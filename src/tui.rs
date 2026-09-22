use std::sync::{Arc, mpsc};
use std::time::Duration;

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::{
    DefaultTerminal,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, List, ListItem, ListState, Paragraph, Wrap},
};
use unicode_width::UnicodeWidthChar;

use agent_session_inspect::core::{
    Block as Content, Registry, Role, Session, SessionMeta, truncate,
};

struct Batch {
    tool: &'static str,
    sessions: Result<Vec<SessionMeta>, String>,
}

pub fn run(registry: Registry) -> Result<()> {
    let mut terminal = ratatui::init();
    let outcome = App::new(registry).run(&mut terminal);
    ratatui::restore();
    outcome
}

struct App {
    registry: Arc<Registry>,
    receiver: mpsc::Receiver<Batch>,
    sessions: Vec<SessionMeta>,
    shown: Vec<usize>,
    tools: Vec<&'static str>,
    tab: usize,
    pending: usize,
    load_errors: Vec<String>,
    list: ListState,
    list_height: u16,
    view: Option<View>,
    error: Option<String>,
}

struct View {
    meta: SessionMeta,
    lines: Vec<Line<'static>>,
    turns: Vec<u16>,
    scroll: u16,
    viewport: u16,
    width: u16,
}

impl App {
    fn new(registry: Registry) -> Self {
        let tools = registry.tool_ids();
        let registry = Arc::new(registry);
        let (sender, receiver) = mpsc::channel();
        let pending = tools.len();
        for tool in tools.clone() {
            let registry = Arc::clone(&registry);
            let sender = sender.clone();
            std::thread::spawn(move || {
                let sessions = registry
                    .sessions_for(tool)
                    .map_err(|error| truncate(&error.to_string(), 160));
                let _ = sender.send(Batch { tool, sessions });
            });
        }
        Self {
            registry,
            receiver,
            sessions: Vec::new(),
            shown: Vec::new(),
            tools,
            tab: 0,
            pending,
            load_errors: Vec::new(),
            list: ListState::default(),
            list_height: 0,
            view: None,
            error: None,
        }
    }

    fn run(&mut self, terminal: &mut DefaultTerminal) -> Result<()> {
        loop {
            terminal.draw(|frame| self.draw(frame))?;
            self.drain();
            if !event::poll(Duration::from_millis(120))? {
                continue;
            }
            let Event::Key(key) = event::read()? else {
                continue;
            };
            if key.kind == KeyEventKind::Release {
                continue;
            }
            let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
            match (key.code, ctrl) {
                (KeyCode::Char('c'), true) => return Ok(()),
                (KeyCode::Char('q'), false) => return Ok(()),
                (KeyCode::Esc, _) => self.view = None,
                (KeyCode::Enter, _) => self.open_selected(),
                (KeyCode::Tab, _) if !key.modifiers.contains(KeyModifiers::SHIFT) => {
                    self.next_tab();
                }
                (KeyCode::Tab, _) | (KeyCode::BackTab, _) => self.prev_tab(),
                (KeyCode::Char(digit), false)
                    if digit.is_ascii_digit()
                        && (digit as usize - '0' as usize) <= self.tools.len() =>
                {
                    self.goto_tab(digit as usize - '0' as usize);
                }
                (KeyCode::Char('f'), true) => self.page(1, 1),
                (KeyCode::Char('b'), true) => self.page(-1, 1),
                (KeyCode::Char('d'), true) => self.page(1, 2),
                (KeyCode::Char('u'), true) => self.page(-1, 2),
                (KeyCode::PageDown, _) => self.page(1, 1),
                (KeyCode::PageUp, _) => self.page(-1, 1),
                (KeyCode::Home, _) => self.edge(true),
                (KeyCode::End, _) => self.edge(false),
                (KeyCode::Char('g'), false) => self.edge(true),
                (KeyCode::Char('G'), false) => self.edge(false),
                (KeyCode::Down, _) | (KeyCode::Char('j'), false) => self.step(1),
                (KeyCode::Up, _) | (KeyCode::Char('k'), false) => self.step(-1),
                (KeyCode::Char('n'), false) => self.turn(1),
                (KeyCode::Char('N'), false) => self.turn(-1),
                _ => {}
            }
        }
    }

    fn step(&mut self, delta: i16) {
        if let Some(view) = &mut self.view {
            view.scroll_by(delta);
            return;
        }
        self.move_selection(delta as isize);
    }

    fn page(&mut self, sign: i16, divisor: u16) {
        if let Some(view) = &mut self.view {
            let amount = i32::from(view.page_amount(divisor).max(1)) * i32::from(sign);
            view.scroll_by(amount.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16);
            return;
        }
        let amount = (self.list_height.max(1) / divisor.max(1)).max(1) as isize * sign as isize;
        self.move_selection(amount);
    }

    fn drain(&mut self) {
        let mut arrived = false;
        loop {
            match self.receiver.try_recv() {
                Ok(batch) => {
                    self.pending = self.pending.saturating_sub(1);
                    match batch.sessions {
                        Ok(loaded) => {
                            self.sessions.extend(loaded);
                            self.sort_sessions();
                            arrived = true;
                        }
                        Err(error) => {
                            self.load_errors.push(format!("{}: {error}", batch.tool));
                        }
                    }
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.pending = 0;
                    break;
                }
            }
        }
        if arrived {
            self.refilter();
        }
    }

    fn sort_sessions(&mut self) {
        let order = |tool| {
            self.tools
                .iter()
                .position(|id| *id == tool)
                .unwrap_or(usize::MAX)
        };
        self.sessions.sort_by_key(|meta| order(meta.tool));
    }

    fn refilter(&mut self) {
        self.shown = filter_indices(&self.sessions, &self.tools, self.tab);
        let len = self.shown.len();
        if len == 0 {
            self.list.select(None);
            return;
        }
        let clamped = self.list.selected().unwrap_or(0).min(len - 1);
        self.list.select(Some(clamped));
    }

    fn next_tab(&mut self) {
        if self.view.is_some() {
            return;
        }
        self.goto_tab((self.tab + 1) % (self.tools.len() + 1));
    }

    fn prev_tab(&mut self) {
        if self.view.is_some() {
            return;
        }
        self.goto_tab((self.tab + self.tools.len()) % (self.tools.len() + 1));
    }

    fn goto_tab(&mut self, tab: usize) {
        if self.view.is_some() || tab > self.tools.len() {
            return;
        }
        self.tab = tab;
        self.list
            .select(if self.shown.is_empty() { None } else { Some(0) });
        self.refilter();
    }

    fn edge(&mut self, top: bool) {
        if let Some(view) = &mut self.view {
            view.edge(top);
            return;
        }
        if self.shown.is_empty() {
            return;
        }
        self.list
            .select(Some(if top { 0 } else { self.shown.len() - 1 }));
    }

    fn turn(&mut self, direction: i16) {
        let Some(view) = &mut self.view else {
            return;
        };
        if direction >= 0 {
            view.next_turn();
        } else {
            view.prev_turn();
        }
    }

    fn move_selection(&mut self, delta: isize) {
        if self.shown.is_empty() {
            return;
        }
        let next = self
            .list
            .selected()
            .unwrap_or(0)
            .saturating_add_signed(delta)
            .min(self.shown.len() - 1);
        self.list.select(Some(next));
    }

    fn open_selected(&mut self) {
        if self.view.is_some() || self.shown.is_empty() {
            return;
        }
        let position = self.list.selected().unwrap_or(0).min(self.shown.len() - 1);
        let meta = &self.sessions[self.shown[position]];
        let session = match self.registry.load(meta) {
            Ok(session) => session,
            Err(error) => {
                self.error = Some(truncate(&error.to_string(), 160));
                return;
            }
        };
        self.error = None;
        let (lines, turns) = render(&session);
        self.view = Some(View {
            meta: SessionMeta {
                id: session.meta.id.clone(),
                tool: session.meta.tool,
                title: session.meta.title.clone(),
                workspace: session.meta.workspace.clone(),
                model: session.meta.model.clone(),
                turns: session.meta.turns,
            },
            lines,
            turns,
            scroll: 0,
            viewport: 0,
            width: 0,
        });
    }

    fn draw(&mut self, frame: &mut ratatui::Frame) {
        let width = frame.area().width.saturating_sub(2);
        let height = frame.area().height.saturating_sub(2);
        if let Some(view) = &mut self.view {
            view.width = width;
            view.viewport = height;
            view.scroll = clamp_scroll(view.scroll, &view.lines, view.width, view.viewport);
            let title = format!(
                "{} · {} [j/k] line [pgup/pgdn] page [n/N] turn [esc] back [q] quit",
                view.meta.title, view.meta.id
            );
            frame.render_widget(
                Paragraph::new(view.lines.clone())
                    .block(Block::bordered().title(title))
                    .wrap(Wrap { trim: false })
                    .scroll((view.scroll, 0)),
                frame.area(),
            );
            return;
        }
        self.list_height = height;
        let items = self.shown.iter().map(|index| {
            let meta = &self.sessions[*index];
            ListItem::new(Line::from(vec![
                Span::styled(
                    format!("{} ", meta.title),
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("{} · {} · {} turns", meta.tool, meta.id, meta.turns),
                    Style::default().fg(Color::DarkGray),
                ),
            ]))
        });
        frame.render_stateful_widget(
            List::new(items)
                .block(Block::bordered().title(self.list_title()))
                .highlight_style(Style::default().bg(Color::DarkGray)),
            frame.area(),
            &mut self.list,
        );
    }

    fn list_title(&self) -> String {
        let mut tabs = vec![tab_label("all", 0, self.tab)];
        tabs.extend(
            self.tools
                .iter()
                .enumerate()
                .map(|(index, tool)| tab_label(tool, index + 1, self.tab)),
        );
        let mut status = if self.pending > 0 {
            format!(
                "loading {}/{}",
                self.tools.len() - self.pending,
                self.tools.len()
            )
        } else {
            format!("{} sessions", self.shown.len())
        };
        let failure = self.load_errors.first().map(|error| {
            if self.load_errors.len() > 1 {
                format!("{error} (+{} more)", self.load_errors.len() - 1)
            } else {
                error.clone()
            }
        });
        if let Some(error) = failure.or(self.error.clone()) {
            status.push_str(&format!(" [!] {error}"));
        }
        format!(
            "{} {status} [tab] tool [j/k] move [enter] open [q] quit",
            tabs.join(" ")
        )
    }
}

fn tab_label(name: &str, index: usize, active: usize) -> String {
    if index == active {
        format!("[{name}]")
    } else {
        format!(" {name} ")
    }
}

fn filter_indices(sessions: &[SessionMeta], tools: &[&str], tab: usize) -> Vec<usize> {
    if tab == 0 || tab > tools.len() {
        return (0..sessions.len()).collect();
    }
    let tool = tools[tab - 1];
    sessions
        .iter()
        .enumerate()
        .filter(|(_, meta)| meta.tool == tool)
        .map(|(index, _)| index)
        .collect()
}

impl View {
    fn max(&self) -> u16 {
        max_scroll(&self.lines, self.width, self.viewport)
    }

    fn scroll_by(&mut self, delta: i16) {
        self.scroll = clamp_scroll(
            self.scroll.saturating_add_signed(delta),
            &self.lines,
            self.width,
            self.viewport,
        );
    }

    fn page_amount(&self, divisor: u16) -> u16 {
        (self.viewport / divisor.max(1)).max(1)
    }

    fn edge(&mut self, top: bool) {
        self.scroll = if top { 0 } else { self.max() };
    }

    fn next_turn(&mut self) {
        self.scroll = next_turn(
            &turn_visual_rows(&self.lines, self.width, &self.turns),
            self.scroll,
            self.max(),
        );
    }

    fn prev_turn(&mut self) {
        self.scroll = prev_turn(
            &turn_visual_rows(&self.lines, self.width, &self.turns),
            self.scroll,
        );
    }
}

fn line_text(line: &Line) -> String {
    line.spans
        .iter()
        .map(|span| span.content.as_ref())
        .collect()
}

fn wrapped_rows(text: &str, max_width: u16) -> usize {
    let max = max_width.max(1) as usize;
    let mut rows = 0usize;
    let mut line_width = 0usize;
    let mut line_items = 0usize;
    let mut word_width = 0usize;
    let mut word_items = 0usize;
    let mut spaces: Vec<usize> = Vec::new();
    let mut space_width = 0usize;
    let mut word_previous = false;
    for ch in text.chars() {
        let space = ch.is_whitespace();
        let width = ch.width().unwrap_or(0);
        if width > max {
            continue;
        }
        if (word_previous && space) || (line_items == 0 && word_width + space_width + width > max) {
            line_width += space_width + word_width;
            line_items += spaces.len() + word_items;
            spaces.clear();
            space_width = 0;
            word_width = 0;
            word_items = 0;
        }
        if line_width >= max || (width > 0 && line_width + space_width + word_width >= max) {
            rows += 1;
            line_width = 0;
            line_items = 0;
            let mut remaining = max;
            while let Some(front) = spaces.first() {
                if *front > remaining {
                    break;
                }
                remaining -= front;
                space_width -= front;
                spaces.remove(0);
            }
            if space && spaces.is_empty() {
                continue;
            }
        }
        if space {
            space_width += width;
            spaces.push(width);
        } else {
            word_width += width;
            word_items += 1;
        }
        word_previous = !space;
    }
    if line_items == 0 && word_items == 0 && !spaces.is_empty() {
        rows += 1;
    }
    line_items += spaces.len() + word_items;
    if line_items > 0 || rows == 0 {
        rows += 1;
    }
    rows
}

fn visual_rows(lines: &[Line], width: u16) -> usize {
    lines
        .iter()
        .map(|line| wrapped_rows(&line_text(line), width))
        .sum()
}

fn turn_visual_rows(lines: &[Line], width: u16, turns: &[u16]) -> Vec<u16> {
    let mut rows = Vec::with_capacity(turns.len());
    let mut visual = 0usize;
    let mut pending = turns.iter().peekable();
    for (index, line) in lines.iter().enumerate() {
        while pending
            .peek()
            .is_some_and(|start| **start as usize == index)
        {
            rows.push(visual.min(u16::MAX as usize) as u16);
            pending.next();
        }
        visual += wrapped_rows(&line_text(line), width);
    }
    for _ in pending {
        rows.push(visual.min(u16::MAX as usize) as u16);
    }
    rows
}

fn max_scroll(lines: &[Line], width: u16, viewport: u16) -> u16 {
    visual_rows(lines, width)
        .saturating_sub(viewport as usize)
        .min(u16::MAX as usize) as u16
}

fn clamp_scroll(scroll: u16, lines: &[Line], width: u16, viewport: u16) -> u16 {
    scroll.min(max_scroll(lines, width, viewport))
}

fn next_turn(turns: &[u16], scroll: u16, max: u16) -> u16 {
    turns
        .iter()
        .copied()
        .find(|start| *start > scroll)
        .unwrap_or(scroll)
        .min(max)
}

fn prev_turn(turns: &[u16], scroll: u16) -> u16 {
    if turns.is_empty() {
        return scroll;
    }
    turns
        .iter()
        .copied()
        .rfind(|start| *start < scroll)
        .unwrap_or(0)
}

fn render(session: &Session) -> (Vec<Line<'static>>, Vec<u16>) {
    let mut lines = vec![Line::from(format!(
        "{} · {} · {} turns",
        session.meta.workspace, session.meta.model, session.meta.turns
    ))];
    let mut turns = Vec::with_capacity(session.turns.len());
    for turn in &session.turns {
        turns.push(lines.len().min(u16::MAX as usize) as u16);
        let (label, color) = match turn.role {
            Role::User => ("user", Color::Green),
            Role::Assistant => ("assistant", Color::Blue),
        };
        lines.push(Line::from(Span::styled(
            format!("── {label} ──"),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        )));
        for block in &turn.blocks {
            match block {
                Content::Text(text) => {
                    lines.extend(text.lines().map(|line| Line::from(line.to_owned())))
                }
                Content::Tools(calls) => {
                    for call in calls {
                        let mark = if call.failed { "✗" } else { "✓" };
                        lines.push(Line::from(format!("$ {} {}", call.name, call.summary)));
                        if let Some(result) = &call.result {
                            lines.push(Line::from(format!("{mark} {}", first_line(result))));
                        }
                    }
                }
                Content::Usage { input, output } => lines.push(Line::from(Span::styled(
                    format!("{input} in / {output} out"),
                    Style::default().fg(Color::DarkGray),
                ))),
            }
        }
    }
    (lines, turns)
}

fn first_line(text: &str) -> String {
    let line = text.lines().next().unwrap_or_default();
    agent_session_inspect::core::truncate(line, 160)
}

#[cfg(test)]
mod tests {
    use super::{
        clamp_scroll, filter_indices, max_scroll, next_turn, prev_turn, render, tab_label,
        turn_visual_rows, visual_rows, wrapped_rows,
    };
    use agent_session_inspect::core::{Block, Role, Session, SessionMeta, Turn};
    use ratatui::{
        buffer::Buffer,
        layout::Rect,
        text::Line,
        widgets::{Paragraph, Widget, Wrap},
    };

    fn metas() -> Vec<SessionMeta> {
        ["muse", "claude", "muse"]
            .iter()
            .enumerate()
            .map(|(index, tool)| SessionMeta {
                id: format!("id{index}"),
                tool,
                title: format!("title{index}"),
                workspace: String::new(),
                model: String::new(),
                turns: 0,
            })
            .collect()
    }

    #[test]
    fn filter_tabs_select_all_or_one_tool() {
        let sessions = metas();
        let tools = ["muse", "claude"];
        assert_eq!(filter_indices(&sessions, &tools, 0), vec![0, 1, 2]);
        assert_eq!(filter_indices(&sessions, &tools, 1), vec![0, 2]);
        assert_eq!(filter_indices(&sessions, &tools, 2), vec![1]);
        assert_eq!(filter_indices(&sessions, &tools, 9), vec![0, 1, 2]);
    }

    #[test]
    fn tab_label_marks_only_the_active_tab() {
        assert_eq!(tab_label("all", 0, 0), "[all]");
        assert_eq!(tab_label("muse", 1, 0), " muse ");
        assert_eq!(tab_label("muse", 1, 1), "[muse]");
    }

    fn fixture() -> Session {
        let turn = |role| Turn {
            role,
            blocks: vec![Block::Text("one\ntwo".to_owned())],
        };
        Session {
            meta: SessionMeta {
                id: "id".to_owned(),
                tool: "muse",
                title: "title".to_owned(),
                workspace: "ws".to_owned(),
                model: "model".to_owned(),
                turns: 3,
            },
            turns: vec![turn(Role::User), turn(Role::Assistant), turn(Role::User)],
        }
    }

    fn text_lines(texts: &[&str]) -> Vec<Line<'static>> {
        texts
            .iter()
            .map(|text| Line::from(text.to_string()))
            .collect()
    }

    fn rendered_pattern(text: &str, width: u16, height: u16) -> Vec<bool> {
        let area = Rect::new(0, 0, width, height);
        let mut buffer = Buffer::empty(area);
        Paragraph::new(text)
            .wrap(Wrap { trim: false })
            .render(area, &mut buffer);
        (0..height)
            .map(|y| (0..width).any(|x| buffer[(x, y)].symbol() != " "))
            .collect()
    }

    #[test]
    fn render_records_one_offset_per_turn_header() {
        let session = fixture();
        let (lines, turns) = render(&session);
        assert_eq!(turns.len(), 3);
        assert_eq!(turns, vec![1, 4, 7]);
        assert_eq!(lines.len(), 10);
    }

    #[test]
    fn turn_jump_moves_between_headers_and_stops_at_ends() {
        let turns = vec![1u16, 4, 7];
        assert_eq!(next_turn(&turns, 0, 20), 1);
        assert_eq!(next_turn(&turns, 1, 20), 4);
        assert_eq!(next_turn(&turns, 5, 20), 7);
        assert_eq!(next_turn(&turns, 7, 20), 7);
        assert_eq!(prev_turn(&turns, 7), 4);
        assert_eq!(prev_turn(&turns, 5), 4);
        assert_eq!(prev_turn(&turns, 1), 0);
        assert_eq!(prev_turn(&turns, 0), 0);
    }

    #[test]
    fn turn_jump_without_turns_is_a_noop() {
        assert_eq!(next_turn(&[], 7, 20), 7);
        assert_eq!(prev_turn(&[], 7), 7);
    }

    #[test]
    fn turn_jump_never_scrolls_past_content() {
        let turns = vec![1u16, 40000];
        assert_eq!(next_turn(&turns, 0, 100), 1);
        assert_eq!(next_turn(&turns, 1, 100), 100);
    }

    #[test]
    fn scroll_clamps_to_wrapped_content_bounds() {
        let long = "a".repeat(81);
        let lines = text_lines(&[long.as_str(), "hi"]);
        assert_eq!(max_scroll(&lines, 80, 4), 0);
        assert_eq!(max_scroll(&lines, 80, 1), 2);
        assert_eq!(clamp_scroll(99, &lines, 80, 1), 2);
        assert_eq!(clamp_scroll(2, &lines, 80, 1), 2);
    }

    #[test]
    fn turn_offsets_follow_wrapped_rows() {
        let long = "a".repeat(81);
        let lines = text_lines(&["head", long.as_str(), "── user ──", "tail"]);
        assert_eq!(turn_visual_rows(&lines, 80, &[2]), vec![3]);
        assert_eq!(visual_rows(&lines, 80), 5);
    }

    #[test]
    fn wrapped_rows_cover_short_blank_and_split_lines() {
        assert_eq!(wrapped_rows("", 80), 1);
        assert_eq!(wrapped_rows("hi", 80), 1);
        assert_eq!(wrapped_rows("   ", 80), 2);
        assert_eq!(wrapped_rows("    code", 80), 1);
        assert_eq!(wrapped_rows(&"a".repeat(80), 80), 1);
        assert_eq!(wrapped_rows(&"a".repeat(81), 80), 2);
        assert_eq!(wrapped_rows(&"a".repeat(200), 80), 3);
        assert_eq!(wrapped_rows(&"한".repeat(40), 80), 1);
        assert_eq!(wrapped_rows(&"한".repeat(40), 79), 1);
        assert_eq!(wrapped_rows("a ", 1), 1);
    }

    #[test]
    fn wrapped_rows_match_paragraph_renderer() {
        let samples = [
            "hello world".to_owned(),
            "a".repeat(80),
            "a".repeat(81),
            "a".repeat(200),
            "ab".repeat(80),
            format!("{} {} {}", "a".repeat(41), "b".repeat(41), "c".repeat(18)),
            "한".repeat(40),
            "hello 한 world foo bar baz qux quux corge grault".to_owned(),
            "indented code stays on one row here".to_owned(),
        ];
        for width in [7u16, 40, 79, 80, 100] {
            for sample in &samples {
                let mut expected = vec![true; wrapped_rows(sample, width)];
                expected.extend([false, false, false, false]);
                let height = expected.len() as u16;
                assert_eq!(
                    rendered_pattern(sample, width, height),
                    expected,
                    "sample {sample:?} at width {width}"
                );
            }
        }
    }

    #[test]
    fn wrapped_rows_match_renderer_on_edge_spacing() {
        for (text, width, expected) in [
            ("a \nb", 1, vec![true, true, false, false]),
            ("a \nb", 80, vec![true, true, false, false]),
            ("   \nxy", 80, vec![false, false, true, false, false]),
        ] {
            let height = expected.len() as u16;
            assert_eq!(
                rendered_pattern(text, width, height),
                expected,
                "text {text:?} at width {width}"
            );
        }
    }
}
