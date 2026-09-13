use anyhow::Result;
use crossterm::event::{self, Event, KeyCode};
use ratatui::{
    DefaultTerminal,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, List, ListItem, ListState, Paragraph, Wrap},
};

use muse_session_inspect::core::{Block as Content, Role, Session, SessionMeta};

pub fn run(
    sessions: Vec<SessionMeta>,
    open: impl Fn(&SessionMeta) -> Result<Session>,
) -> Result<()> {
    let mut terminal = ratatui::init();
    let outcome = App::new(sessions).run(&mut terminal, &open);
    ratatui::restore();
    outcome
}

struct App {
    sessions: Vec<SessionMeta>,
    list: ListState,
    view: Option<View>,
    error: Option<String>,
}

struct View {
    meta: SessionMeta,
    lines: Vec<Line<'static>>,
    scroll: u16,
}

impl App {
    fn new(sessions: Vec<SessionMeta>) -> Self {
        let mut list = ListState::default();
        if !sessions.is_empty() {
            list.select(Some(0));
        }
        Self {
            sessions,
            list,
            view: None,
            error: None,
        }
    }

    fn run(
        &mut self,
        terminal: &mut DefaultTerminal,
        open: &dyn Fn(&SessionMeta) -> Result<Session>,
    ) -> Result<()> {
        loop {
            terminal.draw(|frame| self.draw(frame))?;
            let Event::Key(key) = event::read()? else {
                continue;
            };
            match key.code {
                KeyCode::Char('q') => return Ok(()),
                KeyCode::Esc => self.view = None,
                KeyCode::Enter => self.open_selected(open),
                KeyCode::Down | KeyCode::Char('j') => self.step(1),
                KeyCode::Up | KeyCode::Char('k') => self.step(-1),
                _ => {}
            }
        }
    }

    fn step(&mut self, delta: i16) {
        if let Some(view) = &mut self.view {
            view.scroll = view.scroll.saturating_add_signed(delta * 5);
            return;
        }
        if self.sessions.is_empty() {
            return;
        }
        let next = self
            .list
            .selected()
            .unwrap_or(0)
            .saturating_add_signed(delta as isize)
            .min(self.sessions.len() - 1);
        self.list.select(Some(next));
    }

    fn open_selected(&mut self, open: &dyn Fn(&SessionMeta) -> Result<Session>) {
        if self.view.is_some() || self.sessions.is_empty() {
            return;
        }
        let meta = &self.sessions[self.list.selected().unwrap_or(0)];
        let session = match open(meta) {
            Ok(session) => session,
            Err(error) => {
                self.error = Some(muse_session_inspect::core::truncate(
                    &error.to_string(),
                    160,
                ));
                return;
            }
        };
        self.error = None;
        self.view = Some(View {
            meta: SessionMeta {
                id: session.meta.id.clone(),
                tool: session.meta.tool,
                title: session.meta.title.clone(),
                workspace: session.meta.workspace.clone(),
                model: session.meta.model.clone(),
                turns: session.meta.turns,
            },
            lines: render(&session),
            scroll: 0,
        });
    }

    fn draw(&mut self, frame: &mut ratatui::Frame) {
        if let Some(view) = &self.view {
            let title = format!("{} · {} [esc] back [q] quit", view.meta.title, view.meta.id);
            frame.render_widget(
                Paragraph::new(view.lines.clone())
                    .block(Block::bordered().title(title))
                    .wrap(Wrap { trim: false })
                    .scroll((view.scroll, 0)),
                frame.area(),
            );
            return;
        }
        let items = self.sessions.iter().map(|meta| {
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
        let title = match &self.error {
            Some(error) => format!("sessions [!] {error}"),
            None => "sessions [enter] open [q] quit".to_owned(),
        };
        frame.render_stateful_widget(
            List::new(items)
                .block(Block::bordered().title(title))
                .highlight_style(Style::default().bg(Color::DarkGray)),
            frame.area(),
            &mut self.list,
        );
    }
}

fn render(session: &Session) -> Vec<Line<'static>> {
    let mut lines = vec![Line::from(format!(
        "{} · {} · {} turns",
        session.meta.workspace, session.meta.model, session.meta.turns
    ))];
    for turn in &session.turns {
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
    lines
}

fn first_line(text: &str) -> String {
    let line = text.lines().next().unwrap_or_default();
    muse_session_inspect::core::truncate(line, 160)
}
