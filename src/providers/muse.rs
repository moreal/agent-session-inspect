use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde_json::Value;
use walkdir::WalkDir;

use crate::core::{Block, Provider, Role, Session, SessionMeta, ToolCall, Turn, truncate};

pub struct MuseProvider {
    root: PathBuf,
}

impl MuseProvider {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn default_root() -> PathBuf {
        std::env::var("HOME")
            .map(|home| PathBuf::from(home).join(".local/share/muse/sessions"))
            .unwrap_or_else(|_| PathBuf::from("."))
    }

    fn candidates(&self) -> Vec<PathBuf> {
        let mut paths: Vec<PathBuf> = WalkDir::new(&self.root)
            .into_iter()
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.into_path())
            .filter(|path| path.file_name().is_some_and(|name| name == "session.jsonl"))
            .collect();
        paths.sort();
        paths
    }

    fn log_path(&self, id: &str) -> Option<PathBuf> {
        self.candidates()
            .into_iter()
            .filter(|path| {
                path.parent()
                    .is_some_and(|parent| parent.file_name().is_some_and(|name| name == id))
            })
            .max_by(|first, second| {
                let stat = |path: &PathBuf| {
                    std::fs::metadata(path)
                        .map(|meta| {
                            (
                                meta.len(),
                                meta.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH),
                            )
                        })
                        .unwrap_or((0, std::time::SystemTime::UNIX_EPOCH))
                };
                let (first_size, first_time) = stat(first);
                let (second_size, second_time) = stat(second);
                first_size
                    .cmp(&second_size)
                    .then_with(|| first_time.cmp(&second_time))
                    .then_with(|| first.cmp(second))
            })
    }

    fn records(&self, id: &str) -> Result<Vec<Record>> {
        let path = self.log_path(id).ok_or_else(|| {
            anyhow::anyhow!("session {id} not found under {}", self.root.display())
        })?;
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let mut records = Vec::new();
        for line in text.lines() {
            let Ok(line) = serde_json::from_str::<Value>(line) else {
                continue;
            };
            if let Some(children) = line.get("children").and_then(Value::as_array)
                && !children.is_empty()
            {
                for child in children {
                    if let Some(raw) = child.get("record_json").and_then(Value::as_str)
                        && let Ok(record) = serde_json::from_str(raw)
                    {
                        records.push(record);
                    }
                }
            } else if line.get("payload_type").is_some()
                && let Ok(record) = serde_json::from_value(line)
            {
                records.push(record);
            }
        }
        Ok(records)
    }
}

impl Provider for MuseProvider {
    fn tool_id(&self) -> &'static str {
        "muse"
    }

    fn sessions(&self) -> Result<Vec<SessionMeta>> {
        let mut metas = Vec::new();
        let mut seen = HashSet::new();
        for path in self.candidates() {
            if let Some(id) = path
                .parent()
                .and_then(|parent| parent.file_name())
                .and_then(|name| name.to_str())
                && seen.insert(id.to_owned())
                && let Ok(session) = self.load(id)
            {
                metas.push(session.meta);
            }
        }
        Ok(metas)
    }

    fn load(&self, id: &str) -> Result<Session> {
        let mut fold = Fold::new(id);
        for record in self.records(id)? {
            fold.apply(&record);
        }
        Ok(fold.finish())
    }
}

struct Record {
    payload_type: String,
    payload: Value,
}

impl<'de> serde::Deserialize<'de> for Record {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        Ok(Self {
            payload_type: value
                .get("payload_type")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            payload: value.get("payload").cloned().unwrap_or(Value::Null),
        })
    }
}

struct Fold {
    id: String,
    title: String,
    workspace: String,
    model: String,
    turns: Vec<Turn>,
    calls: HashMap<String, (usize, usize)>,
    tasks: HashMap<String, String>,
    settled: HashSet<String>,
}

impl Fold {
    fn new(id: &str) -> Self {
        Self {
            id: id.to_owned(),
            title: id.to_owned(),
            workspace: String::new(),
            model: String::new(),
            turns: Vec::new(),
            calls: HashMap::new(),
            tasks: HashMap::new(),
            settled: HashSet::new(),
        }
    }

    fn user_turn(&mut self, text: &str) {
        let duplicate = self.turns.last().is_some_and(|turn| {
            matches!(turn.role, Role::User)
                && turn
                    .blocks
                    .iter()
                    .any(|block| matches!(block, Block::Text(existing) if existing == text))
        });
        if !duplicate && !text.is_empty() {
            self.turns.push(Turn {
                role: Role::User,
                blocks: vec![Block::Text(text.to_owned())],
            });
        }
    }

    fn assistant_text(&mut self, event: &Value) {
        let text = event
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if !text.is_empty() {
            self.assistant_turn()
                .blocks
                .push(Block::Text(text.to_owned()));
        }
    }

    fn assistant_turn(&mut self) -> &mut Turn {
        let fresh = self.turns.last().is_some_and(|turn| {
            matches!(turn.role, Role::Assistant)
                && turn
                    .blocks
                    .iter()
                    .any(|block| matches!(block, Block::Usage { .. }))
        });
        if !self
            .turns
            .last()
            .is_some_and(|turn| matches!(turn.role, Role::Assistant))
            || fresh
        {
            self.turns.push(Turn {
                role: Role::Assistant,
                blocks: Vec::new(),
            });
        }
        self.turns.last_mut().expect("just pushed")
    }

    fn find_call(&mut self, call_id: &str) -> Option<&mut ToolCall> {
        let (turn, mut index) = self.calls.get(call_id).copied()?;
        for block in self.turns.get_mut(turn)?.blocks.iter_mut() {
            if let Block::Tools(calls) = block {
                if index < calls.len() {
                    return calls.get_mut(index);
                }
                index -= calls.len();
            }
        }
        None
    }

    fn apply(&mut self, record: &Record) {
        match record.payload_type.as_str() {
            "runtime.session" => self.apply_session_event(&record.payload),
            "runtime.user_intent.accepted" => {
                let text = record
                    .payload
                    .get("refill_blocks")
                    .and_then(Value::as_array)
                    .map(|blocks| {
                        blocks
                            .iter()
                            .filter_map(|block| block.get("text").and_then(Value::as_str))
                            .collect::<Vec<_>>()
                            .join("\n")
                    })
                    .unwrap_or_default();
                self.user_turn(&text);
            }
            "tool_batch.effect.started" => {
                let task = field(&record.payload, "record", "task_id");
                let call = field(&record.payload, "record", "call_id");
                if !task.is_empty() && !call.is_empty() {
                    self.tasks.insert(task, call);
                }
            }
            "runtime.session.metadata" => {
                let workspace = field(&record.payload, "record", "workspace_root");
                if !workspace.is_empty() {
                    self.workspace = workspace;
                }
                let model = field(&record.payload, "record", "model_id");
                if !model.is_empty() {
                    self.model = model;
                }
            }
            "runtime.session.route_facts" => {
                if self.workspace.is_empty() {
                    self.workspace = field(&record.payload, "record", "cwd");
                }
            }
            "run.model.configured" => {
                let model = field(&record.payload, "record", "model_id");
                if !model.is_empty() {
                    self.model = model;
                }
            }
            "runtime.model_reconfigure.completed" => {
                let model = record
                    .payload
                    .get("record")
                    .and_then(|record| record.get("effective"))
                    .and_then(|effective| effective.get("model_id"))
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if !model.is_empty() {
                    self.model = model.to_owned();
                }
            }
            "session.name.changed" => {
                let name = record
                    .payload
                    .get("new_name")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if !name.is_empty() {
                    self.title = name.to_owned();
                }
            }
            _ => {}
        }
    }

    fn apply_session_event(&mut self, payload: &Value) {
        match payload.get("kind").and_then(Value::as_str) {
            Some("run") => self.apply_run_event(&payload["event"]),
            Some("task") => self.apply_task_event(&payload["event"]),
            _ => {}
        }
    }

    fn apply_run_event(&mut self, event: &Value) {
        match event.get("kind").and_then(Value::as_str) {
            Some("started") => {
                let prompt = event
                    .get("prompt")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                self.user_turn(prompt);
            }
            Some("assistant_tool_calls_committed") => {
                let mut seen = HashSet::new();
                let calls: Vec<Value> = event
                    .get("tool_calls")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
                    .filter(|call| {
                        let id = call
                            .get("call_id")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        !id.is_empty() && seen.insert(id.to_owned()) && !self.calls.contains_key(id)
                    })
                    .collect();
                if calls.is_empty() {
                    return;
                }
                let turn_index = self.assistant_turn_index();
                let base = self.turns[turn_index]
                    .blocks
                    .iter()
                    .filter_map(|block| match block {
                        Block::Tools(calls) => Some(calls.len()),
                        _ => None,
                    })
                    .sum::<usize>();
                let mut tools = Vec::new();
                for call in &calls {
                    let call_id = call
                        .get("call_id")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    let name = call.get("name").and_then(Value::as_str).unwrap_or("?");
                    let args = match call.get("args") {
                        Some(Value::String(text)) => text.clone(),
                        Some(Value::Object(map)) => salient_arg(map).unwrap_or_else(|| {
                            truncate(&serde_json::to_string(map).unwrap_or_default(), 300)
                        }),
                        Some(args) => {
                            truncate(&serde_json::to_string(args).unwrap_or_default(), 300)
                        }
                        None => String::new(),
                    };
                    self.calls
                        .insert(call_id.to_owned(), (turn_index, base + tools.len()));
                    tools.push(ToolCall {
                        name: name.to_owned(),
                        summary: summarize_args(&args),
                        result: None,
                        failed: false,
                    });
                }
                self.assistant_turn().blocks.push(Block::Tools(tools));
            }
            Some("tool_result_batch_committed") => {
                let results = event
                    .get("results")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                for result in &results {
                    let call_id = result
                        .get("tool_call_id")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    let text = result
                        .get("text")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    if text.is_empty() || call_id.is_empty() {
                        continue;
                    }
                    if self.calls.contains_key(call_id) {
                        self.settled.insert(call_id.to_owned());
                    }
                    if let Some(call) = self.find_call(call_id) {
                        call.result = Some(unpack_result(text));
                        call.failed |= looks_failed(text);
                    }
                }
            }
            Some("assistant_message_committed") | Some("reasoning_summary_committed") => {
                self.assistant_text(event);
            }
            Some("inbox_item_queued") => {
                let source = event.get("source");
                let steer = source.and_then(Value::as_str) == Some("user_steer")
                    || source
                        .and_then(|source| source.get("source"))
                        .and_then(Value::as_str)
                        == Some("user_steer");
                if steer {
                    let text = event
                        .get("payload")
                        .and_then(|payload| payload.get("prompt"))
                        .and_then(Value::as_str)
                        .or_else(|| event.get("body").and_then(Value::as_str))
                        .unwrap_or_default();
                    self.user_turn(text);
                }
            }
            Some("model_completed") => {
                let input = event
                    .get("usage")
                    .and_then(|usage| usage.get("input_tokens"))
                    .and_then(Value::as_u64)
                    .unwrap_or_default();
                let output = event
                    .get("usage")
                    .and_then(|usage| usage.get("output_tokens"))
                    .and_then(Value::as_u64)
                    .unwrap_or_default();
                if input + output > 0 {
                    self.assistant_turn()
                        .blocks
                        .push(Block::Usage { input, output });
                }
            }
            _ => {}
        }
    }

    fn apply_task_event(&mut self, event: &Value) {
        let task_id = event
            .get("task_id")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let Some(call_id) = self.tasks.get(task_id).cloned() else {
            return;
        };
        let known = self.calls.contains_key(&call_id);
        match event.get("kind").and_then(Value::as_str) {
            Some("output") => {
                let final_output = event.get("final_result").and_then(Value::as_bool) == Some(true);
                if final_output && known {
                    self.settled.insert(call_id.clone());
                }
                let chunk = event
                    .get("chunk")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if chunk.is_empty() {
                    return;
                }
                if final_output {
                    if let Some(call) = self.find_call(&call_id) {
                        call.result = Some(unpack_result(chunk));
                        call.failed |= looks_failed(chunk);
                    }
                } else if !self.settled.contains(&call_id)
                    && let Some(call) = self.find_call(&call_id)
                    && call.result.is_none()
                {
                    call.result = Some(unpack_result(chunk));
                }
            }
            Some("failed") | Some("timed_out") | Some("cancelled") | Some("rejected") => {
                let reason = event
                    .get("reason")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if known {
                    self.settled.insert(call_id.clone());
                }
                if let Some(call) = self.find_call(&call_id) {
                    call.failed = true;
                    if call.result.is_none() && !reason.is_empty() {
                        call.result = Some(truncate(reason, 6000));
                    }
                }
            }
            Some("tool_output_ref") => {
                let reference = event.get("output_ref");
                let kind = reference
                    .and_then(|reference| reference.get("kind"))
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let uri = reference
                    .and_then(|reference| reference.get("uri"))
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let available = event
                    .get("availability")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if uri.is_empty() || (!available.is_empty() && available != "available") {
                    return;
                }
                let settled = self.settled.contains(&call_id);
                if let Some(call) = self.find_call(&call_id)
                    && !call.failed
                    && !settled
                {
                    call.result = Some(if kind.is_empty() {
                        format!("large output: {uri}")
                    } else {
                        format!("large output ({kind}): {uri}")
                    });
                }
            }
            _ => {}
        }
    }

    fn assistant_turn_index(&mut self) -> usize {
        self.assistant_turn();
        self.turns.len() - 1
    }

    fn finish(self) -> Session {
        let turns = self.turns.len();
        Session {
            meta: SessionMeta {
                id: self.id,
                tool: "muse",
                title: self.title,
                workspace: self.workspace,
                model: self.model,
                turns,
            },
            turns: self.turns,
        }
    }
}

fn field(payload: &Value, group: &str, key: &str) -> String {
    payload
        .get(group)
        .and_then(|group| group.get(key))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned()
}

const SALIENT_ARG_KEYS: [&str; 7] = [
    "command",
    "description",
    "path",
    "url",
    "query",
    "pattern",
    "name",
];

fn salient_arg(map: &serde_json::Map<String, Value>) -> Option<String> {
    SALIENT_ARG_KEYS
        .into_iter()
        .filter_map(|key| map.get(key).and_then(Value::as_str))
        .find(|text| !text.is_empty())
        .map(str::to_owned)
}

fn summarize_args(args: &str) -> String {
    let parsed: Value = serde_json::from_str(args).unwrap_or(Value::Null);
    let text = match &parsed {
        Value::Object(map) => salient_arg(map).unwrap_or_else(|| args.to_owned()),
        _ => args.to_owned(),
    };
    truncate(&text, 300)
}

fn unpack_result(chunk: &str) -> String {
    if let Ok(envelope) = serde_json::from_str::<Value>(chunk)
        && let Some(output) = envelope.get("output").and_then(Value::as_str)
    {
        return truncate(output, 6000);
    }
    truncate(chunk, 6000)
}

fn looks_failed(text: &str) -> bool {
    if let Ok(envelope) = serde_json::from_str::<Value>(text) {
        return terminal_failed(&envelope) || exit_failed(envelope.get("exit_code"));
    }
    scan_failed(text)
}

fn terminal_failed(envelope: &Value) -> bool {
    envelope.get("terminal_status").and_then(Value::as_str) == Some("failed")
}

fn exit_failed(code: Option<&Value>) -> bool {
    match code {
        Some(Value::Number(code)) => {
            code.as_i64().is_some_and(|code| code != 0)
                || code.as_u64().is_some_and(|code| code != 0)
                || code
                    .as_f64()
                    .is_some_and(|code| code.fract() == 0.0 && code != 0.0)
        }
        Some(Value::String(code)) => {
            let code = code.trim();
            code.parse::<i64>().is_ok_and(|code| code != 0)
                || code.parse::<u64>().is_ok_and(|code| code != 0)
                || (!code.ends_with('.')
                    && code
                        .parse::<f64>()
                        .is_ok_and(|code| code.fract() == 0.0 && code != 0.0))
        }
        _ => false,
    }
}

fn scan_failed(text: &str) -> bool {
    let stripped = strip_ws_outside_strings(text);
    let bytes = stripped.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        let Some((key, next)) = read_quoted(bytes, index) else {
            index += 1;
            continue;
        };
        index = next;
        skip_ws(bytes, &mut index);
        if !consume(bytes, &mut index, b':') {
            continue;
        }
        skip_ws(bytes, &mut index);
        if key == "terminal_status" {
            if let Some((value, next)) = read_quoted(bytes, index) {
                index = next;
                if value == "failed" {
                    return true;
                }
            }
        } else if key == "exit_code" && number_failed(bytes, &mut index) {
            return true;
        }
    }
    false
}

fn strip_ws_outside_strings(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut index = 0;
    while index < chars.len() {
        let next = chars[index];
        index += 1;
        if next == '"' {
            out.push(next);
            while index < chars.len() {
                let next = chars[index];
                index += 1;
                out.push(next);
                if next == '\\' {
                    if index < chars.len() {
                        out.push(chars[index]);
                        index += 1;
                    }
                } else if next == '"' {
                    break;
                }
            }
        } else if !next.is_whitespace() {
            out.push(next);
        }
    }
    out
}

fn read_quoted(bytes: &[u8], mut index: usize) -> Option<(String, usize)> {
    if bytes.get(index) != Some(&b'"') {
        return None;
    }
    index += 1;
    let mut content = Vec::new();
    while let Some(&byte) = bytes.get(index) {
        if byte == b'\\' {
            if let Some(&next) = bytes.get(index + 1) {
                content.push(byte);
                content.push(next);
                index += 2;
                continue;
            }
            return None;
        }
        if byte == b'"' {
            return String::from_utf8(content).ok().map(|key| (key, index + 1));
        }
        content.push(byte);
        index += 1;
    }
    None
}

fn skip_ws(bytes: &[u8], index: &mut usize) {
    while matches!(
        bytes.get(*index),
        Some(b' ') | Some(b'\t') | Some(b'\n') | Some(b'\r')
    ) {
        *index += 1;
    }
}

fn consume(bytes: &[u8], index: &mut usize, want: u8) -> bool {
    if bytes.get(*index) == Some(&want) {
        *index += 1;
        true
    } else {
        false
    }
}

fn number_failed(bytes: &[u8], index: &mut usize) -> bool {
    let entry = *index;
    let quoted = consume(bytes, index, b'"');
    let restore = |index: &mut usize| {
        if quoted {
            *index = entry;
        }
    };
    if quoted {
        skip_ws(bytes, index);
    }
    let start = *index;
    if matches!(bytes.get(*index), Some(b'-') | Some(b'+')) {
        *index += 1;
    }
    let digits = *index;
    while matches!(bytes.get(*index), Some(b'0'..=b'9')) {
        *index += 1;
    }
    if *index == digits {
        restore(&mut *index);
        return false;
    }
    let mut float = false;
    if consume(bytes, index, b'.') {
        let fraction = *index;
        while matches!(bytes.get(*index), Some(b'0'..=b'9')) {
            *index += 1;
        }
        if *index == fraction {
            if quoted {
                restore(&mut *index);
                return false;
            }
            *index -= 1;
        } else {
            float = true;
        }
    }
    if matches!(bytes.get(*index), Some(b'e') | Some(b'E')) {
        float = true;
        *index += 1;
        if matches!(bytes.get(*index), Some(b'+') | Some(b'-')) {
            *index += 1;
        }
        let exponent = *index;
        while matches!(bytes.get(*index), Some(b'0'..=b'9')) {
            *index += 1;
        }
        if *index == exponent {
            restore(&mut *index);
            return false;
        }
    }
    let end = *index;
    if quoted {
        skip_ws(bytes, index);
        if !consume(bytes, index, b'"') {
            restore(&mut *index);
            return false;
        }
    } else {
        let terminal = bytes.get(*index);
        let closed = matches!(
            terminal,
            None | Some(b',')
                | Some(b'}')
                | Some(b']')
                | Some(b'"')
                | Some(b')')
                | Some(b';')
                | Some(b'!')
                | Some(b'?')
        ) || (!float && terminal == Some(&b'.'));
        if !closed {
            return false;
        }
    }
    let literal = &bytes[start..end.min(bytes.len())];
    let literal = std::str::from_utf8(literal).unwrap_or_default();
    if float {
        literal
            .parse::<f64>()
            .is_ok_and(|code| code.fract() == 0.0 && code != 0.0)
    } else {
        literal
            .trim_start_matches(['-', '+'])
            .chars()
            .any(|digit| digit != '0')
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn fixture() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().expect("tempdir");
        let session = dir.path().join("2026/09/13/abc123");
        std::fs::create_dir_all(&session).expect("mkdirs");
        let mut file = std::fs::File::create(session.join("session.jsonl")).expect("create");
        writeln!(
            file,
            "{{\"payload_type\":\"runtime.user_intent.accepted\",\"payload\":{{\"refill_blocks\":[{{\"kind\":\"text\",\"text\":\"hello\"}}]}}}}"
        )
        .expect("write");
        writeln!(
            file,
            "{{\"payload_type\":\"runtime.session\",\"payload\":{{\"kind\":\"run\",\"event\":{{\"kind\":\"started\",\"prompt\":\"hello\"}}}}}}"
        )
        .expect("write");
        writeln!(
            file,
            "{{\"payload_type\":\"runtime.session\",\"payload\":{{\"kind\":\"run\",\"event\":{{\"kind\":\"assistant_tool_calls_committed\",\"tool_calls\":[{{\"call_id\":\"c1\",\"name\":\"bash\",\"args\":\"{{\\\"command\\\":\\\"ls\\\"}}\"}}]}}}}}}"
        )
        .expect("write");
        writeln!(
            file,
            "{{\"payload_type\":\"tool_batch.effect.started\",\"payload\":{{\"record\":{{\"task_id\":\"t1\",\"call_id\":\"c1\"}}}}}}"
        )
        .expect("write");
        writeln!(
            file,
            "{{\"payload_type\":\"runtime.session\",\"payload\":{{\"kind\":\"task\",\"event\":{{\"kind\":\"output\",\"task_id\":\"t1\",\"final_result\":true,\"chunk\":\"done\"}}}}}}"
        )
        .expect("write");
        (dir, session)
    }

    #[test]
    fn folds_user_prompt_tool_call_and_result() {
        let (dir, _) = fixture();
        let provider = MuseProvider::new(dir.path().to_owned());
        let session = provider.load("abc123").expect("load");
        assert_eq!(session.turns.len(), 2);
        assert!(matches!(session.turns[0].role, Role::User));
        let Block::Tools(calls) = &session.turns[1].blocks[0] else {
            panic!("expected tools block");
        };
        assert_eq!(calls[0].name, "bash");
        assert_eq!(calls[0].result.as_deref(), Some("done"));
    }

    #[test]
    fn framed_children_parse() {
        let (dir, session) = fixture();
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(session.join("session.jsonl"))
            .expect("open");
        writeln!(
            file,
            "{{\"children\":[{{\"record_json\":\"{{\\\"payload_type\\\":\\\"session.name.changed\\\",\\\"payload\\\":{{\\\"new_name\\\":\\\"framed\\\"}}}}\"}}]}}"
        )
        .expect("write");
        let provider = MuseProvider::new(dir.path().to_owned());
        let session = provider.load("abc123").expect("load");
        assert_eq!(session.meta.title, "framed");
    }

    #[test]
    fn lists_sessions() {
        let (dir, _) = fixture();
        let provider = MuseProvider::new(dir.path().to_owned());
        let metas = provider.sessions().expect("sessions");
        assert_eq!(metas.len(), 1);
        assert_eq!(metas[0].id, "abc123");
    }

    fn commit(file: &mut std::fs::File, call_id: &str, command: &str) {
        writeln!(
            file,
            "{{\"payload_type\":\"runtime.session\",\"payload\":{{\"kind\":\"run\",\"event\":{{\"kind\":\"assistant_tool_calls_committed\",\"tool_calls\":[{{\"call_id\":\"{call_id}\",\"name\":\"bash\",\"args\":\"{{\\\"command\\\":\\\"{command}\\\"}}\"}}]}}}}}}"
        )
        .expect("write");
    }

    fn output(file: &mut std::fs::File, task_id: &str, call_id: &str, chunk: &str) {
        link(file, task_id, call_id);
        writeln!(
            file,
            "{{\"payload_type\":\"runtime.session\",\"payload\":{{\"kind\":\"task\",\"event\":{{\"kind\":\"output\",\"task_id\":\"{task_id}\",\"final_result\":true,\"chunk\":\"{chunk}\"}}}}}}"
        )
        .expect("write");
    }

    fn tools_of(session: &Session) -> Vec<(String, Option<String>)> {
        session
            .turns
            .iter()
            .flat_map(|turn| turn.blocks.iter())
            .filter_map(|block| match block {
                Block::Tools(calls) => Some(calls),
                _ => None,
            })
            .flatten()
            .map(|call| (call.summary.clone(), call.result.clone()))
            .collect()
    }

    #[test]
    fn attributes_results_across_commit_batches() {
        let (dir, session) = fixture();
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(session.join("session.jsonl"))
            .expect("open");
        commit(&mut file, "c2", "pwd");
        commit(&mut file, "c3", "whoami");
        output(&mut file, "t2", "c2", "second");
        output(&mut file, "t3", "c3", "third");
        let provider = MuseProvider::new(dir.path().to_owned());
        let session = provider.load("abc123").expect("load");
        assert_eq!(
            tools_of(&session),
            [
                ("ls".to_owned(), Some("done".to_owned())),
                ("pwd".to_owned(), Some("second".to_owned())),
                ("whoami".to_owned(), Some("third".to_owned())),
            ]
        );
    }

    fn append(session: &std::path::Path, line: &str) {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(session.join("session.jsonl"))
            .expect("open");
        writeln!(file, "{line}").expect("write");
    }

    fn run_event(event: &str) -> String {
        format!(
            "{{\"payload_type\":\"runtime.session\",\"payload\":{{\"kind\":\"run\",\"event\":{event}}}}}"
        )
    }

    fn task_event(event: &str) -> String {
        format!(
            "{{\"payload_type\":\"runtime.session\",\"payload\":{{\"kind\":\"task\",\"event\":{event}}}}}"
        )
    }

    fn text_blocks(session: &Session) -> Vec<&str> {
        session
            .turns
            .iter()
            .flat_map(|turn| turn.blocks.iter())
            .filter_map(|block| match block {
                Block::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn renders_assistant_message_and_summary() {
        let (dir, session) = fixture();
        append(
            &session,
            &run_event(
                "{\"kind\":\"reasoning_summary_committed\",\"message_id\":\"m1\",\"response_id\":\"r1\",\"text\":\"Doing the thing\"}",
            ),
        );
        append(
            &session,
            &run_event(
                "{\"kind\":\"assistant_message_committed\",\"message_id\":\"m2\",\"response_id\":\"r1\",\"text\":\"Done did it\"}",
            ),
        );
        let provider = MuseProvider::new(dir.path().to_owned());
        let session = provider.load("abc123").expect("load");
        let texts = text_blocks(&session);
        assert!(texts.contains(&"Doing the thing"));
        assert!(texts.contains(&"Done did it"));
    }

    #[test]
    fn steer_inbox_becomes_user_turn() {
        let (dir, session) = fixture();
        append(
            &session,
            &run_event(
                "{\"kind\":\"inbox_item_queued\",\"source\":{\"source\":\"user_steer\"},\"disposition\":\"steer\",\"body\":\"mid-run note\",\"payload\":{\"prompt\":\"mid-run note\"}}",
            ),
        );
        append(
            &session,
            &run_event(
                "{\"kind\":\"inbox_item_queued\",\"source\":{\"source\":\"subagent_result\"},\"disposition\":\"queue\",\"body\":\"result envelope\"}",
            ),
        );
        append(
            &session,
            &run_event(
                "{\"kind\":\"inbox_item_queued\",\"source\":\"user_steer\",\"disposition\":\"steer\",\"body\":\"plain steer\"}",
            ),
        );
        let provider = MuseProvider::new(dir.path().to_owned());
        let session = provider.load("abc123").expect("load");
        let users: Vec<&str> = session
            .turns
            .iter()
            .filter(|turn| matches!(turn.role, Role::User))
            .flat_map(|turn| turn.blocks.iter())
            .filter_map(|block| match block {
                Block::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert!(users.contains(&"mid-run note"));
        assert!(users.contains(&"plain steer"));
        assert!(!users.contains(&"result envelope"));
    }

    fn link(file: &mut std::fs::File, task_id: &str, call_id: &str) {
        use std::io::Write;
        writeln!(
            file,
            "{{\"payload_type\":\"tool_batch.effect.started\",\"payload\":{{\"record\":{{\"task_id\":\"{task_id}\",\"call_id\":\"{call_id}\"}}}}}}"
        )
        .expect("write");
    }

    fn calls_of(session: &Session) -> Vec<&ToolCall> {
        session
            .turns
            .iter()
            .flat_map(|turn| turn.blocks.iter())
            .filter_map(|block| match block {
                Block::Tools(calls) => Some(calls),
                _ => None,
            })
            .flatten()
            .collect()
    }

    #[test]
    fn failed_and_rejected_tasks_mark_call() {
        let (dir, session) = fixture();
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(session.join("session.jsonl"))
            .expect("open");
        commit(&mut file, "c7", "blasted");
        link(&mut file, "t7", "c7");
        commit(&mut file, "c8", "doomed");
        link(&mut file, "t8", "c8");
        commit(&mut file, "c9", "refused");
        link(&mut file, "t9", "c9");
        drop(file);
        append(
            &session,
            &task_event(
                "{\"kind\":\"failed\",\"task_id\":\"t7\",\"reason\":\"timeout after 10s\"}",
            ),
        );
        append(
            &session,
            &task_event(
                "{\"kind\":\"failed\",\"task_id\":\"t8\",\"reason\":\"process exited with status 1\"}",
            ),
        );
        append(
            &session,
            &task_event("{\"kind\":\"rejected\",\"task_id\":\"t9\"}"),
        );
        append(
            &session,
            &task_event(
                "{\"kind\":\"output\",\"task_id\":\"t8\",\"final_result\":true,\"chunk\":\"late\"}",
            ),
        );
        append(
            &session,
            &task_event("{\"kind\":\"output\",\"task_id\":\"t9\",\"chunk\":\"late-bits\"}"),
        );
        append(
            &session,
            &task_event(
                "{\"kind\":\"tool_output_ref\",\"task_id\":\"t9\",\"output_ref\":{\"kind\":\"blob\",\"uri\":\"tool-output://local/q\"},\"availability\":\"available\"}",
            ),
        );
        let provider = MuseProvider::new(dir.path().to_owned());
        let session = provider.load("abc123").expect("load");
        let calls = calls_of(&session);
        let doomed = calls
            .iter()
            .find(|call| call.summary == "doomed")
            .expect("call");
        assert!(doomed.failed);
        assert_eq!(doomed.result.as_deref(), Some("late"));
        let refused = calls
            .iter()
            .find(|call| call.summary == "refused")
            .expect("call");
        assert!(refused.failed);
        assert_eq!(refused.result, None);
        let blasted = calls
            .iter()
            .find(|call| call.summary == "blasted")
            .expect("call");
        assert!(blasted.failed);
        assert_eq!(blasted.result.as_deref(), Some("timeout after 10s"));
    }

    #[test]
    fn nonfinal_output_fills_absent_result_but_never_overwrites_final() {
        let (dir, session) = fixture();
        append(
            &session,
            &task_event("{\"kind\":\"output\",\"task_id\":\"t1\",\"chunk\":\"stale\"}"),
        );
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(session.join("session.jsonl"))
            .expect("open");
        commit(&mut file, "c8", "partial");
        link(&mut file, "t8", "c8");
        commit(&mut file, "c9", "flaky");
        link(&mut file, "t9", "c9");
        drop(file);
        append(
            &session,
            &task_event("{\"kind\":\"output\",\"task_id\":\"t8\",\"chunk\":\"early\"}"),
        );
        append(
            &session,
            &task_event(
                "{\"kind\":\"output\",\"task_id\":\"t9\",\"chunk\":\"retry {\\\"exit_code\\\": 1}\"}",
            ),
        );
        append(
            &session,
            &task_event(
                "{\"kind\":\"output\",\"task_id\":\"t9\",\"final_result\":true,\"chunk\":\"ok\"}",
            ),
        );
        let provider = MuseProvider::new(dir.path().to_owned());
        let session = provider.load("abc123").expect("load");
        let texts = tools_of(&session);
        assert_eq!(texts[0].1.as_deref(), Some("done"));
        let partial = texts
            .iter()
            .find(|(summary, _)| summary == "partial")
            .expect("call");
        assert_eq!(partial.1.as_deref(), Some("early"));
        let binding = calls_of(&session);
        let flaky = binding
            .iter()
            .find(|call| call.summary == "flaky")
            .expect("call");
        assert_eq!(flaky.result.as_deref(), Some("ok"));
        assert!(!flaky.failed);
    }

    #[test]
    fn output_ref_marks_call_without_clobbering_result() {
        let (dir, session) = fixture();
        append(
            &session,
            &task_event(
                "{\"kind\":\"tool_output_ref\",\"task_id\":\"t1\",\"output_ref\":{\"kind\":\"web_fetch-page\",\"uri\":\"tool-output://local/x\"},\"availability\":\"available\"}",
            ),
        );
        let provider = MuseProvider::new(dir.path().to_owned());
        let session = provider.load("abc123").expect("load");
        let texts = tools_of(&session);
        assert_eq!(texts[0].1.as_deref(), Some("done"));
        let (dir, session) = fixture();
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(session.join("session.jsonl"))
            .expect("open");
        commit(&mut file, "c2", "fetch");
        writeln!(
            file,
            "{{\"payload_type\":\"tool_batch.effect.started\",\"payload\":{{\"record\":{{\"task_id\":\"t2\",\"call_id\":\"c2\"}}}}}}"
        )
        .expect("write");
        drop(file);
        append(
            &session,
            &task_event(
                "{\"kind\":\"tool_output_ref\",\"task_id\":\"t2\",\"output_ref\":{\"kind\":\"web_fetch-page\",\"uri\":\"tool-output://local/y\"},\"availability\":\"available\"}",
            ),
        );
        let provider = MuseProvider::new(dir.path().to_owned());
        let session = provider.load("abc123").expect("load");
        let texts = tools_of(&session);
        let marked = texts
            .iter()
            .find(|(summary, _)| summary == "fetch")
            .expect("call");
        assert_eq!(
            marked.1.as_deref(),
            Some("large output (web_fetch-page): tool-output://local/y")
        );
    }

    #[test]
    fn output_ref_replaces_provisional_partial() {
        let (dir, session) = fixture();
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(session.join("session.jsonl"))
            .expect("open");
        commit(&mut file, "c7", "big");
        link(&mut file, "t7", "c7");
        drop(file);
        append(
            &session,
            &task_event("{\"kind\":\"output\",\"task_id\":\"t7\",\"chunk\":\"early-bits\"}"),
        );
        append(
            &session,
            &task_event(
                "{\"kind\":\"tool_output_ref\",\"task_id\":\"t7\",\"output_ref\":{\"kind\":\"blob\",\"uri\":\"tool-output://local/z\"},\"availability\":\"available\"}",
            ),
        );
        let provider = MuseProvider::new(dir.path().to_owned());
        let session = provider.load("abc123").expect("load");
        let texts = tools_of(&session);
        let big = texts
            .iter()
            .find(|(summary, _)| summary == "big")
            .expect("call");
        assert_eq!(
            big.1.as_deref(),
            Some("large output (blob): tool-output://local/z")
        );
    }

    #[test]
    fn empty_final_settles_against_strays() {
        let (dir, session) = fixture();
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(session.join("session.jsonl"))
            .expect("open");
        commit(&mut file, "c7", "quiet");
        link(&mut file, "t7", "c7");
        drop(file);
        append(
            &session,
            &task_event(
                "{\"kind\":\"output\",\"task_id\":\"t7\",\"final_result\":true,\"chunk\":\"\"}",
            ),
        );
        append(
            &session,
            &task_event("{\"kind\":\"output\",\"task_id\":\"t7\",\"chunk\":\"stray\"}"),
        );
        append(
            &session,
            &task_event(
                "{\"kind\":\"tool_output_ref\",\"task_id\":\"t7\",\"output_ref\":{\"kind\":\"blob\",\"uri\":\"tool-output://local/s\"},\"availability\":\"available\"}",
            ),
        );
        let provider = MuseProvider::new(dir.path().to_owned());
        let session = provider.load("abc123").expect("load");
        let texts = tools_of(&session);
        let quiet = texts
            .iter()
            .find(|(summary, _)| summary == "quiet")
            .expect("call");
        assert_eq!(quiet.1, None);
    }

    #[test]
    fn ref_without_availability_still_marks() {
        let (dir, session) = fixture();
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(session.join("session.jsonl"))
            .expect("open");
        commit(&mut file, "c7", "legacy");
        link(&mut file, "t7", "c7");
        drop(file);
        append(
            &session,
            &task_event(
                "{\"kind\":\"tool_output_ref\",\"task_id\":\"t7\",\"output_ref\":{\"kind\":\"blob\",\"uri\":\"tool-output://local/old\"}}",
            ),
        );
        let provider = MuseProvider::new(dir.path().to_owned());
        let session = provider.load("abc123").expect("load");
        let texts = tools_of(&session);
        let legacy = texts
            .iter()
            .find(|(summary, _)| summary == "legacy")
            .expect("call");
        assert_eq!(
            legacy.1.as_deref(),
            Some("large output (blob): tool-output://local/old")
        );
    }

    #[test]
    fn unavailable_ref_is_ignored() {
        let (dir, session) = fixture();
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(session.join("session.jsonl"))
            .expect("open");
        commit(&mut file, "c7", "gone");
        link(&mut file, "t7", "c7");
        drop(file);
        append(
            &session,
            &task_event(
                "{\"kind\":\"tool_output_ref\",\"task_id\":\"t7\",\"output_ref\":{\"kind\":\"blob\",\"uri\":\"tool-output://local/g\"},\"availability\":\"expired\"}",
            ),
        );
        let provider = MuseProvider::new(dir.path().to_owned());
        let session = provider.load("abc123").expect("load");
        let texts = tools_of(&session);
        let gone = texts
            .iter()
            .find(|(summary, _)| summary == "gone")
            .expect("call");
        assert_eq!(gone.1, None);
    }

    #[test]
    fn recommitted_call_renders_once() {
        let (dir, session) = fixture();
        append(
            &session,
            &run_event(
                "{\"kind\":\"assistant_tool_calls_committed\",\"tool_calls\":[{\"call_id\":\"c1\",\"name\":\"bash\",\"args\":\"{\\\"command\\\":\\\"again\\\"}\"}]}",
            ),
        );
        append(
            &session,
            &run_event(
                "{\"kind\":\"assistant_tool_calls_committed\",\"tool_calls\":[{\"call_id\":\"c9\",\"name\":\"bash\",\"args\":\"{\\\"command\\\":\\\"first\\\"}\"},{\"call_id\":\"c9\",\"name\":\"bash\",\"args\":\"{\\\"command\\\":\\\"second\\\"}\"}]}",
            ),
        );
        let provider = MuseProvider::new(dir.path().to_owned());
        let session = provider.load("abc123").expect("load");
        assert_eq!(
            tools_of(&session),
            [
                ("ls".to_owned(), Some("done".to_owned())),
                ("first".to_owned(), None)
            ]
        );
    }

    #[test]
    fn object_form_args_summarized() {
        let (dir, session) = fixture();
        append(
            &session,
            &run_event(
                "{\"kind\":\"assistant_tool_calls_committed\",\"tool_calls\":[{\"call_id\":\"c9\",\"name\":\"read_file\",\"args\":{\"path\":\"a.txt\"}}]}",
            ),
        );
        let provider = MuseProvider::new(dir.path().to_owned());
        let session = provider.load("abc123").expect("load");
        let texts = tools_of(&session);
        let found = texts
            .iter()
            .find(|(summary, _)| summary == "a.txt")
            .expect("call");
        assert_eq!(found.1, None);
    }

    fn write_log(root: &std::path::Path, rel: &str, lines: &[&str]) {
        let dir = root.join(rel);
        std::fs::create_dir_all(&dir).expect("mkdirs");
        std::fs::write(dir.join("session.jsonl"), lines.join("\n")).expect("write");
    }

    #[test]
    fn duplicate_id_resolves_to_larger_log() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_log(
            dir.path(),
            "2026/09/13/aaa/dup-id",
            &[&run_event("{\"kind\":\"started\",\"prompt\":\"stub\"}")],
        );
        write_log(
            dir.path(),
            "2026/09/13/zzz/dup-id",
            &[
                &run_event("{\"kind\":\"started\",\"prompt\":\"full-a\"}"),
                &run_event("{\"kind\":\"started\",\"prompt\":\"full-b\"}"),
            ],
        );
        let provider = MuseProvider::new(dir.path().to_owned());
        let metas = provider.sessions().expect("sessions");
        assert_eq!(metas.len(), 1);
        let session = provider.load("dup-id").expect("load");
        let texts = text_blocks(&session);
        assert!(texts.contains(&"full-a"));
        assert!(texts.contains(&"full-b"));
        assert!(!texts.contains(&"stub"));
    }

    #[test]
    fn duplicate_id_tie_prefers_newer_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_log(
            dir.path(),
            "2026/09/13/aaa/dup-id",
            &[&run_event("{\"kind\":\"started\",\"prompt\":\"s1\"}")],
        );
        write_log(
            dir.path(),
            "2026/09/13/zzz/dup-id",
            &[&run_event("{\"kind\":\"started\",\"prompt\":\"s2\"}")],
        );
        let provider = MuseProvider::new(dir.path().to_owned());
        let session = provider.load("dup-id").expect("load");
        let texts = text_blocks(&session);
        assert!(texts.contains(&"s2"));
        assert!(!texts.contains(&"s1"));
    }

    #[test]
    fn model_reconfigure_sets_model() {
        let (dir, session) = fixture();
        append(
            &session,
            "{\"payload_type\":\"runtime.model_reconfigure.completed\",\"payload\":{\"kind\":\"model_reconfigure\",\"record\":{\"effective\":{\"model_id\":\"muse-spark-9.9\"}}}}",
        );
        let provider = MuseProvider::new(dir.path().to_owned());
        let session = provider.load("abc123").expect("load");
        assert_eq!(session.meta.model, "muse-spark-9.9");
    }

    #[test]
    fn looks_failed_flags_failures() {
        assert!(looks_failed("{\"exit_code\": 128, \"output\": \"x\"}"));
        assert!(looks_failed("{\"exit_code\":1}"));
        assert!(looks_failed("{\"exit_code\": \"1\"}"));
        assert!(looks_failed("{\"exit_code\": \" 128 \"}"));
        assert!(looks_failed("{\"exit_code\": \"5.0\"}"));
        assert!(!looks_failed("{\"exit_code\": \"5.\"}"));
        assert!(!looks_failed("{\"exit_code\": 0.5}"));
        assert!(looks_failed(
            "{\"exit_code\": 0, \"terminal_status\": \"failed\"}"
        ));
        assert!(!looks_failed("{\"exit_code\": 0, \"output\": \"x\"}"));
        assert!(!looks_failed("{\"exit_code\": \"0\"}"));
        assert!(!looks_failed("plain transcript"));
        assert!(looks_failed("{\"terminal_status\": \"failed\"}"));
        assert!(looks_failed("{\"terminal_status\":\"failed\"}"));
        assert!(looks_failed("Build failed {\"exit_code\":1}"));
        assert!(looks_failed("{\"exit_code\": 1.0}"));
        assert!(!looks_failed("{\"exit_code\": 0.0}"));
        assert!(looks_failed("Build failed {\"exit_code\":2}"));
        assert!(looks_failed("Build failed {\"exit_code\":-1}"));
        assert!(looks_failed("{\"exit_code\": 01}"));
        assert!(looks_failed("Build failed {\"exit_code\": \"1\"}"));
        assert!(!looks_failed("Build ok {\"exit_code\":1.5}"));
        assert!(!looks_failed("{\"terminal_status\": \"fail ed\"}"));
        assert!(looks_failed("(\"exit_code\": 5)"));
        assert!(looks_failed("cmd failed \"exit_code\": 5."));
        assert!(looks_failed("\"exit_code\": 5;"));
        assert!(!looks_failed("\"exit_code\": 5.5.5"));
        assert!(!looks_failed("log \"exit_code\": \"0\" end"));
        assert!(looks_failed("log \"exit_code\": \"5.0\" end"));
        assert!(looks_failed("\"exit_code\":\"n/a\",\"exit_code\":\"3\""));
        assert!(looks_failed("\"exit_code\":\"5.\",\"exit_code\":\"9\""));
        assert!(looks_failed("\"exit_code\":\"1e\",\"exit_code\":\"2\""));
        assert!(looks_failed("log \"exit_code\": \" 3 \" end"));
        assert!(looks_failed("log \"exit_code\": \"-1\" end"));
        assert!(looks_failed("\"exit_code\":\"+1\""));
        assert!(!looks_failed("log \"exit_code\": \" +0 \" end"));
        assert!(!looks_failed("\"exit_code\":\"-+1\""));
    }

    #[test]
    fn empty_children_falls_through_to_payload() {
        let (dir, session) = fixture();
        append(
            &session,
            "{\"children\":[],\"payload_type\":\"runtime.session\",\"payload\":{\"kind\":\"run\",\"event\":{\"kind\":\"started\",\"prompt\":\"framed\"}}}",
        );
        let provider = MuseProvider::new(dir.path().to_owned());
        let session = provider.load("abc123").expect("load");
        let texts = text_blocks(&session);
        assert!(texts.contains(&"framed"));
    }

    #[test]
    fn summarize_args_prefers_salient_keys() {
        assert_eq!(
            summarize_args("{\"path\":\"src/main.rs\",\"limit\":5}"),
            "src/main.rs"
        );
        assert_eq!(
            summarize_args("{\"url\":\"https://example.com\"}"),
            "https://example.com"
        );
        assert_eq!(
            summarize_args("{\"command\":\"\",\"path\":\"fallback.rs\"}"),
            "fallback.rs"
        );
        assert_eq!(summarize_args("{\"todos\":[]}"), "{\"todos\":[]}");
    }

    #[test]
    fn skips_malformed_lines() {
        let (dir, session) = fixture();
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(session.join("session.jsonl"))
            .expect("open");
        writeln!(file, "not json at all").expect("write");
        writeln!(file, "{{\"children\":[{{\"record_json\":\"broken\"}}]}}").expect("write");
        let provider = MuseProvider::new(dir.path().to_owned());
        let session = provider.load("abc123").expect("load");
        assert_eq!(session.turns.len(), 2);
        let metas = provider.sessions().expect("sessions");
        assert_eq!(metas.len(), 1);
    }
}
