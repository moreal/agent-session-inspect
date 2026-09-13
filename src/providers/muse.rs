use std::collections::HashMap;
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

    fn log_path(&self, id: &str) -> Option<PathBuf> {
        WalkDir::new(&self.root)
            .into_iter()
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.into_path())
            .find(|path| {
                path.file_name().is_some_and(|name| name == "session.jsonl")
                    && path
                        .parent()
                        .is_some_and(|parent| parent.file_name().is_some_and(|name| name == id))
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
            if let Some(children) = line.get("children").and_then(Value::as_array) {
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
        for entry in WalkDir::new(&self.root)
            .into_iter()
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_name().to_str() == Some("session.jsonl"))
        {
            if let Some(id) = entry
                .path()
                .parent()
                .and_then(|parent| parent.file_name())
                .and_then(|name| name.to_str())
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
                let calls = event
                    .get("tool_calls")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
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
                    let args = call.get("args").and_then(Value::as_str).unwrap_or_default();
                    self.calls
                        .insert(call_id.to_owned(), (turn_index, base + tools.len()));
                    tools.push(ToolCall {
                        name: name.to_owned(),
                        summary: summarize_args(args),
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
                    if let Some(call) = self.find_call(call_id) {
                        call.result = Some(unpack_result(text));
                        call.failed = looks_failed(text);
                    }
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
        if event.get("kind").and_then(Value::as_str) != Some("output") {
            return;
        }
        if event.get("final_result").and_then(Value::as_bool) != Some(true) {
            return;
        }
        let task_id = event
            .get("task_id")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let chunk = event
            .get("chunk")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if let Some(call_id) = self.tasks.get(task_id).cloned()
            && let Some(call) = self.find_call(&call_id)
        {
            call.result = Some(unpack_result(chunk));
            call.failed = looks_failed(chunk);
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

fn summarize_args(args: &str) -> String {
    let parsed: Value = serde_json::from_str(args).unwrap_or(Value::Null);
    let text = parsed
        .get("command")
        .or_else(|| parsed.get("description"))
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| args.to_owned());
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
    text.contains("\"exit_code\": 1") || text.contains("\"terminal_status\": \"failed\"")
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

    fn output(file: &mut std::fs::File, task_id: &str, chunk: &str) {
        writeln!(
            file,
            "{{\"payload_type\":\"tool_batch.effect.started\",\"payload\":{{\"record\":{{\"task_id\":\"{task_id}\",\"call_id\":\"{task_id}\"}}}}}}"
        )
        .expect("write");
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
        output(&mut file, "c2", "second");
        output(&mut file, "c3", "third");
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
