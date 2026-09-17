use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde_json::Value;
use walkdir::WalkDir;

use crate::core::{Block, Provider, Role, Session, SessionMeta, ToolCall, Turn, truncate};

pub struct CodexProvider {
    root: PathBuf,
}

impl CodexProvider {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn default_root() -> PathBuf {
        if let Ok(dir) = std::env::var("CODEX_HOME") {
            return PathBuf::from(dir).join("sessions");
        }
        std::env::var("HOME")
            .map(|home| PathBuf::from(home).join(".codex/sessions"))
            .unwrap_or_else(|_| PathBuf::from("."))
    }

    fn session_id(path: &Path) -> Option<String> {
        let stem = path.file_stem()?.to_str()?;
        let id = stem.get(stem.len().checked_sub(36)?..)?;
        let bytes = id.as_bytes();
        if [8, 13, 18, 23]
            .iter()
            .all(|&at| bytes.get(at) == Some(&b'-'))
        {
            Some(id.to_owned())
        } else {
            None
        }
    }

    fn candidates(&self) -> Vec<PathBuf> {
        let mut paths: Vec<PathBuf> = WalkDir::new(&self.root)
            .into_iter()
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.into_path())
            .filter(|path| {
                path.extension().is_some_and(|ext| ext == "jsonl")
                    && path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.starts_with("rollout-"))
                    && Self::session_id(path).is_some()
            })
            .collect();
        paths.sort();
        paths
    }

    fn log_path(&self, id: &str) -> Option<PathBuf> {
        self.candidates()
            .into_iter()
            .filter(|path| Self::session_id(path).is_some_and(|found| found == id))
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
}

impl Provider for CodexProvider {
    fn tool_id(&self) -> &'static str {
        "codex"
    }

    fn sessions(&self) -> Result<Vec<SessionMeta>> {
        let mut metas = Vec::new();
        let mut seen = HashSet::new();
        for path in self.candidates() {
            if let Some(id) = Self::session_id(&path)
                && seen.insert(id.clone())
                && let Ok(session) = self.load(&id)
                && (session.meta.turns > 0 || session.meta.title != session.meta.id)
            {
                metas.push(session.meta);
            }
        }
        Ok(metas)
    }

    fn load(&self, id: &str) -> Result<Session> {
        let path = self.log_path(id).ok_or_else(|| {
            anyhow::anyhow!("session {id} not found under {}", self.root.display())
        })?;
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let mut fold = Fold::new(id);
        for line in text.lines() {
            let Ok(entry) = serde_json::from_str::<Value>(line) else {
                continue;
            };
            fold.apply(&entry);
        }
        Ok(fold.finish())
    }
}

struct Fold {
    id: String,
    first_prompt: Option<String>,
    workspace: String,
    model: String,
    turns: Vec<Turn>,
    calls: HashMap<String, (usize, usize)>,
    pending: HashMap<String, Vec<String>>,
    consumed: HashSet<String>,
}

impl Fold {
    fn new(id: &str) -> Self {
        Self {
            id: id.to_owned(),
            first_prompt: None,
            workspace: String::new(),
            model: String::new(),
            turns: Vec::new(),
            calls: HashMap::new(),
            pending: HashMap::new(),
            consumed: HashSet::new(),
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
        if text.is_empty() || duplicate {
            return;
        }
        if self.first_prompt.is_none() {
            self.first_prompt = Some(truncate(text, 80));
        }
        self.turns.push(Turn {
            role: Role::User,
            blocks: vec![Block::Text(text.to_owned())],
        });
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

    fn assistant_text(&mut self, text: &str) {
        if !text.is_empty() {
            self.assistant_turn()
                .blocks
                .push(Block::Text(text.to_owned()));
        }
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

    fn push_tool(&mut self, call: ToolCall) {
        self.assistant_turn().blocks.push(Block::Tools(vec![call]));
    }

    fn absorb_execution(&mut self, call: &ToolCall) -> bool {
        if let Some(queue) = self.pending.get_mut(call.summary.as_str()) {
            queue.retain(|id| !self.consumed.contains(id));
        }
        let Some(call_id) = self
            .pending
            .get(call.summary.as_str())
            .and_then(|queue| queue.first().cloned())
        else {
            return false;
        };
        let Some(pending) = self.find_call(&call_id) else {
            return false;
        };
        if let Some(result) = &call.result {
            pending.result = Some(result.clone());
        }
        pending.failed |= call.failed;
        self.consumed.insert(call_id.clone());
        if let Some(queue) = self.pending.get_mut(call.summary.as_str()) {
            queue.retain(|id| id != &call_id);
        }
        true
    }

    fn register_call(&mut self, call_id: &str, call: ToolCall) {
        let turn_index = self.turn_index();
        let base = self.turns[turn_index]
            .blocks
            .iter()
            .filter_map(|block| match block {
                Block::Tools(calls) => Some(calls.len()),
                _ => None,
            })
            .sum::<usize>();
        self.assistant_turn().blocks.push(Block::Tools(vec![call]));
        self.calls.insert(call_id.to_owned(), (turn_index, base));
    }

    fn turn_index(&mut self) -> usize {
        self.assistant_turn();
        self.turns.len() - 1
    }

    fn apply(&mut self, entry: &Value) {
        match entry.get("type").and_then(Value::as_str) {
            Some("session_meta") => {
                if let Some(workspace) = entry
                    .get("payload")
                    .and_then(|payload| payload.get("cwd"))
                    .and_then(Value::as_str)
                    && !workspace.is_empty()
                {
                    self.workspace = workspace.to_owned();
                }
            }
            Some("turn_context") => {
                let payload = entry.get("payload").unwrap_or(&Value::Null);
                if self.workspace.is_empty()
                    && let Some(workspace) = payload.get("cwd").and_then(Value::as_str)
                    && !workspace.is_empty()
                {
                    self.workspace = workspace
                        .strip_prefix("file://")
                        .unwrap_or(workspace)
                        .to_owned();
                }
                if let Some(model) = payload.get("model").and_then(Value::as_str)
                    && !model.is_empty()
                {
                    self.model = model.to_owned();
                }
            }
            Some("response_item") => {
                self.apply_response(entry.get("payload").unwrap_or(&Value::Null));
            }
            Some("event_msg") => {
                self.apply_event(entry.get("payload").unwrap_or(&Value::Null));
            }
            Some("token_usage_record") => {
                let usage = entry
                    .get("payload")
                    .and_then(|payload| payload.get("usage"))
                    .unwrap_or(&Value::Null);
                let input = usage
                    .get("input_tokens")
                    .and_then(Value::as_u64)
                    .unwrap_or_default();
                let output = usage
                    .get("output_tokens")
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

    fn apply_response(&mut self, payload: &Value) {
        match payload.get("type").and_then(Value::as_str) {
            Some("message") => match payload.get("role").and_then(Value::as_str) {
                Some("user") => {
                    let text = texts(payload.get("content").unwrap_or(&Value::Null));
                    if !injected(&text) {
                        self.user_turn(&text);
                    }
                }
                Some("assistant") => {
                    self.assistant_text(&texts(payload.get("content").unwrap_or(&Value::Null)));
                }
                _ => {}
            },
            Some("function_call") => {
                let call_id = payload
                    .get("call_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if call_id.is_empty() || self.calls.contains_key(call_id) {
                    return;
                }
                let name = payload.get("name").and_then(Value::as_str).unwrap_or("?");
                let summary = summarize_arguments(
                    payload
                        .get("arguments")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                );
                self.register_call(
                    call_id,
                    ToolCall {
                        name: name.to_owned(),
                        summary: summary.clone(),
                        result: None,
                        failed: false,
                    },
                );
                self.pending
                    .entry(summary)
                    .or_default()
                    .push(call_id.to_owned());
            }
            Some("function_call_output") => {
                let call_id = payload
                    .get("call_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let output = payload
                    .get("output")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if call_id.is_empty() || output.is_empty() {
                    return;
                }
                if self.consumed.contains(call_id) {
                    if let Some(call) = self.find_call(call_id) {
                        call.failed |= exited_nonzero(output);
                    }
                    return;
                }
                if let Some(call) = self.find_call(call_id) {
                    call.result = Some(truncate(output, 6000));
                    call.failed |= exited_nonzero(output);
                }
            }
            _ => {}
        }
    }

    fn apply_event(&mut self, payload: &Value) {
        if payload.get("type").and_then(Value::as_str) != Some("item_completed") {
            return;
        }
        let item = payload.get("item").unwrap_or(&Value::Null);
        match item.get("type").and_then(Value::as_str) {
            Some("UserMessage") => {
                let text = texts(item.get("content").unwrap_or(&Value::Null));
                if !injected(&text) {
                    self.user_turn(&text);
                }
            }
            Some("AgentMessage") => {
                self.assistant_text(&texts(item.get("content").unwrap_or(&Value::Null)));
            }
            Some("CommandExecution") => {
                if let Some(call) = execution_tool(item)
                    && !self.absorb_execution(&call)
                {
                    self.push_tool(call);
                }
            }
            Some("Extension") => {
                if let Some(call) = extension_tool(item) {
                    self.push_tool(call);
                }
            }
            _ => {}
        }
    }

    fn finish(self) -> Session {
        let title = self.first_prompt.unwrap_or_else(|| self.id.clone());
        let turns = self.turns.len();
        Session {
            meta: SessionMeta {
                id: self.id,
                tool: "codex",
                title,
                workspace: self.workspace,
                model: self.model,
                turns,
            },
            turns: self.turns,
        }
    }
}

fn texts(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(|block| block.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn injected(text: &str) -> bool {
    const PREFIXES: [&str; 9] = [
        "<environment_context>",
        "<user_instructions>",
        "<permissions",
        "<skills_instructions>",
        "<collaboration_mode>",
        "<turn_aborted>",
        "<recommended_plugins>",
        "# AGENTS.md instructions",
        "Whenever you create or amend a Git commit",
    ];
    let trimmed = text.trim_start();
    PREFIXES.iter().any(|prefix| trimmed.starts_with(prefix))
}

fn summarize_arguments(args: &str) -> String {
    if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(args)
        && let Some(cmd) = map
            .get("cmd")
            .or_else(|| map.get("command"))
            .and_then(Value::as_str)
        && !cmd.is_empty()
    {
        return truncate(cmd, 300);
    }
    truncate(args, 300)
}

fn exited_nonzero(text: &str) -> bool {
    const MARKER: &str = "exited with code ";
    let mut rest = text;
    while let Some(at) = rest.find(MARKER) {
        let digits = rest[at + MARKER.len()..]
            .trim_start()
            .chars()
            .take_while(|char| char.is_ascii_digit())
            .collect::<String>();
        if digits.parse::<i64>().is_ok_and(|code| code != 0) {
            return true;
        }
        rest = &rest[at + 1..];
    }
    false
}

fn execution_tool(item: &Value) -> Option<ToolCall> {
    let parsed = item.get("parsed_cmd").and_then(Value::as_array);
    let summary = parsed
        .and_then(|commands| commands.first())
        .and_then(|command| command.get("cmd"))
        .and_then(Value::as_str)
        .or_else(|| {
            item.get("command")
                .and_then(Value::as_array)
                .and_then(|command| command.last())
                .and_then(Value::as_str)
        })
        .unwrap_or_default();
    if summary.is_empty() {
        return None;
    }
    let stdout = item
        .get("stdout")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let stderr = item
        .get("stderr")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let fallback = item
        .get("aggregated_output")
        .or_else(|| item.get("formatted_output"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let output = if stdout.is_empty() {
        if stderr.is_empty() { fallback } else { stderr }
    } else {
        stdout
    };
    let failed = item
        .get("status")
        .and_then(Value::as_str)
        .is_some_and(|status| status != "completed")
        || item
            .get("exit_code")
            .and_then(Value::as_i64)
            .is_some_and(|code| code != 0);
    Some(ToolCall {
        name: "exec".to_owned(),
        summary: truncate(summary, 300),
        result: (!output.is_empty()).then(|| truncate(output, 6000)),
        failed,
    })
}

fn extension_tool(item: &Value) -> Option<ToolCall> {
    let name = item
        .get("kind")
        .and_then(Value::as_str)
        .unwrap_or("extension");
    let summary = item
        .get("query")
        .and_then(Value::as_str)
        .or_else(|| {
            item.get("action")
                .and_then(|action| action.get("url"))
                .and_then(Value::as_str)
        })
        .unwrap_or_default();
    if summary.is_empty() {
        return None;
    }
    let result = item
        .get("results")
        .and_then(Value::as_array)
        .map(|results| {
            results
                .iter()
                .filter_map(|result| {
                    result
                        .get("snippet")
                        .or_else(|| result.get("text"))
                        .and_then(Value::as_str)
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default();
    Some(ToolCall {
        name: name.to_owned(),
        summary: truncate(summary, 300),
        result: (!result.is_empty()).then(|| truncate(&result, 6000)),
        failed: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    const ID: &str = "019beb04-467d-7bf2-9772-db918251fc4d";

    fn layout(dir: &tempfile::TempDir) -> PathBuf {
        let bucket = dir.path().join("2026/01/23");
        std::fs::create_dir_all(&bucket).expect("mkdirs");
        bucket
    }

    fn rollout(bucket: &Path, name: &str) -> std::fs::File {
        std::fs::File::create(bucket.join(name)).expect("create")
    }

    fn fixture() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().expect("tempdir");
        let bucket = layout(&dir);
        let mut file = rollout(&bucket, &format!("rollout-2026-01-23T22-21-24-{ID}.jsonl"));
        writeln!(
            file,
            "{{\"timestamp\":\"2026-01-23T13:21:24.605Z\",\"type\":\"session_meta\",\"payload\":{{\"id\":\"{ID}\",\"cwd\":\"/repo/seonbi\"}}}}"
        )
        .expect("write");
        writeln!(
            file,
            "{{\"type\":\"response_item\",\"payload\":{{\"type\":\"message\",\"role\":\"user\",\"content\":[{{\"type\":\"input_text\",\"text\":\"<environment_context>\\n  <cwd>/repo/seonbi</cwd>\\n</environment_context>\"}}]}}}}"
        )
        .expect("write");
        writeln!(
            file,
            "{{\"type\":\"turn_context\",\"payload\":{{\"cwd\":\"/repo/seonbi\",\"model\":\"gpt-5.3-codex\"}}}}"
        )
        .expect("write");
        writeln!(
            file,
            "{{\"type\":\"response_item\",\"payload\":{{\"type\":\"message\",\"role\":\"user\",\"content\":[{{\"type\":\"input_text\",\"text\":\"check the plan\"}}]}}}}"
        )
        .expect("write");
        writeln!(
            file,
            "{{\"type\":\"event_msg\",\"payload\":{{\"type\":\"item_completed\",\"item\":{{\"type\":\"UserMessage\",\"content\":[{{\"type\":\"text\",\"text\":\"check the plan\"}}]}}}}}}"
        )
        .expect("write");
        writeln!(
            file,
            "{{\"type\":\"response_item\",\"payload\":{{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{{\"type\":\"output_text\",\"text\":\"On it\"}}]}}}}"
        )
        .expect("write");
        writeln!(
            file,
            "{{\"type\":\"response_item\",\"payload\":{{\"type\":\"function_call\",\"name\":\"exec_command\",\"arguments\":\"{{\\\"cmd\\\":\\\"wc -l plan.md\\\"}}\",\"call_id\":\"call_1\"}}}}"
        )
        .expect("write");
        writeln!(
            file,
            "{{\"type\":\"response_item\",\"payload\":{{\"type\":\"function_call_output\",\"call_id\":\"call_1\",\"output\":\"Chunk ID: a\\nProcess exited with code 0\\nOutput:\\n10 plan.md\"}}}}"
        )
        .expect("write");
        writeln!(
            file,
            "{{\"type\":\"response_item\",\"payload\":{{\"type\":\"function_call\",\"name\":\"exec_command\",\"arguments\":\"{{\\\"cmd\\\":\\\"boom\\\"}}\",\"call_id\":\"call_2\"}}}}"
        )
        .expect("write");
        writeln!(
            file,
            "{{\"type\":\"response_item\",\"payload\":{{\"type\":\"function_call_output\",\"call_id\":\"call_2\",\"output\":\"Process exited with code 1\\nOutput:\\nnope\"}}}}"
        )
        .expect("write");
        (dir, bucket)
    }

    fn tools_of(session: &Session) -> Vec<&ToolCall> {
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
    fn folds_prompt_assistant_and_old_style_tools() {
        let (dir, _) = fixture();
        let session = CodexProvider::new(dir.path().to_owned())
            .load(ID)
            .expect("load");
        assert_eq!(session.meta.tool, "codex");
        assert_eq!(session.meta.title, "check the plan");
        assert_eq!(session.meta.workspace, "/repo/seonbi");
        assert_eq!(session.meta.model, "gpt-5.3-codex");
        assert!(matches!(session.turns[0].role, Role::User));
        let users = session
            .turns
            .iter()
            .filter(|turn| matches!(turn.role, Role::User))
            .count();
        assert_eq!(users, 1);
        let tools = tools_of(&session);
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].summary, "wc -l plan.md");
        assert!(!tools[0].failed);
        assert!(tools[1].failed);
    }

    #[test]
    fn folds_command_execution_and_usage() {
        let (dir, bucket) = fixture();
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(bucket.join(format!("rollout-2026-01-23T22-21-24-{ID}.jsonl")))
            .expect("open");
        writeln!(
            file,
            "{{\"type\":\"event_msg\",\"payload\":{{\"type\":\"item_completed\",\"item\":{{\"type\":\"CommandExecution\",\"command\":[\"/bin/zsh\",\"-lc\",\"nl -ba run.py\"],\"parsed_cmd\":[{{\"type\":\"read\",\"cmd\":\"nl -ba run.py\",\"name\":\"run.py\"}}],\"status\":\"completed\",\"stdout\":\"1\\tprint\",\"exit_code\":0}}}}}}"
        )
        .expect("write");
        writeln!(
            file,
            "{{\"type\":\"token_usage_record\",\"payload\":{{\"usage\":{{\"input_tokens\":10,\"output_tokens\":5}}}}}}"
        )
        .expect("write");
        let provider = CodexProvider::new(dir.path().to_owned());
        let session = provider.load(ID).expect("load");
        let tools = tools_of(&session);
        let read = tools
            .iter()
            .find(|call| call.summary == "nl -ba run.py")
            .expect("call");
        assert_eq!(read.name, "exec");
        assert_eq!(read.result.as_deref(), Some("1\tprint"));
        assert!(!read.failed);
        let usage = session
            .turns
            .iter()
            .flat_map(|turn| turn.blocks.iter())
            .any(|block| {
                matches!(
                    block,
                    Block::Usage {
                        input: 10,
                        output: 5
                    }
                )
            });
        assert!(usage);
    }

    #[test]
    fn command_execution_absorbs_matching_function_call() {
        let (dir, bucket) = fixture();
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(bucket.join(format!("rollout-2026-01-23T22-21-24-{ID}.jsonl")))
            .expect("open");
        writeln!(
            file,
            "{{\"type\":\"response_item\",\"payload\":{{\"type\":\"function_call\",\"name\":\"exec_command\",\"arguments\":\"{{\\\"cmd\\\":\\\"pwd\\\"}}\",\"call_id\":\"call_9\"}}}}"
        )
        .expect("write");
        writeln!(
            file,
            "{{\"type\":\"event_msg\",\"payload\":{{\"type\":\"item_completed\",\"item\":{{\"type\":\"CommandExecution\",\"command\":[\"/bin/zsh\",\"-lc\",\"pwd\"],\"status\":\"completed\",\"stdout\":\"/repo/seonbi\",\"exit_code\":0}}}}}}"
        )
        .expect("write");
        writeln!(
            file,
            "{{\"type\":\"response_item\",\"payload\":{{\"type\":\"function_call_output\",\"call_id\":\"call_9\",\"output\":\"Command: /bin/zsh -lc pwd\\nProcess exited with code 0\\nOutput:\\n/repo/seonbi\"}}}}"
        )
        .expect("write");
        writeln!(
            file,
            "{{\"type\":\"event_msg\",\"payload\":{{\"type\":\"item_completed\",\"item\":{{\"type\":\"CommandExecution\",\"command\":[\"/bin/zsh\",\"-lc\",\"mystery\"],\"stdout\":\"?\"}}}}}}"
        )
        .expect("write");
        let provider = CodexProvider::new(dir.path().to_owned());
        let session = provider.load(ID).expect("load");
        let tools = tools_of(&session);
        let pwd: Vec<&&ToolCall> = tools.iter().filter(|call| call.summary == "pwd").collect();
        assert_eq!(pwd.len(), 1);
        assert_eq!(pwd[0].result.as_deref(), Some("/repo/seonbi"));
        assert!(!pwd[0].failed);
        let mystery = tools
            .iter()
            .find(|call| call.summary == "mystery")
            .expect("call");
        assert!(!mystery.failed);
    }

    #[test]
    fn lists_only_nonempty_sessions() {
        let (dir, bucket) = fixture();
        let _ = rollout(
            &bucket,
            "rollout-2026-01-23T22-22-00-019c220c-ef5c-7750-a9e3-a831c9b705b2.jsonl",
        );
        let provider = CodexProvider::new(dir.path().to_owned());
        let metas = provider.sessions().expect("sessions");
        assert_eq!(metas.len(), 1);
        assert_eq!(metas[0].id, ID);
    }

    #[test]
    fn duplicate_id_resolves_to_larger_log() {
        let dir = tempfile::tempdir().expect("tempdir");
        let small = layout(&dir);
        std::fs::write(
            small.join(format!("rollout-2026-01-23T22-21-24-{ID}.jsonl")),
            format!(
                "{{\"type\":\"session_meta\",\"payload\":{{\"id\":\"{ID}\",\"cwd\":\"/stub\"}}}}\n"
            ),
        )
        .expect("write");
        let big = dir.path().join("2026/01/24");
        std::fs::create_dir_all(&big).expect("mkdirs");
        std::fs::write(
            big.join(format!("rollout-2026-01-24T10-00-00-{ID}.jsonl")),
            format!(
                "{{\"type\":\"session_meta\",\"payload\":{{\"id\":\"{ID}\",\"cwd\":\"/full\"}}}}\n{{\"type\":\"event_msg\",\"payload\":{{\"type\":\"item_completed\",\"item\":{{\"type\":\"UserMessage\",\"content\":[{{\"type\":\"text\",\"text\":\"full prompt\"}}]}}}}}}\n"
            ),
        )
        .expect("write");
        let provider = CodexProvider::new(dir.path().to_owned());
        let session = provider.load(ID).expect("load");
        assert_eq!(session.meta.workspace, "/full");
        assert_eq!(session.meta.title, "full prompt");
    }

    #[test]
    fn skips_malformed_lines() {
        let (dir, bucket) = fixture();
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(bucket.join(format!("rollout-2026-01-23T22-21-24-{ID}.jsonl")))
            .expect("open");
        writeln!(file, "not json at all").expect("write");
        let provider = CodexProvider::new(dir.path().to_owned());
        let session = provider.load(ID).expect("load");
        assert_eq!(session.meta.title, "check the plan");
    }
}
