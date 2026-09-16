use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde_json::Value;
use walkdir::WalkDir;

use crate::core::{Block, Provider, Role, Session, SessionMeta, ToolCall, Turn, truncate};

pub struct ClaudeProvider {
    root: PathBuf,
}

impl ClaudeProvider {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn default_root() -> PathBuf {
        if let Ok(dir) = std::env::var("CLAUDE_CONFIG_DIR") {
            return PathBuf::from(dir).join("projects");
        }
        std::env::var("HOME")
            .map(|home| PathBuf::from(home).join(".claude/projects"))
            .unwrap_or_else(|_| PathBuf::from("."))
    }

    fn candidates(&self) -> Vec<PathBuf> {
        let mut paths: Vec<PathBuf> = WalkDir::new(&self.root)
            .into_iter()
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.into_path())
            .filter(|path| {
                path.extension().is_some_and(|ext| ext == "jsonl")
                    && path
                        .file_stem()
                        .is_some_and(|stem| stem.to_str().is_some_and(|stem| !stem.is_empty()))
            })
            .collect();
        paths.sort();
        paths
    }

    fn log_path(&self, id: &str) -> Option<PathBuf> {
        self.candidates()
            .into_iter()
            .filter(|path| {
                path.file_stem()
                    .and_then(|stem| stem.to_str())
                    .is_some_and(|stem| stem == id)
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
}

impl Provider for ClaudeProvider {
    fn tool_id(&self) -> &'static str {
        "claude"
    }

    fn sessions(&self) -> Result<Vec<SessionMeta>> {
        let mut metas = Vec::new();
        let mut seen = HashSet::new();
        for path in self.candidates() {
            if let Some(id) = path.file_stem().and_then(|stem| stem.to_str())
                && seen.insert(id.to_owned())
                && let Ok(session) = self.load(id)
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
    ai_title: Option<String>,
    summary: Option<String>,
    first_prompt: Option<String>,
    workspace: String,
    model: String,
    turns: Vec<Turn>,
    calls: HashMap<String, (usize, usize)>,
}

impl Fold {
    fn new(id: &str) -> Self {
        Self {
            id: id.to_owned(),
            ai_title: None,
            summary: None,
            first_prompt: None,
            workspace: String::new(),
            model: String::new(),
            turns: Vec::new(),
            calls: HashMap::new(),
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

    fn apply(&mut self, entry: &Value) {
        if entry.get("isSidechain").and_then(Value::as_bool) == Some(true) {
            return;
        }
        if self.workspace.is_empty()
            && let Some(workspace) = entry.get("cwd").and_then(Value::as_str)
            && !workspace.is_empty()
        {
            self.workspace = workspace.to_owned();
        }
        match entry.get("type").and_then(Value::as_str) {
            Some("ai-title") => {
                if let Some(title) = entry.get("aiTitle").and_then(Value::as_str)
                    && !title.is_empty()
                {
                    self.ai_title = Some(title.to_owned());
                }
            }
            Some("summary") => {
                if let Some(summary) = entry.get("summary").and_then(Value::as_str)
                    && !summary.is_empty()
                {
                    self.summary = Some(truncate(summary, 80));
                }
            }
            Some("user") => self.apply_user(entry),
            Some("assistant") => self.apply_assistant(entry),
            _ => {}
        }
    }

    fn apply_user(&mut self, entry: &Value) {
        let message = entry.get("message").unwrap_or(&Value::Null);
        let content = message.get("content").unwrap_or(&Value::Null);
        let mut results = Vec::new();
        if let Some(blocks) = content.as_array() {
            for block in blocks {
                if block.get("type").and_then(Value::as_str) == Some("tool_result") {
                    let id = block
                        .get("tool_use_id")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    if !id.is_empty() {
                        results.push((id.to_owned(), result_text(block)));
                    }
                }
            }
        }
        for (id, (text, failed)) in results {
            if let Some(call) = self.find_call(&id) {
                if !text.is_empty() {
                    call.result = Some(text);
                }
                call.failed |= failed;
            }
        }
        if entry.get("isMeta").and_then(Value::as_bool) == Some(true) {
            return;
        }
        if let Some(text) = prompt_text(content)
            && !text.is_empty()
        {
            self.user_turn(&text);
        }
    }

    fn apply_assistant(&mut self, entry: &Value) {
        let message = entry.get("message").unwrap_or(&Value::Null);
        let model = message
            .get("model")
            .and_then(Value::as_str)
            .or_else(|| entry.get("model").and_then(Value::as_str))
            .unwrap_or_default();
        if !model.is_empty() {
            self.model = model.to_owned();
        }
        let mut tools = Vec::new();
        if let Some(blocks) = message.get("content").and_then(Value::as_array) {
            for block in blocks {
                match block.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        let text = block
                            .get("text")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        if !text.is_empty() {
                            self.assistant_turn()
                                .blocks
                                .push(Block::Text(text.to_owned()));
                        }
                    }
                    Some("tool_use") => {
                        let id = block.get("id").and_then(Value::as_str).unwrap_or_default();
                        if id.is_empty() || self.calls.contains_key(id) {
                            continue;
                        }
                        let name = block.get("name").and_then(Value::as_str).unwrap_or("?");
                        let summary = block.get("input").map(summarize_input).unwrap_or_default();
                        tools.push((id.to_owned(), name.to_owned(), summary));
                    }
                    _ => {}
                }
            }
        }
        if !tools.is_empty() {
            let turn_index = self.turn_index();
            let base = self.turns[turn_index]
                .blocks
                .iter()
                .filter_map(|block| match block {
                    Block::Tools(calls) => Some(calls.len()),
                    _ => None,
                })
                .sum::<usize>();
            let calls = tools
                .into_iter()
                .enumerate()
                .map(|(offset, (id, name, summary))| {
                    self.calls.insert(id, (turn_index, base + offset));
                    ToolCall {
                        name,
                        summary,
                        result: None,
                        failed: false,
                    }
                })
                .collect();
            self.assistant_turn().blocks.push(Block::Tools(calls));
        }
        let input = message
            .get("usage")
            .and_then(|usage| usage.get("input_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or_default();
        let output = message
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

    fn turn_index(&mut self) -> usize {
        self.assistant_turn();
        self.turns.len() - 1
    }

    fn finish(self) -> Session {
        let title = self
            .ai_title
            .or(self.first_prompt)
            .or(self.summary)
            .unwrap_or_else(|| self.id.clone());
        let turns = self.turns.len();
        Session {
            meta: SessionMeta {
                id: self.id,
                tool: "claude",
                title,
                workspace: self.workspace,
                model: self.model,
                turns,
            },
            turns: self.turns,
        }
    }
}

fn prompt_text(content: &Value) -> Option<String> {
    match content {
        Value::String(text) => Some(text.clone()),
        Value::Array(blocks) => {
            let text = blocks
                .iter()
                .filter_map(|block| match block {
                    Value::String(text) => Some(text.as_str()),
                    Value::Object(_) => block.get("text").and_then(Value::as_str),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            Some(text)
        }
        _ => None,
    }
}

fn result_text(block: &Value) -> (String, bool) {
    let failed = block.get("is_error").and_then(Value::as_bool) == Some(true);
    let text = match block.get("content") {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(blocks)) => blocks
            .iter()
            .filter_map(|block| match block.get("type").and_then(Value::as_str) {
                Some("text") => block.get("text").and_then(Value::as_str),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Some(content) => serde_json::to_string(content).unwrap_or_default(),
    };
    (truncate(&text, 6000), failed)
}

const SALIENT_INPUT_KEYS: [&str; 8] = [
    "command",
    "description",
    "path",
    "url",
    "query",
    "pattern",
    "prompt",
    "skill",
];

fn summarize_input(input: &Value) -> String {
    match input {
        Value::Object(map) => SALIENT_INPUT_KEYS
            .into_iter()
            .filter_map(|key| map.get(key).and_then(Value::as_str))
            .find(|text| !text.is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| truncate(&input.to_string(), 300)),
        Value::String(text) => truncate(text, 300),
        _ => truncate(&input.to_string(), 300),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    const ID: &str = "943b9a19-8466-468c-a5f6-3a7c245e0a37";

    fn fixture() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().expect("tempdir");
        let bucket = dir.path().join("-Users-moreal-github-moreal-rxui");
        std::fs::create_dir_all(&bucket).expect("mkdirs");
        let mut file = std::fs::File::create(bucket.join(format!("{ID}.jsonl"))).expect("create");
        writeln!(
            file,
            "{{\"type\":\"user\",\"message\":{{\"role\":\"user\",\"content\":\"make it run every 3 hours\"}},\"cwd\":\"/repo/rxui\",\"sessionId\":\"{ID}\"}}"
        )
        .expect("write");
        writeln!(
            file,
            "{{\"type\":\"assistant\",\"message\":{{\"model\":\"claude-sonnet-5\",\"content\":[{{\"type\":\"thinking\",\"thinking\":\"plan\"}},{{\"type\":\"text\",\"text\":\"On it\"}},{{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"Bash\",\"input\":{{\"command\":\"ls\"}}}}],\"usage\":{{\"input_tokens\":10,\"output_tokens\":5}}}},\"cwd\":\"/repo/rxui\",\"sessionId\":\"{ID}\"}}"
        )
        .expect("write");
        writeln!(
            file,
            "{{\"type\":\"user\",\"message\":{{\"role\":\"user\",\"content\":[{{\"type\":\"tool_result\",\"tool_use_id\":\"toolu_1\",\"content\":\"ok\"}}]}}}}"
        )
        .expect("write");
        writeln!(
            file,
            "{{\"type\":\"assistant\",\"message\":{{\"model\":\"claude-sonnet-5\",\"content\":[{{\"type\":\"tool_use\",\"id\":\"toolu_2\",\"name\":\"Bash\",\"input\":{{\"command\":\"boom\"}}}}],\"usage\":{{\"input_tokens\":3,\"output_tokens\":2}}}},\"cwd\":\"/repo/rxui\",\"sessionId\":\"{ID}\"}}"
        )
        .expect("write");
        writeln!(
            file,
            "{{\"type\":\"user\",\"message\":{{\"role\":\"user\",\"content\":[{{\"type\":\"tool_result\",\"tool_use_id\":\"toolu_2\",\"content\":\"Exit code 1\",\"is_error\":true}}]}}}}"
        )
        .expect("write");
        writeln!(
            file,
            "{{\"type\":\"ai-title\",\"aiTitle\":\"Benchmark scheduling\",\"sessionId\":\"{ID}\"}}"
        )
        .expect("write");
        (dir, bucket)
    }

    fn append(bucket: &std::path::Path, line: &str) {
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(bucket.join(format!("{ID}.jsonl")))
            .expect("open");
        writeln!(file, "{line}").expect("write");
    }

    #[test]
    fn folds_prompts_tools_results_and_meta() {
        let (dir, _) = fixture();
        let session = ClaudeProvider::new(dir.path().to_owned())
            .load(ID)
            .expect("load");
        assert_eq!(session.meta.tool, "claude");
        assert_eq!(session.meta.title, "Benchmark scheduling");
        assert_eq!(session.meta.workspace, "/repo/rxui");
        assert_eq!(session.meta.model, "claude-sonnet-5");
        assert!(matches!(session.turns[0].role, Role::User));
        let tools: Vec<&ToolCall> = session
            .turns
            .iter()
            .flat_map(|turn| turn.blocks.iter())
            .filter_map(|block| match block {
                Block::Tools(calls) => Some(calls),
                _ => None,
            })
            .flatten()
            .collect();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].summary, "ls");
        assert_eq!(tools[0].result.as_deref(), Some("ok"));
        assert!(!tools[0].failed);
        assert!(tools[1].failed);
        let assistants = session
            .turns
            .iter()
            .filter(|turn| matches!(turn.role, Role::Assistant))
            .count();
        assert_eq!(assistants, 2);
        let usage = session
            .turns
            .iter()
            .flat_map(|turn| turn.blocks.iter())
            .filter_map(|block| match block {
                Block::Usage { input, output } => Some((*input, *output)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(usage, [(10, 5), (3, 2)]);
    }

    #[test]
    fn skips_meta_and_sidechain_prompts() {
        let (dir, bucket) = fixture();
        append(
            &bucket,
            "{\"type\":\"user\",\"isMeta\":true,\"message\":{\"role\":\"user\",\"content\":\"injected\"}}",
        );
        append(
            &bucket,
            "{\"type\":\"user\",\"isSidechain\":true,\"message\":{\"role\":\"user\",\"content\":\"subagent\"}}",
        );
        let session = ClaudeProvider::new(dir.path().to_owned())
            .load(ID)
            .expect("load");
        let texts = session
            .turns
            .iter()
            .flat_map(|turn| turn.blocks.iter())
            .filter_map(|block| match block {
                Block::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(
            !texts
                .iter()
                .any(|text| *text == "injected" || *text == "subagent")
        );
    }

    #[test]
    fn falls_back_to_first_prompt_for_title() {
        let (dir, bucket) = fixture();
        let path = bucket.join(format!("{ID}.jsonl"));
        let text = std::fs::read_to_string(&path).expect("read");
        let kept = text
            .lines()
            .filter(|line| !line.contains("ai-title"))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        std::fs::write(&path, kept).expect("write");
        let session = ClaudeProvider::new(dir.path().to_owned())
            .load(ID)
            .expect("load");
        assert_eq!(session.meta.title, "make it run every 3 hours");
    }

    #[test]
    fn lists_sessions() {
        let (dir, _) = fixture();
        let metas = ClaudeProvider::new(dir.path().to_owned())
            .sessions()
            .expect("sessions");
        assert_eq!(metas.len(), 1);
        assert_eq!(metas[0].id, ID);
        assert_eq!(metas[0].tool, "claude");
    }

    #[test]
    fn prefers_first_prompt_over_summary_for_title() {
        let (dir, bucket) = fixture();
        append(
            &bucket,
            "{\"type\":\"summary\",\"summary\":\"long compaction note nobody wants as a title\"}",
        );
        let path = bucket.join(format!("{ID}.jsonl"));
        let text = std::fs::read_to_string(&path).expect("read");
        let kept = text
            .lines()
            .filter(|line| !line.contains("ai-title"))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        std::fs::write(&path, kept).expect("write");
        let session = ClaudeProvider::new(dir.path().to_owned())
            .load(ID)
            .expect("load");
        assert_eq!(session.meta.title, "make it run every 3 hours");
    }

    #[test]
    fn keeps_mixed_result_and_prompt() {
        let (dir, bucket) = fixture();
        append(
            &bucket,
            "{\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"toolu_2\",\"content\":\"late\"},{\"type\":\"text\",\"text\":\"array prompt\"}]}}",
        );
        let session = ClaudeProvider::new(dir.path().to_owned())
            .load(ID)
            .expect("load");
        let tools: Vec<&ToolCall> = session
            .turns
            .iter()
            .flat_map(|turn| turn.blocks.iter())
            .filter_map(|block| match block {
                Block::Tools(calls) => Some(calls),
                _ => None,
            })
            .flatten()
            .collect();
        assert_eq!(tools[1].result.as_deref(), Some("late"));
        assert!(
            session
                .turns
                .iter()
                .flat_map(|turn| turn.blocks.iter())
                .any(|block| matches!(block, Block::Text(text) if text == "array prompt"))
        );
    }

    #[test]
    fn missing_result_content_stays_unset() {
        let (dir, bucket) = fixture();
        append(
            &bucket,
            "{\"type\":\"assistant\",\"message\":{\"content\":[{\"type\":\"tool_use\",\"id\":\"toolu_9\",\"name\":\"Bash\",\"input\":{}}],\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}",
        );
        append(
            &bucket,
            "{\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"toolu_9\"}]}}",
        );
        let session = ClaudeProvider::new(dir.path().to_owned())
            .load(ID)
            .expect("load");
        let tools: Vec<&ToolCall> = session
            .turns
            .iter()
            .flat_map(|turn| turn.blocks.iter())
            .filter_map(|block| match block {
                Block::Tools(calls) => Some(calls),
                _ => None,
            })
            .flatten()
            .collect();
        assert_eq!(tools.len(), 3);
        assert_eq!(tools[2].result, None);
    }

    #[test]
    fn repeated_tool_use_keeps_first_call() {
        let (dir, bucket) = fixture();
        append(
            &bucket,
            "{\"type\":\"assistant\",\"message\":{\"content\":[{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"Bash\",\"input\":{\"command\":\"retry\"}}],\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}",
        );
        let session = ClaudeProvider::new(dir.path().to_owned())
            .load(ID)
            .expect("load");
        let tools: Vec<&ToolCall> = session
            .turns
            .iter()
            .flat_map(|turn| turn.blocks.iter())
            .filter_map(|block| match block {
                Block::Tools(calls) => Some(calls),
                _ => None,
            })
            .flatten()
            .collect();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].summary, "ls");
        assert_eq!(tools[0].result.as_deref(), Some("ok"));
    }

    #[test]
    fn ignores_sidechain_only_transcripts() {
        let (dir, _) = fixture();
        let sub = dir
            .path()
            .join("-Users-moreal-github-moreal-rxui")
            .join("agent-acf97af50c6132ed2.jsonl");
        std::fs::write(
            &sub,
            "{\"type\":\"user\",\"isSidechain\":true,\"message\":{\"role\":\"user\",\"content\":\"sub\"}}\n",
        )
        .expect("write");
        let metas = ClaudeProvider::new(dir.path().to_owned())
            .sessions()
            .expect("sessions");
        assert_eq!(metas.len(), 1);
        assert_eq!(metas[0].id, ID);
    }

    #[test]
    fn ignores_garbage_jsonl() {
        let (dir, _) = fixture();
        std::fs::write(
            dir.path().join("notes.jsonl"),
            "not json\n{\"type\":\"attachment\",\"x\":1}\n",
        )
        .expect("write");
        let metas = ClaudeProvider::new(dir.path().to_owned())
            .sessions()
            .expect("sessions");
        assert_eq!(metas.len(), 1);
        assert_eq!(metas[0].id, ID);
    }
}
