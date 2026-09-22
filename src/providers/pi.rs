use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde_json::Value;
use walkdir::WalkDir;

use crate::core::{Block, Provider, Role, Session, SessionMeta, ToolCall, Turn, truncate};

pub struct PiProvider {
    root: PathBuf,
}

impl PiProvider {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn default_root() -> PathBuf {
        std::env::var("HOME")
            .map(|home| PathBuf::from(home).join(".pi/agent/sessions"))
            .unwrap_or_else(|_| PathBuf::from("."))
    }

    fn session_id(path: &Path) -> Option<String> {
        let stem = path.file_stem()?.to_str()?;
        let id = stem.rsplit('_').next()?;
        let bytes = id.as_bytes();
        if id.len() == 36
            && [8, 13, 18, 23]
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

impl Provider for PiProvider {
    fn tool_id(&self) -> &'static str {
        "pi"
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
        if text.trim().is_empty() || duplicate {
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
        match entry.get("type").and_then(Value::as_str) {
            Some("session") => {
                if self.workspace.is_empty()
                    && let Some(workspace) = entry.get("cwd").and_then(Value::as_str)
                    && !workspace.is_empty()
                {
                    self.workspace = workspace.to_owned();
                }
            }
            Some("model_change") => {
                let name = model_name(
                    entry
                        .get("provider")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                    entry
                        .get("modelId")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                );
                if !name.is_empty() {
                    self.model = name;
                }
            }
            Some("message") => self.apply_message(entry.get("message").unwrap_or(&Value::Null)),
            _ => {}
        }
    }

    fn apply_message(&mut self, message: &Value) {
        match message.get("role").and_then(Value::as_str) {
            Some("user") => {
                let text = texts(message.get("content").unwrap_or(&Value::Null));
                self.user_turn(&text);
            }
            Some("assistant") => self.apply_assistant(message),
            Some("toolResult") => {
                let call_id = message
                    .get("toolCallId")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if call_id.is_empty() {
                    return;
                }
                let text = texts(message.get("content").unwrap_or(&Value::Null));
                let failed = message.get("isError").and_then(Value::as_bool) == Some(true);
                if let Some(call) = self.find_call(call_id) {
                    if !text.is_empty() {
                        call.result = Some(truncate(&text, 6000));
                    }
                    call.failed |= failed;
                }
            }
            _ => {}
        }
    }

    fn apply_assistant(&mut self, message: &Value) {
        let name = model_name(
            message
                .get("provider")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            message
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or_default(),
        );
        if !name.is_empty() {
            self.model = name;
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
                        if !text.trim().is_empty() {
                            self.assistant_turn()
                                .blocks
                                .push(Block::Text(text.to_owned()));
                        }
                    }
                    Some("toolCall") => {
                        let id = block.get("id").and_then(Value::as_str).unwrap_or_default();
                        if id.is_empty() || self.calls.contains_key(id) {
                            continue;
                        }
                        let name = block.get("name").and_then(Value::as_str).unwrap_or("?");
                        let summary = block
                            .get("arguments")
                            .map(summarize_arguments)
                            .unwrap_or_default();
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
        let usage = message.get("usage").unwrap_or(&Value::Null);
        let input = usage
            .get("input")
            .and_then(Value::as_u64)
            .unwrap_or_default();
        let output = usage
            .get("output")
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
        let title = self.first_prompt.unwrap_or_else(|| self.id.clone());
        let turns = self.turns.len();
        Session {
            meta: SessionMeta {
                id: self.id,
                tool: "pi",
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
            .filter_map(|block| match block.get("type").and_then(Value::as_str) {
                Some("text") => block.get("text").and_then(Value::as_str),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn model_name(provider: &str, model: &str) -> String {
    if model.is_empty() {
        return String::new();
    }
    if provider.is_empty() || provider == model {
        model.to_owned()
    } else {
        format!("{provider}/{model}")
    }
}

const SALIENT_ARGUMENT_KEYS: [&str; 8] = [
    "command",
    "path",
    "url",
    "query",
    "pattern",
    "prompt",
    "description",
    "skill",
];

fn summarize_arguments(arguments: &Value) -> String {
    match arguments {
        Value::Object(map) => SALIENT_ARGUMENT_KEYS
            .into_iter()
            .filter_map(|key| map.get(key).and_then(Value::as_str))
            .find(|text| !text.is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| truncate(&arguments.to_string(), 300)),
        Value::String(text) => truncate(text, 300),
        _ => truncate(&arguments.to_string(), 300),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    const ID: &str = "9cadfc5b-24c4-4173-b543-4691e5f5f5b1";

    fn fixture() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().expect("tempdir");
        let bucket = dir.path().join("--Users-moreal-repo-num-complex--");
        std::fs::create_dir_all(&bucket).expect("mkdirs");
        let mut file =
            std::fs::File::create(bucket.join(format!("2026-03-17T16-10-34-553Z_{ID}.jsonl")))
                .expect("create");
        writeln!(
            file,
            "{{\"type\":\"session\",\"version\":3,\"id\":\"{ID}\",\"timestamp\":\"2026-03-17T16:10:34.553Z\",\"cwd\":\"/repo/num-complex\"}}"
        )
        .expect("write");
        writeln!(
            file,
            "{{\"type\":\"model_change\",\"id\":\"bd2f8c51\",\"parentId\":null,\"timestamp\":\"2026-03-17T16:10:34.554Z\",\"provider\":\"github-copilot\",\"modelId\":\"claude-opus-4.6\"}}"
        )
        .expect("write");
        writeln!(
            file,
            "{{\"type\":\"thinking_level_change\",\"id\":\"de5aedf1\",\"parentId\":\"bd2f8c51\",\"timestamp\":\"2026-03-17T16:10:34.554Z\",\"thinkingLevel\":\"medium\"}}"
        )
        .expect("write");
        writeln!(
            file,
            "{{\"type\":\"message\",\"id\":\"a80dccbc\",\"parentId\":\"0527e331\",\"timestamp\":\"2026-03-17T16:11:28.972Z\",\"message\":{{\"role\":\"user\",\"content\":[{{\"type\":\"text\",\"text\":\"저장소를 살펴보세요\"}}],\"timestamp\":1773763888969}}}}"
        )
        .expect("write");
        writeln!(
            file,
            "{{\"type\":\"message\",\"id\":\"6c8fa65e\",\"parentId\":\"a80dccbc\",\"timestamp\":\"2026-03-17T16:11:33.864Z\",\"message\":{{\"role\":\"assistant\",\"content\":[{{\"type\":\"text\",\"text\":\"\\n\\n\"}},{{\"type\":\"thinking\",\"thinking\":\"구조를 확인하자\"}},{{\"type\":\"toolCall\",\"id\":\"toolu_1\",\"name\":\"bash\",\"arguments\":{{\"command\":\"ls -la\"}}}},{{\"type\":\"toolCall\",\"id\":\"toolu_2\",\"name\":\"read\",\"arguments\":{{\"path\":\"Cargo.toml\"}}}}],\"api\":\"anthropic-messages\",\"provider\":\"github-copilot\",\"model\":\"claude-opus-4.6\",\"usage\":{{\"input\":1761,\"output\":204,\"cacheRead\":0,\"cacheWrite\":0,\"totalTokens\":1965}},\"stopReason\":\"toolUse\",\"timestamp\":1773763888973}}}}"
        )
        .expect("write");
        writeln!(
            file,
            "{{\"type\":\"message\",\"id\":\"e9c816aa\",\"parentId\":\"6c8fa65e\",\"timestamp\":\"2026-03-17T16:11:33.898Z\",\"message\":{{\"role\":\"toolResult\",\"toolCallId\":\"toolu_1\",\"toolName\":\"bash\",\"content\":[{{\"type\":\"text\",\"text\":\"total 72\"}}],\"isError\":false,\"timestamp\":1773763893898}}}}"
        )
        .expect("write");
        writeln!(
            file,
            "{{\"type\":\"message\",\"id\":\"f739cf2c\",\"parentId\":\"e9c816aa\",\"timestamp\":\"2026-03-17T16:11:34.001Z\",\"message\":{{\"role\":\"toolResult\",\"toolCallId\":\"toolu_2\",\"toolName\":\"read\",\"content\":[{{\"type\":\"text\",\"text\":\"boom\"}}],\"isError\":true,\"timestamp\":1773763894001}}}}"
        )
        .expect("write");
        writeln!(
            file,
            "{{\"type\":\"compaction\",\"id\":\"a0f4f6c0\",\"parentId\":\"11a21db1\",\"timestamp\":\"2026-03-19T12:52:36.979Z\",\"summary\":\"## Goal\\nFix things\"}}"
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
    fn folds_prompts_tools_results_and_meta() {
        let (dir, _) = fixture();
        let session = PiProvider::new(dir.path().to_owned())
            .load(ID)
            .expect("load");
        assert_eq!(session.meta.tool, "pi");
        assert_eq!(session.meta.title, "저장소를 살펴보세요");
        assert_eq!(session.meta.workspace, "/repo/num-complex");
        assert_eq!(session.meta.model, "github-copilot/claude-opus-4.6");
        assert!(matches!(session.turns[0].role, Role::User));
        let tools = tools_of(&session);
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].name, "bash");
        assert_eq!(tools[0].summary, "ls -la");
        assert_eq!(tools[0].result.as_deref(), Some("total 72"));
        assert!(!tools[0].failed);
        assert_eq!(tools[1].summary, "Cargo.toml");
        assert!(tools[1].failed);
        let texts: Vec<&str> = session
            .turns
            .iter()
            .flat_map(|turn| turn.blocks.iter())
            .filter_map(|block| match block {
                Block::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, ["저장소를 살펴보세요"]);
        let usage = session
            .turns
            .iter()
            .flat_map(|turn| turn.blocks.iter())
            .filter_map(|block| match block {
                Block::Usage { input, output } => Some((*input, *output)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(usage, [(1761, 204)]);
    }

    #[test]
    fn falls_back_to_id_for_title() {
        let (dir, bucket) = fixture();
        let path = bucket.join(format!("2026-03-17T16-10-34-553Z_{ID}.jsonl"));
        let text = std::fs::read_to_string(&path).expect("read");
        let kept = text
            .lines()
            .filter(|line| !line.contains("\"role\":\"user\""))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        std::fs::write(&path, kept).expect("write");
        let session = PiProvider::new(dir.path().to_owned())
            .load(ID)
            .expect("load");
        assert_eq!(session.meta.title, ID);
    }

    #[test]
    fn lists_sessions() {
        let (dir, _) = fixture();
        let metas = PiProvider::new(dir.path().to_owned())
            .sessions()
            .expect("sessions");
        assert_eq!(metas.len(), 1);
        assert_eq!(metas[0].id, ID);
        assert_eq!(metas[0].tool, "pi");
    }

    #[test]
    fn ignores_non_session_files() {
        let (dir, bucket) = fixture();
        std::fs::write(bucket.join(".task-lineage.summary.json"), "{}").expect("write");
        std::fs::write(bucket.join("notes.txt"), "hi").expect("write");
        let metas = PiProvider::new(dir.path().to_owned())
            .sessions()
            .expect("sessions");
        assert_eq!(metas.len(), 1);
    }
}
