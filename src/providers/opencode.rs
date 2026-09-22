use std::collections::HashSet;
use std::path::PathBuf;

use anyhow::{Context, Result};
use rusqlite::{Connection, OpenFlags};
use serde_json::Value;

use crate::core::{Block, Provider, Role, Session, SessionMeta, ToolCall, Turn, truncate};

pub struct OpenCodeProvider {
    db: PathBuf,
}

impl OpenCodeProvider {
    pub fn new(root: PathBuf) -> Self {
        Self { db: db_path(&root) }
    }

    pub fn default_root() -> PathBuf {
        if let Ok(dir) = std::env::var("OPENCODE_HOME") {
            return PathBuf::from(dir);
        }
        if let Ok(data) = std::env::var("XDG_DATA_HOME") {
            return PathBuf::from(data).join("opencode");
        }
        std::env::var("HOME")
            .map(|home| PathBuf::from(home).join(".local/share/opencode"))
            .unwrap_or_else(|_| PathBuf::from("."))
    }

    fn open(&self) -> Result<Connection> {
        let conn = Connection::open_with_flags(&self.db, OpenFlags::SQLITE_OPEN_READ_ONLY)
            .with_context(|| format!("opening {}", self.db.display()))?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        Ok(conn)
    }
}

fn db_path(root: &std::path::Path) -> PathBuf {
    if root.is_dir() || !root.extension().is_some_and(|ext| ext == "db") {
        root.join("opencode.db")
    } else {
        root.to_owned()
    }
}

impl Provider for OpenCodeProvider {
    fn tool_id(&self) -> &'static str {
        "opencode"
    }

    fn sessions(&self) -> Result<Vec<SessionMeta>> {
        if !self.db.exists() {
            return Ok(Vec::new());
        }
        let conn = self.open()?;
        let mut stmt = conn.prepare("SELECT id FROM session ORDER BY time_updated")?;
        let ids: Vec<String> = stmt
            .query_map([], |row| row.get(0))?
            .filter_map(|id| id.ok())
            .collect();
        let mut metas = Vec::new();
        let mut seen = HashSet::new();
        for id in ids {
            if seen.insert(id.clone())
                && let Ok(session) = self.load(&id)
                && (session.meta.turns > 0 || session.meta.title != session.meta.id)
            {
                metas.push(session.meta);
            }
        }
        Ok(metas)
    }

    fn load(&self, id: &str) -> Result<Session> {
        let conn = self.open()?;
        let (title, workspace, model): (String, String, String) = conn
            .query_row(
                "SELECT title, directory, COALESCE(model, '') FROM session WHERE id = ?1",
                [id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .map_err(|error| match error {
                rusqlite::Error::QueryReturnedNoRows => {
                    anyhow::anyhow!("session {id} not found under {}", self.db.display())
                }
                _ => anyhow::anyhow!("loading session {id}: {error}"),
            })?;
        let mut fold = Fold::new(id, &title, &workspace, &model_name(&model));
        let mut messages = conn.prepare(
            "SELECT id, data FROM message WHERE session_id = ?1 ORDER BY time_created, id",
        )?;
        let rows: Vec<(String, String)> = messages
            .query_map([id], |row| Ok((row.get(0)?, row.get(1)?)))?
            .filter_map(|row| row.ok())
            .collect();
        let mut parts =
            conn.prepare("SELECT data FROM part WHERE message_id = ?1 ORDER BY time_created, id")?;
        for (message_id, data) in &rows {
            let Some(role) = serde_json::from_str::<Value>(data)
                .ok()
                .and_then(|entry| entry.get("role").and_then(Value::as_str).map(str::to_owned))
                .filter(|role| role == "user" || role == "assistant")
            else {
                continue;
            };
            let bodies: Vec<String> = parts
                .query_map([message_id], |row| row.get(0))?
                .filter_map(|body| body.ok())
                .collect();
            for body in &bodies {
                let Ok(part) = serde_json::from_str::<Value>(body) else {
                    continue;
                };
                fold.apply(&role, &part);
            }
        }
        Ok(fold.finish())
    }
}

struct Fold {
    id: String,
    title: Option<String>,
    first_prompt: Option<String>,
    workspace: String,
    model: String,
    turns: Vec<Turn>,
}

impl Fold {
    fn new(id: &str, title: &str, workspace: &str, model: &str) -> Self {
        let title =
            (!title.is_empty() && !title.starts_with("New session")).then(|| title.to_owned());
        Self {
            id: id.to_owned(),
            title,
            first_prompt: None,
            workspace: workspace.to_owned(),
            model: model.to_owned(),
            turns: Vec::new(),
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

    fn apply(&mut self, role: &str, part: &Value) {
        match part.get("type").and_then(Value::as_str) {
            Some("text") => {
                let text = part.get("text").and_then(Value::as_str).unwrap_or_default();
                if role == "user" {
                    self.user_turn(text);
                } else if !text.is_empty() {
                    self.assistant_turn()
                        .blocks
                        .push(Block::Text(text.to_owned()));
                }
            }
            Some("reasoning") => {
                let text = part.get("text").and_then(Value::as_str).unwrap_or_default();
                if !text.is_empty() {
                    self.assistant_turn()
                        .blocks
                        .push(Block::Text(text.to_owned()));
                }
            }
            Some("tool") => {
                if let Some(call) = tool_call(part) {
                    self.assistant_turn().blocks.push(Block::Tools(vec![call]));
                }
            }
            Some("step-finish") => {
                let tokens = part.get("tokens").unwrap_or(&Value::Null);
                let input = tokens
                    .get("input")
                    .and_then(Value::as_u64)
                    .unwrap_or_default();
                let output = tokens
                    .get("output")
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

    fn finish(self) -> Session {
        let title = self
            .title
            .or(self.first_prompt)
            .unwrap_or_else(|| self.id.clone());
        let turns = self.turns.len();
        Session {
            meta: SessionMeta {
                id: self.id,
                tool: "opencode",
                title,
                workspace: self.workspace,
                model: self.model,
                turns,
            },
            turns: self.turns,
        }
    }
}

fn model_name(raw: &str) -> String {
    let Ok(parsed) = serde_json::from_str::<Value>(raw) else {
        return raw.to_owned();
    };
    match parsed {
        Value::String(name) => name,
        Value::Object(map) => {
            let id = map.get("id").and_then(Value::as_str).unwrap_or_default();
            let provider = map
                .get("providerID")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if !id.is_empty() && !provider.is_empty() {
                format!("{provider}/{id}")
            } else {
                id.to_owned()
            }
        }
        _ => raw.to_owned(),
    }
}

const SALIENT_INPUT_KEYS: [&str; 8] = [
    "command",
    "filePath",
    "path",
    "url",
    "query",
    "pattern",
    "prompt",
    "description",
];

fn summarize_input(input: &Value) -> Option<String> {
    let Value::Object(map) = input else {
        return None;
    };
    let target = ["filePath", "path"]
        .into_iter()
        .filter_map(|key| map.get(key).and_then(Value::as_str))
        .find(|text| !text.is_empty());
    let pattern = map
        .get("pattern")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if let Some(target) = target
        && !pattern.is_empty()
    {
        return Some(truncate(&format!("{target} {pattern}"), 300));
    }
    SALIENT_INPUT_KEYS
        .into_iter()
        .filter_map(|key| map.get(key).and_then(Value::as_str))
        .find(|text| !text.is_empty())
        .map(|text| truncate(text, 300))
}

fn part_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Null => String::new(),
        _ => serde_json::to_string(value).unwrap_or_default(),
    }
}

fn tool_call(part: &Value) -> Option<ToolCall> {
    let name = part.get("tool").and_then(Value::as_str).unwrap_or("?");
    let state = part.get("state").unwrap_or(&Value::Null);
    let summary = summarize_input(state.get("input").unwrap_or(&Value::Null))?;
    let output = part_text(state.get("output").unwrap_or(&Value::Null));
    let error = part_text(state.get("error").unwrap_or(&Value::Null));
    let text = if output.is_empty() { error } else { output };
    Some(ToolCall {
        name: name.to_owned(),
        summary,
        result: (!text.is_empty()).then(|| truncate(&text, 6000)),
        failed: state.get("status").and_then(Value::as_str) != Some("completed"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const ID: &str = "ses_12068ca53ffeIPWVNODStq4Agx";

    fn fixture() -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = dir.path().join("opencode.db");
        let conn = Connection::open(&db).expect("open");
        conn.execute_batch(
            "CREATE TABLE session (id TEXT PRIMARY KEY, title TEXT NOT NULL, directory TEXT NOT NULL, model TEXT, time_updated INTEGER NOT NULL);
             CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, time_created INTEGER NOT NULL, data TEXT NOT NULL);
             CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT NOT NULL, session_id TEXT NOT NULL, time_created INTEGER NOT NULL, data TEXT NOT NULL);",
        )
        .expect("schema");
        conn.execute(
            "INSERT INTO session (id, title, directory, model, time_updated) VALUES (?1, ?2, ?3, ?4, 1)",
            rusqlite::params![
                ID,
                "Verso 문법 파싱 설명",
                "/repo/verso",
                r#"{"id":"gpt-5.5","providerID":"openai"}"#,
            ],
        )
        .expect("session");
        conn.execute(
            "INSERT INTO message (id, data, session_id, time_created) VALUES ('msg_user', ?1, ?2, 1)",
            rusqlite::params![r#"{"role":"user"}"#, ID],
        )
        .expect("user message");
        conn.execute(
            "INSERT INTO part (id, data, message_id, session_id, time_created) VALUES ('prt_u1', ?1, 'msg_user', ?2, 1)",
            rusqlite::params![
                r#"{"type":"text","text":"Verso 문법은 어떻게 파싱하나요?"}"#,
                ID,
            ],
        )
        .expect("user part");
        conn.execute(
            "INSERT INTO message (id, data, session_id, time_created) VALUES ('msg_asst', ?1, ?2, 2)",
            rusqlite::params![r#"{"role":"assistant"}"#, ID],
        )
        .expect("assistant message");
        conn.execute(
            "INSERT INTO part (id, data, message_id, session_id, time_created) VALUES ('prt_a1', ?1, 'msg_asst', ?2, 2)",
            rusqlite::params![r#"{"type":"text","text":"저장소 구조부터 확인하겠습니다."}"#, ID],
        )
        .expect("assistant text");
        conn.execute(
            "INSERT INTO part (id, data, message_id, session_id, time_created) VALUES ('prt_a2', ?1, 'msg_asst', ?2, 3)",
            rusqlite::params![
                r#"{"type":"tool","tool":"read","callID":"call_1","state":{"status":"completed","input":{"filePath":"/repo/verso"},"output":"<entries>src/</entries>"}}"#,
                ID,
            ],
        )
        .expect("tool part");
        conn.execute(
            "INSERT INTO part (id, data, message_id, session_id, time_created) VALUES ('prt_a3', ?1, 'msg_asst', ?2, 4)",
            rusqlite::params![
                r#"{"type":"tool","tool":"bash","callID":"call_2","state":{"status":"error","input":{"command":"boom"},"error":"Unknown: exit 1"}}"#,
                ID,
            ],
        )
        .expect("failed tool");
        conn.execute(
            "INSERT INTO part (id, data, message_id, session_id, time_created) VALUES ('prt_a4', ?1, 'msg_asst', ?2, 5)",
            rusqlite::params![
                r#"{"type":"step-finish","reason":"tool-calls","tokens":{"input":6698,"output":152}}"#,
                ID,
            ],
        )
        .expect("usage");
        dir
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
    fn folds_text_tools_and_usage() {
        let dir = fixture();
        let session = OpenCodeProvider::new(dir.path().to_owned())
            .load(ID)
            .expect("load");
        assert_eq!(session.meta.tool, "opencode");
        assert_eq!(session.meta.title, "Verso 문법 파싱 설명");
        assert_eq!(session.meta.workspace, "/repo/verso");
        assert_eq!(session.meta.model, "openai/gpt-5.5");
        assert!(matches!(session.turns[0].role, Role::User));
        let tools = tools_of(&session);
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].summary, "/repo/verso");
        assert!(!tools[0].failed);
        assert!(tools[1].failed);
        let usage = session
            .turns
            .iter()
            .flat_map(|turn| turn.blocks.iter())
            .any(|block| {
                matches!(
                    block,
                    Block::Usage {
                        input: 6698,
                        output: 152
                    }
                )
            });
        assert!(usage);
    }

    #[test]
    fn falls_back_to_first_prompt_for_new_session() {
        let dir = fixture();
        let db = dir.path().join("opencode.db");
        Connection::open(&db)
            .expect("open")
            .execute(
                "UPDATE session SET title = 'New session - 2026-09-12' WHERE id = ?1",
                [ID],
            )
            .expect("rename");
        let session = OpenCodeProvider::new(dir.path().to_owned())
            .load(ID)
            .expect("load");
        assert_eq!(session.meta.title, "Verso 문법은 어떻게 파싱하나요?");
    }

    #[test]
    fn missing_db_lists_nothing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sessions = OpenCodeProvider::new(dir.path().to_owned())
            .sessions()
            .expect("sessions");
        assert!(sessions.is_empty());
    }

    fn insert_message(conn: &Connection, id: &str, data: &str, created: i64) {
        conn.execute(
            "INSERT INTO message (id, data, session_id, time_created) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![id, data, ID, created],
        )
        .expect("message");
    }

    fn insert_part(conn: &Connection, id: &str, message: &str, data: &str, created: i64) {
        conn.execute(
            "INSERT INTO part (id, data, message_id, session_id, time_created) VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![id, data, message, ID, created],
        )
        .expect("part");
    }

    #[test]
    fn skips_unparseable_or_foreign_roles() {
        let dir = fixture();
        let db = dir.path().join("opencode.db");
        let conn = Connection::open(&db).expect("open");
        insert_message(&conn, "msg_broken", "not json", 10);
        insert_part(
            &conn,
            "prt_b1",
            "msg_broken",
            r#"{"type":"text","text":"phantom"}"#,
            10,
        );
        insert_message(&conn, "msg_sys", r#"{"role":"system"}"#, 11);
        insert_part(
            &conn,
            "prt_s1",
            "msg_sys",
            r#"{"type":"text","text":"injected"}"#,
            11,
        );
        drop(conn);
        let session = OpenCodeProvider::new(dir.path().to_owned())
            .load(ID)
            .expect("load");
        let texts: Vec<&str> = session
            .turns
            .iter()
            .flat_map(|turn| turn.blocks.iter())
            .filter_map(|block| match block {
                Block::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert!(!texts.contains(&"phantom"));
        assert!(!texts.contains(&"injected"));
    }

    #[test]
    fn skips_tools_without_salient_input() {
        let dir = fixture();
        let db = dir.path().join("opencode.db");
        let conn = Connection::open(&db).expect("open");
        insert_part(
            &conn,
            "prt_null",
            "msg_asst",
            r#"{"type":"tool","tool":"bash","callID":"call_9","state":{"status":"completed"}}"#,
            9,
        );
        drop(conn);
        let session = OpenCodeProvider::new(dir.path().to_owned())
            .load(ID)
            .expect("load");
        let tools = tools_of(&session);
        assert_eq!(tools.len(), 2);
        assert!(tools.iter().all(|tool| tool.summary != "null"));
    }

    #[test]
    fn marks_unfinished_tools_failed() {
        let dir = fixture();
        let db = dir.path().join("opencode.db");
        let conn = Connection::open(&db).expect("open");
        insert_part(
            &conn,
            "prt_pending",
            "msg_asst",
            r#"{"type":"tool","tool":"bash","callID":"call_8","state":{"status":"pending","input":{"command":"sleep 60"}}}"#,
            8,
        );
        drop(conn);
        let session = OpenCodeProvider::new(dir.path().to_owned())
            .load(ID)
            .expect("load");
        let pending = tools_of(&session)
            .into_iter()
            .find(|tool| tool.summary == "sleep 60")
            .expect("pending tool");
        assert!(pending.failed);
    }

    #[test]
    fn empty_session_stays_out_of_listing() {
        let dir = fixture();
        let db = dir.path().join("opencode.db");
        let conn = Connection::open(&db).expect("open");
        conn.execute(
            "INSERT INTO session (id, title, directory, model, time_updated) VALUES ('ses_empty', 'ses_empty', '', '', 2)",
            [],
        )
        .expect("empty session");
        drop(conn);
        let sessions = OpenCodeProvider::new(dir.path().to_owned())
            .sessions()
            .expect("sessions");
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].id, ID);
    }
}
