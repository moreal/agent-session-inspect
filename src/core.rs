use anyhow::Result;

pub struct SessionMeta {
    pub id: String,
    pub tool: &'static str,
    pub title: String,
    pub workspace: String,
    pub model: String,
    pub turns: usize,
}

pub enum Role {
    User,
    Assistant,
}

pub struct ToolCall {
    pub name: String,
    pub summary: String,
    pub result: Option<String>,
    pub failed: bool,
}

pub enum Block {
    Text(String),
    Tools(Vec<ToolCall>),
    Usage { input: u64, output: u64 },
}

pub struct Turn {
    pub role: Role,
    pub blocks: Vec<Block>,
}

pub struct Session {
    pub meta: SessionMeta,
    pub turns: Vec<Turn>,
}

pub trait Provider: Send + Sync {
    fn tool_id(&self) -> &'static str;
    fn sessions(&self) -> Result<Vec<SessionMeta>>;
    fn load(&self, id: &str) -> Result<Session>;
}

pub fn truncate(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_owned();
    }
    let mut end = max;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &text[..end])
}

pub struct Registry {
    providers: Vec<Box<dyn Provider>>,
}

impl Registry {
    pub fn new(providers: Vec<Box<dyn Provider>>) -> Self {
        Self { providers }
    }

    pub fn tool_ids(&self) -> Vec<&'static str> {
        let mut ids = Vec::new();
        for provider in &self.providers {
            if !ids.contains(&provider.tool_id()) {
                ids.push(provider.tool_id());
            }
        }
        ids
    }

    pub fn sessions_for(&self, tool: &str) -> Result<Vec<SessionMeta>> {
        let mut all = Vec::new();
        let mut known = false;
        for provider in &self.providers {
            if provider.tool_id() == tool {
                known = true;
                all.extend(provider.sessions()?);
            }
        }
        if known {
            Ok(all)
        } else {
            Err(anyhow::anyhow!("no provider for tool {tool}"))
        }
    }

    pub fn sessions(&self) -> Result<Vec<SessionMeta>> {
        let mut all = Vec::new();
        for provider in &self.providers {
            all.extend(provider.sessions()?);
        }
        Ok(all)
    }

    pub fn load(&self, meta: &SessionMeta) -> Result<Session> {
        self.providers
            .iter()
            .find(|provider| provider.tool_id() == meta.tool)
            .ok_or_else(|| anyhow::anyhow!("no provider for tool {}", meta.tool))?
            .load(&meta.id)
    }
}

#[cfg(test)]
mod tests {
    use super::{Registry, SessionMeta, truncate};

    struct Stub {
        tool: &'static str,
        ids: Vec<&'static str>,
    }

    impl super::Provider for Stub {
        fn tool_id(&self) -> &'static str {
            self.tool
        }

        fn sessions(&self) -> anyhow::Result<Vec<SessionMeta>> {
            Ok(self
                .ids
                .iter()
                .map(|id| SessionMeta {
                    id: id.to_string(),
                    tool: self.tool,
                    title: id.to_string(),
                    workspace: String::new(),
                    model: String::new(),
                    turns: 0,
                })
                .collect())
        }

        fn load(&self, id: &str) -> anyhow::Result<super::Session> {
            anyhow::bail!("no session {id}")
        }
    }

    #[test]
    fn registry_routes_sessions_by_tool() {
        let registry = Registry::new(vec![
            Box::new(Stub {
                tool: "a",
                ids: vec!["a1"],
            }),
            Box::new(Stub {
                tool: "b",
                ids: vec!["b1", "b2"],
            }),
        ]);
        assert_eq!(registry.tool_ids(), vec!["a", "b"]);
        assert_eq!(registry.sessions_for("b").unwrap().len(), 2);
        assert_eq!(registry.sessions().unwrap().len(), 3);
        assert!(registry.sessions_for("missing").is_err());
    }

    #[test]
    fn truncate_never_splits_a_char() {
        let text = "한".repeat(100);
        let cut = truncate(&text, 160);
        assert!(cut.len() < 160 + 3);
        assert!(cut.ends_with('…'));
        assert_eq!(cut[..cut.len() - 3].chars().count(), 53);
    }
}
