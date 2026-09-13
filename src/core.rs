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

pub trait Provider {
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
    use super::truncate;

    #[test]
    fn truncate_never_splits_a_char() {
        let text = "한".repeat(100);
        let cut = truncate(&text, 160);
        assert!(cut.len() < 160 + 3);
        assert!(cut.ends_with('…'));
        assert_eq!(cut[..cut.len() - 3].chars().count(), 53);
    }
}
