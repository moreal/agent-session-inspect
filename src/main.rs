mod tui;

use std::path::PathBuf;

use anyhow::Result;
use muse_session_inspect::core::Registry;
use muse_session_inspect::providers::claude::ClaudeProvider;
use muse_session_inspect::providers::muse::MuseProvider;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let muse_root = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(MuseProvider::default_root);
    let claude_root = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(ClaudeProvider::default_root);
    let registry = Registry::new(vec![
        Box::new(MuseProvider::new(muse_root)),
        Box::new(ClaudeProvider::new(claude_root)),
    ]);
    let sessions = registry.sessions()?;
    tui::run(sessions, |meta| registry.load(meta))
}
