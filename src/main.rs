mod tui;

use std::path::PathBuf;

use anyhow::Result;
use muse_session_inspect::core::Registry;
use muse_session_inspect::providers::claude::ClaudeProvider;
use muse_session_inspect::providers::codex::CodexProvider;
use muse_session_inspect::providers::muse::MuseProvider;
use muse_session_inspect::providers::opencode::OpenCodeProvider;
use muse_session_inspect::providers::pi::PiProvider;

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
    let codex_root = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(CodexProvider::default_root);
    let opencode_root = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(OpenCodeProvider::default_root);
    let pi_root = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(PiProvider::default_root);
    let registry = Registry::new(vec![
        Box::new(MuseProvider::new(muse_root)),
        Box::new(ClaudeProvider::new(claude_root)),
        Box::new(CodexProvider::new(codex_root)),
        Box::new(OpenCodeProvider::new(opencode_root)),
        Box::new(PiProvider::new(pi_root)),
    ]);
    tui::run(registry)
}
