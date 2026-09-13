mod tui;

use std::path::PathBuf;

use anyhow::Result;
use muse_session_inspect::core::Registry;
use muse_session_inspect::providers::muse::MuseProvider;

fn main() -> Result<()> {
    let root = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(MuseProvider::default_root);
    let registry = Registry::new(vec![Box::new(MuseProvider::new(root))]);
    let sessions = registry.sessions()?;
    tui::run(sessions, |meta| registry.load(meta))
}
