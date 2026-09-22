# AGENTS.md

Coding-agent session inspector. Rust 2024 edition, TUI with ratatui, dev shell via Nix flakes.

## Commands

Enter the dev shell with `nix develop` (or direnv via `.envrc`). Set `CARGO_HOME`
outside the sandbox when the default registry path is not writable.

- `cargo build` — build `agent-session-inspect`
- `cargo test` — run the suite; keep provider fixtures hermetic under `src/`
- `cargo clippy --all-targets` — must be warning-free before finishing
- `cargo fmt --check` — must pass before finishing
- Run with optional sessions roots: `cargo run -- [<muse-root> [<claude-root> [<codex-root> [<opencode-root> [<pi-root>]]]]]`

## Architecture

New tools join without touching the TUI:

- `src/core.rs` — tool-agnostic model (`SessionMeta`, `Session`, `Turn`, `Block`)
  plus the `Provider` trait (discover + load) and `Registry` routing by tool id.
- `src/providers/<tool>.rs` — one `Provider` impl per tool. Owns log discovery,
  native-format parsing, and folding into the core model.
- `src/tui.rs` — renders only the core model. Never imports a provider module.

`Provider::sessions` may scan the filesystem; keep `load` pure over one session id.

## Working agreements

- 주석을 사용하게 되는 것은 코드를 이해하기 어려운 상태라는 신호이므로 코드만을 보고 이해할 수 있도록 리팩터링을 진행하십시오.
  Comment-worthy code is a signal to refactor until the code reads on its own.
  Keep comments to why-this-way records that renaming cannot express.
- subagent로 하여금 적대적 리뷰를 하게끔 하여 코드를 더 나은 상태로 만듭시다.
  Before finishing a behavior change, spawn a subagent for adversarial review and
  address or explicitly rebut each finding.
- 토큰 효율적이게 동작하십시오. Prefer small diffs, shared helpers over new
  abstractions, truncated display strings, and no duplicated context.

## Commits

- Sign commits directly; signing is expected to work. If signing fails,
  fall back to `git commit --no-gpg-sign` and report the failure.
- Never change signing config to work around a failure.
