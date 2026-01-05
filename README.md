# agent-streams

agent-streams is a local runner for long-running coding agents with a skeptical overseer gate.
DONE is a claim, not a state: a separate overseer reviews against the spec and only then allows a merge.

```text
Spec -> Builder -> DONE -> Overseer -> APPROVED | ISSUES.md -> fix loop -> merge
```

## Requirements

- Python 3.11+
- `git`
- `tmux` (recommended; used by default)
- The `claude` CLI by default (override with `AGENT_STREAMS_CLAUDE_BIN`)

## Install (local dev)

```bash
./install.sh
```

This uses `uv tool install --editable` under the hood. Ensure your tool bin dir is on `PATH` (commonly `~/.local/bin`).

## Quickstart

```bash
# Launch stream N in tmux (default)
agent-streams launch 3

# Show global run status
agent-streams status

# JSON status (for callers)
agent-streams status --format json

# Print repo slug for current repo
agent-streams repo-slug

# Resume a run (e.g. if tmux died)
agent-streams resume <run_id> --repo /path/to/repo

# Merge an approved run (recovery for approved-but-not-merged)
agent-streams merge <run_id> --repo /path/to/repo
```

## How it works

- Each stream runs in its own tmux session and git worktree.
- The builder runs until it believes the work is done, then creates `DONE` in the agent state dir.
- The overseer reads the spec, reviews the diff, and runs tests. It never edits code.
- The overseer writes either `APPROVED` (merge) or `ISSUES.md` (keep iterating).

## Defaults and layout

Prompts live outside your repo by default:

- `~/.agent-streams/repos/<repo_slug>/streams/streamN/prompt.md`

Run metadata, metrics, and worktrees live under:

- `~/.agent-streams/repos/<repo_slug>/runs/<run_id>`
- `~/.agent-streams/repos/<repo_slug>/metrics/<run_id>/<stream>.prom`
- `~/.agent-streams/worktrees/<repo_slug>/<run_id>/<stream>`

## Configuration

- `AGENT_STREAMS_HOME`: override the agent home (defaults to `~/.agent-streams`)
- `AGENT_STREAMS_CLAUDE_BIN`: override the Claude CLI path
- `AGENT_STREAMS_SESSION_PREFIX`: override tmux session name prefix

## Observability

- Streaming Claude output is shown by default during runs.
- Prometheus textfile metrics are written per stream.
- A lightweight local dashboard is available:

```bash
uv run python scripts/metrics_dashboard.py
```

Then open `http://127.0.0.1:8765` in your browser. Use `--agent-home`, `--host`, or `--port` to customize, or `--dump` to print the JSON snapshot.

## Testing

```bash
uv run pytest -v
uv run ruff check .
uv run ty check
```

Integration tests (real tools; skipped by default):

```bash
uv run pytest -v -m integration
```

## Branches

Runs work from any checked-out branch. The current branch becomes the base branch for the run; detached HEAD is not supported.

## Background

Blog post: https://www.john-rush.com/posts/agent-streams-20260101.html

## License

MIT. See `LICENSE`.
