# agent-streams

A small local runner for “build → review → fix → merge” agent loops, designed to work across many repos using git worktrees and tmux.

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

## Usage

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

Prompts live outside your repo by default under `~/.agent-streams/repos/<repo_slug>/streams/streamN/prompt.md`.

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

## Metrics Dashboard (local)

Run a quick local collector + dashboard that reads `*.prom` files from `~/.agent-streams` and serves a live table.

```bash
uv run python scripts/metrics_dashboard.py
```

Then open `http://127.0.0.1:8765` in your browser. Use `--agent-home`, `--host`, or `--port` to customize, or `--dump` to print the JSON snapshot.

## License

MIT. See `LICENSE`.
