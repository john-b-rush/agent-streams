# Contributing

Thanks for taking a look.

## Development setup

Requirements:
- Python 3.11+
- `git`
- `tmux` (recommended; `agent-streams` defaults to tmux)
- The `claude` CLI (default agent runner), or set `AGENT_STREAMS_CLAUDE_BIN` to another compatible executable

Install for local development:

```bash
./install.sh
```

Run from source without installing:

```bash
./bin/agent-streams --help
```

## What to change

This project aims to stay:
- Single-file, stdlib-only (`agent_streams.py`)
- Opinionated about workflow (git worktrees + tmux)
- Simple to fork and adapt for other agent CLIs

## Reporting issues

Please include:
- Your OS + Python version
- `agent-streams --help` output (version, if present)
- The command you ran and the output
- The contents of the run’s `meta.json` (found under `~/.agent-streams/repos/<repo_slug>/runs/<run_id>/meta.json`)
