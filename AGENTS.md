# AGENTS

Guidelines for automated contributors:

- Keep changes centered in `agent_streams.py`; add new files only when needed (tests, scripts, docs).
- Prefer stdlib-only solutions; avoid new runtime dependencies.
- Use `uv run` for local commands.
- Default checks: `uv run pytest -v`, `uv run ruff check .`, `uv run ty check`.
- Integration tests use real tools and are opt-in: `uv run pytest -v -m integration`.
- Update `README.md` when user-facing behavior or commands change.
