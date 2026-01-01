#!/usr/bin/env python3

from __future__ import annotations

import argparse
import codecs
import contextlib
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

DEFAULT_AGENT_HOME = Path("~/.agent-streams").expanduser()
DEFAULT_SESSION_PREFIX = "agent-streams"


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _die(msg: str, code: int = 1) -> NoReturn:
    _eprint(msg)
    raise SystemExit(code)


def _shell_quote(value: str) -> str:
    return shlex.quote(value)


def _run(
    argv: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(
            argv,
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=capture,
            check=False,
        )
    except FileNotFoundError:
        _die(f"Command not found: {argv[0]}")

    if check and proc.returncode != 0:
        out = ""
        if proc.stdout:
            out += proc.stdout
        if proc.stderr:
            out += proc.stderr
        cmd = shlex.join(argv)
        _die(f"Command failed ({proc.returncode}): {cmd}\n{out}".rstrip(), code=proc.returncode)
    return proc


def _git_out(repo_root: Path, *args: str) -> str:
    proc = _run(["git", "-C", str(repo_root), *args], capture=True)
    return (proc.stdout or "").strip()


def _resolve_repo_root(repo: str | None) -> Path:
    base = Path(repo).expanduser() if repo else Path.cwd()
    proc = _run(["git", "-C", str(base), "rev-parse", "--show-toplevel"], capture=True)
    root = (proc.stdout or "").strip()
    if not root:
        _die(f"Failed to resolve repo root from: {base}")
    return Path(root)


def _repo_slug(repo_root: Path) -> str:
    repo_name = repo_root.name
    repo_hash = hashlib.sha1(str(repo_root).encode("utf-8")).hexdigest()[:8]
    return f"{repo_name}-{repo_hash}"


def _run_id() -> str:
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"


def _stream_name(raw: str) -> str:
    if raw.startswith("stream"):
        return raw
    if raw.isdigit():
        return f"stream{raw}"
    _die(f"Invalid stream: {raw} (expected N or streamN)")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _write_text(path: Path, content: str) -> None:
    _ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def _resolve_agent_home(agent_home: str | None) -> Path:
    if agent_home:
        return Path(agent_home).expanduser()
    env = os.environ.get("AGENT_STREAMS_HOME")
    if env:
        return Path(env).expanduser()
    return DEFAULT_AGENT_HOME


def _resolve_prompt_source(
    *,
    prompt: str | None,
    agent_home: Path,
    repo_slug: str,
    stream_name: str,
    repo_root: Path,
) -> Path:
    if prompt:
        path = Path(prompt).expanduser()
        if not path.is_file():
            _die(f"Prompt not found: {path}")
        return path

    candidates = [
        agent_home / "repos" / repo_slug / "streams" / stream_name / "prompt.md",
        agent_home / "streams" / stream_name / "prompt.md",
        agent_home / "prompts" / repo_slug / f"{stream_name}.md",
        agent_home / "prompts" / f"{stream_name}.md",
        # Legacy in-repo location (optional).
        repo_root / "streams" / stream_name / "prompt.md",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    _die(
        "\n".join(
            [
                f"Prompt not found for {stream_name}.",
                "Provide one with: agent-streams launch --prompt PATH <N>",
                f"Or create: {agent_home}/repos/{repo_slug}/streams/{stream_name}/prompt.md",
            ]
        )
    )


def _tmux_available() -> bool:
    return shutil.which("tmux") is not None


def _kill_tmux(session: str) -> None:
    if not session:
        return
    if not _tmux_available():
        return
    _run(["tmux", "has-session", "-t", session], check=False)
    _run(["tmux", "kill-session", "-t", session], check=False)


def _tmux_new_session(*, session: str, window: str, cwd: Path) -> None:
    _run(["tmux", "new-session", "-d", "-s", session, "-n", window, "-c", str(cwd)])


def _tmux_send(*, session: str, window: str, command: str) -> None:
    target = f"{session}:{window}"
    _run(["tmux", "send-keys", "-t", target, command, "Enter"])


def _self_run_argv() -> list[str]:
    # Prefer the currently-running executable (console script, symlink, etc).
    argv0 = Path(sys.argv[0]).expanduser()
    if argv0.is_file():
        return [str(argv0.resolve())]

    found = shutil.which("agent-streams")
    if found:
        return [found]

    # Fallback: run this module via the current Python interpreter.
    return [sys.executable, str(Path(__file__).resolve())]


def _append_runtime_block(
    *,
    prompt_source: Path,
    prompt_dest: Path,
    stream_name: str,
    run_id: str,
    worktree_path: Path,
    agent_dir: Path,
) -> None:
    base = _read_text(prompt_source)
    runtime = f"""

<!-- AGENT_STREAMS_RUNTIME -->
## Agent Streams Runtime

This stream is being run by an external runner. Ignore any hard-coded paths in the prompt for `streams/<stream>/.agent/*` and use the agent state dir below instead.

- Stream: `{stream_name}`
- Run id: `{run_id}`
- Repo (worktree): `{worktree_path}`
- Agent state dir: `{agent_dir}`

### Required loop + markers
1. If `{agent_dir}/ISSUES.md` exists, fix those issues first.
2. Track progress in `{agent_dir}/TODO.md`.
3. Run the tests specified in the prompt.
4. Commit changes with clear messages.
5. When everything is complete and tests pass, create `{agent_dir}/DONE`.
""".lstrip(
        "\n"
    )
    _write_text(prompt_dest, base.rstrip() + "\n" + runtime)


def _worktree_list_paths(repo_root: Path) -> set[str]:
    proc = _run(["git", "-C", str(repo_root), "worktree", "list", "--porcelain"], capture=True)
    paths: set[str] = set()
    for line in (proc.stdout or "").splitlines():
        if line.startswith("worktree "):
            paths.add(line.split(" ", 1)[1].strip())
    return paths


def _ensure_worktree(
    *,
    repo_root: Path,
    base_dir: Path,
    branch_prefix: str,
    stream_name: str,
) -> Path:
    _ensure_dir(base_dir)
    branch_name = f"{branch_prefix}{stream_name}"
    worktree_path = base_dir / stream_name

    if str(worktree_path) in _worktree_list_paths(repo_root):
        return worktree_path

    if worktree_path.exists():
        _die(f"Error: {worktree_path} exists but is not a git worktree.")

    # Ensure branch exists (re-use if so).
    exists = _run(
        ["git", "-C", str(repo_root), "show-ref", "--verify", "--quiet", f"refs/heads/{branch_name}"],
        check=False,
    ).returncode == 0
    if exists:
        _run(["git", "-C", str(repo_root), "worktree", "add", str(worktree_path), branch_name])
    else:
        _run(["git", "-C", str(repo_root), "worktree", "add", "-b", branch_name, str(worktree_path)])
    return worktree_path


def _safe_unlink(path: Path) -> None:
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    except FileNotFoundError:
        return


def _strip_agent_path_from_repo(repo_root: Path, stream_name: str) -> None:
    agent_path = repo_root / "streams" / stream_name / ".agent"
    if not agent_path.exists():
        return
    _safe_unlink(agent_path)
    _run(["git", "-C", str(repo_root), "add", "-A", "--", str(agent_path.relative_to(repo_root))], check=False)


def _find_first_run_tests_line(prompt_file: Path) -> str:
    try:
        for line in prompt_file.read_text(encoding="utf-8").splitlines():
            if "Run tests:" in line:
                return line
    except OSError:
        return ""
    return ""


def _claude_bin() -> str:
    return os.environ.get("AGENT_STREAMS_CLAUDE_BIN", "claude")


class Colors:
    GRAY = "\033[90m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def _use_color() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    return sys.stderr.isatty()


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _stringify_stream_value(value: Any, *, limit: int) -> str:
    if isinstance(value, str):
        return _truncate_text(value, limit)
    if isinstance(value, (list, dict)):
        text = _content_to_text(value)
        if text:
            return _truncate_text(text, limit)
        try:
            rendered = json.dumps(value, ensure_ascii=True)
        except TypeError:
            rendered = str(value)
        return _truncate_text(rendered, limit)
    return _truncate_text(str(value), limit)


def _format_stream_event(data: dict[str, Any]) -> str | None:
    event_type = data.get("type")
    subtype = data.get("subtype")
    use_color = _use_color()

    def wrap(text: str, color: str) -> str:
        if use_color:
            return f"{color}{text}{Colors.RESET}"
        return text

    if event_type == "system" and subtype == "init":
        model = data.get("model", "unknown")
        return wrap(f"[init] model={model}", Colors.GRAY)

    if event_type == "assistant":
        msg = data.get("message") or {}
        content = msg.get("content") or []
        parts: list[str] = []
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type == "tool_use":
                    tool_name = item.get("name", "?")
                    tool_input = item.get("input", {})
                    try:
                        input_str = json.dumps(tool_input, ensure_ascii=True)
                    except TypeError:
                        input_str = str(tool_input)
                    input_str = _truncate_text(input_str, 160)
                    parts.append(wrap(f"[tool] {tool_name}: {input_str}", Colors.CYAN))
                elif item_type == "text":
                    text = _stringify_stream_value(item.get("text", ""), limit=200)
                    if text:
                        parts.append(wrap(f"[text] {text}", Colors.GREEN))
        return "\n".join(parts) if parts else None

    if event_type == "user":
        msg = data.get("message") or {}
        content = msg.get("content") or []
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "tool_result":
                    result = _stringify_stream_value(item.get("content", ""), limit=150)
                    is_error = bool(item.get("is_error", False))
                    status = "error" if is_error else "result"
                    color = Colors.RED if is_error else Colors.YELLOW
                    return wrap(f"[{status}] {result}", color)
        return None

    if event_type == "result":
        cost_raw = data.get("total_cost_usd", 0)
        turns_raw = data.get("num_turns", 0)
        duration_raw = data.get("duration_ms", 0)
        try:
            cost = float(cost_raw)
        except (TypeError, ValueError):
            cost = 0.0
        try:
            turns = int(turns_raw)
        except (TypeError, ValueError):
            turns = 0
        try:
            duration = float(duration_raw) / 1000
        except (TypeError, ValueError):
            duration = 0.0
        return wrap(f"[done] {turns} turns, ${cost:.4f}, {duration:.1f}s", Colors.BOLD)

    return None


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
    return ""


def _extract_text(obj: Any) -> str:
    if isinstance(obj, dict):
        if "content" in obj:
            text = _content_to_text(obj["content"])
            if text:
                return text
        if "message" in obj:
            text = _extract_text(obj["message"])
            if text:
                return text
        completion = obj.get("completion")
        if isinstance(completion, str) and completion:
            return completion
        if obj.get("type") == "text" and isinstance(obj.get("text"), str):
            return obj["text"]
        for value in obj.values():
            text = _extract_text(value)
            if text:
                return text
    elif isinstance(obj, list):
        parts: list[str] = []
        for item in obj:
            text = _extract_text(item)
            if text:
                parts.append(text)
        return "".join(parts)
    return ""


def _first_int(*values: Any) -> int | None:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _find_usage(obj: Any) -> dict[str, Any] | None:
    if isinstance(obj, dict):
        usage = obj.get("usage")
        if isinstance(usage, dict):
            return usage
        for value in obj.values():
            found = _find_usage(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_usage(item)
            if found is not None:
                return found
    return None


def _extract_usage(obj: Any) -> dict[str, int] | None:
    usage = _find_usage(obj)
    if not usage:
        return None
    prompt = _first_int(usage.get("prompt_tokens"), usage.get("input_tokens"), usage.get("input"))
    completion = _first_int(usage.get("completion_tokens"), usage.get("output_tokens"), usage.get("output"))
    total = _first_int(usage.get("total_tokens"))
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    if prompt is None and completion is None and total is None:
        return None
    return {
        "prompt_tokens": int(prompt or 0),
        "completion_tokens": int(completion or 0),
        "total_tokens": int(total or 0),
    }


def _load_json_maybe(raw: str) -> Any | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            pass
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


@dataclass
class ClaudeResult:
    raw: str
    text: str
    usage: dict[str, int] | None


def _run_claude_json(*, prompt_file: Path, cwd: Path) -> ClaudeResult:
    return _run_claude_json_stdin(prompt_text=_read_text(prompt_file), cwd=cwd)


def _run_claude_json_stdin(*, prompt_text: str, cwd: Path) -> ClaudeResult:
    claude = _claude_bin()
    try:
        proc = subprocess.run(
            [claude, "-p", "--output-format", "json", "--dangerously-skip-permissions"],
            cwd=str(cwd),
            input=prompt_text,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        _die(f"Command not found: {claude}")

    raw = ""
    if proc.stdout:
        raw += proc.stdout
    if proc.stderr:
        raw += proc.stderr
    data = _load_json_maybe(raw)
    text = _extract_text(data) if data is not None else ""
    usage = _extract_usage(data) if data is not None else None
    if not text:
        text = raw
    return ClaudeResult(raw=raw, text=text, usage=usage)


def _run_claude_stream(*, prompt_file: Path, cwd: Path, quiet: bool = False) -> ClaudeResult:
    return _run_claude_stream_stdin(
        prompt_text=_read_text(prompt_file), cwd=cwd, quiet=quiet
    )


def _run_claude_stream_stdin(*, prompt_text: str, cwd: Path, quiet: bool = False) -> ClaudeResult:
    claude = _claude_bin()
    cmd = [
        claude,
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--dangerously-skip-permissions",
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError:
        _die(f"Command not found: {claude}")

    stdin = proc.stdin
    if stdin is None:
        _die("Failed to open stdin for Claude process.")
    assert stdin is not None
    stdin.write(prompt_text)
    stdin.close()

    raw_lines: list[str] = []
    final_result: dict[str, Any] | None = None

    if proc.stdout is not None:
        for raw_line in proc.stdout:
            raw_lines.append(raw_line)
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                if not quiet:
                    print(line, file=sys.stderr, flush=True)
                continue
            if data.get("type") == "result":
                final_result = data
            if not quiet:
                formatted = _format_stream_event(data)
                if formatted:
                    print(formatted, file=sys.stderr, flush=True)

    proc.wait()
    raw = "".join(raw_lines)

    if final_result:
        text = final_result.get("result", "")
        usage = _extract_usage(final_result)
    else:
        data = _load_json_maybe(raw)
        text = _extract_text(data) if data is not None else ""
        usage = _extract_usage(data) if data is not None else None
        if not text:
            text = raw
    return ClaudeResult(raw=raw, text=text, usage=usage)


def _default_review_prompt() -> str:
    return """# Overseer Review

You are reviewing work completed by another agent. Be SKEPTICAL and THOROUGH.

Your job is to find problems. The builder agent tends to do cut-rate work and declare victory early.

**Trust nothing:**
- Don't trust the builder's claims in DONE or TODO.md
- Don't trust "optional" or "enhancement" labels in specs - investigate whether they're actually needed
- Don't trust comments like "works correctly" - verify it yourself
- Don't trust that tests passing means the feature is complete

## Your Task

1. Read the stream prompt first (it is the spec of record for this stream)
2. Review the implementation changes in the repo
3. Run the tests listed in the stream prompt (or the most relevant subset)
4. Find gaps (functional, tests, integration, error handling)

## Output

Default to finding issues. An empty ISSUES.md is suspicious.

If issues found (expected):
- Write them to `ISSUES.md` in the agent state directory
- Be specific: file, line (if possible), what's wrong, what to add
- Prioritize: critical > important > nice-to-have
- Do NOT create APPROVED

If genuinely complete (rare):
- Create `APPROVED` with detailed justification
- Delete `ISSUES.md` if it exists
"""


def _resolve_review_prompt(*, agent_home: Path, repo_slug: str, override: str | None) -> str:
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            _die(f"Review prompt not found: {path}")
        return _read_text(path)
    repo_specific = agent_home / "repos" / repo_slug / "review.md"
    if repo_specific.is_file():
        return _read_text(repo_specific)
    global_default = agent_home / "review.md"
    if global_default.is_file():
        return _read_text(global_default)
    return _default_review_prompt()


def _emit_metrics(
    *,
    metrics_file: Path,
    labels: str,
    iteration: int,
    max_iterations: int,
    state: str,
    last_agent_duration: int,
    last_review_duration: int,
    review_approved: int,
    review_issues: int,
    review_none: int,
    llm_calls: int,
    llm_usage_missing: int,
    tokens_prompt_total: int,
    tokens_completion_total: int,
    tokens_total: int,
    tokens_prompt_last: int,
    tokens_completion_last: int,
    tokens_last_total: int,
) -> None:
    now = int(time.time())
    _ensure_dir(metrics_file.parent)
    states = ["init", "running", "review", "waiting", "approved", "maxed"]
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=str(metrics_file.parent),
            prefix=f".{metrics_file.name}.",
            suffix=".tmp",
        ) as f:
            tmp_path = Path(f.name)
            f.write("# HELP streams_iteration_total Total iterations executed for a stream.\n")
            f.write("# TYPE streams_iteration_total counter\n")
            f.write(f"streams_iteration_total{{{labels}}} {iteration}\n")
            f.write("# HELP streams_max_iterations Max iterations configured for the stream.\n")
            f.write("# TYPE streams_max_iterations gauge\n")
            f.write(f"streams_max_iterations{{{labels}}} {max_iterations}\n")
            f.write("# HELP streams_state Current lifecycle state for the stream.\n")
            f.write("# TYPE streams_state gauge\n")
            for s in states:
                v = 1 if state == s else 0
                f.write(f"streams_state{{{labels},state=\"{s}\"}} {v}\n")
            f.write("# HELP streams_last_update_timestamp Unix timestamp of the last metrics update.\n")
            f.write("# TYPE streams_last_update_timestamp gauge\n")
            f.write(f"streams_last_update_timestamp{{{labels}}} {now}\n")
            f.write("# HELP streams_last_action_duration_seconds Duration of the last agent or review action.\n")
            f.write("# TYPE streams_last_action_duration_seconds gauge\n")
            f.write(f"streams_last_action_duration_seconds{{{labels},action=\"agent\"}} {last_agent_duration}\n")
            f.write(f"streams_last_action_duration_seconds{{{labels},action=\"review\"}} {last_review_duration}\n")
            f.write("# HELP streams_review_total Review results by outcome.\n")
            f.write("# TYPE streams_review_total counter\n")
            f.write(f"streams_review_total{{{labels},result=\"approved\"}} {review_approved}\n")
            f.write(f"streams_review_total{{{labels},result=\"issues\"}} {review_issues}\n")
            f.write(f"streams_review_total{{{labels},result=\"none\"}} {review_none}\n")
            f.write("# HELP streams_llm_calls_total Total Claude calls for this stream.\n")
            f.write("# TYPE streams_llm_calls_total counter\n")
            f.write(f"streams_llm_calls_total{{{labels}}} {llm_calls}\n")
            f.write("# HELP streams_llm_usage_missing_total Claude calls missing token usage.\n")
            f.write("# TYPE streams_llm_usage_missing_total counter\n")
            f.write(f"streams_llm_usage_missing_total{{{labels}}} {llm_usage_missing}\n")
            f.write("# HELP streams_tokens_total Total tokens by type for this stream.\n")
            f.write("# TYPE streams_tokens_total counter\n")
            f.write(f"streams_tokens_total{{{labels},kind=\"prompt\"}} {tokens_prompt_total}\n")
            f.write(f"streams_tokens_total{{{labels},kind=\"completion\"}} {tokens_completion_total}\n")
            f.write(f"streams_tokens_total{{{labels},kind=\"total\"}} {tokens_total}\n")
            f.write("# HELP streams_tokens_last Tokens for the last Claude call.\n")
            f.write("# TYPE streams_tokens_last gauge\n")
            f.write(f"streams_tokens_last{{{labels},kind=\"prompt\"}} {tokens_prompt_last}\n")
            f.write(f"streams_tokens_last{{{labels},kind=\"completion\"}} {tokens_completion_last}\n")
            f.write(f"streams_tokens_last{{{labels},kind=\"total\"}} {tokens_last_total}\n")
        if tmp_path is not None:
            tmp_path.replace(metrics_file)
    finally:
        if tmp_path and tmp_path.exists():
            with contextlib.suppress(OSError):
                tmp_path.unlink()


def _run_loop(
    *,
    agent_home: Path,
    repo_root: Path,
    repo_label: str,
    repo_slug: str,
    stream_name: str,
    agent_dir: Path,
    prompt_file: Path,
    metrics_dir: Path,
    max_iterations: int,
    review_prompt_override: str | None,
    quiet: bool = False,
) -> bool:
    metrics_file = metrics_dir / f"{stream_name}.prom"
    labels = f"repo=\"{repo_label}\",stream=\"{stream_name}\""

    iteration = 0
    last_agent_duration = 0
    last_review_duration = 0

    tokens_prompt_total = 0
    tokens_completion_total = 0
    tokens_total = 0
    tokens_prompt_last = 0
    tokens_completion_last = 0
    tokens_last_total = 0

    llm_calls = 0
    llm_usage_missing = 0
    review_approved = 0
    review_issues = 0
    review_none = 0
    state = "init"

    def emit() -> None:
        _emit_metrics(
            metrics_file=metrics_file,
            labels=labels,
            iteration=iteration,
            max_iterations=max_iterations,
            state=state,
            last_agent_duration=last_agent_duration,
            last_review_duration=last_review_duration,
            review_approved=review_approved,
            review_issues=review_issues,
            review_none=review_none,
            llm_calls=llm_calls,
            llm_usage_missing=llm_usage_missing,
            tokens_prompt_total=tokens_prompt_total,
            tokens_completion_total=tokens_completion_total,
            tokens_total=tokens_total,
            tokens_prompt_last=tokens_prompt_last,
            tokens_completion_last=tokens_completion_last,
            tokens_last_total=tokens_last_total,
        )

    _ensure_dir(agent_dir)
    _ensure_dir(metrics_dir)

    print(f"=== Starting: {stream_name} (max {max_iterations} iterations) ===")
    print(f"Prompt: {prompt_file}")
    print("---")
    emit()

    while True:
        if iteration >= max_iterations:
            print(f"=== MAX ITERATIONS ({max_iterations}) reached ===")
            print(f"Stream not complete. Check {agent_dir}/ISSUES.md for remaining work.")
            state = "maxed"
            emit()
            return False

        if (agent_dir / "APPROVED").exists():
            print("Stream already approved. Nothing to do.")
            state = "approved"
            emit()
            return True

        if (agent_dir / "DONE").exists():
            print(f"[{time.strftime('%H:%M:%S')}] Agent signaled DONE. Running overseer review...")
            state = "review"
            emit()

            review_start = time.time()
            review_template = _resolve_review_prompt(
                agent_home=agent_home,
                repo_slug=repo_slug,
                override=review_prompt_override,
            )
            tests_line = _find_first_run_tests_line(prompt_file)
            review_prompt = (
                review_template.rstrip()
                + "\n\n"
                + f"## Stream to Review: {stream_name}\n\n"
                + f"Repo root: `{repo_root}`\n"
                + f"Stream prompt: `{prompt_file}`\n"
                + f"Agent state dir: `{agent_dir}`\n"
                + "\nRead the stream prompt first. Review the implementation changes in the repo, run the tests listed in the stream prompt, and look for missing spec requirements referenced by the prompt.\n"
            )
            if tests_line:
                review_prompt += "\nRun the tests listed in the stream prompt:\n" + tests_line + "\n"
            review_prompt += (
                "\nWrite issues to `ISSUES.md` in the agent state directory, or approval to `APPROVED`.\n"
            )

            result = _run_claude_stream_stdin(
                prompt_text=review_prompt, cwd=repo_root, quiet=quiet
            )
            print(result.text, end="" if result.text.endswith("\n") else "\n")

            llm_calls += 1
            if result.usage:
                tokens_prompt_last = int(result.usage.get("prompt_tokens", 0))
                tokens_completion_last = int(result.usage.get("completion_tokens", 0))
                tokens_last_total = int(result.usage.get("total_tokens", 0))
                tokens_prompt_total += tokens_prompt_last
                tokens_completion_total += tokens_completion_last
                tokens_total += tokens_last_total
            else:
                llm_usage_missing += 1
                tokens_prompt_last = 0
                tokens_completion_last = 0
                tokens_last_total = 0

            last_review_duration = int(time.time() - review_start)

            if (agent_dir / "APPROVED").exists():
                print("=== APPROVED by overseer ===")
                print(_read_text(agent_dir / "APPROVED").rstrip())
                review_approved += 1
                state = "approved"
                emit()
                return True

            if (agent_dir / "ISSUES.md").exists():
                print("=== ISSUES found by overseer ===")
                print(_read_text(agent_dir / "ISSUES.md").rstrip())
                print("\nRemoving DONE flag, continuing loop...")
                _safe_unlink(agent_dir / "DONE")
                review_issues += 1
                state = "waiting"
                emit()
                time.sleep(2)
                continue

            print("Overseer didn't create APPROVED or ISSUES. Continuing...")
            _safe_unlink(agent_dir / "DONE")
            review_none += 1
            state = "waiting"
            emit()
            time.sleep(2)
            continue

        iteration += 1
        print(f"[{time.strftime('%H:%M:%S')}] Starting iteration {iteration}/{max_iterations}...")
        state = "running"
        emit()

        agent_start = time.time()
        result = _run_claude_stream_stdin(
            prompt_text=_read_text(prompt_file), cwd=repo_root, quiet=quiet
        )
        print(result.text, end="" if result.text.endswith("\n") else "\n")
        last_agent_duration = int(time.time() - agent_start)

        llm_calls += 1
        if result.usage:
            tokens_prompt_last = int(result.usage.get("prompt_tokens", 0))
            tokens_completion_last = int(result.usage.get("completion_tokens", 0))
            tokens_last_total = int(result.usage.get("total_tokens", 0))
            tokens_prompt_total += tokens_prompt_last
            tokens_completion_total += tokens_completion_last
            tokens_total += tokens_last_total
        else:
            llm_usage_missing += 1
            tokens_prompt_last = 0
            tokens_completion_last = 0
            tokens_last_total = 0

        print(f"[{time.strftime('%H:%M:%S')}] Iteration {iteration} complete")
        state = "waiting"
        emit()
        time.sleep(2)


def cmd_launch(args: argparse.Namespace) -> int:
    stream = _stream_name(args.stream)
    repo_root = _resolve_repo_root(args.repo)
    base_branch = _git_out(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    if base_branch == "HEAD":
        _die("Base branch is detached (HEAD). Checkout a branch before launching.")

    agent_home = _resolve_agent_home(args.agent_home)
    slug = _repo_slug(repo_root)
    run_id = _run_id()

    prompt_source = _resolve_prompt_source(
        prompt=args.prompt,
        agent_home=agent_home,
        repo_slug=slug,
        stream_name=stream,
        repo_root=repo_root,
    )

    metrics_root = Path(args.metrics_dir).expanduser() if args.metrics_dir else agent_home / "repos" / slug / "metrics"
    worktree_root = Path(args.worktree_base).expanduser() if args.worktree_base else agent_home / "worktrees" / slug

    metrics_dir = metrics_root / run_id
    run_dir = agent_home / "repos" / slug / "runs" / run_id
    worktree_run_base = worktree_root / run_id
    branch_prefix = f"stream/{run_id}/"

    _ensure_dir(metrics_root)
    _ensure_dir(metrics_dir)
    _ensure_dir(run_dir)
    _ensure_dir(run_dir / ".agent")

    prefix = os.environ.get("AGENT_STREAMS_SESSION_PREFIX", DEFAULT_SESSION_PREFIX)
    session = f"{prefix}-{slug}-{stream}-{run_id}"

    # Write metadata up-front so status/merge can find the run even if tmux dies early.
    predicted_worktree_path = worktree_run_base / stream
    predicted_prompt_file = run_dir / "prompt.md"
    meta = {
        "run_id": run_id,
        "stream_name": stream,
        "repo_root": str(repo_root),
        "repo_slug": slug,
        "base_branch": base_branch,
        "worktree_path": str(predicted_worktree_path),
        "branch_prefix": branch_prefix,
        "metrics_dir": str(metrics_dir),
        "run_dir": str(run_dir),
        "agent_dir": str(run_dir / ".agent"),
        "prompt_source": str(prompt_source),
        "prompt_file": str(predicted_prompt_file),
        "tmux_session": session,
        "max_iterations": int(args.max_iterations),
    }
    _write_text(run_dir / "meta.json", json.dumps(meta, indent=2, sort_keys=True) + "\n")
    meta_env_lines = [
        f"RUN_ID={_shell_quote(run_id)}",
        f"STREAM_NAME={_shell_quote(stream)}",
        f"REPO_ROOT={_shell_quote(str(repo_root))}",
        f"REPO_SLUG={_shell_quote(slug)}",
        f"BASE_BRANCH={_shell_quote(base_branch)}",
        f"WORKTREE_PATH={_shell_quote(str(predicted_worktree_path))}",
        f"BRANCH_PREFIX={_shell_quote(branch_prefix)}",
        f"METRICS_DIR={_shell_quote(str(metrics_dir))}",
        f"RUN_DIR={_shell_quote(str(run_dir))}",
        f"AGENT_DIR={_shell_quote(str(run_dir / '.agent'))}",
        f"PROMPT_SOURCE={_shell_quote(str(prompt_source))}",
        f"PROMPT_FILE={_shell_quote(str(predicted_prompt_file))}",
        f"TMUX_SESSION={_shell_quote(session)}",
        f"MAX_ITERATIONS={int(args.max_iterations)}",
    ]
    _write_text(run_dir / "meta.env", "\n".join(meta_env_lines) + "\n")

    run_stream_cmd = [
        *_self_run_argv(),
        "run-stream",
        "--repo-root",
        str(repo_root),
        "--repo-slug",
        slug,
        "--repo-label",
        repo_root.name,
        "--stream",
        stream,
        "--run-id",
        run_id,
        "--base-branch",
        base_branch,
        "--agent-home",
        str(agent_home),
        "--run-dir",
        str(run_dir),
        "--prompt-source",
        str(prompt_source),
        "--metrics-dir",
        str(metrics_dir),
        "--worktree-base",
        str(worktree_run_base),
        "--branch-prefix",
        branch_prefix,
        "--tmux-session",
        session,
        "--max-iterations",
        str(args.max_iterations),
    ]
    if args.review_prompt:
        run_stream_cmd += ["--review-prompt", args.review_prompt]
    if args.quiet:
        run_stream_cmd += ["--quiet"]
    if args.quiet:
        run_stream_cmd += ["--quiet"]

    if args.no_tmux:
        proc = _run(run_stream_cmd, check=False)
        return int(proc.returncode)

    if not _tmux_available():
        _die("tmux not found (required). Install tmux or run with --no-tmux.")

    _tmux_new_session(session=session, window=stream, cwd=repo_root)
    cmd_str = " ".join(_shell_quote(s) for s in run_stream_cmd)
    _tmux_send(session=session, window=stream, command=cmd_str)

    print("")
    print(f"Stream launched: {stream}")
    print(f"Session: {session}")
    print(f"Run id: {run_id}")
    print(f"Repo slug: {slug}")
    print(f"Metrics dir: {metrics_dir}")
    print(f"Run dir: {run_dir}")
    print(f"Prompt: {prompt_source}")
    print("")
    print("Commands:")
    print(f"  tmux attach -t {session}       # Attach to session")
    print("  Ctrl-b d                        # Detach")
    return 0


def cmd_run_stream(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).expanduser()
    slug = args.repo_slug or _repo_slug(repo_root)
    stream = _stream_name(args.stream)
    run_id = args.run_id
    base_branch = args.base_branch or _git_out(repo_root, "rev-parse", "--abbrev-ref", "HEAD")

    agent_home = _resolve_agent_home(args.agent_home)
    run_dir = Path(args.run_dir).expanduser() if args.run_dir else agent_home / "repos" / slug / "runs" / run_id
    agent_dir = run_dir / ".agent"
    metrics_dir = Path(args.metrics_dir).expanduser()
    worktree_base = Path(args.worktree_base).expanduser()
    branch_prefix = args.branch_prefix or f"stream/{run_id}/"

    prompt_source = Path(args.prompt_source).expanduser()
    if not prompt_source.is_file():
        _die(f"Prompt source not found: {prompt_source}")

    _ensure_dir(run_dir)
    _ensure_dir(agent_dir)
    _ensure_dir(metrics_dir)

    tmux_session = args.tmux_session or ""
    try:
        worktree_path = _ensure_worktree(
            repo_root=repo_root,
            base_dir=worktree_base,
            branch_prefix=branch_prefix,
            stream_name=stream,
        )

        prompt_file_used = run_dir / "prompt.md"
        _append_runtime_block(
            prompt_source=prompt_source,
            prompt_dest=prompt_file_used,
            stream_name=stream,
            run_id=run_id,
            worktree_path=worktree_path,
            agent_dir=agent_dir,
        )

        # Optional compatibility: if repo has streams/<stream>/, symlink its .agent to the run agent dir.
        stream_dir_in_worktree = worktree_path / "streams" / stream
        if stream_dir_in_worktree.is_dir():
            agent_link = stream_dir_in_worktree / ".agent"
            if not agent_link.exists():
                with contextlib.suppress(OSError):
                    os.symlink(str(agent_dir), str(agent_link))

        meta = {
            "run_id": run_id,
            "stream_name": stream,
            "repo_root": str(repo_root),
            "repo_slug": slug,
            "base_branch": base_branch,
            "worktree_path": str(worktree_path),
            "branch_prefix": branch_prefix,
            "metrics_dir": str(metrics_dir),
            "run_dir": str(run_dir),
            "agent_dir": str(agent_dir),
            "prompt_source": str(prompt_source),
            "prompt_file": str(prompt_file_used),
            "tmux_session": tmux_session,
            "max_iterations": int(args.max_iterations),
        }
        _write_text(run_dir / "meta.json", json.dumps(meta, indent=2, sort_keys=True) + "\n")

        # Human-friendly (shell) metadata.
        meta_env_lines = [
            f"RUN_ID={_shell_quote(run_id)}",
            f"STREAM_NAME={_shell_quote(stream)}",
            f"REPO_ROOT={_shell_quote(str(repo_root))}",
            f"REPO_SLUG={_shell_quote(slug)}",
            f"BASE_BRANCH={_shell_quote(base_branch)}",
            f"WORKTREE_PATH={_shell_quote(str(worktree_path))}",
            f"BRANCH_PREFIX={_shell_quote(branch_prefix)}",
            f"METRICS_DIR={_shell_quote(str(metrics_dir))}",
            f"RUN_DIR={_shell_quote(str(run_dir))}",
            f"AGENT_DIR={_shell_quote(str(agent_dir))}",
            f"PROMPT_SOURCE={_shell_quote(str(prompt_source))}",
            f"PROMPT_FILE={_shell_quote(str(prompt_file_used))}",
            f"TMUX_SESSION={_shell_quote(tmux_session)}",
            f"MAX_ITERATIONS={int(args.max_iterations)}",
        ]
        _write_text(run_dir / "meta.env", "\n".join(meta_env_lines) + "\n")

        approved = _run_loop(
            agent_home=agent_home,
            repo_root=worktree_path,
            repo_label=args.repo_label or repo_root.name,
            repo_slug=slug,
            stream_name=stream,
            agent_dir=agent_dir,
            prompt_file=prompt_file_used,
            metrics_dir=metrics_dir,
            max_iterations=int(args.max_iterations),
            review_prompt_override=args.review_prompt,
            quiet=args.quiet,
        )

        if not approved:
            print("No approval; skipping merge.")
            return 0

        print(f"Approved; squashing into {base_branch}...")
        branch_name = _git_out(worktree_path, "rev-parse", "--abbrev-ref", "HEAD")
        if branch_name == "HEAD":
            _die("Worktree is detached; cannot merge.")

        _run(["git", "-C", str(repo_root), "checkout", base_branch])

        merge_proc = _run(
            ["git", "-C", str(repo_root), "merge", "--squash", branch_name],
            check=False,
            capture=True,
        )
        if merge_proc.returncode != 0:
            conflict_files = _git_out(repo_root, "diff", "--name-only", "--diff-filter=U")
            if conflict_files.strip():
                _eprint("Attempting Claude conflict resolution...")
                prompt_text = "\n".join(
                    [
                        "Resolve merge conflicts in the repo.",
                        f"Stream: {stream}",
                        f"Conflicting files: {conflict_files.replace(os.linesep, ' ')}",
                        "Please resolve conflict markers, keep intended behavior, and stage the files.",
                        "Do not run tests.",
                        "",
                    ]
                )
                _run_claude_stream_stdin(
                    prompt_text=prompt_text, cwd=repo_root, quiet=args.quiet
                )

            remaining = _git_out(repo_root, "diff", "--name-only", "--diff-filter=U")
            if remaining.strip():
                _die("Conflicts remain after Claude resolution attempt.")

            _run(["git", "-C", str(repo_root), "add", "-A"])
        else:
            _run(["git", "-C", str(repo_root), "add", "-A"])

        _strip_agent_path_from_repo(repo_root, stream)

        if _run(["git", "-C", str(repo_root), "diff", "--cached", "--quiet"], check=False).returncode == 0:
            print("No changes to commit.")
            return 0

        _run(["git", "-C", str(repo_root), "commit", "-m", f"Squash stream: {stream} ({run_id})"])

        # Cleanup worktree + branch.
        _run(["git", "-C", str(repo_root), "worktree", "remove", "--force", str(worktree_path)], check=False)
        _run(["git", "-C", str(repo_root), "worktree", "prune"], check=False)
        _run(["git", "-C", str(repo_root), "branch", "-D", branch_name], check=False)
        return 0
    finally:
        if tmux_session:
            _kill_tmux(tmux_session)


def _agent_status(agent_dir: Path) -> str:
    if (agent_dir / "APPROVED").exists():
        return "APPROVED"
    if (agent_dir / "ISSUES.md").exists():
        return "ISSUES"
    if (agent_dir / "DONE").exists():
        return "DONE"
    todo = agent_dir / "TODO.md"
    if todo.exists():
        try:
            lines = todo.read_text(encoding="utf-8").splitlines()
        except OSError:
            return "TODO"
        tasks = sum(1 for line in lines if line.startswith("- ["))
        done = sum(1 for line in lines if line.startswith("- [x]"))
        return f"TODO {done}/{tasks}"
    return "not started"


def _parse_metrics_value(metrics_path: Path, metric_prefix: str, stream_name: str) -> str:
    try:
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if not line.startswith(metric_prefix):
                continue
            if f'stream="{stream_name}"' not in line:
                continue
            return line.rsplit(" ", 1)[-1].strip()
    except OSError:
        return ""
    return ""


def _parse_state(metrics_path: Path, stream_name: str) -> str:
    try:
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("streams_state"):
                continue
            if f'stream="{stream_name}"' not in line:
                continue
            if not line.rstrip().endswith(" 1"):
                continue
            m = re.search(r'state="([^"]+)"', line)
            if m:
                return m.group(1)
    except OSError:
        return ""
    return ""


def _env_unquote(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""

    # Handle bash ANSI-C quoting ($'...') used by printf %q in some shells.
    if raw.startswith("$'") and raw.endswith("'") and len(raw) >= 3:
        inner = raw[2:-1]
        try:
            return codecs.decode(inner, "unicode_escape")
        except Exception:
            return inner

    try:
        parts = shlex.split(raw, posix=True)
    except ValueError:
        return raw
    return parts[0] if parts else ""


def _load_meta_env(meta_env_path: Path) -> dict[str, Any]:
    raw: dict[str, str] = {}
    try:
        for line in meta_env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            raw[key.strip()] = _env_unquote(value)
    except OSError:
        return {}

    mapping = {
        "RUN_ID": "run_id",
        "STREAM_NAME": "stream_name",
        "REPO_ROOT": "repo_root",
        "REPO_SLUG": "repo_slug",
        "BASE_BRANCH": "base_branch",
        "WORKTREE_PATH": "worktree_path",
        "BRANCH_PREFIX": "branch_prefix",
        "METRICS_DIR": "metrics_dir",
        "RUN_DIR": "run_dir",
        "AGENT_DIR": "agent_dir",
        "PROMPT_SOURCE": "prompt_source",
        "PROMPT_FILE": "prompt_file",
        "TMUX_SESSION": "tmux_session",
        "MAX_ITERATIONS": "max_iterations",
    }
    meta: dict[str, Any] = {}
    for k, v in raw.items():
        mapped = mapping.get(k)
        if mapped:
            meta[mapped] = v
    return meta


def _load_run_meta(run_dir: Path) -> dict[str, Any]:
    meta_json = run_dir / "meta.json"
    if meta_json.is_file():
        try:
            meta = json.loads(meta_json.read_text(encoding="utf-8"))
            if isinstance(meta, dict):
                meta.setdefault("run_dir", str(run_dir))
                return meta
        except Exception:
            pass

    meta_env = run_dir / "meta.env"
    if meta_env.is_file():
        meta = _load_meta_env(meta_env)
        meta.setdefault("run_dir", str(run_dir))
        return meta
    return {}


def _discover_run_dirs(agent_home: Path) -> list[Path]:
    run_dirs: list[Path] = []
    for run_dir in agent_home.glob("repos/*/runs/*"):
        if not run_dir.is_dir():
            continue
        if (run_dir / "meta.json").is_file() or (run_dir / "meta.env").is_file():
            run_dirs.append(run_dir)

    def sort_key(path: Path) -> float:
        meta = _load_run_meta(path)
        stream = meta.get("stream_name")
        metrics_dir = meta.get("metrics_dir")
        if stream and metrics_dir:
            metrics_path = Path(str(metrics_dir)) / f"{stream}.prom"
            if metrics_path.is_file():
                try:
                    return metrics_path.stat().st_mtime
                except OSError:
                    pass
        for meta_file in (path / "meta.json", path / "meta.env"):
            if meta_file.is_file():
                try:
                    return meta_file.stat().st_mtime
                except OSError:
                    pass
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    return sorted(run_dirs, key=sort_key, reverse=True)


def _tmux_session_alive(session: str) -> bool | None:
    if not session:
        return None
    if not _tmux_available():
        return None
    proc = _run(["tmux", "has-session", "-t", session], check=False, capture=True)
    return proc.returncode == 0


def _resolve_filter_repo_root(repo: str | None) -> Path | None:
    if not repo:
        return None
    try:
        return _resolve_repo_root(repo).resolve()
    except SystemExit:
        return Path(repo).expanduser().resolve()


def _git_is_dirty(repo_root: Path) -> bool:
    proc = _run(["git", "-C", str(repo_root), "status", "--porcelain"], capture=True, check=False)
    return bool((proc.stdout or "").strip())


def _merge_squash(*, meta: dict[str, Any]) -> None:
    stream = meta.get("stream_name") or "stream?"
    run_id = meta.get("run_id") or "run?"

    repo_root = Path(str(meta.get("repo_root") or "")).expanduser()
    worktree_path = Path(str(meta.get("worktree_path") or "")).expanduser()
    run_dir = Path(str(meta.get("run_dir") or "")).expanduser()
    agent_dir = Path(str(meta.get("agent_dir") or (run_dir / ".agent"))).expanduser()

    if not repo_root.is_dir():
        _die(f"Repo root not found: {repo_root}")

    merged = _git_out(repo_root, "log", "--oneline", "--grep", f"Squash stream: {stream} ({run_id})", "-1")
    if merged:
        print(f"Already merged: {merged}")
        return

    if not worktree_path.is_dir():
        _die(f"Worktree not found: {worktree_path}")

    base_branch = str(meta.get("base_branch") or "")
    if not base_branch:
        _die("Missing base_branch in metadata.")

    approved_path = agent_dir / "APPROVED"
    if not approved_path.is_file():
        _die(f"Not approved yet (missing): {approved_path}")

    if _git_is_dirty(repo_root):
        _die(f"Repo has uncommitted changes; clean it first: {repo_root}")

    print(f"Squashing {stream} ({run_id}) into {base_branch}...")
    branch_name = _git_out(worktree_path, "rev-parse", "--abbrev-ref", "HEAD")
    if branch_name == "HEAD":
        _die("Worktree is detached; cannot merge.")

    _run(["git", "-C", str(repo_root), "checkout", base_branch])

    merge_proc = _run(
        ["git", "-C", str(repo_root), "merge", "--squash", branch_name],
        check=False,
        capture=True,
    )
    if merge_proc.returncode != 0:
        conflict_files = _git_out(repo_root, "diff", "--name-only", "--diff-filter=U")
        if conflict_files.strip():
            _eprint("Attempting Claude conflict resolution...")
            prompt_text = "\n".join(
                [
                    "Resolve merge conflicts in the repo.",
                    f"Stream: {stream}",
                    f"Conflicting files: {conflict_files.replace(os.linesep, ' ')}",
                    "Please resolve conflict markers, keep intended behavior, and stage the files.",
                    "Do not run tests.",
                    "",
                ]
            )
            _run_claude_stream_stdin(prompt_text=prompt_text, cwd=repo_root, quiet=False)

        remaining = _git_out(repo_root, "diff", "--name-only", "--diff-filter=U")
        if remaining.strip():
            _die("Conflicts remain after Claude resolution attempt.")

        _run(["git", "-C", str(repo_root), "add", "-A"])
    else:
        _run(["git", "-C", str(repo_root), "add", "-A"])

    _strip_agent_path_from_repo(repo_root, stream)

    if _run(["git", "-C", str(repo_root), "diff", "--cached", "--quiet"], check=False).returncode == 0:
        print("No changes to commit.")
        return

    _run(["git", "-C", str(repo_root), "commit", "-m", f"Squash stream: {stream} ({run_id})"])

    # Cleanup worktree + branch.
    _run(["git", "-C", str(repo_root), "worktree", "remove", "--force", str(worktree_path)], check=False)
    _run(["git", "-C", str(repo_root), "worktree", "prune"], check=False)
    _run(["git", "-C", str(repo_root), "branch", "-D", branch_name], check=False)


def _locate_run_dir(*, agent_home: Path, run_id: str, repo: str | None) -> Path:
    if repo:
        repo_root = _resolve_repo_root(repo)
        slug = _repo_slug(repo_root)
        candidate = agent_home / "repos" / slug / "runs" / run_id
        if candidate.is_dir():
            return candidate
        _die(f"Run not found: {candidate}")

    matches = [p for p in agent_home.glob(f"repos/*/runs/{run_id}") if p.is_dir()]
    if not matches:
        _die(f"Run not found: {run_id}")
    if len(matches) > 1:
        _die(f"Run id is ambiguous across repos; pass --repo. Matches: {', '.join(str(m) for m in matches)}")
    return matches[0]


def cmd_status(args: argparse.Namespace) -> int:
    agent_home = _resolve_agent_home(args.agent_home)
    run_dirs = _discover_run_dirs(agent_home)
    if not run_dirs:
        print(f"No runs found under: {agent_home}/repos/*/runs/*/(meta.json|meta.env)")
        return 0

    filter_repo_root = _resolve_filter_repo_root(args.repo)
    filter_stream = _stream_name(args.stream) if args.stream else None

    runs: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        meta = _load_run_meta(run_dir)
        if not meta:
            continue

        meta_repo = meta.get("repo_root")
        if filter_repo_root and meta_repo:
            try:
                if Path(str(meta_repo)).expanduser().resolve() != filter_repo_root:
                    continue
            except Exception:
                continue

        if filter_stream and filter_stream != meta.get("stream_name"):
            continue

        stream = str(meta.get("stream_name") or "?")
        run_id = str(meta.get("run_id") or run_dir.name)
        repo_root_str = str(meta.get("repo_root") or "")
        worktree_str = str(meta.get("worktree_path") or "")
        base_branch = str(meta.get("base_branch") or "")
        prompt_source = ""
        for candidate in (meta.get("prompt_source"), meta.get("prompt_file")):
            if not candidate:
                continue
            try:
                p = Path(str(candidate)).expanduser()
            except Exception:
                continue
            if p.is_file():
                prompt_source = str(p)
                break
            if not prompt_source:
                prompt_source = str(p)
        run_dir_str = str(meta.get("run_dir") or str(run_dir))
        agent_dir_str = str(meta.get("agent_dir") or str(Path(run_dir_str) / ".agent"))
        tmux_session = str(meta.get("tmux_session") or "")

        agent_dir = Path(agent_dir_str).expanduser()
        agent_status = _agent_status(agent_dir)

        repo_root = Path(repo_root_str).expanduser()
        merged_commit = ""
        if repo_root.is_dir() and stream != "?" and run_id:
            merged_commit = _git_out(repo_root, "log", "--oneline", "--grep", f"Squash stream: {stream} ({run_id})", "-1")

        metrics_dir = Path(str(meta.get("metrics_dir") or "")).expanduser()
        metrics_path = metrics_dir / f"{stream}.prom" if metrics_dir and stream != "?" else Path("/dev/null")
        state = ""
        iteration = ""
        tokens_total = ""
        if metrics_path.is_file():
            state = _parse_state(metrics_path, stream)
            iteration = _parse_metrics_value(metrics_path, "streams_iteration_total", stream)
            try:
                for line in metrics_path.read_text(encoding="utf-8").splitlines():
                    if not line.startswith("streams_tokens_total"):
                        continue
                    if f'stream="{stream}"' not in line:
                        continue
                    if 'kind="total"' not in line:
                        continue
                    tokens_total = line.rsplit(" ", 1)[-1].strip()
                    break
            except OSError:
                pass

        tmux_alive = _tmux_session_alive(tmux_session)

        next_action = ""
        next_command = ""
        if agent_status == "APPROVED" and not merged_commit:
            next_action = "merge"
            next_command = f"agent-streams merge {run_id} --repo {repo_root_str}" if repo_root_str else f"agent-streams merge {run_id}"
        elif tmux_alive:
            next_action = "attach"
            next_command = f"tmux attach -t {tmux_session}"
        elif not merged_commit and agent_status not in ("APPROVED",):
            next_action = "resume"
            next_command = f"agent-streams resume {run_id} --repo {repo_root_str}" if repo_root_str else f"agent-streams resume {run_id}"

        runs.append(
            {
                "run_id": run_id,
                "stream": stream,
                "repo_root": repo_root_str,
                "worktree_path": worktree_str,
                "base_branch": base_branch,
                "prompt_source": prompt_source,
                "run_dir": run_dir_str,
                "agent_dir": agent_dir_str,
                "agent_status": agent_status,
                "merged_commit": merged_commit or None,
                "metrics_path": str(metrics_path) if metrics_path.is_file() else None,
                "state": state or None,
                "iterations": int(iteration) if iteration.isdigit() else None,
                "tokens_total": int(tokens_total) if tokens_total.isdigit() else None,
                "tmux_session": tmux_session or None,
                "tmux_alive": tmux_alive,
                "next_action": next_action or None,
                "next_command": next_command or None,
            }
        )

    limit = args.limit
    if limit is not None:
        if limit < 0:
            _die("--limit must be >= 0")
        if limit > 0:
            runs = runs[:limit]

    if args.format == "json":
        runs = list(reversed(runs))
        print(json.dumps(runs, indent=2))
        return 0

    runs = list(reversed(runs))
    print("=== Agent Streams Runs ===\n")
    for item in runs:
        print(f"--- {item['stream']} ({item['run_id']}) ---")
        if item["repo_root"]:
            print(f"  Repo: {item['repo_root']}")
        if item["worktree_path"]:
            print(f"  Worktree: {item['worktree_path']}")
        if item["base_branch"]:
            print(f"  Base branch: {item['base_branch']}")
        if item["prompt_source"]:
            print(f"  Prompt: {item['prompt_source']}")
        if item["run_dir"]:
            print(f"  Run dir: {item['run_dir']}")
        if item["agent_dir"]:
            print(f"  Agent dir: {item['agent_dir']}")

        print(f"  Agent markers: {item['agent_status']}")
        print(f"  Merge: {item['merged_commit'] or 'not merged'}")

        if item["state"]:
            print(f"  State: {item['state']}")
        if item["iterations"] is not None:
            print(f"  Iterations: {item['iterations']}")
        if item["tokens_total"] is not None:
            print(f"  Tokens total: {item['tokens_total']}")
        if item["metrics_path"]:
            print(f"  Metrics: {item['metrics_path']}")
        else:
            print("  Metrics: none")

        if item["tmux_session"]:
            alive = item["tmux_alive"]
            alive_text = "unknown"
            if alive is True:
                alive_text = "alive"
            elif alive is False:
                alive_text = "dead"
            print(f"  Tmux: {item['tmux_session']} ({alive_text})")

        if item["next_command"]:
            print(f"  Next: {item['next_command']}")
            if item["next_action"] == "attach" and item["tmux_session"]:
                print(f"  Capture: tmux capture-pane -p -t {item['tmux_session']} -S -200")
        print("")

    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    agent_home = _resolve_agent_home(args.agent_home)
    run_dir = _locate_run_dir(agent_home=agent_home, run_id=args.run_id, repo=args.repo)
    meta = _load_run_meta(run_dir)
    if not meta:
        _die(f"Missing metadata for run: {run_dir}")
    _merge_squash(meta=meta)
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    agent_home = _resolve_agent_home(args.agent_home)
    run_dir = _locate_run_dir(agent_home=agent_home, run_id=args.run_id, repo=args.repo)
    meta = _load_run_meta(run_dir)
    if not meta:
        _die(f"Missing metadata for run: {run_dir}")

    repo_root = Path(str(meta.get("repo_root") or "")).expanduser()
    if not repo_root.is_dir():
        _die(f"Repo root not found: {repo_root}")

    slug = str(meta.get("repo_slug") or _repo_slug(repo_root))
    stream = _stream_name(str(meta.get("stream_name") or ""))
    run_id = str(meta.get("run_id") or args.run_id)
    base_branch = str(meta.get("base_branch") or _git_out(repo_root, "rev-parse", "--abbrev-ref", "HEAD"))

    prompt_source = ""
    prompt_candidates = [meta.get("prompt_source"), meta.get("prompt_file")]
    for candidate in prompt_candidates:
        if not candidate:
            continue
        try:
            p = Path(str(candidate)).expanduser()
        except Exception:
            continue
        if p.is_file():
            prompt_source = str(p)
            break

    if not prompt_source:
        _die("Prompt not found (prompt_source/prompt_file missing or deleted).")

    metrics_dir = Path(str(meta.get("metrics_dir") or (agent_home / "repos" / slug / "metrics" / run_id))).expanduser()

    worktree_path = Path(str(meta.get("worktree_path") or "")).expanduser()
    if worktree_path.is_dir():
        worktree_base = worktree_path.parent
    else:
        worktree_base = agent_home / "worktrees" / slug / run_id

    branch_prefix = str(meta.get("branch_prefix") or f"stream/{run_id}/")

    prefix = os.environ.get("AGENT_STREAMS_SESSION_PREFIX", DEFAULT_SESSION_PREFIX)
    session = str(meta.get("tmux_session") or f"{prefix}-{slug}-{stream}-{run_id}")

    max_iterations = args.max_iterations
    meta_max = meta.get("max_iterations")
    if meta_max is not None:
        with contextlib.suppress(TypeError, ValueError):
            max_iterations = int(meta_max)

    run_stream_cmd = [
        *_self_run_argv(),
        "run-stream",
        "--repo-root",
        str(repo_root),
        "--repo-slug",
        slug,
        "--repo-label",
        repo_root.name,
        "--stream",
        stream,
        "--run-id",
        run_id,
        "--base-branch",
        base_branch,
        "--agent-home",
        str(agent_home),
        "--run-dir",
        str(Path(str(meta.get("run_dir") or run_dir)).expanduser()),
        "--prompt-source",
        prompt_source,
        "--metrics-dir",
        str(metrics_dir),
        "--worktree-base",
        str(worktree_base),
        "--branch-prefix",
        branch_prefix,
        "--tmux-session",
        session,
        "--max-iterations",
        str(max_iterations),
    ]
    if args.review_prompt:
        run_stream_cmd += ["--review-prompt", args.review_prompt]

    if args.no_tmux:
        proc = _run(run_stream_cmd, check=False)
        return int(proc.returncode)

    if not _tmux_available():
        _die("tmux not found (required). Install tmux or run with --no-tmux.")

    alive = _tmux_session_alive(session)
    if alive:
        print(f"Tmux session already running: {session}")
        print(f"Attach: tmux attach -t {session}")
        return 0

    _tmux_new_session(session=session, window=stream, cwd=repo_root)
    cmd_str = " ".join(_shell_quote(s) for s in run_stream_cmd)
    _tmux_send(session=session, window=stream, command=cmd_str)

    print("")
    print(f"Resumed: {stream} ({run_id})")
    print(f"Session: {session}")
    print("")
    print("Commands:")
    print(f"  tmux attach -t {session}       # Attach to session")
    print("  Ctrl-b d                        # Detach")
    return 0


def cmd_repo_slug(args: argparse.Namespace) -> int:
    repo_root = _resolve_repo_root(args.repo)
    print(_repo_slug(repo_root))
    return 0


_STREAM_DIR_RE = re.compile(r"^stream(?P<num>[0-9]+)$")
_DEPENDS_LINE_RE = re.compile(r"^\\s*\\*\\*Depends on:\\*\\*\\s*(?P<value>.+?)\\s*$", re.IGNORECASE)
_STREAM_REF_RE = re.compile(r"\\bstream\\s*(?P<num>[0-9]+)\\b", re.IGNORECASE)
_NONE_RE = re.compile(r"^(none|n/?a)\\b", re.IGNORECASE)


def _stream_sort_key(name: str) -> int:
    match = _STREAM_DIR_RE.match(name)
    if not match:
        return 10**9
    return int(match.group("num"))


def _parse_deps_value(raw: str) -> tuple[str, ...]:
    value = raw.strip()
    if not value or _NONE_RE.match(value):
        return ()
    deps = {f"stream{int(m.group('num'))}" for m in _STREAM_REF_RE.finditer(value)}
    return tuple(sorted(deps, key=_stream_sort_key))


def _parse_prompt_deps(prompt_path: Path) -> tuple[str, ...]:
    try:
        lines = prompt_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ()
    for line in lines[:80]:
        match = _DEPENDS_LINE_RE.match(line)
        if match:
            return _parse_deps_value(match.group("value"))
    return ()


def _iter_stream_deps(streams_dir: Path) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    if not streams_dir.is_dir():
        return result
    for entry in sorted(streams_dir.iterdir(), key=lambda p: _stream_sort_key(p.name)):
        if not entry.is_dir():
            continue
        if not _STREAM_DIR_RE.match(entry.name):
            continue
        result[entry.name] = _parse_prompt_deps(entry / "prompt.md")
    return result


def _validate_deps(all_deps: dict[str, tuple[str, ...]]) -> list[str]:
    errors: list[str] = []
    known = set(all_deps.keys())
    for stream, deps in all_deps.items():
        for dep in deps:
            if dep not in known:
                errors.append(f"{stream} depends on missing stream: {dep}")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str, stack: list[str]) -> None:
        if node in visited:
            return
        if node in visiting:
            cycle_start = stack.index(node) if node in stack else 0
            cycle = stack[cycle_start:] + [node]
            errors.append(f"dependency cycle: {' -> '.join(cycle)}")
            return
        visiting.add(node)
        stack.append(node)
        for dep in all_deps.get(node, ()):
            if dep in known:
                visit(dep, stack)
        stack.pop()
        visiting.remove(node)
        visited.add(node)

    for stream in sorted(known, key=_stream_sort_key):
        visit(stream, [])

    return errors


def cmd_deps(args: argparse.Namespace) -> int:
    agent_home = _resolve_agent_home(args.agent_home)

    if args.streams_dir:
        streams_dir = Path(args.streams_dir).expanduser()
    else:
        repo_root = _resolve_repo_root(args.repo)
        slug = _repo_slug(repo_root)
        streams_dir = agent_home / "repos" / slug / "streams"

    all_deps = _iter_stream_deps(streams_dir)

    if args.check:
        errors = _validate_deps(all_deps)
        if errors:
            for err in errors:
                print(err)
            return 2

    selected = all_deps
    if args.stream:
        stream = _stream_name(args.stream)
        if stream not in all_deps:
            _die(f"unknown stream: {stream}", code=2)
        selected = {stream: all_deps[stream]}

    fmt = args.format
    if fmt == "json":
        print(json.dumps({k: list(v) for k, v in selected.items()}, indent=2, sort_keys=True))
        return 0
    if fmt == "csv":
        if args.stream:
            stream = _stream_name(args.stream)
            print(",".join(selected[stream]))
        else:
            for stream in sorted(selected.keys(), key=_stream_sort_key):
                row = ",".join((stream, *selected[stream]))
                print(row)
        return 0
    if fmt == "dot":
        print("digraph streams {")
        print("  rankdir=LR;")
        for stream in sorted(selected.keys(), key=_stream_sort_key):
            deps = selected[stream]
            if not deps:
                print(f"  {stream};")
                continue
            for dep in deps:
                print(f"  {dep} -> {stream};")
        print("}")
        return 0

    for stream in sorted(selected.keys(), key=_stream_sort_key):
        deps = selected[stream]
        print(f"{stream}: {', '.join(deps) if deps else 'none'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-streams")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_launch = sub.add_parser("launch", help="Launch a stream (tmux by default)")
    p_launch.add_argument("stream", help="Stream number (N) or name (streamN)")
    p_launch.add_argument("--repo", help="Target git repo (default: current repo)")
    p_launch.add_argument("--prompt", help="Prompt markdown file (default: auto-resolve)")
    p_launch.add_argument("--agent-home", help="Agent streams home (default: ~/.agent-streams)")
    p_launch.add_argument("--no-tmux", action="store_true", help="Run in current terminal (no tmux)")
    p_launch.add_argument("--worktree-base", help="Set worktree root directory (run id appended)")
    p_launch.add_argument("--metrics-dir", help="Set metrics root directory (run id appended)")
    p_launch.add_argument("--review-prompt", help="Override review prompt markdown")
    p_launch.add_argument("--max-iterations", type=int, default=10)
    p_launch.add_argument("--quiet", action="store_true", help="Suppress streaming status output")
    p_launch.set_defaults(func=cmd_launch)

    p_run = sub.add_parser("run-stream", help=argparse.SUPPRESS)
    p_run.add_argument("--repo-root", required=True)
    p_run.add_argument("--repo-slug")
    p_run.add_argument("--repo-label")
    p_run.add_argument("--stream", required=True)
    p_run.add_argument("--run-id", required=True)
    p_run.add_argument("--base-branch")
    p_run.add_argument("--agent-home")
    p_run.add_argument("--run-dir")
    p_run.add_argument("--prompt-source", required=True)
    p_run.add_argument("--metrics-dir", required=True)
    p_run.add_argument("--worktree-base", required=True)
    p_run.add_argument("--branch-prefix")
    p_run.add_argument("--tmux-session")
    p_run.add_argument("--review-prompt")
    p_run.add_argument("--max-iterations", default="10")
    p_run.add_argument("--quiet", action="store_true", help="Suppress streaming status output")
    p_run.set_defaults(func=cmd_run_stream)

    p_status = sub.add_parser("status", help="Show runs under ~/.agent-streams")
    p_status.add_argument("--agent-home", help="Agent streams home (default: ~/.agent-streams)")
    p_status.add_argument("--repo", help="Filter by repo root path")
    p_status.add_argument("--stream", help="Filter by stream")
    p_status.add_argument("--format", choices=("text", "json"), default="text")
    p_status.add_argument("--limit", type=int, default=10, help="Limit runs shown (0 for all)")
    p_status.set_defaults(func=cmd_status)

    p_slug = sub.add_parser("repo-slug", help="Print repo slug (used in ~/.agent-streams paths)")
    p_slug.add_argument("repo", nargs="?", help="Repo path (default: current repo)")
    p_slug.set_defaults(func=cmd_repo_slug)

    p_deps = sub.add_parser("deps", help="Parse dependencies from prompt metadata")
    p_deps.add_argument("--agent-home", help="Agent streams home (default: ~/.agent-streams)")
    p_deps.add_argument("--repo", help="Repo path (default: current repo)")
    p_deps.add_argument("--streams-dir", help="Path containing streamN/ prompt.md (default: ~/.agent-streams/.../streams)")
    p_deps.add_argument("--stream", help="Only print dependencies for this stream (e.g. stream11)")
    p_deps.add_argument("--format", choices=("text", "csv", "json", "dot"), default="text")
    p_deps.add_argument("--check", action="store_true")
    p_deps.set_defaults(func=cmd_deps)

    p_merge = sub.add_parser("merge", help="Squash-merge an approved run back to base branch")
    p_merge.add_argument("run_id", help="Run id (e.g. 20250101_120000_12345)")
    p_merge.add_argument("--repo", help="Repo path to disambiguate run ids")
    p_merge.add_argument("--agent-home", help="Agent streams home (default: ~/.agent-streams)")
    p_merge.set_defaults(func=cmd_merge)

    p_resume = sub.add_parser("resume", help="Resume a run in tmux (recreates session if needed)")
    p_resume.add_argument("run_id", help="Run id (e.g. 20250101_120000_12345)")
    p_resume.add_argument("--repo", help="Repo path to disambiguate run ids")
    p_resume.add_argument("--agent-home", help="Agent streams home (default: ~/.agent-streams)")
    p_resume.add_argument("--no-tmux", action="store_true", help="Run in current terminal (no tmux)")
    p_resume.add_argument("--review-prompt", help="Override review prompt markdown")
    p_resume.add_argument("--max-iterations", type=int, default=10)
    p_resume.add_argument("--quiet", action="store_true", help="Suppress streaming status output")
    p_resume.set_defaults(func=cmd_resume)

    ns = parser.parse_args(argv)
    return int(ns.func(ns))


if __name__ == "__main__":
    raise SystemExit(main())
