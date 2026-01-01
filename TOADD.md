# Proposed Additions to agent-streams

## 1. Streaming Output Support

### Problem
Currently, `_run_claude_json_stdin` uses `--output-format json` which outputs a single JSON blob at the end. During long iterations (15+ minutes), there's no visibility into what the agent is doing.

### Solution
Use `--output-format stream-json --verbose` which emits JSON lines in real-time:

```
{"type":"system","subtype":"init"...}           # session started
{"type":"assistant"...tool_use...}              # tool call happening
{"type":"user"...tool_result...}                # tool result
{"type":"assistant"..."text":"..."}             # response text
{"type":"result"...}                            # final summary with costs
```

### Implementation

```python
class Colors:
    GRAY = "\033[90m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def _format_stream_event(data: dict) -> Optional[str]:
    """Format a stream-json event for human-readable output."""
    event_type = data.get("type")
    subtype = data.get("subtype")

    if event_type == "system" and subtype == "init":
        model = data.get("model", "unknown")
        return f"{Colors.GRAY}[init] model={model}{Colors.RESET}"

    elif event_type == "assistant":
        msg = data.get("message", {})
        content = msg.get("content", [])
        parts = []
        for item in content:
            if item.get("type") == "tool_use":
                tool_name = item.get("name", "?")
                tool_input = item.get("input", {})
                input_str = json.dumps(tool_input)
                if len(input_str) > 100:
                    input_str = input_str[:100] + "..."
                parts.append(f"{Colors.CYAN}[tool] {tool_name}: {input_str}{Colors.RESET}")
            elif item.get("type") == "text":
                text = item.get("text", "")
                if text and len(text) > 200:
                    text = text[:200] + "..."
                if text:
                    parts.append(f"{Colors.GREEN}[text] {text}{Colors.RESET}")
        return "\n".join(parts) if parts else None

    elif event_type == "user":
        msg = data.get("message", {})
        content = msg.get("content", [])
        for item in content:
            if item.get("type") == "tool_result":
                result = item.get("content", "")
                is_error = item.get("is_error", False)
                if len(result) > 150:
                    result = result[:150] + "..."
                color = Colors.RED if is_error else Colors.YELLOW
                status = "error" if is_error else "result"
                return f"{color}[{status}] {result}{Colors.RESET}"

    elif event_type == "result":
        cost = data.get("total_cost_usd", 0)
        turns = data.get("num_turns", 0)
        duration = data.get("duration_ms", 0) / 1000
        return f"{Colors.BOLD}[done] {turns} turns, ${cost:.4f}, {duration:.1f}s{Colors.RESET}"

    return None


def _run_claude_stream_stdin(*, prompt_text: str, cwd: Path, quiet: bool = False) -> ClaudeResult:
    """Run claude with streaming output, displaying progress in real-time."""
    claude = _claude_bin()

    cmd = [
        claude, "-p",
        "--output-format", "stream-json",
        "--verbose",
        "--dangerously-skip-permissions"
    ]

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    proc.stdin.write(prompt_text)
    proc.stdin.close()

    raw_lines = []
    final_result = None

    for line in proc.stdout:
        raw_lines.append(line)
        line = line.strip()
        if not line:
            continue

        try:
            data = json.loads(line)
            if data.get("type") == "result":
                final_result = data
            if not quiet:
                formatted = _format_stream_event(data)
                if formatted:
                    print(formatted, file=sys.stderr)
        except json.JSONDecodeError:
            if not quiet:
                print(line, file=sys.stderr)

    proc.wait()
    raw = "".join(raw_lines)

    if final_result:
        text = final_result.get("result", "")
        usage = _extract_usage(final_result)
    else:
        data = _load_json_maybe(raw)
        text = _extract_text(data) if data is not None else ""
        usage = _extract_usage(data) if data is not None else None

    return ClaudeResult(raw=raw, text=text, usage=usage)
```

### Changes Required
1. Add `Colors` class
2. Add `_format_stream_event` function
3. Add `_run_claude_stream_stdin` function
4. Replace calls to `_run_claude_json_stdin` with `_run_claude_stream_stdin`
5. Optionally add `--quiet` flag to suppress streaming

---

## 2. Test Suite

### Problem
No tests exist. Changes risk regressions.

### Proposed Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── test_helpers.py          # Unit tests for helper functions
├── test_claude_runner.py    # Tests for claude invocation
├── test_git_operations.py   # Tests for git/worktree operations
├── test_stream_lifecycle.py # Integration tests for stream lifecycle
└── test_cli.py              # CLI argument parsing tests
```

### Test Cases

#### test_helpers.py
```python
def test_extract_text_from_json():
    """_extract_text extracts text from various JSON structures."""
    assert _extract_text({"content": [{"type": "text", "text": "hello"}]}) == "hello"
    assert _extract_text({"message": {"content": [{"type": "text", "text": "world"}]}}) == "world"
    assert _extract_text({"result": "done"}) == "done"

def test_extract_text_empty():
    """_extract_text returns empty string for missing content."""
    assert _extract_text({}) == ""
    assert _extract_text(None) == ""

def test_extract_usage():
    """_extract_usage parses token counts."""
    usage = _extract_usage({"usage": {"input_tokens": 100, "output_tokens": 50}})
    assert usage["prompt"] == 100
    assert usage["completion"] == 50

def test_load_json_maybe_valid():
    """_load_json_maybe parses valid JSON."""
    assert _load_json_maybe('{"key": "value"}') == {"key": "value"}

def test_load_json_maybe_with_prefix():
    """_load_json_maybe handles JSON with non-JSON prefix."""
    assert _load_json_maybe('some text {"key": "value"} more') == {"key": "value"}

def test_load_json_maybe_invalid():
    """_load_json_maybe returns None for invalid JSON."""
    assert _load_json_maybe("not json") is None

def test_repo_slug_generation():
    """_repo_slug generates consistent slugs."""
    # Should be deterministic
    slug1 = _repo_slug("/path/to/repo")
    slug2 = _repo_slug("/path/to/repo")
    assert slug1 == slug2
    # Should include repo name
    assert "repo" in slug1
```

#### test_claude_runner.py
```python
@pytest.fixture
def mock_claude(monkeypatch):
    """Mock claude CLI that returns canned responses."""
    def mock_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout='{"type":"result","result":"test done","total_cost_usd":0.01}',
            stderr=""
        )
    monkeypatch.setattr(subprocess, "run", mock_run)

def test_run_claude_json_success(mock_claude):
    """_run_claude_json_stdin returns ClaudeResult on success."""
    result = _run_claude_json_stdin(prompt_text="test", cwd=Path("."))
    assert result.text == "test done"

def test_run_claude_json_extracts_usage(mock_claude):
    """_run_claude_json_stdin extracts usage stats."""
    # Mock response with usage
    result = _run_claude_json_stdin(prompt_text="test", cwd=Path("."))
    # Check usage extraction

def test_run_claude_missing_binary(monkeypatch):
    """_run_claude_json_stdin fails gracefully if claude not found."""
    monkeypatch.setenv("AGENT_STREAMS_CLAUDE_BIN", "/nonexistent/claude")
    with pytest.raises(SystemExit):
        _run_claude_json_stdin(prompt_text="test", cwd=Path("."))
```

#### test_stream_lifecycle.py
```python
def test_stream_creates_worktree(tmp_path):
    """Stream launch creates git worktree."""
    # Setup mock repo
    # Run stream launch
    # Verify worktree created

def test_stream_done_triggers_review():
    """DONE file triggers overseer review."""
    # Setup stream with DONE
    # Verify review is triggered

def test_stream_approved_triggers_merge():
    """APPROVED file triggers squash merge."""
    # Setup stream with APPROVED
    # Verify merge occurs

def test_stream_issues_continues_iteration():
    """ISSUES.md causes another iteration."""
    # Setup stream with ISSUES.md
    # Verify iteration count increases
    # Verify DONE is removed

def test_stream_max_iterations_stops():
    """Stream stops after max iterations."""
    # Run stream to max iterations
    # Verify it stops gracefully
```

#### test_cli.py
```python
def test_launch_requires_stream_number():
    """launch command requires stream number."""
    with pytest.raises(SystemExit):
        main(["launch"])

def test_status_runs_without_args():
    """status command works without arguments."""
    # Should not raise
    main(["status"])

def test_repo_slug_command():
    """repo-slug command outputs slug."""
    # Capture stdout
    main(["repo-slug"])
    # Verify output
```

### Dependencies to Add
```toml
[dependency-groups]
dev = ["ruff", "ty", "pytest", "pytest-cov"]
```

### Running Tests
```bash
uv run pytest tests/ -v
uv run pytest tests/ -v --cov=agent_streams
```
