from pathlib import Path
import subprocess

import pytest

import agent_streams as mod


def test_run_claude_json_stdin_parses_text_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args, **kwargs):
        stdout = (
            '{"content":[{"type":"text","text":"hello"}],'
            '"usage":{"input_tokens":2,"output_tokens":3}}'
        )
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=stdout,
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = mod._run_claude_json_stdin(prompt_text="test", cwd=Path("."))
    assert result.text == "hello"
    assert result.usage == {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}


def test_run_claude_json_stdin_falls_back_to_raw(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="not json",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = mod._run_claude_json_stdin(prompt_text="test", cwd=Path("."))
    assert result.text == "not json"
    assert result.usage is None


def test_run_claude_json_stdin_missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(SystemExit):
        mod._run_claude_json_stdin(prompt_text="test", cwd=Path("."))


class _FakeStdin:
    def __init__(self) -> None:
        self.data = ""
        self.closed = False

    def write(self, value: str) -> None:
        self.data += value

    def close(self) -> None:
        self.closed = True


class _FakePopen:
    def __init__(self, lines: list[str]) -> None:
        self.stdin = _FakeStdin()
        self.stdout = lines

    def wait(self) -> int:
        return 0


def test_run_claude_stream_stdin_parses_result(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = [
        '{"type":"system","subtype":"init","model":"test"}\n',
        '{"type":"assistant","message":{"content":[{"type":"tool_use","name":"rg","input":{"q":"x"}}]}}\n',
        '{"type":"user","message":{"content":[{"type":"tool_result","content":"ok","is_error":false}]}}\n',
        '{"type":"result","result":"done","usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}}\n',
    ]

    def fake_popen(*args, **kwargs):
        return _FakePopen(lines)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    result = mod._run_claude_stream_stdin(prompt_text="test", cwd=Path("."), quiet=True)
    assert result.text == "done"
    assert result.usage == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    assert result.raw == "".join(lines)


def test_run_claude_stream_stdin_falls_back_to_raw(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = ["not json\n"]

    def fake_popen(*args, **kwargs):
        return _FakePopen(lines)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    result = mod._run_claude_stream_stdin(prompt_text="test", cwd=Path("."), quiet=True)
    assert result.text == "not json\n"
    assert result.usage is None


def test_format_stream_event_tool_result_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    data = {
        "type": "user",
        "message": {
            "content": [{"type": "tool_result", "content": [{"type": "text", "text": "ok"}]}]
        },
    }
    assert mod._format_stream_event(data) == "[result] ok"
