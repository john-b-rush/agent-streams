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
