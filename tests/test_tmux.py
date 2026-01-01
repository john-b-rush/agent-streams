from pathlib import Path
import subprocess

import pytest

import agent_streams as mod


def test_tmux_session_alive_empty_session() -> None:
    assert mod._tmux_session_alive("") is None


def test_tmux_session_alive_tmux_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "_tmux_available", lambda: False)
    assert mod._tmux_session_alive("session") is None


def test_tmux_session_alive_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "_tmux_available", lambda: True)

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(mod, "_run", fake_run)

    assert mod._tmux_session_alive("session") is True


def test_tmux_session_alive_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "_tmux_available", lambda: True)

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="")

    monkeypatch.setattr(mod, "_run", fake_run)

    assert mod._tmux_session_alive("session") is False


def test_tmux_new_session_invokes_tmux(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(mod, "_run", fake_run)

    mod._tmux_new_session(session="session", window="stream1", cwd=tmp_path)

    assert calls == [
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            "session",
            "-n",
            "stream1",
            "-c",
            str(tmp_path),
        ]
    ]


def test_tmux_send_invokes_tmux(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(mod, "_run", fake_run)

    mod._tmux_send(session="session", window="stream1", command="echo hi")

    assert calls == [
        ["tmux", "send-keys", "-t", "session:stream1", "echo hi", "Enter"]
    ]


def test_kill_tmux_skips_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(mod, "_tmux_available", lambda: False)
    monkeypatch.setattr(mod, "_run", fake_run)

    mod._kill_tmux("session")

    assert calls == []


def test_kill_tmux_calls_has_and_kill(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(mod, "_tmux_available", lambda: True)
    monkeypatch.setattr(mod, "_run", fake_run)

    mod._kill_tmux("session")

    assert calls == [
        ["tmux", "has-session", "-t", "session"],
        ["tmux", "kill-session", "-t", "session"],
    ]
