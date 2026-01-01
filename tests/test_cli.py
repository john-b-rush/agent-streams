from pathlib import Path

import pytest

import agent_streams as mod


def test_main_requires_subcommand() -> None:
    with pytest.raises(SystemExit):
        mod.main([])


def test_launch_requires_stream_argument() -> None:
    with pytest.raises(SystemExit):
        mod.main(["launch"])


def test_status_no_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(mod, "_resolve_agent_home", lambda _: tmp_path)
    rc = mod.main(["status"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "No runs found under:" in out


def test_repo_slug_command_prints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_root = tmp_path / "repo"
    monkeypatch.setattr(mod, "_resolve_repo_root", lambda _: repo_root)
    expected = mod._repo_slug(repo_root)
    rc = mod.main(["repo-slug"])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert out == expected
