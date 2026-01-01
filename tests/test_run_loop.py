import shutil
import subprocess
from pathlib import Path

import pytest

import agent_streams as mod

if shutil.which("git") is None:
    pytest.skip("git not installed", allow_module_level=True)


def _git(cmd: list[str], *, cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr


def test_run_loop_dirty_done_writes_issues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git(["git", "init"], cwd=repo_root)
    _git(["git", "config", "user.email", "test@example.com"], cwd=repo_root)
    _git(["git", "config", "user.name", "Test User"], cwd=repo_root)
    (repo_root / "README.md").write_text("hello\n", encoding="utf-8")
    _git(["git", "add", "README.md"], cwd=repo_root)
    _git(["git", "commit", "-m", "init"], cwd=repo_root)
    (repo_root / "README.md").write_text("dirty\n", encoding="utf-8")

    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "DONE").write_text("done\n", encoding="utf-8")

    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("hello\n", encoding="utf-8")
    metrics_dir = tmp_path / "metrics"

    calls = {"count": 0}

    def fake_run_claude(*, prompt_text: str, cwd: Path, quiet: bool = False) -> mod.ClaudeResult:
        calls["count"] += 1
        return mod.ClaudeResult(raw="ok", text="ok", usage=None)

    monkeypatch.setattr(mod, "_run_claude_stream_stdin", fake_run_claude)
    monkeypatch.setattr(mod.time, "sleep", lambda _: None)

    result = mod._run_loop(
        agent_home=tmp_path,
        repo_root=repo_root,
        repo_label="repo",
        repo_slug="slug",
        stream_name="stream1",
        agent_dir=agent_dir,
        prompt_file=prompt_file,
        metrics_dir=metrics_dir,
        max_iterations=1,
        review_prompt_override=None,
        quiet=True,
    )

    assert result is False
    assert calls["count"] == 1
    assert not (agent_dir / "DONE").exists()
    issues = (agent_dir / "ISSUES.md").read_text(encoding="utf-8")
    assert "Worktree has uncommitted" in issues
