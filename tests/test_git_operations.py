from pathlib import Path
import subprocess

import pytest

import agent_streams as mod


def test_worktree_list_paths_parses_porcelain(monkeypatch: pytest.MonkeyPatch) -> None:
    output = """worktree /repo
HEAD abcdef1234567890
branch refs/heads/main
worktree /repo/worktrees/stream1
HEAD 1234567890abcdef
branch refs/heads/stream1
"""

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout=output, stderr="")

    monkeypatch.setattr(mod, "_run", fake_run)

    paths = mod._worktree_list_paths(Path("/repo"))
    assert paths == {"/repo", "/repo/worktrees/stream1"}


def test_ensure_worktree_returns_existing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base_dir = tmp_path / "worktrees"
    stream_name = "stream1"
    worktree_path = base_dir / stream_name

    monkeypatch.setattr(mod, "_worktree_list_paths", lambda _: {str(worktree_path)})

    def fake_run(*args, **kwargs):
        raise AssertionError("_run should not be called")

    monkeypatch.setattr(mod, "_run", fake_run)

    result = mod._ensure_worktree(
        repo_root=tmp_path,
        base_dir=base_dir,
        branch_prefix="agent-",
        stream_name=stream_name,
    )
    assert result == worktree_path


def test_ensure_worktree_existing_path_not_worktree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base_dir = tmp_path / "worktrees"
    stream_name = "stream1"
    worktree_path = base_dir / stream_name
    worktree_path.mkdir(parents=True)

    monkeypatch.setattr(mod, "_worktree_list_paths", lambda _: set())

    with pytest.raises(SystemExit):
        mod._ensure_worktree(
            repo_root=tmp_path,
            base_dir=base_dir,
            branch_prefix="agent-",
            stream_name=stream_name,
        )


def test_ensure_worktree_reuses_existing_branch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base_dir = tmp_path / "worktrees"
    stream_name = "stream1"
    worktree_path = base_dir / stream_name
    branch_name = f"agent-{stream_name}"
    calls: list[list[str]] = []

    monkeypatch.setattr(mod, "_worktree_list_paths", lambda _: set())

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        if "show-ref" in argv:
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(mod, "_run", fake_run)

    result = mod._ensure_worktree(
        repo_root=tmp_path,
        base_dir=base_dir,
        branch_prefix="agent-",
        stream_name=stream_name,
    )

    assert result == worktree_path
    assert any(
        call[:6]
        == ["git", "-C", str(tmp_path), "worktree", "add", str(worktree_path)]
        and call[6:] == [branch_name]
        for call in calls
    )


def test_ensure_worktree_creates_branch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base_dir = tmp_path / "worktrees"
    stream_name = "stream1"
    worktree_path = base_dir / stream_name
    branch_name = f"agent-{stream_name}"
    calls: list[list[str]] = []

    monkeypatch.setattr(mod, "_worktree_list_paths", lambda _: set())

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        if "show-ref" in argv:
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(mod, "_run", fake_run)

    result = mod._ensure_worktree(
        repo_root=tmp_path,
        base_dir=base_dir,
        branch_prefix="agent-",
        stream_name=stream_name,
    )

    assert result == worktree_path
    assert any(
        call[:7]
        == [
            "git",
            "-C",
            str(tmp_path),
            "worktree",
            "add",
            "-b",
            branch_name,
        ]
        and call[7:] == [str(worktree_path)]
        for call in calls
    )


def test_strip_agent_path_from_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    agent_path = repo_root / "streams" / "stream1" / ".agent"
    agent_path.parent.mkdir(parents=True)
    agent_path.write_text("data", encoding="utf-8")

    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(mod, "_run", fake_run)

    mod._strip_agent_path_from_repo(repo_root, "stream1")

    assert not agent_path.exists()
    assert [
        "git",
        "-C",
        str(repo_root),
        "add",
        "-A",
        "--",
        str(agent_path.relative_to(repo_root)),
    ] in calls


def test_git_is_dirty_true(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout=" M file\n", stderr="")

    monkeypatch.setattr(mod, "_run", fake_run)

    assert mod._git_is_dirty(tmp_path) is True


def test_git_is_dirty_false(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(mod, "_run", fake_run)

    assert mod._git_is_dirty(tmp_path) is False
