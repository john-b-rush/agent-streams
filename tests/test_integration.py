import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _script_path() -> Path:
    return Path(__file__).resolve().parents[1] / "agent_streams.py"


def _require_claude() -> str:
    claude = os.environ.get("AGENT_STREAMS_CLAUDE_BIN", "claude")
    if "/" in claude or "\\" in claude:
        if not Path(claude).is_file():
            pytest.fail(f"Claude binary not found at {claude}")
        return claude
    if shutil.which(claude) is None:
        pytest.fail(f"Claude binary not found on PATH: {claude}")
    return claude


def _run(
    cmd: list[str], *, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
    )


def _init_repo(repo_root: Path) -> str:
    init = _run(["git", "init"], cwd=repo_root)
    assert init.returncode == 0, init.stderr
    cfg_email = _run(
        ["git", "config", "user.email", "test@example.com"], cwd=repo_root
    )
    assert cfg_email.returncode == 0, cfg_email.stderr
    cfg_name = _run(["git", "config", "user.name", "Test User"], cwd=repo_root)
    assert cfg_name.returncode == 0, cfg_name.stderr
    (repo_root / "README.md").write_text("hello\n", encoding="utf-8")
    add = _run(["git", "add", "README.md"], cwd=repo_root)
    assert add.returncode == 0, add.stderr
    commit = _run(["git", "commit", "-m", "init"], cwd=repo_root)
    assert commit.returncode == 0, commit.stderr
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    assert branch.returncode == 0, branch.stderr
    return branch.stdout.strip()


def _repo_slug(script: Path, repo_root: Path) -> str:
    proc = _run([sys.executable, str(script), "repo-slug", str(repo_root)])
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def test_run_stream_creates_worktree_and_metrics(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    _require_claude()

    script = _script_path()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    base_branch = _init_repo(repo_root)

    agent_home = tmp_path / "agent_home"
    slug = _repo_slug(script, repo_root)
    run_id = "20250101_000000_12345"
    stream = "stream1"

    run_dir = agent_home / "repos" / slug / "runs" / run_id
    metrics_dir = agent_home / "repos" / slug / "metrics" / run_id
    worktree_base = agent_home / "worktrees" / slug / run_id

    prompt_source = tmp_path / "prompt.md"
    prompt_source.write_text("hello\n", encoding="utf-8")

    proc = _run(
        [
            sys.executable,
            str(script),
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
            str(worktree_base),
            "--branch-prefix",
            "stream/test/",
            "--max-iterations",
            "1",
        ]
    )
    assert proc.returncode == 0, proc.stderr

    worktree_path = worktree_base / stream
    assert worktree_path.is_dir()
    inside = _run(
        ["git", "-C", str(worktree_path), "rev-parse", "--is-inside-work-tree"]
    )
    assert inside.returncode == 0
    assert inside.stdout.strip() == "true"

    prompt_file = run_dir / "prompt.md"
    assert prompt_file.is_file()
    assert "AGENT_STREAMS_RUNTIME" in prompt_file.read_text(encoding="utf-8")

    metrics_file = metrics_dir / f"{stream}.prom"
    assert metrics_file.is_file()
    metrics_text = metrics_file.read_text(encoding="utf-8")
    assert (
        f'streams_iteration_total{{repo="{repo_root.name}",stream="{stream}"}} 1'
        in metrics_text
    )
    assert (
        f'streams_state{{repo="{repo_root.name}",stream="{stream}",state="maxed"}} 1'
        in metrics_text
    )
    assert (
        f'streams_llm_calls_total{{repo="{repo_root.name}",stream="{stream}"}} 1'
        in metrics_text
    )

    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["worktree_path"] == str(worktree_path)
    assert meta["prompt_source"] == str(prompt_source)
    assert meta["prompt_file"] == str(prompt_file)


def test_status_reports_metrics(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    _require_claude()

    script = _script_path()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    base_branch = _init_repo(repo_root)

    agent_home = tmp_path / "agent_home"
    slug = _repo_slug(script, repo_root)
    run_id = "20250101_000000_12346"
    stream = "stream1"

    run_dir = agent_home / "repos" / slug / "runs" / run_id
    metrics_dir = agent_home / "repos" / slug / "metrics" / run_id
    worktree_base = agent_home / "worktrees" / slug / run_id

    prompt_source = tmp_path / "prompt.md"
    prompt_source.write_text("hello\n", encoding="utf-8")

    run_proc = _run(
        [
            sys.executable,
            str(script),
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
            str(worktree_base),
            "--branch-prefix",
            "stream/test/",
            "--max-iterations",
            "1",
        ]
    )
    assert run_proc.returncode == 0, run_proc.stderr

    status_proc = _run(
        [
            sys.executable,
            str(script),
            "status",
            "--agent-home",
            str(agent_home),
            "--format",
            "json",
        ]
    )
    assert status_proc.returncode == 0, status_proc.stderr

    runs = json.loads(status_proc.stdout)
    match = next(item for item in runs if item["run_id"] == run_id)
    assert match["stream"] == stream
    assert match["state"] == "maxed"
    assert match["iterations"] == 1
    assert match["tokens_total"] is not None
    assert match["metrics_path"] == str(metrics_dir / f"{stream}.prom")
