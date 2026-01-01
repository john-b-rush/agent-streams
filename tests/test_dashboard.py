import json
import subprocess
import sys
import time
from pathlib import Path


def test_metrics_dashboard_dump(tmp_path: Path) -> None:
    agent_home = tmp_path / "agent_home"
    metrics_path = (
        agent_home / "repos" / "demo-repo" / "metrics" / "20250101_000000_12345"
    )
    metrics_path.mkdir(parents=True)
    stream = "stream1"
    ts = int(time.time()) - 5
    content = "\n".join(
        [
            "# HELP streams_state Current lifecycle state for the stream.",
            "# TYPE streams_state gauge",
            f'streams_state{{repo="demo",stream="{stream}",state="running"}} 1',
            f'streams_iteration_total{{repo="demo",stream="{stream}"}} 2',
            f'streams_max_iterations{{repo="demo",stream="{stream}"}} 5',
            f'streams_last_update_timestamp{{repo="demo",stream="{stream}"}} {ts}',
            f'streams_llm_calls_total{{repo="demo",stream="{stream}"}} 3',
            f'streams_tokens_total{{repo="demo",stream="{stream}",kind="total"}} 99',
        ]
    )
    (metrics_path / f"{stream}.prom").write_text(content + "\n", encoding="utf-8")

    script = Path(__file__).resolve().parents[1] / "scripts" / "metrics_dashboard.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dump",
            "--agent-home",
            str(agent_home),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    payload = json.loads(proc.stdout)
    assert payload["summary"]["total"] == 1
    assert payload["summary"]["states"]["running"] == 1

    run = payload["runs"][0]
    assert run["repo"] == "demo"
    assert run["repo_slug"] == "demo-repo"
    assert run["run_id"] == "20250101_000000_12345"
    assert run["stream"] == stream
    assert run["state"] == "running"
    assert run["iteration"] == 2
    assert run["max_iterations"] == 5
    assert run["tokens_total"] == 99
    assert run["llm_calls"] == 3
