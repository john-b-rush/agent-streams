from pathlib import Path

import agent_streams as mod


def test_emit_metrics_atomic_write(tmp_path: Path) -> None:
    metrics_file = tmp_path / "stream1.prom"
    labels = 'repo="repo",stream="stream1"'

    mod._emit_metrics(
        metrics_file=metrics_file,
        labels=labels,
        iteration=1,
        max_iterations=2,
        state="running",
        last_agent_duration=3,
        last_review_duration=4,
        review_approved=0,
        review_issues=0,
        review_none=0,
        llm_calls=1,
        llm_usage_missing=0,
        tokens_prompt_total=5,
        tokens_completion_total=6,
        tokens_total=11,
        tokens_prompt_last=5,
        tokens_completion_last=6,
        tokens_last_total=11,
    )

    assert metrics_file.is_file()
    metrics_text = metrics_file.read_text(encoding="utf-8")
    assert (
        f'streams_state{{{labels},state="running"}} 1' in metrics_text
    )
    assert f"streams_iteration_total{{{labels}}} 1" in metrics_text

    tmp_files = list(tmp_path.glob(f".{metrics_file.name}.*.tmp"))
    assert tmp_files == []
