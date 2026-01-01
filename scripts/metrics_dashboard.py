#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

DEFAULT_AGENT_HOME = Path("~/.agent-streams").expanduser()
METRIC_LINE_RE = re.compile(
    r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\{([^}]*)\}\s+([-+]?[\d.]+(?:[eE][-+]?\d+)?)$'
)
LABEL_RE = re.compile(r'(\w+)="((?:\\.|[^"\\])*)"')


def _resolve_agent_home(value: str | None) -> Path:
    if value:
        return Path(value).expanduser()
    env = os.environ.get("AGENT_STREAMS_HOME")
    if env:
        return Path(env).expanduser()
    return DEFAULT_AGENT_HOME


def _parse_labels(raw: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    for key, value in LABEL_RE.findall(raw):
        labels[key] = value.replace('\\"', '"').replace("\\\\", "\\")
    return labels


def _parse_int(value: str) -> int | None:
    try:
        return int(float(value))
    except ValueError:
        return None


def _collect_metrics(agent_home: Path) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    now = int(time.time())
    metrics_root = agent_home / "repos"
    if not metrics_root.is_dir():
        return {"generated_at": now, "runs": [], "summary": {"total": 0, "states": {}}}

    for metrics_file in metrics_root.glob("*/metrics/*/*.prom"):
        run_id = metrics_file.parent.name
        repo_slug = metrics_file.parents[2].name
        stream = metrics_file.stem

        record: dict[str, Any] = {
            "repo": None,
            "repo_slug": repo_slug,
            "run_id": run_id,
            "stream": stream,
            "metrics_path": str(metrics_file),
            "state": "unknown",
            "iteration": None,
            "max_iterations": None,
            "last_update": int(metrics_file.stat().st_mtime),
            "llm_calls": None,
            "tokens_total": None,
            "tokens_prompt": None,
            "tokens_completion": None,
        }

        try:
            lines = metrics_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            runs.append(record)
            continue

        for line in lines:
            if not line or line.startswith("#"):
                continue
            match = METRIC_LINE_RE.match(line)
            if not match:
                continue
            metric, labels_str, value_str = match.groups()
            labels = _parse_labels(labels_str)
            value = _parse_int(value_str)
            if value is None:
                continue
            if labels.get("repo"):
                record["repo"] = labels["repo"]

            if metric == "streams_state" and labels.get("state") and value == 1:
                record["state"] = labels["state"]
            elif metric == "streams_iteration_total":
                record["iteration"] = value
            elif metric == "streams_max_iterations":
                record["max_iterations"] = value
            elif metric == "streams_last_update_timestamp":
                record["last_update"] = value
            elif metric == "streams_llm_calls_total":
                record["llm_calls"] = value
            elif metric == "streams_tokens_total":
                kind = labels.get("kind")
                if kind == "total":
                    record["tokens_total"] = value
                elif kind == "prompt":
                    record["tokens_prompt"] = value
                elif kind == "completion":
                    record["tokens_completion"] = value

        if record["repo"] is None:
            record["repo"] = repo_slug

        last_update = record.get("last_update") or 0
        record["age_sec"] = max(0, now - int(last_update))
        runs.append(record)

    runs.sort(key=lambda item: (item.get("last_update") or 0, item.get("run_id") or ""))
    state_counts: dict[str, int] = {}
    for item in runs:
        state = item.get("state") or "unknown"
        state_counts[state] = state_counts.get(state, 0) + 1

    return {"generated_at": now, "runs": runs, "summary": {"total": len(runs), "states": state_counts}}


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "agent-streams-dashboard/0.1"

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._serve_index()
            return
        if path == "/metrics.json":
            self._serve_metrics()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def _serve_index(self) -> None:
        index_path = Path(__file__).resolve().parents[1] / "dashboard" / "index.html"
        try:
            payload = index_path.read_bytes()
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "index.html missing")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _serve_metrics(self) -> None:
        agent_home = self.server.agent_home  # type: ignore[attr-defined]
        payload = json.dumps(_collect_metrics(agent_home)).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve a local agent-streams metrics dashboard.")
    parser.add_argument("--agent-home", help="Agent home (default: AGENT_STREAMS_HOME or ~/.agent-streams)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind (default: 8765)")
    parser.add_argument("--dump", action="store_true", help="Print metrics JSON and exit")
    args = parser.parse_args()

    agent_home = _resolve_agent_home(args.agent_home)
    if args.dump:
        print(json.dumps(_collect_metrics(agent_home), indent=2, sort_keys=True))
        return 0

    server = HTTPServer((args.host, args.port), DashboardHandler)
    server.agent_home = agent_home  # type: ignore[attr-defined]
    print(f"Dashboard: http://{args.host}:{args.port}")
    print(f"Agent home: {agent_home}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
