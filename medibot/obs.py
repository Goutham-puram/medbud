"""Observability hooks: a request id per call, timers, and ONE structured JSON log line per request.

This is the seam Assignment 3 plugs into (LangSmith tracing, heuristic checks over the trace):
every /chat response carries request_id, route, scores and timings, and the same fields are logged.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid

LOG_FILE = os.getenv("MEDIBOT_LOG_FILE", "")  # e.g. logs/medbud.jsonl; empty = stdout only

log = logging.getLogger("medbud")
if not log.handlers:
    log.setLevel(logging.INFO)
    log.propagate = False
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(stream)
    if LOG_FILE:
        os.makedirs(os.path.dirname(LOG_FILE) or ".", exist_ok=True)
        fh = logging.FileHandler(LOG_FILE)
        fh.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(fh)


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


class Timer:
    """with Timer(timings, "retrieval"): ...   -> timings["retrieval_ms"] = elapsed"""

    def __init__(self, sink: dict, name: str):
        self.sink, self.name = sink, name

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.sink[f"{self.name}_ms"] = round((time.perf_counter() - self.t0) * 1000)


def log_event(**fields) -> None:
    fields.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S"))
    log.info(json.dumps(fields, default=str))
