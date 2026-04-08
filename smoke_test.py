#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request


BASE_URL = "http://127.0.0.1:7860"


def request(method: str, path: str, payload=None):
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"{BASE_URL}{path}", data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_server() -> None:
    for _ in range(30):
        try:
            request("GET", "/health")
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("Server did not become healthy in time.")


def main() -> None:
    server = subprocess.Popen([sys.executable, "app.py"])
    try:
        wait_for_server()
        tasks = request("GET", "/tasks")["tasks"]
        assert len(tasks) >= 4
        for task in tasks:
            reset = request("POST", "/reset", {"task_id": task["task_id"]})
            session_id = reset["session_id"]
            assert reset["task"]["task_id"] == task["task_id"]
            state = request("GET", f"/state?session_id={session_id}")
            assert state["session_id"] == session_id
            assert state["observation"]["task_id"] == task["task_id"]
        print("Smoke test passed.")
    finally:
        server.terminate()
        server.wait(timeout=10)


if __name__ == "__main__":
    main()
