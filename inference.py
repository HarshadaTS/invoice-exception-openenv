#!/usr/bin/env python3
"""
Baseline inference script for Invoice Exception OpenEnv.

MANDATORY STDOUT FORMAT:
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from openai import OpenAI


BENCHMARK = "invoice-exception-openenv"
API_BASE_URL = os.environ.get("API_BASE_URL", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "deterministic-baseline")
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
ENV_URL = os.environ.get("ENV_URL", "http://127.0.0.1:7860")

TASK_PLANS: Dict[str, List[str]] = {
    "invoice_easy": ["review_invoice", "compare_po", "check_receipt", "approve", "done"],
    "invoice_medium": [
        "review_invoice",
        "compare_po",
        "check_receipt",
        "request_supporting_doc",
        "route_to_review",
        "done",
    ],
    "invoice_hard": [
        "review_invoice",
        "compare_po",
        "check_receipt",
        "search_duplicate",
        "flag_duplicate",
        "reject",
        "done",
    ],
    "invoice_expert": [
        "review_invoice",
        "compare_po",
        "request_supporting_doc",
        "check_receipt",
        "approve",
        "done",
    ],
}
def http_json(method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"{ENV_URL}{path}", data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def log_start(task_name: str) -> None:
    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def log_step(step: int, action_type: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_text = "null" if not error else str(error).replace("\n", " ").replace("\r", " ")
    print(
        f"[STEP] step={step} action={action_type} reward={reward:.2f} done={str(done).lower()} error={error_text}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def llm_next_action(task_id: str, observation: Dict[str, Any]) -> Optional[str]:
    if not (API_BASE_URL and MODEL_NAME and API_KEY):
        return None

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    prompt = (
        "You are resolving a supplier invoice exception.\n"
        "Return exactly one action from this list: "
        "review_invoice, compare_po, check_receipt, search_duplicate, request_supporting_doc, "
        "approve, route_to_review, flag_duplicate, reject, done.\n"
        f"Task: {task_id}\n"
        f"Observation JSON: {json.dumps(observation, separators=(',', ':'))}\n"
        'Answer with JSON only, like {"action_type":"compare_po"}.'
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a precise invoice-operations agent."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    text = response.choices[0].message.content or ""
    try:
        parsed = json.loads(text)
        action = parsed.get("action_type")
        if isinstance(action, str):
            return action
    except json.JSONDecodeError:
        return None
    return None


def deterministic_next_action(task_id: str, step_index: int) -> str:
    plan = TASK_PLANS[task_id]
    if step_index >= len(plan):
        return "done"
    return plan[step_index]


def run_task(task: Dict[str, Any]) -> Dict[str, Any]:
    task_id = task["task_id"]
    task_name = task["name"]
    reset = http_json("POST", "/reset", {"task_id": task_id})
    session_id = reset["session_id"]
    observation = reset["observation"]
    rewards: List[float] = []
    success = False
    steps_taken = 0
    score = 0.0

    log_start(task_name)
    try:
        while True:
            action_type = llm_next_action(task_id, observation) or deterministic_next_action(task_id, steps_taken)
            result = http_json(
                "POST",
                "/step",
                {
                    "session_id": session_id,
                    "action_type": action_type,
                    "rationale": "baseline_policy",
                },
            )
            reward = float(result["reward"])
            done = bool(result["done"])
            observation = result["observation"]
            rewards.append(reward)
            steps_taken += 1
            log_step(steps_taken, action_type, reward, done, observation.get("last_action_error"))
            if done:
                score = float(result["score"])
                success = score >= 0.8
                break
    except urllib.error.URLError as exc:
        log_step(steps_taken + 1, "network_error", 0.0, True, str(exc.reason))
        success = False
    except Exception as exc:
        log_step(steps_taken + 1, "runtime_error", 0.0, True, str(exc))
        success = False
    finally:
        log_end(success, steps_taken, score, rewards)
    return {
        "task_id": task_id,
        "task_name": task_name,
        "score": score,
        "success": success,
        "steps": steps_taken,
        "rewards": rewards,
    }


def main() -> None:
    tasks = http_json("GET", "/tasks")["tasks"]
    results = [run_task(task) for task in tasks]
    average_score = sum(result["score"] for result in results) / len(results)
    with open("inference_results.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "benchmark": BENCHMARK,
                "env_url": ENV_URL,
                "model_name": MODEL_NAME,
                "average_score": round(average_score, 4),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "results": results,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
