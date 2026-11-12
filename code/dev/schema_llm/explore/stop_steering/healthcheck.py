"""Health check for the unattended STOP-steering regeneration.

Progress counts alone do not catch failures: a pilot can die and leave its count frozen, a watcher
can exit, a job can OOM and log a traceback while the others carry on. This checks the things that
actually go wrong and prints a single verdict line, so a 15-minute polling loop surfaces problems
instead of requiring them to be noticed in a number that stopped moving.
"""

from __future__ import annotations

import datetime
import re
import subprocess
from pathlib import Path

LOGS = Path("/tmp/fullcap-regen")
JOBS = {"dose-specificity": 221, "layer08": 119, "layer24": 119, "sampling": 255}
WAVE1 = sum(JOBS.values())
OUT = Path("results/schema_llm/analysis/stop_steering")
ERROR_PATTERNS = ("Traceback", "CUDA out of memory", "OutOfMemoryError", "Error:")


def count(pattern: str) -> int:
    try:
        return int(subprocess.run(["pgrep", "-fc", pattern], capture_output=True,
                                  text=True).stdout.strip() or 0)
    except (ValueError, OSError):
        return 0


def gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True).stdout.strip().splitlines()[0]
        used, total = (int(x) for x in out.split(","))
        return used, total
    except (ValueError, OSError, IndexError):
        return -1, -1


def job_state():
    rows, stalled = [], []
    for name, target in JOBS.items():
        log = LOGS / f"{name}.log"
        if not log.exists():
            rows.append((name, 0, target, "missing"))
            continue
        text = log.read_text(errors="ignore")
        n = text.count("->")
        errors = [p for p in ERROR_PATTERNS if p in text]
        done = (OUT / f"20260921-fullcap-{name}" / "pilot.parquet").exists()
        state = "DONE" if done else ("ERR:" + errors[-1][:18] if errors else "running")
        rows.append((name, n, target, state))
        if errors and not done:
            stalled.append(name)
    return rows, stalled


def main():
    rows, stalled = job_state()
    total = sum(r[1] for r in rows)
    pilots = count("[o]pt/venv.*stop_steering.pilot")
    orch = count("[f]ullcap_orchestrate")
    prim = count("[p]rimitive_mix")
    used, cap = gpu()
    now = datetime.datetime.now().strftime("%H:%M")

    problems = []
    if orch == 0:
        problems.append("orchestrator not running")
    if prim == 0:
        problems.append("primitive-mix watcher not running")
    finished = all(r[3] == "DONE" for r in rows)
    if pilots == 0 and not finished:
        problems.append("no pilots alive but wave 1 incomplete")
    problems += [f"{n}: error in log" for n in stalled]

    detail = " ".join(f"{n[:4]}:{c}/{t}" for n, c, t, _ in rows)
    verdict = "OK" if not problems else "PROBLEM: " + "; ".join(problems)
    print(f"[{now}] wave1 {total}/{WAVE1} ({total / WAVE1:.0%}) | {detail} | "
          f"pilots={pilots} orch={orch} prim={prim} | gpu {used}/{cap} | {verdict}")

    # surface the tail of any log that contains an error, once
    for name in stalled:
        text = (LOGS / f"{name}.log").read_text(errors="ignore")
        tail = [ln for ln in text.splitlines() if re.search(r"Error|Traceback|memory", ln)]
        for line in tail[-2:]:
            print(f"    {name}: {line[:150]}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
