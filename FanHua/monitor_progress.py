import argparse
import csv
import json
import time
from pathlib import Path


def latest_step(path):
    try:
        with (path / "updates.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        return int(rows[-1]["global_steps"]) if rows else 0
    except (FileNotFoundError, IndexError, KeyError, ValueError):
        return 0


def main():
    parser = argparse.ArgumentParser(description="Print one message per 4000 observed training steps")
    parser.add_argument("run")
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()
    directory = Path(args.run)
    reported = set()
    while True:
        step = latest_step(directory)
        for milestone in range(4000, 40001, 4000):
            if step >= milestone and milestone not in reported:
                print(f"[progress] 已完成 {milestone} 步。当前日志步数 {step}。", flush=True)
                reported.add(milestone)
        status = {}
        try:
            status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            pass
        if status.get("status") in {"completed_training_not_yet_evaluated", "failed", "interrupted"}:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
