import argparse
import traceback

import numpy as np

from fanhua.artifacts import new_run, write_json
from fanhua.envs import TASKS, WemiEnv


def main():
    parser = argparse.ArgumentParser(description="Check original WemiEnv protocol; no model training")
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--connect-timeout", type=float, default=600)
    parser.add_argument("--port", type=int, default=7891)
    args = parser.parse_args()
    if args.episodes <= 0 or args.max_steps <= 0:
        parser.error("episodes and max-steps must be positive")
    directory = new_run(f"probe_{args.task}")
    print(f"[probe] Output: {directory}", flush=True)
    environment = None
    report = {"mode": "random_protocol_probe_no_training", "task": args.task,
              "status": "running", "episodes": [], "client_seed_supported": False}
    write_json(directory / "result.json", report)
    try:
        environment = WemiEnv(args.task, directory / "protocol.jsonl", port=args.port,
                              connect_timeout=args.connect_timeout, max_episode_steps=args.max_steps)
        generator = np.random.default_rng(0)
        for episode in range(args.episodes):
            state = environment.reset()
            total = 0.0
            for step in range(args.max_steps):
                state, reward, terminated, truncated, info = environment.step(int(generator.integers(2)))
                total += reward
                if terminated or truncated:
                    break
            row = {"episode": episode + 1, "steps": step + 1, "return": total,
                   "terminated": terminated, "truncated": truncated,
                   "score": info["score"], "encoded_dim": len(state)}
            report["episodes"].append(row)
            print(f"[probe] {row}", flush=True)
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
        raise
    finally:
        if environment is not None:
            environment.close()
        write_json(directory / "result.json", report)


if __name__ == "__main__":
    main()
