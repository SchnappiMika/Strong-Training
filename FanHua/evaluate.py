import argparse
import hashlib
import json
import statistics
import traceback
from pathlib import Path

import numpy as np
import torch

from fanhua.agent import Agent
from fanhua.artifacts import CsvLog, inside_root, new_run, write_json
from fanhua.envs import TASKS, WemiEnv
from fanhua.replay import Replay


def main():
    parser = argparse.ArgumentParser(description="Independent WemiEnv evaluation with no gradient updates")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--connect-timeout", type=float, default=600)
    args = parser.parse_args()
    if args.episodes < 2 or args.max_steps <= 0:
        parser.error("Use at least two evaluation episodes and positive max-steps")
    checkpoint_path = inside_root(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    agent = Agent(config)
    replay = Replay(config["capacity"])
    generator = np.random.default_rng(args.seed)
    agent.load(checkpoint_path, replay, generator)
    if args.task not in agent.ready_tasks:
        raise ValueError(f"{args.task} has not passed action representation warmup in this checkpoint")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    agent.policy.eval()
    agent.critic.eval()
    directory = new_run(f"evaluate_{args.task}")
    print(f"[evaluate] Output: {directory}", flush=True)
    log = CsvLog(directory / "episodes.csv", ["episode", "task", "steps", "raw_return", "learning_return",
                                             "score", "terminated", "truncated"])
    report = {"task": args.task, "checkpoint": str(checkpoint_path),
              "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
              "status": "running", "client_seed_supported": False, "deterministic_policy": True,
              "requested_episodes": args.episodes, "evaluation_max_steps": args.max_steps,
              "episodes": [], "updates_before": agent.updates,
              "parameters_before": agent.parameter_digest(), "action_counts": {"0": 0, "1": 0}}
    environment = None
    write_json(directory / "evaluation.json", report)
    try:
        environment = WemiEnv(args.task, directory / "protocol.jsonl", connect_timeout=args.connect_timeout,
                              max_episode_steps=args.max_steps)
        with torch.inference_mode():
            for episode in range(args.episodes):
                state = environment.reset()
                learning_return = 0.0
                raw_return = 0.0
                for step in range(args.max_steps):
                    action, embedding = agent.select(state, args.task, deterministic=True)
                    report["action_counts"][str(action)] += 1
                    state, reward, terminated, truncated, info = environment.step(action)
                    learning_return += reward
                    raw_return += info["raw_reward"]
                    if terminated or truncated:
                        break
                row = {"episode": episode, "task": args.task, "steps": step + 1, "raw_return": raw_return,
                       "learning_return": learning_return, "score": info["score"],
                       "terminated": terminated, "truncated": truncated}
                report["episodes"].append(row)
                log.write(row)
                write_json(directory / "evaluation.json", report)
                outcome = "自然死亡" if terminated else "达到步数上限，无死亡"
                score_text = f"，得分 {info['score']:g}" if info["score"] is not None else ""
                print(f"[evaluate] 第 {episode + 1} 局结束：{step + 1} 步{score_text}，"
                      f"学习回报 {learning_return:.4f}，{outcome}", flush=True)
        report["mean_learning_return"] = statistics.mean(row["learning_return"] for row in report["episodes"])
        report["std_learning_return"] = statistics.stdev(row["learning_return"] for row in report["episodes"])
        report["mean_steps"] = statistics.mean(row["steps"] for row in report["episodes"])
        if args.task == "timberman":
            report["mean_score"] = statistics.mean(row["score"] for row in report["episodes"])
        report["updates_after"] = agent.updates
        report["parameters_after"] = agent.parameter_digest()
        if report["updates_before"] != report["updates_after"]:
            raise RuntimeError("Evaluation unexpectedly changed training update count")
        if report["parameters_before"] != report["parameters_after"]:
            raise RuntimeError("Evaluation unexpectedly changed model parameters")
        report["status"] = "completed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
        raise
    finally:
        log.close()
        if environment:
            environment.close()
        write_json(directory / "evaluation.json", report)


if __name__ == "__main__":
    main()
