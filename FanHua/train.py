import argparse
import hashlib
import json
import math
import signal
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from fanhua.agent import Agent
from fanhua.artifacts import ROOT, CsvLog, append_event, inside_root, new_run, write_json
from fanhua.envs import TASKS, WemiEnv
from fanhua.diagnostics import check_focused_timber_policy
from fanhua.evaluation import evaluate_environment
from fanhua.replay import Replay
from fanhua.supervision import Teachers
from fanhua.prerequisites import check_prerequisites


def check_config(config):
    for field in ("hidden_dim", "embedding_dim", "batch_size", "capacity", "warmup_steps",
                  "representation_steps", "cql_samples", "checkpoint_interval", "max_episode_steps"):
        if type(config[field]) is not int or config[field] <= 0:
            raise ValueError(f"{field} must be a positive integer")
    if config["capacity"] < config["batch_size"] or config["warmup_steps"] < config["batch_size"]:
        raise ValueError("capacity and warmup_steps must cover batch_size")
    for field in ("gamma", "tau", "history_probability"):
        if type(config[field]) not in (int, float) or not 0 <= config[field] <= 1:
            raise ValueError(f"Invalid {field}")
    for field in ("learning_rate", "representation_lr", "connect_timeout", "response_timeout"):
        if type(config[field]) not in (int, float) or not math.isfinite(config[field]) or config[field] <= 0:
            raise ValueError(f"Invalid {field}")
    if type(config["calibration"]) is not bool or type(config["seed"]) is not int:
        raise ValueError("calibration must be boolean and seed must be integer")
    if type(config.get("policy_focus_timber", False)) is not bool:
        raise ValueError("policy_focus_timber must be boolean")
    if type(config["cql_weight"]) not in (int, float) or not math.isfinite(config["cql_weight"]) or config["cql_weight"] < 0:
        raise ValueError("cql_weight must be finite and nonnegative")
    if type(config.get("behavior_weight", 0.0)) not in (int, float) or not math.isfinite(config.get("behavior_weight", 0.0)) or config.get("behavior_weight", 0.0) < 0:
        raise ValueError("behavior_weight must be finite and nonnegative")
    if type(config.get("stage_evaluation_episodes", 0)) is not int or config.get("stage_evaluation_episodes", 0) < 0:
        raise ValueError("stage_evaluation_episodes must be a nonnegative integer")
    if type(config.get("stage_evaluation_steps", 300)) is not int or config.get("stage_evaluation_steps", 300) <= 0:
        raise ValueError("stage_evaluation_steps must be a positive integer")
    if not config["stages"]:
        raise ValueError("At least one stage is required")
    for stage in config["stages"]:
        if stage["task"] not in TASKS or type(stage["steps"]) is not int or stage["steps"] <= 0:
            raise ValueError(f"Invalid stage: {stage}")


def train(config, directory, resume=None):
    check_config(config)
    check_prerequisites(config, resume)
    torch.set_num_threads(1)
    generator = np.random.default_rng(config["seed"])
    agent = Agent(config)
    replay = Replay(config["capacity"], config["history_probability"], config["seed"])
    progress = {"stage": 0, "stage_steps": 0, "global_steps": 0, "episodes": 0, "last_checkpoint_step": 0,
                "status": "running", "client_seed_supported": False}
    if resume:
        progress = agent.load(resume, replay, generator)
        append_event(directory / "events.jsonl", "resume", checkpoint=str(resume), previous_status=progress["status"],
                     previous_error=progress.pop("error", None), global_steps=progress["global_steps"])
        progress["status"] = "running"
    else:
        append_event(directory / "events.jsonl", "start", config=config)
    training_rng = torch.get_rng_state()
    teachers = Teachers(config["teacher_paths"]) if config.get("teacher_paths") else None
    torch.set_rng_state(training_rng)
    write_json(directory / "config.json", config)
    progress["phase"] = "online_train"
    write_json(directory / "status.json", progress)
    episodes = CsvLog(directory / "episodes.csv", ["stage", "task", "episode", "global_steps", "steps",
                       "raw_return", "learning_return", "score", "terminated", "truncated", "complete", "seconds"])
    updates = CsvLog(directory / "updates.csv", ["stage", "task", "global_steps", "updates", "critic_loss", "td_loss",
                     "cql_loss", "policy_loss", "alpha", "q_mean", "bound_fraction", "history_fraction",
                     "critic_gradient_norm", "actor_gradient_norm"])
    behavior_log = CsvLog(directory / "behavior_updates.csv", ["updates", "behavior_loss"])
    environment = None
    pending = []
    episode_steps = 0
    episode_return = 0.0
    raw_return = 0.0
    state = None
    start = time.monotonic()
    task = None
    info = {"score": None}
    next_progress_report = ((progress["global_steps"] // 4000) + 1) * 4000
    try:
        if config.get("policy_focus_timber") and replay.history["timberman"].rows:
            diagnostic = check_focused_timber_policy(agent, replay)
            write_json(directory / "initial_policy_diagnostic.json", diagnostic)
            if not diagnostic["passed"]:
                raise RuntimeError("Initial policy fails Timberman regression diagnostic")
        while progress["stage"] < len(config["stages"]):
            stage = config["stages"][progress["stage"]]
            task = stage["task"]
            append_event(directory / "events.jsonl", "stage_start", task=task, stage=progress["stage"],
                         global_steps=progress["global_steps"])
            if progress["stage_steps"] == 0:
                replay.begin_stage()
            print(f"[train] stage {progress['stage']} {task}, target={stage['steps']} steps", flush=True)
            environment = WemiEnv(task, directory / "protocol.jsonl",
                                  timeout=config["response_timeout"], connect_timeout=config["connect_timeout"],
                                  max_episode_steps=config["max_episode_steps"])
            state = environment.reset()
            episode_steps = 0
            episode_return = 0.0
            raw_return = 0.0
            pending = []
            start = time.monotonic()
            while progress["stage_steps"] < stage["steps"]:
                if (directory / "STOP").exists():
                    (directory / "STOP").unlink()
                    raise KeyboardInterrupt()
                random_action = task not in agent.ready_tasks
                action, embedding = agent.select(state, task, generator, random_action=random_action)
                next_state, reward, terminated, truncated, info = environment.step(action)
                pending.append({"task": task, "stage": progress["stage"], "state": state.tolist(),
                                "next_state": next_state.tolist(), "action": action, "embedding": embedding,
                                "reward": reward, "terminated": terminated, "truncated": truncated})
                state = next_state
                episode_steps += 1
                episode_return += reward
                raw_return += info["raw_reward"]
                progress["stage_steps"] += 1
                progress["global_steps"] += 1
                if progress["global_steps"] >= next_progress_report:
                    print(f"[progress] 已完成 {progress['global_steps']} 步，当前阶段 {task}，"
                          f"更新次数 {agent.updates}", flush=True)
                    next_progress_report += 4000
                boundary = progress["stage_steps"] == stage["steps"]
                if boundary and not terminated:
                    truncated = True
                    pending[-1]["truncated"] = True
                if terminated or truncated or boundary:
                    bootstrap = 0.0 if terminated else agent.value(state, task)
                    replay.add_episode(pending, config["gamma"], bootstrap, complete=terminated)
                    episodes.write({"stage": progress["stage"], "task": task, "episode": progress["episodes"],
                                    "global_steps": progress["global_steps"], "steps": episode_steps,
                                    "raw_return": raw_return, "learning_return": episode_return, "score": info["score"],
                                    "terminated": terminated, "truncated": truncated, "complete": terminated,
                                    "seconds": time.monotonic() - start})
                    print(f"[episode] {task} steps={episode_steps} return={episode_return:.3f} score={info['score']}", flush=True)
                    progress["episodes"] += 1
                    pending = []
                    action_coverage = {row["action"] for row in replay.online.rows if row["task"] == task}
                    if task not in agent.ready_tasks and len(replay.online.rows) >= config["warmup_steps"] and action_coverage == {0, 1}:
                        loss = agent.fit_representation(replay, task, config["representation_steps"])
                        print(f"[representation] {task} warmup passed, loss={loss:.6f}", flush=True)
                if task in agent.ready_tasks and len(replay.online.rows) >= config["batch_size"]:
                    batch = replay.sample(config["batch_size"])
                    if teachers:
                        batch["teacher_action"] = teachers.actions(batch["state"], batch["task_id"])
                    stats = agent.update(batch)
                    behavior_log.write({"updates": stats["updates"], "behavior_loss": stats.pop("behavior_loss")})
                    updates.write({"stage": progress["stage"], "task": task,
                                   "global_steps": progress["global_steps"], **stats})
                if progress["global_steps"] - progress.get("last_checkpoint_step", 0) >= config["checkpoint_interval"] and not pending:
                    progress["last_checkpoint_step"] = progress["global_steps"]
                    agent.save(directory / "checkpoints" / "latest.pt", replay, progress, generator)
                    write_json(directory / "status.json", progress)
                    if config.get("policy_focus_timber") and replay.history["timberman"].rows:
                        diagnostic = check_focused_timber_policy(agent, replay)
                        append_event(directory / "policy_diagnostics.jsonl", "checkpoint_policy_check",
                                     global_steps=progress["global_steps"], **diagnostic)
                        if not diagnostic["passed"]:
                            raise RuntimeError("Timberman policy regression detected; checkpoint retained for repair")
                if (terminated or truncated) and not boundary:
                    state = environment.reset()
                    episode_steps = 0
                    episode_return = 0.0
                    raw_return = 0.0
                    start = time.monotonic()
            if task not in agent.ready_tasks:
                raise RuntimeError(f"Stage budget too small to complete {task} representation warmup")
            evaluation_count = config.get("stage_evaluation_episodes", 0)
            if evaluation_count:
                progress["status"] = "evaluating_stage"
                agent.save(directory / "checkpoints" / "latest.pt", replay, progress, generator)
                write_json(directory / "status.json", progress)
                result = evaluate_environment(agent, environment, task, evaluation_count,
                                              config.get("stage_evaluation_steps", 300))
                result["stage"] = progress["stage"]
                result["checkpoint"] = str(directory / "checkpoints" / "latest.pt")
                result["checkpoint_sha256"] = hashlib.sha256((directory / "checkpoints" / "latest.pt").read_bytes()).hexdigest()
                write_json(directory / f"stage_{progress['stage'] + 1:02d}_{task}_evaluation.json", result)
                progress["status"] = "running"
            environment.close()
            environment = None
            progress["stage"] += 1
            progress["stage_steps"] = 0
            append_event(directory / "events.jsonl", "stage_completed", task=task, stage=progress["stage"] - 1,
                         global_steps=progress["global_steps"], updates=agent.updates)
            agent.save(directory / "checkpoints" / f"stage_{progress['stage']:02d}.pt", replay, progress, generator)
            agent.save(directory / "checkpoints" / "latest.pt", replay, progress, generator)
            write_json(directory / "status.json", progress)
        progress["status"] = "completed_training_not_yet_evaluated"
    except KeyboardInterrupt:
        progress["status"] = "interrupted"
        append_event(directory / "events.jsonl", "interrupted", global_steps=progress["global_steps"])
        print("[train] interrupted; saving checkpoint", flush=True)
    except BaseException as error:
        progress["status"] = "failed"
        progress["error"] = f"{type(error).__name__}: {error}"
        append_event(directory / "events.jsonl", "failed", error=progress["error"], global_steps=progress["global_steps"])
        traceback.print_exc()
        raise
    finally:
        if pending:
            replay.add_episode(pending, config["gamma"], complete=False)
            episodes.write({"stage": progress["stage"], "task": task, "episode": progress["episodes"],
                            "global_steps": progress["global_steps"], "steps": episode_steps,
                            "raw_return": raw_return, "learning_return": episode_return,
                            "score": info["score"], "terminated": False, "truncated": True,
                            "complete": False, "seconds": time.monotonic() - start})
        agent.save(directory / "checkpoints" / "latest.pt", replay, progress, generator)
        write_json(directory / "status.json", progress)
        episodes.close()
        updates.close()
        behavior_log.close()
        if environment is not None:
            environment.close()
        print(f"[train] {progress['status']} Output: {directory}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="WemiEnv two-task Cal-LAICA-style PyTorch extension")
    parser.add_argument("--config", default=str(ROOT / "configs" / "smoke.json"))
    parser.add_argument("--resume")
    parser.add_argument("--new-run", action="store_true", help="Resume weights/progress into a fresh experiment directory")
    parser.add_argument("--label", default="cal_laica_smoke")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    if args.resume:
        checkpoint = inside_root(args.resume)
        source_directory = inside_root(checkpoint.parent.parent)
        if args.new_run or ((source_directory / "pretrain.json").exists() and not (source_directory / "episodes.csv").exists()):
            directory = new_run(args.label)
        else:
            directory = source_directory
    else:
        check_config(config)
        directory = new_run(args.label)
    print(f"[train] Output: {directory}", flush=True)
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt()))
    train(config, directory, args.resume)


if __name__ == "__main__":
    main()
