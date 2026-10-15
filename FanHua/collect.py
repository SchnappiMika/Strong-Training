import argparse
import hashlib
import json
import signal

import numpy as np
import torch
from torch import nn

from fanhua.agent import Agent
from fanhua.artifacts import ROOT, inside_root, new_run, write_json
from fanhua.envs import TASKS, WemiEnv
from fanhua.replay import Replay


def main():
    parser = argparse.ArgumentParser(description="Collect WemiEnv behavior data for offline pretraining")
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--behavior", choices=("random", "legacy-ppo", "checkpoint"), default="random")
    parser.add_argument("--legacy-model")
    parser.add_argument("--checkpoint")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.episodes <= 0 or args.max_steps <= 0:
        parser.error("episodes/max-steps must be positive")
    if args.behavior == "legacy-ppo" and (args.task != "flappybird" or not args.legacy_model):
        parser.error("legacy-ppo requires flappybird and --legacy-model")
    if args.behavior == "checkpoint" and not args.checkpoint:
        parser.error("checkpoint behavior requires --checkpoint")
    config = json.loads((ROOT / "configs" / "smoke.json").read_text(encoding="utf-8"))
    agent = Agent(config)
    actor = None
    if args.behavior == "legacy-ppo":
        actor = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))
        saved = torch.load(args.legacy_model, map_location="cpu", weights_only=True)
        actor.load_state_dict({name.removeprefix("logits_net."): value for name, value in saved["actor"].items()})
        actor.eval()
    generator = np.random.default_rng(config["seed"])
    teacher = None
    source_hash = None
    if args.behavior == "checkpoint":
        source_path = inside_root(args.checkpoint)
        saved = torch.load(source_path, map_location="cpu", weights_only=True)
        teacher = Agent(saved["config"])
        teacher_replay = Replay(saved["config"]["capacity"])
        teacher.load(source_path, teacher_replay, generator)
        if args.task not in teacher.ready_tasks or saved["config"]["gamma"] != config["gamma"]:
            raise ValueError("Teacher task readiness or gamma does not match collector")
        teacher.policy.eval()
        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        teacher_digest = teacher.parameter_digest()
    replay = Replay(config["capacity"], seed=config["seed"])
    directory = new_run(f"collect_{args.task}_{args.behavior}")
    environment = None
    summary = {"status": "running", "behavior": args.behavior, "task": args.task,
               "source_model": args.checkpoint if teacher else args.legacy_model,
               "source_sha256": source_hash, "episodes": [], "client_seed_supported": False,
               "maximum_episodes": args.episodes, "maximum_steps_per_episode": args.max_steps}
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt()))
    write_json(directory / "dataset.json", summary)
    print(f"[collect] Output: {directory}", flush=True)
    try:
        environment = WemiEnv(args.task, directory / "protocol.jsonl", connect_timeout=config["connect_timeout"],
                              max_episode_steps=args.max_steps)
        for episode in range(args.episodes):
            state = environment.reset()
            rows = []
            total = 0.0
            for step in range(args.max_steps):
                action, embedding = agent.select(state, args.task, generator, random_action=True)
                if teacher is not None:
                    action, embedding = teacher.select(state, args.task, deterministic=True)
                if actor is not None:
                    with torch.inference_mode():
                        action = int(actor(torch.tensor(state[:3])).argmax().item())
                    embedding = agent.representation.embeddings(torch.tensor([0]))[0, action].detach().tolist()
                next_state, reward, terminated, truncated, info = environment.step(action)
                rows.append({"task": args.task, "stage": -1, "state": state.tolist(),
                             "next_state": next_state.tolist(), "action": action, "embedding": embedding,
                             "reward": reward, "terminated": terminated, "truncated": truncated})
                total += reward
                state = next_state
                if terminated or truncated:
                    break
            replay.add_episode(rows, config["gamma"], complete=terminated)
            summary["episodes"].append({"episode": episode, "steps": len(rows), "return": total,
                                        "terminated": terminated, "truncated": truncated, "score": info["score"]})
            temporary = directory / "dataset.tmp"
            torch.save({"format_version": 1, "rows": replay.online.rows,
                        "gamma": config["gamma"], "behavior": args.behavior,
                        "source_model": summary["source_model"], "source_sha256": source_hash,
                        "encoding_version": "wemi_12_v1", "reward_version": "flappy_scaled_timber_delta_v1"},
                       temporary)
            temporary.replace(directory / "dataset.pt")
            summary["transitions"] = len(replay.online.rows)
            write_json(directory / "dataset.json", summary)
            print(f"[collect] episode={episode} steps={len(rows)} return={total:.3f}", flush=True)
        summary["status"] = "completed"
        if teacher is not None:
            if teacher.parameter_digest() != teacher_digest:
                raise RuntimeError("Collection changed teacher model parameters")
            summary["teacher_parameters_unchanged"] = True
    except KeyboardInterrupt:
        summary["status"] = "interrupted_completed_episodes_preserved"
        print("[collect] interrupted; dataset contains previously completed episodes", flush=True)
    except BaseException as error:
        summary["status"] = "failed"
        summary["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if environment:
            environment.close()
        write_json(directory / "dataset.json", summary)


if __name__ == "__main__":
    main()
