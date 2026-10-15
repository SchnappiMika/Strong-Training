import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from fanhua.agent import Agent
from fanhua.artifacts import ROOT, CsvLog, inside_root, new_run, write_json
from fanhua.envs import TASKS
from fanhua.replay import Replay, validate_transitions
from fanhua.supervision import Teachers, behavior_loss


def main():
    parser = argparse.ArgumentParser(description="Repair a shared policy using training-only teacher labels; preserve actual critic transitions")
    parser.add_argument("--source", required=True)
    parser.add_argument("--flappy-dataset", required=True)
    parser.add_argument("--timber-teacher", required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--focus-timber", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("steps must be positive")
    torch.set_num_threads(1)
    source_path = inside_root(args.source)
    saved = torch.load(source_path, map_location="cpu", weights_only=True)
    config = dict(saved["config"])
    generator = np.random.default_rng(config["seed"])
    agent = Agent(config)
    replay = Replay(config["capacity"], config["history_probability"], config["seed"])
    agent.load(source_path, replay, generator)
    if args.focus_timber:
        config["policy_focus_timber"] = True
        agent.policy.focus_timber = True
    teacher_paths = {"flappybird": str(ROOT.parent / "flappybird_rl/logs/ppo_20260925_174115_model.pt"),
                     "timberman": str(inside_root(args.timber_teacher))}
    training_rng = torch.get_rng_state()
    teachers = Teachers(teacher_paths)
    torch.set_rng_state(training_rng)
    dataset_path = inside_root(args.flappy_dataset)
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=True)
    if dataset["gamma"] != config["gamma"] or dataset.get("encoding_version") != "wemi_12_v1":
        raise ValueError("Repair dataset does not match checkpoint encoding/gamma")
    validate_transitions(dataset["rows"], config["embedding_dim"])
    pools = {}
    teacher_labels = {}
    for index, task in enumerate(TASKS):
        rows = dataset["rows"] if task == "flappybird" else replay.history[task].rows
        rows = [row for row in rows if row["task"] == task and
                (task != "timberman" or row["state"][0] != row["state"][-1])]
        if not rows:
            raise ValueError(f"No usable {task} policy repair states")
        pools[task] = torch.tensor([row["state"] for row in rows])
        teacher_labels[task] = teachers.actions(pools[task], torch.full((len(rows),), index, dtype=torch.long))
    directory = new_run("repair_shared_policy")
    config["teacher_paths"] = teacher_paths
    config["behavior_target"] = "training_only_teacher_actions_CE_plus_embedding_regression"
    agent.config = config
    write_json(directory / "config.json", config)
    report = {"status": "running", "source": str(source_path), "teacher_paths": teacher_paths,
              "teacher_sha256": {task: hashlib.sha256(Path(path).read_bytes()).hexdigest() for task, path in teacher_paths.items()},
              "training_states": {task: len(pool) for task, pool in pools.items()},
              "supervised_updates": 0, "sac_updates_unchanged": agent.updates,
              "critic_transitions_relabelled": False, "teacher_models_used_at_inference": False}
    log = CsvLog(directory / "repair_updates.csv", ["iteration", "loss", "gradient_norm"])
    original_critics = {name: value.detach().clone() for name, value in agent.critic.state_dict().items()}
    try:
        for iteration in range(args.steps):
            state_parts, action_parts, task_parts = [], [], []
            for index, task in enumerate(TASKS):
                chosen = generator.integers(len(pools[task]), size=config["batch_size"] // 2)
                state_parts.append(pools[task][chosen])
                action_parts.append(teacher_labels[task][chosen])
                task_parts.append(torch.full((len(chosen),), index, dtype=torch.long))
            loss = behavior_loss(agent.policy, agent.representation, torch.cat(state_parts),
                                 torch.cat(task_parts), torch.cat(action_parts))
            gradient = agent.optimize(loss, agent.policy_optimizer, agent.policy.parameters())
            log.write({"iteration": iteration + 1, "loss": float(loss.detach()), "gradient_norm": gradient})
            report["supervised_updates"] = iteration + 1
            if (iteration + 1) % 1000 == 0:
                print(f"[repair] steps={iteration + 1} loss={float(loss.detach()):.6f}", flush=True)
        report["teacher_agreement"] = {}
        with torch.inference_mode():
            for index, task in enumerate(TASKS):
                tasks = torch.full((len(pools[task]),), index, dtype=torch.long)
                embeddings, _ = agent.policy(pools[task], tasks, deterministic=True)
                actions = agent.representation.action(embeddings, tasks)
                report["teacher_agreement"][task] = float((actions == teacher_labels[task]).float().mean())
        if any(not torch.equal(original_critics[name], value) for name, value in agent.critic.state_dict().items()):
            raise RuntimeError("Policy repair unexpectedly changed critic weights")
        report["status"] = "completed_policy_repair_requires_real_evaluation"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        log.close()
        progress = {"stage": 0, "stage_steps": 0, "global_steps": 0, "episodes": 0,
                    "last_checkpoint_step": 0, "status": report["status"], "phase": "offline_pretrain",
                    "client_seed_supported": False}
        checkpoint_name = "latest.pt" if report["status"].startswith("completed") else "failed_diagnostic.pt"
        agent.save(directory / "checkpoints" / checkpoint_name, replay, progress, generator)
        write_json(directory / "repair.json", report)
        print(f"[repair] Output: {directory}", flush=True)


if __name__ == "__main__":
    main()
