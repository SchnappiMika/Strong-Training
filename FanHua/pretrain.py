import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fanhua.agent import Agent
from fanhua.artifacts import ROOT, CsvLog, inside_root, new_run, write_json
from fanhua.envs import TASKS
from fanhua.replay import Replay, validate_transitions
from train import check_config


def main():
    parser = argparse.ArgumentParser(description="CQL offline pretraining using collected WemiEnv transitions")
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs" / "smoke.json"))
    parser.add_argument("--updates", type=int, default=1000)
    args = parser.parse_args()
    if args.updates <= 0:
        parser.error("updates must be positive")
    config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    check_config(config)
    agent = Agent(config)
    replay = Replay(config["capacity"], config["history_probability"], config["seed"])
    replay.history_probability = 1.0
    generator = np.random.default_rng(config["seed"])
    torch.set_num_threads(1)
    datasets = []
    provenance = []
    for dataset_path in args.dataset:
        dataset_path = inside_root(dataset_path)
        dataset = torch.load(dataset_path, map_location="cpu", weights_only=True)
        if dataset["format_version"] != 1 or dataset["gamma"] != config["gamma"]:
            raise ValueError("Dataset version/gamma does not match config")
        if dataset.get("encoding_version") != "wemi_12_v1" or dataset.get("reward_version") != "flappy_scaled_timber_delta_v1":
            raise ValueError("Dataset observation/reward encoding does not match this training pipeline")
        validate_transitions(dataset["rows"], config["embedding_dim"])
        datasets.extend(dataset["rows"])
        provenance.append({"path": str(dataset_path), "behavior": dataset.get("behavior"),
                           "transitions": len(dataset["rows"]),
                           "valid_bounds": sum(row["valid_bound"] for row in dataset["rows"])})
    directory = new_run("offline_pretrain")
    write_json(directory / "config.json", config)
    report = {"status": "running", "datasets": provenance, "updates": 0}
    log = None
    try:
        for task in TASKS:
            rows = [row for row in datasets if row["task"] == task]
            if not rows:
                continue
            for row in rows:
                replay.online.append(row)
            if len(replay.online.rows) < config["batch_size"]:
                raise ValueError(f"Too few {task} offline samples")
            agent.fit_representation(replay, task, config["representation_steps"])
            replay.begin_stage()
        if not any(pool.rows for pool in replay.history.values()):
            raise ValueError("Offline pretraining requires non-empty data")
        replay.history_probability = config["history_probability"]
        for iteration in range(args.updates):
            config_calibration = agent.config["calibration"]
            agent.config["calibration"] = False
            try:
                stats = agent.update(replay.sample(config["batch_size"]))
            finally:
                agent.config["calibration"] = config_calibration
            if log is None:
                log = CsvLog(directory / "updates.csv", list(stats))
            log.write(stats)
            report["updates"] += 1
            if report["updates"] % 200 == 0:
                print(f"[pretrain] updates={report['updates']} critic_loss={stats['critic_loss']:.5f}", flush=True)
        report["status"] = "completed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if log:
            log.close()
        if agent.ready_tasks:
            progress = {"stage": 0, "stage_steps": 0, "global_steps": 0, "episodes": 0,
                        "last_checkpoint_step": 0, "status": report["status"], "client_seed_supported": False,
                        "phase": "offline_pretrain"}
            checkpoint_name = "latest.pt" if report["status"] == "completed" else "failed_diagnostic.pt"
            agent.save(directory / "checkpoints" / checkpoint_name, replay, progress, generator)
        write_json(directory / "pretrain.json", report)
        print(f"[pretrain] {report['status']} Output: {directory}", flush=True)


if __name__ == "__main__":
    main()
