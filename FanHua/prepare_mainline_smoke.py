import copy
import hashlib
import json

import numpy as np
import torch

from fanhua.agent import Agent
from fanhua.artifacts import ROOT, inside_root, new_run, write_json
from fanhua.diagnostics import check_focused_timber_policy
from fanhua.replay import Replay
from train import check_config


def main():
    config = json.loads((ROOT / "configs/mainline_shared_model.json").read_text(encoding="utf-8"))
    source = inside_root(ROOT / config["source_checkpoint"])
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = json.loads((ROOT / "reports/mainline_starting_evaluations.json").read_text(encoding="utf-8"))
    if source_hash != manifest["checkpoint_sha256"]:
        raise RuntimeError("Starting checkpoint changed after independent evaluations")
    evaluation_reports = []
    for task in ("flappybird", "timberman"):
        result = json.loads((ROOT / manifest[task]["run"] / "evaluation.json").read_text(encoding="utf-8"))
        if (result["status"] != "completed" or result["task"] != task
                or result["checkpoint_sha256"] != source_hash or len(result["episodes"]) != 20
                or result["evaluation_max_steps"] != 3000
                or result["parameters_before"] != result["parameters_after"]
                or result["updates_before"] != result["updates_after"]):
            raise RuntimeError(f"Starting evaluation verification failed: {task}")
        evaluation_reports.append(result)
    if evaluation_reports[0]["parameters_after"] != evaluation_reports[1]["parameters_after"]:
        raise RuntimeError("Evaluations used different model parameters")
    replay = Replay(config["capacity"], config["history_probability"], config["seed"])
    generator = np.random.default_rng(config["seed"])
    agent = Agent(config)
    progress = agent.load(source, replay, generator)
    if progress["stage"] != 0 or progress["stage_steps"] != 0 or progress["global_steps"] != 0:
        raise RuntimeError("Smoke initialization requires the verified pre-online checkpoint")
    digest_before = agent.parameter_digest()
    diagnostic = check_focused_timber_policy(agent, replay)
    if not diagnostic["passed"]:
        raise RuntimeError("Starting checkpoint failed Timberman diagnostic")
    smoke_config = copy.deepcopy(config)
    for stage in smoke_config["stages"]:
        stage["steps"] = 768
    smoke_config["max_episode_steps"] = 256
    smoke_config["experiment_role"] = "shared_model_smoke"
    check_config(smoke_config)
    agent.config = smoke_config
    directory = new_run("shared_mainline_smoke")
    initialized = directory / "checkpoints/initialized.pt"
    agent.save(initialized, replay, progress, generator)
    restored_agent = Agent(smoke_config)
    restored_replay = Replay(smoke_config["capacity"])
    restored_generator = np.random.default_rng(0)
    restored_progress = restored_agent.load(initialized, restored_replay, restored_generator)
    if restored_agent.parameter_digest() != digest_before or restored_progress != progress:
        raise RuntimeError("Smoke checkpoint preparation changed model parameters or progress")
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError("Original checkpoint changed during preparation")
    write_json(directory / "config.json", smoke_config)
    report = {
        "status": "prepared_not_started",
        "run": str(directory.relative_to(ROOT)),
        "config": str((directory / "config.json").relative_to(ROOT)),
        "initialized_checkpoint": str(initialized.relative_to(ROOT)),
        "source_checkpoint": config["source_checkpoint"],
        "source_checkpoint_sha256": source_hash,
        "source_parameters": digest_before,
        "parameter_updates_performed": 0,
        "source_updates": agent.updates,
        "stage_steps": 768,
        "max_training_episode_steps": 256,
        "total_steps": 3072,
        "stage_evaluation_episodes": 5,
        "stage_evaluation_steps": 300,
        "training_settings_change": "Stage budgets and training episode cap shortened; model, optimizers, replay and RNG retained",
        "timberman_diagnostic": diagnostic,
        "long_training_started": False,
    }
    write_json(directory / "preflight.json", report)
    write_json(ROOT / "reports/mainline_smoke_preparation.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
