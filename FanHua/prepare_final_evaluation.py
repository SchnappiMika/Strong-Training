import hashlib
import json

import numpy as np
import torch

from fanhua.agent import Agent
from fanhua.artifacts import ROOT, write_json
from fanhua.replay import Replay


def main():
    directory = ROOT / "experiments/20261008_122246_330413_mainline_shared_model"
    path = directory / "checkpoints/latest.pt"
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if checkpoint["config"] != config or checkpoint["progress"] != status:
        raise RuntimeError("Checkpoint differs from saved training configuration or status")
    if (status["status"] != "completed_training_not_yet_evaluated"
            or status["stage"] != 4 or status["global_steps"] != 80000):
        raise RuntimeError("Mainline training is incomplete")
    if set(checkpoint["ready_tasks"]) != {"flappybird", "timberman"}:
        raise RuntimeError("Both games must have ready action representations")
    for name in ("policy", "critic", "target", "representation"):
        if any(not torch.isfinite(value).all() for value in checkpoint[name].values()):
            raise RuntimeError(f"Non-finite model tensor: {name}")
    if not torch.isfinite(checkpoint["log_alpha"]).all():
        raise RuntimeError("Non-finite entropy temperature")
    agent = Agent(config)
    agent.load(path, Replay(config["capacity"]), np.random.default_rng(1000))
    if hashlib.sha256(path.read_bytes()).hexdigest() != file_hash:
        raise RuntimeError("Checkpoint changed during inspection")
    report = {
        "preflight_status": "passed",
        "checkpoint": str(path),
        "checkpoint_sha256": file_hash,
        "parameters_before": agent.parameter_digest(),
        "updates_before": agent.updates,
        "training_steps": status["global_steps"],
        "evaluation_policy": "deterministic_shared_actor_without_teachers",
        "requested_episodes_per_game": 20,
        "max_steps_per_episode": 3000,
        "evaluation_seed": 1000,
        "client_seed_supported": False,
        "parameter_updates_performed": 0,
        "flappybird": {"status": "not_started"},
        "timberman": {"status": "not_started"},
        "automatic_retraining": False,
    }
    write_json(ROOT / "reports/mainline_final_evaluation.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
