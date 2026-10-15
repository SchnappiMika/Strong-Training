import hashlib
import json

import numpy as np

from fanhua.agent import Agent
from fanhua.artifacts import ROOT, new_run, write_json
from fanhua.replay import Replay
from train import check_config


def main():
    config = json.loads((ROOT / "configs/mainline_shared_model.json").read_text(encoding="utf-8"))
    smoke_run = ROOT / "experiments/20261008_113537_446916_shared_mainline_smoke"
    smoke_checkpoint = smoke_run / "checkpoints/latest.pt"
    smoke_config = json.loads((smoke_run / "config.json").read_text(encoding="utf-8"))
    check_config(config)
    replay = Replay(config["capacity"], config["history_probability"], config["seed"])
    generator = np.random.default_rng(config["seed"])
    smoke_agent = Agent(smoke_config)
    progress = smoke_agent.load(smoke_checkpoint, replay, generator)
    source_hash = hashlib.sha256(smoke_checkpoint.read_bytes()).hexdigest()
    if progress["status"] != "completed_training_not_yet_evaluated" or progress["global_steps"] != 3072:
        raise RuntimeError("Smoke checkpoint is not complete")
    if set(smoke_agent.ready_tasks) != {"flappybird", "timberman"}:
        raise RuntimeError("Smoke checkpoint does not contain both ready action representations")
    agent = Agent(config)
    agent.policy.load_state_dict(smoke_agent.policy.state_dict())
    agent.critic.load_state_dict(smoke_agent.critic.state_dict())
    agent.target.load_state_dict(smoke_agent.target.state_dict())
    agent.representation.load_state_dict(smoke_agent.representation.state_dict())
    agent.log_alpha.data.copy_(smoke_agent.log_alpha.data)
    agent.policy_optimizer.load_state_dict(smoke_agent.policy_optimizer.state_dict())
    agent.critic_optimizer.load_state_dict(smoke_agent.critic_optimizer.state_dict())
    agent.representation_optimizer.load_state_dict(smoke_agent.representation_optimizer.state_dict())
    agent.alpha_optimizer.load_state_dict(smoke_agent.alpha_optimizer.state_dict())
    agent.ready_tasks = list(smoke_agent.ready_tasks)
    agent.updates = smoke_agent.updates
    directory = new_run("mainline_shared_model")
    reset_progress = {"stage": 0, "stage_steps": 0, "global_steps": 0, "episodes": 0,
                      "last_checkpoint_step": 0, "status": "prepared_not_started",
                      "client_seed_supported": False, "phase": "online_train",
                      "initialization": "continuation_from_smoke_final_checkpoint",
                      "smoke_source_global_steps": 3072, "smoke_source_updates": smoke_agent.updates}
    agent.save(directory / "checkpoints/initialized.pt", replay, reset_progress, generator)
    write_json(directory / "config.json", config)
    report = {
        "status": "prepared_not_started",
        "run": str(directory.relative_to(ROOT)),
        "config": "configs/mainline_shared_model.json",
        "source_smoke_checkpoint": str(smoke_checkpoint.relative_to(ROOT)),
        "source_smoke_checkpoint_sha256": source_hash,
        "source_smoke_global_steps": 3072,
        "source_smoke_updates": smoke_agent.updates,
        "carried_replay_tasks": {task: len(pool.rows) for task, pool in replay.history.items()},
        "carried_ready_tasks": agent.ready_tasks,
        "new_training_steps": 0,
        "new_training_updates": 0,
        "planned_steps": 80000,
        "stage_evaluation_episodes": 5,
        "stage_evaluation_steps": 300,
        "reset_reason": "Begin the long mainline counter after smoke validation while retaining smoke-learned weights and replay",
    }
    write_json(directory / "preflight.json", report)
    write_json(ROOT / "reports/mainline_long_preflight.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
