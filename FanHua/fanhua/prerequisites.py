import hashlib
import json

from .artifacts import ROOT, inside_root
from .envs import TASKS


def verify_baseline_result(baseline, manifest):
    config_path = inside_root(ROOT / baseline["config"])
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != baseline["config_sha256"]:
        raise RuntimeError("Baseline configuration changed after experiment design was recorded")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    directory = inside_root(ROOT / baseline["run"])
    recorded_config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    task = baseline["task"]
    check_prerequisites(config)
    if (recorded_config != config or config["baseline_task"] != task
            or config["seed"] != baseline["seed"]
            or status["status"] != "completed_training_not_yet_evaluated"
            or status["global_steps"] != manifest["baseline_steps_per_task"]
            or status["stage"] != len(config["stages"])):
        raise RuntimeError(f"Baseline training budget, task, seed, or configuration is invalid: {task}")
    evaluation = json.loads(inside_root(ROOT / baseline["evaluation"]).read_text(encoding="utf-8"))
    checkpoint = inside_root(evaluation["checkpoint"])
    expected_checkpoint = directory / "checkpoints/latest.pt"
    if (checkpoint != expected_checkpoint or evaluation["status"] != "completed"
            or evaluation["task"] != task
            or len(evaluation["episodes"]) != manifest["baseline_final_evaluation_episodes"]
            or evaluation.get("evaluation_max_steps") != manifest["baseline_final_evaluation_steps"]
            or evaluation["parameters_before"] != evaluation["parameters_after"]
            or evaluation["updates_before"] != evaluation["updates_after"]
            or evaluation["checkpoint_sha256"] != hashlib.sha256(checkpoint.read_bytes()).hexdigest()):
        raise RuntimeError(f"Invalid independent baseline evaluation: {task}")
    check_prerequisites(config, checkpoint)
    import torch
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["config"] != config or saved["progress"] != status or saved["updates"] != evaluation["updates_before"]:
        raise RuntimeError(f"Baseline checkpoint does not match training and evaluation records: {task}")
    return evaluation


def check_prerequisites(config, resume=None):
    role = config.get("experiment_role")
    if role == "single_game_baseline":
        task = config.get("baseline_task")
        if task not in TASKS or {stage["task"] for stage in config["stages"]} != {task}:
            raise ValueError("A single-game baseline must train exactly its designated task")
        if config.get("teacher_paths") or config.get("behavior_weight", 0) != 0:
            raise ValueError("Independent baseline must not use external teacher supervision")
        if config.get("policy_focus_timber", False):
            raise ValueError("Core baseline must not enable the separately studied input focusing extension")
        if resume:
            import torch
            checkpoint = torch.load(inside_root(resume), map_location="cpu", weights_only=True)
            recorded_tasks = set(checkpoint["ready_tasks"])
            recorded_tasks.update(row["task"] for row in checkpoint["replay"]["online"]["rows"])
            for pool in checkpoint["replay"]["history"].values():
                recorded_tasks.update(row["task"] for row in pool["rows"])
            if recorded_tasks - {task}:
                raise ValueError("Baseline resume checkpoint contains another game's training experience")
        return
    if role != "formal_continual_experiment":
        return
    manifest = json.loads(inside_root(config["prerequisites_manifest"]).read_text(encoding="utf-8"))
    if not manifest.get("formal_training_ready") or not manifest.get("formal_experiments_authorized"):
        raise RuntimeError("Formal experiments remain held until prerequisites are verified and the user resumes them")
    active_seeds = {manifest["primary_seed"], *manifest.get("planned_replication_seeds", [])}
    if config["seed"] not in active_seeds:
        raise RuntimeError("This formal seed was excluded by the approved replication policy")
    for seed in active_seeds:
        for task in TASKS:
            completed = [item for item in manifest["experiments"]
                         if item["role"] == "single_game_baseline" and item["task"] == task
                         and item["seed"] == seed and item["status"] == "completed"]
            if len(completed) != 1:
                raise RuntimeError(f"Missing completed baseline: {task}, seed {seed}")
            verify_baseline_result(completed[0], manifest)
