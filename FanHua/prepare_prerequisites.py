import copy
import hashlib
import json
from pathlib import Path

from fanhua.artifacts import ROOT, write_json
from fanhua.envs import TASKS
from train import check_config


def prepare():
    base = json.loads((ROOT / "configs/formal_candidate.json").read_text(encoding="utf-8"))
    base.update(policy_focus_timber=False, behavior_weight=0.0, teacher_paths={},
                stage_evaluation_episodes=20, stage_evaluation_steps=300)
    destination = ROOT / "configs/prerequisites_v2"
    gate_path = ROOT / "reports/前置工作验收.json"
    if gate_path.exists():
        raise FileExistsError("Prerequisite manifest already exists; preserve recorded results before regenerating")
    records = []
    groups = {
        "core_full": {},
        "core_without_calibration": {"calibration": False},
        "core_without_history": {"history_probability": 0.0},
        "latent_sac": {"history_probability": 0.0, "calibration": False, "cql_weight": 0.0},
    }
    repair_config = json.loads((ROOT / "experiments/20261006_171810_268739_repair_shared_policy/config.json")
                               .read_text(encoding="utf-8"))
    guidance = {"teacher_paths": repair_config["teacher_paths"], "behavior_weight": 5.0}
    groups.update({
        "guided_focused": dict(guidance, policy_focus_timber=True),
        "focused_without_guidance": {"policy_focus_timber": True},
        "guided_without_focus": guidance,
        "guided_focused_without_calibration": dict(guidance, policy_focus_timber=True, calibration=False),
        "guided_focused_without_history": dict(guidance, policy_focus_timber=True, history_probability=0.0),
    })
    for seed in (0, 1, 2):
        for task in TASKS:
            values = copy.deepcopy(base)
            values.update(seed=seed, history_probability=0.0,
                          stages=[{"task": task, "steps": 40000}],
                          experiment_role="single_game_baseline", baseline_task=task,
                          initialization="random_without_external_data")
            check_config(values)
            path = destination / f"baseline_{task}_seed_{seed}.json"
            write_json(path, values)
            records.append({"role": "single_game_baseline", "task": task, "seed": seed,
                            "config": str(path.relative_to(ROOT)), "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "status": "pending", "run": None, "evaluation": None})
        for group, overrides in groups.items():
            values = copy.deepcopy(base)
            values.update(overrides, seed=seed, experiment_role="formal_continual_experiment",
                          experiment_group=group, initialization="random_without_external_data",
                          prerequisites_manifest=str(gate_path))
            check_config(values)
            path = destination / f"{group}_seed_{seed}.json"
            write_json(path, values)
            records.append({"role": "formal_continual_experiment", "group": group, "seed": seed,
                            "config": str(path.relative_to(ROOT)), "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "status": "held", "run": None})
    teacher_provenance = {}
    for task, path in guidance["teacher_paths"].items():
        teacher_provenance[task] = {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                                    "use": "training supervision only; never deployment action selection"}
    manifest = {
        "format_version": 1, "formal_training_ready": False,
        "formal_experiments_authorized": False, "client_seed_supported": False,
        "reference_commit": "35ba6fc167d9f3257a1c12d943600b096e79f6b9",
        "baseline_steps_per_task": 40000, "formal_steps_per_task": 40000,
        "baseline_final_evaluation_episodes": 20, "baseline_final_evaluation_steps": 3000,
        "short_stage_evaluation_episodes": 20, "short_stage_evaluation_steps": 300,
        "primary_seed": 0, "planned_replication_seeds": [1, 2],
        "design_review": "pending_baseline_results_and_budget_review",
        "teacher_provenance": teacher_provenance, "experiments": records,
        "notes": [
            "Single-game baselines start independently from random initialization with no teachers, offline data, or shared checkpoints",
            "Formal groups start from scratch; the repaired shared checkpoint is an exploratory reference, not their initializer",
            "The Timberman teacher itself comes from prior shared smoke training; guidance is external prior knowledge and is not pure RL",
            "Guidance and input focusing are engineering additions, not original Cal-LAICA components",
            "No-calibration changes only calibration; no-history changes only history within each matched family",
            "Final evaluations are separate no-update runs on each saved baseline checkpoint",
            "Baseline failure is retained as an outcome, not hidden by substituting a teacher or changing its budget",
            "Seed 0 baselines run first; remaining replication configs are prepared, not claimed completed",
        ],
    }
    write_json(gate_path, manifest)
    print(f"[prerequisites] {len(records)} configs prepared; formal training remains gated. Manifest: {gate_path}", flush=True)
    return manifest


if __name__ == "__main__":
    prepare()
