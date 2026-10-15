import json

from fanhua.artifacts import ROOT, write_json
from fanhua.envs import TASKS


def main():
    manifest_path = ROOT / "reports/前置工作验收.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["planned_replication_seeds"] = [1]
    manifest["replication_policy"] = "two seeds only: primary seed 0 plus replication seed 1; seed 2 excluded by time budget"
    manifest["formal_training_ready"] = False
    manifest["formal_experiments_authorized"] = False
    for record in manifest["experiments"]:
        if record["seed"] != 2:
            continue
        if record["role"] == "single_game_baseline":
            record["status"] = "skipped_by_time_budget"
            record["skip_reason"] = "User approved two-seed plan; seed 2 is not run"
        else:
            record["status"] = "excluded_by_time_budget"
            record["skip_reason"] = "User approved two-seed formal plan; seed 2 is not run"
    manifest["design_review"] = "two_seed_plan_pending_review"
    manifest["design_review_report"] = "reports/前置工作设计核对_两种子.json"
    write_json(manifest_path, manifest)
    report = {
        "decision": "seed_2_excluded",
        "active_seeds": [0, 1],
        "excluded_seeds": [2],
        "reason": "User time constraint; two independent seeds remain for a minimum stability check",
        "limitations": [
            "Two seeds provide a comparison of initialization and sampling variation, not a strong multi-seed statistical estimate",
            "No claim of seed-independent mastery or confidence interval is made",
            "All seed 2 configurations and pending entries remain recorded but are not executed",
        ],
        "completed_baselines": [
            {"task": record["task"], "seed": record["seed"], "run": record["run"], "evaluation": record["evaluation"]}
            for record in manifest["experiments"]
            if record["role"] == "single_game_baseline" and record["status"] == "completed"
        ],
        "formal_groups": 9,
        "formal_seeds": 2,
        "formal_runs": 18,
        "formal_training_steps": 18 * len(TASKS) * manifest["formal_steps_per_task"],
        "formal_training_ready": False,
        "formal_experiments_authorized": False,
    }
    write_json(ROOT / "reports/前置工作设计核对_两种子.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
