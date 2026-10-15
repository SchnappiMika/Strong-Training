import argparse
import json

from fanhua.artifacts import ROOT, inside_root, write_json
from fanhua.prerequisites import verify_baseline_result


def record(config_path, run_path, evaluation_path=None):
    manifest_path = ROOT / "reports/前置工作验收.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_path = inside_root(config_path)
    matching = [item for item in manifest["experiments"]
                if item["role"] == "single_game_baseline" and inside_root(ROOT / item["config"]) == config_path]
    if len(matching) != 1:
        raise ValueError("Configuration must identify a registered independent baseline")
    baseline = matching[0]
    baseline["run"] = str(inside_root(run_path).relative_to(ROOT))
    if evaluation_path:
        baseline["evaluation"] = str(inside_root(evaluation_path).relative_to(ROOT))
        evaluation = verify_baseline_result(baseline, manifest)
        baseline["status"] = "completed"
        baseline["result"] = {key: evaluation[key] for key in
                              ("mean_steps", "mean_learning_return", "std_learning_return", "checkpoint_sha256")}
        if "mean_score" in evaluation:
            baseline["result"]["mean_score"] = evaluation["mean_score"]
        baseline["result"]["natural_terminations"] = sum(row["terminated"] for row in evaluation["episodes"])
        baseline["result"]["truncated_episodes"] = sum(row["truncated"] for row in evaluation["episodes"])
    else:
        if baseline["status"] == "completed":
            raise ValueError("Do not overwrite a completed baseline registration")
        status = json.loads((inside_root(run_path) / "status.json").read_text(encoding="utf-8"))
        baseline["status"] = "awaiting_independent_evaluation" if status["status"].startswith("completed") else status["status"]
    write_json(manifest_path, manifest)
    print(f"[baseline] {baseline['task']} seed {baseline['seed']}: {baseline['status']}; formal readiness unchanged", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Record baseline progress or verify a finished independent baseline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--evaluation")
    args = parser.parse_args()
    record(args.config, args.run, args.evaluation)


if __name__ == "__main__":
    main()
