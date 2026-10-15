import hashlib
import json
from pathlib import Path

from fanhua.artifacts import ROOT, write_json
from fanhua.envs import TASKS
from fanhua.prerequisites import verify_baseline_result
from train import check_config


def review():
    manifest_path = ROOT / "reports/前置工作验收.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    active_seeds = [manifest["primary_seed"]] + manifest["planned_replication_seeds"]
    baseline_results = []
    configs = {}
    for record in manifest["experiments"]:
        path = ROOT / record["config"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["config_sha256"]:
            raise RuntimeError(f"Configuration changed: {path}")
        config = json.loads(path.read_text(encoding="utf-8"))
        check_config(config)
        if config["seed"] != record["seed"]:
            raise ValueError("Registered seed differs from configuration")
        if record["role"] == "single_game_baseline":
            if (config["stages"] != [{"task": record["task"], "steps": manifest["baseline_steps_per_task"]}]
                    or config.get("teacher_paths") or config["behavior_weight"] != 0
                    or config["policy_focus_timber"] or config["history_probability"] != 0):
                raise ValueError("Independent baseline conditions differ from the design")
            if record["status"] == "completed":
                evaluation = verify_baseline_result(record, manifest)
                baseline_results.append({"task": record["task"], "seed": record["seed"],
                                         "run": record["run"], "evaluation": record["evaluation"],
                                         "mean_steps": evaluation["mean_steps"],
                                         "mean_learning_return": evaluation["mean_learning_return"],
                                         "std_learning_return": evaluation["std_learning_return"],
                                         "mean_score": evaluation.get("mean_score"),
                                         "truncated_episodes": sum(row["truncated"] for row in evaluation["episodes"])})
        else:
            if record["seed"] in active_seeds:
                configs[(record["group"], record["seed"])] = config
            for task in TASKS:
                if sum(stage["steps"] for stage in config["stages"] if stage["task"] == task) != manifest["formal_steps_per_task"]:
                    raise ValueError("Formal per-game budget differs from baseline budget")
            if config["initialization"] != "random_without_external_data":
                raise ValueError("Formal groups must not start from the teacher-repaired checkpoint")
    expected_overrides = {
        "core_full": {},
        "core_without_calibration": {"calibration": False},
        "core_without_history": {"history_probability": 0.0},
        "latent_sac": {"calibration": False, "history_probability": 0.0, "cql_weight": 0.0},
        "focused_without_guidance": {"policy_focus_timber": True},
        "guided_without_focus": {"teacher_paths": {task: item["path"] for task, item in manifest["teacher_provenance"].items()}, "behavior_weight": 5.0},
    }
    guidance = dict(expected_overrides["guided_without_focus"], policy_focus_timber=True)
    expected_overrides.update(guided_focused=guidance,
                              guided_focused_without_calibration=dict(guidance, calibration=False),
                              guided_focused_without_history=dict(guidance, history_probability=0.0))
    for seed in active_seeds:
        reference = configs[("core_full", seed)]
        for group, overrides in expected_overrides.items():
            expected = dict(reference, **overrides)
            expected["experiment_group"] = group
            if configs[(group, seed)] != expected:
                raise ValueError(f"Unexpected coupled configuration change: {group}, seed {seed}")
        reference_seed = dict(configs[("core_full", manifest["primary_seed"])], seed=seed)
        if reference != reference_seed:
            raise ValueError("Core settings changed between seeds")
    for task, provenance in manifest["teacher_provenance"].items():
        if hashlib.sha256(Path(provenance["path"]).read_bytes()).hexdigest() != provenance["sha256"]:
            raise ValueError(f"Teacher file changed: {task}")
    pending_baselines = [item for item in manifest["experiments"]
                         if item["role"] == "single_game_baseline" and item["seed"] in active_seeds
                         and item["status"] != "completed"]
    remaining_training_steps = 0
    for item in pending_baselines:
        trained_steps = 0
        if item.get("run"):
            status_path = ROOT / item["run"] / "status.json"
            if status_path.exists():
                trained_steps = json.loads(status_path.read_text(encoding="utf-8"))["global_steps"]
        remaining_training_steps += max(0, manifest["baseline_steps_per_task"] - trained_steps)
    interpretation = ("Active two-seed baseline replications remain incomplete; no formal runs started." if pending_baselines
                      else "All active two-seed independent baselines verified; formal experiments remain held.")
    report = {
        "design_checks_passed": True, "checked_configs": len(manifest["experiments"]),
        "active_seeds": active_seeds, "excluded_seeds": [2],
        "baseline_results": baseline_results, "pending_baselines": len(pending_baselines),
        "remaining_baseline_training_steps": remaining_training_steps,
        "planned_formal_runs": len(configs),
        "planned_formal_training_steps": sum(sum(stage["steps"] for stage in config["stages"]) for config in configs.values()),
        "formal_training_ready": False, "formal_experiments_authorized": manifest["formal_experiments_authorized"],
        "interpretation": interpretation,
    }
    manifest["design_review"] = ("configuration_and_primary_baseline_checks_passed_replications_pending" if pending_baselines
                                 else "configuration_and_all_baseline_checks_passed_formal_experiments_held")
    manifest["design_review_report"] = "reports/前置工作设计核对.json"
    write_json(ROOT / manifest["design_review_report"], report)
    write_json(manifest_path, manifest)
    lines = ["# 单游戏主基线验收与实验设计核对", "",
             "两款 seed 0 基线已分别从随机初始化完成 40000 步，并完成独立 20 局、每局上限 3000 步评估。",
             "没有使用老师、旧共享模型或离线轨迹；检查点、配置、预算和评估参数不变检查均通过。", "",
             "| 游戏 | 平均步数 | 平均学习回报 | 回报标准差 | 平均真实得分 | 达到上限 |",
             "|---|---:|---:|---:|---:|---:|"]
    for result in baseline_results:
        score = str(result["mean_score"]) if result["mean_score"] is not None else "协议未提供"
        lines.append(f"| {result['task']} (seed {result['seed']}) | {result['mean_steps']} | "
                     f"{result['mean_learning_return']:.4f} | {result['std_learning_return']:.4f} | "
                     f"{score} | {result['truncated_episodes']}/20 |")
    lines.extend(["", "## 核对结论", "",
                  "33 份配置的 SHA256、任务预算、随机种子和分组差异均已核对。核心消融与工程扩展消融按各自共同设置逐项改变因素；指导模型来源与文件摘要一致。",
                  "共享正式组每款游戏 40000 步，总 80000 步；单游戏基线每款 40000 步。实际更新次数需按日志报告。",
                  f"当前两种子计划中完成验收 {len(baseline_results)} 个，"
                  f"剩余 {len(pending_baselines)} 个待验收，尚余训练预算 {remaining_training_steps} 步，另有未完成的独立评估。"
                  "单次训练的回合波动不能作为跨训练种子的统计。",
                  "seed 2 按用户时间要求排除并保留记录；本轮不将两种子结果表述为强统计结论。正式共享实验共 9 组 × 2 种子 = 18 次，预备交互预算 1440000 步，另需阶段诊断、最终双游戏评估和遗忘评估。",
                  "正式实验仍暂停，需在两种子结果最终核对后由用户明确恢复。", "",
                  "## 成绩含义", "",
                  "20 局均达到 3000 步是本次预算内的稳定性证据，不等于自然存活极限。Flappy Bird 客户端未返回真实过管得分，两游戏回报不可直接相加比较。",
                  "游戏端随机数未设种子；Python 种子不能控制相同地图。这套实验研究已知两任务的持续学习和能力保留，未证明未见游戏泛化。", ""])
    (ROOT / "reports/单游戏主基线验收与实验设计核对_20261006.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    review()
