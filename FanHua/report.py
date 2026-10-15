import argparse
import csv
import json
import math
import statistics

from fanhua.artifacts import inside_root, write_json
from fanhua.envs import TASKS


def read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summary(rows, field):
    values = [float(row[field]) for row in rows if row.get(field) not in (None, "")]
    if not values:
        return None
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Non-finite values in {field}")
    return {"count": len(values), "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else None, "max": max(values)}


def main():
    parser = argparse.ArgumentParser(description="Generate Chinese summaries and plots from actual experiment logs")
    parser.add_argument("--run", required=True)
    parser.add_argument("--evaluation", action="append", default=[])
    args = parser.parse_args()
    directory = inside_root(args.run)
    if not directory.is_dir():
        raise ValueError("run must be an existing experiment directory")
    episodes = read_csv(directory / "episodes.csv")
    updates = read_csv(directory / "updates.csv")
    status = json.loads((directory / "status.json").read_text(encoding="utf-8")) if (directory / "status.json").exists() else {}
    if not status and (directory / "pretrain.json").exists():
        status = json.loads((directory / "pretrain.json").read_text(encoding="utf-8"))
        status["phase"] = "offline_pretrain"
    if not status and (directory / "repair.json").exists():
        status = json.loads((directory / "repair.json").read_text(encoding="utf-8"))
        status["phase"] = "supervised_policy_repair"
    report = {"status": status, "training_episode_rows": len(episodes), "update_rows": len(updates),
              "training_by_task": {}, "evaluations": []}
    lines = ["# 双游戏实验记录", "", f"训练状态：`{status.get('status', 'unknown')}`。",
             f"已记录交互步数：{status.get('global_steps', '未记录')}；更新日志行数：{len(updates)}。", "",
             "以下训练指标只统计客户端真正终止的完整回合，阶段末尾截断片段不计入均值。", "",
             "| 游戏 | 完整回合 | 平均训练回报 | 回报标准差 | 平均步数 | 最高实际分数 |",
             "|---|---:|---:|---:|---:|---:|"]
    if status.get("phase") == "supervised_policy_repair":
        lines[3] = (f"本阶段监督修复更新 {status['supervised_updates']} 次；"
                    f"SAC 更新计数保持 {status['sac_updates_unchanged']}，本阶段未新增游戏交互。")
    for task in TASKS:
        completed = [row for row in episodes if row["task"] == task and row["complete"] == "True"]
        details = {field: summary(completed, field) for field in ("learning_return", "steps", "score")}
        report["training_by_task"][task] = details
        returns, steps, score = details["learning_return"], details["steps"], details["score"]
        mean = f"{returns['mean']:.3f}" if returns else "—"
        deviation = f"{returns['std']:.3f}" if returns and returns["std"] is not None else "—"
        length = f"{steps['mean']:.2f}" if steps else "—"
        maximum = f"{score['max']:.0f}" if score else "协议未提供"
        lines.append(f"| {task} | {len(completed)} | {mean} | {deviation} | {length} | {maximum} |")
    lines.extend(["", "## 独立评估", ""])
    lines.append("训练状态来自训练器运行记录；后续独立评估结果单独列在这里，不改写原始训练检查点。")
    lines.append("")
    for evaluation_path in args.evaluation:
        evaluation_path = inside_root(evaluation_path)
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        report["evaluations"].append(evaluation)
        if evaluation.get("status") != "completed":
            lines.append(f"- `{evaluation_path.name}` 未完成，不能计为成功评估。")
            continue
        rows = evaluation["episodes"]
        truncated_count = sum(row["truncated"] for row in rows)
        lines.append(f"- {evaluation['task']}：{len(rows)} 局；平均回报 {evaluation['mean_learning_return']:.3f}，"
                     f"标准差 {evaluation['std_learning_return']:.3f}，平均步数 {evaluation['mean_steps']:.2f}，"
                     f"截断 {truncated_count}/{len(rows)}。")
        if "mean_score" in evaluation:
            lines.append(f"  平均实际得分：{evaluation['mean_score']:.2f}。")
            if truncated_count:
                lines.append("  达到步数上限的得分是该局最终可达到分数的下界，不等于自然结束时的极限成绩。")
        lines.append(f"  检查点 SHA256：`{evaluation['checkpoint_sha256']}`。")
        if "action_counts" in evaluation:
            lines.append(f"  动作计数：`{evaluation['action_counts']}`。")
    completed_evaluations = [item for item in report["evaluations"] if item.get("status") == "completed"]
    shared = {}
    for evaluation in completed_evaluations:
        shared.setdefault(evaluation["checkpoint_sha256"], set()).add(evaluation["task"])
    report["same_checkpoint_evaluated_on_both_tasks"] = any(tasks == set(TASKS) for tasks in shared.values())
    if not report["same_checkpoint_evaluated_on_both_tasks"]:
        lines.extend(["", "尚未具备同一检查点的双游戏完整评估证据，不能据此确认最终模型掌握两款游戏。"])
    else:
        lines.extend(["", "已确认两游戏评估包含同一个检查点。是否达成能力目标仍需看各游戏成绩，不能仅凭检查点相同判定精通。"])
    lines.extend(["", "## 指标解释", "",
                  "训练中最高分不代表稳定水平。Flappy Bird 的客户端奖励不是过管得分，"
                  "两游戏的奖励尺度不同，应分别比较。游戏端随机数未设置种子，"
                  "Python 种子不能保证相同的树枝或管道序列。", ""])
    if updates:
        for field in ("critic_loss", "policy_loss", "alpha", "history_fraction", "bound_fraction"):
            report[field] = summary(updates, field)
    write_json(directory / "report.json", report)
    inside_root(directory / "训练记录.md").write_text("\n".join(lines), encoding="utf-8")
    if episodes:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as pyplot
        figure, axes = pyplot.subplots(2, 1, figsize=(10, 7), sharex=True)
        for task in TASKS:
            completed = [row for row in episodes if row["task"] == task and row["complete"] == "True"]
            coordinates = [int(row["global_steps"]) for row in completed]
            values = [float(row["learning_return"]) for row in completed]
            rolling = [statistics.mean(values[max(0, index - 19):index + 1]) for index in range(len(values))]
            axes[0].plot(coordinates, rolling, label=f"{task} mean of last 20 completed episodes")
            axes[1].plot(coordinates, [int(row["steps"]) for row in completed], alpha=0.7, label=task)
        axes[0].set_ylabel("Training return")
        axes[1].set_ylabel("Episode steps")
        axes[1].set_xlabel("Global interaction steps")
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend()
        figure.tight_layout()
        figure.savefig(inside_root(directory / "training_curves.png"), dpi=160)
        pyplot.close(figure)
    print(f"[report] Output: {directory}")


if __name__ == "__main__":
    main()
