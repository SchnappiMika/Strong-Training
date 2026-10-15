import statistics

import torch


def evaluate_environment(agent, environment, task, episodes, max_steps):
    training_cap = getattr(environment, "max_episode_steps", None)
    if training_cap is not None:
        environment.max_episode_steps = max_steps
    try:
        return evaluate_with_fixed_cap(agent, environment, task, episodes, max_steps)
    finally:
        if training_cap is not None:
            environment.max_episode_steps = training_cap


def evaluate_with_fixed_cap(agent, environment, task, episodes, max_steps):
    before = agent.parameter_digest()
    updates_before = agent.updates
    rows = []
    action_counts = {"0": 0, "1": 0}
    with torch.inference_mode():
        for episode in range(episodes):
            state = environment.reset()
            total = 0.0
            raw_total = 0.0
            for step in range(max_steps):
                action, embedding = agent.select(state, task, deterministic=True)
                action_counts[str(action)] += 1
                state, reward, terminated, truncated, info = environment.step(action)
                total += reward
                raw_total += info["raw_reward"]
                if terminated or truncated:
                    break
            truncated = truncated or not terminated
            row = {"episode": episode, "task": task, "steps": step + 1, "learning_return": total,
                   "raw_return": raw_total, "score": info["score"], "terminated": terminated, "truncated": truncated}
            rows.append(row)
            outcome = "自然死亡" if terminated else "达到步数上限，无死亡"
            score_text = f"，得分 {info['score']:g}" if info["score"] is not None else ""
            print(f"[stage_evaluation] 第 {episode + 1} 局结束：{step + 1} 步{score_text}，"
                  f"学习回报 {total:.4f}，{outcome}", flush=True)
    after = agent.parameter_digest()
    if before != after or agent.updates != updates_before:
        raise RuntimeError("Stage evaluation changed model parameters or training update count")
    result = {"status": "completed", "task": task, "episodes": rows, "action_counts": action_counts,
              "evaluation_max_steps": max_steps,
              "deterministic_policy": True, "client_seed_supported": False,
              "parameters_before": before, "parameters_after": after,
              "updates_before": updates_before, "updates_after": agent.updates,
              "mean_learning_return": statistics.mean(row["learning_return"] for row in rows),
              "std_learning_return": statistics.stdev(row["learning_return"] for row in rows) if len(rows) > 1 else None,
              "mean_steps": statistics.mean(row["steps"] for row in rows)}
    if task == "timberman":
        result["mean_score"] = statistics.mean(row["score"] for row in rows)
    return result
