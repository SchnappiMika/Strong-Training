# -*- coding: utf-8 -*-
"""
用训练好的模型玩 Flappybird（效果演示）
============================================================
载入训练好的 PPO 模型，用【确定性动作】（取概率最大的动作，不再随机探索）
连玩若干局，直观看看策略学得怎么样。

用法（在 rl 环境下）：
    python flappybird_rl/play.py                # 默认玩 5 局
    python flappybird_rl/play.py 10             # 玩 10 局
    python flappybird_rl/play.py 5 --max-steps 2000
    python flappybird_rl/play.py 5 --model flappybird_rl/logs/xxx_model.pt

运行后会阻塞等待，此时请到微信开发者工具点【编译】。
============================================================
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.flappybird_env import FlappybirdEnv   # noqa: E402
from algos.ppo import PPO                        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")

# 随机策略的实测水平，用于对照
RANDOM_BASELINE_STEPS = 13.0


def find_latest_model():
    """按修改时间取最新的模型文件。"""
    files = sorted(glob.glob(os.path.join(LOG_DIR, "*_model.pt")),
                   key=os.path.getmtime)
    if not files:
        raise SystemExit(f"[错误] 在 {LOG_DIR} 里没找到任何 *_model.pt，先训练一次吧。")
    return files[-1]


def main():
    parser = argparse.ArgumentParser(description="用训练好的模型玩 Flappybird")
    parser.add_argument("episodes", type=int, nargs="?", default=5,
                        help="玩几局（默认 5）")
    parser.add_argument("--model", type=str, default=None,
                        help="模型文件路径（默认自动取最新的）")
    parser.add_argument("--max-steps", type=int, default=3000,
                        help="单局步数上限（防止模型太强一直飞）")
    args = parser.parse_args()

    model_path = args.model or find_latest_model()

    print("=" * 70)
    print("  Flappybird 演示 —— 用训练好的模型玩")
    print(f"  模型文件 : {model_path}")
    print(f"  局数     : {args.episodes}")
    print("=" * 70)
    print("  >>> 请到微信开发者工具点【编译】（项目要是 vchat_Flappybird）")

    # 建立环境（此处阻塞等待客户端）
    env = FlappybirdEnv(max_steps_per_episode=args.max_steps)
    obs_dim = env.observation_space.shape[0]
    act_dim = int(env.action_space.n)

    # 建同结构的网络并载入权重
    algo = PPO(obs_dim=obs_dim, act_dim=act_dim)
    ckpt = torch.load(model_path, map_location="cpu")
    algo.actor.load_state_dict(ckpt["actor"])
    algo.critic.load_state_dict(ckpt["critic"])
    print(f"  模型载入成功（obs_dim={obs_dim}, act_dim={act_dim}）")
    print("-" * 70)

    all_steps = []
    all_returns = []

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()          # gymnasium 接口：reset() 返回 (obs, info)
        steps = 0
        ret = 0.0
        done = False
        t0 = time.time()

        while not done:
            # deterministic=True：取概率最大的动作，不做随机探索
            a, _ = algo.select_action(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(a)
            ret += r
            steps += 1
            done = terminated or truncated

        dt = time.time() - t0
        all_steps.append(steps)
        all_returns.append(ret)

        print(f"  第 {ep} 局 | 飞行 {steps:>4} 步 | 累计回报 {ret:>10.1f} | "
              f"用时 {dt:>6.1f}s")

    # ---------------- 汇总 ----------------
    mean_steps = float(np.mean(all_steps))
    mean_return = float(np.mean(all_returns))

    print("\n" + "=" * 70)
    print("  演示结束，成绩单：")
    print(f"    平均飞行步数 : {mean_steps:.1f} 步/局")
    print(f"    最高飞行步数 : {int(np.max(all_steps))} 步")
    print(f"    平均累计回报 : {mean_return:.1f}")
    print(f"    随机策略对照 : {RANDOM_BASELINE_STEPS:.0f} 步/局")
    if mean_steps > 0:
        print(f"    → 比随机策略强 {mean_steps / RANDOM_BASELINE_STEPS:.1f} 倍")
    print("=" * 70)


if __name__ == "__main__":
    main()
