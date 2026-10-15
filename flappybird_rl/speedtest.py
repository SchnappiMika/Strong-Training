# -*- coding: utf-8 -*-
"""
Flappybird 速度实测
============================================================
目的：测出真实速度，用来决定后面 on-policy 算法（VPG/PPO/TRPO）的
      「每次采样多少步」「训练多少轮」等参数。

为什么要单独测：
    本游戏客户端每步有 100ms 的固定节流，Python 侧没有 sleep。
    真实速度必须实测，不能拍脑袋。

用法（在 rl 环境下）：
    python flappybird_rl/speedtest.py            # 默认跑 3 局
    python flappybird_rl/speedtest.py 5          # 跑 5 局

运行后会阻塞等待，此时请到微信开发者工具点【编译】。
============================================================
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.flappybird_env import FlappybirdEnv  # noqa: E402


def main():
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    print("=" * 66)
    print(f"  Flappybird 速度实测（随机动作，{episodes} 局）")
    print("=" * 66)
    print("  >>> 请到微信开发者工具点【编译】（项目要是 vchat_Flappybird）")

    env = FlappybirdEnv()
    print(f"  观测空间 = {env.observation_space}")
    print(f"  动作空间 = {env.action_space}")

    all_steps = []
    total_steps = 0
    global_start = time.time()

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()          # gymnasium 接口：reset() 返回 (obs, info)
        ep_steps = 0
        ep_reward = 0.0
        ep_start = time.time()

        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            ep_steps += 1
            total_steps += 1
            if terminated or truncated:
                break

        ep_time = time.time() - ep_start
        all_steps.append(ep_steps)
        print(f"  第 {ep} 局 | {ep_steps:>4} 步 | 累计回报 {ep_reward:>9.1f} | "
              f"用时 {ep_time:>5.2f}s | {ep_steps / ep_time:>6.2f} 步/秒")

    # ---------------- 汇总 ----------------
    total_time = time.time() - global_start
    speed = total_steps / total_time
    mean_steps = sum(all_steps) / len(all_steps)

    print("\n" + "=" * 66)
    print("  实测结果：")
    print(f"    总步数       : {total_steps}")
    print(f"    总用时       : {total_time:.2f} 秒")
    print(f"    平均速度     : {speed:.2f} 步/秒")
    print(f"    平均每局步数 : {mean_steps:.1f}")
    print(f"    → 1 小时可跑 : {speed * 3600:,.0f} 步 ≈ {3600 * speed / mean_steps:,.0f} 局")
    print("=" * 66)


if __name__ == "__main__":
    main()
