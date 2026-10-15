# -*- coding: utf-8 -*-
"""
Flappybird × PPO 训练主程序
============================================================
流程（on-policy）：
    1. 用当前策略连续采样 steps_per_epoch 步，记录 obs/action/reward/V/logp
    2. 用 GAE 计算优势与回报
    3. 做一轮 PPO 更新（clip + KL 早停）
    4. 重复，直到时间上限

记录两个 CSV：
    logs/<tag>_epochs_<时间>.csv    每轮的汇总（用来画主曲线）
    logs/<tag>_episodes_<时间>.csv  每局的明细

用法（在 rl 环境下）：
    python flappybird_rl/train.py                          # 默认 60 分钟
    python flappybird_rl/train.py --minutes 5              # 冒烟测试
    python flappybird_rl/train.py --minutes 60 --steps-per-epoch 1000

运行后会阻塞等待，此时请到微信开发者工具点【编译】。
============================================================
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.flappybird_env import FlappybirdEnv          # noqa: E402
from algos.ppo import PPO, compute_gae                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")


def main():
    parser = argparse.ArgumentParser(description="Flappybird × PPO 训练")
    parser.add_argument("--minutes", type=float, default=60.0, help="时间上限（分钟）")
    parser.add_argument("--steps-per-epoch", type=int, default=1000,
                        help="每轮采样步数（Spinning Up 默认 4000，"
                             "但我们的总预算小，取 1000 以获得更多轮更新）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default="ppo")
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    epoch_csv = os.path.join(LOG_DIR, f"{args.tag}_epochs_{stamp}.csv")
    ep_csv = os.path.join(LOG_DIR, f"{args.tag}_episodes_{stamp}.csv")

    print("=" * 72)
    print("  Flappybird × PPO 训练")
    print(f"  时间上限     : {args.minutes} 分钟")
    print(f"  每轮采样步数 : {args.steps_per_epoch}")
    print(f"  随机种子     : {args.seed}")
    print(f"  轮级日志     : {epoch_csv}")
    print(f"  局级日志     : {ep_csv}")
    print("=" * 72)
    print("  >>> 请到微信开发者工具点【编译】（项目要是 vchat_Flappybird）")

    env = FlappybirdEnv()
    obs_dim = env.observation_space.shape[0]
    act_dim = int(env.action_space.n)
    print(f"  环境就绪：obs_dim={obs_dim}  act_dim={act_dim}")

    algo = PPO(obs_dim=obs_dim, act_dim=act_dim, seed=args.seed)
    print("  PPO 已初始化（γ=0.99, clip=0.2, lr=0.0003/0.001, λ=0.97, 网络 64×64 ReLU）")
    print("-" * 72)

    f_epoch = open(epoch_csv, "w", newline="", encoding="utf-8")
    w_epoch = csv.writer(f_epoch)
    w_epoch.writerow(["epoch", "global_step", "mean_return", "mean_ep_len",
                      "episodes", "pi_loss", "v_loss", "kl", "entropy",
                      "elapsed_sec"])
    f_epoch.flush()

    f_ep = open(ep_csv, "w", newline="", encoding="utf-8")
    w_ep = csv.writer(f_ep)
    w_ep.writerow(["global_step", "episode", "episode_return", "episode_steps",
                   "elapsed_sec"])
    f_ep.flush()

    # ---------------- 训练循环 ----------------
    obs, _ = env.reset()
    episode = 1
    ep_return = 0.0
    ep_steps = 0
    global_step = 0
    epoch = 0
    all_ep_returns = []

    start_time = time.time()
    deadline = start_time + args.minutes * 60

    try:
        while time.time() < deadline:
            epoch += 1
            n = args.steps_per_epoch

            obs_buf = np.zeros((n, obs_dim), dtype=np.float32)
            act_buf = np.zeros(n, dtype=np.int64)
            rew_buf = np.zeros(n, dtype=np.float32)
            val_buf = np.zeros(n, dtype=np.float32)
            logp_buf = np.zeros(n, dtype=np.float32)
            done_buf = np.zeros(n, dtype=np.float32)   # 真终止标记

            # ---------- 1) 采样 ----------
            for t in range(n):
                a, logp = algo.select_action(obs)
                v = algo.value(obs)
                next_obs, r, terminated, truncated, _ = env.step(a)
                done = terminated or truncated

                obs_buf[t] = obs
                act_buf[t] = a
                rew_buf[t] = r
                val_buf[t] = v
                logp_buf[t] = logp
                done_buf[t] = float(terminated)     # 截断时仍可 bootstrap

                obs = next_obs
                ep_return += r
                ep_steps += 1
                global_step += 1

                if done:
                    all_ep_returns.append(ep_return)
                    w_ep.writerow([global_step, episode, round(ep_return, 3),
                                   ep_steps, round(time.time() - start_time, 2)])
                    f_ep.flush()

                    obs, _ = env.reset()
                    episode += 1
                    ep_return = 0.0
                    ep_steps = 0

            # ---------- 2) GAE ----------
            last_val = 0.0 if done_buf[-1] else algo.value(obs)
            adv, ret = compute_gae(rew_buf, val_buf, done_buf, last_val,
                                   algo.gamma, algo.lam)

            # ---------- 3) 更新 ----------
            stats = algo.update(obs_buf, act_buf, adv, ret, logp_buf)

            # ---------- 4) 记录本轮 ----------
            # 统计本轮内结束的局
            n_ep_done = len(all_ep_returns)
            elapsed = time.time() - start_time

            # 取最近若干局的均值作为"当前水平"
            recent = all_ep_returns[-max(1, n_ep_done // max(1, epoch)):] if all_ep_returns else [0.0]
            mean_return = float(np.mean(recent)) if recent else 0.0

            w_epoch.writerow([epoch, global_step, round(mean_return, 3),
                              round(mean_return, 2), n_ep_done,
                              round(stats["pi_loss"], 6), round(stats["v_loss"], 6),
                              round(stats["kl"], 6), round(stats["entropy"], 4),
                              round(elapsed, 2)])
            f_epoch.flush()

            speed = global_step / elapsed if elapsed > 0 else 0
            print(f"  轮 {epoch:>3} | 步数 {global_step:>6} | 最近均回报 {mean_return:>8.2f} | "
                  f"pi_loss {stats['pi_loss']:>8.3f} | v_loss {stats['v_loss']:>9.2f} | "
                  f"KL {stats['kl']:.4f} | 用时 {elapsed / 60:>4.1f}min | "
                  f"{speed:>5.2f} 步/秒")

    except KeyboardInterrupt:
        print("\n  [已手动中断，正在保存 ……]")
    finally:
        f_epoch.close()
        f_ep.close()

        model_path = os.path.join(LOG_DIR, f"{args.tag}_{stamp}_model.pt")
        algo.save(model_path)

        total_time = time.time() - start_time
        print("-" * 72)
        print("  训练结束，汇总：")
        print(f"    轮数        : {epoch}")
        print(f"    总步数      : {global_step}")
        print(f"    总局数      : {episode - 1}")
        print(f"    总用时      : {total_time / 60:.2f} 分钟")
        print(f"    平均速度    : {global_step / total_time:.2f} 步/秒")
        if all_ep_returns:
            print(f"    全局平均回报: {np.mean(all_ep_returns):.2f}")
            print(f"    最后 20 局  : {np.mean(all_ep_returns[-20:]):.2f}")
            print(f"    最高回报    : {np.max(all_ep_returns):.2f}")
        print(f"    模型文件    : {model_path}")
        print("-" * 72)
        print("  下一步：python flappybird_rl/plot.py")


if __name__ == "__main__":
    main()
