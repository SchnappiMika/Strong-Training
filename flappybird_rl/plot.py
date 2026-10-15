# -*- coding: utf-8 -*-
"""
画 Flappybird 训练曲线（论文 Fig.5 风格）
============================================================
读取 logs/ 下的训练记录，画「总交互步数 vs 平均回报」曲线。

用法（在 rl 环境下）：
    python flappybird_rl/plot.py                    # 画最新的训练
    python flappybird_rl/plot.py --all              # 把所有训练画在一起对比
    python flappybird_rl/plot.py --window 20        # 局级滑动平均窗口
    python flappybird_rl/plot.py --kind episodes    # 用局级明细画（更细腻）

图会保存到 figures/ 目录。
============================================================
"""
import argparse
import csv
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
FIG_DIR = os.path.join(HERE, "figures")

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def moving_average(x, window):
    window = max(1, min(window, len(x)))
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def load_epochs(path):
    steps, returns = [], []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            steps.append(int(row["global_step"]))
            returns.append(float(row["mean_return"]))
    return np.array(steps), np.array(returns)


def load_episodes(path):
    steps, returns = [], []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            steps.append(int(row["global_step"]))
            returns.append(float(row["episode_return"]))
    return np.array(steps), np.array(returns)


def main():
    parser = argparse.ArgumentParser(description="画 Flappybird 训练曲线")
    parser.add_argument("--all", action="store_true", help="把所有训练画在一起")
    parser.add_argument("--kind", choices=["epochs", "episodes"], default="epochs",
                        help="epochs=每轮均值（默认）；episodes=每局明细")
    parser.add_argument("--window", type=int, default=20, help="局级滑动平均窗口")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)

    pattern = os.path.join(LOG_DIR, f"*_{args.kind}_*.csv")
    # 按【文件修改时间】排序。
    # 注意：不能用文件名字母序 —— 例如 "smoke_..." 会排在 "ppo_..." 之后，
    # 导致「取最新一个」取到冒烟测试的数据。
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not args.all:
        files = files[-1:]

    if not files:
        raise SystemExit(f"[错误] 在 {LOG_DIR} 找不到匹配 {pattern} 的文件。")

    fig, ax = plt.subplots(figsize=(7.5, 5))

    for path in files:
        if args.kind == "epochs":
            x, y = load_epochs(path)
            label = os.path.basename(path).split("_epochs_")[0]
            ax.plot(x, y, marker="o", markersize=3, linewidth=1.6, label=label)
        else:
            x, y = load_episodes(path)
            label = os.path.basename(path).split("_episodes_")[0]
            sm = moving_average(y, args.window)
            x2 = x[len(x) - len(sm):]
            ax.plot(x2, sm, linewidth=1.6, label=label)
            ax.plot(x, y, alpha=0.15, linewidth=0.6)

        print(f"  {os.path.basename(path)}: {len(x)} 个点, "
              f"最后值 {y[-1]:.2f}, 最高 {y.max():.2f}")

    ax.set_xlabel("Total Interactions (global step)")
    ax.set_ylabel("Average Return")
    ax.set_title("Flappy Bird — 训练曲线")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    out_name = args.out or f"flappybird_{args.kind}"
    out_path = os.path.join(FIG_DIR, f"{out_name}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"\n  图已保存：{out_path}")


if __name__ == "__main__":
    main()
