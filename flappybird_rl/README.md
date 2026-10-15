# Flappybird 强化学习复现（论文 WemiEnv / Flappy Bird 任务）

本文件夹是**独立的一个整体**，所有复现工作产生的文件都放在这里。

## 复现目标

复现论文 *WemiEnv: A WeChat Mini-Game Platform for Reinforcement Learning*（IEEE ToG）
中 **Fig.5** 关于 **Flappy Bird** 任务的训练曲线。

论文对**离散动作空间**任务使用 **VPG / PPO / TRPO** 三种算法。

## 为什么选这个游戏（相比 BottleJump 的优势）

| 项目 | BottleJump | Flappybird |
|---|---|---|
| Python 侧延迟 | 0.7 + 1.5 秒 | **0**（原代码无 sleep） |
| 客户端延迟 | ssx(0.7~1.1s) + 500ms | **100 ms** |
| 实测速度 | 0.33 步/秒 | 待测（预估 5~10 步/秒） |
| 奖励类型 | 稀疏（±1、+2） | **密集**（离管道中心越近分越高） |

## 环境规格（来自客户端 `lib/wxbird.js` 与 `Flappybird.py`）

| 项目 | 值 |
|---|---|
| 动作空间 | `Discrete(2)`：0 = 不飞，1 = 飞（拍翅） |
| 观测空间 | `Box(0, 600, (3,))` = `[鸟的 y, 下管道顶部 y, 上管道高度]` |
| 奖励 | 鸟在管道间隙内 → 到最近管壁的正距离；撞管道/掉地 → **−10** |
| 结束条件 | `done = stoped`（撞击或落地） |

## 目录结构

```
flappybird_rl/
├─ README.md                 ← 本文件
├─ envs/
│   └─ flappybird_env.py     ← 环境封装
├─ algos/
│   └─ ppo.py                ← PPO 实现（GAE + clip + KL 早停）
├─ speedtest.py              ← 速度实测（实测 8.47 步/秒）
├─ train.py                  ← 训练主程序
├─ plot.py                   ← 画 Fig.5 风格曲线
├─ logs/                     ← 训练记录（轮级 + 局级 CSV、模型）
└─ figures/                  ← 输出的图
```

## ✅ 训练结果（2026-09-25 完成）

| 指标 | 值 |
|---|---|
| 训练时长 | 61.6 分钟 |
| 总步数 / 局数 / 轮数 | 33,000 / 401 / 33 |
| 平均速度 | 8.93 步/秒 |
| 平均回报（最后 20 局） | **54,140.82** |
| 最高单局回报 | **164,956.21** |
| 曲线趋势 | **从 −254 单调上升至 +71,643，全程无崩溃** |

主曲线见 `figures/flappybird_epochs.png` 与 `figures/flappybird_episodes.png`。

## 论文给出的超参数（本复现严格采用）

| 参数 | 值 |
|---|---|
| 折扣因子 γ | 0.99 |
| 网络结构（VPG/PPO/TRPO） | 3 层 × 64 神经元，ReLU |
| VPG actor / critic 学习率 | 0.0003 / 0.001 |
| VPG GAE λ | 0.97 |
| PPO clip ε | 0.2 |
| PPO actor / critic 学习率 | 0.0003 / 0.001 |
| PPO GAE λ | 0.97 |
| TRPO KL 约束 | 0.01 |
| TRPO actor / critic 学习率 | 0.01 / 0.01 |
| TRPO 回溯线搜索 | 10 次共轭梯度迭代，最多 10 步，步长比例 0.8 |
| 输入 | 做归一化 |

## 使用方式

每一步都需要：**先启动 Python 脚本，再去微信开发者工具点【编译】**（顺序不能反）。

```powershell
conda activate rl
python flappybird_rl/speedtest.py
```
