# Strong Training · WemiEnv 双游戏共享强化学习

在原 WemiEnv 微信小游戏环境中，用 PyTorch 训练一份任务条件共享模型，使同一个 actor 能够分别完成 Flappy Bird 的飞行决策和 Timberman 的砍树、换边决策。

本仓库保存实际使用的训练代码、配置、实验记录、模型检查点、最终验收和中文学习资料。训练已完成，最终交付是一个共享 checkpoint；不同阶段的检查点是它的历史版本。游戏客户端轮流连接，最终评估在两款游戏分别运行。

## 1. 最终结果与证据

| 游戏 | 独立评估 | 每局上限 | 达到上限 | 自然死亡 | 平均指标 | 样本标准差 |
|---|---:|---:|---:|---:|---:|---:|
| Flappy Bird | 20 局 | 3000 步 | 20/20 | 0 | 学习回报 2267.3487 | 23.8071 |
| Timberman | 20 局 | 3000 步 | 20/20 | 0 | 实际分数 2505.85 | 12.1970 |

两次评估使用同一份 `latest.pt`，参数摘要一致，SAC 更新计数均为 `73056 → 73056`。达到上限表示在本次预算内没有死亡，随后人为截断。Flappy 的学习回报不是通过管道的数量，两款游戏的指标不能直接相加。

最终模型原始位置：

```text
FanHua/experiments/20261008_122246_330413_mainline_shared_model/checkpoints/latest.pt
```

文件 SHA256：

```text
f3ed7900942db70f70aa2479533c0622bf78d6b98a786a6534a1c43657b08bf9
```

主要证据：

- [最终验收 JSON](FanHua/reports/最终验收_单模型双游戏.json)
- [最终总结](FanHua/reports/最终总结_单模型双游戏.md)
- [主线训练报告](FanHua/experiments/20261008_122246_330413_mainline_shared_model/report.json)
- [Flappy Bird 最终逐局结果](FanHua/experiments/20261008_172006_553338_evaluate_flappybird/evaluation.json)
- [Timberman 最终逐局结果](FanHua/experiments/20261008_200326_813205_evaluate_timberman/evaluation.json)

主线起点已经通过相同上限的双游戏检查，最终 Flappy 均值相对起点约 −0.94%，Timberman 约 +0.25%。当前结果支持交替续训后的双游戏能力保持，没有证明额外 80000 步带来了显著性能增益。共享主线只有一条训练链，没有完整多 seed、消融或未见第三款游戏实验。

## 2. 下载完整项目与训练产物

GitHub 普通文件页面保存代码、配置、报告、逐局 CSV 和小型结果文件。完整的 `.pt`、原始 `protocol.jsonl`、更新明细和参考数据放在同一私有仓库的 [Release](https://github.com/SchnappiMika/Strong-Training/releases/tag/training-archive-20261015) 中。这样保留全部有用产物，同时避免大文件进入普通 Git 历史。仓库和附件都需要获得仓库访问权限才能下载。

| Release 附件 | 内容 |
|---|---|
| `Strong-Training-full-20261015.zip` | 完整项目快照，含所有归档实验、历史模型、数据和日志；解压后顶层为 `Strong-Training/` |
| `shared_model_final.pt` | 最终共享模型的逐字节副本，方便单独下载 |
| `SHA256SUMS.txt` | 两个大附件的 SHA256 校验值 |

PowerShell 下载与解压示例，需要安装并登录 GitHub CLI：

```powershell
gh auth login
gh repo clone SchnappiMika/Strong-Training
gh release download training-archive-20261015 --repo SchnappiMika/Strong-Training --pattern Strong-Training-full-20261015.zip --dir downloads
Expand-Archive -LiteralPath downloads/Strong-Training-full-20261015.zip -DestinationPath . -Force
Set-Location Strong-Training
python tools/verify_archive.py --full
```

也可以直接从 Release 下载完整 ZIP，不必先 clone。GitHub 的绿色 Code 按钮“Download ZIP”只包含普通 Git 文件，不包含 Release 中的模型和大日志。完整归档的解压覆盖恢复的是原始文件内容；已有训练记录未因整理而改写。

只查看代码的用户可运行 `python tools/verify_archive.py`，它只检查普通 Git 文件。`--full` 才检查所有归档产物。校验脚本只使用 Python 标准库，不连接游戏、不更新模型。

## 3. 目录结构

```text
Strong-Training/
├── README.md                          # 项目入口、结果、安装、复现与限制
├── .gitignore                         # 缓存、大产物及本机私有设置的分流规则
├── docs/
│   ├── DIRECTORY_STRUCTURE.md         # 完整目录树与各目录用途
│   ├── EXPERIMENT_INDEX.md            # 全部历史实验的状态与角色索引
│   ├── ARCHIVE_MANIFEST.json          # 文件路径、大小、SHA256、Git/Release 位置
│   ├── PACKAGING_POLICY.md            # 纳入范围、排除项、原始记录保护
│   └── THIRD_PARTY.md                 # 上游来源、提交与使用说明
├── tools/
│   ├── verify_archive.py             # 校验归档、核心验收和单模型一致性
│   └── prepare_portable_mainline.py   # 显式生成可迁移的续训起点，不启动训练
├── FanHua/
│   ├── README.md                     # 算法入口和历史配置说明
│   ├── requirements.txt              # 原训练环境的版本记录
│   ├── requirements-compatible.txt   # 新机器安装的兼容版本范围
│   ├── fanhua/                       # agent、WemiEnv 适配、回放和行为监督
│   ├── configs/                      # 主线、冒烟、基线和未执行消融配置
│   ├── experiments/                  # 全部历史运行，重型产物在 Release
│   ├── reports/                      # 最终报告、HTML 教材、汇报与核对记录
│   ├── reference/Cal-LAICA-Experiments/ # 使用过的上游参考快照
│   ├── train.py / evaluate.py         # 真实在线训练和冻结评估入口
│   ├── collect.py / pretrain.py       # 行为采集和离线预训练
│   ├── repair_policy.py              # 共享 actor 的教师行为修复
│   └── 其他准备、诊断、报告脚本
├── WemiEnv/                          # 当时本地 WemiEnv 代码与游戏客户端快照
│   └── vchat_minigame/
│       ├── vchat_Flappybird/          # 本次 Flappy Bird 客户端
│       └── vchat_Timberman/           # 本次 Timberman 客户端
├── flappybird_rl/                    # 旧 PPO 代码与教师权重来源
└── 文献/                             # 提供的两篇 PDF，保留于私有归档
```

[完整目录说明](docs/DIRECTORY_STRUCTURE.md) 和 [实验索引](docs/EXPERIMENT_INDEX.md) 可以定位每一份记录。原实验目录名保留时间戳，方便对照已有报告。课程笔记、BottleJump 独立训练项目及其他无关工作区文件不属于本次双游戏归档。

## 4. 模型与方法

执行动作的路线为：

```text
原 WemiEnv 观测
  → 12 维状态编码 + 2 维任务标签
  → 同一个共享 actor：14 → 64 → 64 → 64
  → 二维潜在动作
  → 当前任务的动作原型最近邻映射
  → 客户端动作 0 或 1
```

| 组件 | 本次的具体处理 |
|---|---|
| 共享 actor | task one-hot 区分游戏；输出潜在动作均值与标准差信息 |
| SAC | 双 critic、目标副本、自动熵温度；训练随机采样，评估确定性均值映射 |
| 动作表示 | 初期逆动力学与原型预热，预热后冻结任务原型 |
| CQL | 随机与策略候选的保守正则，权重 0.1 |
| 价值校准 | 自然终止完整轨迹提供有效 MC 回报，仅替换有效策略候选正则分支 |
| 历史回放 | 历史池按任务保留，历史采样概率 0.25 |
| 行为约束 | 冻结教师的动作标签，CE 加潜在原型回归；权重 5.0 |
| Timberman 聚焦 | actor 保留最近两段树枝和玩家侧别，critic 保留完整编码 |

教师只参与训练辅助，最终推理使用同一个共享 actor 与任务原型。双 critic 是共同服务两款游戏的训练组件，不是两个按游戏部署的策略。

WemiEnv 是游戏交互平台，PyTorch 是网络与优化工具。本次新增训练代码不依赖 TensorFlow、Gym 或 Gymnasium。旧 PPO 和上游参考代码作为历史材料保存，运行共享模型无需调用它们的旧训练入口。

主要参考为 [Cal-LAICA-Experiments](https://github.com/bailei951/Cal-LAICA-Experiments) 和提供的 Cal-LAICA 文稿。原论文研究动作集合逐步扩展，本次两款游戏动作集合固定，增加了任务条件与行为指导，因此属于工程适配。[算法迁移说明](FanHua/reports/算法迁移与验证说明.md) 保留细节与限制，部分文字是当时的阶段记录；当前结果以最终验收为准。

## 5. 实际训练链与预算

1. 核对原客户端协议、观测编码、奖励与随机通道。
2. 共享短程训练发现 Flappy 不拍翅，采集旧 PPO 真实行为。保留 2400 条 Flappy 记录，初次预训练实际采用 1600 条，加上 1536 条 Timberman 混合行为数据，完成 3000 次 SAC 更新。
3. 定位 Timberman 提前换边问题。第二版修复完成 5000 次监督更新，调整共享 actor 的输入聚焦；监督次数与 SAC 计数单独记录。
4. 对共享起点做双游戏长评估；单游戏 seed 0/1 基线作为参考，未合并进共享模型。
5. 主线前做 3072 步冒烟，增加 2052 次 SAC 更新；进入主线时累计计数为 5052。
6. 同一共享起点交替完成四阶段真实在线续训：

| 阶段 | 游戏 | 新增训练步 | 累计主线步 | 阶段短评估 |
|---:|---|---:|---:|---|
| 1 | Flappy Bird | 20000 | 20000 | 5 局 × 300 步，均值 226.7753 |
| 2 | Timberman | 20000 | 40000 | 5 局 × 300 步，均分 251.4 |
| 3 | Flappy Bird | 20000 | 60000 | 5 局 × 300 步，均值 225.3225 |
| 4 | Timberman | 20000 | 80000 | 5 局 × 300 步，均分 249.6 |

7. 停止训练，从同一终点做双游戏各 20 局 × 3000 步冻结验收，并生成总结。

主线新增 SAC 更新 `68004`，最终累计 `3000 + 2052 + 68004 = 73056`。交互步数不等于更新次数，因为经验在回合或阶段边界入池，样本不足时不更新。阶段评估的 6000 步和最终评估的 120000 步均不计入 80000 训练预算，也不更新参数。

## 6. 环境安装

原始运行记录见 [runtime.json](FanHua/reports/runtime.json)：Windows、Python 3.12、CPU PyTorch，以及当时保存的精确包版本。`requirements.txt` 保留该记录。不同日期、平台或软件源未必提供所有记录版本，新机器可以先使用兼容范围文件；这代表重新搭建兼容环境，不是声称逐包复现原环境。

```powershell
conda create -n fanhua python=3.12 -y
conda activate fanhua
python -m pip install -r FanHua/requirements-compatible.txt
```

若需要精确环境且软件源提供相应版本：

```powershell
python -m pip install -r FanHua/requirements.txt
```

另外安装 [微信开发者工具](https://developers.weixin.qq.com/miniprogram/dev/devtools/download.html)，导入 `WemiEnv/vchat_minigame/vchat_Flappybird` 和 `vchat_Timberman`。按工具要求使用自己有权限的 AppID 或测试项目；原本的 `project.private.config.json` 未上传，需要在本机配置开发者工具允许本地 WebSocket 调试。项目主要针对 Windows 与原微信模拟器；只有 Python 部分并不足以运行真实游戏。

## 7. 查看和校验已有结果

以下命令不连接游戏，也不训练模型：

```powershell
python tools/verify_archive.py --full
Get-FileHash -Algorithm SHA256 FanHua/experiments/20261008_122246_330413_mainline_shared_model/checkpoints/latest.pt
```

完整归档中的 CSV、JSON、协议日志与 `.pt` 保留原内容，迁移后仍可能包含原机器的绝对路径。它们作为历史来源证据保存，校验使用归档内的相对路径，不要求另一台电脑拥有原磁盘目录。

中文 HTML 页面可离线打开：

- [知识点互动教程](FanHua/reports/知识点教程_双游戏泛化.html)
- [组会汇报](FanHua/reports/组会汇报_双游戏泛化训练.html)
- [算法论文阅读清单](FanHua/reports/算法与方法_权威论文阅读清单.md)

GitHub 通常展示 HTML 源码，下载后在浏览器打开才能看到页面和互动。页面支持手机和平板；外部 PDF、原始记录链接需要保留相关目录。私有仓库没有在本次发布中开通公开 Pages 网站。

## 8. 显式运行新的独立评估

以下命令会实际启动游戏通信并生成新的评估记录。仅在需要复验时手动运行；它们不会优化模型参数。

```powershell
Set-Location FanHua
python evaluate.py --checkpoint experiments/20261008_122246_330413_mainline_shared_model/checkpoints/latest.pt --task flappybird --episodes 20 --max-steps 3000
```

看到服务监听 `7891` 后，在开发者工具编译 Flappy Bird，并保持单个调试窗口运行。评估完成、服务关闭后，再运行：

```powershell
python evaluate.py --checkpoint experiments/20261008_122246_330413_mainline_shared_model/checkpoints/latest.pt --task timberman --episodes 20 --max-steps 3000
```

这次切换并编译 Timberman。每次 Python 新服务启动后都需要对应客户端重新连接。两次必须读取同一 checkpoint。

## 9. 需要重新训练时的可迁移入口

历史主线配置与 checkpoint 保存了原机器教师路径，不能假定迁移后直接可用。本仓库提供一个单独准备工具：读取原始主线 `initialized.pt`，保留网络、优化器、回放及随机数状态，生成新的续训目录，只把教师路径改成当前机器位置。原始检查点和历史记录保持不变。

从仓库根目录执行：

```powershell
python tools/prepare_portable_mainline.py
```

工具只创建新的起点和配置，不连接客户端、不做优化更新。随后可以手动运行它输出的 `train.py --config ... --resume ...` 命令，重新进行 80000 步续训。四阶段运行期间按提示切换原游戏客户端。该流程是从保存的共享起点重新续训，不是从随机初始化完整重建全部早期实验。

若需要从随机参数进行小预算流程测试：

```powershell
Set-Location FanHua
python train.py --config configs/smoke.json --label new_smoke
```

这会更新新模型的参数，但小预算不能保证学会两款游戏。旧候选/消融配置用于历史参考，18 个比较实验未执行，不能把配置存在当成已完成实验。

## 10. 主要代码入口

| 文件 | 作用 |
|---|---|
| `FanHua/fanhua/envs.py` | 原 WebSocket 协议、严格观测校验、奖励转换、终止与截断 |
| `FanHua/fanhua/agent.py` | actor、双 critic、原型、SAC/CQL/校准更新和 checkpoint |
| `FanHua/fanhua/replay.py` | 当前/历史采样、MC 回报与有效下界 |
| `FanHua/fanhua/supervision.py` | 冻结教师加载、动作建议、CE/MSE 行为损失 |
| `FanHua/fanhua/diagnostics.py` | Timberman 经验模式的危险动作诊断 |
| `FanHua/fanhua/artifacts.py` | 强制本次运行输出留在 FanHua |
| `FanHua/train.py` | 多阶段训练、短评估、记录、断点恢复 |
| `FanHua/evaluate.py` | 确定性冻结评估与参数一致性检查 |
| `FanHua/collect.py` / `pretrain.py` | 真实行为采集与离线预训练 |
| `FanHua/repair_policy.py` | 共享 actor 行为监督修复 |
| `FanHua/report.py` | 日志汇总与训练曲线 |
| `FanHua/validate.py` | 合成环境、算法和本机通信自动验证；会创建测试产物 |
| `tools/verify_archive.py` | 标准库只读归档和最终结果校验 |

`validate.py` 包含合成训练更新与本机 WebSocket 测试，属于代码验证，不能代替真实双游戏验收。本次上传准备只运行归档校验，没有启动真实训练、游戏评估或自动实验。

## 11. 常见问题

| 现象 | 检查与处理 |
|---|---|
| clone 后没有 `.pt` 或原始协议日志 | 下载并解压完整 Release ZIP，Git 仓库本身只含轻型文件 |
| 等待客户端超时 | 先启动 Python 服务，再编译对应游戏；只保留当前游戏窗口 |
| 7891 被占用 | 检查是否有另一训练/评估服务，结束不用的服务后再显式启动 |
| 观测长度错误 | Flappy 原始 3 项，Timberman 34 项；通常是客户端选错或协议变化 |
| 教师路径不存在或恢复配置不匹配 | 用可迁移准备工具生成新目录，避免改写历史 config 与 checkpoint |
| 精确依赖版本无法安装 | 保留原记录，使用兼容范围安装并检查本机实际版本 |
| 断线后如何继续 | 原目录保留 checkpoint；按相同配置 `--resume`，微信关卡与计时不能精确恢复 |
| 如何停止新训练 | 在新实验目录创建 `STOP` 空文件，训练器检测后保存并停止；网络阻塞可能需等超时 |

## 12. 资料来源和归档边界

上游来源与版本见 [THIRD_PARTY.md](docs/THIRD_PARTY.md)。归档保留本地实际快照，不把上游代码、游戏素材、数据集或论文的许可改写成项目自己的许可。私有备份不代表获得再次公开分发的授权。

原始 `.git`、Python 缓存、浏览器调试用户目录、已安装 `node_modules`、密钥文件和开发者工具私有配置不纳入归档。完整排除清单及文件校验值见 [归档政策](docs/PACKAGING_POLICY.md) 和 [ARCHIVE_MANIFEST.json](docs/ARCHIVE_MANIFEST.json)。本次所有生成的发布文件保存在原工作区的 `FanHua/publishing/` 内。
