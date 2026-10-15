# FanHua：基于 WemiEnv 的双游戏持续学习

此目录保存本次 Flappy Bird + Timberman 的全部新增代码、数据、模型和报告。游戏仍由原来的微信开发者工具项目运行；Python 通过原有 WebSocket 协议与 WemiEnv 游戏客户端交互。

## 运行环境

使用已有 `rl` 环境：Python 3.12.14、PyTorch 2.14.0+cpu、NumPy、websocket-server。当前 PyTorch 没有 CUDA 支持，因此使用 CPU。新增执行代码没有导入 TensorFlow、Gym 或 Gymnasium。Gymnasium 是环境接口，与 PyTorch 不是替代关系；本次按用户要求直接实现 WemiEnv 适配接口。

PowerShell 的环境激活仅影响当前终端。在工作区根目录可运行：

```powershell
& 'C:/Users/Schnappi SaMa/Anaconda3/shell/condabin/conda-hook.ps1'
conda activate rl
Set-Location './FanHua'
python validate.py
```

在工作区根目录，无需激活也可以显式调用同一个解释器：

```powershell
& 'C:/Users/Schnappi SaMa/Anaconda3/envs/rl/python.exe' './FanHua/validate.py'
```

微信开发者工具每次需在 Python 服务开始监听 `7891` **之后**编译对应游戏。一次仅保留一个游戏调试窗口。切换阶段时旧连接会关闭，需要切换游戏并重新编译。

## 代码与输出

| 文件 | 作用 |
|---|---|
| `fanhua/envs.py` | 原客户端协议、观测编码、奖励转换、终止/截断处理 |
| `fanhua/agent.py` | 共享 PyTorch actor、双 critic、目标网络、动作表示、CQL/校准更新 |
| `fanhua/replay.py` | 当前阶段经验池、按任务保留的历史池、完整轨迹 MC 回报 |
| `probe.py` | 随机交互测试，不更新模型 |
| `collect.py` | 随机策略或已有 Flappy PPO 的行为数据采集 |
| `pretrain.py` | 基于真实采集数据的离线 CQL 预训练 |
| `export_dataset.py` | 从真实训练回放导出数据，保留来源；不将混合策略数据称为专家数据 |
| `diagnose_timberman.py` | 对照新旧模型在实际失败日志中的动作，定位提前换边错误 |
| `repair_policy.py` | 对共享 actor 做训练指导修复，保留 critic 的真实转移记录 |
| `fanhua/diagnostics.py` | 检查回放中有效 Timberman 模式的危险动作，供训练退步检测使用 |
| `train.py` | 在线阶段训练、日志、检查点、恢复 |
| `evaluate.py` | 独立确定性评估，校验参数哈希与更新计数不变 |
| `validate.py` | 算法、恢复、断线及本机 WebSocket 的自动验证 |
| `report.py` | 读取实验日志，生成中文摘要与曲线 |
| `prepare_prerequisites.py` | 生成独立基线及新版正式实验组配置，保留正式实验暂停状态 |
| `record_baseline.py` | 登记基线进展，核对完整训练预算和最终独立评估 |
| `review_prerequisites.py` | 核对基线验收、各组配置摘要、预算与消融差异，生成前置核对报告 |
| `configs/` | 短程配置、正式候选配置与消融配置 |
| `experiments/` | 每次运行的独立目录；测试数据也保存在这里 |
| `reference/Cal-LAICA-Experiments/` | 原作者代码，用于核对方法，不是本次训练入口 |
| `reports/` | 算法迁移说明与已验证事实 |

`python prepare_configs.py` 生成同预算的校准/回放消融配置和 3 个 Python 随机种子配置，并保存当前环境版本。它只创建配置，不运行实验。`events.jsonl` 记录新入口启动后的阶段、错误与恢复；此前短程运行的第一次断线见报告及原始协议日志。

训练目录包含 `config.json`、`status.json`、`episodes.csv`、`updates.csv`、`protocol.jsonl`、`checkpoints/latest.pt` 和 `checkpoints/stage_XX.pt`。`latest.pt` 保存模型参数、优化器、温度系数、回放数据、随机数状态和阶段进度；`stage_XX.pt` 保留阶段结束时的模型。扩展名 `.pt` 与 `.ckpt` 本身不能决定内容，必须看保存代码；`.checkout` 不是这里采用的 PyTorch 检查点约定。

## 分阶段运行

2026-10-06 补齐工作见 `reports/前置工作补齐与实验设计_v2_20261006.md`。新版配置位于 `configs/prerequisites_v2`：先从随机初始化分别训练单游戏基线，各 40000 步，再另起进程独立评估 20 局、每局最多 3000 步。当前仅执行基线；长程共享模型与正式实验继续搁置。正式配置在前置验收及明确恢复正式实验之前会拒绝启动，旧候选配置仅作历史参考。

```powershell
python train.py --config configs/prerequisites_v2/baseline_flappybird_seed_0.json --label baseline_flappybird_seed_0
python evaluate.py --checkpoint experiments/<基线目录>/checkpoints/latest.pt --task flappybird --episodes 20 --max-steps 3000
python record_baseline.py --config configs/prerequisites_v2/baseline_flappybird_seed_0.json --run experiments/<基线目录> --evaluation experiments/<评估目录>/evaluation.json
```

`record_baseline.py` 验证通过后才将该项标记完成；登记结果不会自动开放正式实验。

在 FanHua 目录内执行以下命令。短程测试没有离线预训练，作用是验证实际训练链路，不能据此宣布模型已经精通游戏。

```powershell
python train.py --config configs/timber_first_smoke.json --label timber_first_smoke
```

这个配置按 Timberman → Flappy Bird → Timberman → Flappy Bird 各运行 768 步，使用同一个 actor 与同一组 critic。出现 `[WemiEnv] Compile ... now.` 时，切换并编译相应项目。

中断或断线后保留检查点，恢复时配置必须一致：

```powershell
python train.py --config configs/timber_first_smoke.json --resume experiments/<运行目录>/checkpoints/latest.pt
```

模型和优化器可恢复；微信游戏随机数、计时器和中途状态不能恢复，恢复时会重新开始当前游戏的一局，不声称精确重放同一段游戏。

从离线预训练检查点转入在线训练时，入口自动创建新的实验目录，保留离线检查点和日志。在线检查点默认在原运行目录续训；添加 `--new-run --label <名称>` 可保存到新目录。这样在线与离线 CSV 的不同字段不会混在同一个文件。

需要停止较长的训练时，在该实验目录创建名为 `STOP` 的空文件，训练器在下一步交互前停止并保存检查点；它会消费这个文件，恢复后可正常续训。断线仍按失败记录，避免把断线当作正常完成。

修复模型启用 `policy_focus_timber` 时，恢复及定期检查点会检查历史回放中的有效 Timberman 模式；出现危险预测则保存失败现场并停止。结果见 `initial_policy_diagnostic.json`、`policy_diagnostics.jsonl` 和 `status.json`。这项检查只覆盖已观察到的模式，真实双游戏评估仍须完成。训练指导模型只为 actor 提供训练目标；评估时使用同一个共享 actor。

## 正式训练前的数据与评估

正式候选配置 `configs/formal_candidate.json` 需要通过短程训练与独立评估后再使用。预算是起点，不是精通的保证。先用 `collect.py` 采集真正的客户端轨迹；已有 PPO 权重是 Flappy Bird 行为模型，不能直接当作 Timberman 的老师：

初始短程模型的 Flappy Bird 出现一直不跳，结果保留在报告中。`configs/behavior_pretrain_candidate.json` 提供额外的行为约束候选，权重为 5；这是本次工程扩展，原论文没有该损失项，必须与原配置对比。详见迁移报告；当前候选尚待真实双游戏评估，不应把离线动作一致率当成精通证明。

```powershell
python collect.py --task flappybird --behavior legacy-ppo --legacy-model ../flappybird_rl/logs/ppo_20260925_174115_model.pt --episodes 20
python collect.py --task timberman --behavior random --episodes 100
python pretrain.py --config configs/formal_candidate.json --dataset experiments/<Flappy采集目录>/dataset.pt --dataset experiments/<Timber采集目录>/dataset.pt --updates 1000
python train.py --config configs/formal_candidate.json --resume experiments/<预训练目录>/checkpoints/latest.pt
```

随机 Timberman 数据不叫专家数据；采集后必须检查动作覆盖、轨迹长度、回报与终止状态。离线预训练数据必须与训练的奖励定义、观测编码和折扣系数一致。

每个阶段的模型都应在两个游戏分别评估，最终两个游戏必须使用**同一检查点**：

正式候选配置在每个训练阶段结束后自动进行当前游戏 20 局、每局最多 300 步的确定性评估，保存 `stage_XX_<游戏>_evaluation.json`，不加入经验池，也不修改模型。此时已使用当前连接，无需额外编译。同一阶段检查点在另一款游戏的评估仍需手动切换后运行独立评估脚本。

```powershell
python evaluate.py --checkpoint experiments/<运行目录>/checkpoints/latest.pt --task flappybird --episodes 20
python evaluate.py --checkpoint experiments/<运行目录>/checkpoints/latest.pt --task timberman --episodes 20
python report.py --run experiments/<运行目录> --evaluation experiments/<Flappy评估目录>/evaluation.json --evaluation experiments/<Timber评估目录>/evaluation.json
```

Flappy Bird 协议不发送实际过管得分，因此记录客户端回报与存活步数；Timberman 记录实际分数。不能把 Flappy Bird 的奖励当作过管分数。评估阶段不更新任何模型参数。

## 如何判断达到预期

1. 协议门槛：正确游戏、观测长度、两个动作、重置、奖励差分和真正终止均验证通过。
2. 算法门槛：有限损失、非零梯度、参数确实改变、目标网络更新、历史样本存在，检查点恢复下一次更新一致。
3. 短程门槛：两款游戏能用同一模型完成多阶段训练；切换任务时回放样本比例正常，无断线或数值错误。
4. 效果门槛：独立评估与同预算对照；不能用训练中最高分替代平均表现。记录学习新任务后旧任务平均回报的变化，衡量遗忘。
5. 最终门槛：同一检查点在两款游戏上分别重复评估，报告均值、标准差、截断比例。游戏端 `Math.random` 未支持种子，不能声称严格可重复的地图序列。

遇到观测不符时检查是否开错游戏；等待超时时重新编译正确项目并从检查点恢复；数值不有限时保留诊断数据并暂停训练。若分数没有改善，先检查游戏逻辑、状态信息和动作映射，再结合对照调整学习率、CQL 权重、历史比例和训练预算。详细方法限制见 `reports/算法迁移与验证说明.md`。
