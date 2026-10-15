# FanHua：一个共享模型的双游戏持续训练

本目录保存 Flappy Bird 与 Timberman 共享训练的新增代码、全部实验产物和中文材料。当前训练、最终双游戏验收与总结均已完成。项目安装、Release 下载和可迁移运行入口见[仓库根 README](../README.md)。

最终共享 checkpoint：`experiments/20261008_122246_330413_mainline_shared_model/checkpoints/latest.pt`。两款游戏最终各 20 局、每局最多 3000 步，均达到上限且没有自然死亡；评估参数未改变。主线在线预算为 80000 步，四阶段各 20000 步，最终 SAC 累计更新 73056。完整证据见 [最终验收](reports/最终验收_单模型双游戏.json)。

## 目录用途

| 目录/入口 | 说明 |
|---|---|
| `fanhua/` | PyTorch 共享 agent、原 WemiEnv 协议适配、回放、行为监督、诊断 |
| `configs/mainline_shared_model.json` | 实际主线超参数，原教师绝对路径作为历史信息保存 |
| `configs/smoke.json` | 随机初始化、小预算的共享流程检查 |
| `configs/prerequisites_v2/` | 已执行单游戏 seed 0/1 基线及未运行候选/消融配置 |
| `configs/ablations/` | 早期对照方案，配置存在不代表完成实验 |
| `experiments/` | 时间戳目录保存原始运行结果；重型数据从 Release 下载 |
| `reports/` | 最终总结、HTML 教材、组会材料、阶段诊断和历史核对 |
| `reference/Cal-LAICA-Experiments/` | 对照的上游参考代码快照，不是本次训练入口 |
| `collect.py`, `pretrain.py`, `repair_policy.py` | 真实数据采集、离线学习和共享 actor 修复 |
| `train.py`, `evaluate.py`, `report.py` | 在线训练、冻结评估、汇总曲线 |
| `prepare_*.py` | 各时期准备脚本，有些含原机器路径，保留以追溯流程 |
| `record_baseline.py`, `review_prerequisites.py` | 早期基线登记与候选设计核对 |
| `validate.py` | 合成环境和本机通信验证，会生成测试产物 |

输出约束由 `fanhua/artifacts.py` 实现：新实验、报告、数据和 checkpoint 必须留在本 FanHua 目录内。代码运行时应先切换到 FanHua。

## 配置与记录怎么读

本次主线以 SAC 为骨架，加 CQL/校准、潜在动作、历史回放和训练期行为指导。教师冻结，最终模型执行时不调用教师。Timberman 输入聚焦只改变 actor 可见特征，未加推理动作覆盖规则。

历史报告记录了各自生成时的状态，部分含“尚待评估”“正式实验暂停”等文字。最终状态以 `reports/最终验收_单模型双游戏.json` 与根 README 为准；历史文件没有改写。单游戏基线是参照，不是最终双游戏部署策略；原 9 组 × 2 seed 的比较实验未运行。

`.pt` 包含网络、优化器、回放和随机数状态。`stage_01.pt` 等是同一运行的时间版本，`latest.pt` 为最终版本。原始二进制保持校验一致，不直接修改里面保存的配置来迁移电脑。新的续训准备使用根目录 `tools/prepare_portable_mainline.py`，单独输出新的 checkpoint。

## 独立评估示例

需要先下载完整 Release，并在开发者工具准备原客户端。仅在手动需要复验时执行：

```powershell
python evaluate.py --checkpoint experiments/20261008_122246_330413_mainline_shared_model/checkpoints/latest.pt --task flappybird --episodes 20 --max-steps 3000
```

等待监听 7891 后编译 Flappy Bird。完成后用同一 checkpoint 运行 `--task timberman`，切换并重新编译 Timberman。两次均不更新参数。查看报告不需要执行这些命令。

## 学习与汇报入口

- [中文互动教材](reports/知识点教程_双游戏泛化.html)
- [组会汇报](reports/组会汇报_双游戏泛化训练.html)
- [算法原始论文阅读清单](reports/算法与方法_权威论文阅读清单.md)
- [Cal-LAICA 中文整理](AAAI_论文中文总结_动态动作空间持续强化学习.md)
- [最终总结](reports/最终总结_单模型双游戏.md)
