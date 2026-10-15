# 第三方来源与本地快照

本项目保留上游与历史代码的名称、目录和已有说明。以下提交标识描述本地基础版本；实际快照还包含本机已修改、未提交的游戏适配文件，不能把提交号理解为快照全部内容都未修改。

| 内容 | 来源 | 基础提交/说明 |
|---|---|---|
| WemiEnv | https://github.com/Ada9io/WemiEnv | `5970399a0e8f1c29ed2da5b453606a323136882d`；归档本地修改与实际客户端 |
| Cal-LAICA-Experiments | https://github.com/bailei951/Cal-LAICA-Experiments | `35ba6fc167d9f3257a1c12d943600b096e79f6b9`；作方法对照，非共享模型执行入口 |
| flappybird_rl | 原工作区中的 PPO 工程 | 旧 Flappy 行为教师和数据来源，非上游公开仓库声明 |
| Cal-LAICA 论文 | 用户提供的匿名 PDF | 文件名不代表已被 AAAI 接收；方法参考与学习材料 |
| WemiEnv 论文 | 用户提供的 camera-ready PDF | 平台论文，来源信息见原 WemiEnv README |
| 参考仓库附带数据 | 上游 `data/` | MovieLens/AB-1M 等数据保留上游已有说明；未用于本次双游戏训练 |

第三方代码、数据集、游戏素材和论文的权利归各自作者或权利人。已有许可文件原样保留。缺少明确许可时，本项目不自行授予公开再分发或商业使用权；本次上传为用户指定的私有备份。

本次共享训练的 SAC、CQL、Cal-QL、动作表示等论文来源与进一步阅读见 `FanHua/reports/算法与方法_权威论文阅读清单.md`。
