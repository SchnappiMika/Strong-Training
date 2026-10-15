# main.py

import os
import random
from pathlib import Path
import numpy as np
import torch

from recsys_calql.utils.Parser import Parser
from recsys_calql.utils.Config import Config
from recsys_calql.utils.Runner import Runner

PROJECT_ROOT = Path(__file__).resolve().parent


# ================================================================
# 工具：固定随机种子
# ================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"🌱 随机种子已固定: {seed}")


# ================================================================
# 工具：打印实验概览
# ================================================================

def print_experiment_info(args, config: Config):
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"🧪 实验概览")
    print(f"   数据集:      {args.experiment}")
    print(f"   算法:        CAL-QL")
    print(f"   随机种子:    {args.seed}")
    print(f"   时间戳:      {args.timestamp}")
    print(f"   设备:        {args.device}")
    print(f"   阶段数:      {config.change_num}")
    print(f"   每阶段回合:  {config.episodes_per_phase}")
    print(f"   Buffer 大小: {config.buffer_size:,}")
    print(f"   Batch 大小:  {config.sac_batch_size}")
    print(f"   CAL-QL:      {config.use_cal_ql}")
    print(f"   日志目录:    {config.log_base_dir}/{config.dataset}/CAL-QL/seed_{args.seed}/")
    print(f"{sep}\n")


# ================================================================
# 主函数
# ================================================================

def main():
    # ── Step 1: 解析命令行参数 ──────────────────────────────────
    parser = Parser()
    args   = parser.parse()

    torch_num_threads = os.environ.get("TORCH_NUM_THREADS")
    if torch_num_threads is not None:
        try:
            torch.set_num_threads(int(torch_num_threads))
        except Exception:
            pass
    torch_num_interop_threads = os.environ.get("TORCH_NUM_INTEROP_THREADS")
    if torch_num_interop_threads is not None:
        try:
            torch.set_num_interop_threads(int(torch_num_interop_threads))
        except Exception:
            pass

    # ── Step 2: 固定随机种子 ────────────────────────────────────
    set_seed(args.seed)

    # ── Step 3: 选择设备 ────────────────────────────────────────
    if args.device == "cpu":
        device = "cpu"
    else:
        device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
        if not torch.cuda.is_available():
            print(f"⚠️  GPU 不可用，已自动切换到 CPU")

    # ── Step 4: 初始化 Config ───────────────────────────────────
    print("\n" + "=" * 65)
    print("⚙️  初始化 Config & 构建环境...")
    print("=" * 65)

    data_path = args.data_path
    if data_path is None:
        exp_key = str(args.experiment).upper().replace("_", "-")
        data_name = "ab-1m" if exp_key in {"AB-1M", "AB1M"} else "ml-1m"
        data_path = PROJECT_ROOT / "data" / data_name
    else:
        data_path = Path(data_path)
        if not data_path.is_absolute():
            data_path = PROJECT_ROOT / data_path

    if args.run_group is not None:
        run_group = args.run_group
        env_name = args.experiment
    else:
        exp_key = str(args.experiment).upper().replace("_", "-")
        if exp_key in {"ML-1M", "AB-1M", "AB1M"}:
            run_group = args.experiment
            env_name = args.experiment
        else:
            run_group = args.experiment
            env_name = "ML-1M"

    config = Config(
        # 数据
        data_path        = str(data_path),
        test_ratio       = args.test_ratio,
        by_time          = args.by_time,
        # 环境
        embedding_dim    = args.embedding_dim,
        history_len      = args.history_len,
        device           = device,
        use_initial_emb  = args.use_initial_emb,
        change_num       = args.change_num,
        # FunkSVD 训练
        n_epochs         = args.n_epochs,
        batch_size       = args.batch_size,
        learning_rate    = args.learning_rate,
        weight_decay     = args.weight_decay,
        # SAC / CQL / CAL-QL Agent
        item_embedding_dim  = args.item_embedding_dim,
        state_hidden_dim    = args.state_hidden_dim,
        action_emb_dim      = args.action_emb_dim,
        policy_hidden       = args.policy_hidden,
        critic_hidden       = args.critic_hidden,
        rnn_type            = args.rnn_type,
        gamma               = args.gamma,
        tau                 = args.tau,
        alpha_init          = args.alpha_init,
        lr                  = args.lr,
        buffer_size         = args.buffer_size,
        max_seq_len         = args.max_seq_len,
        sac_batch_size      = args.sac_batch_size,
        # CQL / CAL-QL
        cql_alpha           = args.cql_alpha,
        cql_num_random      = args.cql_num_random,
        cql_with_lagrange   = args.cql_with_lagrange,
        cql_lagrange_thresh = args.cql_lagrange_thresh,
        use_cal_ql          = True,
        env_name            = env_name,
        legacy_cal_ql_mode  = True,
        kg_feature_alpha    = args.kg_feature_alpha,
        user_feature_alpha  = args.user_feature_alpha,
        min_user_interactions = args.min_user_interactions,
    )

    # ── Step 5: 注入 Runner 所需的额外字段 ─────────────────────
    # Config 本身没有这些字段，Runner.__init__ 通过 getattr 读取
    config.dataset         = run_group               # 目录层级: Environments/<run_group>/
    config.exp_name        = "CAL-QL"                # 目录层级: CAL-QL
    config.seed            = args.seed                # 目录层级: seed_42
    config.log_base_dir    = str(PROJECT_ROOT / "outputs")
    config.episodes_per_phase = args.episodes_per_phase
    config.log_interval    = args.log_interval
    config.save_interval   = args.save_interval
    config.test_user_count = args.test_user_count
    config.eval_interval   = args.eval_interval
    config.eval_user_count = args.eval_user_count
    config.eval_phase      = args.eval_phase
    config.disable_offline_online_buffer = args.disable_offline_online_buffer
    config.offline_sample_prob = args.offline_sample_prob
    config.save_q_values = args.save_q_values
    config.q_output_filename = args.q_output_filename

    # ── Step 6: 打印实验概览 ────────────────────────────────────
    print_experiment_info(args, config)

    # ── Step 7: 初始化 Runner ───────────────────────────────────
    runner = Runner(config)

    # ── Step 8: 启动训练 ────────────────────────────────────────
    rewards, std_rewards = runner.Cal_QL_Run()

    # ── Step 9: 简单汇报结果 ────────────────────────────────────
    print("\n" + "=" * 65)
    print("📈 训练结束，结果摘要")
    print(f"   总回合数:          {len(rewards)}")
    print(f"   最终均值(后100):   {np.mean(rewards[-100:]):.4f}")
    print(f"   历史最佳单回合:    {max(rewards):.4f}")
    print(f"   结果文件目录:")
    print(f"   {config.log_base_dir}/{config.dataset}/CAL-QL/seed_{args.seed}/")
    print("=" * 65)


if __name__ == "__main__":
    main()
