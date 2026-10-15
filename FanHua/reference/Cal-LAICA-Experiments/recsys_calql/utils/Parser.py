import argparse
from datetime import datetime

class Parser(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description="推荐系统强化学习实验参数解析")

        # 实验基础参数
        parser.add_argument("--experiment", default='AB-1M', help="环境名（例如 AB-1M 或 ML-1M）；也可用于输出目录分组名")
        parser.add_argument("--run_group", default=None, help="输出目录分组名（用于 Environments/<run_group>/...），默认沿用 experiment")
        parser.add_argument("--seed", default=12349, type=int, help="随机种子")
        parser.add_argument("--log_output", default='term_file', choices=['term_file', 'term', 'file'], help="日志输出方式")
        parser.add_argument("--restore", default=False, type=self.str2bool, help="断点续训/恢复模型")
        parser.add_argument("--save_model", default=True, type=self.str2bool, help="是否保存模型")
        parser.add_argument("--gpu", default=0, help="GPU编号", type=int)

        # 配置文件
        parser.add_argument("--config", default=None, help="配置文件路径（yaml/json）")

        # 主要Config参数（可命令行覆盖）
        parser.add_argument("--data_path", default=None, type=str)
        parser.add_argument("--test_ratio", default=0.2, type=float)
        parser.add_argument("--by_time", default=True, type=self.str2bool)
        parser.add_argument("--embedding_dim", default=64, type=int)
        parser.add_argument("--history_len", default=32, type=int)
        parser.add_argument("--device", default="cpu", type=str)
        parser.add_argument("--use_initial_emb", default=False, type=self.str2bool)
        parser.add_argument("--change_num", default=5, type=int)
        parser.add_argument("--n_epochs", default=20, type=int)
        parser.add_argument("--batch_size", default=1024, type=int)
        parser.add_argument("--learning_rate", default=0.01, type=float)
        parser.add_argument("--weight_decay", default=0.01, type=float)
        parser.add_argument("--item_embedding_dim", default=64, type=int)
        parser.add_argument("--state_hidden_dim", default=128, type=int)
        parser.add_argument("--action_emb_dim", default=32, type=int)
        parser.add_argument("--policy_hidden", default="256,256", type=str)
        parser.add_argument("--critic_hidden", default="256,256", type=str)
        parser.add_argument("--rnn_type", default="GRU", type=str)
        parser.add_argument("--gamma", default=0.99, type=float)
        parser.add_argument("--tau", default=0.005, type=float)
        parser.add_argument("--alpha_init", default=0.01, type=float)
        parser.add_argument("--lr", default=1e-3, type=float)
        parser.add_argument("--buffer_size", default=100000, type=int)
        parser.add_argument("--max_seq_len", default=50, type=int)
        parser.add_argument("--sac_batch_size", default=256, type=int)
        parser.add_argument("--cql_alpha", default=5.0, type=float)
        parser.add_argument("--cql_num_random", default=10, type=int)
        parser.add_argument("--cql_with_lagrange", default=False, type=self.str2bool)
        parser.add_argument("--cql_lagrange_thresh", default=5.0, type=float)
        parser.add_argument("--disable_offline_online_buffer", default=False, type=self.str2bool)
        parser.add_argument("--offline_sample_prob", default=0.25, type=float)
        parser.add_argument("--save_q_values", default=False, type=self.str2bool)
        parser.add_argument("--q_output_filename", default="q_values_eval.npy", type=str)

        parser.add_argument("--episodes_per_phase", default=1000, type=int)
        parser.add_argument("--log_interval", default=50, type=int)
        parser.add_argument("--save_interval", default=50, type=int)
        parser.add_argument("--eval_interval", default=20, type=int)
        parser.add_argument("--eval_user_count", default=100, type=int)
        parser.add_argument("--test_user_count", default=5, type=int)

        parser.add_argument("--eval_phase", default=None, type=int, help="评估时强制使用指定阶段的 action_mask（用于旧任务评估）")

        parser.add_argument("--kg_feature_alpha", default=0.25, type=float)
        parser.add_argument("--user_feature_alpha", default=0.15, type=float)
        parser.add_argument("--min_user_interactions", default=5, type=int)

        # 时间戳
        now = datetime.now()
        timestamp = f"{now.month:02d}-{now.day:02d}_{now.hour:02d}-{now.minute:02d}-{now.second:02d}"
        parser.add_argument("--timestamp", default=timestamp, help="实验时间戳")

        self.parser = parser

    def str2bool(self, text):
        if isinstance(text, bool):
            return text
        if text.lower() in ('true', '1', 'yes'):
            return True
        elif text.lower() in ('false', '0', 'no'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    def parse(self):
        args = self.parser.parse_args()

        # 处理元组参数
        args.policy_hidden = tuple(int(x) for x in args.policy_hidden.split(',')) if isinstance(args.policy_hidden, str) else args.policy_hidden
        args.critic_hidden = tuple(int(x) for x in args.critic_hidden.split(',')) if isinstance(args.critic_hidden, str) else args.critic_hidden

        return args

if __name__ == "__main__":
    parser = Parser()
    args = parser.parse()
    print("实验名:", args.experiment)
    print("算法名: CAL-QL")
    print("随机种子:", args.seed)
    print("全部参数:", args)
