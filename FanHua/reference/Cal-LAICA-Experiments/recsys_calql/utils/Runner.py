import os
import os.path
import logging
from time import time
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import random

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')          # 无 GUI 环境下安全渲染
import matplotlib.pyplot as plt
from torch import long, float32, Tensor
from recsys_calql.agent.SAC import RecCalQLAgent
from recsys_calql.environments.recWorld import RecommendationEnv


# ================================================================
# 工具函数：构建实验目录 & 初始化 logger
# ================================================================

def _build_exp_dir(base: str, dataset: str, exp_name: str, seed: int) -> str:
    """
    构建并返回实验目录路径。
    例: ./outputs/ML-1M/CAL-QL/seed_42
    """
    path = os.path.join(base, dataset, exp_name, f"seed_{seed}")
    os.makedirs(path, exist_ok=True)
    return path


def _init_logger(log_path: str, name: str = "runner") -> logging.Logger:
    """
    初始化 Python logging，同时输出到控制台和 .log 文件。
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 避免重复添加 handler（多次实例化时）
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 文件 handler（追加模式）
    fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # 控制台 handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ================================================================
# 工具函数：保存 npy & 绘图
# ================================================================

def _save_npy(exp_dir: str, rewards: list, std_rewards: list):
    """将 rewards / std_rewards 覆盖保存为 npy。"""
    np.save(os.path.join(exp_dir, "rewards.npy"),     np.array(rewards,     dtype=np.float32))
    np.save(os.path.join(exp_dir, "std_rewards.npy"), np.array(std_rewards, dtype=np.float32))


def _plot_curve(
    exp_dir: str,
    rewards: list,
    std_rewards: list,
    phase_boundaries: list,   # [(phase_id, episode_index), ...]
    smooth_window: int = 20,
):
    """
    绘制并覆盖保存 reward_curve.png。
    - 原始曲线（浅色）+ 滑动平均（深色）
    - 阶段切换竖线
    - std 阴影
    """
    if len(rewards) < 2:
        return

    r   = np.array(rewards,     dtype=np.float32)
    std = np.array(std_rewards, dtype=np.float32)
    x   = np.arange(1, len(r) + 1)

    kernel = min(smooth_window, len(r))
    idx = np.arange(1, len(r) + 1)
    start = np.maximum(0, idx - kernel)
    csum = np.cumsum(np.insert(r, 0, 0.0))
    csum_std = np.cumsum(np.insert(std, 0, 0.0))
    w = idx - start
    smooth = (csum[idx] - csum[start]) / w
    std_smooth = (csum_std[idx] - csum_std[start]) / w

    fig, ax = plt.subplots(figsize=(12, 5))

    # 原始曲线（半透明）
    ax.plot(x, r, color='steelblue', alpha=0.25, linewidth=0.8, label='Raw eval')

    ax.fill_between(
        x,
        smooth - std_smooth,
        smooth + std_smooth,
        alpha=0.15, color='steelblue'
    )

    # 滑动平均主线
    ax.plot(x, smooth, color='steelblue', linewidth=1.8,
            label=f'Smooth (w={kernel})')

    # 阶段切换竖线
    for phase_id, ep_idx in phase_boundaries:
        ax.axvline(x=ep_idx, color='tomato', linestyle='--',
                   linewidth=1.0, alpha=0.7)
        ax.text(ep_idx + 1, ax.get_ylim()[1] * 0.95,
                f'P{phase_id}', color='tomato', fontsize=7, va='top')

    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Eval Score', fontsize=11)
    ax.set_title('CAL-QL Training Curve', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(exp_dir, "reward_curve.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ================================================================
# Runner
# ================================================================

class Runner(object):
    def __init__(self, config):
        self.config = config
        self.env: RecommendationEnv = config.env
        self.test_env: Optional[RecommendationEnv] = getattr(config, 'test_env', None)
        self.agent = RecCalQLAgent(
            num_items=config.meta['n_items'],
            item_embedding_dim=config.item_embedding_dim,
            state_hidden_dim=config.state_hidden_dim,
            action_emb_dim=config.action_emb_dim,
            policy_hidden=config.policy_hidden,
            critic_hidden=config.critic_hidden,
            rnn_type=config.rnn_type,
            gamma=config.gamma,
            tau=config.tau,
            alpha_init=config.alpha_init,
            cql_alpha=config.cql_alpha,
            cql_num_random=config.cql_num_random,
            cql_with_lagrange=config.cql_with_lagrange,
            cql_lagrange_thresh=config.cql_lagrange_thresh,
            use_cal_ql=True,
            disable_offline_online_buffer=getattr(config, "disable_offline_online_buffer", False),
            offline_sample_prob=getattr(config, "offline_sample_prob", 0.25),
            lr=config.lr,
            buffer_size=config.buffer_size,
            max_seq_len=config.max_seq_len,
            batch_size=config.sac_batch_size,
            device=config.device,
        )

        # ---- 数据相关 ----
        self.user_subsequences: Dict[int, List[List[dict]]] = {}
        self.all_users:  List[int] = []
        self.test_users: List[int] = []
        self.max_seq_len:     int = getattr(config, 'max_seq_len',      64)
        self.test_user_count: int = getattr(config, 'test_user_count',   5)
        self.n_items:         int = config.meta['n_items']

        # ---- 实验目录 & 日志 ----
        dataset  = getattr(config, 'dataset',  'ML-1M')
        exp_name = getattr(config, 'exp_name', 'Cal-QL')
        seed     = getattr(config, 'seed',     0)
        base_dir = getattr(config, 'log_base_dir', './Environments')

        self.exp_dir = _build_exp_dir(base_dir, dataset, exp_name, seed)
        self.logger  = _init_logger(
            log_path=os.path.join(self.exp_dir, "train.log"),
            name=f"{dataset}_{exp_name}_seed{seed}",
        )

        # ---- 持久化频率 ----
        # 每隔 save_interval 个 episode 保存一次 npy + 图
        self.save_interval: int = getattr(config, 'save_interval', 50)
        self.eval_q_values: list = []
        self.disable_progress: bool = bool(int(os.environ.get("CALQL_DISABLE_TQDM", "1")))

    # ==============================================================
    # 内部：统一日志打印（同时写文件 + 控制台）
    # ==============================================================

    def _log(self, msg: str):
        self.logger.info(msg)

    def _save_q_values(self):
        if not bool(getattr(self.config, "save_q_values", False)):
            return
        q_path = os.path.join(self.exp_dir, str(getattr(self.config, "q_output_filename", "q_values_eval.npy")))
        np.save(q_path, np.asarray(self.eval_q_values, dtype=np.float32))

    # ==============================================================
    # 公用函数：数据准备
    # ==============================================================

    def _prepare_data(self):
        """
        从环境 user_history 切分子序列。
        所有用户都参与 buffer 填充，测试用户单独记录用于评估。
        """
        user_history = self.env.user_history
        max_len = self.max_seq_len

        self.user_subsequences = {}

        for u_idx, history in user_history.items():
            if len(history) < 3:
                continue

            subsequences = []
            for start in range(0, len(history), max_len):
                sub = history[start: start + max_len]
                if len(sub) >= 2:
                    subsequences.append(sub)

            if subsequences:
                self.user_subsequences[u_idx] = subsequences

        self.all_users = list(self.user_subsequences.keys())

        np.random.shuffle(self.all_users)
        self.test_users = self.all_users[:self.test_user_count]

        total_subseqs = sum(len(v) for v in self.user_subsequences.values())

        sep = '=' * 60
        self._log(sep)
        self._log("📊 数据准备完成")
        self._log(f"   总用户数:       {len(self.all_users)}")
        self._log(f"   测试用户池:     {len(self.test_users)}")
        self._log(f"   最大序列长度:   {max_len}")
        self._log(f"   总子序列数:     {total_subseqs}")
        self._log(f"   ⚠️ 所有用户（含测试）均参与 buffer 填充")
        self._log(sep)

    # ==============================================================
    # 离线填充 buffer
    # ==============================================================

    def _fill_buffer_one_subsequence(
        self,
        user_idx: int,
        subsequence: List[dict],
    ) -> Tuple[float, int, dict]:
        if len(subsequence) < 2:
            return 0.0, 0, {}

        WINDOW = 32  # 最长支持的窗口大小

        current_matrix = self.env.get_current_matrix()
        next_matrix    = self.env.get_next_matrix()

        transitions    = []
        episode_reward = 0.0

        for t in range(len(subsequence) - 1):
            # ---- 当前状态序列：取 subsequence[0..t] 的 item_idx，只保留最近 WINDOW 个 ----
            history_items = [s["item_idx"] for s in subsequence[max(0, t + 1 - WINDOW): t + 1]]
            seq           = history_items          # 长度 ∈ [1, WINDOW]
            seq_len       = len(seq)

            # ---- 动作 ----
            action_item = subsequence[t + 1]["item_idx"]
            action_emb  = self.agent.action_vae.get_action_embedding(action_item)

            # ---- 奖励 ----
            reward    = current_matrix[user_idx, action_item].item()
            mc_reward = next_matrix[user_idx, action_item].item()

            # ---- 下一状态序列：取 subsequence[0..t+1] 的 item_idx，只保留最近 WINDOW 个 ----
            next_history = [s["item_idx"] for s in subsequence[max(0, t + 2 - WINDOW): t + 2]]
            next_seq     = next_history            # 长度 ∈ [1, WINDOW]
            next_seq_len = len(next_seq)

            # ---- done：子序列结束 或 每 WINDOW 步截断一次 ----
            is_last_step   = (t + 1 == len(subsequence) - 1)
            is_window_edge = ((t + 1) % WINDOW == 0)
            done = is_last_step or is_window_edge

            transitions.append({
                "item_seq":      seq,
                "seq_len":       seq_len,
                "action":        action_item,
                "action_emb":    action_emb,
                "reward":        reward,
                "mc_reward":     mc_reward,
                "next_item_seq": next_seq,
                "next_seq_len":  next_seq_len,
                "done":          done,
            })
            episode_reward += reward

        # ---- MC return：在每个 done 边界内独立计算 ----
        gamma     = self.agent.gamma
        mc_return = 0.0
        for i in reversed(range(len(transitions))):
            if transitions[i]["done"]:
                mc_return = 0.0                    # 遇到 done 则重置
            mc_return = transitions[i]["mc_reward"] + gamma * mc_return
            transitions[i]["mc_return"] = mc_return

        # ---- 写入 replay buffer ----
        for tr in transitions:
            self.agent.memory.add(
                item_seq      = tr["item_seq"],
                seq_len       = tr["seq_len"],
                action        = tr["action"],
                action_emb    = tr["action_emb"],
                reward        = tr["reward"],
                next_item_seq = tr["next_item_seq"],
                next_seq_len  = tr["next_seq_len"],
                done          = tr["done"],
                mc_return     = tr["mc_return"],
            )

        # ---- 触发一次 agent 更新 ----
        update_info = self.agent.step(
            item_seq      = tr["item_seq"],
            seq_len       = tr["seq_len"],
            action        = tr["action"],
            action_emb    = tr["action_emb"],
            reward        = tr["reward"],
            next_item_seq = tr["next_item_seq"],
            next_seq_len  = tr["next_seq_len"],
            done          = tr["done"],
            mc_return     = tr["mc_return"],
        )
        return episode_reward, len(transitions), update_info

    # ==============================================================
    # 每回合评估
    # ==============================================================

    def _evaluate_episode(self, first_phase = True) -> Tuple[float, float, float]:
        eval_user_count = int(getattr(self.config, "eval_user_count", 100))
        candidates = [u for u in self.all_users if u in self.user_subsequences]

        if len(candidates) < eval_user_count:
            sampled_users = candidates
        else:
            sampled_users = np.random.choice(
                candidates, size=eval_user_count, replace=False
            ).tolist()

        device = self.agent.device
        max_len = self.max_seq_len

        user_total_scores = []
        q_values = []

        for user_idx in sampled_users:
            obs, _ = self.env.reset(user_idx=user_idx)

            raw_items = obs.history_items
            if len(raw_items) == 0:
                if user_idx in self.user_subsequences:
                    seed_seq = self.user_subsequences[user_idx][0]
                    item_seq = [s["item_idx"] for s in seed_seq[:3]]
                else:
                    item_seq = [0]
            else:
                item_seq = raw_items[-max_len:]

            seq_len = len(item_seq)
            episode_reward = 0.0
            done = False

            while not done:
                item_seq_tensor = torch.tensor(item_seq, dtype=torch.long, device=device)
                action_id, action_emb = self.agent.get_action(item_seq_tensor, seq_length=seq_len, deterministic=True)
                if bool(getattr(self.config, "save_q_values", False)) and hasattr(self.agent, "evaluate_q"):
                    q_values.append(self.agent.evaluate_q(item_seq_tensor, seq_len, action_emb))
                next_obs, reward, done, info, _ = self.env.step(action_id)

                next_raw = next_obs.history_items
                if len(next_raw) == 0:
                    next_item_seq = item_seq
                else:
                    next_item_seq = next_raw[-max_len:]
                next_seq_len = len(next_item_seq)

                if not first_phase:
                    self.agent.memory.add(
                        item_seq,
                        seq_len,
                        action_id,
                        action_emb,
                        reward,
                        next_item_seq,
                        next_seq_len,
                        done,
                        mc_return=info.get('mc_return', 0.0)
                    )

                episode_reward += reward
                item_seq = next_item_seq
                seq_len = next_seq_len

            user_total_scores.append(episode_reward)

        if not user_total_scores:
            return 0.0, 0.0, 0.0

        avg_q = float(np.mean(q_values)) if len(q_values) > 0 else 0.0
        return float(np.max(user_total_scores)), float(np.std(user_total_scores)), avg_q

    # ==============================================================
    # 🚀 CAL-QL 主训练循环
    # ==============================================================

    def Cal_QL_Run(self):
        self._prepare_data()

        change_num   = self.env.change_num
        log_interval = getattr(self.config, 'log_interval', 50)
        rewards:          list = []
        std_rewards:      list = []
        phase_boundaries: list = []

        episodes_per_phase = getattr(self.config, 'episodes_per_phase', 1000)

        global_episode = 0
        global_step    = 0
        t0             = time()

        sep70 = '=' * 70
        self._log(sep70)
        self._log("🚀 CAL-QL 训练开始")
        self._log(f"   阶段数: {change_num} | 每阶段正式训练回合: {episodes_per_phase}")
        self._log(f"   实验目录: {self.exp_dir}")
        self._log(sep70)

        for phase in range(1, change_num + 1):

            # ── 阶段切换 ──────────────────────────────────────────────
            mask = self.env.update_mask(update_mask=True)
            self.agent.update_mask(mask)
            first_phase = True if phase == 1 else False
            if phase > 1:
                self.agent.update_memory()          # 重置 memory，重新预热
                self.agent.initial_phase = True     # 重置 VAE 初始化标志

            unlocked       = self.env.reward_mask.sum().item()
            current_matrix = self.env.get_current_matrix()
            active_count   = current_matrix[current_matrix != 0].numel()

            phase_boundaries.append((phase, global_episode + 1))

            sep65 = '─' * 65
            self._log(sep65)
            self._log(
                f"🔓 阶段 {phase}/{change_num} | "
                f"已解锁: {int(unlocked)}/{self.n_items} "
                f"({unlocked / self.n_items:.1%})"
            )
            self._log(f"🔄 [矩阵更新] 有效评分总数: {active_count}")
            self._log(sep65)

            # ══════════════════════════════════════════════════════════
            # 阶段一：全量遍历预热
            # 遍历所有用户的所有子序列，填 buffer 并尝试触发 VAE 初始化
            # ══════════════════════════════════════════════════════════
            self._log("📦 [预热] 开始全量遍历填充 buffer ...")

            all_subseqs = []
            for user_idx in self.all_users:
                if user_idx not in self.user_subsequences:
                    continue
                for subseq in self.user_subsequences[user_idx]:
                    all_subseqs.append((user_idx, subseq))
            np.random.shuffle(all_subseqs)

            self._log(f"   全量子序列数: {len(all_subseqs)}")

            for user_idx, subseq in tqdm(all_subseqs, desc="Processing subsequences", disable=self.disable_progress):
                global_step += 1
                if random.random() < 0.5:
                    continue
                # 填 buffer
                _, _, _ = self._fill_buffer_one_subsequence(user_idx, subseq)
                if not self.agent.initial_phase and not first_phase:
                    break

            if not first_phase:
                self.agent.clear_memory()

                # # 判断是否满足 VAE 初始化条件
                # if self.agent.initial_phase:
                #     if self.agent.memory.size >= self.agent.memory.max_size // 2:
                #         self._log(
                #             f"   ✅ VAE 初始化条件满足 "
                #             f"(buffer={self.agent.memory.size}) → 开始训练 VAE ..."
                #         )
                #         self.agent.initial_vae_training(max_epochs=100, batch_size=32, lr=1e-3)
                #         self.agent.initial_phase = False
                #         self._log("   ✅ VAE 初始化完成，开始更新 policy/critic")
                #     # 未满足：继续填，不更新
                # else:
                #     # VAE 已初始化：正常更新
                #     self.agent.update()

            # ══════════════════════════════════════════════════════════
            # 阶段二（补救）：若全量遍历后 VAE 仍未初始化
            # 随机抽用户继续填，直到 VAE 完成初始化
            # ══════════════════════════════════════════════════════════
            # if self.agent.initial_phase:
            #     self._log("⚠️  全量遍历后 VAE 仍未初始化，随机补充填充 ...")
            #     candidates = [u for u in self.all_users if u in self.user_subsequences]

            #     while self.agent.initial_phase:
            #         user_idx = np.random.choice(candidates)
            #         subseqs  = self.user_subsequences[user_idx]
            #         subseq   = subseqs[np.random.randint(len(subseqs))]

            #         global_step += 1
            #         self._fill_buffer_one_subsequence(user_idx, subseq)

            #         if self.agent.memory.size >= self.agent.memory.max_size // 2:
            #             self._log(
            #                 f"   ✅ 补充填充后 VAE 初始化条件满足 "
            #                 f"(buffer={self.agent.memory.size}) → 开始训练 VAE ..."
            #             )
            #             self.agent.initial_vae_training(max_epochs=100, batch_size=32, lr=1e-3)
            #             self.agent.initial_phase = False
            #             self._log("   ✅ VAE 初始化完成")

            # ══════════════════════════════════════════════════════════
            # 阶段三：正式训练
            # 每回合：随机抽用户 → 填 buffer → update → evaluate
            # ══════════════════════════════════════════════════════════
            self._log(f"🎯 [正式训练] 开始，共 {episodes_per_phase} 回合 ...")

            candidates    = [u for u in self.all_users if u in self.user_subsequences]
            phase_rewards = []

            for ep in range(1, episodes_per_phase + 1):
                global_episode += 1

                # 随机抽一个用户的一条真实子序列
                user_idx = np.random.choice(candidates)
                subseqs  = self.user_subsequences[user_idx]
                subseq   = subseqs[np.random.randint(len(subseqs))]

                # 填 buffer + 更新参数
                train_reward, n_steps, update_info = \
                    self._fill_buffer_one_subsequence(user_idx, subseq)
                global_step += n_steps

                self.agent.update()

                # 评估
                eval_mean, eval_std, eval_q = self._evaluate_episode(first_phase=first_phase)
                rewards.append(eval_mean)
                std_rewards.append(eval_std)
                phase_rewards.append(eval_mean)
                if bool(getattr(self.config, "save_q_values", False)):
                    self.eval_q_values.append(float(eval_q))

                # 定期保存
                if global_episode % self.save_interval == 0:
                    _save_npy(self.exp_dir, rewards, std_rewards)
                    self._save_q_values()
                    _plot_curve(
                        self.exp_dir, rewards, std_rewards,
                        phase_boundaries,
                        smooth_window=max(1, log_interval // 5),
                    )

                # 定期日志
                if ep % log_interval == 0:
                    recent_eval = np.mean(phase_rewards[-log_interval:])
                    recent_std  = np.mean(std_rewards[-log_interval:])
                    elapsed     = time() - t0
                    self._log(
                        f"Phase {phase}/{change_num} | "
                        f"Ep {ep:4d}/{episodes_per_phase} "
                        f"(Global {global_episode:5d}) | "
                        f"Eval(recent {log_interval}): "
                        f"{recent_eval:.3f} ± {recent_std:.3f} | "
                        f"Buffer: {self.agent.memory.size:5d} | "
                        f"Steps: {global_step:6d} | "
                        f"{elapsed:.0f}s"
                    )

            # ── 阶段结束 ──────────────────────────────────────────────
            phase_mean = np.mean(phase_rewards) if phase_rewards else 0.0
            phase_std  = np.std(phase_rewards)  if phase_rewards else 0.0
            self._log(
                f"✅ 阶段 {phase} 完成 | "
                f"正式训练回合: {len(phase_rewards)} | "
                f"平均评估: {phase_mean:.3f} ± {phase_std:.3f}"
            )

            _save_npy(self.exp_dir, rewards, std_rewards)
            self._save_q_values()
            _plot_curve(
                self.exp_dir, rewards, std_rewards,
                phase_boundaries,
                smooth_window=max(1, log_interval // 5),
            )

        # ── 训练结束 ──────────────────────────────────────────────────
        _save_npy(self.exp_dir, rewards, std_rewards)
        self._save_q_values()
        _plot_curve(
            self.exp_dir, rewards, std_rewards,
            phase_boundaries,
            smooth_window=max(1, log_interval // 5),
        )

        self._log(sep70)
        self._log("🏁 训练完成")
        self._log(f"   全局步数: {global_step} | 评估次数: {global_episode}")
        self._log(
            f"   最终评估(最近100): "
            f"{np.mean(rewards[-100:]):.3f} ± {np.mean(std_rewards[-100:]):.3f}"
        ) if rewards else None
        self._log(f"   历史最佳: {max(rewards):.3f}") if rewards else None
        self._log(f"   总耗时: {time() - t0:.1f}s")
        self._log(f"   结果保存至: {self.exp_dir}")
        self._log(sep70)

        return rewards, std_rewards

