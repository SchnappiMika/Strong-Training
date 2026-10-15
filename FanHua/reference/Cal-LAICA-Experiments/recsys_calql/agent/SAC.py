import copy
import os
from typing import Tuple, Any, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor, float32, long
from torch.distributions import Normal

from recsys_calql.agent.VAE import ActionVAE
from recsys_calql.agent.GRU import SequenceStateEncoder

# ============================================================
# 工具函数：构建 MLP
# ============================================================
def build_net(layer_shape, hidden_activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for i in range(len(layer_shape) - 1):
        layers.append(nn.Linear(layer_shape[i], layer_shape[i + 1]))
        if i < len(layer_shape) - 2:
            layers.append(hidden_activation())
        else:
            layers.append(output_activation())
    return nn.Sequential(*layers)


def soft_update(target_net: nn.Module, source_net: nn.Module, tau: float):
    for tp, sp in zip(target_net.parameters(), source_net.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


# ============================================================
# SAC Policy Network（连续动作空间 → action embedding）
# ============================================================
class PolicyNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_shape: Tuple[int, ...] = (256, 256),
    ):
        super().__init__()
        layers = [state_dim] + list(hidden_shape)
        self.net = build_net(layers, hidden_activation=nn.ReLU, output_activation=nn.ReLU)
        self.mu_layer = nn.Linear(layers[-1], action_dim)
        self.log_std_layer = nn.Linear(layers[-1], action_dim)
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def forward(
        self, state: Tensor, deterministic: bool = False, return_log_prob: bool = True
    ) -> Tuple[Tensor, Optional[Tensor]]:
        h = self.net(state)
        mu = self.mu_layer(h)
        log_std = self.log_std_layer(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()

        dist = Normal(mu, std)
        u = mu if deterministic else dist.rsample()
        action = torch.tanh(u)

        if return_log_prob:
            # tanh 压缩后的 log_prob 修正
            log_prob = dist.log_prob(u).sum(dim=-1, keepdim=True) \
                       - (2.0 * (np.log(2) - u - F.softplus(-2.0 * u))).sum(dim=-1, keepdim=True)
        else:
            log_prob = None

        return action, log_prob


# ============================================================
# Double Critic Network
# ============================================================
class DoubleCriticNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_shape: Tuple[int, ...] = (256, 256),
    ):
        super().__init__()
        layers = [state_dim + action_dim] + list(hidden_shape) + [1]
        self.critic1 = build_net(layers, hidden_activation=nn.ReLU, output_activation=nn.Identity)
        self.critic2 = build_net(layers, hidden_activation=nn.ReLU, output_activation=nn.Identity)

    def forward(self, state: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        sa = torch.cat([state, action], dim=-1)
        return self.critic1(sa), self.critic2(sa)
    



# ============================================================
# 经验回放
# ============================================================
class ReplayBuffer:
    """适配序列状态的经验回放"""
    def __init__(self, max_size: int, state_dim: int, action_emb_dim: int, max_seq_len: int, device="cpu"):
        self.max_size = max_size
        self.device = device
        self.ptr = 0
        self.size = 0

        self.item_seqs = torch.zeros((max_size, max_seq_len), dtype=long)
        self.seq_lengths = torch.zeros(max_size, dtype=long)
        self.actions = torch.zeros(max_size, dtype=long)
        self.action_embs = torch.zeros((max_size, action_emb_dim), dtype=float32)
        self.rewards = torch.zeros((max_size, 1), dtype=float32)
        self.next_item_seqs = torch.zeros((max_size, max_seq_len), dtype=long)
        self.next_seq_lengths = torch.zeros(max_size, dtype=long)
        self.dones = torch.zeros((max_size, 1), dtype=float32)
        self.mc_returns = torch.zeros((max_size, 1), dtype=float32)

    def add(self, item_seq, seq_len, action, action_emb, reward, next_item_seq, next_seq_len, done, mc_return = 0.0):
        i = self.ptr
        self.item_seqs[i, :len(item_seq)] = torch.tensor(item_seq, dtype=long) if not isinstance(item_seq, Tensor) else item_seq
        self.seq_lengths[i] = seq_len
        self.actions[i] = action
        self.action_embs[i] = action_emb.detach().cpu() if isinstance(action_emb, Tensor) else torch.tensor(action_emb)
        self.rewards[i] = reward
        self.next_item_seqs[i, :len(next_item_seq)] = torch.tensor(next_item_seq, dtype=long) if not isinstance(next_item_seq, Tensor) else next_item_seq
        self.next_seq_lengths[i] = next_seq_len
        self.dones[i] = float(done)
        self.mc_returns[i] = mc_return

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.item_seqs[idx].to(self.device),
            self.seq_lengths[idx].to(self.device),
            self.actions[idx].to(self.device),
            self.action_embs[idx].to(self.device),
            self.rewards[idx].to(self.device),
            self.next_item_seqs[idx].to(self.device),
            self.next_seq_lengths[idx].to(self.device),
            self.dones[idx].to(self.device),
            self.mc_returns[idx].to(self.device)
        )

    def reset(self):
        self.ptr = 0
        self.size = 0


# ============================================================
# RecSAC Agent：序列状态 + SAC + ActionVAE
# ============================================================
class RecSACAgent(nn.Module):
    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int = 64,
        state_hidden_dim: int = 128,
        action_emb_dim: int = 32,
        policy_hidden: Tuple[int, ...] = (256, 256),
        critic_hidden: Tuple[int, ...] = (256, 256),
        rnn_type: str = "GRU",
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha_init: float = 0.01,
        lr: float = 1e-3,
        buffer_size: int = 100000,
        max_seq_len: int = 50,
        batch_size: int = 256,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.num_items = num_items

        # ---------- 1. 序列状态编码器 ----------
        self.state_encoder = SequenceStateEncoder(
            num_items=num_items,
            embedding_dim=item_embedding_dim,
            hidden_dim=state_hidden_dim,
            rnn_type=rnn_type,
        ).to(device)
        state_dim = self.state_encoder.output_dim  # = state_hidden_dim

        # ---------- 2. Action VAE ----------
        self.action_vae = ActionVAE(
            num_items=num_items,
            embedding_dim=action_emb_dim,
            state_dim=state_dim,
            device=device,
        ).to(device)

        # ---------- 3. SAC 网络 ----------
        self.policy = PolicyNetwork(
            state_dim=state_dim,
            action_dim=action_emb_dim,
            hidden_shape=policy_hidden,
        ).to(device)

        self.critic = DoubleCriticNetwork(
            state_dim=state_dim,
            action_dim=action_emb_dim,
            hidden_shape=critic_hidden,
        ).to(device)

        self.target_critic = DoubleCriticNetwork(
            state_dim=state_dim,
            action_dim=action_emb_dim,
            hidden_shape=critic_hidden,
        ).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_critic.parameters():
            p.requires_grad = False

        # ---------- 4. 自动温度调节 alpha ----------
        self.target_entropy = -float(action_emb_dim)
        self.log_alpha = torch.tensor(
            np.log(alpha_init), dtype=float32, requires_grad=True, device=device
        )
        self.alpha = self.log_alpha.exp().detach()

        # ---------- 5. 优化器 ----------
        self.encoder_optimizer = optim.AdamW(self.state_encoder.parameters(), lr=lr)
        self.vae_optimizer = optim.AdamW(self.action_vae.parameters(), lr=lr)
        self.policy_optimizer = optim.AdamW(self.policy.parameters(), lr=lr)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=lr)
        self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=lr)

        # ---------- 6. 经验回放 ----------
        self.memory = ReplayBuffer(
            max_size=buffer_size,
            state_dim=state_dim,
            action_emb_dim=action_emb_dim,
            max_seq_len=max_seq_len,
            device=device,
        )

    # ----------------------------------------------------------
    # 编码状态：item 序列 → 向量
    # ----------------------------------------------------------
    def encode_state(self, item_seq: Tensor, seq_lengths: Tensor = None) -> Tensor:
        return self.state_encoder(item_seq, seq_lengths)

    # ----------------------------------------------------------
    # 选动作：核心推理流程
    # ----------------------------------------------------------
    @torch.no_grad()
    def get_action(
        self,
        item_seq: Tensor,
        seq_length: Optional[int] = None,
        deterministic: bool = False,
        explore_eps: float = 0.1,
    ) -> Tuple[int, Tensor]:
        """
        输入用户历史 item 序列，输出推荐的 item_id 和 action_embedding。

        流程：
          item_seq → GRU → state_vector → Policy → action_embedding → VAE.get_best_match → item_id
        """
        # 确保 tensor 格式
        if not isinstance(item_seq, Tensor):
            item_seq = torch.tensor(item_seq, dtype=long, device=self.device)
        if item_seq.dim() == 1:
            item_seq = item_seq.unsqueeze(0)  # (1, seq_len)

        seq_len_tensor = None
        if seq_length is not None:
            seq_len_tensor = torch.tensor([seq_length], dtype=long, device=self.device)

        # Step 1: 序列 → 状态向量
        state = self.encode_state(item_seq, seq_len_tensor)  # (1, state_dim)

        # Step 2: 探索 or 利用
        if not deterministic and np.random.rand() < explore_eps:
            # 随机探索：从可用 item 中随机选
            valid_ids = torch.where(self.action_vae.action_mask)[0] \
                if self.action_vae.action_mask is not None \
                else torch.arange(self.num_items)
            action_id = valid_ids[np.random.randint(len(valid_ids))].item()
            action_emb = self.action_vae.get_action_embedding(action_id)
        else:
            # Policy 输出连续 action embedding
            action_emb, _ = self.policy(state, deterministic=deterministic, return_log_prob=False)
            # Step 3: 通过 VAE 找到最近的真实 item
            action_id = self.action_vae.get_best_match(action_emb).item()
            action_emb = action_emb.squeeze(0)

        return action_id, action_emb

    # ----------------------------------------------------------
    # 存储经验
    # ----------------------------------------------------------
    def store_transition(self, item_seq, seq_len, action, action_emb,
                         reward, next_item_seq, next_seq_len, done):
        self.memory.add(item_seq, seq_len, action, action_emb,
                        reward, next_item_seq, next_seq_len, done)

    # ----------------------------------------------------------
    # SAC 更新
    # ----------------------------------------------------------
    def update(self):
        if self.memory.size < self.batch_size:
            return {}

        # ---- 采样 ----
        (item_seqs, seq_lens, actions, action_embs,
         rewards, next_item_seqs, next_seq_lens, dones) = self.memory.sample(self.batch_size)

        # ---- 编码状态 ----
        states = self.encode_state(item_seqs, seq_lens)          # (B, state_dim)
        next_states = self.encode_state(next_item_seqs, next_seq_lens)  # (B, state_dim)

        # ============ Critic 更新 ============
        with torch.no_grad():
            next_action_emb, next_log_prob = self.policy(next_states, deterministic=False, return_log_prob=True)
            target_q1, target_q2 = self.target_critic(next_states, next_action_emb)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1.0 - dones) * self.gamma * (target_q - self.alpha * next_log_prob)

        current_q1, current_q2 = self.critic(states, action_embs)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()  # 状态编码器梯度也通过 critic 回传
        critic_loss.backward()
        self.critic_optimizer.step()
        self.encoder_optimizer.step()

        # ============ Policy 更新 ============
        # 重新编码（因为 encoder 刚更新过）
        states_detached = self.encode_state(item_seqs, seq_lens)

        pred_action_emb, log_prob = self.policy(states_detached, deterministic=False, return_log_prob=True)
        q1, q2 = self.critic(states_detached, pred_action_emb)
        min_q = torch.min(q1, q2)

        policy_loss = (self.alpha * log_prob - min_q).mean()

        # Action VAE 自监督损失（联合训练）
        vae_loss = self.action_vae.unsupervised_loss(states_detached.detach(), actions, 
                                                      self.encode_state(next_item_seqs, next_seq_lens).detach())
        total_policy_loss = policy_loss + vae_loss

        self.policy_optimizer.zero_grad()
        self.vae_optimizer.zero_grad()
        total_policy_loss.backward()
        self.policy_optimizer.step()
        self.vae_optimizer.step()

        # ============ Alpha 更新 ============
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # ============ 软更新 Target Critic ============
        soft_update(self.target_critic, self.critic, self.tau)

        return {
            "critic_loss": critic_loss.item(),
            "policy_loss": policy_loss.item(),
            "vae_loss": vae_loss.item(),
            "alpha": self.alpha.item(),
        }

    def update_mask(self, reward_mask):
        self.action_vae.update_action_mask(reward_mask)
    
    def update_memory(self):
        self.memory.reset()

class RecCQLAgent(nn.Module):
    """
    CQL = SAC + Conservative Q-Learning 正则项

    CQL 正则项的直觉：
      - 对随机采样的 OOD 动作，Q 值应该低（logsumexp 项惩罚高 Q）
      - 对数据集中实际出现的动作，Q 值应该高（减去 dataset Q 项）
      - 净效果：Q 函数对未见过的动作保守估计，避免离线策略过度乐观
    """

    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int = 64,
        state_hidden_dim: int = 128,
        action_emb_dim: int = 32,
        policy_hidden: Tuple[int, ...] = (256, 256),
        critic_hidden: Tuple[int, ...] = (256, 256),
        rnn_type: str = "GRU",
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha_init: float = 0.01,
        # ---- CQL 专属参数 ----
        cql_alpha: float = 5.0,           # CQL 正则项权重
        cql_num_random: int = 10,         # 每个样本采多少随机动作
        cql_with_lagrange: bool = False,  # 是否用 Lagrange 自动调节 cql_alpha
        cql_lagrange_thresh: float = 5.0, # Lagrange 阈值
        # ----------------------
        lr: float = 1e-3,
        buffer_size: int = 100000,
        max_seq_len: int = 50,
        batch_size: int = 256,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.num_items = num_items
        self.action_emb_dim = action_emb_dim

        # CQL 参数
        self.cql_alpha = cql_alpha
        self.cql_num_random = cql_num_random
        self.cql_with_lagrange = cql_with_lagrange
        self.cql_lagrange_thresh = cql_lagrange_thresh

        # 如果使用 Lagrange，cql_alpha 变成可学习参数
        if self.cql_with_lagrange:
            self.log_cql_alpha = torch.tensor(
                np.log(cql_alpha), dtype=float32, requires_grad=True, device=device
            )
            self.cql_alpha_optimizer = optim.AdamW([self.log_cql_alpha], lr=lr)

        # ---------- 序列状态编码器 ----------
        self.state_encoder = SequenceStateEncoder(
            num_items=num_items,
            embedding_dim=item_embedding_dim,
            hidden_dim=state_hidden_dim,
            rnn_type=rnn_type,
        ).to(device)
        state_dim = self.state_encoder.output_dim

        # ---------- Action VAE ----------
        self.action_vae = ActionVAE(
            num_items=num_items,
            embedding_dim=action_emb_dim,
            state_dim=state_dim,
            device=device,
        ).to(device)

        # ---------- SAC 网络 ----------
        self.policy = PolicyNetwork(state_dim, action_emb_dim, policy_hidden).to(device)
        self.critic = DoubleCriticNetwork(state_dim, action_emb_dim, critic_hidden).to(device)
        self.target_critic = DoubleCriticNetwork(state_dim, action_emb_dim, critic_hidden).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_critic.parameters():
            p.requires_grad = False

        # ---------- SAC 温度 alpha ----------
        self.target_entropy = -float(action_emb_dim)
        self.log_alpha = torch.tensor(np.log(alpha_init), dtype=float32, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().detach()

        # ---------- 优化器 ----------
        self.encoder_optimizer = optim.AdamW(self.state_encoder.parameters(), lr=lr)
        self.vae_optimizer = optim.AdamW(self.action_vae.parameters(), lr=lr)
        self.policy_optimizer = optim.AdamW(self.policy.parameters(), lr=lr)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=lr)
        self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=lr)

        # ---------- 经验回放 ----------
        self.memory = ReplayBuffer(
            max_size=buffer_size,
            action_emb_dim=action_emb_dim,
            max_seq_len=max_seq_len,
            device=device,
        )

    # ----------------------------------------------------------
    def encode_state(self, item_seq: Tensor, seq_lengths: Tensor = None) -> Tensor:
        return self.state_encoder(item_seq, seq_lengths)

    # ----------------------------------------------------------
    @torch.no_grad()
    def get_action(self, item_seq, seq_length=None, deterministic=False, explore_eps=0.1):
        if not isinstance(item_seq, Tensor):
            item_seq = torch.tensor(item_seq, dtype=long, device=self.device)
        if item_seq.dim() == 1:
            item_seq = item_seq.unsqueeze(0)

        seq_len_tensor = torch.tensor([seq_length], dtype=long, device=self.device) if seq_length is not None else None
        state = self.encode_state(item_seq, seq_len_tensor)

        if not deterministic and np.random.rand() < explore_eps:
            valid_ids = torch.where(self.action_vae.action_mask)[0] \
                if self.action_vae.action_mask is not None \
                else torch.arange(self.num_items)
            action_id = valid_ids[np.random.randint(len(valid_ids))].item()
            action_emb = self.action_vae.get_action_embedding(action_id)
        else:
            action_emb, _ = self.policy(state, deterministic=deterministic, return_log_prob=False)
            action_id = self.action_vae.get_best_match(action_emb).item()
            action_emb = action_emb.squeeze(0)

        return action_id, action_emb

    # ----------------------------------------------------------
    def store_transition(self, item_seq, seq_len, action, action_emb,
                         reward, next_item_seq, next_seq_len, done):
        self.memory.add(item_seq, seq_len, action, action_emb,
                        reward, next_item_seq, next_seq_len, done)

    def update_mask(self, mask):
        self.action_vae.update_action_mask(mask)
    
    def update_memory(self):
        self.memory.reset()

    # ----------------------------------------------------------
    # 🔑 CQL 核心：计算保守正则项
    # ----------------------------------------------------------
    def _compute_cql_regularizer(
        self, states: Tensor, next_states: Tensor, action_embs: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        CQL 正则项 = logsumexp(Q(s, a_sampled)) - E_{a~D}[Q(s, a_data)]

        采样三类动作来估计 logsumexp：
          1. 均匀随机动作（探索 OOD 区域）
          2. 当前策略在 s 上采样的动作
          3. 当前策略在 s' 上采样的动作
        """
        B = states.shape[0]
        N = self.cql_num_random  # 每个样本的随机采样数

        # ---- 1) 均匀随机动作 ----
        # action embedding 经过 tanh，范围在 [-1, 1]
        random_actions = torch.rand(B * N, self.action_emb_dim, device=self.device) * 2.0 - 1.0
        random_log_prob = np.log(0.5 ** self.action_emb_dim)  # 均匀分布的 log_prob（常数）

        # ---- 2) 当前策略采样的动作 ----
        # 将 states 重复 N 次: (B, D) → (B*N, D)
        states_repeated = states.unsqueeze(1).repeat(1, N, 1).view(B * N, -1)
        next_states_repeated = next_states.unsqueeze(1).repeat(1, N, 1).view(B * N, -1)

        curr_actions, curr_log_probs = self.policy(states_repeated, deterministic=False, return_log_prob=True)
        next_actions, next_log_probs = self.policy(next_states_repeated, deterministic=False, return_log_prob=True)

        # ---- 3) 计算所有采样动作的 Q 值 ----
        # Q(s, a_random)
        q1_random, q2_random = self.critic(states_repeated, random_actions)
        # Q(s, a_curr_policy)
        q1_curr, q2_curr = self.critic(states_repeated, curr_actions)
        # Q(s, a_next_policy)  注意这里仍然用 states_repeated 而非 next_states
        q1_next, q2_next = self.critic(states_repeated, next_actions)

        # ---- 4) 拼接并减去 importance weight (log_prob) ----
        # reshape 回 (B, N, 1)
        q1_random = q1_random.view(B, N, 1)
        q2_random = q2_random.view(B, N, 1)
        q1_curr = q1_curr.view(B, N, 1)
        q2_curr = q2_curr.view(B, N, 1)
        q1_next = q1_next.view(B, N, 1)
        q2_next = q2_next.view(B, N, 1)

        curr_log_probs = curr_log_probs.view(B, N, 1)
        next_log_probs = next_log_probs.view(B, N, 1)

        # 拼接: (B, 3*N, 1)
        # 减去 log_prob 做 importance sampling 修正
        q1_cat = torch.cat([
            q1_random - random_log_prob,       # 均匀分布
            q1_curr - curr_log_probs.detach(), # 当前策略
            q1_next - next_log_probs.detach(), # 下一步策略
        ], dim=1)

        q2_cat = torch.cat([
            q2_random - random_log_prob,
            q2_curr - curr_log_probs.detach(),
            q2_next - next_log_probs.detach(),
        ], dim=1)

        # ---- 5) logsumexp 估计 ----
        # logsumexp(Q) 近似 log E_a[exp(Q(s,a))]
        cql_q1_ood = torch.logsumexp(q1_cat.squeeze(-1), dim=1).mean()
        cql_q2_ood = torch.logsumexp(q2_cat.squeeze(-1), dim=1).mean()

        # ---- 6) 数据集内动作的 Q 值 ----
        cql_q1_data, cql_q2_data = self.critic(states, action_embs)
        cql_q1_data = cql_q1_data.mean()
        cql_q2_data = cql_q2_data.mean()

        # ---- 7) CQL 正则 = OOD 高 Q 惩罚 - 数据内 Q 鼓励 ----
        cql_loss1 = cql_q1_ood - cql_q1_data
        cql_loss2 = cql_q2_ood - cql_q2_data

        return cql_loss1, cql_loss2

    # ----------------------------------------------------------
    # 🔄 CQL 完整更新步
    # ----------------------------------------------------------
    def update(self) -> Dict[str, float]:
        if self.memory.size < self.batch_size:
            return {}

        # ---- 采样 ----
        (item_seqs, seq_lens, actions, action_embs,
         rewards, next_item_seqs, next_seq_lens, dones) = self.memory.sample(self.batch_size)

        # ---- 编码状态 ----
        states = self.encode_state(item_seqs, seq_lens)
        next_states = self.encode_state(next_item_seqs, next_seq_lens)

        # ================================================================
        # 📌 CRITIC 更新 = 标准 SAC TD loss + CQL 正则项
        # ================================================================

        # ---- 标准 SAC TD target ----
        with torch.no_grad():
            next_action_emb, next_log_prob = self.policy(next_states, deterministic=False, return_log_prob=True)
            target_q1, target_q2 = self.target_critic(next_states, next_action_emb)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1.0 - dones) * self.gamma * (target_q - self.alpha * next_log_prob)

        current_q1, current_q2 = self.critic(states, action_embs)
        td_loss1 = F.mse_loss(current_q1, target_q)
        td_loss2 = F.mse_loss(current_q2, target_q)
        td_loss = td_loss1 + td_loss2

        # ---- CQL 正则项 ----
        cql_loss1, cql_loss2 = self._compute_cql_regularizer(
            states.detach(), next_states.detach(), action_embs
        )

        # 获取当前 cql_alpha（可能是固定值或可学习参数）
        if self.cql_with_lagrange:
            current_cql_alpha = self.log_cql_alpha.exp().clamp(min=0.0)
        else:
            current_cql_alpha = self.cql_alpha

        # 总 critic loss
        critic_loss = td_loss + current_cql_alpha * (cql_loss1 + cql_loss2)

        self.critic_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.state_encoder.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        self.encoder_optimizer.step()

        # ---- Lagrange 更新 cql_alpha（可选）----
        cql_alpha_loss = 0.0
        if self.cql_with_lagrange:
            cql_alpha_loss = -current_cql_alpha * (
                (cql_loss1 + cql_loss2).detach() / 2.0 - self.cql_lagrange_thresh
            )
            self.cql_alpha_optimizer.zero_grad()
            cql_alpha_loss.backward()
            self.cql_alpha_optimizer.step()
            self.cql_alpha = self.log_cql_alpha.exp().item()

        # ================================================================
        # 📌 POLICY 更新 = SAC 策略损失 + VAE 自监督损失
        # ================================================================
        # 重新编码（encoder 刚更新过）
        states_new = self.encode_state(item_seqs, seq_lens)
        next_states_new = self.encode_state(next_item_seqs, next_seq_lens)

        pred_action_emb, log_prob = self.policy(states_new, deterministic=False, return_log_prob=True)
        q1, q2 = self.critic(states_new, pred_action_emb)
        min_q = torch.min(q1, q2)

        policy_loss = (self.alpha * log_prob - min_q).mean()

        # Action VAE 自监督损失
        vae_loss = self.action_vae.unsupervised_loss(
            states_new.detach(), actions, next_states_new.detach()
        )

        total_policy_loss = policy_loss + vae_loss

        self.policy_optimizer.zero_grad()
        self.vae_optimizer.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.policy_optimizer.step()
        self.vae_optimizer.step()

        # ================================================================
        # 📌 ALPHA（SAC 温度）更新
        # ================================================================
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # ================================================================
        # 📌 软更新 Target Critic
        # ================================================================
        soft_update(self.target_critic, self.critic, self.tau)

        return {
            "td_loss": td_loss.item(),
            "cql_loss": (cql_loss1 + cql_loss2).item(),
            "critic_loss": critic_loss.item(),
            "policy_loss": policy_loss.item(),
            "vae_loss": vae_loss.item(),
            "alpha": self.alpha.item(),
            "cql_alpha": current_cql_alpha if not self.cql_with_lagrange else self.cql_alpha,
        }


class RecCalQLAgent(nn.Module):
    """
    Cal-QL = CQL + Monte Carlo Return 校准

    核心改进（相比 CQL）：
      - 在 CQL 正则项中，对策略采样动作的 Q 值用 MC 回报做下界校准
      - 即 q_calibrated = max(q_policy, mc_return)
      - 这样 CQL 的保守惩罚不会把 Q 压到比真实回报还低，避免过度保守
      - 均匀随机动作的 Q 值不做校准（它们本来就是 OOD，应该被惩罚）
    """

    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int = 64,
        state_hidden_dim: int = 128,
        action_emb_dim: int = 32,
        policy_hidden: Tuple[int, ...] = (256, 256),
        critic_hidden: Tuple[int, ...] = (256, 256),
        rnn_type: str = "GRU",
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha_init: float = 0.01,
        # ---- Cal-QL / CQL 参数 ----
        cql_alpha: float = 5.0,
        cql_num_random: int = 10,
        cql_with_lagrange: bool = False,
        cql_lagrange_thresh: float = 5.0,
        use_cal_ql: bool = True,          # 🆕 是否启用 Cal-QL 校准
        disable_offline_online_buffer: bool = False,
        offline_sample_prob: float = 0.25,
        # ----------------------
        lr: float = 1e-3,
        buffer_size: int = 100000,
        max_seq_len: int = 50,
        batch_size: int = 256,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.num_items = num_items
        self.action_emb_dim = action_emb_dim

        # CQL / Cal-QL 参数
        self.cql_alpha = cql_alpha
        self.cql_num_random = cql_num_random
        self.cql_with_lagrange = cql_with_lagrange
        self.cql_lagrange_thresh = cql_lagrange_thresh
        self.use_cal_ql = use_cal_ql  # 🆕
        self.disable_offline_online_buffer = bool(disable_offline_online_buffer)
        self.offline_sample_prob = float(offline_sample_prob)

        # Lagrange 可学习 cql_alpha
        if self.cql_with_lagrange:
            self.log_cql_alpha = torch.tensor(
                np.log(cql_alpha), dtype=float32, requires_grad=True, device=device
            )
            self.cql_alpha_optimizer = optim.AdamW([self.log_cql_alpha], lr=lr)

        # ---------- 序列状态编码器 ----------
        self.state_encoder = SequenceStateEncoder(
            num_items=num_items,
            embedding_dim=item_embedding_dim,
            hidden_dim=state_hidden_dim,
            rnn_type=rnn_type,
        ).to(device)
        state_dim = self.state_encoder.output_dim

        # ---------- Action VAE ----------
        self.action_vae = ActionVAE(
            action_dim=num_items,      # num_items 即为动作空间大小
            latent_dim=action_emb_dim, # action_emb_dim 即为潜在嵌入维度
            state_dim=state_dim,
        ).to(device)

        # ---------- SAC 网络 ----------
        self.policy = PolicyNetwork(state_dim, action_emb_dim, policy_hidden).to(device)
        self.critic = DoubleCriticNetwork(state_dim, action_emb_dim, critic_hidden).to(device)
        self.target_critic = DoubleCriticNetwork(state_dim, action_emb_dim, critic_hidden).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_critic.parameters():
            p.requires_grad = False

        # ---------- SAC 温度 alpha ----------
        self.target_entropy = -float(action_emb_dim)
        self.log_alpha = torch.tensor(
            np.log(alpha_init), dtype=float32, requires_grad=True, device=device
        )
        self.alpha = self.log_alpha.exp().detach()

        # ---------- 优化器 ----------
        self.encoder_optimizer = optim.AdamW(self.state_encoder.parameters(), lr=lr)
        self.vae_optimizer = optim.AdamW(self.action_vae.parameters(), lr=lr)
        self.policy_optimizer = optim.AdamW(self.policy.parameters(), lr=lr)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=lr)
        self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=lr)

        # ---------- 经验回放（需要支持 mc_returns）----------
        self.memory = ReplayBuffer(
            max_size=buffer_size,
            state_dim=state_dim,        # ← 补上
            action_emb_dim=action_emb_dim,
            max_seq_len=max_seq_len,
            device=device,
        )

        # 第二处：self.offline_memory
        self.offline_memory = ReplayBuffer(
            max_size=buffer_size,
            state_dim=state_dim,        # ← 补上
            action_emb_dim=action_emb_dim,
            max_seq_len=max_seq_len,
            device=device,
        )
        self.initial_phase = True
    # ----------------------------------------------------------
    def encode_state(self, item_seq: Tensor, seq_lengths: Tensor = None) -> Tensor:
        return self.state_encoder(item_seq, seq_lengths)

    # ----------------------------------------------------------
    @torch.no_grad()
    def get_action(self, item_seq, seq_length=None, deterministic=False, explore_eps=0.1):
        if item_seq is None:
            valid_ids = (
                torch.where(self.action_vae.action_mask)[0]
                if self.action_vae.action_mask is not None
                else torch.arange(self.num_items)
            )
            action_id = valid_ids[np.random.randint(len(valid_ids))].item()
            action_emb = self.action_vae.get_action_embedding(action_id)
            return action_id, action_emb
        if not isinstance(item_seq, Tensor):
            item_seq = torch.tensor(item_seq, dtype=long, device=self.device)
        if item_seq.dim() == 1:
            item_seq = item_seq.unsqueeze(0)

        seq_len_tensor = (
            torch.tensor([seq_length], dtype=long, device=self.device) if seq_length is not None else None
        )
        state = self.encode_state(item_seq, seq_len_tensor)

        if not deterministic and np.random.rand() < explore_eps:
            valid_ids = (
                torch.where(self.action_vae.action_mask)[0]
                if self.action_vae.action_mask is not None
                else torch.arange(self.num_items)
            )
            action_id = valid_ids[np.random.randint(len(valid_ids))].item()
            action_emb = self.action_vae.get_action_embedding(action_id)
        else:
            action_emb, _ = self.policy(state, deterministic=deterministic, return_log_prob=False)
            action_id = self.action_vae.get_best_match(action_emb).item()
            action_emb = action_emb.squeeze(0)

        return action_id, action_emb

    @torch.no_grad()
    def evaluate_q(self, item_seq, seq_length, action_emb: Tensor) -> float:
        if not isinstance(item_seq, Tensor):
            item_seq = torch.tensor(item_seq, dtype=long, device=self.device)
        if item_seq.dim() == 1:
            item_seq = item_seq.unsqueeze(0)
        seq_len_tensor = torch.tensor([int(seq_length)], dtype=long, device=self.device)
        state = self.encode_state(item_seq, seq_len_tensor)
        a = action_emb.view(1, -1).to(self.device)
        q1, q2 = self.critic(state, a)
        return float(torch.min(q1, q2).item())

    # ----------------------------------------------------------
    def store_transition(
        self, item_seq, seq_len, action, action_emb,
        reward, next_item_seq, next_seq_len, done, mc_return = 0.0
    ):
        """🆕 新增 mc_return 参数，用于 Cal-QL 校准"""
        self.memory.add(
            item_seq, seq_len, action, action_emb,
            reward, next_item_seq, next_seq_len, done, mc_return
        )

    # ----------------------------------------------------------
    # 🔑 Cal-QL 核心：带 MC 回报校准的保守正则项
    # ----------------------------------------------------------
    def _compute_cql_regularizer(
        self,
        states: Tensor,
        next_states: Tensor,
        action_embs: Tensor,
        mc_returns: Tensor = None,
        is_online = True
    ) -> Tuple[Tensor, Tensor]:
        """
        Cal-QL 正则项 = logsumexp(Q_calibrated(s, a_sampled)) - E_{a~D}[Q(s, a_data)]

        与 CQL 的区别：
          - 对策略采样动作（curr/next），Q 值用 mc_returns 做下界校准
          - 均匀随机动作不校准（它们是纯 OOD 探测）
          - 校准公式: q_cal = max(q, mc_return)
        """
        B = states.shape[0]
        N = self.cql_num_random

        # ---- 1) 均匀随机动作（不校准）----
        random_actions = (
            torch.rand(B * N, self.action_emb_dim, device=self.device) * 2.0 - 1.0
        )
        random_log_prob = np.log(0.5 ** self.action_emb_dim)

        # ---- 2) 策略采样动作 ----
        states_repeated = states.unsqueeze(1).repeat(1, N, 1).view(B * N, -1)
        next_states_repeated = next_states.unsqueeze(1).repeat(1, N, 1).view(B * N, -1)

        curr_actions, curr_log_probs = self.policy(
            states_repeated, deterministic=False, return_log_prob=True
        )
        next_actions, next_log_probs = self.policy(
            next_states_repeated, deterministic=False, return_log_prob=True
        )

        # ---- 3) 计算 Q 值 ----
        q1_random, q2_random = self.critic(states_repeated, random_actions)
        q1_curr, q2_curr = self.critic(states_repeated, curr_actions)
        q1_next, q2_next = self.critic(states_repeated, next_actions)

        # ---- 🆕 4) Cal-QL 校准：用 MC 回报做 Q 值下界 ----
        if self.use_cal_ql and mc_returns is not None:
            # mc_returns: (B, 1) → (B*N, 1)
            mc_lower = mc_returns.unsqueeze(1).repeat(1, N, 1).view(B * N, -1)
            # 只校准策略采样的动作，不校准随机动作
            q1_curr = torch.maximum(q1_curr, mc_lower)
            q2_curr = torch.maximum(q2_curr, mc_lower)
            q1_next = torch.maximum(q1_next, mc_lower)
            q2_next = torch.maximum(q2_next, mc_lower)

        # ---- 5) reshape 回 (B, N, 1) ----
        q1_random = q1_random.view(B, N, 1)
        q2_random = q2_random.view(B, N, 1)
        q1_curr = q1_curr.view(B, N, 1)
        q2_curr = q2_curr.view(B, N, 1)
        q1_next = q1_next.view(B, N, 1)
        q2_next = q2_next.view(B, N, 1)

        curr_log_probs = curr_log_probs.view(B, N, 1)
        next_log_probs = next_log_probs.view(B, N, 1)

        # ---- 6) 拼接并减去 importance weight ----
        q1_cat = torch.cat([
            q1_random - random_log_prob,
            q1_curr - curr_log_probs.detach(),
            q1_next - next_log_probs.detach(),
        ], dim=1)

        q2_cat = torch.cat([
            q2_random - random_log_prob,
            q2_curr - curr_log_probs.detach(),
            q2_next - next_log_probs.detach(),
        ], dim=1)

        # ---- 7) logsumexp 估计 OOD Q ----
        cql_q1_ood = torch.logsumexp(q1_cat.squeeze(-1), dim=1).mean()
        cql_q2_ood = torch.logsumexp(q2_cat.squeeze(-1), dim=1).mean()

        # ---- 8) 数据集内动作的 Q 值 ----
        cql_q1_data, cql_q2_data = self.critic(states, action_embs)
        cql_q1_data = cql_q1_data.mean()
        cql_q2_data = cql_q2_data.mean()

        # ---- 9) CQL 正则 ----
        cql_loss1 = cql_q1_ood - cql_q1_data
        cql_loss2 = cql_q2_ood - cql_q2_data

        return cql_loss1, cql_loss2

    # ----------------------------------------------------------
    # 🔄 Cal-QL 完整更新步
    # ----------------------------------------------------------
    def update(self) -> Dict[str, float]:
        if self.memory.size < self.batch_size:
            return {}

        # ✅ 修复：先检查 offline_memory 是否有足够数据
        use_offline = (
            (not self.disable_offline_online_buffer)
            and self.offline_memory.size >= self.batch_size
            and np.random.rand() < self.offline_sample_prob
        )

        if use_offline:
            (item_seqs, seq_lens, actions, action_embs,
            rewards, next_item_seqs, next_seq_lens, dones,
            mc_returns) = self.offline_memory.sample(self.batch_size)
            action_embs = self.action_vae.get_action_embedding(actions)
        else:
            (item_seqs, seq_lens, actions, action_embs,
            rewards, next_item_seqs, next_seq_lens, dones,
            mc_returns) = self.memory.sample(self.batch_size)
        # ---- 编码状态 ----
        states = self.encode_state(item_seqs, seq_lens)
        next_states = self.encode_state(next_item_seqs, next_seq_lens)

        # ================================================================
        # 📌 CRITIC 更新 = SAC TD loss + Cal-QL 正则项
        # ================================================================

        # ---- 标准 SAC TD target ----
        with torch.no_grad():
            next_action_emb, next_log_prob = self.policy(
                next_states, deterministic=False, return_log_prob=True
            )
            target_q1, target_q2 = self.target_critic(next_states, next_action_emb)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1.0 - dones) * self.gamma * (
                target_q - self.alpha * next_log_prob
            )

        current_q1, current_q2 = self.critic(states, action_embs)
        td_loss1 = F.mse_loss(current_q1, target_q)
        td_loss2 = F.mse_loss(current_q2, target_q)
        td_loss = td_loss1 + td_loss2

        # ---- Cal-QL 正则项（🆕 传入 mc_returns）----
        cql_loss1, cql_loss2 = self._compute_cql_regularizer(
            states.detach(), next_states.detach(), action_embs,
            mc_returns=mc_returns,  # 🆕
        )

        # 获取当前 cql_alpha
        if self.cql_with_lagrange:
            current_cql_alpha = self.log_cql_alpha.exp().clamp(min=0.0)
        else:
            current_cql_alpha = self.cql_alpha

        # 总 critic loss
        critic_loss = td_loss + current_cql_alpha * (cql_loss1 + cql_loss2)

        self.critic_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.state_encoder.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        self.encoder_optimizer.step()

        # ---- Lagrange 更新（可选）----
        cql_alpha_loss = 0.0
        if self.cql_with_lagrange:
            cql_alpha_loss = -current_cql_alpha * (
                (cql_loss1 + cql_loss2).detach() / 2.0 - self.cql_lagrange_thresh
            )
            self.cql_alpha_optimizer.zero_grad()
            cql_alpha_loss.backward()
            self.cql_alpha_optimizer.step()
            self.cql_alpha = self.log_cql_alpha.exp().item()

        # ================================================================
        # 📌 POLICY 更新 = SAC 策略损失 + VAE 自监督损失
        # ================================================================
        states_new = self.encode_state(item_seqs, seq_lens)
        next_states_new = self.encode_state(next_item_seqs, next_seq_lens)

        pred_action_emb, log_prob = self.policy(
            states_new, deterministic=False, return_log_prob=True
        )
        q1, q2 = self.critic(states_new, pred_action_emb)
        min_q = torch.min(q1, q2)

        policy_loss = (self.alpha * log_prob - min_q).mean()

        vae_loss = self.action_vae.unsupervised_loss(
            states_new.detach(), actions, next_states_new.detach()
        )

        total_policy_loss = policy_loss + vae_loss

        self.policy_optimizer.zero_grad()
        self.vae_optimizer.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.policy_optimizer.step()
        self.vae_optimizer.step()

        # ================================================================
        # 📌 ALPHA 更新
        # ================================================================
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # ================================================================
        # 📌 软更新 Target Critic
        # ================================================================
        soft_update(self.target_critic, self.critic, self.tau)

        return {
            "td_loss": td_loss.item(),
            "cql_loss": (cql_loss1 + cql_loss2).item(),
            "critic_loss": critic_loss.item(),
            "policy_loss": policy_loss.item(),
            "vae_loss": vae_loss.item(),
            "alpha": self.alpha.item(),
            "cql_alpha": (
                current_cql_alpha if not self.cql_with_lagrange else self.cql_alpha
            ),
            "cal_ql_enabled": self.use_cal_ql,  # 🆕
        }
    
    def clear_memory(self):
        self.memory.reset()

    def update_memory(self):
        self.offline_memory = copy.deepcopy(self.memory)
        self.memory.reset()
        self.initial_phase = True

    def update_action_mask(self, reward_mask):
        self.action_vae.update_action_mask(reward_mask)
    def diagnose_vae(self, batch_size: int = 4):
        """
        诊断 ActionVAE 训练流程中的中间变量，定位 inf/nan 来源
        """
        buffer = self.memory if self.memory.size >= batch_size else self.offline_memory
        if buffer.size < batch_size:
            print("❌ 缓冲区数据不足，无法诊断")
            return

        print("\n" + "=" * 70)
        print("🔍 ActionVAE 诊断开始 (batch_size={})".format(batch_size))
        print("=" * 70)

        # ---- 1. 采样数据 ----
        (item_seqs, seq_lens, actions, action_embs,
        rewards, next_item_seqs, next_seq_lens, dones,
        mc_returns) = buffer.sample(batch_size)

        print("\n📦 [Step 0] 采样数据检查:")
        print(f"  item_seqs shape: {item_seqs.shape}, dtype: {item_seqs.dtype}")
        print(f"  item_seqs sample:\n    {item_seqs[0][:10]}...")
        print(f"  seq_lens: {seq_lens}")
        print(f"  actions: {actions}")
        print(f"  actions min/max: {actions.min().item()} / {actions.max().item()}")
        print(f"  rewards: {rewards.squeeze()}")

        # ---- 2. 编码状态 ----
        with torch.no_grad():
            s1 = self.encode_state(item_seqs, seq_lens)
            s2 = self.encode_state(next_item_seqs, next_seq_lens)

        print("\n📦 [Step 1] 状态编码:")
        print(f"  s1 shape: {s1.shape}")
        print(f"  s1 min/max/mean: {s1.min().item():.6f} / {s1.max().item():.6f} / {s1.mean().item():.6f}")
        print(f"  s1 has nan: {torch.isnan(s1).any().item()}, has inf: {torch.isinf(s1).any().item()}")
        print(f"  s2 min/max/mean: {s2.min().item():.6f} / {s2.max().item():.6f} / {s2.mean().item():.6f}")
        print(f"  s2 has nan: {torch.isnan(s2).any().item()}, has inf: {torch.isinf(s2).any().item()}")

        # ---- 3. VAE 编码器 ----
        vae = self.action_vae
        s_pair = torch.cat([s1, s2], dim=-1)
        print(f"\n📦 [Step 2] 编码器输入 s_pair:")
        print(f"  shape: {s_pair.shape}")
        print(f"  min/max: {s_pair.min().item():.6f} / {s_pair.max().item():.6f}")

        h = vae.encoder(s_pair)
        print(f"\n📦 [Step 3] 编码器隐层 h:")
        print(f"  shape: {h.shape}")
        print(f"  min/max/mean: {h.min().item():.6f} / {h.max().item():.6f} / {h.mean().item():.6f}")
        print(f"  has nan: {torch.isnan(h).any().item()}, has inf: {torch.isinf(h).any().item()}")

        mu = vae.fc_mu(h)
        logvar = vae.fc_logvar(h)
        print(f"\n📦 [Step 4] mu & logvar:")
        print(f"  mu    min/max/mean: {mu.min().item():.6f} / {mu.max().item():.6f} / {mu.mean().item():.6f}")
        print(f"  logvar min/max/mean: {logvar.min().item():.6f} / {logvar.max().item():.6f} / {logvar.mean().item():.6f}")
        print(f"  mu has nan: {torch.isnan(mu).any().item()}, logvar has nan: {torch.isnan(logvar).any().item()}")

        # ---- 4. 重参数化 ----
        std = (logvar * 0.5).exp()
        eps = torch.randn_like(std)
        z_raw = mu + std * eps
        z = torch.tanh(z_raw)
        print(f"\n📦 [Step 5] 重参数化:")
        print(f"  std min/max: {std.min().item():.6f} / {std.max().item():.6f}")
        print(f"  z_raw (before tanh) min/max: {z_raw.min().item():.6f} / {z_raw.max().item():.6f}")
        print(f"  z (after tanh) min/max: {z.min().item():.6f} / {z.max().item():.6f}")
        print(f"  z has nan: {torch.isnan(z).any().item()}")

        # ---- 5. 动作嵌入 ----
        embeddings = vae.get_effective_embeddings()
        print(f"\n📦 [Step 6] 动作嵌入表:")
        print(f"  shape: {embeddings.shape}")
        print(f"  min/max/mean: {embeddings.min().item():.6f} / {embeddings.max().item():.6f} / {embeddings.mean().item():.6f}")
        print(f"  has nan: {torch.isnan(embeddings).any().item()}")
        print(f"  trainable: {vae.trainable_embeddings}")

        # ---- 6. 相似度计算 ----
        similarity = vae.compute_similarity(z)
        print(f"\n📦 [Step 7] 相似度 (similarity):")
        print(f"  shape: {similarity.shape}")
        finite_sim = similarity[similarity != float('-inf')]
        if finite_sim.numel() > 0:
            print(f"  finite values min/max/mean: {finite_sim.min().item():.6f} / {finite_sim.max().item():.6f} / {finite_sim.mean().item():.6f}")
        else:
            print(f"  ⚠️ 所有值都是 -inf！")
        print(f"  has nan: {torch.isnan(similarity).any().item()}")
        print(f"  has inf: {torch.isinf(similarity).any().item()}")
        print(f"  -inf count: {(similarity == float('-inf')).sum().item()} / {similarity.numel()}")

        # ---- 7. action_mask 检查 ----
        mask = vae._action_mask
        print(f"\n📦 [Step 8] Action Mask:")
        print(f"  shape: {mask.shape}")
        print(f"  valid actions: {mask.sum().item()} / {mask.numel()}")
        print(f"  mask dtype: {mask.dtype}")

        # 检查 actions 是否在 mask 范围内
        actions_flat = actions.view(-1)
        actions_valid = mask[actions_flat]
        print(f"  采样 actions 是否都在 mask 内: {actions_valid.all().item()}")
        if not actions_valid.all():
            invalid_idx = torch.where(~actions_valid)[0]
            print(f"  ⚠️ 无效动作索引: {invalid_idx.tolist()}")
            print(f"  ⚠️ 对应动作 id: {actions_flat[invalid_idx].tolist()}")

        # ---- 8. 交叉熵损失 ----
        print(f"\n📦 [Step 9] 交叉熵损失:")
        print(f"  actions for CE: {actions_flat}, dtype: {actions_flat.dtype}")
        print(f"  actions range: [{actions_flat.min().item()}, {actions_flat.max().item()}]")
        print(f"  similarity shape for CE: {similarity.shape}")
        print(f"  num_classes (action_dim): {vae.action_dim}")

        try:
            ce_loss = F.cross_entropy(similarity, actions_flat, reduction='mean')
            print(f"  cross_entropy loss: {ce_loss.item():.6f}")
            print(f"  loss is nan: {torch.isnan(ce_loss).item()}, is inf: {torch.isinf(ce_loss).item()}")
        except Exception as e:
            print(f"  ❌ cross_entropy 报错: {e}")

        # ---- 9. _norm_const ----
        print(f"\n📦 [Step 10] 归一化常数:")
        print(f"  _norm_const: {vae._norm_const.item():.6f}")
        if vae._norm_const.item() == 0:
            print(f"  ⚠️ _norm_const 为 0，除法会导致 inf！")

        # ---- 10. KL 散度 ----
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        print(f"\n📦 [Step 11] KL 散度: {kl_loss.item():.6f}")

        # ---- 11. 完整 compute_loss ----
        print(f"\n📦 [Step 12] 完整 compute_loss 调用:")
        try:
            loss_dict = vae.compute_loss(s1, actions_flat, s2)
            for k, v in loss_dict.items():
                print(f"  {k}: {v.item():.6f}")
        except Exception as e:
            print(f"  ❌ compute_loss 报错: {e}")

        # ---- 12. unsupervised_loss (实际训练用的) ----
        print(f"\n📦 [Step 13] unsupervised_loss 调用:")
        try:
            unsup_loss = vae.unsupervised_loss(s1, actions_flat, s2)
            print(f"  unsupervised_loss: {unsup_loss.item():.6f}")
        except Exception as e:
            print(f"  ❌ unsupervised_loss 报错: {e}")

        print("\n" + "=" * 70)
        print("🔍 诊断结束")
        print("=" * 70 + "\n")

    def initial_vae_training(self, max_epochs: int = 50, batch_size: int = 64, lr: float = 1e-3):
        """
        VAE 初始化预训练阶段
        
        使用较大学习率对 ActionVAE 进行自监督预训练，
        通过 (s1, a, s2) 三元组学习动作嵌入空间，
        收敛后切回正常学习率。
        
        Args:
            max_epochs: 最大训练轮数（-1 表示仅靠收敛条件停止）
            batch_size: 每批样本数
            lr: 初始化阶段学习率（通常比正常训练大）
        """
        # 确保有足够数据
        buffer = self.memory if self.memory.size >= batch_size else self.memory
        if buffer.size < batch_size:
            print("[initial_vae_training] 缓冲区数据不足，跳过预训练。")
            return

        print("=" * 60)
        print("🚀 ActionVAE 初始化预训练阶段开始")
        print(f"   数据量: {buffer.size} | batch_size: {batch_size} | lr: {lr}")
        print("=" * 60)

        # 临时使用较大学习率
        vae_pretrain_optimizer = torch.optim.Adam(self.action_vae.parameters(), lr=3e-3)

        epoch_losses = []
        if os.environ.get("CALQL_VAE_DIAGNOSE", "0") == "1":
            self.diagnose_vae(batch_size=4)
        for epoch in range(max_epochs if max_epochs > 0 else 50):
            batch_losses = []

            # ---- 按 batch 遍历整个缓冲区 ----
            num_batches = max(1, buffer.size // batch_size)
            for batch_idx in range(num_batches):
                (item_seqs, seq_lens, actions, action_embs,
                rewards, next_item_seqs, next_seq_lens, dones,
                mc_returns) = buffer.sample(batch_size)

                # 编码状态（不更新 encoder 梯度）
                with torch.no_grad():
                    s1 = self.encode_state(item_seqs, seq_lens)
                    s2 = self.encode_state(next_item_seqs, next_seq_lens)

                # VAE 自监督损失
                loss_dict = self.action_vae.compute_loss(s1, actions.view(-1), s2)
                loss = loss_dict['total']

                vae_pretrain_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.action_vae.parameters(), max_norm=1.0)
                vae_pretrain_optimizer.step()

                batch_losses.append(loss.item())

                # 打印每一步损失
                # print(
                #     f"  [Epoch {epoch:3d} | Batch {batch_idx:3d}] "
                #     f"total={loss_dict['total'].item():.5f}  "
                #     f"recon={loss_dict['recon'].item():.5f}  "
                #     f"kl={loss_dict['kl'].item():.5f}  "
                #     f"embed_reg={loss_dict['embed_reg'].item():.5f}"
                # )

            epoch_mean = np.mean(batch_losses)
            epoch_losses.append(epoch_mean)
            print(f"  ── Epoch {epoch:3d} 平均损失: {epoch_mean:.5f} "
                f"(近10轮均值: {np.mean(epoch_losses[-10:]):.5f})")

            # ---- 收敛判断 ----
            if len(epoch_losses) >= 20:
                recent_10 = np.mean(epoch_losses[-10:])
                recent_5 = np.mean(epoch_losses[-5:])
                if abs(recent_10 - recent_5) < 1e-5:
                    print(f"\n✅ 已收敛！Epoch {epoch} | 最终损失: {epoch_mean:.5f}")
                    break
        else:
            print(f"\n⚠️ 达到最大轮数 {max_epochs}，未完全收敛 | 最终损失: {epoch_losses[-1]:.5f}")

        # 恢复正常学习率优化器
        self.vae_optimizer = torch.optim.AdamW(self.action_vae.parameters(), lr=3e-3)

        print("=" * 60)
        print("🏁 ActionVAE 预训练完成，已切换回正常学习率优化器")
        print("=" * 60)

    def reset(self, mask=None, is_change=False):
        if is_change and mask:
            self.update_action_mask(mask)
            self.update_memory()

    def step(self, item_seq, seq_len, action, action_emb,
        reward, next_item_seq, next_seq_len, done, mc_return = 0.0):
        self.memory.add(
            item_seq,seq_len,action,action_emb,reward,next_item_seq,next_seq_len,done,mc_return
        )
        if self.initial_phase and self.memory.size >= self.memory.max_size // 2:
            self.initial_vae_training(max_epochs=50, batch_size=32, lr=1e-3)
            self.initial_phase = False
        elif not self.initial_phase and self.memory.size >= self.memory.max_size // 2:
            self.update()
        # if 
        #     def update(self, traj: Trajectory = None, memory_reset=False):
        # if traj is not None:
        #     s1, a1, a_emb, r1, s2, done = traj.get_current_traj()
        #     for idx in range(s1.shape[0]):
        #         self._memory.add(s1=s1[idx],
        #                          a1=a1[idx],
        #                          a_emb=a_emb[idx],
        #                          r1=r1[idx],
        #                          s2=s2[idx],
        #                          done=done[idx])

        # if self.initial_phase and self._memory.size >= self.config.buffer_size:
        #     self.initial_phase_training(max_epochs=self.config.initial_phase_epochs)
        #     if memory_reset:
        #         self._memory.reset()
        # elif not self.initial_phase and self._memory.size >= self.config.sup_batch_size:
        #     self.CQL()


    def update_mask(self, mask):
        self.action_vae.update_action_mask(mask)

        # self.initial_phase = True
        # self.offline_memory = copy.deepcopy(self.memory)
        # self.memory.reset()
