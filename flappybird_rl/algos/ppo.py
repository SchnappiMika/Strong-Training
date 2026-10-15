# -*- coding: utf-8 -*-
"""
PPO（Proximal Policy Optimization）
============================================================
严格采用论文 WemiEnv 给出的超参数：

    γ（折扣因子）        : 0.99
    clip 参数 ε          : 0.2
    actor 学习率         : 0.0003
    critic 学习率        : 0.001
    GAE λ                : 0.97
    网络结构             : 3 层 × 64 神经元，ReLU
                          （论文的 "three layers" = 2 个隐藏层 + 1 个输出层）
    输入                 : 已归一化

论文未给出、采用 OpenAI Spinning Up 默认值的部分：
    train_pi_iters = 80, train_v_iters = 80, target_kl = 0.01

算法要点（PPO-clip）：
    1. 用「旧策略」采样一批轨迹
    2. 计算 GAE 优势
    3. 用 clip 后的重要性比率做多轮梯度上升，并在 KL 过大时提前停止
============================================================
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ----------------------------------------------------------------------
def mlp(sizes, activation, output_activation=nn.Identity):
    """按尺寸列表搭多层感知机。"""
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """策略网络：观测 → 离散动作的 logits。结构 obs→64→64→act_dim。"""

    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.logits_net = mlp([obs_dim, hidden, hidden, act_dim], nn.ReLU)

    def forward(self, obs):
        return self.logits_net(obs)


class Critic(nn.Module):
    """价值网络：观测 → V(s)。结构 obs→64→64→1。"""

    def __init__(self, obs_dim, hidden=64):
        super().__init__()
        self.v_net = mlp([obs_dim, hidden, hidden, 1], nn.ReLU)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)


# ----------------------------------------------------------------------
def compute_gae(rew, val, done, last_val, gamma, lam):
    """用 Generalized Advantage Estimation 计算优势与回报。

    参数
    ----
    rew, val : (n,) 每一步的奖励与 V(s)
    done     : (n,) 该步之后 episode 是否结束（1 结束 / 0 继续）
    last_val : 采样段结束后那一步的 V(s)（若恰好是终止状态则传 0）
    """
    n = len(rew)
    adv = np.zeros(n, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(n)):
        if done[t]:
            # 本步之后是终止状态，后续价值为 0
            next_val = 0.0
            next_gae = 0.0
        else:
            next_val = val[t + 1] if t + 1 < n else last_val
            next_gae = last_gae

        delta = rew[t] + gamma * next_val - val[t]
        last_gae = delta + gamma * lam * next_gae
        adv[t] = last_gae

    ret = adv + val
    return adv, ret


# ----------------------------------------------------------------------
class PPO:
    def __init__(self,
                 obs_dim,
                 act_dim,
                 actor_lr=3e-4,
                 critic_lr=1e-3,
                 gamma=0.99,
                 lam=0.97,
                 clip_ratio=0.2,
                 train_pi_iters=80,
                 train_v_iters=80,
                 target_kl=0.01,
                 device="cpu",
                 seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.target_kl = target_kl
        self.device = device

        self.actor = Actor(obs_dim, act_dim).to(device)
        self.critic = Critic(obs_dim).to(device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    # ------------------------------------------------------------------
    def _to_tensor(self, obs):
        o = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        if o.dim() == 1:
            o = o.unsqueeze(0)
        return o

    def value(self, obs):
        """估计 V(s)，返回 numpy 标量。"""
        with torch.no_grad():
            v = self.critic(self._to_tensor(obs))
        return float(v.item())

    def select_action(self, obs, deterministic=False):
        """按当前策略选动作，返回 (action, logp)。"""
        with torch.no_grad():
            logits = self.actor(self._to_tensor(obs))
            dist = Categorical(logits=logits)
            if deterministic:
                a = torch.argmax(logits, dim=-1)
            else:
                a = dist.sample()
            logp = dist.log_prob(a)
        return int(a.item()), float(logp.item())

    # ------------------------------------------------------------------
    def update(self, obs, act, adv, ret, logp_old):
        """执行一轮 PPO 更新。

        参数均为 numpy 数组：
            obs (n, obs_dim) / act (n,) / adv (n,) / ret (n,) / logp_old (n,)
        返回统计字典。
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(act, dtype=torch.long, device=self.device)
        ret_t = torch.as_tensor(ret, dtype=torch.float32, device=self.device)
        logp_old_t = torch.as_tensor(logp_old, dtype=torch.float32, device=self.device)

        # 优势标准化（稳定训练）
        adv = np.asarray(adv, dtype=np.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)

        # ---------------- 更新策略 ----------------
        pi_loss_val, kl_val, ent_val = 0.0, 0.0, 0.0
        stop_iter = self.train_pi_iters

        for i in range(self.train_pi_iters):
            logits = self.actor(obs_t)
            dist = Categorical(logits=logits)
            logp = dist.log_prob(act_t)

            ratio = torch.exp(logp - logp_old_t)
            clip_adv = torch.clamp(ratio, 1 - self.clip_ratio,
                                   1 + self.clip_ratio) * adv_t
            pi_loss = -(torch.min(ratio * adv_t, clip_adv)).mean()

            self.actor_opt.zero_grad()
            pi_loss.backward()
            self.actor_opt.step()

            # 统计（不参与梯度）
            with torch.no_grad():
                kl = float((logp_old_t - logp).mean().item())
                ent = float(dist.entropy().mean().item())

            pi_loss_val, kl_val, ent_val = float(pi_loss.item()), kl, ent

            # KL 过大提前停止（Spinning Up 的做法）
            if kl > 1.5 * self.target_kl:
                stop_iter = i + 1
                break

        # ---------------- 更新价值函数 ----------------
        v_loss_val = 0.0
        for _ in range(self.train_v_iters):
            v = self.critic(obs_t)
            v_loss = F.mse_loss(v, ret_t)

            self.critic_opt.zero_grad()
            v_loss.backward()
            self.critic_opt.step()

            v_loss_val = float(v_loss.item())

        return {
            "pi_loss": pi_loss_val,
            "v_loss": v_loss_val,
            "kl": kl_val,
            "entropy": ent_val,
            "pi_iters": stop_iter,
        }

    # ------------------------------------------------------------------
    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)
