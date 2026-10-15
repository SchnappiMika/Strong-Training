import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from recsys_calql.utils.utils import pair_distance, weight_init

class ActionVAE(nn.Module):
    """
    用于强化学习的动作表示 VAE
    
    核心思想：
    - 将离散动作嵌入到连续的低维空间
    - 通过状态转移 (s1 -> s2) 来学习动作的语义表示
    - 使用 VAE 框架增加表示的平滑性和泛化能力
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 16,
        hidden_dim: int = 128,
        beta: float = 0.01,
        embedding_reg: float = 0.001,
        use_pretrained_embeddings: bool = False,
        pretrained_embeddings: Tensor = None,
        learning_rate: float = 1e-3,
    ):
        super(ActionVAE, self).__init__()
        
        # 维度配置
        self.state_dim = state_dim
        self.action_dim = action_dim
        print("Action dim is ", self.action_dim)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # 损失函数超参数
        self.beta = beta  # KL 散度权重
        self.embedding_reg = embedding_reg  # 嵌入正则化权重
        
        # 动作掩码（用于屏蔽无效动作）
        self._action_mask = torch.ones(action_dim, dtype=torch.bool)
        self._norm_const = torch.log(torch.tensor(action_dim, dtype=torch.float32))
        
        # ==================== 动作嵌入 ====================
        if use_pretrained_embeddings and pretrained_embeddings is not None:
            # 使用预训练嵌入（归一化到 [-1, 1]）
            embeddings = pretrained_embeddings.clone()
            embeddings = self._normalize_embeddings(embeddings)
            self.embeddings = nn.Parameter(embeddings, requires_grad=False)
            self.latent_dim = embeddings.shape[1]
            self.trainable_embeddings = False
        else:
            # 随机初始化可训练嵌入
            init_embeddings = torch.rand(action_dim, latent_dim) * 2 - 1
            self.embeddings = nn.Parameter(init_embeddings, requires_grad=True)
            self.trainable_embeddings = True
        
        # ==================== 编码器网络 ====================
        # 输入：拼接的状态对 (s1, s2)
        self.encoder = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, self.latent_dim)
        
        # ==================== 优化器 ====================
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # 初始化权重
        self.apply(weight_init)
        
        self._print_architecture()
    
    def _normalize_embeddings(self, embeddings: Tensor) -> Tensor:
        """将嵌入归一化到 [-1, 1] 范围"""
        max_val, min_val = embeddings.max(), embeddings.min()
        return ((embeddings - min_val) / (max_val - min_val + 1e-8)) * 2 - 1
    
    def _print_architecture(self):
        """打印模型架构信息"""
        print("=" * 50)
        print("ActionVAE Architecture")
        print("=" * 50)
        for name, param in self.named_parameters():
            print(f"  {name}: {param.shape}")
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Total trainable parameters: {total_params:,}")
        print("=" * 50)
    
    # ==================== 核心方法 ====================
    
    def encode(self, state1: Tensor, state2: Tensor) -> tuple[Tensor, Tensor]:
        """
        编码状态转移为潜在分布参数
        
        Args:
            state1: 初始状态 (batch, state_dim)
            state2: 下一状态 (batch, state_dim)
        
        Returns:
            mu: 均值 (batch, latent_dim)
            logvar: 对数方差 (batch, latent_dim)
        """
        state_cat = torch.cat([state1, state2], dim=-1)
        h = self.encoder(state_cat)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """重参数化技巧"""
        std = (logvar * 0.5).exp()
        eps = torch.randn_like(std)
        z = mu + std * eps
        return torch.tanh(z)  # 约束到 [-1, 1]
    
    def forward(self, state1: Tensor, state2: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        前向传播
        
        Returns:
            z: 采样的潜在动作表示
            mu: 均值
            logvar: 对数方差
        """
        mu, logvar = self.encode(state1, state2)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    # ==================== 动作匹配 ====================
    
    def get_effective_embeddings(self) -> Tensor:
        """获取有效的动作嵌入（应用 tanh 约束）"""
        if self.trainable_embeddings:
            return torch.tanh(self.embeddings)
        return self.embeddings
    
    def compute_similarity(self, z: Tensor) -> Tensor:
        """
        计算潜在表示与所有动作嵌入的相似度
        
        Args:
            z: 潜在动作表示 (batch, latent_dim)
        
        Returns:
            similarity: 相似度分数 (batch, action_dim)
        """
        embeddings = self.get_effective_embeddings()
        # 负欧氏距离作为相似度
        similarity = -pair_distance(z, embeddings)
        # 屏蔽无效动作
        similarity[:, ~self._action_mask] = float('-inf')
        return similarity
    
    def get_action(self, state1: Tensor, state2: Tensor) -> Tensor:
        """根据状态转移预测最可能的动作"""
        z, _, _ = self.forward(state1, state2)
        similarity = self.compute_similarity(z)
        return similarity.argmax(dim=-1)
    
    def get_best_match(self, z: Tensor) -> Tensor:
        """
        给定连续潜在向量 z，返回最近邻的 item id
        
        Args:
            z: policy 输出的连续 action embedding (batch, latent_dim) 或 (1, latent_dim)
        
        Returns:
            最近邻 item 的索引 (batch,)
        """
        similarity = self.compute_similarity(z)
        return similarity.argmax(dim=-1)
    

    def get_action_embedding(self, action: Tensor) -> Tensor:
        """获取指定动作的嵌入向量"""
        embeddings = self.get_effective_embeddings()
        return embeddings[action]
    
    def get_action_embeddings(self, actions: Tensor) -> Tensor:
        """
        批量获取动作嵌入向量
        
        Args:
            actions: 动作索引张量 (batch,) 或 (batch, k)
        
        Returns:
            嵌入向量 (batch, latent_dim) 或 (batch, k, latent_dim)
        """
        embeddings = self.get_effective_embeddings()
        return embeddings[actions]

    # ==================== 动作掩码 ====================
    
    def update_action_mask(self, mask: Tensor) -> None:
        """更新有效动作掩码"""
        self._action_mask = mask.bool().detach()
        num_valid = self._action_mask.sum().float()
        self._norm_const = torch.log(num_valid.clamp(min=1.0))
    
    # ==================== 损失函数 ====================
    
    def compute_loss(
        self,
        state1: Tensor,
        action: Tensor,
        state2: Tensor,
        normalize: bool = True
    ) -> dict[str, Tensor]:
        # 前向传播
        z, mu, logvar = self.forward(state1, state2)
        
        # 1. 重建损失（交叉熵）—— 过滤掉 action 不在 mask 内的样本
        similarity = self.compute_similarity(z)
        
        # ✅ 找出 action 在当前 mask 内的有效样本
        valid_mask = self._action_mask[action]  # (batch,) bool
        
        if valid_mask.any():
            # 只对有效样本计算交叉熵
            recon_loss = F.cross_entropy(
                similarity[valid_mask], 
                action[valid_mask], 
                reduction='mean'
            )
            if normalize:
                recon_loss = recon_loss / self._norm_const
        else:
            # 全部无效，recon_loss 归零
            recon_loss = torch.tensor(0.0, device=state1.device)
        
        # 2. KL 散度（所有样本都参与，与 mask 无关）
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # 3. 嵌入正则化
        if self.trainable_embeddings:
            embed_reg = self.embeddings.pow(2).mean()
        else:
            embed_reg = torch.tensor(0.0, device=state1.device)
        
        # 总损失
        total_loss = recon_loss + self.beta * kl_loss + self.embedding_reg * embed_reg
        
        return {
            'total': total_loss,
            'recon': recon_loss,
            'kl': kl_loss,
            'embed_reg': embed_reg,
            'valid_ratio': valid_mask.float().mean(),  # 🆕 监控有效比例
        }
    
    def train_step(
        self,
        state1: Tensor,
        action: Tensor,
        state2: Tensor,
        normalize: bool = True
    ) -> dict[str, float]:
        """执行一步训练"""
        self.optimizer.zero_grad()
        losses = self.compute_loss(state1, action, state2, normalize)
        losses['total'].backward()
        self.optimizer.step()
        
        return {k: v.item() for k, v in losses.items()}
    
    def unsupervised_loss(
        self,
        state: torch.Tensor,          # 当前状态 (B, state_dim)
        actions: torch.Tensor,        # 离散动作 id (B,)
        next_state: torch.Tensor,     # 下一状态 (B, state_dim)
    ) -> torch.Tensor:
        """
        VAE 无监督损失：
        - 编码器将 (s1, s2) 编码为潜在向量 z
        - 用 z 与动作嵌入表做对比，拉近真实动作嵌入
        - KL 散度正则化
        - 自动过滤 action_mask 外的无效样本
        """
        # 0. 过滤无效动作
        valid_mask = self._action_mask[actions]  # (B,) bool
        
        # 如果全部无效，返回仅含 KL + 正则的损失（保持梯度流）
        if not valid_mask.any():
            s_pair = torch.cat([state, next_state], dim=-1)
            h = self.encoder(s_pair)
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            emb_reg = self.embeddings.norm(dim=-1).mean() if self.trainable_embeddings else 0.0
            return self.beta * kl_loss + self.embedding_reg * emb_reg

        # 只保留有效样本
        state_v = state[valid_mask]
        next_state_v = next_state[valid_mask]
        actions_v = actions[valid_mask]

        # 1. 编码 (s1, s2) → (mu, logvar)
        s_pair = torch.cat([state_v, next_state_v], dim=-1)   # (B', state_dim*2)
        h = self.encoder(s_pair)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # 2. 重参数化采样
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std                                     # (B', latent_dim)

        # 3. 取出真实动作嵌入
        action_emb = self.get_effective_embeddings()[actions_v] # (B', latent_dim)

        # 4. 重建损失：z 应接近真实动作嵌入
        recon_loss = F.mse_loss(z, action_emb)

        # 5. KL 散度
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # 6. 嵌入正则化（防止嵌入爆炸）
        emb_reg = self.embeddings.norm(dim=-1).mean() if self.trainable_embeddings else 0.0

        total_loss = recon_loss + self.beta * kl_loss + self.embedding_reg * emb_reg
        return total_loss
