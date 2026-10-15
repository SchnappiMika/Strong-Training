import torch
import torch.nn as nn
from typing import Tuple, Optional
from tqdm import tqdm


class FunkSVD(nn.Module):
    """
    Funk SVD 矩阵分解模型
    
    用于学习 User 和 Item 的低维表征
    """
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.device = device
        
        # User 和 Item 嵌入
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # 偏置项
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # 初始化
        self._init_weights()
        self.to(device)
    
    def _init_weights(self):
        """Xavier 初始化"""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
    
    def forward(
        self, 
        user_ids: torch.Tensor, 
        item_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        预测评分
        
        Args:
            user_ids: shape (batch_size,)
            item_ids: shape (batch_size,)
            
        Returns:
            predictions: shape (batch_size,)
        """
        user_emb = self.user_embedding(user_ids)  # (batch, dim)
        item_emb = self.item_embedding(item_ids)  # (batch, dim)
        
        # 点积 + 偏置
        dot_product = (user_emb * item_emb).sum(dim=1)  # (batch,)
        u_bias = self.user_bias(user_ids).squeeze()
        i_bias = self.item_bias(item_ids).squeeze()
        
        prediction = dot_product + u_bias + i_bias + self.global_bias
        return prediction
    
    def get_user_embeddings(self, user_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """获取用户表征"""
        if user_ids is None:
            return self.user_embedding.weight.detach()
        return self.user_embedding(user_ids).detach()
    
    def get_item_embeddings(self, item_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """获取物品表征"""
        if item_ids is None:
            return self.item_embedding.weight.detach()
        return self.item_embedding(item_ids).detach()


class FunkSVDTrainer:
    """Funk SVD 训练器"""
    
    def __init__(
        self,
        model: FunkSVD,
        learning_rate: float = 0.01,
        weight_decay: float = 0.01
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()
    
    def train(
        self,
        rating_matrix: torch.Tensor,
        n_epochs: int = 20,
        batch_size: int = 1024,
        verbose: bool = True
    ) -> list:
        """
        训练 Funk SVD 模型
        
        Args:
            rating_matrix: 评分矩阵 (n_users, n_items)，0表示未评分
            n_epochs: 训练轮数
            batch_size: 批大小
            
        Returns:
            losses: 每轮的损失值
        """
        self.model.train()
        
        # 提取非零评分的 (user, item, rating) 三元组
        nonzero_mask = rating_matrix != 0
        user_ids, item_ids = torch.where(nonzero_mask)
        ratings = rating_matrix[nonzero_mask]
        
        n_samples = len(ratings)
        losses = []
        
        for epoch in range(n_epochs):
            # 打乱数据
            perm = torch.randperm(n_samples)
            user_ids_shuffled = user_ids[perm]
            item_ids_shuffled = item_ids[perm]
            ratings_shuffled = ratings[perm]
            
            epoch_loss = 0.0
            n_batches = (n_samples + batch_size - 1) // batch_size
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                
                batch_users = user_ids_shuffled[start_idx:end_idx].to(self.model.device)
                batch_items = item_ids_shuffled[start_idx:end_idx].to(self.model.device)
                batch_ratings = ratings_shuffled[start_idx:end_idx].to(self.model.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(batch_users, batch_items)
                loss = self.criterion(predictions, batch_ratings.float())
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)
            
            if verbose:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
        
        return losses
    
    def get_embeddings(
        self,
        missing_items: Optional[set] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取训练后的表征
        
        Args:
            missing_items: 缺失物品的索引集合（用于区分初始/完整表征）
            
        Returns:
            all_user_emb: 所有用户表征
            all_item_emb: 所有物品表征
            init_user_emb: 初始用户表征（考虑缺失）
            init_item_emb: 初始物品表征（缺失物品用零向量）
        """
        self.model.eval()
        
        all_user_emb = self.model.get_user_embeddings()
        all_item_emb = self.model.get_item_embeddings()
        
        # 初始表征（考虑缺失物品）
        init_user_emb = all_user_emb.clone()
        init_item_emb = all_item_emb.clone()
        
        if missing_items:
            for item_idx in missing_items:
                init_item_emb[item_idx] = torch.zeros(self.model.embedding_dim)
        
        return all_user_emb, all_item_emb, init_user_emb, init_item_emb
