import copy
import os
from typing import Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor, float32, long
from torch.distributions import Normal

# ============================================================
# 序列状态编码器：将 item 序列编码为固定维度的状态向量
# ============================================================
class SequenceStateEncoder(nn.Module):
    """
    将用户历史交互的 item 序列编码为固定维度向量。
    支持 GRU / LSTM 两种模式。
    """
    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1,
        rnn_type: str = "GRU",       # "GRU" 或 "LSTM"
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type.upper()

        # item embedding 层（padding_idx=0 表示用 0 做 padding）
        self.item_embedding = nn.Embedding(
            num_embeddings=num_items + 1,  # +1 给 padding
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        # 序列编码器
        rnn_cls = nn.GRU if self.rnn_type == "GRU" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 输出维度就是 hidden_dim
        self.output_dim = hidden_dim

    def forward(self, item_seq: Tensor, seq_lengths: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            item_seq: (batch_size, max_seq_len) — item ID 序列，0 为 padding
            seq_lengths: (batch_size,) — 每条序列的真实长度（可选）
        Returns:
            state: (batch_size, hidden_dim)
        """
        batch_size = item_seq.size(0)

        # ✅ 空序列保护：确保 seq_len 至少为 1，且 item_seq 至少有 1 列
        if item_seq.size(1) == 0:
            # 整个输入序列为空 → 补一个 padding token
            item_seq = torch.zeros(
                batch_size, 1, dtype=item_seq.dtype, device=item_seq.device
            )
            if seq_lengths is not None:
                seq_lengths = torch.ones(batch_size, dtype=seq_lengths.dtype)

        if seq_lengths is not None:
            seq_lengths = seq_lengths.clamp(min=1)
        # (batch_size, max_seq_len, embedding_dim)
        embedded = self.item_embedding(item_seq)

        if seq_lengths is not None:
            # pack 以忽略 padding，提升效率和准确性
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            rnn_out, hidden = self.rnn(packed)
        else:
            rnn_out, hidden = self.rnn(embedded)

        # 取最后一层的 hidden state 作为状态表示
        if self.rnn_type == "LSTM":
            # hidden = (h_n, c_n)，取 h_n
            state = hidden[0][-1]  # (batch_size, hidden_dim)
        else:
            # hidden = h_n
            state = hidden[-1]     # (batch_size, hidden_dim)

        return state
