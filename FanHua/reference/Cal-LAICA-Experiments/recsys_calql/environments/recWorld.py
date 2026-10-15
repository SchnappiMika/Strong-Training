import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass


@dataclass
class EnvConfig:
    """环境配置"""
    history_len: int = 32
    embedding_dim: int = 64
    device: str = 'cpu'
    use_initial_emb: bool = False
    cap_reward_by_phase: bool = True
    release_by_rating_level: bool = False


class RecEnvState:
    """环境状态封装"""

    def __init__(
        self,
        user_idx: int,
        history_items: List[int],
        history_ratings: List[float],
        item_embeddings: torch.Tensor,
        user_embedding: torch.Tensor,
        config: EnvConfig
    ):
        self.user_idx = user_idx
        self.history_items = history_items
        self.history_ratings = history_ratings
        self.item_embeddings = item_embeddings
        self.user_embedding = user_embedding
        self.config = config

    def to_tensor(self) -> torch.Tensor:
        device = self.config.device
        history_len = self.config.history_len
        embedding_dim = self.config.embedding_dim

        if len(self.history_items) == 0:
            history_emb = torch.zeros(history_len, embedding_dim, device=device)
        else:
            recent_items = self.history_items[-history_len:]
            item_indices = torch.tensor(recent_items, device=device)
            history_emb = self.item_embeddings[item_indices]

            actual_len = len(recent_items)
            if actual_len < history_len:
                padding = torch.zeros(
                    history_len - actual_len,
                    embedding_dim,
                    device=device
                )
                history_emb = torch.cat([padding, history_emb], dim=0)

        return history_emb

    def get_state_with_user(self) -> torch.Tensor:
        history_state = self.to_tensor()
        user_emb = self.user_embedding.unsqueeze(0)
        return torch.cat([user_emb, history_state], dim=0)


class RecommendationEnv:
    """
    推荐系统强化学习环境
    
    支持分阶段解锁奖励：
    - 初始时所有物品奖励被遮蔽（返回 0）
    - 每次调用 update_mask() 解锁一批物品的真实奖励
    - 预计算 change_num+1 个遮蔽评分矩阵，按阶段切换
    - episode 结束时返回 mc_return（基于下一阶段矩阵）
    """

    def __init__(
        self,
        rating_matrix: torch.Tensor,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        user_history: Dict[int, List[dict]],
        config: EnvConfig,
        initial_user_emb: Optional[torch.Tensor] = None,
        initial_item_emb: Optional[torch.Tensor] = None,
        candidate_items: Optional[Set[int]] = None,
        change_num: int = 5
    ):
        """
        Args:
            rating_matrix: 完整评分矩阵 (n_users, n_items)
            user_embeddings: 用户表征 (n_users, embedding_dim)
            item_embeddings: 物品表征 (n_items, embedding_dim)
            user_history: 用户历史交互
            config: 环境配置
            initial_user_emb: 初始用户表征
            initial_item_emb: 初始物品表征
            candidate_items: 候选物品集合
            change_num: 奖励分几次解锁
        """
        self.config = config
        self.rating_matrix = rating_matrix.to(config.device)
        self.user_embeddings = user_embeddings.to(config.device)
        self.item_embeddings = item_embeddings.to(config.device)
        self.user_history = user_history

        self.initial_user_emb = (
            initial_user_emb.to(config.device) if initial_user_emb is not None
            else user_embeddings.to(config.device)
        )
        self.initial_item_emb = (
            initial_item_emb.to(config.device) if initial_item_emb is not None
            else item_embeddings.to(config.device)
        )

        self.n_users, self.n_items = rating_matrix.shape
        self.candidate_items = candidate_items or set(range(self.n_items))
        self.change_num = change_num

        # ========== 核心新增：奖励遮蔽系统 ==========
        
        # 当前阶段索引：0 表示初始（全遮蔽），1~change_num 表示逐步解锁
        self.current_phase: int = 0

        # reward_mask: shape (n_items,), True = 该物品奖励可见
        self.reward_mask = torch.zeros(
            self.n_items, dtype=torch.bool, device=config.device
        )

        # 预计算解锁计划和所有阶段的遮蔽评分矩阵
        self._init_unlock_schedule()
        self._precompute_masked_matrices()

        # ========== 原有状态变量 ==========
        self.current_user_idx: int = 0
        self.current_step: int = 0
        self.current_history_items: List[int] = []
        self.current_history_ratings: List[float] = []

        self.user_list = list(user_history.keys())
        self.user_iterator_idx = 0
        self.episodes = 0

    # ------------------------------------------------------------------
    #  解锁计划 & 预计算矩阵
    # ------------------------------------------------------------------

    def _init_unlock_schedule(self):
        """
        将 n_items 个物品均匀分成 change_num 批。
        
        例: n_items=256, change_num=5
            每批 base=51, 余数 r=1
            批次大小: [52, 51, 51, 51, 51]  (前 r 批多 1 个)
        
        unlock_schedule[phase] = 该阶段新解锁的物品索引列表
        # """
        # if not update_mask:
        #     return self.unlock_schedule
        indices = list(range(self.n_items))
        np.random.seed(42)  # 可复现的随机打乱（如需顺序分配可去掉）
        # 如果你希望按原始顺序分配，注释掉下面这行
        # np.random.shuffle(indices)

        base = self.n_items // self.change_num
        remainder = self.n_items % self.change_num

        self.unlock_schedule: Dict[int, List[int]] = {}
        offset = 0
        for phase in range(1, self.change_num + 1):
            batch_size = base + (1 if phase <= remainder else 0)
            self.unlock_schedule[phase] = indices[offset: offset + batch_size]
            offset += batch_size

        # 打印解锁计划
        print(f"📋 奖励解锁计划 (总物品={self.n_items}, 分{self.change_num}次):")
        for phase, items in self.unlock_schedule.items():
            preview = items[:6]
            suffix = f"... (共{len(items)}个)" if len(items) > 6 else f" (共{len(items)}个)"
            print(f"   阶段 {phase}: {preview}{suffix}")

    def _precompute_masked_matrices(self):
        """
        预计算 change_num + 1 个评分矩阵：
        
        - masked_matrices[0]: 全零矩阵（初始，全部遮蔽）
        - masked_matrices[k]: 前 k 批物品解锁后的评分矩阵
        
        遮蔽逻辑：被遮蔽的物品列对应的评分全部置 0
        """
        self.masked_matrices: List[torch.Tensor] = []

        self.masked_matrices.append(
            torch.zeros_like(self.rating_matrix)
        )

        if bool(getattr(self.config, "release_by_rating_level", False)):
            for _ in range(1, self.change_num + 1):
                self.masked_matrices.append(self.rating_matrix.clone())
            print(f"✅ 预计算完成: {len(self.masked_matrices)} 个评分矩阵 "
                  f"(阶段 0~{self.change_num})")
            return

        cumulative_mask = torch.zeros(
            self.n_items, dtype=torch.bool, device=self.config.device
        )

        for phase in range(1, self.change_num + 1):
            for item_idx in self.unlock_schedule[phase]:
                cumulative_mask[item_idx] = True

            masked_matrix = self.rating_matrix.clone()
            masked_matrix[:, ~cumulative_mask] = 0.0
            self.masked_matrices.append(masked_matrix)

        print(f"✅ 预计算完成: {len(self.masked_matrices)} 个评分矩阵 "
              f"(阶段 0~{self.change_num})")

    # ------------------------------------------------------------------
    #  update_mask: 推进阶段
    # ------------------------------------------------------------------

    def update_mask(self, update_mask=False):
        """
        推进到下一阶段，解锁新一批物品的奖励。
        
        - current_phase += 1
        - 更新 reward_mask
        - 后续 step() 自动使用新阶段对应的评分矩阵
        """
        if not update_mask:
            return self.reward_mask
        if self.current_phase >= self.change_num:
            print(f"⚠️ 已达最大阶段 ({self.change_num})，无法继续解锁")
            return

        self.current_phase += 1
        if bool(getattr(self.config, "release_by_rating_level", False)):
            self.reward_mask[:] = True
            unlocked_count = self.reward_mask.sum().item()
            print(f"🔓 update_mask → 阶段 {self.current_phase}/{self.change_num}: "
                  f"奖励等级上限={self.current_phase}, 可用动作 {int(unlocked_count)}/{self.n_items}")
            return self.reward_mask
        
        for item_idx in self.unlock_schedule[self.current_phase]:
            self.reward_mask[item_idx] = True

        unlocked_count = self.reward_mask.sum().item()
        print(f"🔓 update_mask → 阶段 {self.current_phase}/{self.change_num}: "
              f"新解锁 {len(self.unlock_schedule[self.current_phase])} 个, "
              f"累计可用 {int(unlocked_count)}/{self.n_items}")
        return self.reward_mask

    def set_phase(self, phase: int):
        phase = int(phase)
        if phase < 0:
            phase = 0
        if phase > self.change_num:
            phase = self.change_num
        self.current_phase = phase
        self.reward_mask.zero_()
        if bool(getattr(self.config, "release_by_rating_level", False)):
            if phase > 0:
                self.reward_mask[:] = True
            num_valid = int(self.reward_mask.sum().item())
            return self.reward_mask, num_valid
        for p in range(1, phase + 1):
            for item_idx in self.unlock_schedule.get(p, []):
                self.reward_mask[item_idx] = True
        num_valid = int(self.reward_mask.sum().item())
        return self.reward_mask, num_valid

    # ------------------------------------------------------------------
    #  获取当前/下一阶段的评分矩阵
    # ------------------------------------------------------------------

    def get_current_matrix(self) -> torch.Tensor:
        """获取当前阶段的遮蔽评分矩阵"""
        return self.masked_matrices[self.current_phase]

    def get_next_matrix(self) -> torch.Tensor:
        """
        获取下一阶段的评分矩阵（用于 MC return 计算）。
        如果已是最后阶段，返回当前矩阵。
        """
        next_phase = min(self.current_phase + 1, self.change_num)
        return self.masked_matrices[next_phase]

    # ------------------------------------------------------------------
    #  reset & step
    # ------------------------------------------------------------------

    def reset(self, user_idx: Optional[int] = None, is_change_action_space=False) -> RecEnvState:
        """
        重置环境到指定用户的初始状态。
        
        注意：阶段推进请显式调用 update_mask()，reset 不改变阶段。
        """
        if is_change_action_space:
            reward_mask = self.update_mask()
        else:
            reward_mask = None

        if user_idx is None:
            user_idx = self.user_list[self.user_iterator_idx]
            self.user_iterator_idx = (self.user_iterator_idx + 1) % len(self.user_list)

        self.current_user_idx = user_idx
        self.current_step = 0
        self.current_history_items = []
        self.current_history_ratings = []
        self.episodes += 1

        return self._get_state(), reward_mask

    def step(self, action: int) -> Tuple[RecEnvState, float, bool, dict]:
        """
        执行推荐动作。
        
        奖励逻辑：
        - 使用 masked_matrices[current_phase] 查询评分
        - 被遮蔽的物品奖励为 0
        
        每步返回 mc_return：
        - 用下一阶段矩阵计算该动作的评分
        - 若已是最后阶段，使用当前矩阵
        
        episode 结束时额外返回 mc_episode_return（整个 episode 的累积）
        """
        current_matrix = self.get_current_matrix()
        next_matrix = self.get_next_matrix()

        raw_reward = current_matrix[self.current_user_idx, action].item()
        if bool(getattr(self.config, "cap_reward_by_phase", True)):
            reward = min(raw_reward, self.current_phase)
        else:
            reward = float(raw_reward)
        
        # 🆕 下一阶段矩阵对该动作的评分（每步都有）
        next_phase_reward = next_matrix[self.current_user_idx, action].item()

        # 更新历史
        self.current_history_items.append(action)
        self.current_history_ratings.append(reward)
        self.current_step += 1

        # 判断结束
        user_total = 32
        done = self.current_step >= user_total

        next_state = self._get_state()

        info = {
            'user_idx': self.current_user_idx,
            'item_idx': action,
            'step': self.current_step,
            'actual_rating': self.rating_matrix[self.current_user_idx, action].item(),
            'observed_rating': reward,
            'is_masked': not self.reward_mask[action].item(),
            'current_phase': self.current_phase,
            'mc_return': next_phase_reward,  # 🆕 每步都有：下一阶段对该动作的评分
        }

        # episode 结束时额外返回整个 episode 的累积 mc_return
        if done:
            info['mc_episode_return'] = self._compute_mc_return()

        return next_state, reward, done, info, next_phase_reward


    def _compute_mc_return(self) -> float:
        """
        用下一阶段的评分矩阵，对本 episode 的所有推荐动作
        重新计算累积奖励（无折扣 MC return）。
        
        若已是最后阶段，使用当前矩阵。
        """
        next_matrix = self.get_next_matrix()
        user_idx = self.current_user_idx

        mc_return = 0.0
        for item_idx in self.current_history_items:
            mc_return += next_matrix[user_idx, item_idx].item()

        return mc_return

    # ------------------------------------------------------------------
    #  状态 & 动作空间
    # ------------------------------------------------------------------

    def _get_state(self) -> RecEnvState:
        if self.config.use_initial_emb:
            item_emb = self.initial_item_emb
            user_emb = self.initial_user_emb[self.current_user_idx]
        else:
            item_emb = self.item_embeddings
            user_emb = self.user_embeddings[self.current_user_idx]

        return RecEnvState(
            user_idx=self.current_user_idx,
            history_items=self.current_history_items.copy(),
            history_ratings=self.current_history_ratings.copy(),
            item_embeddings=item_emb,
            user_embedding=user_emb,
            config=self.config
        )

    def get_available_actions(self) -> torch.Tensor:
        """获取当前已解锁且未交互的物品"""
        interacted = set(self.current_history_items)
        available = [
            i for i in range(self.n_items)
            if self.reward_mask[i].item() and i not in interacted
            and i in self.candidate_items
        ]
        return torch.tensor(available, device=self.config.device)

    def get_all_candidate_actions(self) -> torch.Tensor:
        """获取所有候选动作（含被遮蔽的）"""
        interacted = set(self.current_history_items)
        available = [i for i in self.candidate_items if i not in interacted]
        return torch.tensor(available, device=self.config.device)

    def get_action_embeddings(self, action_indices: torch.Tensor) -> torch.Tensor:
        if self.config.use_initial_emb:
            return self.initial_item_emb[action_indices]
        return self.item_embeddings[action_indices]

    # ------------------------------------------------------------------
    #  查询接口
    # ------------------------------------------------------------------

    def get_reward(self, user_idx: int, item_idx: int) -> float:
        """查询当前阶段的评分（遮蔽后）"""
        return self.get_current_matrix()[user_idx, item_idx].item()

    def get_true_reward(self, user_idx: int, item_idx: int) -> float:
        """查询真实评分（忽略遮蔽）"""
        return self.rating_matrix[user_idx, item_idx].item()

    def get_unlock_status(self) -> dict:
        unlocked = int(self.reward_mask.sum().item())
        return {
            'current_phase': self.current_phase,
            'total_phases': self.change_num,
            'unlocked_count': unlocked,
            'total_items': self.n_items,
            'unlock_ratio': unlocked / self.n_items
        }

    def iterate_users(self):
        for user_idx in self.user_list:
            state, _ = self.reset(user_idx)
            yield user_idx, state
