import yaml
import json
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, Optional
from recsys_calql.environments.recWorld import RecommendationEnv, EnvConfig
from recsys_calql.environments.data_loader import ML1MDataLoader, AB1MDataLoader
from recsys_calql.environments.funk_svd import FunkSVD, FunkSVDTrainer

class Config:
    def __init__(self,
                 data_path: Optional[str] = None,
                 test_ratio: float = 0.2,
                 by_time: bool = True,

                 embedding_dim: int = 64,
                 history_len: int = 32,
                 device: str = "cpu",
                 use_initial_emb: bool = False,
                 change_num: int = 5,
                 candidate_items: Optional[set] = None,

                 n_epochs: int = 20,
                 batch_size: int = 1024,
                 learning_rate: float = 0.01,
                 weight_decay: float = 0.01,

                 item_embedding_dim: int = 64,
                 state_hidden_dim: int = 128,
                 action_emb_dim: int = 32,
                 policy_hidden: tuple = (256, 256),
                 critic_hidden: tuple = (256, 256),
                 rnn_type: str = "GRU",
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha_init: float = 0.01,
                 lr: float = 1e-3,
                 buffer_size: int = 100000,
                 max_seq_len: int = 50,
                 sac_batch_size: int = 256,

                 cql_alpha: float = 5.0,
                 cql_num_random: int = 10,
                 cql_with_lagrange: bool = False,
                 cql_lagrange_thresh: float = 5.0,
                 use_cal_ql: bool = True,
                 env_name = "ML-1m",
                 legacy_cal_ql_mode: bool = False,
                 kg_feature_alpha: float = 0.25,
                 user_feature_alpha: float = 0.15,
                 min_user_interactions: int = 5
                 ):
        # 数据相关
        self.data_path = self._resolve_data_path(data_path, env_name)
        self.test_ratio = test_ratio
        self.by_time = by_time

        # 环境相关
        self.embedding_dim = embedding_dim
        self.history_len = history_len
        self.device = device
        self.use_initial_emb = use_initial_emb
        self.change_num = change_num
        self.candidate_items = candidate_items

        # FunkSVD训练
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # SAC/CQL/Cal-QL agent相关
        self.item_embedding_dim = item_embedding_dim
        self.state_hidden_dim = state_hidden_dim
        self.action_emb_dim = action_emb_dim
        self.policy_hidden = policy_hidden
        self.critic_hidden = critic_hidden
        self.rnn_type = rnn_type
        self.gamma = gamma
        self.tau = tau
        self.alpha_init = alpha_init
        self.lr = lr
        self.buffer_size = buffer_size
        self.max_seq_len = max_seq_len
        self.sac_batch_size = sac_batch_size

        # CQL/Cal-QL
        self.cql_alpha = cql_alpha
        self.cql_num_random = cql_num_random
        self.cql_with_lagrange = cql_with_lagrange
        self.cql_lagrange_thresh = cql_lagrange_thresh
        self.use_cal_ql = use_cal_ql
        self.legacy_cal_ql_mode = legacy_cal_ql_mode
        self.kg_feature_alpha = kg_feature_alpha
        self.user_feature_alpha = user_feature_alpha
        self.min_user_interactions = min_user_interactions
        self.get_env(env_name)

    @staticmethod
    def _resolve_data_path(data_path: Optional[str], env_name: str) -> str:
        project_root = Path(__file__).resolve().parents[2]
        if data_path is None:
            env_key = str(env_name).upper().replace("_", "-")
            data_name = "ab-1m" if env_key in {"AB-1M", "AB1M"} else "ml-1m"
            return str(project_root / "data" / data_name)
        path = Path(data_path)
        if not path.is_absolute():
            path = project_root / path
        return str(path)

    def update(self, params: Dict[str, Any]):
        """用字典批量更新参数"""
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)

    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path, 'r') as f:
            params = yaml.safe_load(f)
        return cls(**params)

    @classmethod
    def from_json(cls, json_path: str):
        with open(json_path, 'r') as f:
            params = json.load(f)
        return cls(**params)

    def to_dict(self):
        return self.__dict__

    def __repr__(self):
        return f"Config({self.to_dict()})"

    def get_env(self, env_name: str = None, verbose: bool = True):
        """
        根据数据集名称构建推荐 RL 环境
        
        流程：数据加载 → 评分矩阵 → FunkSVD 训练 → 数据划分 → 环境构建
        
        Args:
            env_name: 数据集名称，默认使用 self.env_name
            verbose:  是否打印详细信息
        
        Returns:
            train_env:  训练环境 (RecommendationEnv)
            test_env:   测试环境 (RecommendationEnv)
            meta:       元信息字典
        """
        env_name = str(env_name or getattr(self, 'env_name', 'ML-1M')).strip()
        self.env_name = env_name
        env_key = env_name.upper().replace("_", "-")
        if env_key in {"AB-1M", "AB1M"}:
            self.env, self.test_env, meta = self._build_ab1m_env(verbose=verbose)
        else:
            self.env, self.test_env, meta = self._build_ml1m_env(verbose=verbose)
        self.meta = meta


    def _build_ml1m_env(self, verbose: bool = True):
        """ML-1M 数据集 → FunkSVD → RecommendationEnv"""

        # ==============================================================
        # 📦 Step 1: 加载数据
        # ==============================================================
        if verbose:
            print("=" * 60)
            print("📦 Step 1/5: 加载 ML-1M 数据集")
            print("=" * 60)

        loader = ML1MDataLoader(data_path=self.data_path)
        loader.load_all()

        # ==============================================================
        # 🧮 Step 2: 构建评分矩阵 & 用户历史
        # ==============================================================
        if verbose:
            print("\n" + "=" * 60)
            print("🧮 Step 2/5: 构建评分矩阵与用户历史")
            print("=" * 60)

        rating_matrix = loader.build_rating_matrix(device=self.device)
        user_history = loader.build_user_history()
        if bool(getattr(self, "legacy_cal_ql_mode", False)):
            for u_idx in user_history:
                user_history[u_idx] = sorted(
                    user_history[u_idx],
                    key=lambda x: x["timestamp"]
                )

        if verbose:
            n_ratings = (rating_matrix != 0).sum().item()
            sparsity = 1.0 - n_ratings / (loader.n_users * loader.n_items)
            print(f"   评分数: {int(n_ratings):,}")
            print(f"   稀疏度: {sparsity:.4%}")

        # ==============================================================
        # 🎯 Step 3: 训练 FunkSVD
        # ==============================================================
        if verbose:
            print("\n" + "=" * 60)
            print(f"🎯 Step 3/5: 训练 FunkSVD "
                f"(dim={self.embedding_dim}, epochs={self.n_epochs})")
            print("=" * 60)

        svd_model = FunkSVD(
            n_users=loader.n_users,
            n_items=loader.n_items,
            embedding_dim=self.embedding_dim,
            device=self.device,
        )

        trainer = FunkSVDTrainer(
            model=svd_model,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        svd_losses = trainer.train(
            rating_matrix=rating_matrix,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            verbose=verbose,
        )

        all_user_emb, all_item_emb, init_user_emb, init_item_emb = (
            trainer.get_embeddings()
        )

        if verbose:
            print(f"\n   用户嵌入: {all_user_emb.shape}")
            print(f"   物品嵌入: {all_item_emb.shape}")
            print(f"   最终 SVD Loss: {svd_losses[-1]:.4f}")

        # ==============================================================
        # ✂️ Step 4: 划分训练/测试集
        # ==============================================================
        if verbose:
            print("\n" + "=" * 60)
            print(f"✂️ Step 4/5: 划分训练/测试集 "
                f"(test_ratio={self.test_ratio})")
            print("=" * 60)

        if bool(getattr(self, "legacy_cal_ql_mode", False)):
            train_history, test_history = loader.get_train_test_split(
                test_ratio=self.test_ratio,
                by_time=self.by_time,
            )
        else:
            train_history, test_history = loader.get_train_test_split(
                test_ratio=self.test_ratio,
                by_time=self.by_time,
                user_history=user_history,
            )

        train_interactions = sum(len(v) for v in train_history.values())
        test_interactions = sum(len(v) for v in test_history.values())
        test_users_with_data = sum(1 for v in test_history.values() if len(v) > 0)

        if verbose:
            print(f"   训练交互数: {train_interactions:,}")
            print(f"   测试交互数: {test_interactions:,}")
            print(f"   有测试数据的用户: {test_users_with_data:,}")

        # ==============================================================
        # 🏗️ Step 5: 构建 RL 环境
        # ==============================================================
        if verbose:
            print("\n" + "=" * 60)
            print(f"🏗️ Step 5/5: 构建推荐 RL 环境 "
                f"(change_num={self.change_num})")
            print("=" * 60)

        # ---- 训练环境配置 ----
        train_env_config = EnvConfig(
            history_len=self.history_len,
            embedding_dim=self.embedding_dim,
            device=self.device,
            use_initial_emb=self.use_initial_emb,
            cap_reward_by_phase=(not bool(getattr(self, "legacy_cal_ql_mode", False))),
        )

        candidate_items = self.candidate_items or set(range(loader.n_items))

        train_env = RecommendationEnv(
            rating_matrix=rating_matrix,
            user_embeddings=all_user_emb,
            item_embeddings=all_item_emb,
            user_history=train_history,
            config=train_env_config,
            initial_user_emb=init_user_emb,
            initial_item_emb=init_item_emb,
            candidate_items=candidate_items,
            change_num=self.change_num,
        )

        # ---- 测试环境（一次性全部解锁）----
        test_env_config = EnvConfig(
            history_len=self.history_len,
            embedding_dim=self.embedding_dim,
            device=self.device,
            use_initial_emb=False,
            cap_reward_by_phase=(not bool(getattr(self, "legacy_cal_ql_mode", False))),
        )

        test_env = RecommendationEnv(
            rating_matrix=rating_matrix,
            user_embeddings=all_user_emb,
            item_embeddings=all_item_emb,
            user_history=test_history,
            config=test_env_config,
            initial_user_emb=init_user_emb,
            initial_item_emb=init_item_emb,
            candidate_items=candidate_items,
            change_num=1,
        )
        test_env.update_mask(update_mask=True)

        if verbose:
            print(f"\n   训练环境: {len(train_history)} 用户, "
                f"{self.change_num} 阶段解锁")
            print(f"   测试环境: {test_users_with_data} 用户, 全部解锁")

        # ==============================================================
        # 📋 元信息
        # ==============================================================
        meta = {
            # 数据
            'loader': loader,
            'n_users': loader.n_users,
            'n_items': loader.n_items,
            'rating_matrix': rating_matrix,
            'train_history': train_history,
            'test_history': test_history,
            # 嵌入
            'user_embeddings': all_user_emb,
            'item_embeddings': all_item_emb,
            'initial_user_emb': init_user_emb,
            'initial_item_emb': init_item_emb,
            # 模型
            'svd_model': svd_model,
            'svd_losses': svd_losses,
            # 配置
            'embedding_dim': self.embedding_dim,
            'env_config': train_env_config,
        }

        if verbose:
            print("\n" + "=" * 60)
            print("✅ 环境初始化完成！")
            print(f"   用户数: {loader.n_users:,} | 物品数: {loader.n_items:,}")
            print(f"   嵌入维度: {self.embedding_dim} | 历史长度: {self.history_len}")
            print(f"   训练阶段: {self.change_num} | 设备: {self.device}")
            print("=" * 60)

        return train_env, test_env, meta

    def _feature_project(self, features, target_dim: int, alpha: float, seed: int = 12345):
        if features is None:
            return None
        if isinstance(features, torch.Tensor):
            f = features.detach().cpu().float()
        else:
            f = torch.tensor(features, dtype=torch.float32)
        if f.numel() == 0:
            return None
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        proj = torch.randn((f.shape[1], target_dim), generator=gen, dtype=torch.float32) / np.sqrt(max(1, f.shape[1]))
        out = f @ proj
        out = out / (out.norm(dim=1, keepdim=True) + 1e-6)
        return float(alpha) * out

    def _build_user_stat_features(self, train_history: Dict[int, list], n_users: int) -> np.ndarray:
        feats = np.zeros((n_users, 4), dtype=np.float32)
        for u_idx in range(n_users):
            hist = train_history.get(u_idx, [])
            if len(hist) == 0:
                continue
            ratings = np.array([x["rating"] for x in hist], dtype=np.float32)
            times = np.array([x["timestamp"] for x in hist], dtype=np.float32)
            feats[u_idx, 0] = np.log1p(float(len(hist)))
            feats[u_idx, 1] = float(ratings.mean())
            feats[u_idx, 2] = float(ratings.std()) if len(ratings) > 1 else 0.0
            feats[u_idx, 3] = np.log1p(float(times.max() - times.min()) / 86400.0)
        mean = feats.mean(axis=0, keepdims=True)
        std = feats.std(axis=0, keepdims=True) + 1e-6
        return ((feats - mean) / std).astype(np.float32)

    def _build_ab1m_env(self, verbose: bool = True):
        if verbose:
            print("=" * 60)
            print("📦 Step 1/5: 加载 AB-1M 数据集")
            print("=" * 60)

        loader = AB1MDataLoader(data_path=self.data_path)
        loader.load_all()

        if verbose:
            print("\n" + "=" * 60)
            print("🧮 Step 2/5: 构建评分矩阵与用户历史")
            print("=" * 60)

        rating_matrix = loader.build_rating_matrix(device=self.device)
        user_history = loader.build_user_history()
        min_user_interactions = int(getattr(self, "min_user_interactions", 5))
        if min_user_interactions > 1:
            user_history = {u: h for u, h in user_history.items() if len(h) >= min_user_interactions}
            kept_users = sorted(user_history.keys())
            if len(kept_users) > 0:
                user_remap = {old: new for new, old in enumerate(kept_users)}
                remapped_history: Dict[int, list] = {}
                remapped_rows = []
                for old_u, hist in user_history.items():
                    new_u = user_remap[old_u]
                    remapped = []
                    for x in hist:
                        remapped.append({"item_idx": x["item_idx"], "rating": x["rating"], "timestamp": x["timestamp"]})
                        remapped_rows.append((new_u, x["item_idx"], x["rating"]))
                    remapped_history[new_u] = remapped
                user_history = remapped_history
                new_n_users = len(kept_users)
                new_rating = torch.zeros((new_n_users, loader.n_items), dtype=torch.float32)
                if len(remapped_rows) > 0:
                    u_idx = torch.tensor([x[0] for x in remapped_rows], dtype=torch.long)
                    i_idx = torch.tensor([x[1] for x in remapped_rows], dtype=torch.long)
                    val = torch.tensor([x[2] for x in remapped_rows], dtype=torch.float32)
                    new_rating[u_idx, i_idx] = val
                rating_matrix = new_rating.to(self.device)
                loader.n_users = new_n_users

        if verbose:
            n_ratings = (rating_matrix != 0).sum().item()
            sparsity = 1.0 - n_ratings / (max(1, loader.n_users) * max(1, loader.n_items))
            print(f"   评分数: {int(n_ratings):,}")
            print(f"   稀疏度: {sparsity:.4%}")
            if hasattr(loader, "dataset_profile") and len(loader.dataset_profile) > 0:
                print(f"   KG覆盖率: {loader.dataset_profile.get('kg_coverage', 0.0):.2%}")
                print(f"   高分占比(5分): {loader.dataset_profile.get('rating_5_ratio', 0.0):.2%}")

        if verbose:
            print("\n" + "=" * 60)
            print(f"🎯 Step 3/5: 训练 FunkSVD "
                f"(dim={self.embedding_dim}, epochs={self.n_epochs})")
            print("=" * 60)

        svd_model = FunkSVD(
            n_users=loader.n_users,
            n_items=loader.n_items,
            embedding_dim=self.embedding_dim,
            device=self.device,
        )

        trainer = FunkSVDTrainer(
            model=svd_model,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        svd_losses = trainer.train(
            rating_matrix=rating_matrix,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            verbose=verbose,
        )

        all_user_emb, all_item_emb, init_user_emb, init_item_emb = trainer.get_embeddings()
        item_feat = loader.item_kg_feature_matrix if hasattr(loader, "item_kg_feature_matrix") else None
        item_proj = self._feature_project(item_feat, self.embedding_dim, float(getattr(self, "kg_feature_alpha", 0.0)))
        if item_proj is not None and item_proj.shape[0] == all_item_emb.shape[0]:
            item_proj = item_proj.to(all_item_emb.device)
            all_item_emb = all_item_emb + item_proj
            init_item_emb = init_item_emb + item_proj

        if verbose:
            print(f"\n   用户嵌入: {all_user_emb.shape}")
            print(f"   物品嵌入: {all_item_emb.shape}")
            print(f"   最终 SVD Loss: {svd_losses[-1]:.4f}")

        if verbose:
            print("\n" + "=" * 60)
            print(f"✂️ Step 4/5: 划分训练/测试集 "
                f"(test_ratio={self.test_ratio})")
            print("=" * 60)

        train_history, test_history = loader.get_train_test_split(
            test_ratio=self.test_ratio,
            by_time=self.by_time,
            user_history=user_history,
        )

        user_feat = self._build_user_stat_features(train_history, loader.n_users)
        user_proj = self._feature_project(user_feat, self.embedding_dim, float(getattr(self, "user_feature_alpha", 0.0)))
        if user_proj is not None and user_proj.shape[0] == all_user_emb.shape[0]:
            user_proj = user_proj.to(all_user_emb.device)
            all_user_emb = all_user_emb + user_proj
            init_user_emb = init_user_emb + user_proj

        train_interactions = sum(len(v) for v in train_history.values())
        test_interactions = sum(len(v) for v in test_history.values())
        test_users_with_data = sum(1 for v in test_history.values() if len(v) > 0)

        if verbose:
            print(f"   训练交互数: {train_interactions:,}")
            print(f"   测试交互数: {test_interactions:,}")
            print(f"   有测试数据的用户: {test_users_with_data:,}")

        if verbose:
            print("\n" + "=" * 60)
            print(f"🏗️ Step 5/5: 构建推荐 RL 环境 "
                f"(change_num={self.change_num})")
            print("=" * 60)

        train_env_config = EnvConfig(
            history_len=self.history_len,
            embedding_dim=self.embedding_dim,
            device=self.device,
            use_initial_emb=self.use_initial_emb,
            cap_reward_by_phase=True,
            release_by_rating_level=True,
        )

        candidate_items = self.candidate_items or set(range(loader.n_items))

        train_env = RecommendationEnv(
            rating_matrix=rating_matrix,
            user_embeddings=all_user_emb,
            item_embeddings=all_item_emb,
            user_history=train_history,
            config=train_env_config,
            initial_user_emb=init_user_emb,
            initial_item_emb=init_item_emb,
            candidate_items=candidate_items,
            change_num=self.change_num,
        )

        test_env_config = EnvConfig(
            history_len=self.history_len,
            embedding_dim=self.embedding_dim,
            device=self.device,
            use_initial_emb=False,
            cap_reward_by_phase=True,
            release_by_rating_level=True,
        )

        test_env = RecommendationEnv(
            rating_matrix=rating_matrix,
            user_embeddings=all_user_emb,
            item_embeddings=all_item_emb,
            user_history=test_history,
            config=test_env_config,
            initial_user_emb=init_user_emb,
            initial_item_emb=init_item_emb,
            candidate_items=candidate_items,
            change_num=1,
        )
        test_env.update_mask(update_mask=True)

        if verbose:
            print(f"\n   训练环境: {len(train_history)} 用户, "
                f"{self.change_num} 阶段解锁")
            print(f"   测试环境: {test_users_with_data} 用户, 全部解锁")

        meta = {
            'loader': loader,
            'n_users': loader.n_users,
            'n_items': loader.n_items,
            'rating_matrix': rating_matrix,
            'train_history': train_history,
            'test_history': test_history,
            'user_embeddings': all_user_emb,
            'item_embeddings': all_item_emb,
            'initial_user_emb': init_user_emb,
            'initial_item_emb': init_item_emb,
            'svd_model': svd_model,
            'svd_losses': svd_losses,
            'embedding_dim': self.embedding_dim,
            'env_config': train_env_config,
            'dataset_profile': getattr(loader, "dataset_profile", {}),
            'item_kg_feature_matrix': item_feat,
            'user_stat_feature_matrix': user_feat,
        }

        if verbose:
            print("\n" + "=" * 60)
            print("✅ AB环境初始化完成！")
            print(f"   用户数: {loader.n_users:,} | 物品数: {loader.n_items:,}")
            print(f"   嵌入维度: {self.embedding_dim} | 历史长度: {self.history_len}")
            print(f"   训练阶段: {self.change_num} | 设备: {self.device}")
            print("=" * 60)

        return train_env, test_env, meta
