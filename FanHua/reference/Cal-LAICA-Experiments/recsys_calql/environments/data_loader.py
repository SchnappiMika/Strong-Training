import torch
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from pathlib import Path
import hashlib
import json
import os


class ML1MDataLoader:
    def __init__(self, data_path: str = "./data/ml-1m"):
        self.data_path = Path(data_path)
        self.ratings_df: Optional[pd.DataFrame] = None
        self.users_df: Optional[pd.DataFrame] = None
        self.movies_df: Optional[pd.DataFrame] = None
        self.user_id_map: Dict[int, int] = {}
        self.item_id_map: Dict[int, int] = {}
        self.reverse_user_map: Dict[int, int] = {}
        self.reverse_item_map: Dict[int, int] = {}
        self.n_users: int = 0
        self.n_items: int = 0

    def load_ratings(self) -> pd.DataFrame:
        ratings_file = self.data_path / "ratings.dat"
        self.ratings_df = pd.read_csv(
            ratings_file,
            sep='::',
            engine='python',
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            encoding='latin-1'
        )
        self.ratings_df = self.ratings_df.sort_values(['user_id', 'timestamp'])
        print(f"✅ 加载评分数据: {len(self.ratings_df)} 条记录")
        return self.ratings_df

    def load_users(self) -> pd.DataFrame:
        users_file = self.data_path / "users.dat"
        self.users_df = pd.read_csv(
            users_file,
            sep='::',
            engine='python',
            names=['user_id', 'gender', 'age', 'occupation', 'zip_code'],
            encoding='latin-1'
        )
        print(f"✅ 加载用户数据: {len(self.users_df)} 个用户")
        return self.users_df

    def load_movies(self) -> pd.DataFrame:
        movies_file = self.data_path / "movies.dat"
        self.movies_df = pd.read_csv(
            movies_file,
            sep='::',
            engine='python',
            names=['movie_id', 'title', 'genres'],
            encoding='latin-1'
        )
        print(f"✅ 加载电影数据: {len(self.movies_df)} 部电影")
        return self.movies_df

    def build_id_mappings(self) -> None:
        if self.ratings_df is None:
            self.load_ratings()
        unique_users = sorted(self.ratings_df['user_id'].unique())
        unique_items = sorted(self.ratings_df['movie_id'].unique())
        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}
        self.reverse_user_map = {idx: uid for uid, idx in self.user_id_map.items()}
        self.reverse_item_map = {idx: iid for iid, idx in self.item_id_map.items()}
        self.n_users = len(unique_users)
        self.n_items = len(unique_items)
        print(f"✅ ID映射构建完成: {self.n_users} 用户, {self.n_items} 物品")

    def build_rating_matrix(self, device: str = 'cpu') -> torch.Tensor:
        if not self.user_id_map:
            self.build_id_mappings()
        u_idx = self.ratings_df["user_id"].map(self.user_id_map).to_numpy(dtype=np.int64)
        i_idx = self.ratings_df["movie_id"].map(self.item_id_map).to_numpy(dtype=np.int64)
        ratings = self.ratings_df["rating"].to_numpy(dtype=np.float32)
        rating_matrix = torch.zeros((self.n_users, self.n_items), dtype=torch.float32)
        rating_matrix[torch.from_numpy(u_idx), torch.from_numpy(i_idx)] = torch.from_numpy(ratings)
        if device != "cpu":
            rating_matrix = rating_matrix.to(device)
        print(f"✅ 评分矩阵构建完成: shape={rating_matrix.shape}")
        return rating_matrix

    def build_user_history(self) -> Dict[int, list]:
        if not self.user_id_map:
            self.build_id_mappings()
        user_history = {u_idx: [] for u_idx in range(self.n_users)}
        for row in self.ratings_df.itertuples(index=False):
            u_idx = self.user_id_map[int(row.user_id)]
            i_idx = self.item_id_map[int(row.movie_id)]
            user_history[u_idx].append({
                'item_idx': i_idx,
                'rating': float(row.rating),
                'timestamp': int(row.timestamp),
            })
        print(f"✅ 用户历史构建完成")
        return user_history

    def get_train_test_split(
        self,
        test_ratio: float = 0.2,
        by_time: bool = True,
        user_history: Optional[Dict[int, list]] = None,
    ) -> Tuple[Dict, Dict]:
        if user_history is None:
            user_history = self.build_user_history()
        train_history = {}
        test_history = {}
        for u_idx, history in user_history.items():
            if len(history) < 5:
                train_history[u_idx] = history
                test_history[u_idx] = []
            else:
                split_idx = int(len(history) * (1 - test_ratio))
                train_history[u_idx] = history[:split_idx]
                test_history[u_idx] = history[split_idx:]
        print(f"✅ 数据划分完成: 训练/测试比例 = {1-test_ratio:.0%}/{test_ratio:.0%}")
        return train_history, test_history

    def load_all(self) -> 'ML1MDataLoader':
        self.load_ratings()
        self.load_users()
        self.load_movies()
        self.build_id_mappings()
        return self


class AB1MDataLoader:
    def __init__(self, data_path: str = "./data/ab-1m"):
        self.data_path = Path(data_path)
        self.ratings_df: Optional[pd.DataFrame] = None
        self.books_df: Optional[pd.DataFrame] = None
        self.triplet_df: Optional[pd.DataFrame] = None
        self.ab2fb_df: Optional[pd.DataFrame] = None
        self.user_id_map: Dict[int, int] = {}
        self.item_id_map: Dict[int, int] = {}
        self.reverse_user_map: Dict[int, int] = {}
        self.reverse_item_map: Dict[int, int] = {}
        self.n_users: int = 0
        self.n_items: int = 0
        self.item_kg_feature_matrix: Optional[np.ndarray] = None
        self.dataset_profile: Dict[str, float] = {}
        self.enable_cache = os.environ.get("AB1M_ENABLE_CACHE", "1").lower() in {"1", "true", "yes"}
        self.cache_dir = self.data_path / ".cache"
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_ab2fb(self) -> Path:
        cands = [self.data_path / "ab2fb.txt", self.data_path / "ab2fb"]
        for p in cands:
            if p.exists():
                return p
        return cands[0]

    def _build_cache_key(self) -> str:
        files = [
            self.data_path / "ab_1_20.dat",
            self.data_path / "books.dat",
            self.data_path / "triplet.dat",
            self._resolve_ab2fb(),
        ]
        payload = []
        for p in files:
            if p.exists():
                st = p.stat()
                payload.append(f"{p.name}:{st.st_size}:{int(st.st_mtime)}")
            else:
                payload.append(f"{p.name}:missing")
        payload.append(f"n_users={self.n_users}")
        payload.append(f"n_items={self.n_items}")
        joined = "|".join(payload)
        return hashlib.md5(joined.encode("utf-8")).hexdigest()

    def load_ratings(self) -> pd.DataFrame:
        ratings_file = self.data_path / "ab_1_20.dat"
        self.ratings_df = pd.read_csv(
            ratings_file,
            sep='::',
            engine='python',
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            encoding='utf-8'
        )
        self.ratings_df["user_id"] = self.ratings_df["user_id"].astype(int)
        self.ratings_df["item_id"] = self.ratings_df["item_id"].astype(int)
        self.ratings_df["rating"] = self.ratings_df["rating"].astype(float)
        self.ratings_df["timestamp"] = self.ratings_df["timestamp"].astype(int)
        self.ratings_df = self.ratings_df.sort_values(['user_id', 'timestamp'])
        print(f"✅ 加载AB评分数据: {len(self.ratings_df)} 条记录")
        return self.ratings_df

    def load_books(self) -> pd.DataFrame:
        books_file = self.data_path / "books.dat"
        isbns = []
        titles = []
        with open(books_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                p = line.find("::")
                if p <= 0:
                    continue
                isbns.append(line[:p])
                titles.append(line[p + 2:])
        self.books_df = pd.DataFrame({"isbn": isbns, "title": titles})
        print(f"✅ 加载AB图书数据: {len(self.books_df)} 条")
        return self.books_df

    def load_triplet(self) -> pd.DataFrame:
        triplet_file = self.data_path / "triplet.dat"
        self.triplet_df = pd.read_csv(
            triplet_file,
            sep='\t',
            names=['h', 'r', 't'],
            engine='python',
            encoding='utf-8',
            dtype={'h': str, 'r': str, 't': str}
        )
        print(f"✅ 加载AB-KG三元组: {len(self.triplet_df)} 条")
        return self.triplet_df

    def load_ab2fb(self) -> pd.DataFrame:
        ab2fb_file = self._resolve_ab2fb()
        self.ab2fb_df = pd.read_csv(
            ab2fb_file,
            sep='\t',
            names=['isbn', 'kg_id'],
            engine='python',
            encoding='utf-8',
            dtype={'isbn': str, 'kg_id': str}
        )
        print(f"✅ 加载AB-KG映射: {len(self.ab2fb_df)} 条")
        return self.ab2fb_df

    def build_id_mappings(self) -> None:
        if self.ratings_df is None:
            self.load_ratings()
        unique_users = sorted(self.ratings_df['user_id'].unique())
        unique_items = sorted(self.ratings_df['item_id'].unique())
        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}
        self.reverse_user_map = {idx: uid for uid, idx in self.user_id_map.items()}
        self.reverse_item_map = {idx: iid for iid, idx in self.item_id_map.items()}
        self.n_users = len(unique_users)
        self.n_items = len(unique_items)
        print(f"✅ AB-ID映射完成: {self.n_users} 用户, {self.n_items} 物品")

    def build_rating_matrix(self, device: str = 'cpu') -> torch.Tensor:
        if not self.user_id_map:
            self.build_id_mappings()
        u_idx = self.ratings_df["user_id"].map(self.user_id_map).to_numpy(dtype=np.int64)
        i_idx = self.ratings_df["item_id"].map(self.item_id_map).to_numpy(dtype=np.int64)
        ratings = self.ratings_df["rating"].to_numpy(dtype=np.float32)
        rating_matrix = torch.zeros((self.n_users, self.n_items), dtype=torch.float32)
        rating_matrix[torch.from_numpy(u_idx), torch.from_numpy(i_idx)] = torch.from_numpy(ratings)
        if device != "cpu":
            rating_matrix = rating_matrix.to(device)
        print(f"✅ AB评分矩阵完成: shape={rating_matrix.shape}")
        return rating_matrix

    def build_user_history(self) -> Dict[int, list]:
        if not self.user_id_map:
            self.build_id_mappings()
        user_history = {u_idx: [] for u_idx in range(self.n_users)}
        for row in self.ratings_df.itertuples(index=False):
            u_idx = self.user_id_map[int(row.user_id)]
            i_idx = self.item_id_map[int(row.item_id)]
            user_history[u_idx].append({
                'item_idx': i_idx,
                'rating': float(row.rating),
                'timestamp': int(row.timestamp),
            })
        print("✅ AB用户历史完成")
        return user_history

    def _build_item_raw_to_isbn(self) -> Dict[int, str]:
        if self.books_df is None:
            self.load_books()
        raw_to_isbn: Dict[int, str] = {}
        max_row = len(self.books_df)
        for raw_item in self.item_id_map.keys():
            if 1 <= int(raw_item) <= max_row:
                raw_to_isbn[int(raw_item)] = str(self.books_df.iloc[int(raw_item) - 1]["isbn"])
            else:
                raw_to_isbn[int(raw_item)] = str(raw_item)
        return raw_to_isbn

    def build_item_kg_features(self) -> np.ndarray:
        if self.triplet_df is None:
            self.load_triplet()
        if self.ab2fb_df is None:
            self.load_ab2fb()
        if not self.item_id_map:
            self.build_id_mappings()
        cache_key = self._build_cache_key()
        cache_npz = self.cache_dir / f"ab1m_itemkg_{cache_key}.npz"
        cache_json = self.cache_dir / f"ab1m_profile_{cache_key}.json"
        if self.enable_cache and cache_npz.exists() and cache_json.exists():
            cache_data = np.load(cache_npz)
            self.item_kg_feature_matrix = cache_data["item_kg_feature_matrix"].astype(np.float32)
            with open(cache_json, "r", encoding="utf-8") as f:
                self.dataset_profile = json.load(f)
            print(f"✅ AB特征缓存命中: {cache_npz.name}")
            return self.item_kg_feature_matrix

        raw_to_isbn = self._build_item_raw_to_isbn()
        isbn_to_kg = {
            str(row.isbn): str(row.kg_id)
            for row in self.ab2fb_df.itertuples(index=False)
            if isinstance(row.isbn, str) and isinstance(row.kg_id, str)
        }

        h_counts = self.triplet_df["h"].value_counts().to_dict()
        t_counts = self.triplet_df["t"].value_counts().to_dict()
        rel_div = self.triplet_df.groupby("h")["r"].nunique().to_dict()

        item_pop = self.ratings_df.groupby("item_id")["rating"].size().to_dict()
        item_mean = self.ratings_df.groupby("item_id")["rating"].mean().to_dict()
        item_ts_span = (
            self.ratings_df.groupby("item_id")["timestamp"].agg(lambda x: int(x.max()) - int(x.min()))
        ).to_dict()

        n = self.n_items
        feats = np.zeros((n, 7), dtype=np.float32)
        for raw_item, idx in self.item_id_map.items():
            raw_key = str(int(raw_item))
            isbn = raw_to_isbn.get(int(raw_item))
            kg_id = isbn_to_kg.get(str(isbn), None)
            h_deg_raw = float(h_counts.get(raw_key, 0.0))
            t_deg_raw = float(t_counts.get(raw_key, 0.0))
            h_deg_mid = float(h_counts.get(kg_id, 0.0)) if kg_id is not None else 0.0
            t_deg_mid = float(t_counts.get(kg_id, 0.0)) if kg_id is not None else 0.0
            h_deg = max(h_deg_raw, h_deg_mid)
            t_deg = max(t_deg_raw, t_deg_mid)
            total_deg = h_deg + t_deg
            reln = max(float(rel_div.get(raw_key, 0.0)), float(rel_div.get(kg_id, 0.0)) if kg_id is not None else 0.0)
            has_kg = 1.0 if (h_deg > 0 or t_deg > 0 or reln > 0) else 0.0
            pop = float(item_pop.get(int(raw_item), 0.0))
            mean_r = float(item_mean.get(int(raw_item), 0.0))
            span = float(item_ts_span.get(int(raw_item), 0.0))

            feats[idx, 0] = has_kg
            feats[idx, 1] = np.log1p(h_deg)
            feats[idx, 2] = np.log1p(t_deg)
            feats[idx, 3] = np.log1p(total_deg)
            feats[idx, 4] = np.log1p(reln)
            feats[idx, 5] = np.log1p(pop)
            feats[idx, 6] = mean_r + 0.1 * np.log1p(span / 86400.0)

        mean = feats.mean(axis=0, keepdims=True)
        std = feats.std(axis=0, keepdims=True) + 1e-6
        feats = (feats - mean) / std
        self.item_kg_feature_matrix = feats.astype(np.float32)

        kg_covered = 0
        if len(self.item_id_map) > 0:
            kg_covered = int((self.item_kg_feature_matrix[:, 0] > 0).sum())

        density = 0.0
        if self.n_users > 0 and self.n_items > 0:
            density = float(len(self.ratings_df)) / float(self.n_users * self.n_items)
        rating_cnt = self.ratings_df["rating"].value_counts().sort_index().to_dict()
        self.dataset_profile = {
            "n_users": float(self.n_users),
            "n_items": float(self.n_items),
            "n_ratings": float(len(self.ratings_df)),
            "density": float(density),
            "kg_coverage": float(kg_covered / max(1, len(self.item_id_map))),
            "rating_mean": float(self.ratings_df["rating"].mean()),
            "rating_std": float(self.ratings_df["rating"].std()),
            "rating_min": float(self.ratings_df["rating"].min()),
            "rating_max": float(self.ratings_df["rating"].max()),
            "rating_1_ratio": float(rating_cnt.get(1.0, 0) / max(1, len(self.ratings_df))),
            "rating_5_ratio": float(rating_cnt.get(5.0, 0) / max(1, len(self.ratings_df))),
        }
        print(
            f"✅ AB特征完成: 稠密度={self.dataset_profile['density']:.4%}, "
            f"KG覆盖={self.dataset_profile['kg_coverage']:.2%}, "
            f"高分占比={self.dataset_profile['rating_5_ratio']:.2%}"
        )
        if self.enable_cache:
            np.savez_compressed(cache_npz, item_kg_feature_matrix=self.item_kg_feature_matrix)
            with open(cache_json, "w", encoding="utf-8") as f:
                json.dump(self.dataset_profile, f, ensure_ascii=False)
        return self.item_kg_feature_matrix

    def get_train_test_split(
        self,
        test_ratio: float = 0.2,
        by_time: bool = True,
        user_history: Optional[Dict[int, list]] = None,
    ) -> Tuple[Dict, Dict]:
        if user_history is None:
            user_history = self.build_user_history()
        train_history: Dict[int, list] = {}
        test_history: Dict[int, list] = {}
        for u_idx, history in user_history.items():
            if len(history) < 5:
                train_history[u_idx] = history
                test_history[u_idx] = []
                continue
            if by_time:
                history = sorted(history, key=lambda x: x["timestamp"])
            split_idx = int(len(history) * (1 - test_ratio))
            split_idx = min(max(1, split_idx), len(history) - 1)
            train_history[u_idx] = history[:split_idx]
            test_history[u_idx] = history[split_idx:]
        print(f"✅ AB数据划分完成: 训练/测试比例 = {1-test_ratio:.0%}/{test_ratio:.0%}")
        return train_history, test_history

    def load_all(self) -> 'AB1MDataLoader':
        self.load_ratings()
        self.load_books()
        self.load_triplet()
        self.load_ab2fb()
        self.build_id_mappings()
        self.build_item_kg_features()
        return self


def download_ml1m(save_path: str = "./ml-1m"):
    print("""
    ⚠️ ML-1M 数据集需要手动下载:
    
    1. 访问: https://grouplens.org/datasets/movielens/1m/
    2. 下载 ml-1m.zip
    3. 解压到项目目录下的 ml-1m 文件夹
    
    目录结构应为:
    ml-1m/
    ├── ratings.dat
    ├── users.dat
    └── movies.dat
    """)
