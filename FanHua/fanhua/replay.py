import numpy as np
import torch

from .envs import TASKS


def validate_transitions(rows, embedding_dim):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Dataset must contain a non-empty transition list")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("task") not in TASKS:
            raise ValueError(f"Invalid task in dataset row {index}")
        if type(row.get("action")) is not int or row["action"] not in (0, 1):
            raise ValueError(f"Invalid action in dataset row {index}")
        for field, size in (("state", 12), ("next_state", 12), ("embedding", embedding_dim)):
            values = row.get(field)
            if not isinstance(values, list) or len(values) != size:
                raise ValueError(f"Invalid {field} shape in dataset row {index}")
            if any(type(value) not in (int, float) or not np.isfinite(value) for value in values):
                raise ValueError(f"Non-numeric {field} in dataset row {index}")
        for field in ("reward", "mc_return"):
            if type(row.get(field)) not in (int, float) or not np.isfinite(row[field]):
                raise ValueError(f"Invalid {field} in dataset row {index}")
        for field in ("terminated", "truncated", "valid_bound"):
            if type(row.get(field)) is not bool:
                raise ValueError(f"Invalid {field} in dataset row {index}")
        if row["terminated"] and row["truncated"]:
            raise ValueError(f"Dataset row {index} cannot both terminate and truncate")


class Ring:
    def __init__(self, capacity):
        self.capacity = capacity
        self.rows = []
        self.position = 0

    def append(self, row):
        if len(self.rows) < self.capacity:
            self.rows.append(row)
        else:
            self.rows[self.position] = row
        self.position = (self.position + 1) % self.capacity

    def draw(self, generator):
        return self.rows[int(generator.integers(len(self.rows)))]

    def state_dict(self):
        return {"capacity": self.capacity, "rows": self.rows, "position": self.position}

    def load_state_dict(self, state):
        self.capacity = state["capacity"]
        self.rows = state["rows"]
        self.position = state["position"]


class Replay:
    def __init__(self, capacity, history_probability=0.25, seed=0):
        if capacity <= 0 or not 0 <= history_probability <= 1:
            raise ValueError("Invalid replay configuration")
        self.online = Ring(capacity)
        self.history = {task: Ring(capacity) for task in TASKS}
        self.history_probability = history_probability
        self.generator = np.random.default_rng(seed)

    def begin_stage(self):
        if self.history_probability > 0:
            for row in self.online.rows:
                self.history[row["task"]].append(row)
        self.online = Ring(self.online.capacity)

    def add_episode(self, rows, gamma, bootstrap=0.0, complete=True):
        discounted_return = float(bootstrap)
        completed_rows = []
        for row in reversed(rows):
            if row["terminated"]:
                discounted_return = 0.0
            discounted_return = row["reward"] + gamma * discounted_return
            stored = dict(row)
            stored["mc_return"] = discounted_return
            stored["valid_bound"] = bool(complete and bootstrap == 0 and rows[-1]["terminated"])
            completed_rows.append(stored)
        for row in reversed(completed_rows):
            self.online.append(row)

    def sample(self, batch_size):
        available_history = [pool for pool in self.history.values() if pool.rows]
        if not self.online.rows and not available_history:
            raise ValueError("Replay is empty")
        rows = []
        history_count = 0
        for index in range(batch_size):
            use_history = available_history and (not self.online.rows or self.generator.random() < self.history_probability)
            if use_history:
                pool = available_history[int(self.generator.integers(len(available_history)))]
                rows.append(pool.draw(self.generator))
                history_count += 1
            else:
                rows.append(self.online.draw(self.generator))
        fields = ("state", "next_state", "embedding", "reward", "terminated", "mc_return", "valid_bound")
        batch = {field: torch.tensor([row[field] for row in rows], dtype=torch.float32) for field in fields}
        batch["task_id"] = torch.tensor([TASKS.index(row["task"]) for row in rows], dtype=torch.long)
        batch["action"] = torch.tensor([row["action"] for row in rows], dtype=torch.long)
        batch["history_fraction"] = history_count / batch_size
        return batch

    def state_dict(self):
        return {"online": self.online.state_dict(),
                "history": {task: pool.state_dict() for task, pool in self.history.items()},
                "history_probability": self.history_probability,
                "rng": self.generator.bit_generator.state}

    def load_state_dict(self, state):
        self.online.load_state_dict(state["online"])
        for task, pool in state["history"].items():
            self.history[task].load_state_dict(pool)
        self.history_probability = state["history_probability"]
        self.generator.bit_generator.state = state["rng"]
