from pathlib import Path

import numpy as np
import torch
from torch import nn

from .artifacts import inside_root
from .envs import TASKS


class Teachers:
    def __init__(self, paths):
        from .agent import Agent
        from .replay import Replay

        self.flappy = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))
        source = torch.load(Path(paths["flappybird"]), map_location="cpu", weights_only=True)
        self.flappy.load_state_dict({name.removeprefix("logits_net."): value for name, value in source["actor"].items()})
        self.flappy.eval().requires_grad_(False)
        path = inside_root(paths["timberman"])
        source = torch.load(path, map_location="cpu", weights_only=True)
        self.timber = Agent(source["config"])
        self.timber.load(path, Replay(source["config"]["capacity"]), np.random.default_rng(0))
        self.timber.policy.eval().requires_grad_(False)

    def actions(self, states, task_ids):
        result = torch.zeros(len(states), dtype=torch.long)
        with torch.inference_mode():
            for index, task in enumerate(TASKS):
                chosen = task_ids == index
                if not chosen.any():
                    continue
                if task == "flappybird":
                    result[chosen] = self.flappy(states[chosen, :3]).argmax(-1)
                else:
                    embeddings, _ = self.timber.policy(states[chosen], task_ids[chosen], deterministic=True)
                    result[chosen] = self.timber.representation.action(embeddings, task_ids[chosen])
        return result


def behavior_loss(policy, representation, states, task_ids, actions, temperature=0.1):
    embedding, _ = policy(states, task_ids, deterministic=True)
    prototypes = representation.embeddings(task_ids).detach()
    distances = (embedding[:, None] - prototypes).square().sum(-1)
    classification = torch.nn.functional.cross_entropy(-distances / temperature, actions, reduction="none")
    selected = prototypes[torch.arange(len(actions)), actions]
    regression = (embedding - selected).square().mean(-1)
    valid = (task_ids != TASKS.index("timberman")) | (states[:, 0] != states[:, -1])
    weights = valid.float()
    return ((classification + regression) * weights).sum() / weights.sum().clamp_min(1)
