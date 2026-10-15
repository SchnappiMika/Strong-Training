import copy
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as functional

from .artifacts import inside_root
from .envs import STATE_DIM, TASKS
from .supervision import behavior_loss


RESUME_METADATA_FIELDS = {
    "max_episode_steps",
    "stages",
    "stage_evaluation_episodes",
    "stage_evaluation_steps",
    "experiment_role",
    "experiment_group",
    "initialization",
    "prerequisites_manifest",
    "source_checkpoint",
}


def resume_config(config):
    return {key: value for key, value in config.items() if key not in RESUME_METADATA_FIELDS}


def network(input_dim, hidden_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                         nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                         nn.Linear(hidden_dim, output_dim))


def conditioned(state, task_id):
    return torch.cat([state, functional.one_hot(task_id, len(TASKS)).float()], dim=-1)


class Policy(nn.Module):
    def __init__(self, hidden_dim, embedding_dim, focus_timber=False):
        super().__init__()
        self.focus_timber = focus_timber
        observation_mask = torch.ones(len(TASKS), STATE_DIM)
        observation_mask[TASKS.index("timberman"), 2:11] = 0
        self.register_buffer("observation_mask", observation_mask, persistent=False)
        self.backbone = network(STATE_DIM + len(TASKS), hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, embedding_dim)
        self.log_std = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, state, task_id, deterministic=False):
        if self.focus_timber:
            state = state * self.observation_mask[task_id]
        features = self.backbone(conditioned(state, task_id))
        distribution = Normal(self.mean(features), self.log_std(features).clamp(-5, 2).exp())
        sample = distribution.mean if deterministic else distribution.rsample()
        embedding = sample.tanh()
        correction = 2 * (math.log(2) - sample - functional.softplus(-2 * sample))
        log_probability = (distribution.log_prob(sample) - correction).sum(dim=-1)
        return embedding, log_probability


class Critic(nn.Module):
    def __init__(self, hidden_dim, embedding_dim):
        super().__init__()
        self.first = network(STATE_DIM + len(TASKS) + embedding_dim, hidden_dim, 1)
        self.second = network(STATE_DIM + len(TASKS) + embedding_dim, hidden_dim, 1)

    def forward(self, state, task_id, embedding):
        features = torch.cat([conditioned(state, task_id), embedding], dim=-1)
        return self.first(features).squeeze(-1), self.second(features).squeeze(-1)


class ActionRepresentation(nn.Module):
    def __init__(self, hidden_dim, embedding_dim):
        super().__init__()
        self.prototypes = nn.ParameterList([
            nn.Parameter(torch.stack([torch.full((embedding_dim,), -0.6),
                                     torch.full((embedding_dim,), 0.6)]))
            for task in TASKS
        ])
        self.inverse = network(STATE_DIM * 2 + len(TASKS), hidden_dim, embedding_dim * 2)

    def embeddings(self, task_id):
        all_embeddings = torch.stack([parameter.tanh() for parameter in self.prototypes])
        return all_embeddings[task_id]

    def action(self, embedding, task_id):
        distances = (embedding.unsqueeze(1) - self.embeddings(task_id)).square().sum(-1)
        return distances.argmin(-1)

    def loss(self, state, next_state, task_id, action):
        features = torch.cat([state, next_state, functional.one_hot(task_id, len(TASKS)).float()], dim=-1)
        mean, log_variance = self.inverse(features).chunk(2, dim=-1)
        log_variance = log_variance.clamp(-5, 2)
        embedding = (mean + torch.randn_like(mean) * (0.5 * log_variance).exp()).tanh()
        distances = (embedding.unsqueeze(1) - self.embeddings(task_id)).square().sum(-1)
        classification = functional.cross_entropy(-distances, action)
        divergence = -0.5 * (1 + log_variance - mean.square() - log_variance.exp()).mean()
        return classification + 0.01 * divergence


def calibrated_values(values, returns, valid_bounds):
    return torch.where(valid_bounds.bool(), torch.maximum(values, returns), values)


def conservative_gap(candidates, data_values, returns, valid_bounds, calibrated):
    if calibrated:
        candidates = calibrated_values(candidates, returns[:, None], valid_bounds[:, None])
    return (torch.logsumexp(candidates, dim=1) - math.log(candidates.shape[1]) - data_values).mean()


class Agent:
    def __init__(self, config):
        self.config = dict(config)
        torch.manual_seed(config["seed"])
        hidden = config["hidden_dim"]
        embedding_dim = config["embedding_dim"]
        self.policy = Policy(hidden, embedding_dim, config.get("policy_focus_timber", False))
        self.critic = Critic(hidden, embedding_dim)
        self.target = copy.deepcopy(self.critic).requires_grad_(False)
        self.representation = ActionRepresentation(hidden, embedding_dim)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config["learning_rate"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config["learning_rate"])
        self.representation_optimizer = torch.optim.Adam(self.representation.parameters(), lr=config["representation_lr"])
        self.log_alpha = torch.tensor(math.log(0.05), requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config["learning_rate"])
        self.ready_tasks = []
        self.updates = 0

    def select(self, state, task, generator=None, random_action=False, deterministic=False):
        task_id = torch.tensor([TASKS.index(task)])
        with torch.no_grad():
            if random_action:
                action = int(generator.integers(2))
                embedding = self.representation.embeddings(task_id)[0, action]
            else:
                predicted, _ = self.policy(torch.tensor(state).float().unsqueeze(0), task_id, deterministic)
                action = int(self.representation.action(predicted, task_id).item())
                embedding = predicted[0]
        return action, embedding.tolist()

    def value(self, state, task):
        with torch.no_grad():
            states = torch.tensor(state).float().unsqueeze(0)
            task_id = torch.tensor([TASKS.index(task)])
            embedding, log_probability = self.policy(states, task_id, deterministic=True)
            first, second = self.target(states, task_id, embedding)
            return float((torch.minimum(first, second) - self.log_alpha.detach().exp() * log_probability).item())

    @staticmethod
    def optimize(loss, optimizer, parameters):
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(parameters, 10.0, error_if_nonfinite=True)
        optimizer.step()
        return float(gradient_norm)

    def fit_representation(self, replay, task, steps):
        if task in self.ready_tasks:
            raise ValueError("Do not refit frozen prototypes without re-encoding all historical transitions")
        task_index = TASKS.index(task)
        rows = [row for row in replay.online.rows if row["task"] == task]
        if not rows or {row["action"] for row in rows} != {0, 1}:
            raise ValueError("Representation warmup requires samples for both actions")
        for index, parameter in enumerate(self.representation.prototypes):
            parameter.requires_grad_(index == task_index)
        final_loss = None
        try:
            for iteration in range(steps):
                chosen = [rows[int(replay.generator.integers(len(rows)))] for index in range(self.config["batch_size"])]
                states = torch.tensor([row["state"] for row in chosen])
                next_states = torch.tensor([row["next_state"] for row in chosen])
                tasks = torch.full((len(chosen),), task_index, dtype=torch.long)
                actions = torch.tensor([row["action"] for row in chosen])
                loss = self.representation.loss(states, next_states, tasks, actions)
                self.optimize(loss, self.representation_optimizer, self.representation.parameters())
                final_loss = float(loss.detach())
        finally:
            self.representation.requires_grad_(True)
        embeddings = self.representation.prototypes[task_index].tanh().detach()
        if torch.linalg.vector_norm(embeddings[0] - embeddings[1]) < 0.1:
            raise RuntimeError("Action representation collapsed; do not start SAC training")
        for row in rows:
            row["embedding"] = embeddings[row["action"]].tolist()
        self.ready_tasks.append(task)
        return final_loss

    def parameter_digest(self):
        digest = hashlib.sha256()
        for module in (self.policy, self.critic, self.target, self.representation):
            for name, value in module.state_dict().items():
                digest.update(name.encode("utf-8"))
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        digest.update(self.log_alpha.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    def update(self, batch):
        config = self.config
        states, next_states, task_id = batch["state"], batch["next_state"], batch["task_id"]
        alpha = self.log_alpha.detach().exp()
        with torch.no_grad():
            next_embedding, next_log_probability = self.policy(next_states, task_id)
            next_first, next_second = self.target(next_states, task_id, next_embedding)
            target = batch["reward"] + config["gamma"] * (1 - batch["terminated"]) * (
                torch.minimum(next_first, next_second) - alpha * next_log_probability)
        first, second = self.critic(states, task_id, batch["embedding"])
        td_loss = functional.mse_loss(first, target) + functional.mse_loss(second, target)
        candidate_count = config["cql_samples"]
        repeated_states = states.repeat_interleave(candidate_count, dim=0)
        repeated_tasks = task_id.repeat_interleave(candidate_count)
        with torch.no_grad():
            uniform = torch.rand(len(repeated_states), config["embedding_dim"]) * 2 - 1
            current_actions, current_logp = self.policy(repeated_states, repeated_tasks)
            next_actions, next_logp = self.policy(next_states.repeat_interleave(candidate_count, dim=0), repeated_tasks)
        uniform_first, uniform_second = self.critic(repeated_states, repeated_tasks, uniform)
        current_first, current_second = self.critic(repeated_states, repeated_tasks, current_actions)
        next_first, next_second = self.critic(repeated_states, repeated_tasks, next_actions)
        uniform_logp = -config["embedding_dim"] * math.log(2)
        gaps = []
        active_bounds = config["calibration"] and bool(batch["valid_bound"].any())
        for uniform_values, current_values, next_values, data_values in (
            (uniform_first, current_first, next_first, first),
            (uniform_second, current_second, next_second, second),
        ):
            bounds = batch["mc_return"].repeat_interleave(candidate_count)
            valid = batch["valid_bound"].repeat_interleave(candidate_count).bool()
            if active_bounds:
                current_values = calibrated_values(current_values, bounds, valid)
                next_values = calibrated_values(next_values, bounds, valid)
            candidates = torch.cat([(uniform_values - uniform_logp).view(len(states), -1),
                                    (current_values - current_logp).view(len(states), -1),
                                    (next_values - next_logp).view(len(states), -1)], dim=1)
            gaps.append(conservative_gap(candidates, data_values, batch["mc_return"],
                                         batch["valid_bound"], calibrated=False))
        cql_loss = sum(gaps)
        critic_loss = td_loss + config["cql_weight"] * cql_loss
        critic_norm = self.optimize(critic_loss, self.critic_optimizer, self.critic.parameters())
        self.critic.requires_grad_(False)
        try:
            embedding, log_probability = self.policy(states, task_id)
            policy_first, policy_second = self.critic(states, task_id, embedding)
            policy_loss = (alpha * log_probability - torch.minimum(policy_first, policy_second)).mean()
            mean_embedding, _ = self.policy(states, task_id, deterministic=True)
            if "teacher_action" in batch:
                behavior_objective = behavior_loss(self.policy, self.representation, states, task_id, batch["teacher_action"])
            else:
                behavior_objective = functional.mse_loss(mean_embedding, batch["embedding"])
            policy_loss = policy_loss + config.get("behavior_weight", 0.0) * behavior_objective
            actor_norm = self.optimize(policy_loss, self.policy_optimizer, self.policy.parameters())
        finally:
            self.critic.requires_grad_(True)
        alpha_loss = -(self.log_alpha * (log_probability.detach() - config["embedding_dim"])).mean()
        self.optimize(alpha_loss, self.alpha_optimizer, [self.log_alpha])
        with torch.no_grad():
            self.log_alpha.clamp_(-10, 2)
            for target_parameter, parameter in zip(self.target.parameters(), self.critic.parameters()):
                target_parameter.lerp_(parameter, config["tau"])
        self.updates += 1
        return {"updates": self.updates, "critic_loss": float(critic_loss.detach()),
                "td_loss": float(td_loss.detach()), "cql_loss": float(cql_loss.detach()),
                "policy_loss": float(policy_loss.detach()), "alpha": float(self.log_alpha.detach().exp()),
                "behavior_loss": float(behavior_objective.detach()),
                "q_mean": float(torch.minimum(first, second).detach().mean()),
                "bound_fraction": float(batch["valid_bound"].mean()),
                "history_fraction": batch["history_fraction"],
                "critic_gradient_norm": critic_norm, "actor_gradient_norm": actor_norm}

    def save(self, path, replay, progress, generator):
        path = inside_root(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        state = {"format_version": 1, "algorithm": "task_conditioned_latent_SAC_Cal_LAICA_extension",
                 "config": self.config, "policy": self.policy.state_dict(),
                 "critic": self.critic.state_dict(), "target": self.target.state_dict(),
                 "representation": self.representation.state_dict(), "log_alpha": self.log_alpha.detach(),
                 "policy_optimizer": self.policy_optimizer.state_dict(),
                 "critic_optimizer": self.critic_optimizer.state_dict(),
                 "representation_optimizer": self.representation_optimizer.state_dict(),
                 "alpha_optimizer": self.alpha_optimizer.state_dict(), "updates": self.updates,
                 "ready_tasks": self.ready_tasks, "replay": replay.state_dict(), "progress": progress,
                 "torch_rng": torch.get_rng_state(), "action_rng": generator.bit_generator.state}
        torch.save(state, temporary)
        temporary.replace(path)

    def load(self, path, replay, generator):
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        if (checkpoint["format_version"] != 1
                or resume_config(checkpoint["config"]) != resume_config(self.config)):
            raise ValueError("Checkpoint version or config does not match")
        for name in ("policy", "critic", "target", "representation"):
            getattr(self, name).load_state_dict(checkpoint[name])
        with torch.no_grad():
            self.log_alpha.copy_(checkpoint["log_alpha"])
        for name in ("policy_optimizer", "critic_optimizer", "representation_optimizer", "alpha_optimizer"):
            getattr(self, name).load_state_dict(checkpoint[name])
        self.ready_tasks = checkpoint["ready_tasks"]
        self.updates = checkpoint["updates"]
        replay.load_state_dict(checkpoint["replay"])
        generator.bit_generator.state = checkpoint["action_rng"]
        torch.set_rng_state(checkpoint["torch_rng"])
        return checkpoint["progress"]
