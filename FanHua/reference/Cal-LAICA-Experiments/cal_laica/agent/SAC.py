import copy
import numpy as np
import os.path
import torch
from torch import Tensor, long, float32
from torch import nn
from torch import optim
from torch.distributions import Normal
from torch.nn import functional as F
from typing import Tuple, Any

from cal_laica.agent.Agent import AgentBase
from cal_laica.utils.utils import VAE, Trajectory
from cal_laica.utils.utils import build_net, MemoryBuffer, soft_update


class PolicyNetwork(nn.Module):
  def __init__(self,
               state_dim: int = 8,
               action_dim: int = 4,
               hidden_shape: Tuple = (16, 16),
               hidden_activation=nn.ReLU,
               output_activation=nn.ReLU):
    super(PolicyNetwork, self).__init__()
    layers = [state_dim] + list(hidden_shape)
    self._policy = build_net(
      layer_shape=layers,
      hidden_activation=hidden_activation,
      output_activation=output_activation,
    )
    self.mu_layer = nn.Linear(layers[-1], action_dim)
    self.log_std_layer = nn.Linear(layers[-1], action_dim)
    self.LOG_STD_MAX = 2
    self.LOG_STD_MIN = -20

  def forward(self, state: Tensor, deterministic: bool = True, return_log_prob: bool = False) -> Tuple[Tensor, Any]:
    net_out: Tensor = self._policy(state)
    mu: Tensor = self.mu_layer(net_out)
    log_std: Tensor = self.log_std_layer(net_out)
    log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
    std: Tensor = torch.exp(log_std)
    dist = Normal(mu, std)
    u: Tensor = mu if deterministic else dist.rsample()
    action_embedding: Tensor = torch.tanh(u)
    if return_log_prob:
      log_prob = dist.log_prob(u).sum(dim=1, keepdim=True) \
                 - (2.0 * (np.log(2) - u - F.softplus(-2.0 * u))).sum(dim=1, keepdim=True)
    else:
      log_prob = None
    return action_embedding, log_prob

  def evaluate(self, state):
    # with torch.no_grad():
    action_embedding, _ = self.forward(state, deterministic=False, return_log_prob=False)
    mean, log_std = self.mu_layer(self.policy(state)), self.log_std_layer(self.policy(state))
    log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
    std = torch.exp(log_std)
    dist = Normal(mean, std)
    action_log_probs: Tensor = dist.log_prob(action_embedding).sum(dim=-1, keepdim=True)
    dist_entropy = dist.entropy().sum(dim=-1)
    return action_log_probs, dist_entropy


class DoubleCriticNetwork(nn.Module):
  def __init__(self,
               state_dim: int = 8,
               action_dim: int = 4,
               hidden_shape: Tuple = (16, 16)):
    super(DoubleCriticNetwork, self).__init__()
    layers = [state_dim + action_dim] + list(hidden_shape) + [1]
    self.critic1 = build_net(layer_shape=layers,
                             hidden_activation=nn.ReLU,
                             output_activation=nn.Identity)

    self.critic2 = build_net(layer_shape=layers,
                             hidden_activation=nn.ReLU,
                             output_activation=nn.Identity)

  def forward(self, state, action_embedding) -> Tuple[Tensor, Tensor]:
    state_action = torch.cat([state, action_embedding], dim=1)
    q1 = self.critic1.forward(state_action)
    q2 = self.critic2.forward(state_action)
    return q1, q2


class Agent(AgentBase):
  def __init__(self, config, action_mask):
    super(Agent, self).__init__(config)
    self._batch_norm = False
    self._initial_phase = not config.true_embeddings and not config.load_embed and not config.restore
    self.fourier = False
    self._action_mask = action_mask

    # self._state_encoders = Fourier_Network(
    #     config=config
    # )
    if self.fourier:
      state_dim = self._state_encoders.feature_dim
    else:
      state_dim = 4
    self._action_encoder = VAE(
      action_dim=self._action_dim,
      state_dim=state_dim,
      config=config
    ).to(self.config.device)
    hidden_shape = (32, 32)
    self._policy = PolicyNetwork(
      action_dim=self._action_encoder.reduced_action_dim,
      state_dim=state_dim,
      hidden_shape=hidden_shape
    ).to(self.config.device)
    self._critic = DoubleCriticNetwork(
      state_dim=state_dim,
      action_dim=self._action_encoder.reduced_action_dim,
      hidden_shape=hidden_shape
    ).to(self.config.device)

    self._target_critic = DoubleCriticNetwork(
      state_dim=state_dim,
      action_dim=self._action_encoder.reduced_action_dim,
      hidden_shape=hidden_shape
    ).to(self.config.device)

    self._action_encoder_optimizer = optim.AdamW(self._action_encoder.parameters(), lr=1e-3)
    self._policy_optimizer = optim.AdamW(self._policy.parameters(), lr=1e-3)
    self._critic_optimizer = optim.AdamW(self._critic.parameters(), lr=1e-3)

    for param in self._target_critic.parameters():
      param.requires_grad = False

    self._target_entropy = torch.tensor(-self.config.reduced_action_dim, dtype=float32, requires_grad=False)
    self._log_alpha = torch.tensor(np.log(0.01), requires_grad=True, dtype=float32, device=self.config.device)
    self._log_alpha_optim = optim.AdamW([self._log_alpha], lr=self.config.critic_lr)
    self._alpha = self._log_alpha.exp()

    self._current_action_set = torch.where(self._action_mask)[0].numpy()
    self._update_mask(action_mask)
    self._memory = MemoryBuffer(max_len=self.config.buffer_size,
                                state_dim=self._state_dim,
                                action_dim=1,
                                atype=long,
                                stype=float32,
                                config=config,
                                a_emb_dim=self._action_encoder.reduced_action_dim)
    self._offline_memory = MemoryBuffer(
      max_len=self.config.buffer_size,
      state_dim=self._state_dim,
      action_dim=1,
      atype=long,
      stype=float32,
      config=config,
      a_emb_dim=self._action_encoder.reduced_action_dim
    )
    self._save_name = 1
    self.init()
    self.q_values = []
    self.average_q_values = []

  def _update_mask(self, action_mask: Tensor = None):
    assert action_mask is not None
    self._action_mask: Tensor = action_mask.detach()
    self._current_action_set = np.where(self._action_mask.numpy())[0]
    self._action_encoder.update_mask(action_mask)

  def update_mask(self, action_mask: Tensor):
    if not isinstance(action_mask, Tensor):
      action_mask = torch.tensor(action_mask, dtype=torch.uint8, requires_grad=False)
    self._update_mask(action_mask)

  def _reset(self,
             action_mask: Tensor = None,
             change_flag: bool = False,
             reset_memory: bool = True,
             is_save=False,
             is_save_q_values=False
             ):
    if self.q_values and is_save_q_values:
      average_q_value = np.mean(self.q_values)
      self.average_q_values.append(average_q_value)

    self.q_values = []
    if change_flag:
      if is_save:
        torch.save(self.state_dict(), os.path.join(self.config.paths['ckpt'], f"{self._save_name}.pth"))
        self._save_name += 1
      self.update_mask(action_mask)
      self._initial_phase = True
      if reset_memory:
        self._memory.reset()

  def reset(self, action_mask, change_flag, reset_memory=True, is_save=False, is_save_q_values=False):
    self._reset(action_mask, change_flag, reset_memory, is_save, is_save_q_values)

  def get_action(self, state: Tensor, explore: float = 0.1, random_action=False):
    if self._batch_norm:
      self._policy.eval()
    is_random_action = False
    if self._initial_phase or (random_action and np.random.rand() < 0.1):
      action = np.random.choice(self._current_action_set)
      action_embedding = self._action_encoder.get_action_embedding(action)
      is_random_action = True
    else:
      deterministic = np.random.rand() >= explore
      if not isinstance(state, Tensor):
        state = torch.tensor(state, dtype=float32, requires_grad=False, device=self.config.device).view(1, -1)
      else:
        state = state.view(1, -1).to(self.config.device).detach()
      # state = self._state_encoders(state)
      action_embedding, _ = self._policy.forward(state, deterministic, False)
      action_tensor: Tensor = self._action_encoder.get_best_match(action_embedding)
      action = action_tensor.item()
      action_embedding = action_embedding.view(-1)
    self.track_entropy_cont(action_embedding.cpu().data.numpy())

    if is_random_action:
      tmp_state = state.unsqueeze(0)
      tmp_action = action_embedding.unsqueeze(0)
    else:
      tmp_state = state.clone()
      tmp_action = action_embedding.clone().unsqueeze(0)
    q1, q2 = self._critic.forward(tmp_state, tmp_action)
    self.q_values.append(min(q1.item(), q2.item()))
    return action, action_embedding

  def optimizer(self):
    if self._batch_norm:
      self._policy.train()
    s1, a1, a_emb, r1, s2, done, _ = self._memory.sample(batch_size=16)

    with torch.no_grad():
      a_emb_next, log_pi_a_next = self._policy.forward(s2, deterministic=False, return_log_prob=True)
      target_q1, target_q2 = self._target_critic.forward(s2, a_emb_next)
      target_q = torch.min(target_q1, target_q2)
      target_q = r1 + (1 - done) * self._config.gamma * (target_q - self._alpha * log_pi_a_next)

    current_q1, current_q2 = self._critic.forward(s1, a_emb)
    loss_critic = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
    self._clear_gradient()
    loss_critic.backward()
    self._critic_optimizer.step()

    pred_action_embedding, log_pi_a = self._policy.forward(s1, deterministic=False, return_log_prob=True)
    current_q1, current_q2 = self._critic.forward(s1, pred_action_embedding)
    current_q = torch.min(current_q1, current_q2)
    loss_policy = (self._alpha * log_pi_a - current_q).mean() + self._action_encoder.unsupervised_loss(s1,
                                                                                                       a1.view(-1),
                                                                                                       s2)
    self._clear_gradient()
    loss_policy.backward()
    self._policy_optimizer.step()
    alpha_loss = -(self._log_alpha * (log_pi_a + self._target_entropy).detach()).mean()
    self._log_alpha_optim.zero_grad()
    alpha_loss.backward()
    self._log_alpha_optim.step()
    self._alpha = self._log_alpha.exp().detach()
    soft_update(self._target_critic, self._critic, self._config.tau)

  def initial_phase_training(self, max_epochs=-1):
    if self._batch_norm:
      self.policy.eval()

    initial_losses = []
    print("Initial training phase started!!!")
    self._action_encoder_optimizer = optim.Adam(self._action_encoder.parameters(), lr=1e-2)
    for epoch in range(max_epochs):
      losses = []
      for s1, a1, _, _, s2, _, _ in self._memory.batch_sample(batch_size=self.config.sup_batch_size,
                                                              randomize=True):
        loss = self.self_supervised_update(s1, a1, s2)
        losses.append(loss)
      initial_losses.append(np.mean(losses))
      print(f"Epoch {epoch} Loss is {np.mean(initial_losses[-10:])}")

      if len(initial_losses) >= 20 and abs(np.mean(initial_losses[-10:]) - np.mean(initial_losses[-5:])) < 1e-5:
        print("Converged!!!")
        break

    self._initial_phase = False
    self._action_encoder_optimizer = optim.SGD(self._action_encoder.parameters(), lr=self.config.embed_lr)

  def CQL(self, is_cal_ql=False):
    batch_size = 32
    if np.random.rand() < 0.75 or self._offline_memory.size <= batch_size:
      s1, a1, a_emb, r1, s2, done, mc_returns = self._memory.sample(batch_size=batch_size)
    else:
      s1, a1, a_emb, r1, s2, done, mc_returns = self._offline_memory.sample(batch_size=batch_size)
      for idx in range(s1.shape[0]):
        a_emb[idx] = self._action_encoder.get_action_embedding(a1[idx])

    with torch.no_grad():
      a_emb_next, log_pi_a_next = self._policy.forward(s2, deterministic=False, return_log_prob=True)
      target_q1, target_q2 = self._target_critic.forward(s2, a_emb_next)
      target_q = torch.min(target_q1, target_q2)
      target_q = r1 + (1 - done) * self._config.gamma * (target_q - self._alpha * log_pi_a_next)

    current_q1, current_q2 = self._critic.forward(s1, a_emb)
    critic1_loss = torch.mean(F.mse_loss(current_q1, target_q.detach()))
    critic2_loss = torch.mean(F.mse_loss(current_q2, target_q.detach()))
    random_uniform_action_emb = torch.rand(current_q1.shape[0] * 10, a_emb.shape[-1], device=self._config.device,
                                           requires_grad=False, dtype=float32) * 2 - 1
    random_uniform_log_prob = torch.tensor(np.log(0.5 ** a_emb.shape[-1]), dtype=float32)
    num_repeat = int(random_uniform_action_emb.shape[0] // s1.shape[0])
    tmp_s1 = s1.unsqueeze(1).repeat(1, num_repeat, 1).view(s1.shape[0] * num_repeat, s1.shape[1])
    tmp_s2 = s2.unsqueeze(1).repeat(1, num_repeat, 1).view(s2.shape[0] * num_repeat, s2.shape[1])
    random_curr_action_emb, random_curr_log_prob = self._policy.forward(tmp_s1, deterministic=False,
                                                                        return_log_prob=True)
    random_next_action_emb, random_next_log_prob = self._policy.forward(tmp_s2, deterministic=False,
                                                                        return_log_prob=True)
    q1_uniform, q2_uniform = self._critic.forward(state=tmp_s1, action_embedding=random_uniform_action_emb)
    q1_current, q2_current = self._critic.forward(state=tmp_s1, action_embedding=random_curr_action_emb)
    q1_next, q2_next = self._critic.forward(state=tmp_s2, action_embedding=random_next_action_emb)
    if is_cal_ql:
      lower_bounds = mc_returns.unsqueeze(1)
      lower_bounds = lower_bounds.repeat(1, num_repeat, 1).view(mc_returns.shape[0] * num_repeat,
                                                                mc_returns.shape[1])
      q1_current = torch.maximum(q1_current, lower_bounds)
      q2_current = torch.maximum(q2_current, lower_bounds)
      q1_next = torch.maximum(q1_next, lower_bounds)
      q2_next = torch.maximum(q2_next, lower_bounds)
    q1_cat = torch.cat([
      q1_uniform - random_uniform_log_prob,
      q1_current - random_curr_log_prob,
      q1_next - random_next_log_prob,
    ], dim=1)
    q2_cat = torch.cat([
      q2_uniform - random_uniform_log_prob,
      q2_current - random_curr_log_prob,
      q2_next - random_next_log_prob,
    ], dim=1)

    qf1_loss1 = torch.logsumexp(q1_cat, dim=1).mean()
    qf2_loss1 = torch.logsumexp(q2_cat, dim=1).mean()
    qf1_loss2, qf2_loss2 = self._critic.forward(s1, a_emb)
    qf1_loss2 = qf1_loss2.mean()
    qf2_loss2 = qf2_loss2.mean()
    qf1_loss = critic1_loss + num_repeat * (qf1_loss1 - qf1_loss2)
    qf2_loss = critic2_loss + num_repeat * (qf2_loss1 - qf2_loss2)
    loss_critic = qf1_loss + qf2_loss
    self._clear_gradient()
    loss_critic.backward()
    self._critic_optimizer.step()

    pred_action_embedding, log_pi_a = self._policy.forward(s1, deterministic=False, return_log_prob=True)
    current_q1, current_q2 = self._critic.forward(s1, pred_action_embedding)
    current_q = torch.min(current_q1, current_q2)
    loss_policy = (self._alpha * log_pi_a - current_q).mean()
    loss_rep = self._action_encoder.unsupervised_loss(s1, a1.view(-1), s2)
    loss = loss_rep + loss_policy
    self._clear_gradient()
    loss.backward()
    self._policy_optimizer.step()
    self._clear_gradient()
    alpha_loss = -(self._log_alpha * (log_pi_a + self._target_entropy).detach()).mean()
    self._log_alpha_optim.zero_grad()
    alpha_loss.backward()
    self._log_alpha_optim.step()
    self._alpha = self._log_alpha.exp()

    soft_update(self._target_critic, self._critic, self.config.tau)

  def update_offline_memory(self):
    self._offline_memory = copy.deepcopy(self._memory)

  @property
  def initial_phase(self):
    return self._initial_phase

  @property
  def entropy(self):
    return self._entropy

  def update(self, traj: Trajectory = None, memory_reset=False):
    if traj is not None:
      s1, a1, a_emb, r1, s2, done = traj.get_current_traj()
      for idx in range(s1.shape[0]):
        self._memory.add(s1=s1[idx],
                         a1=a1[idx],
                         a_emb=a_emb[idx],
                         r1=r1[idx],
                         s2=s2[idx],
                         done=done[idx])

    if self.initial_phase and self._memory.size >= self.config.buffer_size:
      self.initial_phase_training(max_epochs=self.config.initial_phase_epochs)
      if memory_reset:
        self._memory.reset()

  def offline_update(self, traj: Trajectory = None, action_encoder: VAE = None):
    if traj is not None:
      s1, a1, a_emb, r1, s2, done = traj.get_current_traj()
      for idx in range(s1.shape[0]):
        self._memory.add(s1=s1[idx],
                         a1=a1[idx],
                         a_emb=a_emb[idx],
                         r1=r1[idx],
                         s2=s2[idx],
                         done=done[idx])
    if self.initial_phase and self._memory.size >= self.config.buffer_size:
      self._action_encoder.load_state_dict(action_encoder.state_dict())
      self._initial_phase = False
    elif traj is None:
      self._action_encoder.load_state_dict(action_encoder.state_dict())
      self._initial_phase = False

  def self_supervised_update(self, s1, a1, s2, reg=1):
    if self.fourier:
      s1 = self._state_encoders(s1)
      s2 = self._state_encoders(s2)
    loss = self._action_encoder.unsupervised_loss(s1, a1.view(-1), s2)
    self._clear_gradient()
    loss.backward()
    self._action_encoder_optimizer.step()
    return loss.item()

  @property
  def memory_len(self):
    return self._memory.size

  @property
  def action_encoder(self):
    return self._action_encoder

  @property
  def online_memory_len(self):
    return self._online_memory.size

  @property
  def memory(self):
    return self._memory

  def update_memory(self, agent):
    self._memory = copy.copy(agent.memory)

  @initial_phase.setter
  def initial_phase(self, initial_phase: bool):
    self._initial_phase = initial_phase
