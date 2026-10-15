import itertools
import math
import numpy as np
import shutil
import statistics
import sys
import torch
from matplotlib import pyplot as plt
from os import fsync
from os import listdir
from os import mkdir
from os import path
from time import time
from torch import Tensor, uint8, float32, long
from torch import nn
from torch import tensor
from torch.nn import functional as F
from typing import Tuple


def check_n_create(dir_path, overwrite=False):
  try:
    if not path.exists(dir_path):
      mkdir(dir_path)
    else:
      if overwrite:
        shutil.rmtree(dir_path)
        mkdir(dir_path)
  except FileExistsError:
    print("\n ##### Warning File Exists... perhaps multi-threading error? \n")


def create_directory_tree(dir_path):
  dir_path = str.split(dir_path, sep='/')[1:-1]
  for i in range(len(dir_path)):
    check_n_create(path.join('/', *(dir_path[:i + 1])))


def build_net(layer_shape, hidden_activation, output_activation):
  layers = []  # 初始化一个空列表，用于存放网络的各层及其激活函数
  for j in range(len(layer_shape) - 1):  # 遍历 layer_shape 列表，构建各层
    # 如果当前层不是最后一层，使用 hidden_activation 激活函数；否则使用 output_activation 激活函数
    act = hidden_activation if j < len(layer_shape) - 2 else output_activation
    # 添加一个线性层和对应的激活函数到 layers 列表
    layers += [nn.Linear(layer_shape[j], layer_shape[j + 1]), act()]
  return nn.Sequential(*layers)  # 使用 nn.Sequential 将所有层和激活函数组合成一个顺序容器并返回


def search(dir_name, name, exact=False):
  all_files = listdir(dir_name)
  for file in all_files:
    if exact and name == file:
      return path.join(dir_name, name)
    if not exact and name in file:
      return path.join(dir_name, name)
  else:
    # recursive scan
    for file in all_files:
      if file == 'Experiments':
        continue
      _path = path.join(dir_name, file)
      if path.isdir(_path):
        location = search(_path, name, exact)
        if location:
          return location


def weight_init(m):
  if isinstance(m, nn.Linear):
    variance = 0
    m.weight.data.normal_(0.0, variance)
  elif isinstance(m, nn.BatchNorm2d):
    m.weight.data.fill_(1)
    m.bias.data.zero_()


def pair_distance(x: Tensor, y: Tensor):
  x_norm = torch.pow(x, 2).sum(1).view(-1, 1)
  y_norm = torch.pow(y, 2).sum(1).view(1, -1)
  y_t = torch.transpose(y, 0, 1)
  dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
  return dist


def binary_encoding(num: int, size: int) -> Tensor:
  binary_encode = torch.zeros(size, dtype=float32, requires_grad=False)
  idx = -1
  while num > 0:
    binary_encode[idx] = num % 2
    num //= 2
    idx -= 1
  return binary_encode.detach()


def calculate_mean_and_stddev(data):
  # 计算平均值
  mean = statistics.mean(data)

  # 计算标准差
  stddev = statistics.stdev(data)

  return mean, stddev


def save_plots(rewards, std_results, config, episodes=0):
  # Convert lists to NumPy arrays if they are not already
  rewards = np.array(rewards)
  std_results = np.array(std_results)

  # Save rewards array
  np.save(config.paths['results'] + f"rewards", rewards)
  np.save(config.paths['results'] + f"rewards_std", std_results)

  # if config.debug:
  #     # Save the heatmap
  #     plt.figure()
  #     plt.title("Exploration Heatmap")
  #     plt.xlabel("100x position in x coordinate")
  #     plt.ylabel("100x position in y coordinate")
  #     plt.imshow(config.env.heatmap, cmap='hot', interpolation='nearest', origin='lower')
  #     plt.savefig(config.paths['results'] + f'{episodes} heatmap.png')
  #     np.save(config.paths['results'] + f"{episodes} heatmap", config.env.heatmap)
  #     config.env.heatmap.fill(0)  # reset the heat map
  #     plt.close()

  # Plot performance curve with shaded standard deviation
  plt.figure()
  plt.ylabel("Total return")
  plt.xlabel("Episode")
  plt.title("Performance")

  # Plot the mean rewards
  plt.plot(rewards, label="Mean Return")

  # Add shaded region for standard deviation
  plt.fill_between(
    range(len(rewards)),  # x-axis values
    rewards - std_results,  # lower bound of the shaded region
    rewards + std_results,  # upper bound of the shaded region
    alpha=0.3,  # transparency of the shaded region
    label="Standard Deviation"
  )

  plt.legend()  # Show legend
  plt.savefig(config.paths['results'] + "performance.png")
  plt.close()


def soft_update(target, source, tau):
  for target_param, param in zip(target.parameters(), source.parameters()):
    target_param.data.copy_(
      target_param.data * (1.0 - tau) + param.data * tau
    )


class VAE(nn.Module):
  def __init__(self,
               state_dim,
               action_dim,
               config):
    super(VAE, self).__init__()
    self._state_dim = state_dim
    self._action_dim = action_dim
    self._config = config
    self._norm_const = tensor(np.log(self._action_dim), dtype=float32)
    self._action_mask = torch.zeros(self._action_dim, dtype=uint8)

    if config.true_embeddings:
      embeddings: Tensor = config.env.get_embeddings()
      self.reduced_action_dim = embeddings.shape[1]
      max_i, min_i = torch.max(embeddings), torch.min(embeddings)
      embeddings = ((embeddings - min_i) / (max_i - min_i)) * 2 - 1
      self.embeddings: Tensor = embeddings.detach().requires_grad_(False)
    else:
      self.reduced_action_dim = self._config.reduced_action_dim
      init_embeddings = torch.rand(self._action_dim, self.reduced_action_dim) * 2 - 1
      self.embeddings = nn.Parameter(init_embeddings.type(float32), requires_grad=True)

    self._fc_mean = nn.Linear(self._state_dim * 2, self.reduced_action_dim)
    self._fc_std = nn.Linear(self._state_dim * 2, self.reduced_action_dim)
    self.optim = config.optim(self.parameters(), lr=self._config.embed_lr)
    print("Action Representation: ", [(name, param.shape) for name, param in self.named_parameters()])

  def _update_mask(self, action_mask: Tensor) -> None:
    assert isinstance(action_mask, Tensor)
    self._action_mask = action_mask.detach()
    self._norm_const = torch.log(torch.sum(action_mask))

  def _get_match_scores(self, action_embedding: Tensor) -> Tensor:
    embeddings = torch.tanh(self.embeddings) \
      if not self._config.true_embeddings else self.embeddings
    similarity = -pair_distance(action_embedding, embeddings)
    similarity[:, self._action_mask == 0] = float('-inf')
    return similarity

  def _get_best_match(self, action_embedding: Tensor) -> int:
    similarity = self._get_match_scores(action_embedding)
    _, best_action = torch.max(similarity, dim=1)
    return best_action

  def get_action_embedding(self, action: Tensor) -> Tensor:
    action_embedding = torch.tanh(self.embeddings[action]) \
      if not self._config.true_embeddings else self.embeddings[action]
    return action_embedding

  def forward(self, state1: Tensor, state2: Tensor):
    state_cat = torch.cat([state1, state2], dim=1)
    mu: Tensor = self._fc_mean(state_cat)
    log_std: Tensor = self._fc_std(state_cat)
    std = log_std.div(2).exp()
    eps = torch.randn_like(std)
    x = mu + std * eps
    x = torch.tanh(x)
    return x, mu, log_std

  def update_mask(self, action_mask: Tensor) -> None:
    self._update_mask(action_mask)

  def get_best_match(self, action_embedding: Tensor):
    return self._get_best_match(action_embedding)

  def custom_weight_init(self):
    for m in self.modules():
      weight_init(m)

  def unsupervised_loss(self, s1, a, s2, normalized=True):
    x, mu, log_std = self.forward(state1=s1,
                                  state2=s2)
    similarity = self._get_match_scores(x)
    klds = -0.5 * (1 + log_std - mu.pow(2) - log_std.exp())
    if normalized:
      loss = F.cross_entropy(similarity, a, reduction='mean') / self._norm_const \
             + self.config.emb_reg * torch.pow(self.embeddings, 2).mean() / (1.0 * self.reduced_action_dim) \
             + self.config.beta_vae * klds.mean()
    else:
      loss = F.cross_entropy(similarity, a, reduction='mean') \
             + self.config.emb_reg * torch.pow(self.embeddings, 2).mean()

    return loss

  @property
  def config(self):
    return self._config


class Space(object):
  def __init__(self,
               low=None,
               high=None,
               dtype=uint8,
               size=-1):
    if low is None:
      low = np.zeros(1, dtype=np.uint8)
    if high is None:
      high = np.ones(1, dtype=np.uint8)

    self._shape = np.shape(low) if size == -1 else (size,)
    self._low = tensor(low, dtype=dtype, requires_grad=False)
    self._high = tensor(high, dtype=dtype, requires_grad=False)
    self._dtype = dtype
    self.n = len(self._low)

  @property
  def low(self):
    return self._low

  @property
  def high(self):
    return self._high

  @property
  def shape(self):
    return self._shape


class Fourier_Network(nn.Module):
  def __init__(self, config):
    super(Fourier_Network, self).__init__()

    self._config = config
    observation_space: Space = config.env.observation_space
    self._state_low = observation_space.low.detach()
    self._state_high = observation_space.high.detach()
    self._state_diff = self._state_high - self._state_low
    self._state_dim = len(self._state_low)
    self.flag = (self._state_diff > 1e3).any().item()

    print(f"State low: {self._state_low} ::  State high: {self._state_high}")

    pos_dim = 2  # pos 的维度为 2
    # goal_dim = 2  # goal 的维度为 2
    order = self._config.fourier_coupled
    if order:
      if math.pow(order + 1, pos_dim) >= 1000:
        raise ValueError("Reduced Fourier order please...")
      pos_coeff = np.arange(0, order + 1)
      # goal_coeff = np.arange(0, order + 1)
      pos_weights = torch.from_numpy(np.array(list(itertools.product(pos_coeff, repeat=pos_dim))).T)
      # goal_weights = torch.from_numpy(np.array(list(itertools.product(goal_coeff, repeat=goal_dim))).T)
      self.get_basis = self.coupled
      self.feature_dim = pos_weights.shape[-1]  # + goal_weights.shape[-1]
    else:
      pos_weights = torch.from_numpy(np.arange(1, order + 1))
      # goal_weights = torch.from_numpy(np.arange(1, order + 1))
      self.get_basis = self.uncoupled
      self.feature_dim = (pos_weights.shape[-1])  # + goal_weights.shape[-1]) * 2
    self.pos_basis_weights = pos_weights.type(torch.float32).requires_grad_(False).to(self._config.device)
    # self.goal_basis_weights = goal_weights.type(torch.float32).requires_grad_(False).to(self._config.device)
    self.dummy_param = torch.nn.Parameter(torch.rand(1))

  @staticmethod
  def coupled(state: Tensor, basis_weights: Tensor) -> Tensor:
    basis = torch.matmul(state[:2], basis_weights)
    basis = torch.cos(basis * np.pi)
    return basis.view(state.shape[0], -1)

  @staticmethod
  def uncoupled(state: Tensor, basis_weights: Tensor) -> Tensor:
    state = state[:2].detach()
    state = state.unsqueeze(2)
    basis = state * basis_weights
    basis = torch.cos(basis * np.pi)
    return basis.view(state.shape[0], -1)

  def forward(self, state: Tensor) -> Tensor:
    pos, goal = self.preprocess(state)
    pos_basis = self.get_basis(pos, self.pos_basis_weights)
    # goal_basis = self.get_basis(goal, self.goal_basis_weights)
    return pos_basis

  def custom_weight_init(self):
    for module in self.modules():
      weight_init(module)

  def preprocess(self, state: Tensor):
    pos = state[:, :2]
    goal = state[:, 2:]
    if not self.flag:
      pos = (pos - self._state_low[:2]) / self._state_diff[:2]
      goal = (goal - self._state_low[2:]) / self._state_diff[2:]
    return pos, goal


class Logger(object):
  fwrite_frequency = 1800  # 30 min * 60 sec
  temp = 0

  def __init__(self, log_path, restore, method):
    self.terminal = sys.stdout
    self.file = 'file' in method
    self.term = 'term' in method

    if self.file:
      if restore:
        self.log = open(path.join(log_path, "logfile.log"), "a")
      else:
        self.log = open(path.join(log_path, "logfile.log"), "w")

  def write(self, message):
    if self.term:
      self.terminal.write(message)

    if self.file:
      self.log.write(message)

      if (time() - self.temp) > self.fwrite_frequency:
        self.flush()
        self.temp = time()

  def flush(self):
    self.log.flush()
    fsync(self.log.fileno())


class MemoryBuffer(object):
  def __init__(self,
               max_len: int = 1e6,
               state_dim: int = 32,
               action_dim: int = 1,
               a_emb_dim: int = 2,
               stype=float32,
               atype=long,
               config=None):
    super().__init__()
    assert max_len > 0 and config is not None
    self._config = config

    self._s1 = torch.zeros((max_len, state_dim), dtype=stype, requires_grad=False, device=self._config.device)
    self._a1 = torch.zeros((max_len, action_dim), dtype=atype, requires_grad=False, device=self._config.device)
    self._a1_emb = torch.zeros((max_len, a_emb_dim), dtype=stype, requires_grad=False, device=self._config.device)
    self._r1 = torch.zeros((max_len, 1), dtype=float32, requires_grad=False, device=self._config.device)
    self._s2 = torch.zeros((max_len, state_dim), dtype=stype, requires_grad=False, device=self._config.device)
    self._done = torch.zeros((max_len, 1), dtype=float32, requires_grad=False, device=self._config.device)
    self._mc_return = torch.zeros((max_len, 1), dtype=float32, requires_grad=False, device=self._config.device)
    self._length = 0
    self._max_len = max_len
    self._atype = atype
    self._stype = stype
    self._cnt = 0

  @property
  def size(self) -> int:
    return self._length

  def reset(self) -> None:
    self._length = 0
    self._cnt = 0

  def _get(self, ids) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return self._s1[ids], self._a1[ids], self._a1_emb[ids], self._r1[ids], self._s2[ids], self._done[ids], \
      self._mc_return[ids]

  def batch_sample(self, batch_size: int, randomize=True):
    indices = np.random.permutation(self.size) if randomize else np.arange(self.size)

    for ids in [indices[i:i + batch_size] for i in range(0, self.size, batch_size)]:
      yield self._get(ids)

  def _sample(self, batch_size: int):
    count = np.min([batch_size, self.size])
    return self._get(np.random.choice(self.size, count))

  def _add(self, s1, a1, a_emb, r1, s2, done, mc_return=0):
    pos = self._cnt
    self._s1[pos] = s1.detach() if isinstance(s1, Tensor) else tensor(s1,
                                                                      dtype=self._stype,
                                                                      requires_grad=False,
                                                                      device=self._config.device)
    self._a1[pos] = a1.detach() if isinstance(a1, Tensor) else tensor(a1,
                                                                      dtype=self._atype,
                                                                      requires_grad=False,
                                                                      device=self._config.device)
    self._a1_emb[pos] = a_emb.detach() if isinstance(a_emb, Tensor) else tensor(a_emb,
                                                                                dtype=self._stype,
                                                                                requires_grad=False,
                                                                                device=self._config.device)
    self._r1[pos] = r1.detach() if isinstance(r1, Tensor) else tensor(r1,
                                                                      dtype=float32,
                                                                      requires_grad=False,
                                                                      device=self._config.device)
    self._s2[pos] = s2.detach() if isinstance(s2, Tensor) else tensor(s2,
                                                                      dtype=self._stype,
                                                                      requires_grad=False,
                                                                      device=self._config.device)
    self._done[pos] = done.detach() if isinstance(done, Tensor) else tensor(done,
                                                                            dtype=float32,
                                                                            requires_grad=False,
                                                                            device=self._config.device)
    self._mc_return[pos] = mc_return.detach() if isinstance(mc_return, Tensor) else tensor(mc_return,
                                                                                           dtype=float32,
                                                                                           requires_grad=False,
                                                                                           device=self._config.device)

    self._cnt = (self._cnt + 1) % self._max_len
    self._length = min(self._length + 1, self._max_len)

  def add(self, s1, a1, a_emb, r1, s2, done,
          mc_return=0):
    self._add(s1, a1, a_emb, r1, s2, done, mc_return)

  def sample(self, batch_size: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return self._sample(batch_size)


class Trajectory(object):
  def __init__(self, max_len, state_dim, action_dim, a_emb_dim, atype, stype, config):
    self._s1 = torch.zeros((max_len, state_dim), dtype=stype, requires_grad=False, device=config.device)
    self._a1 = torch.zeros((max_len, action_dim), dtype=atype, requires_grad=False, device=config.device)
    self._a_emb = torch.zeros((max_len, a_emb_dim), dtype=float32, requires_grad=False, device=config.device)
    self._r1 = torch.zeros((max_len, 1), dtype=float32, requires_grad=False, device=config.device)
    self._s2 = torch.zeros((max_len, state_dim), dtype=stype, requires_grad=False, device=config.device)
    self._done = torch.zeros((max_len, 1), dtype=uint8, requires_grad=False, device=config.device)

    self._mc_return = torch.zeros((max_len, 1), dtype=float32, requires_grad=False, device=config.device)
    self._max_len = max_len
    self._cnt = 0
    self._atype = atype
    self._stype = stype
    self._length = 0
    self.config = config

  def add(self, s1, a1, a_emb, r1, s2, done):
    if self._cnt > self._max_len:
      print(f"Traj Length is {self._cnt} and Max Length is {self._max_len}")
    if self._length > self._max_len:
      raise OverflowError
    self._s1[self._cnt] = s1.detach() if isinstance(s1, Tensor) else tensor(s1, dtype=self._stype).detach()
    self._a1[self._cnt] = a1.detach() if isinstance(a1, Tensor) else tensor(a1, dtype=self._atype)
    self._a_emb[self._cnt] = a_emb.detach() if isinstance(a_emb, Tensor) else tensor(a_emb, dtype=float32)
    self._r1[self._cnt] = r1.detach() if isinstance(r1, Tensor) else tensor(r1, dtype=float32)
    self._s2[self._cnt] = s2.detach() if isinstance(s2, Tensor) else tensor(s2, dtype=self._stype)
    self._done[self._cnt] = done.detach() if isinstance(done, Tensor) else tensor(done, dtype=uint8)

    self._length += 1
    self._cnt += 1

  @property
  def size(self):
    return self._length

  def reset(self):
    max_len = self._max_len
    state_dim = self._s1.shape[1]
    action_dim = self._a1.shape[1]
    a_emb_dim = self._a_emb.shape[1]
    stype = self._stype
    atype = self._atype
    config = self.config
    self._s1 = torch.zeros((max_len, state_dim), dtype=stype, requires_grad=False, device=config.device)
    self._a1 = torch.zeros((max_len, action_dim), dtype=atype, requires_grad=False, device=config.device)
    self._a_emb = torch.zeros((max_len, a_emb_dim), dtype=float32, requires_grad=False, device=config.device)
    self._r1 = torch.zeros((max_len, 1), dtype=float32, requires_grad=False, device=config.device)
    self._s2 = torch.zeros((max_len, state_dim), dtype=stype, requires_grad=False, device=config.device)
    self._done = torch.zeros((max_len, 1), dtype=uint8, requires_grad=False, device=config.device)

    self._cnt = 0

  def get_all(self):
    return self._s1.detach(), self._a1.detach(), self._a_emb.detach(), self._r1.detach(), self._s2.detach(), self._done.detach()

  def reset_goal(self, new_goals, new_goal_regions):
    if len(new_goals) > 0:
      new_goal = tensor(new_goals[0], dtype=float32, requires_grad=False)
      new_goal_region = new_goal_regions[0]
      goal_cnt = 0
      self._length = 0
      for idx in range(self._cnt):
        self._s1[idx][-2:] = new_goal
        self._s2[idx][-2:] = new_goal
        pos = self._s2[idx][:2]
        self._cnt += 1
        self._length += 1
        if self.in_region(pos, new_goal_region):
          self._r1[idx] = 100
          self._done[idx] = True
          goal_cnt += 1
          if goal_cnt >= len(new_goals):
            break
          else:
            new_goal = tensor(new_goals[goal_cnt], dtype=float32, requires_grad=False)
            new_goal_region = new_goal_regions[goal_cnt]
        else:
          self._r1[idx] = 0.0
          self._done[idx] = False
      self._cnt = self.size

  @staticmethod
  def in_region(pos, region):
    if region[0] <= pos[0] <= region[2] and region[1] <= pos[1] <= region[3]:
      return True
    return False

  def get_current_traj(self):
    pos = self.size
    self.compute_mc_returns()
    return self._s1[:pos].detach(), self._a1[:pos].detach(), self._a_emb[:pos].detach(), self._r1[
                                                                                         :pos].detach(), self._s2[
                                                                                                         :pos].detach(), self._done[
                                                                                                                         :pos].detach()

  def compute_mc_returns(self, gamma=0.99):
    mc_return = 0
    for idx in reversed(range(self.size)):
      if self._done[idx]:
        mc_return = 0
      mc_return = self._r1[idx].item() + gamma * mc_return
      self._mc_return[idx] = mc_return


class Linear_schedule:
  def __init__(self, max_len, max_data=1, min_data=0):
    self.max_len = max_len
    self.max = max_data
    self.min = min_data
    self.temp = self.max / self.max_len

  def get(self, idx):
    return max(self.min, (self.max_len - idx) * self.temp)
