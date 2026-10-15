import numpy as np
from torch import nn
from typing import List

from cal_laica.utils.utils import Space


class AgentBase(nn.Module):
  def __init__(self, config):
    super(AgentBase, self).__init__()
    observation_space: Space = config.env.observation_space
    self._state_low = observation_space.low
    self._state_high = observation_space.high
    self._state_diff = self._state_high - self._state_low
    try:
      action_space: Space = config.env.action_space
      self._action_low = action_space.low
      self._action_high = action_space.high
      self._action_diff = self._action_high - self._action_low
    except AttributeError as _:
      raise AttributeError("Action space not found")

    self._state_dim = observation_space.shape[0]
    self._action_dim = action_space.shape[0] if len(action_space.shape) > 0 else config.env.action_space.n
    self._config = config
    self._entropy, self._tracker = 0, 0

  def init(self):
    for name, module in self.named_modules():
      module.to(self._config.device)

  def _clear_gradient(self):
    for m in self.modules():
      m.zero_grad()

  def save(self):
    if self.config.save_model:
      for name, module in self.named_modules():
        module.save(self.config.paths['ckpt'] + name + '.pt')

  def load(self):
    try:
      for name, module in self.named_modules():
        module.load(self.config.paths['ckpt'] + name + '.pt')
      print("Loaded model from last checkpoint!!!")
    except ValueError as error:
      print(f"Loading failed: {error}")

  def get_grads(self) -> List[List]:
    grads = []
    # if self._config.debug:
    #     for _, module in self.named_modules():
    #         tmp = []
    #         for param in module.parameters():
    #             try:
    #                 tmp.append(np.mean(np.abs(param.grad.data.cpu().numpy())))
    #             except ValueError as error:
    #                 print(error)
    #         grads.append(tmp)
    return grads

  def track_entropy_cont(self, action):
    if self._config.debug:
      self._entropy = 0.5 * self._entropy + 0.5 * np.sum((action - self._tracker) ** 2)
      self._tracker = 0.99 * self._tracker + 0.01 * action

  @property
  def config(self):
    return self._config
