import numpy as np
import os.path
import torch
from time import time
from torch import long, float32, Tensor

from cal_laica.agent.SAC import Agent
from cal_laica.utils import utils
from cal_laica.utils.utils import Linear_schedule


class RandomArchiver(object):
  def __init__(self, config):
    wall_width = config.env.wall_width
    width = int(1.0 / wall_width)
    height = int(1.0 / wall_width)
    self.width = width
    self.height = height
    self.archiver = torch.zeros(((width // 2) * (height // 2), 2), dtype=float32, requires_grad=False,
                                device=config.device)
    self._length = 0
    self.archiver_ids = {}

  def add(self, state: Tensor):
    pos = state[:2]
    row = int(pos[0] * self.width)
    col = int(pos[1] * self.height)
    new_pos = torch.tensor([float(row) / self.width, float(col) / self.height], dtype=float32, requires_grad=False)
    idx = int((pos[0] / 2) * self.height + pos[1] / 2)
    if idx not in self.archiver_ids:
      self._length += 1
      self.archiver[idx] = new_pos
      self.archiver_ids[idx] = True
    else:
      self.archiver[idx] = state[:2]

  def sample(self):
    random_key = np.random.choice(list(self.archiver_ids.keys()))
    state = self.archiver[random_key].clone()
    return state.detach()

  @property
  def size(self):
    return self._length


class Runner(object):
  def __init__(self, config):
    self.config = config
    self.env = config.env

    self.state_dim = self.env.observation_space.shape[0]
    _, action_mask, _ = self.env.reset(is_train=False, is_initial=True)
    self.model = Agent(config, action_mask)
    self.offline_model = Agent(config, action_mask)

  def run(self):
    tmp_rewards = []
    st_episode = 0
    steps = 0
    schedule = Linear_schedule(max_len=self.config.max_episodes)
    t0 = time()
    is_initial = True if self.offline_model.initial_phase else False
    add_trajectory = False
    is_cal_ql = False
    state, _, _ = self.env.reset(is_train=False)
    All_results = []
    rm = 0.0
    for episode in range(st_episode, self.config.max_episodes):

      state, action_mask, change_flag = self.env.reset(is_train=True)
      self.offline_model.reset(action_mask, change_flag, reset_memory=False, is_save_q_values=True)
      self.model.reset(action_mask, change_flag, reset_memory=True)
      trajectory = utils.Trajectory(
        max_len=500,
        state_dim=self.state_dim,
        action_dim=1,
        atype=long,
        stype=float32,
        config=self.config,
        a_emb_dim=self.config.reduced_action_dim
      )
      if self.offline_model.initial_phase:
        self.offline_model.update_offline_memory()
        if not is_initial:
          is_cal_ql = True
          add_trajectory = True
          self.offline_model.reset(action_mask, change_flag, reset_memory=True, is_save=True,
                                   is_save_q_values=True)
        else:
          is_cal_ql = False
        self.online_learning(is_initial)
        state, action_mask, change_flag = self.env.reset(is_train=False)
      else:
        add_trajectory = True
      is_initial = False
      is_terminal = False
      exp = schedule.get(episode)
      step, total_reward = 0, 0.0
      while not is_terminal:
        raw_action, action_embedding = self.offline_model.get_action(state, exp, random_action=True)
        new_state, reward, done, is_terminal, info = self.env.step(raw_action)
        trajectory.add(
          s1=state,
          a1=raw_action,
          a_emb=action_embedding,
          r1=reward,
          s2=new_state,
          done=done
        )
        if self.offline_model.memory_len >= 128 and not self.offline_model.initial_phase:
          self.offline_model.CQL(is_cal_ql=is_cal_ql)
        state = new_state
        step += 1

        if reward > 0:
          total_reward += reward / float(step)
          steps += step
          step = 0
      rm = 0.99 * rm + total_reward * 0.01
      steps += step
      tmp_rewards.append(rm)
      if add_trajectory or self.offline_model.initial_phase:
        self.offline_model.update(trajectory, False)

      All_results.append(total_reward)

  def online_learning(self, is_initial):
    archiver = RandomArchiver(self.config)
    state, _, _ = self.env.reset(is_train=False)
    archiver.add(state)
    episode = 0
    initial_episode = 0 if is_initial else 10000
    while self.offline_model.initial_phase:
      trajectory = utils.Trajectory(
        max_len=500,
        state_dim=self.state_dim,
        action_dim=1,
        atype=long,
        stype=float32,
        config=self.config,
        a_emb_dim=self.config.reduced_action_dim
      )
      if episode >= 15000:
        break
      episode += 1
      # exp = 1.0
      new_st = archiver.sample() if self.model.initial_phase else None
      state, action_mask, change_flag = self.env.reset(new_st, is_train=False)
      self.model.reset(action_mask=action_mask, change_flag=change_flag)
      is_terminal = False
      new_goals, new_goal_regions = [], []
      step, total_r = 0, 0.0
      while not is_terminal:
        raw_action, action_embedding = self.model.get_action(state, 0.0)
        new_state, reward, done, is_terminal, info = self.env.step(raw_action)
        step += 1
        trajectory.add(
          s1=state,
          a1=raw_action,
          a_emb=action_embedding,
          r1=reward,
          done=done,
          s2=new_state
        )
        state = new_state
        total_r += reward

        if not self.model.initial_phase:
          self.model.optimizer()
        if (reward > 0 or np.random.rand() < 0.1) and self.model.initial_phase:
          if not done:
            archiver.add(state)
          wall_width = self.config.env.wall_width
          new_goals.append([int(state[0] / wall_width) // 2 * wall_width + wall_width,
                            int(state[1] / wall_width) // 2 * wall_width + wall_width])
          new_goal_regions.append(
            (new_goals[-1][0] - wall_width, new_goals[-1][1] - wall_width, new_goals[-1][0] + wall_width,
             new_goals[-1][1] + wall_width))
      if self.model.initial_phase:
        trajectory.reset_goal(new_goals, new_goal_regions)
      elif initial_episode <= 1000:
        initial_episode += 1
      else:
        if not is_initial:
          trajectory = None
        self.offline_model.offline_update(trajectory, action_encoder=self.model.action_encoder)
        if not is_initial:
          self.offline_model.update_memory(self.model)
          return

      self.model.update(trajectory)
