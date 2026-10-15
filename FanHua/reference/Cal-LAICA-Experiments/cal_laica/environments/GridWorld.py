import copy
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Tensor
from torch import float32, uint8
from torch import tensor
from typing import Tuple, Callable, List, Dict

from cal_laica.utils.utils import Space, binary_encoding


class GridWorld(object):
  def __init__(self,
               n_actions: int = 8,
               debug: bool = True,
               max_step_length: float = 0.2,
               max_steps=40,
               max_episodes=5e5,
               change_interval: int = -1,
               change_count: int = 3):
    self._debug = debug
    self._episode = 0
    self._n_actions = n_actions if n_actions > 0 else 4
    self._steps = 0
    self._action_space = Space(size=2 ** n_actions)
    self._observation_space = Space(
      low=np.zeros(4, dtype=np.float32),
      high=np.ones(4, dtype=np.float32),
      dtype=torch.float32
    )
    self._display_flag = False
    movement = self.get_movement(n_actions)
    self._motions = self.get_Actions(movement, n_actions)

    self._wall_width = 0.05
    self._step_unit = self._wall_width - 0.005
    self._repeat = int(max_step_length / self._step_unit)

    self._max_steps = int(max_steps / max_step_length)
    self._step_reward = 0.0
    self._collision_reward = 0.0
    self._randomness = 0.00

    self._rng = np.random.RandomState(0)
    self._change_interval = change_interval if change_interval > 0 else max_episodes // change_count
    self.change_add_count = int(1.0 / change_count * 2 ** n_actions)
    self._change_count = change_count
    self._action_mask = torch.zeros(2 ** n_actions, dtype=uint8, requires_grad=False)

    self._obstacle = self._get_obstacle()

    self._true_goals = [
      tensor([0.25, 0.55], dtype=float32, requires_grad=False),
      tensor([0.50, 0.75], dtype=float32, requires_grad=False),
      tensor([0.60, 0.70], dtype=float32, requires_grad=False),
      tensor([0.75, 0.55], dtype=float32, requires_grad=False),
      tensor([0.75, 0.45], dtype=float32, requires_grad=False),
    ]

    self._change_cnt = 0
    self._true_goal_cnt = 0
    self._goal_cnt = 0
    self._goal = self._true_goals[0].detach()
    self._goal_region = self._get_goal_region(self._goal, self.wall_width)
    self._terminal_goal_region = self._get_goal_region(self._goal, self.wall_width)
    self._agent_pos = torch.tensor([0.10, 0.25], dtype=float32, requires_grad=False)
    self._state = self._make_state(self._agent_pos, self._goal)
    self._seed = 0
    self._initial_pos = tensor([0.10, 0.25], dtype=float32, requires_grad=False)
    if self._debug:
      self.heatmap_scale = 99
      self.heatmap = np.zeros((self.heatmap_scale + 1, self.heatmap_scale + 1), dtype=np.int32)

  @staticmethod
  def _get_obstacle() -> List[Tuple]:
    obstacles = []
    return obstacles

  def _reset(self, is_train=True, is_initial=False) -> Tuple[Tensor, Tensor, bool]:
    change_flag = self._update_mask()
    if change_flag:
      self._change_cnt += 1
      self._true_goal_cnt = min(self._true_goal_cnt + 1, len(self._true_goals))
      # if self._change_cnt == self._change_count:
      #     self._true_goal_cnt = len(self._true_goals) - 1
      self._terminal_goal_region = self._get_goal_region(self._true_goals[self._true_goal_cnt - 1],
                                                         self.wall_width)
    self._goal_cnt = 0
    # print(f"Goals cnt is {self._true_goal_cnt}")
    self._goal = self._true_goals[self._goal_cnt]
    self._goal_region = self._get_goal_region(self._goal, self._wall_width)
    self._steps = 0
    self.episode += 1 if is_initial or is_train else 0
    self._agent_pos = copy.deepcopy(self._initial_pos)
    self._state = self._make_state(self._agent_pos, self._goal)
    return self._state.detach(), self._action_mask.detach(), change_flag

  def _step(self, action) -> Tuple[Tensor, float, bool, bool, Dict]:
    assert action >= 0 and self._action_mask[action]

    self._steps += 1
    reward = 0.0

    term = self._is_terminal()
    if term:
      return self._state.detach(), 0.0, True, term, {}
    motion = self._motions[action]
    reward += self._step_reward
    done = False
    for _ in range(self._repeat):
      if np.random.rand() < self._randomness:
        noise = np.random.rand(2) / 1.415
        delta = noise * self._step_unit
        delta = tensor(delta, dtype=float32, requires_grad=False)
      else:
        delta = motion * self._step_unit
      new_pos = self._agent_pos + delta
      if self._valid_pos(new_pos):
        self._agent_pos = new_pos.detach()
        reward += self._compute_reward(new_pos, self._goal_region, self._in_region)
        if reward > 0:
          self._goal_cnt += 1
          if self._goal_cnt < self._true_goal_cnt:
            self._goal = self._true_goals[self._goal_cnt]
            self._goal_region = self._get_goal_region(self._goal, self._wall_width)
          done = True
          # 增加60次移动机会
          self._steps = max(self._steps - 60, 10)
          break
        else:
          done = False
      else:
        reward += self._collision_reward
        done = False
        break
      self._state = self._make_state(self._agent_pos, self._goal)
      x_h, y_h = self._agent_pos * self.heatmap_scale
      self.heatmap[min(int(y_h), 99), min(int(x_h), 99)] += 1
      if self._is_terminal():
        break
    return self._state.detach(), reward, done, self._is_terminal(), {}

  def _render(self):
    raise NotImplemented

  def _valid_pos(self, pos: Tensor) -> bool:
    try:
      if not self._in_region(pos, (0, 0, 1, 1)):
        return False
    except:
      print(f"pos is {pos}!!!")
    for region in self._obstacle:
      if self._in_region(pos, region):
        return False
    return True

  @staticmethod
  def _make_state(pos: Tensor, goal: Tensor):
    pox_x, pos_y = pos[0].item(), pos[1].item()
    goal_x, goal_y = goal[0].item(), goal[1].item()
    return tensor([pox_x, pos_y, goal_x, goal_y], dtype=float32, requires_grad=False)

  @staticmethod
  def _get_goal_region(goal: Tensor, wall_width: float = 0.05) -> Tuple:
    region = (goal[0] - wall_width, goal[1] - wall_width, goal[0] + wall_width, goal[1] + wall_width)
    return region

  @staticmethod
  def _in_region(pos: Tensor, region: Tuple) -> bool:
    return region[0] <= pos[0] <= region[2] and region[1] <= pos[1] <= region[3]

  def seed(self, seed) -> None:
    self._seed = seed
    self._rng = np.random.RandomState(seed)

  @staticmethod
  def _compute_reward(pos: Tensor, goal_region: Tuple, in_regin: Callable[[Tensor, Tuple], bool]) -> int:
    if in_regin(pos, goal_region):
      return 100
    return 0

  def _update_mask(self) -> bool:
    if self.episode % self._change_interval == 0:
      add_actions = []
      curr_no_actions = np.where(self._action_mask.cpu().numpy() == 0)[0]
      if len(curr_no_actions) >= self.change_add_count:
        length = len(curr_no_actions)
        skip = int(length / self.change_add_count)
        add_actions = curr_no_actions[np.arange(0, length, skip)]
      if len(add_actions) <= 1:
        return False
      self._action_mask[add_actions] = 1
      print(f"Actions added {len(add_actions)},Total actions now: {self._action_mask}")
      return True
    return False

  # @property
  def get_embeddings(self) -> Tensor:
    return self._motions.detach()

  @staticmethod
  def get_movement(n_actions: int = 12) -> Tensor:
    x: Tensor = torch.linspace(0, 2 * torch.pi, n_actions + 1, requires_grad=False)
    y: Tensor = torch.linspace(0, 2 * torch.pi, n_actions + 1, requires_grad=False)
    motion_x = torch.round(torch.cos(x)[:-1] * 1000) / 1000
    motion_y = torch.round(torch.sin(y)[:-1] * 1000) / 1000
    movement = torch.stack((motion_x, motion_y), dim=1)
    print(f"Movements:: {movement}")
    return movement.detach()

  @staticmethod
  def get_Actions(movement: Tensor, n_actions: int = 12) -> Tensor:
    motions: Tensor = torch.zeros((2 ** n_actions, 2), dtype=float32, requires_grad=False)
    for idx in range(motions.shape[0]):
      binary_encode = binary_encoding(idx, n_actions)
      motions[idx] = torch.matmul(binary_encode, movement)

    max_dist = torch.max(torch.norm(motions, p=2, dim=-1))
    motions = motions / max_dist
    return motions.detach()

  @property
  def action_space(self) -> Space:
    return self._action_space

  @property
  def observation_space(self) -> Space:
    return self._observation_space

  @property
  def episode(self) -> int:
    return self._episode

  def _is_terminal(self):
    if self._steps >= self._max_steps:
      return True
    elif self._in_region(self._agent_pos, self._terminal_goal_region) \
        and self._in_region(self._agent_pos,
                            self._goal_region):
      return True
    return False

  @episode.setter
  def episode(self, value: int) -> None:
    self._episode = value

  @property
  def wall_width(self) -> float:
    return self._wall_width

  @property
  def true_goal_cnt(self):
    return self._true_goal_cnt

  @property
  def goal_cnt(self):
    return self._goal_cnt

  @property
  def goal_region(self):
    return self._goal_region

  @property
  def terminal_goal_region(self):
    return self._terminal_goal_region


class FourRoom(GridWorld):
  def __init__(self,
               n_actions: int = 8,
               debug: bool = True,
               max_step_length: float = 0.2,
               max_steps=40,
               max_episodes=5e5,
               change_interval: int = -1,
               change_count: int = 3):
    super(FourRoom, self).__init__(n_actions=n_actions,
                                   debug=debug,
                                   max_step_length=max_step_length,
                                   max_steps=max_steps,
                                   max_episodes=max_episodes,
                                   change_interval=change_interval,
                                   change_count=change_count)

  def _get_obstacle(self) -> List[Tuple]:
    wall_width = self._wall_width

    obstacles = [
      (0.50, 0.00, 0.50 + self.wall_width, 0.25, 1.00),
      (0.50, 0.25 + self.wall_width, 0.50 + self.wall_width, 0.75),
      (0.50, 0.75 + wall_width, 0.50 + wall_width, 1.00),
      (0.00, 0.50, 0.25, 0.50 + wall_width),
      (0.25 + wall_width, 0.50, 0.75, 0.50 + wall_width),
      (0.75 + wall_width, 0.50, 1.00, 0.50 + wall_width)
    ]
    return obstacles

  def reset(self, new_state=None, is_train=True, is_initial=False):
    state, action_mask, change_flag = self._reset(is_train=is_train, is_initial=is_initial)
    if new_state is not None:
      self._agent_pos = tensor([new_state[0], new_state[1]], dtype=float32, requires_grad=False)
      self._state = self._make_state(self._agent_pos, self._goal)
      state = self._state

    return state, action_mask, change_flag

  def step(self, action):
    return self._step(action)


class Cross(GridWorld):
  def __init__(self,
               n_actions: int = 8,
               debug: bool = True,
               max_step_length: float = 0.2,
               max_steps=40,
               max_episodes=5e5,
               change_interval: int = -1,
               change_count: int = 3):
    super(Cross, self).__init__(n_actions=n_actions,
                                debug=debug,
                                max_step_length=max_step_length,
                                max_steps=max_steps,
                                max_episodes=max_episodes,
                                change_interval=change_interval,
                                change_count=change_count)
    self._agent_pos = tensor([0.75, 0.25], dtype=float32, requires_grad=False)
    self._initial_pos = tensor([0.75, 0.25], dtype=float32, requires_grad=False)
    self._true_goals = [
      tensor([0.50, 0.60], dtype=float32, requires_grad=False),
      tensor([0.45, 0.85], dtype=float32, requires_grad=False),
      tensor([0.15, 0.85], dtype=float32, requires_grad=False),
      tensor([0.00, 0.75], dtype=float32, requires_grad=False),
      tensor([0.15, 0.45], dtype=float32, requires_grad=False),
    ]

  def _get_obstacle(self) -> List[Tuple]:
    obstacle = [
      (0.00, 0.35, 0.40, 0.35 + self.wall_width),
      (0.40, 0.35, 0.40 + self.wall_width, 0.80),
      (0.05, 0.75, 0.40, 0.75 + self.wall_width)
    ]
    return obstacle

  def reset(self, new_state=None, is_train=True, is_initial=False):
    state, action_mask, change_flag = self._reset(is_train=is_train, is_initial=is_initial)
    if new_state is not None:
      self._agent_pos = tensor([new_state[0], new_state[1]], dtype=float32, requires_grad=False)
      self._state = self._make_state(self._agent_pos, self._goal)
      state = self._state

    return state, action_mask, change_flag

  def step(self, action):
    return self._step(action)


class DistShift(GridWorld):
  def __init__(self,
               n_actions: int = 8,
               debug: bool = True,
               max_step_length: float = 0.2,
               max_steps=40,
               max_episodes=5e5,
               change_interval: int = -1,
               change_count: int = 3):
    super(DistShift, self).__init__(
      n_actions=n_actions,
      debug=debug,
      max_step_length=max_step_length,
      max_steps=max_steps,
      max_episodes=max_episodes,
      change_interval=change_interval,
      change_count=change_count
    )
    self._true_goals = [
      tensor([0.40, 0.60], dtype=float32, requires_grad=False),
      tensor([0.50, 0.55], dtype=float32, requires_grad=False),
      tensor([0.70, 0.50], dtype=float32, requires_grad=False),
      tensor([0.80, 0.40], dtype=float32, requires_grad=False),
      tensor([0.85, 0.25], dtype=float32, requires_grad=False)
    ]
    # self._true_goals=[
    #     tensor([0.])
    # ]

  def reset(self, new_state=None, is_train=True, is_initial=False):
    state, action_mask, change_flag = self._reset(is_train=is_train, is_initial=is_initial)
    if new_state is not None:
      self._agent_pos = tensor([new_state[0], new_state[1]], dtype=float32, requires_grad=False)
      self._state = self._make_state(self._agent_pos, self._goal)
      state = self._state
    return state, action_mask, change_flag

  def step(self, action):
    return self._step(action)

  def _get_obstacle(self) -> List[Tuple]:
    obstacles = [
      # (0.0, 0.30, 0.50, 0.80)
      (0.25, 0.0, 0.75, 0.45)
    ]
    return obstacles


class MovieRecommendationEnv(object):
    def __init__(self,
                 n_actions: int = 10,
                 debug: bool = True,
                 max_steps: int = 50,
                 max_episodes: int = 1000,
                 change_interval: int = 10,  # 每隔多少回合增加可推荐电影数量
                 change_count: int = 5):  # 可推荐电影数量增加的次数
        self._debug = debug
        self._episode = 0
        self._steps = 0
        self._n_actions = n_actions
        self._max_steps = max_steps
        self._max_episodes = max_episodes
        self._change_interval = change_interval
        self._change_count = change_count
        self._action_space = torch.zeros(n_actions, dtype=uint8, requires_grad=False)
        self._observation_space = torch.zeros((n_actions,), dtype=float32, requires_grad=False)
        self._rng = np.random.RandomState(0)

        # 模拟电影数据集（电影特征）
        self._movie_embeddings = self._generate_movie_embeddings(n_actions)
        self._user_preferences = self._generate_user_preferences()
        self._watched_movies = set()

        # 初始化环境状态
        self._state = self._make_state()
        self._actions_unlocked = n_actions

    def reset(self) -> Tuple[Tensor, Tensor]:
        """重置环境"""
        self._steps = 0
        self._episode += 1
        self._watched_movies = set()
        self._state = self._make_state()
        self._update_action_space()
        return self._state.detach(), self._action_space.detach()

    def step(self, action: int) -> Tuple[Tensor, float, bool, Dict]:
        """执行一个推荐动作"""
        assert 0 <= action < len(self._action_space), "动作超出范围"
        assert self._action_space[action] == 1, "动作未解锁"

        self._steps += 1
        movie_embedding = self._movie_embeddings[action]
        reward = self._compute_reward(movie_embedding)
        done = self._steps >= self._max_steps

        # 更新用户状态
        self._watched_movies.add(action)
        self._state = self._make_state()

        # 如果达到最大回合数或动作空间变化，更新动作空间
        if self._episode % self._change_interval == 0:
            self._update_action_space()

        return self._state.detach(), reward, done, {}

    def _make_state(self) -> Tensor:
        """生成当前状态：包含用户偏好和已观看电影"""
        watched_vector = torch.zeros(len(self._movie_embeddings), dtype=float32)
        for movie_id in self._watched_movies:
            watched_vector[movie_id] = 1.0
        return torch.cat([self._user_preferences, watched_vector])

    def _compute_reward(self, movie_embedding: Tensor) -> float:
        """计算推荐奖励：根据用户偏好与电影特征的匹配程度"""
        similarity = torch.dot(self._user_preferences, movie_embedding)
        reward = similarity.item()
        return reward

    def _update_action_space(self) -> None:
        """更新动作空间：解锁更多电影"""
        unlocked_count = min(self._actions_unlocked + self._change_count, len(self._movie_embeddings))
        self._action_space[:unlocked_count] = 1
        self._actions_unlocked = unlocked_count
        if self._debug:
            print(f"解锁电影数量：{self._actions_unlocked}")

    @staticmethod
    def _generate_movie_embeddings(n_movies: int) -> Tensor:
        """生成电影特征向量"""
        embeddings = torch.randn((n_movies, 10), dtype=float32)  # 每个电影有10维特征
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # 标准化
        return embeddings

    @staticmethod
    def _generate_user_preferences() -> Tensor:
        """生成用户偏好向量"""
        preferences = torch.randn(10, dtype=float32)  # 用户偏好也为10维
        preferences = torch.nn.functional.normalize(preferences, p=2, dim=0)  # 标准化
        return preferences

    @property
    def action_space(self) -> Tensor:
        return self._action_space

    @property
    def observation_space(self) -> Tensor:
        return self._observation_space

    @property
    def episode(self) -> int:
        return self._episode



if __name__ == "__main__":
  env = Cross()

  env.reset(is_initial=True)
  env.render()
