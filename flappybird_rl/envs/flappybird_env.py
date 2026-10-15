# -*- coding: utf-8 -*-
"""
Flappy Bird 环境封装（论文 WemiEnv 的 Flappy Bird 任务）
============================================================
本文件基于 WemiEnv/WemiEnv/Flappybird.py 改写，改动如下：

1. **接口标准化**为 gymnasium 风格
   - reset() -> (obs, info)
   - step()  -> (obs, reward, terminated, truncated, info)
2. 观测可选归一化（论文提到"输入做归一化"）
3. 等待客户端消息改为小睡轮询（原版是忙等 pass，会吃满一个 CPU 核）

**动作映射与原版完全一致**：直接把 0/1 发给客户端。

环境规格（来自客户端 lib/wxbird.js）：
    观测 = [鸟的 y, 下管道顶部 y, 上管道高度]，范围 0~600
    动作 = 0 不飞 / 1 拍翅
    奖励 = 若在管道间隙内，取「到最近管壁的正距离」；否则为负距离；撞击/落地 = −10
    结束 = stoped

⚠️ 与 BottleJump 最大的不同：
   本游戏的客户端延迟只有 **100 毫秒**（`setTimeout(()=>{socket_send();}, 100)`），
   而且原版 Flappybird.py 里**没有任何 time.sleep**，
   所以默认 step_delay / reset_delay 都是 0。
============================================================
"""
import json
import logging
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from websocket_server import WebsocketServer

# 观测各分量的量级（客户端里坐标范围约 0~600），用于归一化
OBS_SCALE = 600.0


class Connection:
    """WebSocket 服务端：负责与微信开发者工具里运行的小游戏通信。

    复制自 WemiEnv/WemiEnv/interact.py，仅把「等待客户端」的忙等改为小睡。
    """

    def __init__(self):
        self.latest_msg = None
        self.server = None

    def create_server(self, host='0.0.0.0', port=7891, loglevel=logging.INFO):
        def new_client(client, server):
            print("[WemiEnv] 客户端已连接")

        def left_client(client, server):
            print("[WemiEnv] 客户端已断开")

        def new_received(client, server, msg):
            self.latest_msg = msg

        self.server = WebsocketServer(host, port, loglevel)
        self.server.set_fn_new_client(new_client)
        self.server.set_fn_client_left(left_client)
        self.server.set_fn_message_received(new_received)
        self.server.run_forever(threaded=True)

    def send_message(self, msg_send):
        if self.server is None:
            raise RuntimeError("服务端尚未创建")
        if len(self.server.clients) == 0:
            raise RuntimeError("没有客户端连接，无法发送消息")
        self.server.send_message(self.server.clients[0], msg_send)

    def recv_message(self):
        if self.latest_msg is None:
            return None
        msg = self.latest_msg
        self.latest_msg = None
        return msg

    def waiting_client(self):
        if self.server is None:
            raise RuntimeError("服务端尚未创建")
        print("[WemiEnv] 正在等待客户端连接 ……")
        print("[WemiEnv] 请到微信开发者工具点【编译】")
        while len(self.server.clients) == 0:
            time.sleep(0.01)
        print("[WemiEnv] 客户端连接成功！")


class FlappybirdEnv(gym.Env):
    """Flappy Bird 强化学习环境。

    参数
    ----
    host, port   : WebSocket 地址与端口（默认 0.0.0.0:7891）
    step_delay   : 发送动作【之前】等待的秒数（原版无等待，默认 0）
    reset_delay  : 发送重置指令【之前】等待的秒数（默认 0）
    normalize_obs: 是否把观测除以 600（论文提到输入归一化）
    max_steps_per_episode : 单局步数上限（保险丝）
    wait_timeout : 等待回传的最长秒数
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 host='0.0.0.0',
                 port=7891,
                 step_delay=0.0,
                 reset_delay=0.0,
                 normalize_obs=True,
                 max_steps_per_episode=5000,
                 wait_timeout=30.0):
        super().__init__()

        # 动作：0 不飞 / 1 飞
        self.action_space = spaces.Discrete(2)
        # 观测：[鸟y, 下管道顶, 上管道高]，范围 0~600
        self.observation_space = spaces.Box(low=0.0, high=OBS_SCALE,
                                            shape=(3,), dtype=np.float32)

        self.step_delay = float(step_delay)
        self.reset_delay = float(reset_delay)
        self.normalize_obs = bool(normalize_obs)
        self.max_steps_per_episode = int(max_steps_per_episode)
        self.wait_timeout = float(wait_timeout)

        self._step_count = 0

        # 建立通信通道（会阻塞等待客户端）
        self.conn = Connection()
        self.conn.create_server(host=host, port=port)
        self.conn.waiting_client()

    # ------------------------------------------------------------------
    def _exchange(self, message, delay):
        """等待 → 发送 → 等待并解析回传的观测。

        注意 delay 放在【发送之前】（与原版语义一致）。
        对本游戏 delay 通常为 0，因为客户端自身就有 100ms 的节流。
        """
        if delay > 0:
            time.sleep(delay)

        self.conn.send_message(json.dumps(message))

        deadline = time.time() + self.wait_timeout
        while True:
            data = self.conn.recv_message()
            if data is not None:
                payload = json.loads(data)
                obs = np.array(payload["obs"], dtype=np.float32).ravel()
                if self.normalize_obs:
                    obs = obs / OBS_SCALE
                return obs, payload

            if time.time() > deadline:
                raise TimeoutError(
                    f"等待客户端响应超时（{self.wait_timeout} 秒）。"
                    "请检查微信开发者工具是否在运行、是否点了【编译】。"
                )
            time.sleep(0.0005)

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        """重置游戏。原版发送的 action 是 3（不是 0/1，仅作重置标记）。"""
        super().reset(seed=seed)
        self._step_count = 0
        obs, _ = self._exchange({"action": 3, "restart": True}, self.reset_delay)
        return obs, {}

    def step(self, action):
        """执行一步。action 只能是 0 或 1。"""
        a = int(np.asarray(action).ravel()[0])
        a = 1 if a >= 1 else 0            # 归一成 0/1

        obs, payload = self._exchange({"action": a, "restart": False},
                                      self.step_delay)

        reward = float(payload["reward"])
        terminated = bool(payload["done"])

        self._step_count += 1
        truncated = self._step_count >= self.max_steps_per_episode

        return obs, reward, terminated, truncated, {"action": a}
