import json
import logging
import threading
import time
from collections import deque

import numpy as np
from websocket_server import WebsocketServer

from .artifacts import inside_root


TASKS = ("flappybird", "timberman")
STATE_DIM = 12
TREE_COUNT = 11


def timberman_action_hazard(state, action):
    first, second, side = state[0], state[1], state[-1]
    chosen_side = 1 if action == 1 else -1
    if first == side:
        return "already_in_collision"
    if chosen_side != side:
        return "switch_into_branch" if first == chosen_side else None
    return "cut_exposes_branch" if second == side else None


def encode_observation(task, observation):
    if not isinstance(observation, list):
        raise ValueError("WemiEnv obs must be a list")
    state = np.zeros(STATE_DIM, dtype=np.float32)
    if task == "flappybird":
        if len(observation) != 3:
            raise ValueError(f"Expected Flappybird obs length 3, got {len(observation)}")
        if any(type(value) not in (int, float) for value in observation):
            raise ValueError("Flappybird coordinates must be numeric")
        values = np.asarray(observation, dtype=np.float32)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 650).any():
            raise ValueError(f"Invalid Flappybird observation: {observation}")
        state[:3] = values / 600.0
    elif task == "timberman":
        expected = TREE_COUNT * 3 + 1
        if len(observation) != expected:
            raise ValueError(f"Expected Timberman obs length {expected}, got {len(observation)}")
        for index in range(TREE_COUNT):
            direction = observation[index * 3]
            if direction == "center":
                state[index] = 0.0
            elif type(direction) is bool:
                state[index] = 1.0 if direction else -1.0
            else:
                raise ValueError(f"Invalid Timberman branch direction: {direction!r}")
        if type(observation[-1]) is not bool:
            raise ValueError("Timberman npc.posi must be boolean")
        state[-1] = 1.0 if observation[-1] else -1.0
    else:
        raise ValueError(f"Unknown task: {task}")
    return state


class WemiEnv:
    def __init__(self, task, raw_path, host="127.0.0.1", port=7891,
                 timeout=30.0, connect_timeout=300.0, max_episode_steps=3000):
        if task not in TASKS:
            raise ValueError(task)
        self.task = task
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.max_episode_steps = max_episode_steps
        self.condition = threading.Condition()
        self.messages = deque()
        self.client = None
        self.disconnected = False
        self.protocol_error = None
        self.finished = True
        self.steps = 0
        self.last_score = 0.0
        self.raw_file = open(inside_root(raw_path), "a", encoding="utf-8")
        try:
            self.server = WebsocketServer(host=host, port=port, loglevel=logging.WARNING)
        except BaseException:
            self.raw_file.close()
            raise
        self.server.set_fn_new_client(self._connected)
        self.server.set_fn_client_left(self._disconnected)
        self.server.set_fn_message_received(self._received)
        self.server.run_forever(threaded=True)
        print(f"[WemiEnv] {task}: listening ws://{host}:{self.server.server_address[1]}", flush=True)
        print(f"[WemiEnv] Compile vchat_{'Flappybird' if task == 'flappybird' else 'Timberman'} now.", flush=True)

    def _record(self, direction, message):
        self.raw_file.write(json.dumps({"time": time.time(), "task": self.task,
                                      "direction": direction, "message": message},
                                     ensure_ascii=False) + "\n")
        self.raw_file.flush()

    def _connected(self, client, server):
        with self.condition:
            if self.client is not None:
                self.protocol_error = "Multiple clients connected; leave only the target game open"
            else:
                self.client = client
                self.disconnected = False
            self.condition.notify_all()
        print(f"[WemiEnv] client connected for {self.task}", flush=True)

    def _disconnected(self, client, server):
        with self.condition:
            if client is not None and self.client is not None and client["id"] == self.client["id"]:
                self.disconnected = True
                self.condition.notify_all()

    def _received(self, client, server, message):
        with self.condition:
            if client is not None and self.client is not None and client["id"] == self.client["id"]:
                self.messages.append(message)
                self.condition.notify_all()

    def connect(self):
        deadline = time.monotonic() + self.connect_timeout
        with self.condition:
            while self.client is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"No {self.task} client connected; compile the correct WeChat project")
                self.condition.wait(min(remaining, 1.0))
            self._check_connection()

    def _check_connection(self):
        if self.protocol_error:
            raise RuntimeError(self.protocol_error)
        if self.disconnected:
            raise ConnectionError("WeChat client disconnected; recompile and resume from checkpoint")

    def exchange(self, action, restart):
        self.connect()
        request = {"action": action, "restart": restart}
        deadline = time.monotonic() + self.timeout
        with self.condition:
            self._check_connection()
            if self.messages:
                raise RuntimeError("Unexpected queued message: original WemiEnv protocol has no request IDs")
            self._record("send", request)
            self.server.send_message(self.client, json.dumps(request))
            while not self.messages:
                self._check_connection()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for WemiEnv obs/reward/done")
                self.condition.wait(min(remaining, 0.5))
            response = self.messages.popleft()
        self._record("recv", response)
        payload = json.loads(response)
        if not isinstance(payload, dict) or not {"obs", "reward", "done"} <= payload.keys():
            raise ValueError("Response must contain obs, reward and done")
        if type(payload["done"]) is not bool:
            raise ValueError("WemiEnv done must be boolean")
        if type(payload["reward"]) not in (int, float) or not np.isfinite(payload["reward"]):
            raise ValueError("WemiEnv reward must be a finite number")
        return encode_observation(self.task, payload["obs"]), payload

    def reset(self):
        state, payload = self.exchange(3 if self.task == "flappybird" else 5, True)
        if payload["done"]:
            raise RuntimeError("reset returned done=true")
        self.steps = 0
        self.last_score = float(payload["reward"]) if self.task == "timberman" else 0.0
        if self.task == "timberman" and self.last_score != 0:
            raise RuntimeError("Timberman score did not reset to zero")
        self.finished = False
        return state

    def step(self, action):
        if self.finished:
            raise RuntimeError("Call reset before step and after episode end")
        if type(action) is not int or action not in (0, 1):
            raise ValueError("Both games accept only action 0 or 1")
        state, payload = self.exchange(int(action), False)
        self.steps += 1
        terminated = payload["done"]
        truncated = self.steps >= self.max_episode_steps and not terminated
        raw_reward = float(payload["reward"])
        if self.task == "timberman":
            delta = raw_reward - self.last_score
            if delta not in (0.0, 1.0):
                raise ValueError(f"Unexpected Timberman score increment: {delta}")
            reward = delta - float(terminated)
            self.last_score = raw_reward
            info = {"score": raw_reward, "raw_reward": raw_reward, "score_delta": delta}
        else:
            reward = raw_reward / 100.0
            info = {"score": None, "raw_reward": raw_reward}
        self.finished = terminated or truncated
        return state, reward, terminated, truncated, info

    def close(self):
        if not self.raw_file.closed:
            self.server.shutdown()
            self.server.disconnect_clients_gracefully()
            self.server.server_close()
            self.raw_file.close()
