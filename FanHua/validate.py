import base64
import copy
import hashlib
import json
import math
import socket
import struct
import threading
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fanhua.agent import Agent, conservative_gap
from fanhua.artifacts import ROOT, inside_root, new_run, write_json
from fanhua.diagnostics import check_focused_timber_policy
from fanhua.prerequisites import check_prerequisites
from fanhua.envs import WemiEnv, encode_observation
from fanhua.evaluation import evaluate_environment
from fanhua.supervision import behavior_loss
from fanhua.replay import Replay
import pretrain
from train import check_config, train


DIRECTORY = None


def config():
    values = json.loads((ROOT / "configs" / "smoke.json").read_text(encoding="utf-8"))
    values.update(hidden_dim=16, batch_size=4, capacity=128, warmup_steps=4,
                  representation_steps=3, cql_samples=2, checkpoint_interval=4)
    values["stages"] = [{"task": "flappybird", "steps": 12},
                        {"task": "timberman", "steps": 12},
                        {"task": "flappybird", "steps": 12}]
    return values


def episode(task, count=8):
    return [{"task": task, "stage": 0, "state": [index / 10.0] * 12,
             "next_state": [(index + 1) / 10.0] * 12, "action": index % 2,
             "embedding": [-0.5 if index % 2 == 0 else 0.5] * 2,
             "reward": float(index % 2), "terminated": index == count - 1,
             "truncated": False} for index in range(count)]


class SyntheticEnv:
    calls = 0
    fail_at = None

    def __init__(self, task, *args, **kwargs):
        self.task = task
        self.steps = 0

    def reset(self):
        self.steps = 0
        return np.zeros(12, dtype=np.float32)

    def step(self, action):
        type(self).calls += 1
        if type(self).calls == type(self).fail_at:
            raise ConnectionError("Synthetic disconnect")
        self.steps += 1
        state = np.full(12, self.steps / 10, dtype=np.float32)
        return state, 1.0, self.steps == 2, False, {"raw_reward": 1.0,
                                                 "score": self.steps if self.task == "timberman" else None}

    def close(self):
        pass


def read_bytes(connection, count):
    data = b""
    while len(data) < count:
        received = connection.recv(count - len(data))
        if not received:
            raise EOFError()
        data += received
    return data


def local_client(port, errors):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=5) as connection:
            key = base64.b64encode(b"0123456789abcdef").decode()
            request = (f"GET / HTTP/1.1\r\nHost: localhost:{port}\r\nUpgrade: websocket\r\n"
                       f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n")
            connection.sendall(request.encode())
            headers = b""
            while not headers.endswith(b"\r\n\r\n"):
                headers += read_bytes(connection, 1)
            if b"101 Switching Protocols" not in headers:
                raise RuntimeError(headers.decode())
            while True:
                first, second = read_bytes(connection, 2)
                if first & 15 == 8:
                    return
                length = second & 127
                if length == 126:
                    length = struct.unpack("!H", read_bytes(connection, 2))[0]
                elif length == 127:
                    length = struct.unpack("!Q", read_bytes(connection, 8))[0]
                message = json.loads(read_bytes(connection, length))
                observation = [value for index in range(11) for value in ("center", None, None)] + [True]
                response = json.dumps({"obs": observation, "reward": 0 if message["restart"] else 1,
                                       "done": not message["restart"]}).encode()
                mask = b"abcd"
                encoded = bytes(value ^ mask[index % 4] for index, value in enumerate(response))
                frame = bytes([129, 128 | 126]) + struct.pack("!H", len(response)) + mask + encoded
                connection.sendall(frame)
    except EOFError:
        return
    except BaseException as error:
        errors.append(f"{type(error).__name__}: {error}")


class Verification(unittest.TestCase):
    def test_observations_and_game_mismatch(self):
        np.testing.assert_allclose(encode_observation("flappybird", [300, 400, 200])[:3], [0.5, 2/3, 1/3])
        raw = [value for index in range(11) for value in ("center", None, None)] + [False]
        raw[0] = True
        raw[3] = False
        encoded = encode_observation("timberman", raw)
        self.assertEqual(encoded.tolist(), [1, -1] + [0] * 9 + [-1])
        with self.assertRaises(ValueError):
            encode_observation("flappybird", raw)
        with self.assertRaises(ValueError):
            encode_observation("timberman", [300, 400, 200])
        raw[0] = 1
        with self.assertRaises(ValueError):
            encode_observation("timberman", raw)
        with self.assertRaises(ValueError):
            encode_observation("flappybird", [float("nan"), 400, 200])

    def test_reward_difference_and_truncation(self):
        environment = WemiEnv.__new__(WemiEnv)
        environment.task = "timberman"
        environment.finished = True
        environment.max_episode_steps = 2
        responses = iter([0, 1, 1])
        environment.exchange = lambda action, restart: (np.zeros(12), {"reward": next(responses), "done": False})
        environment.reset()
        self.assertEqual(environment.step(0)[1:4], (1.0, False, False))
        self.assertEqual(environment.step(1)[1:4], (0.0, False, True))
        with self.assertRaises(RuntimeError):
            environment.step(0)
        environment.finished = False
        environment.exchange = lambda action, restart: (np.zeros(12), {"reward": 1, "done": True})
        self.assertEqual(environment.step(0)[1:4], (-1.0, True, False))

    def test_mc_returns_and_history(self):
        replay = Replay(64, 0.25, 5)
        rows = episode("flappybird", 2)
        rows[0]["reward"], rows[1]["reward"] = 1.0, 2.0
        replay.add_episode(rows, 0.5)
        self.assertEqual([row["mc_return"] for row in replay.online.rows], [2.0, 2.0])
        self.assertTrue(all(row["valid_bound"] for row in replay.online.rows))
        replay.begin_stage()
        incomplete = episode("timberman", 2)
        incomplete[-1]["terminated"] = False
        replay.add_episode(incomplete, 0.5, bootstrap=4, complete=False)
        self.assertEqual(replay.online.rows[-1]["mc_return"], 3.0)
        self.assertFalse(any(row["valid_bound"] for row in replay.online.rows))
        sample = replay.sample(4000)
        self.assertAlmostEqual(sample["history_fraction"], 0.25, delta=0.035)
        replay.begin_stage()
        self.assertEqual(len(replay.history["flappybird"].rows), 2)
        self.assertEqual(len(replay.history["timberman"].rows), 2)
        disabled = Replay(64, 0)
        disabled.add_episode(rows, 0.5)
        disabled.begin_stage()
        self.assertFalse(any(pool.rows for pool in disabled.history.values()))

    def test_calibration_gradients(self):
        candidates = torch.tensor([[0.0, 1.0]], requires_grad=True)
        data_values = torch.tensor([0.0], requires_grad=True)
        result = conservative_gap(candidates, data_values, torch.tensor([2.0]), torch.tensor([1.0]), True)
        self.assertAlmostEqual(result.item(), 2.0, places=6)
        result.backward()
        self.assertEqual(candidates.grad.tolist(), [[0.0, 0.0]])
        self.assertEqual(data_values.grad.tolist(), [-1.0])
        invalid = conservative_gap(candidates.detach(), data_values.detach(), torch.tensor([2.0]), torch.tensor([0.0]), True)
        expected = math.log((1 + math.e) / 2)
        self.assertAlmostEqual(invalid.item(), expected, places=6)

    def test_updates_frozen_prototypes_and_exact_resume(self):
        values = config()
        values["behavior_weight"] = 5.0
        values["policy_focus_timber"] = True
        agent = Agent(values)
        replay = Replay(128, 0.25, 0)
        generator = np.random.default_rng(0)
        for parameter, target_parameter in zip(agent.critic.parameters(), agent.target.parameters()):
            self.assertTrue(torch.equal(parameter, target_parameter))
        replay.add_episode(episode("flappybird"), values["gamma"])
        agent.fit_representation(replay, "flappybird", 3)
        original = agent.representation.prototypes[0].detach().clone()
        replay.begin_stage()
        replay.add_episode(episode("timberman"), values["gamma"])
        agent.fit_representation(replay, "timberman", 3)
        self.assertTrue(torch.equal(original, agent.representation.prototypes[0]))
        before_policy = copy.deepcopy(agent.policy.state_dict())
        before_critic = copy.deepcopy(agent.critic.state_dict())
        stats = agent.update(replay.sample(4))
        self.assertTrue(all(math.isfinite(value) for value in stats.values()))
        self.assertTrue(any(not torch.equal(before_policy[name], value) for name, value in agent.policy.state_dict().items()))
        self.assertTrue(any(not torch.equal(before_critic[name], value) for name, value in agent.critic.state_dict().items()))
        checkpoint = DIRECTORY / "resume_test.pt"
        digest = agent.parameter_digest()
        agent.save(checkpoint, replay, {"stage": 2}, generator)
        expected_action = int(generator.integers(1000))
        expected_stats = agent.update(replay.sample(4))
        expected_digest = agent.parameter_digest()
        restored = Agent(values)
        restored_replay = Replay(128)
        restored_generator = np.random.default_rng(20)
        self.assertEqual(restored.load(checkpoint, restored_replay, restored_generator), {"stage": 2})
        self.assertEqual(digest, restored.parameter_digest())
        self.assertEqual(expected_action, int(restored_generator.integers(1000)))
        actual_stats = restored.update(restored_replay.sample(4))
        self.assertEqual(expected_stats, actual_stats)
        self.assertEqual(expected_digest, restored.parameter_digest())

    def test_stage_evaluation_overrides_and_restores_training_cap(self):
        class CappedEnv(SyntheticEnv):
            max_episode_steps = 2

            def step(self, action):
                self.steps += 1
                return (np.zeros(12, dtype=np.float32), 1.0, False,
                        self.steps >= self.max_episode_steps, {"raw_reward": 1.0, "score": None})

        environment = CappedEnv("flappybird")
        result = evaluate_environment(Agent(config()), environment, "flappybird", 2, 5)
        self.assertEqual([row["steps"] for row in result["episodes"]], [5, 5])
        self.assertEqual(result["evaluation_max_steps"], 5)
        self.assertEqual(environment.max_episode_steps, 2)
        with patch.object(environment, "step", side_effect=ConnectionError("Evaluation disconnect")):
            with self.assertRaises(ConnectionError):
                evaluate_environment(Agent(config()), environment, "flappybird", 2, 5)
        self.assertEqual(environment.max_episode_steps, 2)

    def test_shutdown_and_port_reuse(self):
        errors = []
        environment = WemiEnv("timberman", DIRECTORY / "socket.jsonl", port=0, timeout=5, connect_timeout=5)
        port = environment.server.server_address[1]
        client = threading.Thread(target=local_client, args=(port, errors), daemon=True)
        client.start()
        try:
            self.assertEqual(environment.reset().shape, (12,))
            self.assertEqual(environment.step(0)[1:4], (0.0, True, False))
        finally:
            environment.close()
            client.join(timeout=5)
        self.assertFalse(client.is_alive())
        self.assertEqual(errors, [])
        reused = WemiEnv("timberman", DIRECTORY / "reused.jsonl", port=port)
        reused.close()
        reused.close()

    def test_runner_stage_boundaries_and_disconnect_resume(self):
        values = config()
        values["stage_evaluation_episodes"] = 2
        values["stage_evaluation_steps"] = 5
        directory = DIRECTORY / "runner"
        directory.mkdir()
        SyntheticEnv.calls, SyntheticEnv.fail_at = 0, 6
        with patch("train.WemiEnv", SyntheticEnv):
            with self.assertRaises(ConnectionError):
                train(values, directory)
            status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["global_steps"], 5)
            failed = torch.load(directory / "checkpoints/latest.pt", weights_only=True)
            self.assertFalse(failed["replay"]["online"]["rows"][-1]["valid_bound"])
            SyntheticEnv.fail_at = None
            train(values, directory, directory / "checkpoints/latest.pt")
        result = torch.load(directory / "checkpoints/latest.pt", weights_only=True)
        self.assertEqual(result["progress"]["global_steps"], 36)
        self.assertEqual(result["progress"]["stage"], 3)
        self.assertEqual(set(result["ready_tasks"]), {"flappybird", "timberman"})
        self.assertTrue(result["replay"]["history"]["flappybird"]["rows"])
        self.assertTrue(result["replay"]["history"]["timberman"]["rows"])
        evaluations = list(directory.glob("stage_*_evaluation.json"))
        self.assertEqual(len(evaluations), 3)
        for path in evaluations:
            evaluation = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(evaluation["parameters_before"], evaluation["parameters_after"])
            self.assertEqual(len(evaluation["checkpoint_sha256"]), 64)

    def test_requested_stop_preserves_checkpoint(self):
        directory = DIRECTORY / "requested_stop"
        directory.mkdir()
        (directory / "STOP").touch()
        with patch("train.WemiEnv", SyntheticEnv):
            train(config(), directory)
        checkpoint = torch.load(directory / "checkpoints/latest.pt", weights_only=True)
        self.assertEqual(checkpoint["progress"]["status"], "interrupted")
        self.assertEqual(checkpoint["progress"]["global_steps"], 0)
        self.assertFalse((directory / "STOP").exists())

    def test_outputs_and_config(self):
        with self.assertRaises(ValueError):
            inside_root(ROOT.parent / "outside.pt")
        check_config(config())
        invalid = config()
        invalid["learning_rate"] = float("nan")
        with self.assertRaises(ValueError):
            check_config(invalid)

    def test_teacher_supervision_preserves_actual_critic_data(self):
        values = config()
        values["behavior_weight"] = 5.0
        agent = Agent(values)
        replay = Replay(128)
        replay.add_episode(episode("flappybird"), values["gamma"])
        batch = replay.sample(8)
        actual_actions = batch["action"].clone()
        actual_embeddings = batch["embedding"].clone()
        actual_rewards = batch["reward"].clone()
        batch["teacher_action"] = 1 - batch["action"]
        stats = agent.update(batch)
        self.assertTrue(math.isfinite(stats["behavior_loss"]))
        self.assertTrue(torch.equal(batch["action"], actual_actions))
        self.assertTrue(torch.equal(batch["embedding"], actual_embeddings))
        self.assertTrue(torch.equal(batch["reward"], actual_rewards))

    def test_policy_focus_preserves_task_specific_inputs(self):
        values = config()
        values["policy_focus_timber"] = True
        agent = Agent(values)
        states = torch.zeros(2, 12)
        perturbed = states.clone()
        perturbed[:, 2:11] = 1
        timber = torch.ones(2, dtype=torch.long)
        first, _ = agent.policy(states, timber, deterministic=True)
        second, _ = agent.policy(perturbed, timber, deterministic=True)
        self.assertTrue(torch.equal(first, second))
        flappy = torch.zeros(2, dtype=torch.long)
        first, _ = agent.policy(states, flappy, deterministic=True)
        second, _ = agent.policy(perturbed, flappy, deterministic=True)
        self.assertFalse(torch.equal(first, second))

    def test_independent_baseline_rejects_teacher_and_other_game_checkpoint(self):
        values = config()
        values.update(experiment_role="single_game_baseline", baseline_task="flappybird",
                      stages=[{"task": "flappybird", "steps": 12}], behavior_weight=0.0,
                      policy_focus_timber=False, teacher_paths={})
        check_prerequisites(values)
        invalid = copy.deepcopy(values)
        invalid["teacher_paths"] = {"flappybird": "unused.pt"}
        with self.assertRaisesRegex(ValueError, "teacher"):
            check_prerequisites(invalid)
        invalid = copy.deepcopy(values)
        invalid["stages"].append({"task": "timberman", "steps": 12})
        with self.assertRaisesRegex(ValueError, "designated task"):
            check_prerequisites(invalid)
        agent = Agent(values)
        replay = Replay(128)
        replay.add_episode(episode("timberman"), values["gamma"])
        checkpoint = DIRECTORY / "contaminated_baseline.pt"
        agent.save(checkpoint, replay, {"status": "running"}, np.random.default_rng(0))
        with self.assertRaisesRegex(ValueError, "another game's"):
            check_prerequisites(values, checkpoint)

    def test_formal_gate_requires_real_budget_and_independent_evaluation(self):
        gate_directory = DIRECTORY / "formal_gate"
        gate_directory.mkdir()
        entries = []
        for task in ("flappybird", "timberman"):
            directory = gate_directory / task
            directory.mkdir()
            values = config()
            values.update(experiment_role="single_game_baseline", baseline_task=task,
                          stages=[{"task": task, "steps": 12}], behavior_weight=0.0,
                          policy_focus_timber=False, teacher_paths={})
            config_path = directory / "config.json"
            write_json(config_path, values)
            status = {"status": "completed_training_not_yet_evaluated", "global_steps": 12, "stage": 1}
            write_json(directory / "status.json", status)
            agent = Agent(values)
            agent.ready_tasks = [task]
            replay = Replay(128)
            replay.add_episode(episode(task), values["gamma"])
            checkpoint = directory / "checkpoints/latest.pt"
            agent.save(checkpoint, replay, status, np.random.default_rng(0))
            SyntheticEnv.fail_at = None
            evaluation = evaluate_environment(agent, SyntheticEnv(task), task, 3, 10)
            evaluation.update(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                              evaluation_max_steps=10)
            evaluation_path = directory / "evaluation.json"
            write_json(evaluation_path, evaluation)
            entries.append({"role": "single_game_baseline", "task": task, "seed": 0,
                            "status": "completed", "config": str(config_path.relative_to(ROOT)),
                            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                            "run": str(directory.relative_to(ROOT)), "evaluation": str(evaluation_path.relative_to(ROOT))})
        manifest = {"formal_training_ready": False, "formal_experiments_authorized": False,
                    "primary_seed": 0, "baseline_steps_per_task": 12,
                    "baseline_final_evaluation_episodes": 3, "baseline_final_evaluation_steps": 10,
                    "experiments": entries, "synthetic": True}
        manifest_path = gate_directory / "manifest.json"
        write_json(manifest_path, manifest)
        formal = config()
        formal.update(experiment_role="formal_continual_experiment", prerequisites_manifest=str(manifest_path))
        with patch("train.WemiEnv") as environment:
            with self.assertRaisesRegex(RuntimeError, "remain held"):
                train(formal, gate_directory)
            environment.assert_not_called()
        manifest.update(formal_training_ready=True, formal_experiments_authorized=True)
        write_json(manifest_path, manifest)
        check_prerequisites(formal)
        status_path = gate_directory / "flappybird/status.json"
        valid_status = json.loads(status_path.read_text(encoding="utf-8"))
        invalid_status = dict(valid_status, global_steps=11)
        write_json(status_path, invalid_status)
        with self.assertRaisesRegex(RuntimeError, "budget"):
            check_prerequisites(formal)
        write_json(status_path, valid_status)
        evaluation_path = gate_directory / "flappybird/evaluation.json"
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        evaluation["evaluation_max_steps"] = 3
        write_json(evaluation_path, evaluation)
        with self.assertRaisesRegex(RuntimeError, "independent baseline evaluation"):
            check_prerequisites(formal)

    def test_timber_regression_diagnostic_detects_recorded_hazards(self):
        replay = Replay(128)
        valid_state = [0.0, 1.0] + [0.0] * 9 + [1.0]
        invalid_state = valid_state.copy()
        invalid_state[0] = 1.0
        rows = episode("timberman", 3)
        rows[0]["state"] = valid_state
        rows[1]["state"] = valid_state.copy()
        rows[2]["state"] = invalid_state
        replay.add_episode(rows, 0.99)
        agent = Agent(config())
        with patch.object(agent, "select", return_value=(0, [-0.5, -0.5])):
            safe = check_focused_timber_policy(agent, replay)
        self.assertTrue(safe["passed"])
        self.assertEqual(safe["checked_observed_patterns"], 1)
        with patch.object(agent, "select", return_value=(1, [0.5, 0.5])):
            unsafe = check_focused_timber_policy(agent, replay)
        self.assertFalse(unsafe["passed"])
        self.assertEqual(unsafe["failures"][0]["hazard"], "cut_exposes_branch")
        self.assertFalse(check_focused_timber_policy(agent, Replay(128))["passed"])

    def test_timber_regression_guards_save_failure_and_block_client(self):
        values = config()
        values["policy_focus_timber"] = True
        agent = Agent(values)
        agent.ready_tasks = {"flappybird", "timberman"}
        replay = Replay(128)
        replay.add_episode(episode("timberman"), values["gamma"])
        replay.begin_stage()
        source = DIRECTORY / "guard_source.pt"
        progress = {"stage": 0, "stage_steps": 0, "global_steps": 0, "episodes": 0,
                    "last_checkpoint_step": 0, "status": "running"}
        agent.save(source, replay, progress, np.random.default_rng(values["seed"]))
        source_bytes = source.read_bytes()
        failed = {"passed": False, "checked_observed_patterns": 1,
                  "failures": [{"hazard": "cut_exposes_branch"}]}
        initial_directory = DIRECTORY / "guard_initial"
        initial_directory.mkdir()
        with patch("train.check_focused_timber_policy", return_value=failed), patch("train.WemiEnv") as environment:
            with self.assertRaisesRegex(RuntimeError, "Initial policy"):
                train(values, initial_directory, source)
            environment.assert_not_called()
        initial = torch.load(initial_directory / "checkpoints/latest.pt", weights_only=True)
        self.assertEqual(initial["progress"]["status"], "failed")
        self.assertEqual(initial["progress"]["global_steps"], 0)
        checkpoint_directory = DIRECTORY / "guard_checkpoint"
        checkpoint_directory.mkdir()
        SyntheticEnv.calls, SyntheticEnv.fail_at = 0, None
        passed = {"passed": True, "checked_observed_patterns": 1, "failures": []}
        with patch("train.check_focused_timber_policy", side_effect=[passed, failed]), patch("train.WemiEnv", SyntheticEnv):
            with self.assertRaisesRegex(RuntimeError, "regression detected"):
                train(values, checkpoint_directory, source)
        checkpoint = torch.load(checkpoint_directory / "checkpoints/latest.pt", weights_only=True)
        self.assertEqual(checkpoint["progress"]["status"], "failed")
        self.assertEqual(checkpoint["progress"]["global_steps"], 4)
        self.assertEqual(checkpoint["updates"], 1)
        self.assertEqual(source.read_bytes(), source_bytes)

    def test_irreversible_collision_not_used_for_actor_supervision(self):
        agent = Agent(config())
        states = torch.zeros(2, 12)
        states[:, -1] = 1
        states[1, 0] = 1
        task_ids = torch.ones(2, dtype=torch.long)
        actions = torch.tensor([0, 1])
        full = behavior_loss(agent.policy, agent.representation, states, task_ids, actions)
        valid_only = behavior_loss(agent.policy, agent.representation, states[:1], task_ids[:1], actions[:1])
        self.assertTrue(torch.allclose(full, valid_only, atol=1e-6))
        invalid_only = behavior_loss(agent.policy, agent.representation, states[1:], task_ids[1:], actions[1:])
        self.assertEqual(float(invalid_only.detach()), 0)

    def test_inference_does_not_update_parameters(self):
        agent = Agent(config())
        environment = SyntheticEnv("flappybird")
        SyntheticEnv.fail_at = None
        result = evaluate_environment(agent, environment, "flappybird", 3, 10)
        self.assertEqual(result["parameters_before"], result["parameters_after"])
        self.assertEqual(result["updates_before"], 0)
        self.assertEqual(result["updates_after"], 0)
        self.assertEqual(len(result["episodes"]), 3)
        self.assertEqual(sum(result["action_counts"].values()), 6)

    def test_offline_pretrain_without_online_history(self):
        values = config()
        values["history_probability"] = 0.0
        replay = Replay(128)
        replay.add_episode(episode("flappybird"), values["gamma"])
        replay.add_episode(episode("timberman"), values["gamma"])
        dataset_path = DIRECTORY / "synthetic_dataset.pt"
        torch.save({"format_version": 1, "gamma": values["gamma"], "rows": replay.online.rows,
                    "encoding_version": "wemi_12_v1", "reward_version": "flappy_scaled_timber_delta_v1",
                    "synthetic": True}, dataset_path)
        config_path = DIRECTORY / "offline_config.json"
        write_json(config_path, values)
        destination = DIRECTORY / "offline_ablation"
        destination.mkdir()
        arguments = ["pretrain.py", "--config", str(config_path), "--dataset", str(dataset_path), "--updates", "4"]
        with patch("sys.argv", arguments), patch("pretrain.new_run", return_value=destination):
            pretrain.main()
        checkpoint = torch.load(destination / "checkpoints/latest.pt", weights_only=True)
        self.assertEqual(checkpoint["updates"], 4)
        self.assertEqual(checkpoint["replay"]["history_probability"], 0.0)
        self.assertEqual(set(checkpoint["ready_tasks"]), {"flappybird", "timberman"})


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.records.append({"test": test.id(), "status": "passed"})

    def addFailure(self, test, error):
        super().addFailure(test, error)
        self.records.append({"test": test.id(), "status": "failed", "error": self._exc_info_to_string(error, test)})

    def addError(self, test, error):
        super().addError(test, error)
        self.records.append({"test": test.id(), "status": "error", "error": self._exc_info_to_string(error, test)})


def main():
    global DIRECTORY
    DIRECTORY = new_run("verification")
    torch.set_num_threads(1)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Verification)
    result = unittest.TextTestRunner(verbosity=2, resultclass=RecordedResult).run(suite)
    write_json(DIRECTORY / "verification.json", {"passed": result.wasSuccessful(), "tests": result.records,
                                               "scope": "Synthetic algorithm and localhost protocol tests; not game mastery"})
    print(f"[verification] Output: {DIRECTORY}", flush=True)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
