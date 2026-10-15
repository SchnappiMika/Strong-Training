import argparse
import json
from collections import Counter

import numpy as np
import torch

from fanhua.agent import Agent
from fanhua.artifacts import inside_root, new_run, write_json
from fanhua.envs import encode_observation, timberman_action_hazard
from fanhua.replay import Replay


def load_agent(path):
    saved = torch.load(inside_root(path), map_location="cpu", weights_only=True)
    agent = Agent(saved["config"])
    agent.load(path, Replay(saved["config"]["capacity"]), np.random.default_rng(0))
    return agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate")
    args = parser.parse_args()
    torch.set_num_threads(1)
    source = inside_root(args.run)
    evaluation = json.loads((source / "evaluation.json").read_text(encoding="utf-8"))
    candidate_path = args.candidate or evaluation["checkpoint"]
    candidate = load_agent(candidate_path)
    reference = load_agent(args.reference)
    state = None
    request = None
    steps = []
    episode = -1
    with (source / "protocol.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            if entry["direction"] == "send":
                request = entry["message"]
            else:
                response = json.loads(entry["message"])
                next_state = encode_observation("timberman", response["obs"])
                if request["restart"]:
                    episode += 1
                else:
                    candidate_action, _ = candidate.select(state, "timberman", deterministic=True)
                    reference_action, _ = reference.select(state, "timberman", deterministic=True)
                    steps.append({"episode": episode, "action": request["action"],
                                  "candidate_action": candidate_action, "reference_action": reference_action,
                                  "first_branch": float(state[0]), "second_branch": float(state[1]),
                                  "side": float(state[-1]), "score": response["reward"], "done": response["done"],
                                  "candidate_hazard": timberman_action_hazard(state, candidate_action),
                                  "reference_hazard": timberman_action_hazard(state, reference_action)})
                state = next_state
    hazards = [row for row in steps if row["candidate_hazard"] not in (None, "already_in_collision")]
    report = {"source": str(source), "candidate": candidate_path, "reference": args.reference,
              "step_count": len(steps), "candidate_matches_logged_actions": all(row["action"] == row["candidate_action"] for row in steps),
              "action_disagreements": sum(row["candidate_action"] != row["reference_action"] for row in steps),
              "candidate_hazards": dict(Counter(row["candidate_hazard"] for row in steps if row["candidate_hazard"])),
              "reference_hazards": dict(Counter(row["reference_hazard"] for row in steps if row["reference_hazard"])),
              "episodes_with_avoidable_hazard": len({row["episode"] for row in hazards if row["reference_hazard"] is None}),
              "hazard_examples": hazards[:20], "steps": steps,
              "scope": "Counterfactual actions on recorded states, using original client run()/collisionDetection() semantics; not simulated score evidence"}
    directory = new_run("diagnose_timberman")
    write_json(directory / "diagnosis.json", report)
    print(json.dumps({key: value for key, value in report.items() if key not in ("steps", "hazard_examples")}, indent=2))
    print(f"[diagnosis] Output: {directory}")


if __name__ == "__main__":
    main()
