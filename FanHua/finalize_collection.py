import argparse
import json

import torch

from fanhua.artifacts import inside_root, write_json


def main():
    parser = argparse.ArgumentParser(description="Record preserved completed data after externally stopping collection")
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    directory = inside_root(args.run)
    dataset = torch.load(directory / "dataset.pt", map_location="cpu", weights_only=True)
    recorded = []
    actions = []
    rewards = []
    with (directory / "protocol.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            if entry["direction"] == "send" and entry["message"]["restart"]:
                actions = []
                rewards = []
            elif entry["direction"] == "send":
                actions.append(entry["message"]["action"])
            elif entry["direction"] == "recv":
                payload = json.loads(entry["message"])
                if actions:
                    rewards.append(payload["reward"] / 100.0)
                    if payload["done"] or len(rewards) == 800:
                        recorded.append({"episode": len(recorded), "steps": len(rewards),
                                         "return": sum(rewards), "terminated": payload["done"],
                                         "truncated": not payload["done"], "score": None})
    saved_count = len(dataset["rows"])
    preserved = []
    total = 0
    for row in recorded:
        if total + row["steps"] > saved_count:
            break
        preserved.append(row)
        total += row["steps"]
    if total != saved_count:
        raise ValueError("Protocol completed episodes do not match preserved dataset size")
    write_json(directory / "dataset.json", {
        "status": "stopped_after_sufficient_smoke_data_completed_episodes_preserved",
        "task": "flappybird", "behavior": dataset["behavior"], "transitions": saved_count,
        "episodes": preserved, "client_seed_supported": False,
        "reason": "Preserved completed data is sufficient for initial offline pretraining; stopped original 20-episode budget to evaluate candidate sooner",
        "incomplete_tail_in_protocol_only": True})
    print(f"[collection] Preserved {saved_count} transitions from {len(preserved)} completed/capped episodes")


if __name__ == "__main__":
    main()
