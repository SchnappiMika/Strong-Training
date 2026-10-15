import argparse
import hashlib

import torch

from fanhua.artifacts import inside_root, new_run, write_json
from fanhua.envs import TASKS
from fanhua.replay import validate_transitions


def main():
    parser = argparse.ArgumentParser(description="Export actual replay transitions with explicit behavior provenance")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    args = parser.parse_args()
    source = inside_root(args.checkpoint)
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if checkpoint["format_version"] != 1:
        raise ValueError("Unsupported checkpoint format")
    replay = checkpoint["replay"]
    rows = [row for pool in [replay["online"], *replay["history"].values()]
            for row in pool["rows"] if row["task"] == args.task]
    validate_transitions(rows, checkpoint["config"]["embedding_dim"])
    directory = new_run(f"export_{args.task}_training_behavior")
    dataset = {"format_version": 1, "rows": rows, "gamma": checkpoint["config"]["gamma"],
               "behavior": "mixed_random_warmup_and_online_latent_SAC",
               "encoding_version": "wemi_12_v1", "reward_version": "flappy_scaled_timber_delta_v1",
               "source_checkpoint": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    torch.save(dataset, directory / "dataset.pt")
    report = {key: value for key, value in dataset.items() if key != "rows"}
    report.update(task=args.task, transitions=len(rows), valid_bounds=sum(row["valid_bound"] for row in rows),
                  action_counts={str(action): sum(row["action"] == action for row in rows) for action in (0, 1)})
    write_json(directory / "dataset.json", report)
    print(f"[export] {len(rows)} real {args.task} transitions. Output: {directory}")


if __name__ == "__main__":
    main()
