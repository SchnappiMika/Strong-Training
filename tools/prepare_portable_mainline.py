import argparse
import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Prepare a relocated shared starting checkpoint; never starts training")
    parser.add_argument("--label", default="portable_mainline")
    args = parser.parse_args()
    if not args.label or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in args.label):
        parser.error("Label must contain letters, digits, underscores or hyphens")
    import torch

    root = Path(__file__).resolve().parents[1]
    fanhua = root / "FanHua"
    source = fanhua / "experiments/20261008_122246_330413_mainline_shared_model/checkpoints/initialized.pt"
    original = source.read_bytes()
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    progress = checkpoint["progress"]
    if progress["global_steps"] != 0 or progress["stage"] != 0 or checkpoint["updates"] != 5052:
        raise ValueError("Unexpected mainline starting state")
    config = copy.deepcopy(checkpoint["config"])
    config["teacher_paths"] = {
        "flappybird": str(root / "flappybird_rl/logs/ppo_20260925_174115_model.pt"),
        "timberman": str(fanhua / "experiments/20261006_161456_139334_timber_first_smoke/checkpoints/latest.pt"),
    }
    for teacher_path in config["teacher_paths"].values():
        if not Path(teacher_path).is_file():
            raise FileNotFoundError(f"Download full archive first: {teacher_path}")
    checkpoint["config"] = config
    directory = fanhua / "experiments" / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{args.label}"
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "checkpoints").mkdir()
    destination = directory / "checkpoints/initialized.pt"
    torch.save(checkpoint, destination)
    (directory / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {"source": str(source.relative_to(root)), "source_sha256": hashlib.sha256(original).hexdigest(),
              "parameters_updated": False, "training_started": False,
              "change": "Relocated teacher paths only; networks, optimizers, replay, RNG and progress retained"}
    (directory / "portability.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if source.read_bytes() != original:
        raise RuntimeError("Original starting checkpoint changed")
    relative = directory.relative_to(fanhua).as_posix()
    print(f"Prepared: {directory}")
    print("No training or environment connection started. To train explicitly, run from FanHua:")
    print(f"python train.py --config {relative}/config.json --resume {relative}/checkpoints/initialized.pt")


if __name__ == "__main__":
    main()
