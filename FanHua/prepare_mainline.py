import hashlib
import json
from pathlib import Path

import torch

from fanhua.agent import resume_config
from fanhua.artifacts import ROOT, inside_root, write_json
from train import check_config


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    config_path = ROOT / "configs/mainline_shared_model.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    check_config(config)
    source = inside_root(ROOT / config["source_checkpoint"])
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise RuntimeError("Unsupported source checkpoint format")
    if resume_config(checkpoint["config"]) != resume_config(config):
        raise RuntimeError("Source checkpoint and mainline training parameters differ")
    teacher_hashes = {}
    for task, path in config["teacher_paths"].items():
        teacher_path = Path(path)
        if not teacher_path.exists():
            raise FileNotFoundError(f"Missing teacher file: {teacher_path}")
        teacher_hashes[task] = sha256(teacher_path)
    source_hash = sha256(source)
    report = {
        "status": "passed",
        "config": str(config_path.relative_to(ROOT)),
        "source_checkpoint": config["source_checkpoint"],
        "source_checkpoint_sha256": source_hash,
        "source_checkpoint_size": source.stat().st_size,
        "source_checkpoint_format": checkpoint["format_version"],
        "source_checkpoint_algorithm": checkpoint["algorithm"],
        "source_checkpoint_updates": checkpoint["updates"],
        "source_checkpoint_progress": checkpoint["progress"],
        "resume_parameter_match": True,
        "teacher_sha256": teacher_hashes,
        "stage_evaluation_episodes": config["stage_evaluation_episodes"],
        "stage_evaluation_steps": config["stage_evaluation_steps"],
        "total_training_steps": sum(stage["steps"] for stage in config["stages"]),
        "stage_order": config["stages"],
        "parameter_update": "not performed",
        "wemi_connection": "not started",
        "formal_ablation_groups": "held",
    }
    write_json(ROOT / "reports/mainline_preflight.json", report)
    write_json(ROOT / "reports/mainline_source_checkpoint.json", {
        "path": config["source_checkpoint"],
        "sha256": source_hash,
        "size": source.stat().st_size,
        "algorithm": checkpoint["algorithm"],
        "updates": checkpoint["updates"],
        "progress": checkpoint["progress"],
        "teacher_sha256": teacher_hashes,
    })
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
