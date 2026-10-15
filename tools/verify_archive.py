import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify(root, full=False):
    manifest = json.loads((root / "docs/ARCHIVE_MANIFEST.json").read_text(encoding="utf-8"))
    checked = 0
    for entry in manifest["files"]:
        if not full and entry["storage"] == "release":
            continue
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"Manifest path escapes project: {entry['path']}")
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError(f"File integrity mismatch: {entry['path']}")
        checked += 1
    acceptance = json.loads((root / "FanHua/reports/最终验收_单模型双游戏.json").read_text(encoding="utf-8"))
    expected_hash = acceptance["training"]["checkpoint_sha256"]
    evaluations = ["20261008_172006_553338_evaluate_flappybird", "20261008_200326_813205_evaluate_timberman"]
    for run in evaluations:
        report = json.loads((root / "FanHua/experiments" / run / "evaluation.json").read_text(encoding="utf-8"))
        if not (report["status"] == "completed" and report["checkpoint_sha256"] == expected_hash
                and report["updates_before"] == report["updates_after"] == 73056
                and report["parameters_before"] == report["parameters_after"]
                and len(report["episodes"]) == 20
                and all(row["steps"] == 3000 and not row["terminated"] and row["truncated"]
                        for row in report["episodes"])):
            raise ValueError(f"Final evaluation evidence mismatch: {run}")
    if full:
        checkpoint = root / "FanHua" / acceptance["training"]["checkpoint"]
        if digest(checkpoint) != expected_hash:
            raise ValueError("Final shared checkpoint hash mismatch")
    return {"passed": True, "scope": "full archive" if full else "git files",
            "files_verified": checked, "same_shared_checkpoint": True,
            "evaluation_episodes_verified": 40, "training_started": False}


def main():
    parser = argparse.ArgumentParser(description="Verify archived files and existing evaluation evidence without running models")
    parser.add_argument("--full", action="store_true", help="Require Release files, including all checkpoints and large logs")
    args = parser.parse_args()
    print(json.dumps(verify(Path(__file__).resolve().parents[1], args.full), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
