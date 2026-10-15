import argparse
import copy
import importlib.metadata
import json
import sys

import torch

from fanhua.artifacts import ROOT, inside_root, write_json
from train import check_config


def main():
    parser = argparse.ArgumentParser(description="Prepare matched ablation configs and record existing runtime")
    parser.add_argument("--base", default=str(ROOT / "configs" / "formal_candidate.json"))
    args = parser.parse_args()
    base = json.loads(inside_root(args.base).read_text(encoding="utf-8-sig"))
    base_name = inside_root(args.base).stem
    check_config(base)
    groups = {"full": {}, "without_behavior_constraint": {"behavior_weight": 0.0},
              "without_calibration": {"calibration": False},
              "without_history": {"history_probability": 0.0},
              "latent_sac": {"history_probability": 0.0, "calibration": False, "cql_weight": 0.0}}
    manifest = {"reference_commit": "35ba6fc167d9f3257a1c12d943600b096e79f6b9",
                "client_seed_supported": False, "configs": []}
    for group, overrides in groups.items():
        for seed in (0, 1, 2):
            values = copy.deepcopy(base)
            values.update(overrides, seed=seed)
            check_config(values)
            path = ROOT / "configs" / "ablations" / base_name / f"{group}_seed_{seed}.json"
            write_json(path, values)
            manifest["configs"].append({"group": group, "seed": seed, "path": str(path)})
    write_json(ROOT / "configs" / "ablations" / base_name / "manifest.json", manifest)
    runtime = {"python": sys.version, "executable": sys.executable, "cuda_available": torch.cuda.is_available(),
               "packages": {name: importlib.metadata.version(name) for name in
                            ("torch", "numpy", "websocket-server", "matplotlib")}}
    write_json(ROOT / "reports" / "runtime.json", runtime)
    print("[prepare] Matched configs and runtime saved inside FanHua; no training started.")


if __name__ == "__main__":
    main()
