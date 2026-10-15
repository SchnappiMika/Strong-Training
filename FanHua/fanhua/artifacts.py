import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def inside_root(path):
    target = Path(path).resolve()
    if not target.is_relative_to(ROOT):
        raise ValueError(f"Output must stay inside FanHua: {target}")
    return target


def new_run(label):
    if not label or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in label):
        raise ValueError("Run label may contain only letters, digits, _ and -")
    directory = ROOT / "experiments" / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{label}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def write_json(path, data):
    path = inside_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def append_event(path, event, **details):
    path = inside_root(path)
    entry = {"time_utc": datetime.now(timezone.utc).isoformat(), "event": event, **details}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")


class CsvLog:
    def __init__(self, path, fields):
        self.file = open(inside_root(path), "a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fields)
        if self.file.tell() == 0:
            self.writer.writeheader()
            self.file.flush()

    def write(self, data):
        self.writer.writerow(data)
        self.file.flush()

    def close(self):
        self.file.close()
