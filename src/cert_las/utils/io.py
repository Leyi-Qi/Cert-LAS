import csv
import json
from pathlib import Path
from typing import Any, Dict, Sequence, Union


PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: PathLike, payload: Dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def append_jsonl(path: PathLike, payload: Dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def write_csv_rows(path: PathLike, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    path = Path(path)
    ensure_dir(path.parent)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: PathLike) -> list[Dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_float_column(path: PathLike, column: str) -> list[float]:
    values = []
    for row in read_csv_rows(path):
        value = row.get(column)
        if value is None or value == "":
            continue
        values.append(float(value))
    return values
