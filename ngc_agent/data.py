"""Inspect manifests, missing files and split leakage without loading tensors."""
from collections import Counter
import csv
import hashlib
import os
from pathlib import Path


def label_value(value):
    key = str(value).strip().lower()
    if key in {"0", "0.0", "false", "no", "noise", "rfi", "negative", "non-pulsar", "non_pulsar", "nonpulsar"}:
        return 0
    if key in {"1", "1.0", "true", "yes", "pulsar", "psr", "positive"}:
        return 1
    raise ValueError(f"Invalid class label: {value!r}")


def resolve_packed_path(value, csv_path):
    # Manifests exported on Windows should also work on Linux with relative paths.
    value = value.strip().replace("\\", "/")
    if not value:
        raise ValueError("Empty packed_path")
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = (csv_path.parent.parent / path, csv_path.parent / path)
    return next((p.resolve() for p in candidates if p.is_file()), candidates[0].resolve())


def inspect_split(root, value):
    path = Path(value).expanduser()
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_file():
        return {"path": str(path), "ready": False, "error": "CSV 文件不存在", "_paths": set(), "_ids": set()}
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            label_key = next((key for key in ("label_bin", "label_int", "label") if key in columns), None)
            if "packed_path" not in columns or label_key is None:
                raise ValueError("CSV requires packed_path and label_bin/label_int/label")
            rows = list(reader)
        counts, paths, ids, missing, duplicate, errors = Counter(), set(), set(), [], [], []
        for index, row in enumerate(rows, 2):
            try:
                label = label_value(row.get(label_key, ""))
                packed = resolve_packed_path(row.get("packed_path", ""), path)
                canonical = str(packed).casefold() if os.name == "nt" else str(packed)
                if canonical in paths:
                    duplicate.append(index)
                paths.add(canonical)
                identity = (row.get("sample_id") or "").strip()
                if identity:
                    ids.add(identity)
                counts[label] += 1
                if not packed.is_file():
                    missing.append(index)
            except ValueError as exc:
                errors.append(f"row {index}: {exc}")
        return {
            "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": len(rows), "rfi": counts[0], "pulsar": counts[1],
            "missing_count": len(missing), "duplicate_count": len(duplicate),
            "errors": errors[:10], "ready": bool(rows) and not (missing or duplicate or errors),
            "_paths": paths, "_ids": ids,
        }
    except (ValueError, OSError, csv.Error, UnicodeError) as exc:
        return {"path": str(path), "ready": False, "error": str(exc), "_paths": set(), "_ids": set()}


def preflight(root, config, final_test=False):
    keys = ("train", "val", "test") if final_test else ("train", "val")
    splits = {name: inspect_split(root, getattr(config, f"{name}_csv")) for name in keys}
    overlaps = []
    for i, first in enumerate(keys):
        for second in keys[i + 1:]:
            common_paths = splits[first]["_paths"] & splits[second]["_paths"]
            common_ids = splits[first]["_ids"] & splits[second]["_ids"]
            if common_paths or common_ids:
                overlaps.append({"splits": [first, second], "path_count": len(common_paths), "sample_id_count": len(common_ids)})
    for split in splits.values():
        split.pop("_paths")
        split.pop("_ids")
        if split.get("ready") and min(split["rfi"], split["pulsar"]) < config.k_shot + config.q_shot:
            split["ready"] = False
            split["error"] = f"每类至少需要 k_shot + q_shot = {config.k_shot + config.q_shot} 个样本"
    return {
        "ready": all(split["ready"] for split in splits.values()) and not overlaps,
        "splits": splits, "overlaps": overlaps,
        "scope": "train+validation+test manifests" if final_test else "train+validation only; held-out test unopened",
        "limitation": "按文件路径及 sample_id 检查交集；同一观测源的不同候选仍需人工分组划分。",
    }
