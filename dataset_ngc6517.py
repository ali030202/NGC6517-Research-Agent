#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def to_label_int(value: Any) -> int:
    from ngc_agent.data import label_value
    return label_value(value)


def unwrap_npz_value(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
        try:
            return value.item()
        except Exception:
            return value
    return value


def nan_to_num(x: np.ndarray, fill: float = 0.0) -> np.ndarray:
    return np.nan_to_num(np.asarray(x), nan=fill, posinf=fill, neginf=fill)


def zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = nan_to_num(np.asarray(x, dtype=np.float32))
    return ((x - float(x.mean())) / (float(x.std()) + eps)).astype(np.float32)


def row_zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = nan_to_num(np.asarray(x, dtype=np.float32))
    med = np.median(x, axis=1, keepdims=True)
    std = np.std(x - med, axis=1, keepdims=True) + eps
    return ((x - med) / std).astype(np.float32)


def resample_1d(y: np.ndarray, length: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size == int(length):
        return y.astype(np.float32)
    if y.size <= 1:
        return np.zeros((int(length),), dtype=np.float32)
    old_x = np.linspace(0, y.size - 1, y.size, dtype=np.float32)
    new_x = np.linspace(0, y.size - 1, int(length), dtype=np.float32)
    return np.interp(new_x, old_x, y).astype(np.float32)


def resize_2d(x: np.ndarray, size: int) -> np.ndarray:
    t = torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(int(size), int(size)), mode="bilinear", align_corners=False)
    return t.squeeze(0).squeeze(0).numpy().astype(np.float32)


def extract_profile(obj: Any) -> np.ndarray:
    obj = unwrap_npz_value(obj)
    if isinstance(obj, dict):
        for key in ("profile", "prof", "profile_1d", "data"):
            if key in obj:
                return np.asarray(obj[key], dtype=np.float32).reshape(-1)
    return np.asarray(obj, dtype=np.float32).reshape(-1)


def extract_dm_curve(obj: Any) -> np.ndarray:
    obj = unwrap_npz_value(obj)
    if isinstance(obj, dict):
        for key in ("chi2", "dm_curve", "curve", "values", "y", "data"):
            if key in obj:
                return np.asarray(obj[key], dtype=np.float32).reshape(-1)
        for value in obj.values():
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if arr.size > 1:
                return arr
        return np.zeros((0,), dtype=np.float32)
    arr = np.asarray(obj, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, 1].reshape(-1)
    return arr.reshape(-1)


def extract_2d(obj: Any) -> np.ndarray:
    obj = unwrap_npz_value(obj)
    if isinstance(obj, dict):
        for key in ("subints", "subbands", "data", "arr", "image"):
            if key in obj:
                obj = obj[key]
                break
    arr = np.asarray(obj, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.squeeze()
    if arr.ndim != 2:
        flat = arr.reshape(-1)
        side = int(round(np.sqrt(flat.size)))
        if side > 0 and side * side == flat.size:
            arr = flat.reshape(side, side)
        else:
            arr = np.zeros((64, 64), dtype=np.float32)
    return nan_to_num(arr).astype(np.float32)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def width_at_levels(y: np.ndarray, base: float, height: float, levels: List[float]) -> List[float]:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size == 0 or height <= 1e-6:
        return [0.0 for _ in levels]
    out = []
    for level in levels:
        mask = y >= base + float(level) * height
        out.append(float(mask.sum()) / float(max(1, y.size)))
    return out


def profile_peak_features(profile: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    y = nan_to_num(np.asarray(profile, dtype=np.float32).reshape(-1))
    if y.size < 8:
        return np.zeros((9,), dtype=np.float32)
    base = float(np.percentile(y, 25.0))
    low = y[y <= np.percentile(y, 25.0)]
    noise = float(np.std(low)) + eps
    amp = np.maximum(y - base, 0.0)
    peak_i = int(np.argmax(amp))
    peak_h = float(amp[peak_i])
    if peak_h <= eps:
        return np.zeros((9,), dtype=np.float32)
    width50, width30 = width_at_levels(y, base, peak_h, [0.50, 0.30])
    win = max(1, int(round(0.03 * y.size)))
    total = float(np.sum(amp)) + eps
    energy = float(np.sum(amp[max(0, peak_i - win):min(y.size, peak_i + win + 1)]) / total)
    exclusion = max(1, int(round(0.05 * y.size)))
    outside = amp.copy()
    outside[max(0, peak_i - exclusion):min(y.size, peak_i + exclusion + 1)] = 0.0
    second = float(np.max(outside)) if outside.size else 0.0
    half_level = 0.5 * peak_h
    half_peaks = 0
    for i in range(1, amp.size - 1):
        if abs(i - peak_i) <= exclusion:
            continue
        if amp[i] >= half_level and amp[i] >= amp[i - 1] and amp[i] >= amp[i + 1]:
            half_peaks += 1
    deriv = np.abs(np.diff(y, append=y[:1]))
    deriv_conc = float(np.sum(deriv[max(0, peak_i - win):min(y.size, peak_i + win + 1)]) / (np.sum(deriv) + eps))
    return np.asarray(
        [
            float(peak_i) / float(max(1, y.size - 1)),
            float(np.clip(peak_h / noise, 0.0, 50.0) / 50.0),
            float(width50),
            float(width30),
            float(np.clip(energy, 0.0, 1.0)),
            float(np.clip(peak_h / (second + eps), 0.0, 20.0) / 20.0),
            float(np.clip(half_peaks, 0, 8) / 8.0),
            float(np.clip(deriv_conc, 0.0, 1.0)),
            float(np.clip(total / (y.size * (peak_h + eps)), 0.0, 1.0)),
        ],
        dtype=np.float32,
    )


def dm_peak_completeness(dm_curve: np.ndarray, eps: float = 1e-6) -> float:
    y = nan_to_num(np.asarray(dm_curve, dtype=np.float32).reshape(-1))
    if y.size < 8:
        return 0.0
    base = float(np.percentile(y, 25.0))
    peak_i = int(np.argmax(y))
    height = float(y[peak_i] - base)
    if height <= eps:
        return 0.0
    min_side = max(2, int(round(0.05 * y.size)))
    edge_score = min(1.0, peak_i / float(min_side), (y.size - peak_i - 1) / float(min_side))
    left = y[:peak_i]
    right = y[peak_i + 1:]
    left_drop = float(np.max(y[peak_i] - left)) / (height + eps) if left.size else 0.0
    right_drop = float(np.max(y[peak_i] - right)) / (height + eps) if right.size else 0.0
    return float(np.clip(edge_score * min(left_drop, right_drop), 0.0, 1.0))


def dm_peak_features(dm_curve: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    y = nan_to_num(np.asarray(dm_curve, dtype=np.float32).reshape(-1))
    if y.size < 8:
        return np.zeros((8,), dtype=np.float32)
    base = float(np.percentile(y, 25.0))
    peak_i = int(np.argmax(y))
    peak = float(y[peak_i])
    height = peak - base
    if height <= eps:
        return np.zeros((8,), dtype=np.float32)
    width50, width30 = width_at_levels(y, base, height, [0.50, 0.30])
    complete = dm_peak_completeness(y)
    offset = abs(float(peak_i) / float(max(1, y.size - 1)) - 0.5)
    left = y[:peak_i]
    right = y[peak_i + 1:]
    left_drop = float(peak - np.percentile(left, 25.0)) / (height + eps) if left.size else 0.0
    right_drop = float(peak - np.percentile(right, 25.0)) / (height + eps) if right.size else 0.0
    asymmetry = float(np.clip(abs(left_drop - right_drop), 0.0, 1.0))
    above = y >= base + 0.30 * height
    multi = max(0, int(np.sum(np.diff(above.astype(np.int8)) == 1)) - 1)
    return np.asarray(
        [
            float(np.clip(height / (float(np.std(y)) + eps), 0.0, 50.0) / 50.0),
            float(width50),
            float(width30),
            float(complete),
            float(np.clip(offset * 2.0, 0.0, 1.0)),
            float(asymmetry),
            float(np.clip(multi, 0, 8) / 8.0),
            float(np.clip(min(left_drop, right_drop), 0.0, 1.0)),
        ],
        dtype=np.float32,
    )


def local_response_features(profile: np.ndarray, image: np.ndarray, width: int = 7, eps: float = 1e-6) -> np.ndarray:
    prof = nan_to_num(np.asarray(profile, dtype=np.float32).reshape(-1))
    img = nan_to_num(np.asarray(image, dtype=np.float32))
    if prof.size < 4 or img.ndim != 2 or img.size == 0:
        return np.zeros((4,), dtype=np.float32)
    peak_phase = int(np.argmax(prof))
    col = int(round(float(peak_phase) / float(max(1, prof.size - 1)) * float(max(1, img.shape[1] - 1))))
    radius = max(1, int(width) // 2)
    lo = max(0, col - radius)
    hi = min(img.shape[1], col + radius + 1)
    window = img[:, lo:hi]
    global_abs = float(np.mean(np.abs(img))) + eps
    local_abs = float(np.mean(np.abs(window))) if window.size else 0.0
    row_peak = np.max(np.abs(window), axis=1) if window.size else np.zeros((img.shape[0],), dtype=np.float32)
    coverage = float(np.mean(row_peak > (np.mean(np.abs(img), axis=1) + np.std(np.abs(img), axis=1))))
    stability = 1.0 / (1.0 + float(np.std(row_peak)) / (float(np.mean(row_peak)) + eps))
    contrast = local_abs / global_abs
    return np.asarray(
        [
            float(np.clip(contrast / 5.0, 0.0, 1.0)),
            float(np.clip(coverage, 0.0, 1.0)),
            float(np.clip(stability, 0.0, 1.0)),
            float(col) / float(max(1, img.shape[1] - 1)),
        ],
        dtype=np.float32,
    )


def encoder_aux_features(profile: np.ndarray, dm_curve: np.ndarray, subints: np.ndarray, subbands: np.ndarray) -> np.ndarray:
    prof = profile_peak_features(profile)
    dm = dm_peak_features(dm_curve)
    si = local_response_features(profile, subints)
    sb = local_response_features(profile, subbands)
    cross = np.asarray([min(float(si[1]), float(sb[1])), max(float(si[0]), float(sb[0]))], dtype=np.float32)
    return nan_to_num(np.concatenate([prof, dm, si, sb, cross], axis=0)).astype(np.float32)


class NGC6517FewShotDataset(Dataset):
    """Reads split CSV rows and returns normalized multimodal tensors."""

    def __init__(
        self,
        csv_path: str,
        profile_len: int = 128,
        dm_len: int = 256,
        image_size: int = 64,
        cache: bool = True,
    ) -> None:
        self.csv_path = os.path.abspath(csv_path)
        self.profile_len = int(profile_len)
        self.dm_len = int(dm_len)
        self.image_size = int(image_size)
        self.cache_enabled = bool(cache)
        self._cache: Dict[int, Tuple[Dict[str, torch.Tensor], torch.Tensor]] = {}
        self.csv_dir = os.path.dirname(self.csv_path)
        self.package_dir = os.path.dirname(self.csv_dir)

        with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"{csv_path} is empty")
        columns = set(rows[0].keys())
        if "packed_path" not in columns:
            raise ValueError(f"{csv_path} has no packed_path column")
        if not ({"label_bin", "label_int", "label"} & columns):
            raise ValueError(f"{csv_path} has no label, label_int, or label_bin column")
        clean_rows = []
        for row in rows:
            packed_path = self._resolve_packed_path(str(row.get("packed_path", "")).strip())
            if not packed_path or not os.path.isfile(packed_path):
                raise FileNotFoundError(f"Missing sample in {csv_path}: {packed_path}")
            row["packed_path"] = packed_path
            if "label_bin" not in row or str(row.get("label_bin", "")).strip() == "":
                row["label_bin"] = to_label_int(row.get("label_int", row.get("label", "")))
            else:
                row["label_bin"] = to_label_int(row["label_bin"])
            clean_rows.append(row)
        if not clean_rows:
            raise ValueError(f"{csv_path} has no usable packed_path rows")
        self.rows = clean_rows

    def _resolve_packed_path(self, value: str) -> str:
        from pathlib import Path
        from ngc_agent.data import resolve_packed_path
        return str(resolve_packed_path(value, Path(self.csv_path)))

    def __len__(self) -> int:
        return len(self.rows)

    def class_indices(self) -> Dict[int, np.ndarray]:
        labels = np.asarray([int(row["label_bin"]) for row in self.rows], dtype=np.int64)
        return {
            0: np.where(labels == 0)[0].astype(np.int64),
            1: np.where(labels == 1)[0].astype(np.int64),
        }

    def summary(self) -> Dict[str, Any]:
        labels = [int(row["label_bin"]) for row in self.rows]
        return {
            "path": self.csv_path,
            "total": int(len(self.rows)),
            "rfi": int(sum(1 for y in labels if y == 0)),
            "pulsar": int(sum(1 for y in labels if y == 1)),
        }

    def __getitem__(self, index: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        index = int(index)
        if self.cache_enabled and index in self._cache:
            return self._cache[index]

        row = self.rows[index]
        packed_path = os.path.normpath(str(row["packed_path"]))
        # Legacy observation files may contain object arrays; load only trusted local data.
        with np.load(packed_path, allow_pickle=True) as npz:
            profile_raw = extract_profile(npz["profile"])
            dm_raw = extract_dm_curve(npz["dm_curve"])
            subints_raw = extract_2d(npz["subints"])
            subbands_raw = extract_2d(npz["subbands"])
        profile = zscore(resample_1d(profile_raw, self.profile_len))
        dm = zscore(resample_1d(dm_raw, self.dm_len))
        subints = resize_2d(row_zscore(subints_raw), self.image_size)
        subbands = resize_2d(row_zscore(subbands_raw), self.image_size)
        aux = encoder_aux_features(profile, dm_raw if dm_raw.size else dm, subints, subbands)

        meta = np.asarray(
            [
                np.log1p(max(safe_float(row.get("sigma", 0.0)), 0.0)),
                np.log1p(max(safe_float(row.get("red_chi2", 0.0)), 0.0)),
                safe_float(row.get("best_dm", row.get("best_dm_used", 0.0))) / 300.0,
                np.log10(max(safe_float(row.get("p_topo_s", row.get("p_topo_s_used", 0.0))), 1e-6)),
                safe_float(row.get("dm_nu2_score", 0.0)),
            ],
            dtype=np.float32,
        )
        meta = nan_to_num(meta).astype(np.float32)

        sample = {
            "profile": torch.from_numpy(profile).unsqueeze(0),
            "dm": torch.from_numpy(dm).unsqueeze(0),
            "image": torch.from_numpy(np.stack([subints, subbands], axis=0)),
            "meta": torch.from_numpy(meta),
            "aux": torch.from_numpy(aux),
            "index": torch.tensor(index, dtype=torch.long),
        }
        label = torch.tensor(int(row["label_bin"]), dtype=torch.long)
        item = (sample, label)
        if self.cache_enabled:
            self._cache[index] = item
        return item


def stack_items(
    ds: NGC6517FewShotDataset,
    indices: np.ndarray,
    labels: Optional[List[int]] = None,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[str]]:
    xs = []
    ys = []
    sample_ids = []
    if labels is None:
        labels = [None] * len(indices)
    for idx, label_override in zip(indices, labels):
        x, y = ds[int(idx)]
        xs.append(x)
        ys.append(y if label_override is None else torch.tensor(int(label_override), dtype=torch.long))
        sample_ids.append(str(ds.rows[int(idx)].get("sample_id", int(idx))))
    batch = {
        "profile": torch.stack([x["profile"] for x in xs], dim=0),
        "dm": torch.stack([x["dm"] for x in xs], dim=0),
        "image": torch.stack([x["image"] for x in xs], dim=0),
        "meta": torch.stack([x["meta"] for x in xs], dim=0),
        "aux": torch.stack([x["aux"] for x in xs], dim=0),
        "index": torch.stack([x["index"] for x in xs], dim=0),
    }
    return batch, torch.stack(ys, dim=0), sample_ids
