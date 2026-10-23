#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import csv
from datetime import datetime
from pathlib import Path
import json
import os
import random
import sys
from typing import Dict, List, Tuple

import numpy as np

try:
    from tqdm import trange
except ModuleNotFoundError as exc:
    raise SystemExit(
        "tqdm is required for the demo3-style progress bar. "
        "Install it in the active environment, for example: conda install tqdm"
    ) from exc

try:
    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is required for training. Install torch in this Python "
        "environment, or run agent.py inspect to inspect the NGC6517 few-shot "
        "task without torch."
    ) from exc

from dataset_ngc6517 import NGC6517FewShotDataset, stack_items
from model_proto import NGC6517Encoder, prototypical_logits
from ngc_agent.data import preflight


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_run_logging(out_dir: str) -> str:
    ensure_dir(out_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(out_dir, f"train_log_{stamp}.txt")
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.__stdout__, log_file)
    sys.stderr = TeeStream(sys.__stderr__, log_file)
    return log_path


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def sample_indices(pool: np.ndarray, n: int, exclude=None) -> np.ndarray:
    pool = np.asarray(pool, dtype=np.int64)
    if exclude is not None and len(exclude):
        pool = np.setdiff1d(pool, np.asarray(exclude, dtype=np.int64), assume_unique=False)
    if len(pool) == 0:
        raise ValueError("cannot sample from an empty class pool")
    replace = len(pool) < int(n)
    return np.random.choice(pool, size=int(n), replace=replace).astype(np.int64)


def sample_indices_with_rng(pool: np.ndarray, n: int, rng, exclude=None) -> np.ndarray:
    pool = np.asarray(pool, dtype=np.int64)
    if exclude is not None and len(exclude):
        pool = np.setdiff1d(pool, np.asarray(exclude, dtype=np.int64), assume_unique=False)
    if len(pool) == 0:
        raise ValueError("cannot sample from an empty class pool")
    replace = len(pool) < int(n)
    return rng.choice(pool, size=int(n), replace=replace).astype(np.int64)


def make_episode(
    ds: NGC6517FewShotDataset,
    by_class: Dict[int, np.ndarray],
    k_shot: int,
    q_shot: int,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    s0 = sample_indices(by_class[0], k_shot)
    s1 = sample_indices(by_class[1], k_shot)
    q0 = sample_indices(by_class[0], q_shot, exclude=s0)
    q1 = sample_indices(by_class[1], q_shot, exclude=s1)
    support = np.concatenate([s0, s1])
    query = np.concatenate([q0, q1])
    np.random.shuffle(support)
    np.random.shuffle(query)
    sb, sy, _ = stack_items(ds, support)
    qb, qy, _ = stack_items(ds, query)
    return sb, sy, qb, qy


def build_episode_specs(
    ds: NGC6517FewShotDataset,
    episodes: int,
    k_shot: int,
    q_shot: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(int(seed))
    by_class = ds.class_indices()
    specs = []
    for _ in range(int(episodes)):
        s0 = sample_indices_with_rng(by_class[0], k_shot, rng)
        s1 = sample_indices_with_rng(by_class[1], k_shot, rng)
        q0 = sample_indices_with_rng(by_class[0], q_shot, rng, exclude=s0)
        q1 = sample_indices_with_rng(by_class[1], q_shot, rng, exclude=s1)
        support = np.concatenate([s0, s1]).astype(np.int64)
        query = np.concatenate([q0, q1]).astype(np.int64)
        rng.shuffle(support)
        rng.shuffle(query)
        specs.append((support, query))
    return specs


def load_episode_specs_from_bank(
    bank_path: str,
    expected_episodes: int = 0,
    expected_k_shot: int = 0,
    expected_q_shot: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if not bank_path:
        return []
    with open(bank_path, "r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    meta = payload.get("meta", {})
    if expected_k_shot and int(meta.get("k_shot", expected_k_shot)) != int(expected_k_shot):
        raise ValueError(f"{bank_path} k_shot={meta.get('k_shot')} does not match --k_shot={expected_k_shot}")
    if expected_q_shot and int(meta.get("q_query", expected_q_shot)) != int(expected_q_shot):
        raise ValueError(f"{bank_path} q_query={meta.get('q_query')} does not match --q_shot={expected_q_shot}")
    specs = []
    for episode in payload.get("episodes", []):
        support = np.asarray(episode["support_indices"], dtype=np.int64)
        query = np.asarray(episode["query_indices"], dtype=np.int64)
        specs.append((support, query))
    if expected_episodes:
        if len(specs) < expected_episodes:
            raise ValueError("Episode bank is smaller than requested evaluation budget")
        specs = specs[:int(expected_episodes)]
    if not specs:
        raise ValueError(f"{bank_path} has no episode specs")
    return specs


def metrics_from_counts(tp: int, tn: int, fp: int, fn: int, cost_fn: float = 20.0, cost_fp: float = 10.0) -> Dict[str, float]:
    total = tp + tn + fp + fn
    acc = (tp + tn) / max(1, total)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    fpr = fp / max(1, fp + tn)
    fnr = fn / max(1, fn + tp)
    return {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fnr": float(fnr),
        "fpr": float(fpr),
        "cost": float(cost_fn) * float(fnr) + float(cost_fp) * float(fpr),
        "tp": int(tp),
        "fn": int(fn),
        "tn": int(tn),
        "fp": int(fp),
        "n": int(total),
    }


def metrics_from_scores(y_true, prob_pulsar, threshold: float, cost_fn: float = 20.0, cost_fp: float = 10.0) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(prob_pulsar, dtype=np.float32)
    pred = (p >= float(threshold)).astype(np.int64)
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    out = metrics_from_counts(tp, tn, fp, fn, cost_fn=cost_fn, cost_fp=cost_fp)
    out["decision_threshold"] = float(threshold)
    return out


def strip_score_cache(m: Dict[str, float]) -> Dict[str, float]:
    return {k: v for k, v in m.items() if not str(k).startswith("_")}


def tune_threshold(fixed, random_m, fixed_weight: float, cost_fn: float, cost_fp: float, lo: float, hi: float, steps: int):
    thresholds = np.linspace(float(lo), float(hi), int(max(2, steps)), dtype=np.float32)
    best = None
    for thr in thresholds:
        fixed_m = metrics_from_scores(fixed["_y_true"], fixed["_prob"], float(thr), cost_fn=cost_fn, cost_fp=cost_fp)
        random_eval = metrics_from_scores(random_m["_y_true"], random_m["_prob"], float(thr), cost_fn=cost_fn, cost_fp=cost_fp)
        selected = mix_metrics(fixed_m, random_eval, fixed_weight=fixed_weight)
        selected["decision_threshold"] = float(thr)
        if best is None or selected["f1"] > best[2]["f1"] or (
            np.isclose(selected["f1"], best[2]["f1"]) and selected["cost"] < best[2]["cost"]
        ):
            best = (fixed_m, random_eval, selected)
    return best


@torch.no_grad()
def evaluate(
    model: NGC6517Encoder,
    ds: NGC6517FewShotDataset,
    episodes: int,
    k_shot: int,
    q_shot: int,
    temperature: float,
    device: torch.device,
    psr_k: int = 1,
    rfi_k: int = 1,
    cost_fn: float = 20.0,
    cost_fp: float = 10.0,
    out_csv: str = "",
    episode_specs: List[Tuple[np.ndarray, np.ndarray]] = None,
    decision_threshold: float = 0.5,
    return_scores: bool = False,
) -> Dict[str, float]:
    model.eval()
    by_class = ds.class_indices()
    tp = tn = fp = fn = 0
    loss_sum = 0.0
    rows = []
    y_all = []
    prob_all = []
    specs = episode_specs[:int(episodes)] if episode_specs is not None else None
    n_episodes = len(specs) if specs is not None else int(episodes)
    if specs is not None:
        validate_episode_specs(ds, specs, k_shot, q_shot)
    for epi in range(n_episodes):
        if specs is None:
            sb, sy, qb, qy = make_episode(ds, by_class, k_shot, q_shot)
        else:
            support_idx, query_idx = specs[epi]
            sb, sy, _ = stack_items(ds, support_idx)
            qb, qy, _ = stack_items(ds, query_idx)
        sb = move_batch(sb, device)
        qb = move_batch(qb, device)
        sy = sy.to(device)
        qy = qy.to(device)
        logits = prototypical_logits(model(sb), sy, model(qb), temperature=temperature, psr_k=psr_k, rfi_k=rfi_k)
        loss_sum += float(F.cross_entropy(logits, qy).item())
        prob = torch.softmax(logits, dim=1)[:, 1]
        pred = (prob >= float(decision_threshold)).long()
        sample_ids = [str(ds.rows[int(idx)].get("sample_id", int(idx))) for idx in qb["index"].cpu().tolist()]
        for y, p, score, idx, sid in zip(qy.cpu().tolist(), pred.cpu().tolist(), prob.cpu().tolist(), qb["index"].cpu().tolist(), sample_ids):
            y_all.append(int(y))
            prob_all.append(float(score))
            tp += int(y == 1 and p == 1)
            tn += int(y == 0 and p == 0)
            fp += int(y == 0 and p == 1)
            fn += int(y == 1 and p == 0)
            if out_csv:
                rows.append(
                    {
                        "episode": int(epi + 1),
                        "csv_index": int(idx),
                        "sample_id": sid,
                        "true_label": int(y),
                        "pred_label": int(p),
                        "prob_pulsar": float(score),
                        "is_tp": int(y == 1 and p == 1),
                        "is_fn": int(y == 1 and p == 0),
                        "is_fp": int(y == 0 and p == 1),
                        "is_tn": int(y == 0 and p == 0),
                    }
                )
    out = metrics_from_counts(tp, tn, fp, fn, cost_fn=cost_fn, cost_fp=cost_fp)
    out["loss"] = float(loss_sum / max(1, n_episodes))
    out["episodes"] = int(n_episodes)
    out["decision_threshold"] = float(decision_threshold)
    if return_scores:
        out["_y_true"] = y_all
        out["_prob"] = prob_all
    if out_csv:
        ensure_dir(os.path.dirname(out_csv))
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["sample_id"])
            writer.writeheader()
            writer.writerows(rows)
    return out


def fmt(name: str, m: Dict[str, float]) -> str:
    return (
        f"{name} acc={m['acc']:.4f} p={m['precision']:.4f} r={m['recall']:.4f} "
        f"f1={m['f1']:.4f} fnr={m['fnr']:.4f} fpr={m['fpr']:.4f} cost={m['cost']:.6f}"
    )


def mix_metrics(fixed, random_m, fixed_weight: float = 0.7) -> Dict[str, float]:
    if fixed is None:
        out = dict(random_m)
        out["source"] = "random_only"
        return out
    w = float(fixed_weight)
    out = {}
    for key in ["acc", "precision", "recall", "f1", "fnr", "fpr", "cost", "loss"]:
        if key in fixed and key in random_m:
            out[key] = w * float(fixed[key]) + (1.0 - w) * float(random_m[key])
    out["tp"] = int(fixed.get("tp", 0))
    out["fn"] = int(fixed.get("fn", 0))
    out["tn"] = int(fixed.get("tn", 0))
    out["fp"] = int(fixed.get("fp", 0))
    out["n"] = int(fixed.get("n", 0))
    out["source"] = f"fixed_{w:.2f}_random_{1.0 - w:.2f}"
    return out


def progress_write(iterator, message: str) -> None:
    print(message)


def write_summary(out_dir, summary):
    temporary = os.path.join(out_dir, "summary.json.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temporary, os.path.join(out_dir, "summary.json"))


def validate_episode_specs(ds, specs, k_shot, q_shot):
    for support, query in specs:
        if len(support) != 2 * k_shot or len(query) != 2 * q_shot:
            raise ValueError("Episode bank has incompatible support/query counts")
        if len(set(support.tolist())) != len(support) or len(set(query.tolist())) != len(query):
            raise ValueError("Episode bank contains duplicate samples")
        if set(support.tolist()) & set(query.tolist()):
            raise ValueError("Support/query overlap detected in episode bank")
        if any(index < 0 or index >= len(ds) for index in [*support, *query]):
            raise ValueError("Episode bank contains an out-of-bounds index")
        for indices, expected in ((support, k_shot), (query, q_shot)):
            if sum(int(ds.rows[int(index)]["label_bin"]) for index in indices) != expected:
                raise ValueError("Episode bank is not class balanced")


def final_test(args, model, device, threshold):
    ds = NGC6517FewShotDataset(args.test_csv)
    count = int(args.test_fixed_eval_episodes or args.test_episodes)
    seed = int(args.eval_seed if args.eval_seed is not None else args.seed + 10000)
    if args.test_fixed_bank:
        specs = load_episode_specs_from_bank(args.test_fixed_bank, count, args.k_shot, args.q_shot)
    else:
        specs = build_episode_specs(ds, count, args.k_shot, args.q_shot, seed + 1)
    validate_episode_specs(ds, specs, args.k_shot, args.q_shot)
    predictions = os.path.join(args.out_dir, "test_predictions.csv")
    metrics = evaluate(model, ds, count, args.k_shot, args.q_shot, args.temperature, device,
                       psr_k=args.psr_k, rfi_k=args.rfi_k, cost_fn=args.cost_fn, cost_fp=args.cost_fp,
                       out_csv=predictions, episode_specs=specs, decision_threshold=threshold)
    print(f"[FINAL-TEST] {fmt('TEST', metrics)}")
    return metrics, predictions


def main() -> None:
    ap = argparse.ArgumentParser(description="NGC6517 2-way few-shot pulsar/RFI classification")
    ap.add_argument("--train_csv", default="splits1/train.csv")
    ap.add_argument("--val_csv", default="splits1/val.csv")
    ap.add_argument("--test_csv", default="splits1/test.csv")
    ap.add_argument("--out_dir", default=r"out_demo2")
    ap.add_argument("--save_path", default="out_demo2/ngc6517_proto.pt")
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--eval_episodes", type=int, default=100)
    ap.add_argument("--val_fixed_eval_episodes", type=int, default=0)
    ap.add_argument("--val_random_eval_episodes", type=int, default=100)
    ap.add_argument("--val_random_select_weight", type=float, default=0.3)
    ap.add_argument("--test_episodes", type=int, default=200)
    ap.add_argument("--test_fixed_eval_episodes", type=int, default=0)
    ap.add_argument("--val_fixed_bank", default="")
    ap.add_argument("--test_fixed_bank", default="")
    ap.add_argument("--k_shot", type=int, default=20)
    ap.add_argument("--q_shot", type=int, default=20)
    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--no_prior", action="store_true", help="Disable demo3-style prior-guided encoder gates.")
    ap.add_argument("--temperature", type=float, default=10.0)
    ap.add_argument("--psr_k", type=int, default=1)
    ap.add_argument("--rfi_k", type=int, default=1)
    ap.add_argument("--decision_threshold", type=float, default=0.5)
    ap.add_argument("--auto_threshold", action="store_true")
    ap.add_argument("--threshold_min", type=float, default=0.35)
    ap.add_argument("--threshold_max", type=float, default=0.60)
    ap.add_argument("--threshold_steps", type=int, default=41)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--weight_psr", type=float, default=1.0)
    ap.add_argument("--cost_fn", type=float, default=20.0)
    ap.add_argument("--cost_fp", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--eval_seed", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--device", default="")
    ap.add_argument("--disable_tqdm", action="store_true")
    ap.add_argument("--final_test", action="store_true", help="Explicit held-out test evaluation, after all model selection.")
    ap.add_argument("--evaluate_checkpoint", default="", help="Evaluate a frozen checkpoint without any training; requires --final_test.")
    args = ap.parse_args()
    for name in ("episodes", "eval_every", "eval_episodes", "val_random_eval_episodes", "test_episodes", "k_shot", "q_shot", "embed_dim", "psr_k", "rfi_k", "threshold_steps"):
        if getattr(args, name) <= 0:
            ap.error(f"--{name} must be positive")
    for name in ("lr", "temperature", "weight_psr"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            ap.error(f"--{name} must be positive and finite")
    if not 0 <= args.dropout < 1 or not 0 <= args.val_random_select_weight <= 1:
        ap.error("dropout or validation mixing weight is outside its allowed range")
    if not 0 <= args.threshold_min < args.threshold_max <= 1 or not 0 <= args.decision_threshold <= 1:
        ap.error("invalid decision threshold range")
    if args.evaluate_checkpoint and not args.final_test:
        ap.error("--evaluate_checkpoint requires --final_test")
    checkpoint = None
    if args.evaluate_checkpoint:
        checkpoint = torch.load(args.evaluate_checkpoint, map_location="cpu", weights_only=True)
        saved = checkpoint["args"]
        for key in ("embed_dim", "dropout", "no_prior", "temperature", "psr_k", "rfi_k", "k_shot", "q_shot", "cost_fn", "cost_fp"):
            setattr(args, key, saved.get(key, getattr(args, key)))
    checks = preflight(Path.cwd(), args, final_test=args.final_test)
    if not checks["ready"]:
        ap.error("Data preflight failed: " + json.dumps(checks, ensure_ascii=False))

    set_seed(args.seed)
    out_dir = args.out_dir or (os.path.dirname(args.save_path) or ".")
    args.out_dir = out_dir
    ensure_dir(out_dir)
    ensure_dir(os.path.dirname(args.save_path))
    log_path = setup_run_logging(out_dir)

    device_name = args.device or ("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    device = torch.device(device_name)
    print(f"[LOG] full training log txt -> {log_path}")
    print(f"device: {device}")
    print("[DEMO2-PROTO] NGC6517 PSR/RFI prototypical few-shot classifier")
    print("[FSL] symmetric 2-way prototype evidence: support mean prototypes + cosine query matching")
    print(
        f"[EPISODE] support_rfi={args.k_shot}, support_psr={args.k_shot}, "
        f"query_rfi={args.q_shot}, query_psr={args.q_shot}"
    )
    print(
        f"[PROTO] embed_dim={args.embed_dim}, temperature={args.temperature}, "
        f"dropout={args.dropout}, psr_k={args.psr_k}, rfi_k={args.rfi_k}, "
        f"cost_fn={args.cost_fn}, cost_fp={args.cost_fp}"
    )
    print(
        f"[RECALL] weight_psr={args.weight_psr}, decision_threshold={args.decision_threshold} "
        "(lower threshold reduces PSR false negatives at possible FP cost)"
    )
    if args.auto_threshold:
        print(
            f"[THRESHOLD] auto validation search enabled: "
            f"range=[{args.threshold_min}, {args.threshold_max}], steps={args.threshold_steps}"
        )
    print(
        f"[PRIOR] enabled={not args.no_prior}, aux_dim=27, gate_strength=0.2, "
        "profile_prior=9, dm_prior=8, image_prior=10"
    )
    print("[EVIDENCE] profile + DM curve + subints/subbands images + split metadata + diagnostic aux")

    if checkpoint is not None:
        model = NGC6517Encoder(embed_dim=args.embed_dim, dropout=args.dropout, use_prior=not args.no_prior).to(device)
        model.load_state_dict(checkpoint["model"])
        threshold = float(checkpoint.get("decision_threshold", args.decision_threshold))
        test_metrics, predictions = final_test(args, model, device, threshold)
        summary = {"best": checkpoint["best"], "save_path": args.evaluate_checkpoint,
                   "final_test": test_metrics, "final_test_csv": predictions,
                   "best_decision_threshold": threshold, "selection_metric": "val_f1",
                   "mode": "frozen_checkpoint_evaluation"}
        write_summary(out_dir, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    train_ds = NGC6517FewShotDataset(args.train_csv)
    val_ds = NGC6517FewShotDataset(args.val_csv)
    train_by_class = train_ds.class_indices()

    print(f"[DATA] train_rfi={len(train_by_class[0])}, train_psr={len(train_by_class[1])}")
    val_by_class = val_ds.class_indices()
    print(f"[DATA] val_rfi={len(val_by_class[0])}, val_psr={len(val_by_class[1])}")
    eval_seed = int(args.eval_seed if args.eval_seed is not None else int(args.seed) + 10000)
    val_fixed_eval_episodes = int(args.val_fixed_eval_episodes or args.eval_episodes)
    val_random_eval_episodes = int(args.val_random_eval_episodes)
    val_fixed_bank = os.path.normpath(args.val_fixed_bank) if args.val_fixed_bank else ""
    if val_fixed_bank:
        val_fixed_specs = load_episode_specs_from_bank(
            val_fixed_bank,
            expected_episodes=val_fixed_eval_episodes,
            expected_k_shot=args.k_shot,
            expected_q_shot=args.q_shot,
        )
        val_fixed_source = val_fixed_bank
    else:
        val_fixed_specs = build_episode_specs(val_ds, val_fixed_eval_episodes, args.k_shot, args.q_shot, eval_seed)
        val_fixed_source = f"seed:{eval_seed}"
    val_random_specs = build_episode_specs(val_ds, val_random_eval_episodes, args.k_shot, args.q_shot, eval_seed + 2)
    fixed_weight = 1.0 - float(args.val_random_select_weight)
    print(
        f"[EVAL] VAL uses fixed+random: fixed={len(val_fixed_specs)}, "
        f"random={val_random_eval_episodes}, select=fixed_{fixed_weight:.2f}_random_{1.0 - fixed_weight:.2f}"
    )
    print(f"[BANK] VAL_FIXED source={val_fixed_source}")
    print("[EVAL] Held-out test is deferred until explicit final evaluation")

    model = NGC6517Encoder(embed_dim=args.embed_dim, dropout=args.dropout, use_prior=not args.no_prior).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = {"cost": float("inf"), "f1": -1.0, "acc": 0.0, "episode": -1}
    best_threshold = float(args.decision_threshold)
    history = []
    iterator = trange(1, int(args.episodes) + 1, desc="demo2 few-shot", disable=args.disable_tqdm)
    for episode in iterator:
        model.train()
        sb, sy, qb, qy = make_episode(train_ds, train_by_class, args.k_shot, args.q_shot)
        sb = move_batch(sb, device)
        qb = move_batch(qb, device)
        sy = sy.to(device)
        qy = qy.to(device)
        logits = prototypical_logits(
            model(sb),
            sy,
            model(qb),
            temperature=args.temperature,
            psr_k=args.psr_k,
            rfi_k=args.rfi_k,
        )
        class_weight = torch.tensor([1.0, float(args.weight_psr)], dtype=torch.float32, device=device)
        loss = F.cross_entropy(logits, qy, weight=class_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        iterator.set_postfix(loss=float(loss.item()))

        if episode % int(args.eval_every) == 0 or episode == int(args.episodes):
            val_fixed = evaluate(
                model,
                val_ds,
                val_fixed_eval_episodes,
                args.k_shot,
                args.q_shot,
                args.temperature,
                device,
                psr_k=args.psr_k,
                rfi_k=args.rfi_k,
                cost_fn=args.cost_fn,
                cost_fp=args.cost_fp,
                episode_specs=val_fixed_specs,
                decision_threshold=args.decision_threshold,
                return_scores=args.auto_threshold,
            )
            val_random = evaluate(
                model,
                val_ds,
                val_random_eval_episodes,
                args.k_shot,
                args.q_shot,
                args.temperature,
                device,
                psr_k=args.psr_k,
                rfi_k=args.rfi_k,
                cost_fn=args.cost_fn,
                cost_fp=args.cost_fp,
                episode_specs=val_random_specs,
                decision_threshold=args.decision_threshold,
                return_scores=args.auto_threshold,
            )
            selected_threshold = float(args.decision_threshold)
            if args.auto_threshold:
                val_fixed, val_random, val_select = tune_threshold(
                    val_fixed,
                    val_random,
                    fixed_weight=fixed_weight,
                    cost_fn=args.cost_fn,
                    cost_fp=args.cost_fp,
                    lo=args.threshold_min,
                    hi=args.threshold_max,
                    steps=args.threshold_steps,
                )
                selected_threshold = float(val_select["decision_threshold"])
                progress_write(iterator, f"[THRESHOLD] ep={episode} selected={selected_threshold:.3f}")
            else:
                val_select = mix_metrics(val_fixed, val_random, fixed_weight=fixed_weight)
                val_select["decision_threshold"] = selected_threshold
            val_fixed = strip_score_cache(val_fixed)
            val_random = strip_score_cache(val_random)
            val_select = strip_score_cache(val_select)
            progress_write(
                iterator,
                " | ".join(
                    [
                        f"[ep {episode:04d}] loss={float(loss.item()):.4f}",
                        fmt("VAL_FIXED", val_fixed),
                        fmt("VAL_RANDOM", val_random),
                        fmt("VAL_SELECT", val_select),
                    ]
                ),
            )
            row = {
                "episode": int(episode),
                "train_loss": float(loss.item()),
                **{f"val_fixed_{k}": v for k, v in val_fixed.items()},
                **{f"val_random_{k}": v for k, v in val_random.items()},
                **{f"val_select_{k}": v for k, v in val_select.items()},
            }
            history.append(row)
            if val_select["f1"] > best["f1"] or (np.isclose(val_select["f1"], best["f1"]) and val_select["cost"] < best["cost"]):
                best = {"cost": float(val_select["cost"]), "f1": float(val_select["f1"]), "acc": float(val_select["acc"]), "episode": int(episode)}
                best_threshold = float(selected_threshold)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "args": vars(args),
                        "episode": int(episode),
                        "best": best,
                        "decision_threshold": best_threshold,
                        "val": val_select,
                        "val_fixed": val_fixed,
                        "val_random": val_random,
                        "train_summary": train_ds.summary(),
                        "val_summary": val_ds.summary(),
                    },
                    args.save_path,
                )
                progress_write(iterator, f"[SAVE-BEST] {args.save_path}")
    hist_path = os.path.join(out_dir, "history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    ckpt = torch.load(args.save_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    best_threshold = float(ckpt.get("decision_threshold", best_threshold))
    test_m, test_csv = (None, None)
    if args.final_test:
        test_m, test_csv = final_test(args, model, device, best_threshold)
    summary = {
        "best": best, "save_path": args.save_path, "history": hist_path,
        "selection_metric": "val_f1", "validation_protocol": "weighted F1 on two reproducible episode banks",
        "final_test_csv": test_csv, "final_test": test_m,
        "best_decision_threshold": best_threshold,
    }
    write_summary(out_dir, summary)
    print("FINISHED")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
