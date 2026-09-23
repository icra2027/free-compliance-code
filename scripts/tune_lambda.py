#!/usr/bin/env python3
"""Day 12: tune lambda (the compliance/log_k loss weight, proposal §4.2) on
the validation split.

For each candidate lambda: train B5 from scratch for --tune-steps steps on
the train session (demo4), then evaluate (no grad, policy.eval()) on the val
session (demo1 -- Day 10's frozen session-level split) and record the mean
per-head validation loss. `--tune-steps` is deliberately much smaller than a
real B5 run's `--steps` (default 30000) -- this is a lambda *ranking* pass
across candidates, not a final trained checkpoint; whichever lambda wins
here is what Day 12's real, full B5 x 3-seed runs should be launched with
via train_policy.py / launch_day12_runs.sh, not this script's own
short-trained weights.

Selection rule (a documented choice, not a proposal-specified formula, same
spirit as evaluate_gate1.py's (ii)/(iii) or offline_stiffness_benchmark.py's
M8 task definition): report every candidate's val loss_x_eq and loss_log_k
separately, and pick the lambda minimizing their *unweighted* sum on val.
lambda only reweights the *training* objective; picking by a lambda-weighted
validation number would trivially favor lambda -> 0 (it zeroes out the term
being minimized over), so the tie-break has to use something lambda doesn't
appear in.

*** Does not run anything on import or module load -- this script only
trains when invoked, and each invocation is one real (if short) training
run per lambda candidate. Nothing in this repo calls it automatically. ***

Usage:
    python scripts/tune_lambda.py --lambdas 0.1 0.3 1.0 3.0 10.0 --tune-steps 2000
    python scripts/tune_lambda.py --lambdas 0.1 1.0 10.0 --tune-steps 500 --smoke-episodes 6  # quick check
"""

import argparse
import json
import os
import sys
import time

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402

from compliance_vla.policy.compliance_policy import ComplianceSmolVLAConfig, ComplianceSmolVLAPolicy  # noqa: E402
from compliance_vla.policy.dataset import ComplianceWindowDataset, make_collate_fn  # noqa: E402
from scripts.train_policy import STATE_DIM, IMAGE_SHAPE, set_seed  # noqa: E402

TRAIN_SESSION = "demo4"
VAL_SESSION = "demo1"  # Day 10 session-level split: train=demo4, val=demo1, test=demo3


def build_b5_config(args, lam):
    input_features = {
        OBS_STATE: PolicyFeature(FeatureType.STATE, (STATE_DIM,)),
        "observation.images.scene_rgb": PolicyFeature(FeatureType.VISUAL, IMAGE_SHAPE),
        "observation.images.wrist_rgb": PolicyFeature(FeatureType.VISUAL, IMAGE_SHAPE),
    }
    output_features = {ACTION: PolicyFeature(FeatureType.ACTION, (13,))}
    return ComplianceSmolVLAConfig(
        input_features=input_features, output_features=output_features,
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY, "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        },
        device=args.device, load_vlm_weights=not args.no_pretrained,
        chunk_size=32, n_action_steps=32, lam_log_k=lam,
        force_dropout_p=args.force_dropout_p, wrench_bias_range_n=args.wrench_bias_n,
    )


def cap_episodes(dataset, n_episodes):
    if n_episodes is None:
        return
    episodes_seen, capped_index = [], []
    for entry in dataset.index:
        key = entry[:2]
        if key not in episodes_seen:
            if len(episodes_seen) >= n_episodes:
                continue
            episodes_seen.append(key)
        capped_index.append(entry)
    dataset.index = capped_index


def train_and_validate_one_lambda(args, lam):
    set_seed(args.seed)
    config = build_b5_config(args, lam)
    policy = ComplianceSmolVLAPolicy(config)
    policy.to(args.device)

    train_ds = ComplianceWindowDataset(
        [TRAIN_SESSION], chunk_size=config.chunk_size,
        force_history_len=config.force_history_len, force_history_window_sec=config.force_history_window_sec,
    )
    val_ds = ComplianceWindowDataset(
        [VAL_SESSION], chunk_size=config.chunk_size,
        force_history_len=config.force_history_len, force_history_window_sec=config.force_history_window_sec,
    )
    cap_episodes(train_ds, args.smoke_episodes)
    cap_episodes(val_ds, args.smoke_episodes)

    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    collate_fn = make_collate_fn(tokenizer, config.tokenizer_max_length, config.pad_language_to, action_layout="full13")
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)

    optimizer = config.get_optimizer_preset().build(policy.get_optim_params())

    policy.train()
    step = 0
    while step < args.tune_steps:
        for batch in train_loader:
            if step >= args.tune_steps:
                break
            batch = {k: v.to(args.device) for k, v in batch.items()}
            loss, parts = policy.forward(batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.optimizer_grad_clip_norm)
            optimizer.step()
            if not all(v == v for v in parts.values()):
                raise FloatingPointError(f"non-finite loss at lambda={lam} step {step}: {parts}")
            step += 1

    policy.eval()
    val_parts_sum = {"loss_x_eq": 0.0, "loss_log_k": 0.0, "loss_gripper": 0.0, "loss": 0.0}
    n_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            _, parts = policy.forward(batch)
            for k in val_parts_sum:
                val_parts_sum[k] += parts[k]
            n_batches += 1
    val_mean = {k: v / max(n_batches, 1) for k, v in val_parts_sum.items()}

    return {
        "lambda": lam, "tune_steps": step, "n_val_batches": n_batches,
        "val_loss_x_eq": val_mean["loss_x_eq"], "val_loss_log_k": val_mean["loss_log_k"],
        "val_loss_weighted_total": val_mean["loss"],
        "val_unweighted_sum": val_mean["loss_x_eq"] + val_mean["loss_log_k"],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lambdas", type=float, nargs="+", default=[0.1, 0.3, 1.0, 3.0, 10.0])
    p.add_argument("--tune-steps", type=int, default=2000, help="per-lambda training steps before val eval (not a full B5 run)")
    p.add_argument("--seed", type=int, default=0, help="held fixed across lambda candidates for a fair comparison")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--force-dropout-p", type=float, default=0.15)
    p.add_argument("--wrench-bias-n", type=float, default=1.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--smoke-episodes", type=int, default=None, help="cap train/val episodes for a quick check; omit for the real sweep")
    p.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "reports"))
    args = p.parse_args()

    print(f"[tune_lambda] {len(args.lambdas)} candidates: {args.lambdas}, {args.tune_steps} steps each, "
          f"seed={args.seed} (fixed across candidates), val={VAL_SESSION}")

    results = []
    for lam in args.lambdas:
        t0 = time.time()
        print(f"[tune_lambda] === lambda={lam} ===")
        r = train_and_validate_one_lambda(args, lam)
        r["elapsed_sec"] = time.time() - t0
        results.append(r)
        print(f"[tune_lambda] lambda={lam}: val_loss_x_eq={r['val_loss_x_eq']:.4f} "
              f"val_loss_log_k={r['val_loss_log_k']:.4f} unweighted_sum={r['val_unweighted_sum']:.4f} "
              f"({r['elapsed_sec']:.1f}s)")

    best = min(results, key=lambda r: r["val_unweighted_sum"])
    print(f"\n[tune_lambda] best by unweighted val (loss_x_eq + loss_log_k): lambda={best['lambda']}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "lambda_tuning.json")
    with open(out_path, "w") as f:
        json.dump({
            "candidates": args.lambdas, "tune_steps": args.tune_steps, "seed": args.seed,
            "selection_rule": "min val_loss_x_eq + val_loss_log_k (unweighted)",
            "best_lambda": best["lambda"], "results": results,
        }, f, indent=2)
    print(f"[tune_lambda] wrote {out_path}")
    print(f"[tune_lambda] use the winner for the real runs: "
          f"./scripts/launch_day12_runs.sh --lam {best['lambda']}   (only affects b5; b0/b2 ignore --lam)")


if __name__ == "__main__":
    main()
