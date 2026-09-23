#!/usr/bin/env python3
"""Day 12/13: unified launcher for the force/compliance baselines that share
a dataset and evaluation harness (proposal §6.1 table):

  B0  pretrained VLA, position output, fixed high stiffness (no force input)
  B2  pretrained VLA + force input,   position output   (ForceVLA-style)
  B3  pretrained VLA + force input,   HYBRID force-position output (Day 14,
                                       ForceVLA2/Force Policy style)
  B4  ACT/Bi-ACT FROM SCRATCH,        compliance output  (H4 control, Day 13)
  B5  pretrained VLA + force input,   compliance output  ("ours")

B0 is lerobot's unmodified SmolVLAPolicy (no code in this repo). B2 and B5
are compliance_vla.policy.compliance_policy.ComplianceSmolVLAPolicy, differing only in
config.use_compliance_head (see that module's docstring). B3 is
compliance_vla.policy.hybrid_policy.HybridSmolVLAPolicy -- same force-input machinery as
B2/B5, but log_k is decoded by a separate one-shot regression head instead
of being folded into the shared flow-matching target the way B5 does it
(see that module's docstring -- this is §6.1's "closest competing output
parameterization" row). B4 is compliance_vla.policy.bi_act_policy.BiActCompliancePolicy
-- lerobot's unmodified ACT model with no ImageNet-pretrained backbone and a
from-scratch language pathway, isolating whether B5's gain comes from
vision-language pretraining (H4) rather than from the compliance-output
architecture itself (that's what B2/B3 vs B5 already isolate). One script,
one dataset, one training loop for all five, so runs are identical in every
way this project controls except the thing each comparison is about --
exactly what "matched data and backbone" (§6.1) requires for B0/B2/B3/B5;
B4 necessarily differs in backbone by design (that's the H4 manipulation),
but shares B5's data, output space, and loss exactly.

*** This script does not launch the real Day-12 runs by itself. ***
Day 12's exit criterion is "9 runs in flight" (3 policies x 3 seeds) on a
GPU allocation confirmed back on Day 1 -- that's this project's call to make
about queue depth and timing, not something to trigger from here
automatically. Use `--smoke-test` to verify a policy/seed combination trains
without NaNs before committing GPU time to the real run; use
launch_day12_runs.sh (same directory) to fire off all 9 real runs once
you're ready -- see that script's header for exact usage, and the bottom of
this docstring for the one-command version.

Usage:
    # Correctness check only, a few real steps, ~1-2 min:
    python scripts/train_policy.py --policy b0 --seed 0 --smoke-test
    python scripts/train_policy.py --policy b2 --seed 0 --smoke-test
    python scripts/train_policy.py --policy b3 --seed 0 --smoke-test
    python scripts/train_policy.py --policy b4 --seed 0 --smoke-test
    python scripts/train_policy.py --policy b5 --seed 0 --smoke-test

    # One real run (foreground; run 9 of these, one per policy x seed):
    python scripts/train_policy.py --policy b5 --seed 0 --steps 30000

    # All 9 real runs, backgrounded, GPUs round-robin'd -- see that script:
    ./scripts/launch_day12_runs.sh
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
for _p in (SCRIPT_DIR, PROJECT_ROOT):  # SCRIPT_DIR for `import dataset_io`, PROJECT_ROOT for `import training.*`
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature  # noqa: E402
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE  # noqa: E402

import dataset_io as dio  # noqa: E402
from compliance_vla.policy.bi_act_policy import ENV_STATE_DIM, BiActComplianceConfig, build_bi_act_policy  # noqa: E402
from compliance_vla.policy.compliance_policy import B2Config, ComplianceSmolVLAConfig, ComplianceSmolVLAPolicy  # noqa: E402
from compliance_vla.policy.dataset import ComplianceWindowDataset, make_collate_fn  # noqa: E402
from compliance_vla.policy.hybrid_policy import HybridSmolVLAConfig, HybridSmolVLAPolicy  # noqa: E402

STATE_DIM = 20  # q(7) + qdot(7) + x_f(6), proposal §4.2 "(q, q_dot, x_f)"
IMAGE_SHAPE = (3, 224, 224)
TRAIN_SESSIONS = ["demo4"]  # session-level split, scripts/build_dataset_splits.py (Day 10) -- the
# frozen train/val/test = demo4/demo1/demo3 split every other script (tune_lambda.py,
# offline_stiffness_benchmark.py, fit_b1_oracle_stiffness.py) assumes when it reads "val"/"test".
# Stays the default so those comparisons keep meaning what they've always meant; use
# --all-sessions (or --train-sessions demo1 demo3 demo4 directly) to opt into pooling every
# session for training instead, e.g. for the language-grounding retraining ablation in
# language_grounding_issue_handoff.md, which asks to retrain on "the existing 92-episode
# dataset" rather than demo4's 32. Doing so means there is no held-out val/test left for that
# run -- fine for that ablation (it's scored by scripts/diagnose_language_grounding.py, not
# val/test loss), but don't also treat such a run as comparable to the frozen-split baselines.
ALL_SESSIONS = dio.SESSIONS  # ["demo1", "demo3", "demo4"] -- every session collected so far

POLICIES = ("b0", "b2", "b3", "b4", "b5")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def action_dim_for(policy):
    return 7 if policy in ("b0", "b2") else 13


def action_layout_for(policy):
    # b3 (hybrid) needs the full13 layout too, even though flow matching itself only ever
    # sees 7 of those dims -- the other 6 (log_k) are read as a direct regression *target*,
    # not corrupted with noise, so the ground-truth values (not just the mask) must reach
    # the model. See src/compliance_vla/policy/hybrid_policy.py's HybridSmolVLAPolicy.forward.
    return "position7" if policy in ("b0", "b2") else "full13"


def build_config(args):
    action_dim = action_dim_for(args.policy)
    input_features = {
        OBS_STATE: PolicyFeature(FeatureType.STATE, (STATE_DIM,)),
        "observation.images.scene_rgb": PolicyFeature(FeatureType.VISUAL, IMAGE_SHAPE),
        "observation.images.wrist_rgb": PolicyFeature(FeatureType.VISUAL, IMAGE_SHAPE),
    }
    output_features = {ACTION: PolicyFeature(FeatureType.ACTION, (action_dim,))}
    normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.IDENTITY,  # no dataset-wide stats yet, see Day 11's carried-forward gap
        "ACTION": NormalizationMode.IDENTITY,
    }

    if args.policy == "b4":
        # BiActComplianceConfig, not SmolVLAConfig, so it does not share the b0/b2/b5 `common` kwargs below
        # (no `load_vlm_weights` field -- B4 has no VLM at all, that's the point, see src/compliance_vla/policy/bi_act_policy.py).
        # Needs one extra input: environment_state, ACT's slot for the force+language token this project feeds it.
        b4_input_features = dict(input_features, **{OBS_ENV_STATE: PolicyFeature(FeatureType.ENV, (ENV_STATE_DIM,))})
        return BiActComplianceConfig(
            input_features=b4_input_features, output_features=output_features,
            normalization_mapping=normalization_mapping, device=args.device,
            chunk_size=32, n_action_steps=32, lam_log_k=args.lam,
            force_dropout_p=args.force_dropout_p, wrench_bias_range_n=args.wrench_bias_n,
        )

    common = dict(
        input_features=input_features, output_features=output_features,
        normalization_mapping=normalization_mapping, device=args.device,
        load_vlm_weights=not args.no_pretrained,  # b0/b2/b5 are all the *pretrained*-VLA variant
        chunk_size=32, n_action_steps=32,
        # SmolVLAConfig default is train_expert_only=True, which freezes the ENTIRE VLM (vision_model
        # AND text_model -- see lerobot's SmolVLMWithExpertModel.set_requires_grad), leaving only the
        # flow-matching action expert trainable. --unfreeze-lm flips this off so the text tower (the
        # part that would actually have to learn to tell "red" from "blue") gets gradients too. That
        # alone (b0_seed0_allsessions_unfreezelm, 2026-08-26) did not clear chance-level colour
        # discrimination -- see language_grounding_issue_handoff.md's follow-up. One live hypothesis:
        # freeze_vision_encoder was left at its default True in that run, so the text/cross-attention
        # layers were unfrozen but had nothing better to attend to -- the frozen, generically-pretrained
        # SigLIP tower may not represent four small same-shape coloured marks on a whiteboard distinctly
        # enough in the first place. --unfreeze-vision tests that directly.
        train_expert_only=not args.unfreeze_lm,
        freeze_vision_encoder=not args.unfreeze_vision,
    )

    if args.policy == "b0":
        return SmolVLAConfig(**common)
    elif args.policy == "b2":
        return B2Config(**common, force_dropout_p=args.force_dropout_p, wrench_bias_range_n=args.wrench_bias_n)
    elif args.policy == "b3":
        return HybridSmolVLAConfig(
            **common, lam_log_k=args.lam,
            force_dropout_p=args.force_dropout_p, wrench_bias_range_n=args.wrench_bias_n,
        )
    elif args.policy == "b5":
        return ComplianceSmolVLAConfig(
            **common, lam_log_k=args.lam,
            force_dropout_p=args.force_dropout_p, wrench_bias_range_n=args.wrench_bias_n,
        )
    raise ValueError(args.policy)


def build_policy(args, config):
    if args.policy == "b0":
        return SmolVLAPolicy(config)
    if args.policy == "b3":
        return HybridSmolVLAPolicy(config)
    if args.policy == "b4":
        return build_bi_act_policy(config)  # resolves the real tokenizer + vocab size, see src/compliance_vla/policy/bi_act_policy.py
    return ComplianceSmolVLAPolicy(config)  # b2, b5


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True, choices=POLICIES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lam", type=float, default=1.0, help="compliance (log_k) loss weight, §4.2 -- b4/b5 only")
    p.add_argument("--force-dropout-p", type=float, default=0.15, help="b2/b4/b5 only")
    p.add_argument("--wrench-bias-n", type=float, default=1.0, help="b2/b4/b5 only")
    p.add_argument("--lr", type=float, default=None, help="override config default optimizer_lr")
    p.add_argument("--unfreeze-lm", action="store_true",
                    help="set config.train_expert_only=False so the SmolVLM text tower is trained too, not just "
                         "the action expert -- the language-grounding retraining ablation in "
                         "language_grounding_issue_handoff.md. Vision encoder stays frozen unless --unfreeze-vision "
                         "is also given. No-op for b4 (no VLM). Pair with a small --lr (e.g. 1e-5) -- this is a "
                         "single global LR applied to every trainable param, unfrozen backbone included, so the "
                         "default 1e-4 action-expert LR is too large for a pretrained tower on 92 episodes.")
    p.add_argument("--unfreeze-vision", action="store_true",
                    help="set config.freeze_vision_encoder=False so the SigLIP vision tower is trained too -- "
                         "try this if --unfreeze-lm alone doesn't clear chance-level colour discrimination on "
                         "scripts/diagnose_language_grounding.py, since the failure may be that the frozen "
                         "vision encoder doesn't represent the coloured marks distinctly enough for even a "
                         "trainable text tower to attend to. No-op for b4 (no VLM). Combine with --unfreeze-lm "
                         "for a full unfreeze, or use alone to isolate the vision side.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-pretrained", action="store_true", help="debug only: random-init VLM instead of the pretrained backbone every baseline uses")
    p.add_argument("--smoke-test", action="store_true", help="a few real steps, checked for NaNs -- not a real training run")
    p.add_argument("--smoke-steps", type=int, default=8)
    p.add_argument("--smoke-episodes", type=int, default=6, help="cap episodes loaded for --smoke-test, for turnaround time")
    p.add_argument("--overfit-n-per-referent", type=int, default=None,
                    help="colour-balanced overfit sanity check (language_grounding_issue_handoff.md, "
                         "2026-08-28 update): cap the dataset to this many episodes per colour referent per "
                         "session, picked the same way scripts/diagnose_language_grounding.py's "
                         "--n-per-referent does (its pick_balanced_episodes helper), instead of a session's "
                         "full episode set. The point of this check is 'can the model even memorize "
                         "colour->position on a tiny, clean, balanced set' before blaming dataset size for "
                         "chance-level colour discrimination -- an unbalanced subset wouldn't isolate that, "
                         "so this does NOT reuse --max-episodes' first-N-seen order. Unlike --smoke-test, "
                         "this does not cap --steps or skip checkpointing -- it's meant to actually run to "
                         "convergence and be evaluated with scripts/diagnose_language_grounding.py after.")
    p.add_argument("--max-episodes", type=int, default=None,
                    help="cap dataset to at most this many episodes total (first-seen order per session, "
                         "not colour-balanced) without --smoke-test's few-step/no-checkpoint behavior. "
                         "Ignored if --overfit-n-per-referent is set.")
    p.add_argument("--referents", nargs="+", default=None, choices=("red", "blue", "green", "black"),
                    help="2026-09-01 2-colour descope (language_grounding_issue_handoff.md): restrict "
                         "training to episodes whose task text names one of these colours, e.g. "
                         "'--referents red blue'. Every 4-way colour discrimination attempt so far "
                         "(frozen, unfreeze-lm, unfreeze-lm+vision, a full-unfreeze overfit on the "
                         "model's own training episodes, LoRA through step 10000) has landed at or "
                         "below the 25%% chance floor with a persistent 1-2-colour collapse -- this "
                         "raises chance to 50%% for a real go/no-go read on whether the existing weak "
                         "signal (see the positional-prior check, 2026-08-28) clears a 2-way bar. "
                         "Composes with --train-sessions/--all-sessions (applied after session pooling, "
                         "before --max-episodes/--overfit-n-per-referent, which then act on the "
                         "restricted set). Default: no filtering (all 4 colours).")
    p.add_argument("--dataset-root", default=None,
                    help="root every --train-sessions entry is loaded from, overriding dataset_io's default "
                         "dataset/demo{1,3,4} root -- e.g. dio.TWO_COLOR_DATASET_ROOT to train on "
                         "data_two_color sessions instead. Also used by --overfit-n-per-referent's episode "
                         "picker. Applies uniformly to every session in --train-sessions; there's no "
                         "current flag for mixing sessions from more than one root in a single run.")
    p.add_argument("--train-sessions", nargs="+", default=None,
                    help=f"sessions to pool for training (default: {TRAIN_SESSIONS}, the frozen "
                         "train split -- see the TRAIN_SESSIONS comment above). Ignored if --all-sessions is set.")
    p.add_argument("--all-sessions", action="store_true",
                    help=f"train on every collected session ({ALL_SESSIONS}) instead of just the frozen "
                         "train split -- e.g. for the language-grounding retraining ablation, which needs "
                         "all 92 episodes, not demo4's 32. Leaves no held-out val/test for this run.")
    p.add_argument("--lora", action="store_true",
                    help="wrap the attention projection layers that actually see language (SmolVLM's own "
                         "text_model self-attention, plus lm_expert -- the flow-matching expert's attention, "
                         "whose k_proj/v_proj read the shared image+language+state prefix when "
                         "attention_mode='cross_attn') with LoRA adapters (github.com/huggingface/peft) instead "
                         "of fully unfreezing them. language_grounding_issue_handoff.md's full-unfreeze "
                         "experiments (--unfreeze-lm/--unfreeze-vision) showed signs of catastrophic "
                         "forgetting on this project's ~92-episode dataset (a from-scratch overfit run scored "
                         "BELOW chance) -- LoRA trains a couple orders of magnitude fewer parameters, which is "
                         "far less prone to that on a dataset this small. Overrides --unfreeze-lm/"
                         "--unfreeze-vision's effect (peft freezes every non-adapter parameter regardless of "
                         "their requires_grad state), so those flags become no-ops when this is set. No-op for "
                         "b4 (no VLM). The saved checkpoint stores raw adapter weights (not directly loadable "
                         "by serve_policy.py) plus a 'lora_config' entry recording exactly how to reconstruct "
                         "the adapter structure -- run scripts/merge_lora_checkpoint.py on it first to fold the "
                         "adapters into plain weights for serving.")
    p.add_argument("--lora-r", type=int, default=8, help="LoRA rank -- only used with --lora")
    p.add_argument("--lora-alpha", type=int, default=16, help="LoRA scaling factor -- only used with --lora")
    p.add_argument("--lora-dropout", type=float, default=0.05, help="dropout on LoRA's low-rank path -- only used with --lora")
    p.add_argument("--lora-include-vision", action="store_true",
                    help="also adapt the SigLIP vision encoder's attention layers (closer in spirit to "
                         "--unfreeze-vision, but low-rank) -- default off, matching this doc's recommendation "
                         "to try the language-side pathway alone first. Only used with --lora.")
    p.add_argument("--no-manner-calibration", action="store_true",
                    help="disable the per-operator log_k calibration (compliance_vla.policy.labels.calibrate_log_k, "
                         "fit by scripts/fit_manner_force_calibration.py) that's otherwise applied by "
                         "default. That correction exists to reconcile manner grounding (firmly/normally) "
                         "*across* operators -- pass this when --train-sessions is single-operator (e.g. "
                         "the 2026-09-02 Operator-A-only data_two_color run), where there's no cross-"
                         "operator inconsistency to correct and applying it would just nudge that "
                         "operator's own values toward an equal-split reference that includes an operator "
                         "not even present in this training set.")
    p.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "reports"))
    p.add_argument("--checkpoint-dir", default=None, help="if set, save a state_dict here every --checkpoint-every steps")
    p.add_argument("--checkpoint-every", type=int, default=5000)
    p.add_argument("--wandb", action="store_true", help="log metrics to Weights & Biases")
    p.add_argument("--wandb-project", default="compliance-vla-icra2027")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-run-name", default=None, help="default: <policy>_seed<seed>[_smoke]")
    args = p.parse_args()

    if (args.unfreeze_lm or args.unfreeze_vision) and args.policy == "b4":
        print("[train_policy] WARNING: --unfreeze-lm/--unfreeze-vision are no-ops for b4 (BiActCompliancePolicy has no VLM)", file=sys.stderr)

    if args.all_sessions:
        if args.train_sessions is not None:
            print("[train_policy] --all-sessions overrides --train-sessions", file=sys.stderr)
        args.train_sessions = ALL_SESSIONS
    elif args.train_sessions is None:
        args.train_sessions = TRAIN_SESSIONS
    if set(args.train_sessions) != set(TRAIN_SESSIONS):
        print(f"[train_policy] WARNING: training on {args.train_sessions}, not the frozen train "
              f"split {TRAIN_SESSIONS} -- val/test sessions may be included, so this run's val/test "
              "loss (if any) is not comparable to the frozen-split baselines.", file=sys.stderr)

    set_seed(args.seed)

    print(f"[train_policy] policy={args.policy} device={args.device} seed={args.seed} smoke_test={args.smoke_test} "
          f"unfreeze_lm={args.unfreeze_lm} unfreeze_vision={args.unfreeze_vision}")
    config = build_config(args)
    if args.lr is not None:
        config.optimizer_lr = args.lr

    print(f"[train_policy] building policy {args.policy}...")
    t0 = time.time()
    policy = build_policy(args, config)
    policy.to(args.device)
    policy.train()

    lora_target_modules = None
    if args.lora and args.policy == "b4":
        print("[train_policy] WARNING: --lora is a no-op for b4 (BiActCompliancePolicy has no VLM)", file=sys.stderr)
    elif args.lora:
        # Lazy import: peft is only needed for this flag, not every training run (same
        # pattern as diagnose_language_grounding.py's json_numpy/requests, and this script's
        # own --overfit-n-per-referent import below).
        from peft import LoraConfig, get_peft_model

        lora_target_modules = [
            name for name, module in policy.named_modules()
            if isinstance(module, torch.nn.Linear)
            and name.endswith(("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"))
            and ("lm_expert" in name or "text_model" in name
                 or (args.lora_include_vision and "vision_model" in name))
        ]
        if not lora_target_modules:
            raise ValueError(
                f"--lora found no matching attention projection layers on policy {args.policy!r} -- "
                "expected 'lm_expert'/'text_model' (and 'vision_model' if --lora-include-vision) "
                "submodules with self_attn.{q,k,v,o}_proj Linear layers; the model architecture may "
                "have changed since this filter was written (scripts/train_policy.py's --lora block)"
            )
        lora_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=lora_target_modules, bias="none",
        )
        # get_peft_model mutates `policy`'s own submodules in place (targeted nn.Linear layers
        # become peft's lora.Linear wrapper: original weight frozen, small trainable A/B added)
        # and freezes every other parameter -- the returned PeftModel wrapper is only used here
        # for its logging method; `policy` itself keeps working exactly as before (forward(),
        # get_optim_params(), .model.vlm_with_expert.processor.tokenizer, state_dict(), ...)
        # since it's the same underlying object, just with some Linear submodules swapped out.
        # Verified empirically (this session) that this pattern round-trips correctly through a
        # real forward/backward/optimizer.step() -- see language_grounding_issue_handoff.md.
        get_peft_model(policy, lora_config).print_trainable_parameters()

    n_params = sum(p_.numel() for p_ in policy.parameters())
    n_trainable = sum(p_.numel() for p_ in policy.parameters() if p_.requires_grad)
    print(f"[train_policy] policy built in {time.time() - t0:.1f}s -- {n_params / 1e6:.1f}M params, {n_trainable / 1e6:.1f}M trainable")

    print(f"[train_policy] loading dataset (sessions={args.train_sessions}, "
          f"dataset_root={args.dataset_root or 'default (dataset_io.DATASET_ROOT)'}, "
          f"manner_calibration={not args.no_manner_calibration})...")
    t0 = time.time()
    # --dataset-root applies uniformly to every --train-sessions entry -- fine for a
    # single-root run (e.g. all-data_two_color); a run mixing sessions from more than one
    # root would need ComplianceWindowDataset's dataset_roots dict passed directly, not yet
    # exposed as its own flag since nothing has needed it.
    dataset_roots = {s: args.dataset_root for s in args.train_sessions} if args.dataset_root else None
    dataset = ComplianceWindowDataset(
        args.train_sessions, chunk_size=config.chunk_size,
        force_history_len=getattr(config, "force_history_len", 20),
        force_history_window_sec=getattr(config, "force_history_window_sec", 0.5),
        dataset_roots=dataset_roots,
        apply_manner_calibration=not args.no_manner_calibration,
    )
    if args.referents is not None:
        # Lazy import, same reasoning as --overfit-n-per-referent's below: keep
        # diagnose_language_grounding.py's torch/lerobot-free property intact for callers
        # that only need referent_from_task.
        from diagnose_language_grounding import referent_from_task

        keep_referents = set(args.referents)
        before = len(dataset.index)
        kept_episodes = {
            key for key, task in dataset._task_lookup.items()
            if referent_from_task(task) in keep_referents
        }
        dataset.index = [entry for entry in dataset.index if entry[:2] in kept_episodes]
        print(f"[train_policy] --referents {sorted(keep_referents)}: kept {len(kept_episodes)} episodes, "
              f"{len(dataset.index)}/{before} windows")
    if args.overfit_n_per_referent is not None:
        if args.max_episodes is not None:
            print("[train_policy] --overfit-n-per-referent overrides --max-episodes", file=sys.stderr)
        # Lazy import: diagnose_language_grounding.py is deliberately torch/lerobot-free (see its own
        # docstring) so it can run standalone off a served checkpoint; importing it here only when this
        # flag is used avoids adding json_numpy/requests as a hard dependency of every training run.
        from diagnose_language_grounding import _resolve_session_dir, pick_balanced_episodes

        overfit_referents = tuple(args.referents) if args.referents else ("red", "blue", "green", "black")
        keep = set()
        for session in args.train_sessions:
            session_path = _resolve_session_dir(session, args.dataset_root)
            keep.update((session, ep) for ep in pick_balanced_episodes(
                session_path, args.overfit_n_per_referent, referents=overfit_referents
            ))
        dataset.index = [entry for entry in dataset.index if entry[:2] in keep]
        print(f"[train_policy] overfit subset: {len(keep)} episodes {sorted(keep)}, {len(dataset.index)} windows")
    elif args.smoke_test or args.max_episodes is not None:
        cap = args.smoke_episodes if args.smoke_test else args.max_episodes
        episodes_seen, capped_index = [], []
        for entry in dataset.index:
            key = entry[:2]
            if key not in episodes_seen:
                if len(episodes_seen) >= cap:
                    continue
                episodes_seen.append(key)
            capped_index.append(entry)
        dataset.index = capped_index
    print(f"[train_policy] dataset ready in {time.time() - t0:.1f}s -- {len(dataset._episode_cache)} episodes "
          f"({dataset.n_episodes_failed} failed contact-frame fit), {len(dataset)} windows")

    # b4 (BiActCompliancePolicy) exposes its own tokenizer directly (src/compliance_vla/policy/bi_act_policy.py); b0/b2/b5
    # only have one buried in the SmolVLA VLM stack. tokenizer_max_length/pad_language_to are SmolVLAConfig
    # fields with no ACTConfig equivalent -- b4 falls back to make_collate_fn's own defaults.
    tokenizer = getattr(policy, "tokenizer", None) or policy.model.vlm_with_expert.processor.tokenizer
    collate_fn = make_collate_fn(
        tokenizer, getattr(config, "tokenizer_max_length", 48), getattr(config, "pad_language_to", "longest"),
        action_layout=action_layout_for(args.policy),
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
        num_workers=0, drop_last=True,
    )

    # get_optim_params() (lerobot's PreTrainedPolicy default) returns ALL parameters, not just
    # trainable ones -- harmless for correctness (frozen params never get a gradient, so plain
    # SGD/AdamW never updates them) but wasteful for LoRA specifically, where the whole point is
    # a couple orders of magnitude fewer trainable params: AdamW allocates optimizer state
    # (momentum + variance buffers, ~2x param size in fp32) per param regardless of whether it's
    # ever updated, so handing it the frozen 99.7% of the model would burn most of the memory
    # LoRA is meant to save.
    optim_params = policy.get_optim_params()
    if args.lora:
        optim_params = [p_ for p_ in optim_params if p_.requires_grad]
    optimizer = config.get_optimizer_preset().build(optim_params)

    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    wandb_run = None
    if args.wandb:
        import wandb

        run_name = args.wandb_run_name or f"{args.policy}_seed{args.seed}" + ("_smoke" if args.smoke_test else "")
        wandb_run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity, name=run_name,
            config={
                "policy": args.policy, "seed": args.seed, "steps": args.steps,
                "batch_size": args.batch_size, "lam_log_k": args.lam,
                "force_dropout_p": args.force_dropout_p, "wrench_bias_range_n": args.wrench_bias_n,
                "lr": args.lr, "device": args.device, "smoke_test": args.smoke_test,
                "load_vlm_weights": (not args.no_pretrained) if args.policy != "b4" else None,
                "train_sessions": args.train_sessions,
                "unfreeze_lm": args.unfreeze_lm, "unfreeze_vision": args.unfreeze_vision,
                "lora": args.lora, "lora_r": args.lora_r if args.lora else None,
                "lora_alpha": args.lora_alpha if args.lora else None,
                "lora_n_target_modules": len(lora_target_modules) if lora_target_modules else None,
            },
        )

    steps = args.smoke_steps if args.smoke_test else args.steps
    log = []
    step = 0
    t_train_start = time.time()
    while step < steps:
        for batch in loader:
            if step >= steps:
                break
            batch = {k: v.to(args.device) for k, v in batch.items()}

            loss, parts = policy.forward(batch)
            optimizer.zero_grad()
            loss.backward()
            # ACTConfig has no optimizer_grad_clip_norm field (only SmolVLAConfig does) -- b4 falls back to
            # SmolVLAConfig's own default value rather than skipping clipping outright.
            grad_clip_norm = getattr(config, "optimizer_grad_clip_norm", 10.0)
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
            optimizer.step()

            finite = all(v == v and abs(v) != float("inf") for v in parts.values())
            row = {"step": step, "grad_norm": float(grad_norm), **parts}
            log.append(row)
            if wandb_run is not None:
                wandb_run.log(row, step=step)
            flag = "" if finite else "  *** NON-FINITE LOSS ***"
            detail = " | ".join(f"{k} {v:.4f}" for k, v in parts.items() if k != "loss")
            print(f"[train_policy] {args.policy} seed{args.seed} step {step:5d} | loss {parts['loss']:.4f} "
                  f"| {detail} | grad_norm {grad_norm:.3f}{flag}\033[K", end="\r", flush=True)
            if not finite:
                print()
                if wandb_run is not None:
                    wandb_run.finish(exit_code=1)
                raise FloatingPointError(f"non-finite loss at step {step}: {parts}")
            step += 1

            if args.checkpoint_dir and not args.smoke_test and step % args.checkpoint_every == 0:
                ckpt_path = os.path.join(args.checkpoint_dir, f"{args.policy}_seed{args.seed}_step{step}.pt")
                ckpt = {"step": step, "config": config, "model_state_dict": policy.state_dict()}
                if args.lora:
                    # Needed to reconstruct the exact same LoraConfig/target_modules before
                    # load_state_dict -- the raw state_dict alone has the adapter weights but
                    # nothing that says "wrap these Linear layers with LoRA before loading them
                    # in". scripts/merge_lora_checkpoint.py reads this to fold the adapters into
                    # plain weights, producing a checkpoint serve_policy.py can load unmodified.
                    ckpt["lora_config"] = {
                        "r": args.lora_r, "lora_alpha": args.lora_alpha, "lora_dropout": args.lora_dropout,
                        "target_modules": lora_target_modules,
                    }
                torch.save(ckpt, ckpt_path)
                print(f"\n[train_policy] checkpoint -> {ckpt_path}")

    elapsed = time.time() - t_train_start
    print(f"\n[train_policy] {step} steps in {elapsed:.1f}s ({elapsed / max(step, 1):.2f}s/step)")

    os.makedirs(args.out_dir, exist_ok=True)
    tag = "smoke" if args.smoke_test else "full"
    out_path = os.path.join(args.out_dir, f"{args.policy}_seed{args.seed}_{tag}_train_log.json")
    with open(out_path, "w") as f:
        json.dump({
            "policy": args.policy, "seed": args.seed, "smoke_test": args.smoke_test, "steps": step,
            "lam_log_k": args.lam if args.policy in ("b4", "b5") else None,
            "force_dropout_p": args.force_dropout_p if args.policy != "b0" else None,
            "wrench_bias_range_n": args.wrench_bias_n if args.policy != "b0" else None,
            # b4 has no VLM at all (that's the H4 manipulation) -- load_vlm_weights is meaningless for it,
            # reported as None rather than a misleading True/False derived from an unused --no-pretrained flag.
            "load_vlm_weights": (not args.no_pretrained) if args.policy != "b4" else None,
            "n_params": n_params, "n_trainable_params": n_trainable,
            "elapsed_sec": elapsed, "log": log,
        }, f, indent=2)
    print(f"[train_policy] wrote {out_path}")

    all_finite = all(v == v and abs(v) != float("inf") for row in log for v in row.values())
    print(f"[train_policy] {'PASS' if all_finite else 'FAIL'}: all {len(log)} steps finite")

    if wandb_run is not None:
        wandb_run.summary["elapsed_sec"] = elapsed
        wandb_run.summary["all_finite"] = all_finite
        wandb_run.finish()


if __name__ == "__main__":
    main()
