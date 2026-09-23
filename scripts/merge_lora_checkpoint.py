#!/usr/bin/env python3
"""Fold a --lora checkpoint's adapter weights into the base model's own weights, producing
a checkpoint scripts/serve_policy.py can load completely unmodified.

Why this is needed: a checkpoint saved during a `train_policy.py --lora` run has its LoRA
adapters (small lora_A/lora_B matrices) as extra keys in `model_state_dict`, alongside the
frozen base weights. Loading that directly into a freshly-built (non-LoRA) policy -- what
serve_policy.py does -- fails, since the fresh policy's module tree doesn't have those
lora_A/lora_B submodules for the keys to land in. This script rebuilds the exact same LoRA
adapter structure (from the checkpoint's own saved `lora_config`, not re-typed by hand),
loads the raw state dict into that reconstructed structure, then calls peft's
`merge_and_unload()` to bake each adapter's low-rank update directly into its target Linear
layer's weight and remove the adapter wrapper entirely -- what's left is a plain
SmolVLAPolicy state dict, structurally identical to a non-LoRA checkpoint.

Usage:
    python scripts/merge_lora_checkpoint.py \\
        checkpoints/b0_seed0_lora/b0_seed0_step30000.pt \\
        checkpoints/b0_seed0_lora/b0_seed0_step30000_merged.pt

Then serve the merged checkpoint exactly like any other:
    python scripts/serve_policy.py --checkpoint checkpoints/b0_seed0_lora/b0_seed0_step30000_merged.pt
"""
import argparse
import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
for _p in (SCRIPT_DIR, PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402

from compliance_vla.policy.compliance_policy import ComplianceSmolVLAConfig, ComplianceSmolVLAPolicy  # noqa: E402
from compliance_vla.policy.hybrid_policy import HybridSmolVLAConfig, HybridSmolVLAPolicy  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="a checkpoint written by `train_policy.py --lora`")
    p.add_argument("out_path", help="where to write the merged, serve_policy.py-ready checkpoint")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "lora_config" not in ckpt:
        raise ValueError(
            f"{args.checkpoint} has no 'lora_config' entry -- this doesn't look like a checkpoint "
            "written by `train_policy.py --lora` (nothing to merge; if it wasn't trained with "
            "--lora it's already directly loadable by serve_policy.py as-is)"
        )
    lora_cfg = ckpt["lora_config"]
    config = ckpt["config"]
    config.load_vlm_weights = False  # weights come from the checkpoint's own state_dict below

    # Same policy_cls resolution as serve_policy.py's load_policy -- HybridSmolVLAConfig must
    # be checked first since it subclasses ComplianceSmolVLAConfig.
    if isinstance(config, HybridSmolVLAConfig):
        policy_cls = HybridSmolVLAPolicy
    elif isinstance(config, ComplianceSmolVLAConfig):
        policy_cls = ComplianceSmolVLAPolicy
    else:
        policy_cls = SmolVLAPolicy
    policy = policy_cls(config)

    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=lora_cfg["r"], lora_alpha=lora_cfg["lora_alpha"], lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"], bias="none",
    )
    peft_model = get_peft_model(policy, lora_config)
    # Load into `policy` (the original, pre-wrap reference), not `peft_model` -- get_peft_model
    # mutates policy's own submodules in place and peft_model is just an outer wrapper around
    # it, but peft_model's own state_dict() keys are prefixed with "base_model.model." while
    # `policy.state_dict()` (what train_policy.py actually saved, since it keeps using the
    # original `policy` reference rather than the get_peft_model() return value) is not.
    policy.load_state_dict(ckpt["model_state_dict"])
    print(f"[merge_lora_checkpoint] loaded {args.checkpoint} -- {policy_cls.__name__}, step {ckpt.get('step')}, "
          f"{len(lora_cfg['target_modules'])} LoRA target modules (r={lora_cfg['r']}, alpha={lora_cfg['lora_alpha']})")

    peft_model.merge_and_unload()  # mutates `policy` in place: adapters folded in, removed
    print("[merge_lora_checkpoint] merged -- policy is now a plain (non-LoRA) state dict")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    torch.save({"step": ckpt.get("step"), "config": config, "model_state_dict": policy.state_dict()}, args.out_path)
    print(f"[merge_lora_checkpoint] wrote {args.out_path} -- load with scripts/serve_policy.py unmodified")


if __name__ == "__main__":
    main()
