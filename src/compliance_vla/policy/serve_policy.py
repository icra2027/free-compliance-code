#!/usr/bin/env python3
"""Serve a trained B0/B2/B5 checkpoint over HTTP for real-robot evaluation.

Loads one checkpoint written by src/compliance_vla/policy/train_policy.py's --checkpoint-dir
(`torch.save({"step":, "config":, "model_state_dict":}, ...)`) and exposes
it as a FastAPI server with a single POST /act endpoint, modeled on
moojink/openvla-oft's vla-scripts/deploy.py (json_numpy wire format so
numpy arrays round-trip through JSON with no manual encoding, FastAPI +
uvicorn, one endpoint). Unlike that reference, this server is stateless
across requests -- it calls `predict_action_chunk` directly rather than
lerobot's `select_action` action queue, so no server-side state depends on
requests arriving in order. The client sends one full observation per call
and gets back the whole `(chunk_size, action_dim)` predicted action chunk,
then executes however many of those steps it trusts open-loop before
calling again. That fits this project's split better than one action per
HTTP round trip: the client is the robot-control machine (no GPU, possibly
no direct network path to wherever the checkpoint lives), the server is
wherever the checkpoint + venv are.

    python -m compliance_vla.policy.serve_policy --checkpoint checkpoints/b5_seed0/b5_seed0_step30000.pt
    python -m compliance_vla.policy.serve_policy --checkpoint checkpoints/b5_seed0/b5_seed0_step30000.pt --chunk-size 16

Needs fastapi, uvicorn, json-numpy on top of the usual venv (smolvla_policy/README.md):
    bin/uv pip install fastapi uvicorn json-numpy

### Request format (POST /act)

JSON body, numpy arrays encoded via json_numpy's wire format (`{"__numpy__":
<base64>, "dtype":, "shape":}`) -- send it with `json_numpy.dumps`/decode
the response with `json_numpy.loads`, not `requests.post(..., json=...)` or
`resp.json()` (see src/compliance_vla/policy/client_example.py, and the note by the import
below on why this server deliberately does not call `json_numpy.patch()`):

    {
      "task": "insert wiper into holder",         # instruction, str -- must match training's task text
      "scene_rgb":  <uint8 array, (H, W, 3)>,      # any size; resized server-side (SmolVLA's own resize_with_pad)
      "wrist_rgb":  <uint8 array, (H, W, 3)>,
      "state":      <float32 array, (20,)>,        # [q(7), qdot(7), x_f(6)] -- STATE_DIM, src/compliance_vla/policy/train_policy.py
      "force_history": <float32 array, (20, 6)>,   # b2/b5 only, omit for b0; last 500ms of wrench,
                                                    # resampled to 20 pts the same way training does
                                                    # (compliance_vla.policy.force_encoder.resample_to_n_samples)
    }

Response: `{"action_chunk": <float32 array, (chunk_size, action_dim)>}` --
action_dim is 7 ([x_eq(6), gripper(1)]) for b0/b2, 13 ([x_eq(6), log_k(6),
gripper(1)]) for b3/b5 (b3's log_k comes from a direct regression head
instead of the flow-matching target, see src/compliance_vla/policy/hybrid_policy.py, but the
response layout is identical); chunk_size is whatever the checkpoint's
config used (32 for all the baselines). On error:
`{"error": <traceback str>}`, HTTP 400.

### Client access over SSH

From the robot-control machine, tunnel a local port to this server (this is
a plain ssh hop to whichever machine is running it -- not the `hpc` alias
from smolvla_policy/README.md, which is a different machine entirely):

    ssh -N -L 8000:localhost:8000 <user>@<server-host>

Then on the robot machine, POST to http://localhost:8000/act as if the
server were local -- see src/compliance_vla/policy/client_example.py for a minimal client.
"""

import argparse
import os
import traceback

import json_numpy
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

# Deliberately NOT calling json_numpy.patch() here: it monkeypatches the
# stdlib `json` module process-wide, and `import lerobot` below transitively
# imports numpy.testing, which calls plain `json.loads` on its own data
# during import -- json_numpy's object_hook chokes on that (confirmed: raises
# TypeError, unrelated to anything this server sends/receives). Using
# json_numpy.dumps/.loads explicitly, only on this endpoint's own request and
# response bytes, gets the same numpy wire format without patching anything
# global.


from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402

from compliance_vla.policy.compliance_policy import ComplianceSmolVLAConfig, ComplianceSmolVLAPolicy  # noqa: E402
from compliance_vla.policy.hybrid_policy import HybridSmolVLAConfig, HybridSmolVLAPolicy  # noqa: E402


def load_policy(checkpoint_path, device, chunk_size=None):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    config.device = device
    # state_dict below already has the real (pretrained-backbone + finetuned) weights,
    # so building from a random-init VLM and overwriting it skips a redundant HF
    # download -- matters here since the robot-control network may not have it either.
    config.load_vlm_weights = False
    if chunk_size is not None:
        config.chunk_size = chunk_size
        if hasattr(config, "n_action_steps"):
            config.n_action_steps = chunk_size
    print(config)
    # HybridSmolVLAConfig subclasses ComplianceSmolVLAConfig (src/compliance_vla/policy/hybrid_policy.py), so
    # it must be checked FIRST -- the reverse order would route every B3 checkpoint through
    # ComplianceSmolVLAPolicy (B5's class) instead, silently dropping the log_k_head entirely
    # (load_state_dict would then either error on unexpected keys or, worse, partially match).
    if isinstance(config, HybridSmolVLAConfig):
        policy_cls = HybridSmolVLAPolicy
    elif isinstance(config, ComplianceSmolVLAConfig):
        policy_cls = ComplianceSmolVLAPolicy
    else:
        policy_cls = SmolVLAPolicy
    policy = policy_cls(config)
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.to(device)
    policy.eval()

    uses_force = isinstance(config, ComplianceSmolVLAConfig) and config.use_force_input
    action_dim = config.output_features["action"].shape[0]
    print(f"[serve_policy] loaded {checkpoint_path} -- {policy_cls.__name__}, step {ckpt.get('step')}, "
          f"action_dim {action_dim}, chunk_size {config.chunk_size}, force_input={uses_force}")
    return policy, uses_force


def _to_chw_float(img_hwc_uint8, device):
    """(H, W, 3) uint8 -> (1, 3, H, W) float32 in [0, 1], matching
    compliance_vla.policy.dataset._to_chw_float / SmolVLAPolicy.prepare_images's expected input."""
    t = torch.from_numpy(np.asarray(img_hwc_uint8, dtype=np.uint8))
    return t.permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)


class PolicyServer:
    def __init__(self, policy, uses_force, device):
        self.policy = policy
        self.uses_force = uses_force
        self.device = device
        self.tokenizer = policy.model.vlm_with_expert.processor.tokenizer

    def act(self, payload):
        seed = payload.get("seed")
        if seed is not None:
            # Flow-matching sampling starts from torch.normal(...) noise with no seeding
            # anywhere in the request path (VLAFlowMatching.sample_noise) -- two calls with
            # identical (task, observation) otherwise get independent noise draws and can
            # land at different denoised points. Real deployment wants that call-to-call
            # variety, so this stays opt-in: only requests that explicitly pass "seed" get a
            # reseed right before sampling, e.g. diagnose_language_grounding.py holding the
            # seed fixed across colour words to isolate language conditioning from sampling
            # noise.
            torch.manual_seed(int(seed))
            if self.device.startswith("cuda"):
                torch.cuda.manual_seed_all(int(seed))
        tok = self.tokenizer(
            [payload["task"]], padding=self.policy.config.pad_language_to, truncation=True,
            max_length=self.policy.config.tokenizer_max_length, return_tensors="pt",
        )
        batch = {
            "observation.state": torch.as_tensor(
                payload["state"], dtype=torch.float32, device=self.device
            ).unsqueeze(0),
            "observation.images.scene_rgb": _to_chw_float(payload["scene_rgb"], self.device),
            "observation.images.wrist_rgb": _to_chw_float(payload["wrist_rgb"], self.device),
            "observation.language.tokens": tok["input_ids"].to(self.device),
            "observation.language.attention_mask": tok["attention_mask"].bool().to(self.device),
        }
        if self.uses_force:
            force_hist = payload.get("force_history")
            if force_hist is None:
                raise ValueError(
                    "this checkpoint was trained with force input (use_force_input=True) -- "
                    "'force_history' is required in the request"
                )
            batch["force_history"] = torch.as_tensor(
                force_hist, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

        action_chunk = self.policy.predict_action_chunk(batch)  # (1, chunk_size, action_dim)
        return action_chunk[0].cpu().numpy()


def build_app(server: PolicyServer) -> FastAPI:
    app = FastAPI()

    @app.post("/act")
    async def act(request: Request):
        try:
            payload = json_numpy.loads(await request.body())
            action_chunk = server.act(payload)
            body = json_numpy.dumps({"action_chunk": action_chunk})
            return Response(content=body, media_type="application/json")
        except Exception:
            tb = traceback.format_exc()
            print(f"[serve_policy] request failed:\n{tb}")
            return JSONResponse({"error": tb}, status_code=400)

    return app


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True,
                   help="path to a .pt file written by src/compliance_vla/policy/train_policy.py's --checkpoint-dir")
    p.add_argument("--chunk-size", type=int,
                   help="override the checkpoint config's action chunk size at inference time")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    policy, uses_force = load_policy(args.checkpoint, args.device, chunk_size=args.chunk_size)
    server = PolicyServer(policy, uses_force, args.device)
    app = build_app(server)

    print(f"[serve_policy] listening on {args.host}:{args.port} (POST /act)")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
