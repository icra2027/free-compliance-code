#!/usr/bin/env python3
"""Minimal example client for src/compliance_vla/policy/serve_policy.py's /act endpoint.

Run this on the robot-control machine, after tunneling a local port to
wherever serve_policy.py is running:

    ssh -N -L 8000:localhost:8000 <user>@<server-host>

Swap the placeholder scene_rgb/wrist_rgb/state/task values below for
whatever your robot driver actually reads (camera frames, joint state,
wrench), and force_history for the trailing 500ms of 6-DoF wrench if
you're driving a b2/b5 checkpoint (omit it entirely for b0). Only needs
`json_numpy` and `requests` -- neither pulls in torch/lerobot, so this can
run on a machine that has no GPU and none of the training deps installed.

Uses json_numpy.dumps/.loads directly rather than `json_numpy.patch()` +
requests' `json=`/`.json()` convenience -- see serve_policy.py's note on why
that global monkeypatch is worth avoiding (it isn't a problem in this
particular script today, since nothing heavy gets imported after it, but
it's the same footgun for any code added here later, so this script and
the server it talks to use the one pattern that's safe everywhere).
"""

import json_numpy
import numpy as np
import requests

SERVER_URL = "http://localhost:8000/act"


def get_action_chunk(scene_rgb, wrist_rgb, state, task, force_history=None):
    """Returns a (chunk_size, action_dim) numpy array -- action_dim is 7
    ([x_eq(6), gripper(1)]) for b0/b2, 13 ([x_eq(6), log_k(6), gripper(1)])
    for b5. Execute however many leading steps of the chunk you trust
    open-loop, then call again with a fresh observation."""
    payload = {"task": task, "scene_rgb": scene_rgb, "wrist_rgb": wrist_rgb, "state": state}
    if force_history is not None:
        payload["force_history"] = force_history
    resp = requests.post(SERVER_URL, data=json_numpy.dumps(payload),
                          headers={"Content-Type": "application/json"})
    resp.raise_for_status()
    return json_numpy.loads(resp.content)["action_chunk"]


if __name__ == "__main__":
    scene_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    wrist_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    state = np.zeros(20, dtype=np.float32)               # [q(7), qdot(7), x_f(6)]
    force_history = np.zeros((20, 6), dtype=np.float32)  # b2/b5 only; drop for b0
    print(f"scene_rgb shape={scene_rgb.shape}, wrist_rgb shape={wrist_rgb.shape}, "
          f"state shape={state.shape}, force_history shape={force_history.shape}")
    action_chunk = get_action_chunk(scene_rgb, wrist_rgb, state, "insert wiper into holder", force_history)
    print(f"action_chunk shape={action_chunk.shape}\nfirst 3 steps:\n{action_chunk[:3]}")
