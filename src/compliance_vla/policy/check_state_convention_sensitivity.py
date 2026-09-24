r"""Offline A/B: how much does the policy's prediction change when you feed it x_f in
deploy_smolvla.py's frame convention instead of the one src/compliance_vla/policy/labels.py actually
trained on?

Why this exists: `replay_policy_on_data_two_color.py` (open-loop, teacher-forced on a
recorded episode) reportedly produces correct-looking real-robot behaviour from the same
checkpoint that `deploy_smolvla.py` (closed-loop, live sensors) does not. The two scripts
disagree about what `x_f` -- half the policy's 20-dim proprio input -- means:

  training / diagnose_language_grounding.load_observation / replay_policy_...:
      x_f = [ FK_flange_pos(q) + R_flange(q) @ tool_offset ,  rotvec(R_flange(q)) ]
            \____ tip position, EE/leader convention ____/    \__ FLANGE rotation __/
      i.e. a MIXED convention: tip position, but the *flange's* orientation, because
      panda_fk.fk() stops at the flange ("flange frame, no hand/TCP offset", see its
      docstring) and calibrate_tool_offset.py only ever fit a translation.

  deploy_smolvla.get_current_state():
      x_f = [ current_pose.position + R_ee @ (0,0,0.08) ,  rotvec(R_ee) ]
      where current_pose is /franka_robot_state_broadcaster/current_pose -- the robot's
      O_T_EE (hand) frame, NOT the flange frame.

`observation.leader_pose` (which is what x_eq is, and what tool_offset.npy was fit
against) comes from the *same* broadcaster topic on the leader arm
(data_recorder's `leader_pose_topic` default), so the flange->EE rotation gap
is directly measurable from the recorded dataset alone -- that is step 1 below, and it
needs no robot and no server.

Step 2 then asks the question that actually matters: does that gap move the model's
output? It rebuilds each recorded frame's state four ways --

  train     what training/replay feed (the reference)
  deploy    what deploy_smolvla.py feeds (EE rotation + the extra 8cm tool translation)
  rot_only  train, but with the EE-frame rotation      (isolates the rotation gap)
  pos_only  train, but with the extra 8cm on position  (isolates the translation gap)

-- holds images/task/force_history byte-identical across all four, and reports how far
the predicted x_eq moves. Everything is scored against the episode's real recorded wipe
endpoint (`diagnose_language_grounding._true_wipe_endpoint`), and the colour-selection
verdict is recomputed per variant, so this also answers "does the deploy-convention state
destroy colour selectivity, or just displace the target?"

Sampling noise is real here (see diagnose_language_grounding's --n-seeds writeup: the
flow-matching sampler draws fresh noise per request), so every prediction is averaged
over --n-seeds fixed seeds, and the SAME seed set is used for every variant -- otherwise
a variant-to-variant difference is partly just two different noise draws.

Usage:
    # step 1 only, no server needed -- prints the measured flange->EE rotation gap
    python -m compliance_vla.policy.check_state_convention_sensitivity --dataset-root ../data_two_color \
        --sessions demo_blue_firmB demo_red_firmB --dry-run

    # full A/B against a running serve_policy.py
    python -m compliance_vla.policy.check_state_convention_sensitivity --dataset-root ../data_two_color \
        --sessions demo_redB demo_blueB demo_blue_firmB demo_red_firmB \
        --referents red blue --no-force --n-per-referent 3 --n-seeds 5

How to read it:
  - Step 1's angle ~= 45 deg (Franka's standard F_T_EE hand rotation) confirms deploy is
    feeding a rotation the policy never saw in training. ~0 deg would refute this whole
    hypothesis.
  - Step 2: if `deploy` predictions sit several cm from `train` predictions and/or flip
    the colour verdict, the live failure is explained by the state convention, not by
    grounding. If `deploy` predictions are within ~1cm of `train` and score the same, the
    convention gap is NOT the live bug and you should look at the camera feed / closed-loop
    rate / start pose instead.
  - `rot_only` vs `pos_only` attributes the damage to one of the two gaps, which decides
    which fix to make first.
"""

import argparse
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

from compliance_vla.policy._paths import DATA_EXTRACTION_DIR, ensure_on_sys_path  # noqa: E402

ensure_on_sys_path()  # dataset_io / panda_fk are plain scripts under data_extraction/

import dataset_io as dio  # noqa: E402
import panda_fk as fk  # noqa: E402
from compliance_vla.policy import diagnose_language_grounding as dlg  # noqa: E402

# deploy_smolvla.py's own constructor default (tool_offset_translation), applied on top of
# current_pose when tool_offset_apply_to_state=True -- also its default, despite that
# option's docstring in the same file claiming False is the default.
DEPLOY_EXTRA_TOOL_TRANSLATION = np.array([0.0, 0.0, 0.08])

# Same static-frame gating calibrate_tool_offset.py uses to pick frames where the leader
# and follower genuinely coincide (near-zero external force AND near-zero joint velocity,
# not force alone -- bilateral teleop keeps driving the leader off-contact).
FORCE_THRESHOLD_N = 0.5
VEL_THRESHOLD_RAD_S = 0.01


def fit_flange_to_ee_rotation(sessions, dataset_root):
    """Step 1: measure the rotation between the frame training's x_f reports orientation in
    (FK flange) and the frame deploy's current_pose reports it in (broadcaster O_T_EE).

    Uses observation.leader_pose as the stand-in for current_pose: both are the same
    `franka_robot_state_broadcaster/current_pose` message type on the two arms, so their
    frame conventions are identical, and on static frames the leader and follower coincide
    (exactly the assumption calibrate_tool_offset.py already relies on for translation).

    Returns (mean_rotation, per_frame_angles_deg, n_frames)."""
    deltas = []
    for session in sessions:
        df = dio.load_frames(session, dataset_root=dataset_root)
        q = dio.stack_col(df, "observation.state")
        leader_pose = dio.stack_col(df, "observation.leader_pose")
        wrench = dio.stack_col(df, "observation.wrench.external_base")
        vel = dio.stack_col(df, "observation.velocity")
        static = (np.linalg.norm(wrench[:, :3], axis=1) < FORCE_THRESHOLD_N) & (
            np.linalg.norm(vel, axis=1) < VEL_THRESHOLD_RAD_S
        )
        n_static = int(static.sum())
        print(f"  {session}: {n_static} static frames / {len(static)}")
        if n_static == 0:
            continue
        _, rot = fk.fk_batch(q[static])
        r_flange = R.from_matrix(rot)
        r_ee = R.from_rotvec(leader_pose[static][:, 3:6])
        deltas.append((r_flange.inv() * r_ee).as_rotvec())

    if not deltas:
        raise RuntimeError("no static frames found in any session -- cannot fit the rotation gap")
    rotvecs = np.concatenate(deltas, axis=0)
    mean_rotation = R.from_rotvec(rotvecs).mean()
    angles_deg = np.degrees(np.linalg.norm(rotvecs, axis=1))
    return mean_rotation, angles_deg, len(rotvecs)


def build_state_variants(state, tool_offset, flange_to_ee):
    """Rebuilds one recorded frame's 20-dim state under each frame convention.

    `state` is load_observation's output, i.e. the training convention:
    [q(7), qdot(7), tip_pos(3), rotvec(R_flange)(3)]. q/qdot are untouched throughout --
    only x_f's 6 numbers differ between variants."""
    q, qdot = state[:7].astype(np.float64), state[7:14]
    tip_pos = state[14:17].astype(np.float64)
    r_flange = R.from_rotvec(state[17:20].astype(np.float64))

    # What /current_pose would report for this same q: same origin as the fitted tip
    # position (that IS what tool_offset.npy was fit to reproduce), rotated into the EE
    # frame. Then deploy adds its own 8cm on top of that, in the EE frame.
    r_ee = r_flange * flange_to_ee
    deploy_pos = tip_pos + r_ee.apply(DEPLOY_EXTRA_TOOL_TRANSLATION)

    def pack(pos, rot):
        return np.concatenate([q, qdot, pos, rot.as_rotvec()]).astype(np.float32)

    return {
        "train": pack(tip_pos, r_flange),
        "deploy": pack(deploy_pos, r_ee),
        "rot_only": pack(tip_pos, r_ee),
        "pos_only": pack(deploy_pos, r_flange),
    }


def mean_prediction(server_url, scene_rgb, wrist_rgb, state, task, force_history, seeds):
    """Mean predicted final x_eq position over a fixed seed set -- the same
    mean-position-over-seeds estimator diagnose_language_grounding.py settled on, so a
    variant-to-variant difference here isn't just two different noise draws."""
    positions = []
    for seed in seeds:
        chunk = dlg.get_action_chunk(
            server_url, scene_rgb, wrist_rgb, state, task, force_history, seed=seed
        )
        positions.append(np.asarray(chunk[-1][:3], dtype=np.float64))
    return np.mean(positions, axis=0)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--server-url", default="http://localhost:8000/act")
    p.add_argument("--sessions", nargs="+", required=True,
                   choices=dio.SESSIONS + dio.TWO_COLOR_SESSIONS)
    p.add_argument("--dataset-root", default=None)
    p.add_argument("--n-per-referent", type=int, default=2)
    p.add_argument("--frame-index", type=int, default=5,
                   help="frame within each episode (same default/rationale as "
                        "diagnose_language_grounding.py: early, before the arm occludes a mark)")
    p.add_argument("--manner", default="normally",
                   help="manner word used to compose the counterfactual colour prompts")
    p.add_argument("--referents", nargs="+", default=list(dlg.DEFAULT_DESCOPE_REFERENTS),
                   choices=list(dlg.REFERENTS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--no-force", action="store_true", help="omit force_history (b0 checkpoints)")
    p.add_argument("--dry-run", action="store_true",
                   help="step 1 + state-vector deltas only; never contacts the server")
    args = p.parse_args()

    referents = tuple(args.referents)
    seeds = [args.seed + i for i in range(args.n_seeds)]
    tool_offset = np.load(os.path.join(DATA_EXTRACTION_DIR, "tool_offset.npy"))
    print(f"tool_offset.npy = {np.array2string(tool_offset, precision=5)}  "
          f"(|.|={np.linalg.norm(tool_offset) * 100:.2f}cm)\n")

    print("=== Step 1: measured flange -> EE(current_pose) rotation gap ===")
    flange_to_ee, angles_deg, n_frames = fit_flange_to_ee_rotation(args.sessions, args.dataset_root)
    gap_rotvec = flange_to_ee.as_rotvec()
    gap_angle = np.degrees(np.linalg.norm(gap_rotvec))
    axis = gap_rotvec / max(np.linalg.norm(gap_rotvec), 1e-12)
    print(f"\n  pooled static frames: {n_frames}")
    print(f"  mean rotation: {gap_angle:.2f} deg about tool-frame axis "
          f"[{axis[0]:+.3f}, {axis[1]:+.3f}, {axis[2]:+.3f}]")
    print(f"  euler xyz (deg): {np.array2string(flange_to_ee.as_euler('xyz', degrees=True), precision=1)}")
    print(f"  per-frame |angle| deg: p5={np.percentile(angles_deg, 5):.1f} "
          f"p50={np.median(angles_deg):.1f} p95={np.percentile(angles_deg, 95):.1f}")
    print("\n  -> This is how far deploy_smolvla.py's x_f rotation is from the convention the")
    print("     checkpoint was trained on. ~45 deg = Franka's standard hand F_T_EE rotation.\n")

    print("=== Step 2: does that change what the policy predicts? ===")
    rows = []
    for session in args.sessions:
        session_path = dlg._resolve_session_dir(session, args.dataset_root)
        episodes = dlg.pick_balanced_episodes(
            session_path, n_per_referent=args.n_per_referent, referents=referents
        )
        for episode_index in episodes:
            try:
                (scene_rgb, wrist_rgb, state, force_history,
                 true_referent, true_task, true_target) = dlg.load_observation(
                    session, episode_index, args.frame_index, tool_offset, args.dataset_root
                )
            except Exception as e:  # a single bad episode shouldn't kill the sweep
                print(f"  {session}#{episode_index}: SKIPPED ({type(e).__name__}: {e})")
                continue
            if args.no_force:
                force_history = None

            variants = build_state_variants(state, tool_offset, flange_to_ee)
            d_pos = np.linalg.norm(variants["deploy"][14:17] - variants["train"][14:17])
            d_rot = np.degrees(np.linalg.norm(
                (R.from_rotvec(variants["deploy"][17:20].astype(np.float64))
                 * R.from_rotvec(variants["train"][17:20].astype(np.float64)).inv()).as_rotvec()
            ))
            print(f"\n  {session}#{episode_index} true={true_referent!r} "
                  f"task={true_task!r}\n"
                  f"    state gap (deploy vs train): position {d_pos * 100:.2f}cm, "
                  f"rotation {d_rot:.1f}deg")
            if args.dry_run:
                continue

            per_variant = {}
            for name, variant_state in variants.items():
                preds = {
                    colour: mean_prediction(
                        args.server_url, scene_rgb, wrist_rgb, variant_state,
                        f"wipe the {colour} mark {args.manner}", force_history, seeds,
                    )
                    for colour in referents
                }
                closest = min(
                    referents, key=lambda c: np.linalg.norm(preds[c] - true_target)
                )
                per_variant[name] = (preds, closest)

            train_pred = per_variant["train"][0][true_referent]
            for name in ("train", "deploy", "rot_only", "pos_only"):
                preds, closest = per_variant[name]
                pred = preds[true_referent]
                shift = np.linalg.norm(pred - train_pred)
                err = np.linalg.norm(pred - true_target)
                # How far the prediction moves when only the colour word changes -- the
                # within-episode colour spread this project already uses as its direct
                # read on language conditioning strength.
                spread = max(
                    np.linalg.norm(preds[a] - preds[b])
                    for a in referents for b in referents
                )
                print(f"    {name:<9} pred={np.array2string(pred, precision=3)}  "
                      f"shift_vs_train={shift * 100:5.1f}cm  err_vs_truth={err * 100:5.1f}cm  "
                      f"colour_spread={spread * 100:4.1f}cm  closest={closest!r} "
                      f"{'OK' if closest == true_referent else 'MISS'}")
                rows.append((name, closest == true_referent, shift, err, spread))

    if rows and not args.dry_run:
        print("\n=== Pooled ===")
        for name in ("train", "deploy", "rot_only", "pos_only"):
            sel = [r for r in rows if r[0] == name]
            if not sel:
                continue
            correct = sum(1 for r in sel if r[1])
            print(f"  {name:<9} colour correct {correct}/{len(sel)} ({correct / len(sel):.0%})  "
                  f"mean shift_vs_train={np.mean([r[2] for r in sel]) * 100:5.1f}cm  "
                  f"mean err_vs_truth={np.mean([r[3] for r in sel]) * 100:5.1f}cm  "
                  f"mean colour_spread={np.mean([r[4] for r in sel]) * 100:4.1f}cm")


if __name__ == "__main__":
    main()
