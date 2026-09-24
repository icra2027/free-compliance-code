"""How much does each of the policy's inputs actually move its prediction, on IN-
DISTRIBUTION (training-session) data?

Motivation: check_state_convention_sensitivity.py's per-input ablation (colour word /
scene image / wrist image / x_f position, one swapped at a time, rest held fixed) was
first run against `demo_*B` (operator B) sessions -- but the deployed checkpoint
(`b0_seed0_operatorA_two_color`) was trained on operator A only (demo_red_left, demo_blue,
demo_blue_firm, demo_red_firm); operator B is unseen data for it. A low colour-word
sensitivity measured there conflates two different explanations that call for different
fixes -- "the model never learned to condition on the colour word" (fix: training/
architecture) vs. "the model doesn't generalize to operator B's visual setup at all, so
none of its conditioning transfers, colour included" (fix: more/varied training data,
no code change needed) -- and the ablation alone can't distinguish them on OOD data.

This script runs the identical ablation on the checkpoint's OWN training sessions, where
that confound doesn't apply: any input the model is really conditioning on there should
move the prediction, full stop. Compare this run's colour-word sensitivity number against
a same-checkpoint run on operator B (via this same script, different --sessions) to see
whether the OOD gap is real -- i.e. whether colour sensitivity collapses specifically
between A and B, or is uniformly weak on both (which would instead point back at weak
grounding even in-distribution, consistent with the training_data_two_color diagnostic's
modest 63.5% pooled accuracy).

Methodology (same as check_state_convention_sensitivity.py's step 2, factored out):
for each of several recorded (session, episode, frame) observations, get the mean
predicted final position (action_chunk[-1][:3], averaged over --n-seeds fixed seeds --
flow-matching sampling is unseeded per-request, see diagnose_language_grounding.py's own
seed-control writeup) under the real observation, then again with exactly one input
swapped for a donor observation's version of that input (a different episode's scene
image, wrist image, or x_f position; the colour word swapped to the episode's OTHER
referent). Reports the L2 shift in predicted position caused by each swap, and the
episode's own true colour-to-colour target separation as a reference scale.

Usage (needs a running serve_policy.py, e.g. via an SSH tunnel to the GPU host):
    python -m compliance_vla.policy.check_input_sensitivity --dataset-root ../data_two_color \
        --sessions demo_red_left demo_blue demo_blue_firm demo_red_firm \
        --referents red blue --n-episodes 3 --n-seeds 3

    # same checkpoint, operator B, for direct comparison:
    python -m compliance_vla.policy.check_input_sensitivity --dataset-root ../data_two_color \
        --sessions demo_redB demo_blueB demo_blue_firmB demo_red_firmB \
        --referents red blue --n-episodes 3 --n-seeds 3
"""

import argparse
import os
import sys

import numpy as np

from compliance_vla.policy._paths import DATA_EXTRACTION_DIR, ensure_on_sys_path  # noqa: E402

ensure_on_sys_path()  # dataset_io / panda_fk are plain scripts under data_extraction/

import dataset_io as dio  # noqa: E402
from compliance_vla.policy import diagnose_language_grounding as dlg  # noqa: E402

FRAME_INDICES = (5, 40, 80)  # early / mid / late within each (trimmed-length-permitting) episode


def mean_prediction(server_url, scene_rgb, wrist_rgb, state, task, force_history, seeds):
    positions = [
        np.asarray(dlg.get_action_chunk(server_url, scene_rgb, wrist_rgb, state, task,
                                         force_history, seed=s)[-1][:3], dtype=np.float64)
        for s in seeds
    ]
    return np.mean(positions, axis=0)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--server-url", default="http://localhost:8000/act")
    p.add_argument("--sessions", nargs="+", required=True,
                   choices=dio.SESSIONS + dio.TWO_COLOR_SESSIONS)
    p.add_argument("--dataset-root", default=None)
    p.add_argument("--n-episodes", type=int, default=2,
                   help="episodes per session to sample (first N in dataset order)")
    p.add_argument("--referents", nargs="+", default=list(dlg.DEFAULT_DESCOPE_REFERENTS),
                   choices=list(dlg.REFERENTS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--no-force", action="store_true")
    args = p.parse_args()

    seeds = [args.seed + i for i in range(args.n_seeds)]
    tool_offset = np.load(os.path.join(DATA_EXTRACTION_DIR, "tool_offset.npy"))

    # A fixed donor observation (different episode, mid-episode frame) per session, used
    # as the source of "a different scene/wrist image" -- held constant across all target
    # episodes in that session so every ablation in a session is against the same donor.
    donors = {}
    targets = []  # (session, episode_index)
    for session in args.sessions:
        session_path = dlg._resolve_session_dir(session, args.dataset_root)
        eps = dlg.pick_balanced_episodes(
            session_path, n_per_referent=args.n_episodes, referents=args.referents
        )
        if len(eps) < 2:
            print(f"  {session}: fewer than 2 usable episodes, skipping")
            continue
        donors[session] = eps[-1]  # last picked episode is the donor for this session
        targets.extend((session, e) for e in eps[:-1])

    print(f"{'episode@frame':<26}{'baseline xy':<20}{'d_colour':>9}{'d_scene':>9}"
          f"{'d_wrist':>9}{'d_state':>9}{'true_colour_sep':>17}")
    rows = []
    true_targets_by_session = {}
    for session, episode_index in targets:
        donor_ep = donors[session]
        try:
            donor_scene, donor_wrist, *_ = dlg.load_observation(
                session, donor_ep, 40, tool_offset, args.dataset_root
            )
        except Exception as e:
            print(f"  donor {session}#{donor_ep}: SKIPPED ({type(e).__name__}: {e})")
            continue

        for frame_index in FRAME_INDICES:
            try:
                (scene_rgb, wrist_rgb, state, force_history,
                 true_referent, true_task, true_target) = dlg.load_observation(
                    session, episode_index, frame_index, tool_offset, args.dataset_root
                )
            except Exception as e:
                print(f"  {session}#{episode_index}@{frame_index}: SKIPPED "
                      f"({type(e).__name__}: {e})")
                continue
            if args.no_force:
                force_history = None
            other = next(c for c in args.referents if c != true_referent)
            manner = true_task.split()[-1]

            baseline = mean_prediction(
                args.server_url, scene_rgb, wrist_rgb, state, true_task, force_history, seeds
            )
            d_colour = np.linalg.norm(mean_prediction(
                args.server_url, scene_rgb, wrist_rgb, state,
                f"wipe the {other} mark {manner}", force_history, seeds
            ) - baseline)
            d_scene = np.linalg.norm(mean_prediction(
                args.server_url, donor_scene, wrist_rgb, state, true_task, force_history, seeds
            ) - baseline)
            d_wrist = np.linalg.norm(mean_prediction(
                args.server_url, scene_rgb, donor_wrist, state, true_task, force_history, seeds
            ) - baseline)
            state_shifted = state.copy()
            state_shifted[14:17] += np.float32([0.0, 0.0, 0.08])
            d_state = np.linalg.norm(mean_prediction(
                args.server_url, scene_rgb, wrist_rgb, state_shifted, true_task,
                force_history, seeds
            ) - baseline)

            # Same-session true colour separation, for scale: predict for the OTHER
            # colour's own recorded episodes in this session (first one, any frame),
            # compare its true wipe endpoint against this episode's.
            if session not in true_targets_by_session:
                true_targets_by_session[session] = {}
            true_targets_by_session[session][true_referent] = true_target
            other_true = true_targets_by_session[session].get(other)
            sep = np.linalg.norm(true_target - other_true) if other_true is not None else float("nan")

            tag = f"{session}#{episode_index}@{frame_index}"
            print(f"{tag:<26}{np.array2string(baseline[:2], precision=3):<20}"
                  f"{d_colour * 100:>8.1f}{d_scene * 100:>9.1f}{d_wrist * 100:>9.1f}"
                  f"{d_state * 100:>9.1f}{sep * 100 if sep == sep else float('nan'):>16.1f}")
            rows.append((d_colour, d_scene, d_wrist, d_state))

    if rows:
        r = np.array(rows) * 100
        print(f"\nmean shift (cm), n={len(rows)}:  colour={r[:, 0].mean():.2f}  "
              f"scene={r[:, 1].mean():.2f}  wrist={r[:, 2].mean():.2f}  "
              f"state_pos(+8cm)={r[:, 3].mean():.2f}")


if __name__ == "__main__":
    main()
