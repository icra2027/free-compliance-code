"""Cross-operator adverb->force analysis for Figure 6.

Reuses fr3_bilateral_teleop/dataset_tools/evaluation/analyze_adverb_separation.py's own
per-episode contact-force computation, manner parsing, and Cohen's-d
implementation (all already self-tested there) rather than reimplementing
them -- this script only adds the operator_id join (from session_manifest.jsonl,
which analyze_adverb_separation.py deliberately does NOT read for manner/task,
but IS the right source for operator identity) and the cross-operator
comparisons / Figure 6 that script doesn't produce.

Two questions, for operator adverb calibration:
  1. Within each operator, do normally/firmly separate? (H4's core premise,
     per-operator -- analyze_adverb_separation.py answers this pooled-across-
     operators; here it's split.)
  2. Across operators, within each manner, do A and B agree? (Is the
     gently/normally/firmly mapping shared, or person-specific?)
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)          # compliance-vla/
REPO_ROOT = os.path.dirname(PROJECT_ROOT)            # datasets/ (holds dataset/ and external/)
EXTERNAL_SCRIPTS = os.path.join(
    PROJECT_ROOT, "hardware", "fr3_bilateral_teleop", "dataset_tools", "evaluation")  # dataset evaluation
DATA_EXTRACTION_DIR = os.path.join(PROJECT_ROOT, "data_extraction")  # dataset_io, panda_fk, extraction drivers
sys.path.insert(0, EXTERNAL_SCRIPTS)
sys.path.insert(0, DATA_EXTRACTION_DIR)
sys.path.insert(0, SCRIPT_DIR)

import dataset_io as dio  # noqa: E402
from analyze_adverb_separation import (  # noqa: E402
    manner_from_task_text, per_episode_contact_force, cohens_d, load_episode_table,
)

OUT_DIR = os.path.join(PROJECT_ROOT, "reports")
CONTACT_FORCE_THRESHOLD = 2.0

COLOR_NORMALLY = "#2a78d6"  # dataviz reference palette slot 1 (blue)
COLOR_FIRMLY = "#eb6834"    # slot 2 (orange)


def build_per_episode_table():
    rows = []
    for session in dio.SESSIONS:
        dataset_dir = Path(dio.session_dir(session))
        manifest = {r["episode_index"]: r for r in dio.load_session_manifest(session)}
        episodes = load_episode_table(dataset_dir)
        for _, row in episodes.iterrows():
            task_text = row["tasks"][0] if len(row["tasks"]) else ""
            manner = manner_from_task_text(task_text)
            stats = per_episode_contact_force(dataset_dir, row, CONTACT_FORCE_THRESHOLD)
            rows.append({
                "session": session,
                "episode_index": int(row["episode_index"]),
                "task": task_text,
                "manner": manner,
                "operator_id": manifest.get(int(row["episode_index"]), {}).get("operator_id"),
                **stats,
            })
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = build_per_episode_table()
    df = df[df["mean_contact_force_n"].notna() & df["manner"].isin(["normally", "firmly"])]
    df = df[df["operator_id"].notna()]
    df.to_csv(os.path.join(OUT_DIR, "per_episode_contact_force.csv"), index=False)

    summary = (df.groupby(["operator_id", "manner"])["mean_contact_force_n"]
               .agg(n="count", mean_force_n="mean", sd_force_n="std").reset_index())
    print("Per operator x manner force summary:")
    print(summary.to_string(index=False))

    effects = {}
    for op in sorted(df["operator_id"].unique()):
        norm = df[(df.operator_id == op) & (df.manner == "normally")]["mean_contact_force_n"].to_numpy()
        firm = df[(df.operator_id == op) & (df.manner == "firmly")]["mean_contact_force_n"].to_numpy()
        if len(norm) >= 2 and len(firm) >= 2:
            d = cohens_d(firm, norm)
            effects[f"operator_{op}_firmly_vs_normally_d"] = d
            print(f"operator {op}: firmly vs normally Cohen's d = {d:+.2f} "
                  f"(n_normally={len(norm)}, n_firmly={len(firm)})")

    for manner in ["normally", "firmly"]:
        by_op = {op: g["mean_contact_force_n"].to_numpy()
                 for op, g in df[df.manner == manner].groupby("operator_id")}
        ops = sorted(by_op)
        if len(ops) == 2 and all(len(by_op[o]) >= 2 for o in ops):
            d = cohens_d(by_op[ops[1]], by_op[ops[0]])
            effects[f"{manner}_operator_{ops[1]}_vs_{ops[0]}_d"] = d
            print(f"{manner}: operator {ops[1]} vs {ops[0]} Cohen's d = {d:+.2f} "
                  f"(n_{ops[0]}={len(by_op[ops[0]])}, n_{ops[1]}={len(by_op[ops[1]])})")

    with open(os.path.join(OUT_DIR, "cross_operator_adverb_effects.json"), "w") as f:
        json.dump({"summary": summary.to_dict(orient="records"), "cohens_d": effects}, f, indent=2)

    # --- Figure 6 ---
    operators = sorted(df["operator_id"].unique())
    fig, axes = plt.subplots(1, len(operators), figsize=(5.5 * len(operators), 4.4), sharey=True)
    if len(operators) == 1:
        axes = [axes]

    for ax, op in zip(axes, operators):
        sub = df[df.operator_id == op]
        data = [sub[sub.manner == "normally"]["mean_contact_force_n"].to_numpy(),
                sub[sub.manner == "firmly"]["mean_contact_force_n"].to_numpy()]
        bp = ax.boxplot(
            data, positions=[0, 1], widths=0.5, patch_artist=True,
            medianprops=dict(color="#0b0b0b", linewidth=2),
            whiskerprops=dict(color="#52514e", linewidth=1.5),
            capprops=dict(color="#52514e", linewidth=1.5),
            boxprops=dict(linewidth=1.5),
            flierprops=dict(marker="o", markersize=4, markerfacecolor="#52514e",
                             markeredgecolor="none", alpha=0.6),
        )
        for patch, color in zip(bp["boxes"], [COLOR_NORMALLY, COLOR_FIRMLY]):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
            patch.set_edgecolor(color)

        rng = np.random.default_rng(0)
        for i, vals in enumerate(data):
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=18, color="#0b0b0b",
                       alpha=0.5, zorder=3, linewidths=0)

        for band, (lo, hi) in [("normally", (8, 12)), ("firmly", (15, 22))]:
            x = 0 if band == "normally" else 1
            ax.hlines([lo, hi], x - 0.28, x + 0.28, colors="#8a8a86", linestyles="dashed", linewidth=1)

        n_norm, n_firm = len(data[0]), len(data[1])
        d_val = effects.get(f"operator_{op}_firmly_vs_normally_d")
        title = f"Operator {op}  (n={n_norm}+{n_firm})"
        if d_val is not None:
            title += f"\nfirmly vs normally: Cohen's d = {d_val:.2f}"
        ax.set_title(title, fontsize=11, color="#0b0b0b")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["normally", "firmly"], fontsize=10.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#c3c2b7")
        ax.spines["bottom"].set_color("#c3c2b7")
        ax.tick_params(colors="#52514e")
        ax.grid(axis="y", color="#e5e4df", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Mean contact-force magnitude (N)", fontsize=10.5, color="#0b0b0b")
    fig.suptitle("Figure 6: adverb → realized contact force, by operator\n"
                  "(dashed lines: pre-registered live-display target bands)",
                  fontsize=12.5, color="#0b0b0b")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(top=0.78)
    fig_path = os.path.join(OUT_DIR, "figure6_cross_operator_adverb_force.png")
    fig.savefig(fig_path, dpi=200, facecolor="white")
    print(f"saved figure -> {fig_path}")


if __name__ == "__main__":
    main()
