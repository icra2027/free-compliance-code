"""Figure 4: per-axis sensorless wrench noise floor (sigma_f).

Real data, from reports/noise_floor_sigma_f.json (11 Aug 2026 real-hardware
session: fit_residual_bias.py against the 332,626-sample free-space sweep).
This is a single real point estimate per axis (min of raw vs. bias-corrected RMS,
as recorded in that JSON), not a raw-sample distribution -- no per-sample residual CSV exists
in this environment to draw a true distribution from, so a per-axis bar (with the
raw-vs-bias-corrected provenance marked) is what's honestly plottable, not a
violin/box plot. Do not fabricate spread that isn't in the source data.

Caveat carried into the caption: this is a free-space noise floor, not an
independently-verified hanging-mass ground-truth calibration -- that check was
attempted twice on 11 Aug and is still open (see reports/noise_floor_sigma_f.json,
_caveat field).
"""
import json

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BLUE = "#2a78d6"
ORANGE = "#eb6834"
PRIMARY = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"

with open("reports/noise_floor_sigma_f.json") as fh:
    data = json.load(fh)

axes = data["axes"]
sigma_f = data["sigma_f"]
units = data["units"]
helps = data["bias_correction_helps"]
axis_labels = [r"$f_x$", r"$f_y$", r"$f_z$", r"$\tau_x$", r"$\tau_y$", r"$\tau_z$"]

force_idx = [i for i, u in enumerate(units) if u == "N"]
torque_idx = [i for i, u in enumerate(units) if u == "Nm"]

fig, (axF, axT) = plt.subplots(1, 2, figsize=(7.0, 2.9), constrained_layout=True)

for ax, idxs, color, title, unit in (
    (axF, force_idx, BLUE, "Force axes", "N"),
    (axT, torque_idx, ORANGE, "Torque axes", "Nm"),
):
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    xs = range(len(idxs))
    heights = [sigma_f[i] for i in idxs]
    labels = [axis_labels[i] for i in idxs]
    hatches = ["" if helps[axes[i]] else "///" for i in idxs]
    bars = ax.bar(xs, heights, width=0.55, color=color, zorder=3, edgecolor="white", linewidth=0.8)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    for x, rect in zip(xs, bars):
        h = rect.get_height()
        ax.text(x, h + 0.02, f"{h:.3f}", ha="center", fontsize=7.2, color=SECONDARY)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=10, color=PRIMARY)
    ax.set_ylabel(rf"$\sigma_f$ ({unit})", fontsize=8.5, color=SECONDARY)
    ax.set_ylim(0, max(heights) * 1.35)
    ax.tick_params(axis="y", colors=MUTED, labelsize=8)
    ax.grid(axis="y", linewidth=0.6, color=GRID, zorder=0)
    ax.set_title(title, fontsize=9, color=PRIMARY, pad=18)

fig.suptitle(
    "Sensorless wrench noise floor: per-axis $\\sigma_f$, free-space residuals",
    fontsize=8.8, color=PRIMARY,
)
legend_elems = [
    Patch(facecolor="white", edgecolor=MUTED, label="bias-corrected RMS used"),
    Patch(facecolor="white", edgecolor=MUTED, hatch="///", label="raw RMS used (bias model did not help)"),
]
fig.legend(handles=legend_elems, loc="lower center", ncol=2, fontsize=7.2, frameon=False,
           bbox_to_anchor=(0.5, -0.08))

fig.savefig("reports/figures/fig4_noise_floor.pdf", bbox_inches="tight")
fig.savefig("reports/figures/fig4_noise_floor.png", dpi=220, bbox_inches="tight")
print("wrote reports/figures/fig4_noise_floor.{pdf,png}")
