"""Figure 5: offline compliance-prediction benchmark (M8 / Gate 2).

Real data, from reports/offline_stiffness_benchmark_m8.json (Day 10 run,
session-level split train=demo4/val=demo1/test=demo3, cross-operator
held-out test). Per-axis RMSE on log K for all three baselines -- constant,
nearest-neighbour, and the learned RFF-ridge model. No fabricated numbers;
the nearest-neighbour aggregate (needed for Table III's TODO too) is derived
here as the mean of its real per-axis RMSEs, same aggregation the JSON's own
"gate2" block already applies to constant/learned.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

MUTED = "#898781"    # constant (naive baseline)
ORANGE = "#eb6834"   # nearest-neighbour
BLUE = "#2a78d6"     # learned (RFF-ridge) -- this paper's method
PRIMARY = "#0b0b0b"
SECONDARY = "#52514e"
GRID = "#e1e0d9"

with open("reports/offline_stiffness_benchmark_m8.json") as fh:
    data = json.load(fh)

axis_order = ["fx", "fy", "fz", "tx", "ty", "tz"]
axis_labels = [r"$f_x$", r"$f_y$", r"$f_z$", r"$\tau_x$", r"$\tau_y$", r"$\tau_z$"]
per_axis = data["per_axis"]

constant = [per_axis[a]["constant_rmse_log_k"] for a in axis_order]
nn = [per_axis[a]["nearest_neighbour_rmse_log_k"] for a in axis_order]
learned = [per_axis[a]["learned_rmse_log_k"] for a in axis_order]
beats = [per_axis[a]["learned_beats_constant"] for a in axis_order]

nn_aggregate_ratio = data["gate2"]["mean_constant_rmse_log_k_across_axes"] / np.mean(nn)
print(f"nearest-neighbour aggregate RMSE relative to constant: {nn_aggregate_ratio:.3f}x "
      f"(fill-in value for Table III)")

fig, ax = plt.subplots(figsize=(7.0, 3.3), constrained_layout=True)
ax.set_facecolor("#fcfcfb")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)

x = np.arange(len(axis_order))
w = 0.26

b1 = ax.bar(x - w, constant, width=w, color=MUTED, label="constant", zorder=3)
b2 = ax.bar(x, nn, width=w, color=ORANGE, label="nearest-neighbour", zorder=3)
b3 = ax.bar(x + w, learned, width=w, color=BLUE, label="learned (RFF-ridge)", zorder=3)

for bars in (b1, b2, b3):
    for rect in bars:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 0.02, f"{h:.2f}", ha="center",
                fontsize=6.0, color=SECONDARY)

for xi, wins in zip(x, beats):
    if wins:
        ax.text(xi, -0.09, "beats\nconstant", ha="center", va="top", fontsize=5.6,
                color=BLUE, transform=ax.get_xaxis_transform())

ax.set_xticks(x)
ax.set_xticklabels(axis_labels, fontsize=9.5, color=PRIMARY)
ax.set_ylabel(r"RMSE on $\log K$ (held-out test session)", fontsize=8.2, color=SECONDARY)
ax.set_ylim(0, max(constant + nn + learned) * 1.22)
ax.tick_params(axis="y", colors=MUTED, labelsize=8)
ax.grid(axis="y", linewidth=0.6, color=GRID, zorder=0)

handles = [
    plt.Rectangle((0, 0), 1, 1, color=MUTED, label="constant"),
    plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="nearest-neighbour"),
    plt.Rectangle((0, 0), 1, 1, color=BLUE, label="learned (RFF-ridge)"),
]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22),
          fontsize=7.5, frameon=False, ncol=3)
ax.set_title(
    r"Offline compliance-prediction benchmark: aggregate $1.058\times$ over constant "
    "(3/6 axes individually beat it)", fontsize=9, color=PRIMARY, pad=8)

fig.savefig("reports/figures/fig5_benchmark.pdf", bbox_inches="tight")
fig.savefig("reports/figures/fig5_benchmark.png", dpi=220, bbox_inches="tight")
print("wrote reports/figures/fig5_benchmark.{pdf,png}")
