"""Figure 3 (partial): identifiability mask coverage, dataset-wide.

Real data, from reports/mask_coverage.csv (Day 10 extraction run over the full
92-episode T1 dataset, demo1/demo3/demo4). No fabricated numbers.

NOTE -- this is only the mask-coverage half of proposal Figure 3. The other half
("extracted stiffness traces with phase annotation") needs real per-timestep
K(t)/phase data, which is not present in this environment (no raw LeRobot
dataset or per-timestep extraction output here, only the aggregated per-episode
summary in reports/extraction_per_episode.json). That panel is NOT drawn here;
do not fabricate a trace to fill the gap.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

BLUE = "#2a78d6"    # slot 1 -- "normally"
ORANGE = "#eb6834"  # slot 2 -- "firmly"
AQUA = "#1baf7a"    # slot 3 -- "overall"
PRIMARY = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"

df = pd.read_csv("reports/mask_coverage.csv")

axis_order = ["fx", "fy", "fz", "tx", "ty", "tz"]
axis_labels = [r"$f_x$", r"$f_y$", r"$f_z$", r"$\tau_x$", r"$\tau_y$", r"$\tau_z$"]

overall = df[df.task == "__all__"].set_index("axis")["coverage_within_contact"]

per_task = df[df.task != "__all__"].copy()
per_task["manner"] = per_task.task.str.extract(r"mark (\w+)$")

normally = per_task[per_task.manner == "normally"].groupby("axis")["coverage_within_contact"].mean()
firmly = per_task[per_task.manner == "firmly"].groupby("axis")["coverage_within_contact"].mean()

fig, ax = plt.subplots(figsize=(7.0, 3.3), constrained_layout=True)
ax.set_facecolor("#fcfcfb")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)

x = np.arange(len(axis_order))
w = 0.26

b1 = ax.bar(x - w, [overall[a] * 100 for a in axis_order], width=w, color=AQUA, label="overall", zorder=3)
b2 = ax.bar(x, [normally[a] * 100 for a in axis_order], width=w, color=BLUE, label="normally", zorder=3)
b3 = ax.bar(x + w, [firmly[a] * 100 for a in axis_order], width=w, color=ORANGE, label="firmly", zorder=3)

for bars in (b1, b2, b3):
    for rect in bars:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 1.5, f"{h:.0f}", ha="center",
                fontsize=6.3, color=SECONDARY)

ax.set_xticks(x)
ax.set_xticklabels(axis_labels, fontsize=9.5, color=PRIMARY)
ax.set_ylabel("mask coverage within contact (%)", fontsize=8.5, color=SECONDARY)
ax.set_ylim(0, 108)
ax.tick_params(axis="y", colors=MUTED, labelsize=8)
ax.grid(axis="y", linewidth=0.6, color=GRID, zorder=0)
ax.axhline(25, color=MUTED, linestyle=":", linewidth=1.4, zorder=4)

handles = [
    plt.Rectangle((0, 0), 1, 1, color=AQUA, label="overall"),
    plt.Rectangle((0, 0), 1, 1, color=BLUE, label="normally"),
    plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="firmly"),
    Line2D([0], [0], color=MUTED, linestyle=":", linewidth=1.4, label="Gate 1 threshold (25%)"),
]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.18),
          fontsize=7.5, frameon=False, ncol=4)
ax.set_title("Identifiability mask coverage, T1 wiping (91/92 episodes, dataset-wide)",
             fontsize=9.5, color=PRIMARY, pad=8)

fig.savefig("reports/figures/fig3_mask_coverage.pdf")
fig.savefig("reports/figures/fig3_mask_coverage.png", dpi=220)
print("wrote reports/figures/fig3_mask_coverage.{pdf,png}")
print("\noverall coverage used:")
print(overall)
