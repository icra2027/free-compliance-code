"""Figure 3b: extracted stiffness trace with phase annotation, from a REAL demo.

This finishes the half of Figure 3 that could not be built earlier --
the earlier fig3_mask_coverage.py used only the aggregated per-episode summary
that ships in this repo (reports/extraction_per_episode.json). The raw
per-timestep bilateral demo logs turned out to exist after all, in
../../analysis/franka_teleop_demos/ (x_l(t), x_f(t), wrench f(t) at 1kHz,
record_demo.py's own CSV format) -- these are the real pilot demos
(operators A/B, "gently") used for the identifiability evaluation.

IMPORTANT SCOPE CAVEAT, stated here and in the figure caption: this is a
SIMPLIFIED reconstruction, not the paper's real §4.1 pipeline
(extract_impedance_labels.py), which lives in a separate ROS2 workspace
(fr3_bilateral_teleop) not present in this checkout. Differences from the real
pipeline:
  - base/end-effector frame position error (l - f), not a fitted contact frame
    (the method explicitly sanctions reporting an end-effector-frame variant
    "in the appendix so reviewers see the choice was deliberate" -- this is
    that variant, not a replacement for the main contact-frame figure).
  - one static per-window least-squares slope K = sum(f*e)/sum(e^2), no
    log-space regularization, no damping term.
  - a proxy identifiability mask (bounds + sign + R^2 >= 0.3) standing in for
    the real Gram-matrix condition-number + noise-floor mask.
  - a single representative episode, not a dataset-wide aggregate.

Do not caption this in the paper as the primary §4.1 result -- pair it with
fig3_mask_coverage.py's real dataset-wide numbers for that, and use this panel
only as the qualitative "here is what a trace looks like" companion, clearly
labeled as such.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
PRIMARY = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
CONTACT_SHADE = "#fbf0e4"

DEMO_PATH = "../analysis/franka_teleop_demos/T1_wiping_A_gently_1786444370.csv"
CONTACT_FORCE_THRESHOLD = 2.0  # N, matches this project's own real pipeline convention
FS = 1000.0
WIN_SEC = 0.3
OUT_HZ = 30.0
K_MIN, K_MAX = 20.0, 2000.0
R2_MIN = 0.3

df = pd.read_csv(DEMO_PATH)
t = df["t"].values - df["t"].values[0]
e = {a: (df[f"l{a}"].values - df[f"f{a}"].values) for a in "xyz"}
f = {a: df[f"wf{a}"].values for a in "xyz"}
wf_mag = np.sqrt(df["wfx"] ** 2 + df["wfy"] ** 2 + df["wfz"] ** 2).values

win = int(WIN_SEC * FS)
stride = int(FS / OUT_HZ)

out_t, out_contact = [], []
out_K = {a: [] for a in "xyz"}
out_valid = {a: [] for a in "xyz"}

for start in range(0, len(t) - win, stride):
    sl = slice(start, start + win)
    contact = wf_mag[sl].mean() > CONTACT_FORCE_THRESHOLD
    out_t.append(t[sl].mean())
    out_contact.append(contact)
    for a in "xyz":
        ea, fa = e[a][sl], f[a][sl]
        denom = np.sum(ea ** 2)
        if denom < 1e-8:
            out_K[a].append(np.nan)
            out_valid[a].append(False)
            continue
        k = np.sum(fa * ea) / denom
        pred = k * ea
        ss_res = np.sum((fa - pred) ** 2)
        ss_tot = np.sum((fa - fa.mean()) ** 2) + 1e-9
        r2 = 1 - ss_res / ss_tot
        valid = contact and (K_MIN <= k <= K_MAX) and (r2 >= R2_MIN)
        out_K[a].append(k)
        out_valid[a].append(valid)

out_t = np.array(out_t)
out_contact = np.array(out_contact)
n_contact = out_contact.sum()

axis_colors = {"x": AQUA, "y": BLUE, "z": ORANGE}
axis_labels = {"x": "$K_x$", "y": "$K_y$", "z": "$K_z$"}
coverage = {}

fig, ax = plt.subplots(figsize=(7.0, 2.9), constrained_layout=True)
ax.set_facecolor("#fcfcfb")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)

# Contact-phase shading
in_contact = out_contact.astype(int)
edges = np.where(np.diff(in_contact) != 0)[0]
starts = [0] + list(edges + 1)
ends = list(edges + 1) + [len(out_t) - 1]
for s_i, e_i in zip(starts, ends):
    if in_contact[s_i]:
        ax.axvspan(out_t[s_i], out_t[min(e_i, len(out_t) - 1)], color=CONTACT_SHADE, zorder=0)

for a in "xyz":
    k = np.array(out_K[a])
    valid = np.array(out_valid[a])
    coverage[a] = valid.sum() / n_contact
    k_plot = np.where(valid, k, np.nan)
    ax.plot(out_t, k_plot, color=axis_colors[a], linewidth=1.6,
            label=f"{axis_labels[a]} ({coverage[a]*100:.0f}% cov.)", zorder=3)

ax.set_yscale("log")
ax.set_xlabel("time (s)", fontsize=8.5, color=SECONDARY)
ax.set_ylabel("stiffness $K$ (N/m, log scale)", fontsize=8.5, color=SECONDARY)
ax.tick_params(colors=MUTED, labelsize=8)
ax.grid(True, which="major", axis="y", linewidth=0.6, color=GRID, zorder=1)
ax.legend(loc="upper right", fontsize=7.8, frameon=False, ncol=3)
ax.set_title("Extracted stiffness trace -- T1 pilot demo, operator A, \"gently\" (base-frame, simplified)",
              fontsize=9.0, color=PRIMARY, pad=8)

# Legend swatch for contact shading
ax.axvspan(np.nan, np.nan, color=CONTACT_SHADE, label="_nolegend_")
handles, labels = ax.get_legend_handles_labels()
from matplotlib.patches import Patch
handles.append(Patch(facecolor=CONTACT_SHADE, edgecolor="none", label="in contact"))
ax.legend(handles=handles, loc="upper right", fontsize=7.2, frameon=False, ncol=4)

fig.savefig("reports/figures/fig3b_stiffness_trace.pdf")
fig.savefig("reports/figures/fig3b_stiffness_trace.png", dpi=220)
print("wrote fig3b_stiffness_trace.{pdf,png}")
print("per-axis coverage-within-contact (this single demo, proxy mask):", coverage)
