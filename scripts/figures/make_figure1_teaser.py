"""Figure 1 (teaser): the equilibrium/stiffness identifiability ambiguity,
extended to bridge into the paper's headline VLA finding.

Pure illustration, no experimental data required -- draws the two-unknowns
argument from proposal Sec. 2.2(b)/4.1 directly: f = K * e has one equation and
two unknowns (K, e = x_eq - x_f) per axis. Left panel: pose-only teleop observes
a single force f* and cannot separate K from e (every point on the e=f*/K curve
is equally consistent). Middle panel: bilateral teleop additionally measures e
directly via the leader pose x_l, which intersects the same curve at exactly one
point, making K identifiable by regression.

Right panel (added 2026-09-11, per request to connect the mechanism to the
paper's headline VLA result rather than leaving it as an abstract
identifiability argument only): a schematic bridging identifiable labels to
the downstream policy result -- B0 (standard VLA, position-only, fixed
stiffness) vs. B5 (ours, + compliance head, predicts K(t)). Deliberately
qualitative (an ordinal "B5 > B0" arrow, not a bar chart with invented
heights): GATE 3 (B5 >= B0, in-distribution T1 pilot) is a real, confirmed
result, but the full E1 evaluation numbers are not finalized yet, and a bar
chart with specific heights would assert a magnitude nothing here actually
measures. Do not add numbers to this panel until the real E1 table exists --
swap the "[final n / CI pending]" note for the real numbers then, not before.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Reference palette (dataviz skill, references/palette.md) -- unmodified default
# categorical order; slots 1/2 are the adjacent-pair-validated choice.
BLUE = "#2a78d6"      # categorical slot 1 -- the resolved/identifiable line
MUTED = "#898781"     # muted ink -- ambiguous candidates / B0
SECONDARY = "#52514e"
PRIMARY = "#0b0b0b"
GRID = "#e1e0d9"
ORANGE = "#eb6834"    # categorical slot 2 -- the measured-e reference line
SURFACE = "#fcfcfb"

f_star = 10.0  # representative observed force (N), arbitrary for illustration
K = np.linspace(20, 400, 400)
e = f_star / K

fig, axes = plt.subplots(
    1, 3, figsize=(10.2, 2.9), constrained_layout=True,
    gridspec_kw={"width_ratios": [1.0, 1.0, 0.9]},
)

for ax in axes[:2]:
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(True, linewidth=0.6, color=GRID, zorder=0)
    ax.set_xlabel("stiffness $K$ (N/m)", color=SECONDARY, fontsize=9)
    ax.set_ylabel(r"equilibrium offset $e = x_{eq}-x_f$ (mm)", color=SECONDARY, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlim(20, 400)
    ax.set_ylim(0, e.max() * 1000 * 1.15)

# --- Panel 1: pose-only teleoperation ---
ax = axes[0]
ax.plot(K, e * 1000, color=MUTED, linewidth=2.0, zorder=2)
sample_idx = np.linspace(20, 380, 6).astype(int)
for i in sample_idx:
    ax.plot(K[i], e[i] * 1000, "o", color=MUTED, markersize=6, zorder=3,
            markeredgecolor=SURFACE, markeredgewidth=1.0)
ax.annotate("every point equally\nconsistent with the\nsingle observed $f^\\ast$",
            xy=(K[sample_idx[2]], e[sample_idx[2]] * 1000), xytext=(180, 380),
            fontsize=8, color=SECONDARY,
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0))
ax.set_title("Pose-only teleop (VR / SpaceMouse / UMI)\n$K$ and $x_{eq}$ unidentifiable",
             fontsize=9.5, color=PRIMARY, pad=8)

# --- Panel 2: bilateral teleoperation ---
ax = axes[1]
ax.plot(K, e * 1000, color=MUTED, linewidth=1.4, alpha=0.5, zorder=2)
e_measured = 0.033  # illustrative measured pose error via x_l, meters
K_true = f_star / e_measured
ax.axhline(e_measured * 1000, color=ORANGE, linestyle="--", linewidth=1.4, zorder=2)
ax.annotate(r"$e$ measured directly via leader pose $x_l$",
            xy=(340, e_measured * 1000), xytext=(150, e_measured * 1000 + 60),
            fontsize=8, color=ORANGE,
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.0))
ax.plot(K_true, e_measured * 1000, "o", color=BLUE, markersize=9, zorder=4,
        markeredgecolor=SURFACE, markeredgewidth=1.4)
ax.annotate("unique $K$,\nidentifiable\nby regression", xy=(K_true, e_measured * 1000),
            xytext=(K_true - 155, e_measured * 1000 + 130), fontsize=8.5, color=BLUE,
            fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))
ax.set_title("Bilateral teleop (this paper)\n$x_{eq} := x_l$ collapses the ambiguity",
             fontsize=9.5, color=PRIMARY, pad=8)

# --- Panel 3: bridge to the headline VLA result ---
ax = axes[2]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("$\\rightarrow$ compliance as a VLA output",
              fontsize=9.5, color=PRIMARY, pad=8)


def box(xy, w, h, text, ec, fc, fontsize=8.0, textcolor=PRIMARY, lw=1.3, fontweight="normal"):
    x, y = xy
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12,rounding_size=0.15",
                            fc=fc, ec=ec, lw=lw, zorder=3)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color=textcolor, zorder=4, fontweight=fontweight)


box((0.6, 6.0), 8.8, 2.6, "B0: standard VLA\nposition output, fixed $K$",
    ec=MUTED, fc="#f2f1ee", fontsize=8.6)
box((0.6, 1.2), 8.8, 2.6, "B5 (ours): + compliance head\npredicts $K(t)$",
    ec=BLUE, fc="#eef4fc", fontsize=8.6, textcolor=BLUE, fontweight="bold")

arrow = FancyArrowPatch((5.0, 5.9), (5.0, 3.9), arrowstyle="-|>", color=BLUE,
                         lw=1.6, mutation_scale=14, zorder=3)
ax.add_patch(arrow)
ax.text(5.55, 4.9, "outperforms\non real-robot\nrollouts", ha="left", va="center",
        fontsize=7.6, color=BLUE, fontweight="bold", zorder=4)

ax.text(5.0, 0.35, "in-distribution pilot, GATE 3 (B5 $\\geq$ B0);\n"
                    "[final $n$ / CI pending full E1 evaluation]",
        ha="center", va="center", fontsize=6.6, color=MUTED,
        style="italic", zorder=4)

fig.savefig("reports/figures/fig1_teaser.pdf")
fig.savefig("reports/figures/fig1_teaser.png", dpi=220)
print("wrote reports/figures/fig1_teaser.{pdf,png}")
