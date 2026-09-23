"""Figure 2: end-to-end labeling pipeline (Fallback A).

Pure illustration, no experimental data required -- every stage here is a
fixed design choice already committed in methodology.tex (Secs.
4.3/4.4/4.5): bilateral rig -> sensorless wrench estimation (payload ID ->
re-zero -> bias model) -> extraction (windowed regression + identifiability
mask) -> labeled dataset. Replaces the old two-panel rig+policy figure
(make_figure2_architecture.py) now that Fallback A drops the policy panel.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
PRIMARY = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
SURFACE = "#fcfcfb"


def box(ax, xy, w, h, text, fc="white", ec=PRIMARY, fontsize=7.6, textcolor=PRIMARY,
        lw=1.2, zorder=3):
    x, y = xy
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.0",
                            fc=fc, ec=ec, lw=lw, zorder=zorder)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color=textcolor, zorder=zorder + 1)
    return patch


def chip(ax, xy, w, h, text, ec=MUTED, fontsize=6.4):
    return box(ax, xy, w, h, text, fc="white", ec=ec, fontsize=fontsize, lw=0.9, zorder=4)


def arrow(ax, p0, p1, color=PRIMARY, lw=1.4, style="-|>", mutation_scale=11, zorder=2):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, color=color, lw=lw,
                         mutation_scale=mutation_scale, zorder=zorder)
    ax.add_patch(a)


fig, ax = plt.subplots(figsize=(7.1, 2.5), constrained_layout=True)
ax.set_xlim(0, 100)
ax.set_ylim(0, 32)
ax.axis("off")
ax.set_facecolor(SURFACE)

# ---------------- Stage 1: bilateral rig ----------------
box(ax, (1, 11), 15, 12, "Bilateral\nteleoperation\nrig", fc="#eef4fc", ec=BLUE, fontsize=8.2)

# ---------------- Stage 2: sensorless wrench estimation ----------------
box(ax, (20, 2), 26, 26, "", fc="none", ec=ORANGE, lw=1.2)
ax.text(33, 25.6, "Sensorless wrench estimation", ha="center", fontsize=7.4,
        color=ORANGE, weight="bold")
chip(ax, (21.5, 15.5), 9.5, 6.5, "payload ID\n(Eq. 6)")
chip(ax, (33.0, 15.5), 9.5, 6.5, "re-zero")
chip(ax, (27, 5.5), 15, 6.5, "bias model\n(RFF ridge, Eq. 8)")
arrow(ax, (31.2, 18.75), (32.8, 18.75), color=MUTED, lw=1.1, mutation_scale=8)  # payload ID -> re-zero
arrow(ax, (37.75, 15.5), (34.5, 12), color=MUTED, lw=1.1, mutation_scale=8)  # re-zero -> bias model

# ---------------- Stage 3: extraction ----------------
box(ax, (50, 2), 26, 26, "", fc="none", ec=AQUA, lw=1.2)
ax.text(63, 25.6, "Extraction", ha="center", fontsize=7.4, color=AQUA, weight="bold")
chip(ax, (51.5, 15.5), 23, 7.5, "windowed regression\n$(k_i, d_i)$, Eq. 5")
chip(ax, (51.5, 5.5), 23, 7.5, "identifiability mask\n$m_i(t)$, Eq. 9")
arrow(ax, (63, 15.5), (63, 13), color=MUTED, lw=1.0)

# ---------------- Stage 4: labeled dataset ----------------
box(ax, (80, 11), 19, 12, "Labeled\ndataset\n$K(t), D(t), m(t)$", fc="#fbf7ee", ec=PRIMARY,
    fontsize=7.8)

# ---------------- inter-stage arrows + signal labels ----------------
arrow(ax, (16, 17), (19.5, 17), color=PRIMARY, lw=1.6)
ax.text(17.7, 19.2, r"$x_l, x_f,$" + "\n" + r"$q,\dot q,\tau_{\mathrm{meas}}$",
        ha="center", fontsize=6.0, color=SECONDARY)

arrow(ax, (46.3, 17), (49.5, 17), color=PRIMARY, lw=1.6)
ax.text(48, 19.4, r"$\hat f(t)$", ha="center", fontsize=6.6, color=SECONDARY)

arrow(ax, (76.3, 17), (79.5, 17), color=PRIMARY, lw=1.6)
ax.text(78, 20.2, r"$K(t), D(t),$" + "\n" + r"$m(t)$", ha="center", fontsize=5.6,
        color=SECONDARY)

fig.savefig("reports/figures/fig2_pipeline.pdf", bbox_inches="tight")
fig.savefig("reports/figures/fig2_pipeline.png", dpi=220, bbox_inches="tight")
print("wrote reports/figures/fig2_pipeline.{pdf,png}")
