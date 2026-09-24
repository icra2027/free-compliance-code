"""Figure 2: hardware rig + policy architecture diagram.

Pure illustration, no experimental data required -- every element here is a
fixed design choice of the method and the
real implementation (src/compliance_vla/policy/compliance_policy.py, src/compliance_vla/policy/force_encoder.py).
Nothing here is a placeholder.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
PRIMARY = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"


def box(ax, xy, w, h, text, fc="white", ec=PRIMARY, fontsize=8.0, textcolor=PRIMARY,
        lw=1.1, zorder=3):
    x, y = xy
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                            fc=fc, ec=ec, lw=lw, zorder=zorder)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color=textcolor, zorder=zorder + 1, wrap=True)
    return patch


def arrow(ax, p0, p1, color=PRIMARY, style="-|>", lw=1.2, connectionstyle=None, zorder=2):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, color=color, lw=lw,
                         mutation_scale=10, connectionstyle=connectionstyle, zorder=zorder)
    ax.add_patch(a)


fig = plt.figure(figsize=(7.1, 3.4), constrained_layout=True)
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15])
axL = fig.add_subplot(gs[0, 0])
axR = fig.add_subplot(gs[0, 1])
for ax in (axL, axR):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_facecolor(SURFACE)

# ---------------- Left panel: bilateral hardware rig ----------------
axL.set_title("(a) Bilateral teleoperation rig", fontsize=9.5, color=PRIMARY, pad=6)

leader = box(axL, (0.4, 7.2), 3.0, 1.6, "Leader\nFranka\n(operator)", fc="#eef4fc", ec=BLUE)
follower = box(axL, (6.6, 7.2), 3.0, 1.6, "Follower\nFranka\n+ rigid wiper", fc="#eef4fc", ec=BLUE)
arrow(axL, (3.4, 8.2), (6.6, 8.2), color=BLUE, style="<|-|>", lw=1.6)
axL.text(5.0, 8.55, "4-channel\nbilateral coupling", ha="center", fontsize=7.2, color=BLUE)
axL.text(5.0, 7.75, r"$x_l(t),\, \dot x_l(t)$   $\leftrightarrow$   $x_f(t),\, f(t)$",
         ha="center", fontsize=6.8, color=SECONDARY)

wrist_cam = box(axL, (6.8, 5.5), 2.6, 1.0, "wrist RGB\n(D435I)", fc="white", ec=MUTED, fontsize=7)
scene_cam = box(axL, (0.6, 5.5), 2.6, 1.0, "scene RGB\n(Femto Bolt)", fc="white", ec=MUTED, fontsize=7)
board = box(axL, (3.6, 4.0), 2.8, 2.0,
            "whiteboard, 20°\n4 marks (R/G/B/K)\nrandomized layout", fc="#fbf7ee", ec=MUTED, fontsize=7.2)
arrow(axL, (8.1, 7.2), (8.1, 6.5), color=MUTED, lw=1.0)
arrow(axL, (1.9, 7.2), (1.9, 6.5), color=MUTED, lw=1.0)
arrow(axL, (8.0, 5.5), (5.6, 5.0), color=MUTED, lw=1.0, connectionstyle="arc3,rad=-0.2")
arrow(axL, (2.0, 5.5), (4.4, 5.0), color=MUTED, lw=1.0, connectionstyle="arc3,rad=0.2")

controller = box(axL, (3.4, 1.3), 3.2, 1.6,
                  "1kHz Cartesian impedance\n$\\tau = J^T[K_t(x_{eq}-x)+D_t(\\dot x_{eq}-\\dot x)]$\n"
                  "+ log-space rate limit + energy tank",
                  fc="#fff5ee", ec=ORANGE, fontsize=6.6)
arrow(axL, (8.1, 7.2), (6.2, 2.9), color=ORANGE, lw=1.2, connectionstyle="arc3,rad=0.25")
axL.text(7.5, 4.4, "target pose\n+ target $K(t)$", ha="center", fontsize=6.5, color=ORANGE)

# ---------------- Right panel: policy architecture ----------------
axR.set_title("(b) Compliance-VLA policy (B5)", fontsize=9.5, color=PRIMARY, pad=6)

vlm = box(axR, (0.4, 7.6), 4.6, 1.4,
          "frozen SigLIP + SmolLM2\nvision-language backbone\n(pretrained, not fine-tuned)",
          fc="#eef4fc", ec=BLUE, fontsize=7.2)
lang = box(axR, (5.4, 7.6), 4.2, 1.4, '"wipe the {colour}\nmark {manner}"', fc="white",
           ec=MUTED, fontsize=7.5)
arrow(axR, (5.4, 8.3), (5.0, 8.3), color=MUTED, lw=1.0)

prefix = box(axR, (0.4, 6.0), 9.2, 0.9, "shared prefix embedding (image + language tokens)",
             fc="white", ec=MUTED, fontsize=7.2)
arrow(axR, (2.7, 7.6), (2.7, 6.9), color=BLUE, lw=1.2)

force_tok = box(axR, (0.4, 3.9), 3.0, 1.4,
                "force-history token\n500ms wrench -> 20 samples\n1D-conv encoder",
                fc="#fff5ee", ec=ORANGE, fontsize=6.8)
proprio = box(axR, (3.6, 3.9), 2.7, 1.4, r"proprioception" + "\n" + r"$(q,\dot q, x_f)$",
              fc="white", ec=MUTED, fontsize=7.2)
expert = box(axR, (6.5, 3.7), 3.1, 1.8,
             "action-expert\ntransformer\n(flow matching)", fc="#eef4fc", ec=BLUE, fontsize=7.5)

arrow(axR, (5.0, 5.95), (8.0, 5.65), color=MUTED, lw=1.1, connectionstyle="arc3,rad=-0.15")
axR.text(6.9, 5.42, "post-VLM injection\n(ForceVLA-style)", fontsize=6.3, color=ORANGE, ha="center",
          bbox=dict(boxstyle="round,pad=0.15", fc="#fcfcfb", ec="none"))
arrow(axR, (1.9, 3.9), (7.2, 3.9), color=ORANGE, lw=1.1, connectionstyle="arc3,rad=0.35")
arrow(axR, (4.9, 3.9), (7.5, 3.7), color=MUTED, lw=1.0, connectionstyle="arc3,rad=0.2")

out = box(axR, (6.9, 1.2), 2.4, 1.6,
          "$x_{eq}(6)$\n$\\log K(6)$\n$\\text{gripper}(1)$\n(H=32 @ 30Hz)",
          fc="#fbf7ee", ec=PRIMARY, fontsize=7.2)
arrow(axR, (8.0, 3.7), (8.0, 2.8), color=PRIMARY, lw=1.3)

legend_elems = [
    Line2D([0], [0], color=BLUE, lw=2, label="pretrained / VLM path"),
    Line2D([0], [0], color=ORANGE, lw=2, label="force / compliance path"),
    Line2D([0], [0], color=MUTED, lw=2, label="shared / other"),
]
fig.legend(handles=legend_elems, loc="lower center", ncol=3, fontsize=7.5,
           frameon=False, bbox_to_anchor=(0.5, -0.03))

fig.savefig("reports/figures/fig2_architecture.pdf", bbox_inches="tight")
fig.savefig("reports/figures/fig2_architecture.png", dpi=220, bbox_inches="tight")
print("wrote reports/figures/fig2_architecture.{pdf,png}")
