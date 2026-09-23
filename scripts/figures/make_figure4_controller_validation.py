"""Figure 4: commanded vs. realized stiffness + energy-tank trace, REAL hardware.

Source data: ../analysis/franka_teleop_impedance_probe/sinusoid_probe_1786449651.csv
This is the real-hardware confirmation run from 11 Aug (12:00 session, follower at
192.168.101.2, per tasks.md's Day 5 update) -- launched *after* the vetoed-limiter
reset fix was on disk, and *after* the duplicate joint_state_broadcaster spawn bug
was fixed, so this is the clean run, not the one that raced the pre-fix code.

NOTE ON THE SOURCE PNG (already in analysis/): its title hardcodes
"(fake hardware -- wiring/numerics check, not physical dynamics)" even on this
real-hardware run -- that text is a copy-paste artifact in the probe script's
plotting code (the fake-hardware validation run, sinusoid_probe_1786448063.csv,
genuinely is fake hardware and correctly shows the pre-fix flatline bug; this
run is real hardware and the title is simply wrong for it). Do not carry that
title into the paper. This script re-derives the figure from the raw CSV with
an accurate label instead of copying the source PNG.
"""
import pandas as pd
import matplotlib.pyplot as plt

BLUE = "#2a78d6"
ORANGE = "#eb6834"
PRIMARY = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"

df = pd.read_csv("../analysis/franka_teleop_impedance_probe/sinusoid_probe_1786449651.csv")
t = df["t"].values

axes_spec = [
    ("x", "$K_x$ (N/m)"), ("y", "$K_y$ (N/m)"), ("z", "$K_z$ (N/m)"),
    ("rx", "$K_{rx}$ (Nm/rad)"), ("ry", "$K_{ry}$ (Nm/rad)"), ("rz", "$K_{rz}$ (Nm/rad)"),
]

fig = plt.figure(figsize=(7.1, 4.3), constrained_layout=True)
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.8])

for i, (a, ylabel) in enumerate(axes_spec):
    r, c = divmod(i, 3)
    ax = fig.add_subplot(gs[r, c])
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.plot(t, df[f"k_target_{a}"], color=BLUE, linewidth=1.1, label="commanded")
    ax.plot(t, df[f"k_applied_{a}"], color=ORANGE, linewidth=1.1, label="realized")
    ax.set_title(ylabel, fontsize=8.5, color=PRIMARY, pad=4)
    ax.tick_params(colors=MUTED, labelsize=6.5)
    if r == 1:
        ax.set_xlabel("t (s)", fontsize=7.5, color=SECONDARY)
    if i == 0:
        ax.legend(fontsize=6.5, frameon=False, loc="upper right")

ax_e = fig.add_subplot(gs[2, :])
ax_e.set_facecolor("#fcfcfb")
for s in ("top", "right"):
    ax_e.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax_e.spines[s].set_color(GRID)
ax_e.plot(t, df["energy_tank_j"], color=PRIMARY, linewidth=1.3)
ax_e.set_title("energy tank (J)", fontsize=8.5, color=PRIMARY, pad=4)
ax_e.set_xlabel("t (s)", fontsize=7.5, color=SECONDARY)
ax_e.tick_params(colors=MUTED, labelsize=6.5)

fig.suptitle("Commanded vs. realized stiffness, real hardware (20s sinusoidal probe)",
             fontsize=9.5, color=PRIMARY)

fig.savefig("reports/figures/fig4_controller_validation.pdf")
fig.savefig("reports/figures/fig4_controller_validation.png", dpi=220)
print("wrote fig4_controller_validation.{pdf,png}")
