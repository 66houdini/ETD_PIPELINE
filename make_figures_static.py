# -*- coding: utf-8 -*-
"""
Static report figures: circuit schematic, block diagram, flowcharts, and the
hardware GAP timeline (from real measured bench data).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np
import os

OUT = r"C:\Users\HP\Desktop\final-project\figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.size': 9})


def box(ax, x, y, w, h, text, fc='#dae8fc', ec='#4a6f9c', fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                facecolor=fc, edgecolor=ec, linewidth=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, fontweight='bold' if bold else 'normal', linespacing=1.4)


def arrow(ax, p1, p2, color='#333333', lw=1.4, style='-|>'):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=11,
                                 color=color, linewidth=lw,
                                 shrinkA=1, shrinkB=1))


def wire(ax, pts, color='#b85450', lw=2.2):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, linewidth=lw, solid_capstyle='round', zorder=1)


# ── FIG 1: Power circuit schematic ─────────────────────────────
def fig_circuit():
    fig, ax = plt.subplots(figsize=(9, 6.2))
    ax.set_xlim(-3, 100); ax.set_ylim(0, 72); ax.axis('off')

    box(ax, 3, 58, 17, 9, "AC MAINS\n220–240 V, 50 Hz", '#f8cecc', '#b85450', bold=True)
    box(ax, 26, 58, 10, 9, "FUSE\n4 A", '#ffe6cc', '#d79b00')
    box(ax, 42, 57, 18, 11, "PZEM-A\nPoint A\n(total incoming)", '#dae8fc', '#4a6f9c', bold=True)
    ax.plot(72, 62.5, 'o', ms=9, color='#82b366', zorder=3)
    ax.text(72, 66.5, "Node X", ha='center', fontsize=8.5, style='italic')

    # Live path: mains -> fuse -> PZEM-A -> Node X
    wire(ax, [(20, 62.5), (26, 62.5)])
    wire(ax, [(36, 62.5), (42, 62.5)])
    wire(ax, [(60, 62.5), (72, 62.5)])

    # Distribution bus BELOW PZEM-A, so the split is unambiguously at Node X
    wire(ax, [(72, 62.5), (72, 52)], color='#8a6d9c')
    wire(ax, [(38, 52), (84, 52)], color='#8a6d9c')
    wire(ax, [(38, 52), (38, 46)], color='#82b366')
    wire(ax, [(84, 52), (84, 46)], color='#b85450')

    ax.text(38, 48.4, "METERED BRANCH (legitimate)", fontsize=8,
            fontweight='bold', color='#4a7a3a', ha='center')
    ax.text(84, 48.4, "BYPASS BRANCH (theft)", fontsize=8,
            fontweight='bold', color='#b85450', ha='center')

    box(ax, 29, 35, 18, 11, "PZEM-B\nPoint B\n(metered load)", '#dae8fc', '#4a6f9c', bold=True)
    box(ax, 75, 35, 18, 11, "SWITCH S1\n(engages theft)", '#f8cecc', '#b85450')

    wire(ax, [(38, 35), (38, 27)], color='#82b366')
    wire(ax, [(84, 35), (84, 27)], color='#b85450')

    ax.add_patch(Circle((38, 21), 6, facecolor='#fff2cc', edgecolor='#d6b656', lw=1.4))
    ax.text(38, 21, "BULB 1\n10 W", ha='center', va='center', fontsize=8)
    ax.add_patch(Circle((84, 21), 6, facecolor='#f8cecc', edgecolor='#b85450', lw=1.4))
    ax.text(84, 21, "BULB 2\n10 W", ha='center', va='center', fontsize=8)

    wire(ax, [(38, 15), (38, 9)], color='#9673a6')
    wire(ax, [(84, 15), (84, 9)], color='#9673a6')
    box(ax, 24, 5, 74, 4.5, "SHARED NEUTRAL BUS", '#e1d5e7', '#9673a6', bold=True)
    # Neutral return runs down the far left, clear of PZEM-C
    wire(ax, [(24, 7.2), (-1.5, 7.2), (-1.5, 62.5), (3, 62.5)], color='#9673a6')

    # PZEM-C (bypassed)
    ax.add_patch(FancyBboxPatch((4, 16), 17, 12, boxstyle="round,pad=0.02",
                                facecolor='#f5f5f5', edgecolor='#999999',
                                linewidth=1.2, linestyle='--'))
    ax.text(12.5, 22, "PZEM-C (Point C)\nBYPASSED on mains side\nreads 0.00 W",
            ha='center', va='center', fontsize=7.5, color='#666666')

    ax.text(50, 0.5,
            "GAP = P$_A$ − P$_B$    |    S1 open: P$_A$≈P$_B$, GAP≈0 W (normal)"
            "    |    S1 closed: GAP≈11 W (THEFT, alarm at >10.0 W)",
            ha='center', fontsize=8.5, style='italic')
    ax.set_title("Energy-Balance Theft Detection Circuit", fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_circuit_schematic.png", dpi=220, bbox_inches='tight')
    plt.close(fig)


# ── FIG 2: ML pipeline flowchart ───────────────────────────────
def fig_ml_flow():
    fig, ax = plt.subplots(figsize=(5.2, 9))
    ax.set_xlim(0, 40); ax.set_ylim(0, 100); ax.axis('off')
    steps = [
        ("SGCC dataset\n42,372 customers × 1,034 days", '#f8cecc'),
        ("Stratified 80/20 split\n(RAW data, before preprocessing)", '#ffe6cc'),
        ("Row-wise median imputation\n(fit on TRAIN only)", '#dae8fc'),
        ("Feature engineering (173)\nstatistical • theft-specific •\nFFT • calendar • weekly", '#dae8fc'),
        ("StandardScaler\n(fit on TRAIN only)", '#dae8fc'),
        ("Class imbalance handling\nSMOTETomek  vs  class weighting", '#fff2cc'),
        ("STAGE 1 — Base classifiers\nXGBoost | Random Forest | MLP\n5-fold OOF probabilities", '#d5e8d4'),
        ("STAGE 2 — Meta-learner\nLogistic Regression", '#d5e8d4'),
        ("Evaluation on untouched test set\nAUC • FPR • F1 • TPR", '#e1d5e7'),
        ("Threshold selection\n+ SHAP interpretability", '#e1d5e7'),
    ]
    h, gap = 7.4, 2.6
    y = 100 - h
    for i, (txt, c) in enumerate(steps):
        box(ax, 3, y, 34, h, txt, c, '#555555', fs=8)
        if i < len(steps) - 1:
            arrow(ax, (20, y), (20, y - gap))
        y -= (h + gap)
    ax.set_title("Machine-Learning Pipeline", fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_ml_flowchart.png", dpi=220, bbox_inches='tight')
    plt.close(fig)


# ── FIG 3: Arduino firmware flowchart ──────────────────────────
def fig_fw_flow():
    fig, ax = plt.subplots(figsize=(5.4, 7.6))
    ax.set_xlim(0, 40); ax.set_ylim(0, 100); ax.axis('off')
    box(ax, 8, 91, 24, 6.5, "START / setup()", '#d5e8d4', '#82b366', bold=True)
    box(ax, 4, 80, 32, 8, "Init Serial(115200), Serial1/2/3(9600)\nInit PZEMs, LCD, buzzer, LED", '#dae8fc', '#4a6f9c')
    box(ax, 6, 69, 28, 7, "Read P$_A$, P$_B$, P$_C$ (Modbus)", '#dae8fc', '#4a6f9c')
    box(ax, 6, 59, 28, 6.5, "NaN guard → 0.0", '#ffe6cc', '#d79b00')
    box(ax, 6, 49, 28, 6.5, "GAP = P$_A$ − P$_B$", '#fff2cc', '#d6b656', bold=True)

    # decision diamond
    ax.add_patch(plt.Polygon([[20, 44], [34, 36], [20, 28], [6, 36]],
                             facecolor='#f8cecc', edgecolor='#b85450', lw=1.3))
    ax.text(20, 36, "GAP > 10.0 W ?", ha='center', va='center', fontsize=8.5,
            fontweight='bold')

    box(ax, 1, 17, 17, 7, "Buzzer ON\nLED ON\n\"BYPASS\"", '#f8cecc', '#b85450')
    box(ax, 22, 17, 17, 7, "Buzzer OFF\nLED OFF\n\"normal\"", '#d5e8d4', '#82b366')
    box(ax, 5, 6, 30, 7, "Print A/B/C/GAP to serial\nUpdate LCD\ndelay(2000 ms)", '#e1d5e7', '#9673a6')

    arrow(ax, (20, 91), (20, 88)); arrow(ax, (20, 80), (20, 76))
    arrow(ax, (20, 69), (20, 65.5)); arrow(ax, (20, 59), (20, 55.5))
    arrow(ax, (20, 49), (20, 44))
    arrow(ax, (10, 33), (9.5, 24)); ax.text(7, 29, "YES", fontsize=8, color='#b85450')
    arrow(ax, (30, 33), (30.5, 24)); ax.text(31, 29, "NO", fontsize=8, color='#4a7a3a')
    arrow(ax, (9.5, 17), (12, 13)); arrow(ax, (30.5, 17), (28, 13))
    # loop back
    ax.annotate("", xy=(20, 72.5), xytext=(38, 9.5),
                arrowprops=dict(arrowstyle='-|>', color='#666666', lw=1.2,
                                connectionstyle="angle,angleA=0,angleB=90,rad=6"))
    ax.text(38.6, 42, "loop()", fontsize=8, rotation=90, color='#666666',
            va='center', style='italic')
    ax.set_title("Arduino Firmware Logic", fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_firmware_flowchart.png", dpi=220, bbox_inches='tight')
    plt.close(fig)


# ── FIG 4: Hardware GAP timeline (REAL measured bench data) ────
def fig_hw_timeline():
    t = [4.7, 5.4, 7.8, 10.2, 12.6, 14.9, 17.4, 19.8, 22.2, 24.6, 27.0, 29.4]
    A = [11.00, 21.80, 21.70, 21.60, 10.90, 10.90, 10.90, 17.30, 21.70, 21.80, 21.80, 15.20]
    B = [10.80, 10.90, 10.80, 10.60, 11.00, 10.70, 10.70, 10.70, 10.70, 10.70, 10.80, 10.60]
    G = [0.20, 10.90, 10.90, 11.00, -0.10, 0.20, 0.20, 6.60, 11.00, 11.10, 11.00, 4.60]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 6.4), sharex=True,
                                   gridspec_kw={'height_ratios': [1.15, 1]})
    ax1.plot(t, A, 'o-', color='#b85450', lw=2, ms=5, label='Point A (source)')
    ax1.plot(t, B, 's-', color='#4a7a3a', lw=2, ms=5, label='Point B (metered load)')
    ax1.set_ylabel("Real power (W)")
    ax1.legend(loc='center right', framealpha=0.95)
    ax1.grid(alpha=0.3)
    ax1.set_title("Live Bench Measurement — Bypass Switch Toggled", fontweight='bold')
    ax1.annotate("B stays constant:\nthe meter never sees the theft",
                 xy=(10.2, 10.6), xytext=(11.5, 15.5), fontsize=8,
                 arrowprops=dict(arrowstyle='->', color='#4a7a3a'), color='#4a7a3a')

    ax2.plot(t, G, 'o-', color='#2b4c7e', lw=2, ms=5, label='GAP = P$_A$ − P$_B$')
    ax2.axhline(10.0, color='#b85450', ls='--', lw=1.6, label='Threshold (10.0 W)')
    ax2.axhspan(-0.5, 0.5, color='#82b366', alpha=0.18, label='Baseline noise band')
    for x, g in zip(t, G):
        if g > 10.0:
            ax2.plot(x, g, 'o', ms=11, mfc='none', mec='#b85450', mew=1.8)
    ax2.annotate("bulb warm-up:\ntransitional samples\nbelow threshold",
                 xy=(19.8, 6.6), xytext=(14.0, 6.4), fontsize=8,
                 arrowprops=dict(arrowstyle='->', color='#666666'), color='#444444')
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("GAP (W)")
    ax2.legend(loc='center right', fontsize=8, framealpha=0.95)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_hardware_timeline.png", dpi=220, bbox_inches='tight')
    plt.close(fig)


# ── FIG 5: System block diagram ────────────────────────────────
def fig_block():
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 70); ax.axis('off')

    box(ax, 2, 56, 15, 7, "PZEM-A\nPoint A", '#dae8fc', '#4a6f9c')
    box(ax, 2, 47, 15, 7, "PZEM-B\nPoint B", '#dae8fc', '#4a6f9c')
    ax.add_patch(FancyBboxPatch((2, 38), 15, 7, boxstyle="round,pad=0.02",
                                facecolor='#f5f5f5', edgecolor='#999999',
                                linewidth=1.1, linestyle='--'))
    ax.text(9.5, 41.5, "PZEM-C\n(bypassed)", ha='center', va='center',
            fontsize=7.5, color='#666666')

    box(ax, 25, 42, 27, 21,
        "ARDUINO MEGA 2560\n\n• Poll sensors (Modbus, 9600)\n"
        "• GAP = P$_A$ − P$_B$\n• Threshold compare (10.0 W)\n"
        "• Drive LCD / buzzer / LED\n• Stream USB serial (115200)",
        '#d5e8d4', '#82b366', fs=8, bold=False)

    box(ax, 60, 58, 20, 6, "LCD 16×2 (I²C)", '#ffe6cc', '#d79b00', fs=8)
    box(ax, 60, 50.5, 20, 6, "Buzzer (pin 8)", '#ffe6cc', '#d79b00', fs=8)
    box(ax, 60, 43, 20, 6, "LED (pin 9)", '#ffe6cc', '#d79b00', fs=8)
    box(ax, 84, 47, 14, 12, "LAYER 1\nREAL-TIME\nbypass alarm\n(no ML)",
        '#d5e8d4', '#82b366', fs=8, bold=True)

    box(ax, 25, 30, 27, 7, "PC — pyserial listener", '#e1d5e7', '#9673a6', fs=8.5)
    box(ax, 25, 21, 27, 7, "Daily energy accumulation", '#e1d5e7', '#9673a6', fs=8.5)
    box(ax, 25, 12, 27, 7, "Feature engineering (173)", '#dae8fc', '#4a6f9c', fs=8.5)
    box(ax, 20, 2, 37, 8,
        "STACKED ENSEMBLE (etd_model.pkl)\nXGBoost | RandomForest | MLP\n"
        "→ Logistic Regression meta-learner", '#dae8fc', '#4a6f9c', fs=8, bold=True)
    box(ax, 66, 3, 22, 12, "LAYER 2\nACCUMULATED\nPATTERN\nbehavioural\nclassification",
        '#dae8fc', '#4a6f9c', fs=8, bold=True)

    for y in (59.5, 50.5):
        arrow(ax, (17, y), (25, 55 if y > 55 else 50))
    arrow(ax, (17, 41.5), (25, 46), color='#999999')
    for y in (61, 53.5, 46):
        arrow(ax, (52, 55 if y > 58 else (52 if y > 50 else 48)), (60, y))
    arrow(ax, (80, 53.5), (84, 53))
    arrow(ax, (38, 42), (38, 37)); ax.text(39, 39.3, "USB serial", fontsize=7.5)
    arrow(ax, (38, 30), (38, 28)); arrow(ax, (38, 21), (38, 19))
    arrow(ax, (38, 12), (38, 10)); arrow(ax, (57, 6), (66, 9))

    ax.set_title("System Block Diagram — Hardware and Machine-Learning Layers",
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_block_diagram.png", dpi=220, bbox_inches='tight')
    plt.close(fig)


for fn in (fig_circuit, fig_ml_flow, fig_fw_flow, fig_hw_timeline, fig_block):
    fn(); print(f"  {fn.__name__} done")
print(f"Static figures written to {OUT}")
