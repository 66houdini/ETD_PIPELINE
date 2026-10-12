# -*- coding: utf-8 -*-
"""Two additional report figures: ML taxonomy (Ch2) and stacking architecture (Ch3)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

OUT = r"C:\Users\HP\Desktop\final-project\figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.size': 9})


def box(ax, x, y, w, h, text, fc='#dae8fc', ec='#4a6f9c', fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                facecolor=fc, edgecolor=ec, linewidth=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, fontweight='bold' if bold else 'normal', linespacing=1.35)


def arrow(ax, p1, p2, color='#444444', lw=1.2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle='-|>', mutation_scale=10,
                                 color=color, linewidth=lw, shrinkA=1, shrinkB=1))


def elbow(ax, x0, y0, x1, y1, color='#666666', lw=1.1):
    ax.plot([x0, x0], [y0, (y0 + y1) / 2], color=color, lw=lw)
    ax.plot([x0, x1], [(y0 + y1) / 2, (y0 + y1) / 2], color=color, lw=lw)
    ax.plot([x1, x1], [(y0 + y1) / 2, y1], color=color, lw=lw)


# ── ML taxonomy ────────────────────────────────────────────────
def fig_taxonomy():
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis('off')

    box(ax, 36, 54, 28, 6.5, "Techniques for Electricity Theft Detection",
        '#f8cecc', '#b85450', fs=10, bold=True)
    box(ax, 40, 43, 20, 6, "Data-Driven (ML)", '#ffe6cc', '#d79b00', fs=9.5, bold=True)
    arrow(ax, (50, 54), (50, 49))

    cols = [
        (1.5, "Supervised\nLearning",
         "Decision Tree\nK-Nearest Neighbours\nSupport Vector Machine\n"
         "Logistic Regression", '#dae8fc', '#4a6f9c'),
        (21, "Unsupervised\nLearning",
         "K-Means clustering\nFuzzy C-Means\nPCA\nGaussian Mixture\nObserver meters",
         '#d5e8d4', '#82b366'),
        (40.5, "Ensemble\nMethods",
         "Random Forest\nExtraTrees (bagging)\nXGBoost / CatBoost\n"
         "Stacked generalisation", '#fff2cc', '#d6b656'),
        (60, "Deep\nLearning",
         "CNN (1D / 2D)\nRNN / LSTM\nDeep NN\nAutoencoders", '#e1d5e7', '#9673a6'),
        (79.5, "Reinforcement\nLearning",
         "Q-Learning\nDeep Q-Network\nDouble DQN", '#f8cecc', '#b85450'),
    ]
    for x, head, leaves, fc, ec in cols:
        box(ax, x, 30, 19, 8, head, fc, ec, fs=9, bold=True)
        elbow(ax, 50, 43, x + 9.5, 38)
        box(ax, x, 8, 19, 19, leaves, '#fbfbfb', '#aaaaaa', fs=8)
        arrow(ax, (x + 9.5, 30), (x + 9.5, 27))

    ax.add_patch(FancyBboxPatch((30.5, 0.5), 39, 5, boxstyle="round,pad=0.02",
                                facecolor='#f5f5f5', edgecolor='#999999',
                                linewidth=1.1, linestyle='--'))
    ax.text(50, 3, "Adopted in this work: ensemble methods (stacked generalisation)\n"
                   "with a deep-learning baseline for comparison",
            ha='center', va='center', fontsize=8.5, style='italic', color='#555555')
    ax.set_title("Taxonomy of Machine Learning Techniques in Electricity Theft Detection",
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_taxonomy.png", dpi=220, bbox_inches='tight')
    plt.close(fig)


# ── Stacked ensemble architecture ──────────────────────────────
def fig_stacking():
    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 70); ax.axis('off')

    box(ax, 25, 62, 50, 6, "Engineered feature matrix  X  (n × 173)",
        '#f8cecc', '#b85450', fs=9.5, bold=True)

    ax.text(50, 57.8, "STAGE 1 — Base classifiers, 5-fold stratified cross-validation",
            ha='center', fontsize=9, fontweight='bold', color='#2b4c7e', zorder=5,
            bbox=dict(facecolor='white', edgecolor='none', pad=2.5))

    names = [("XGBoost", 4), ("Random Forest", 36.5), ("MLP", 69)]
    for nm, x in names:
        box(ax, x, 44, 27, 10,
            f"{nm}\n\nfit on 4 folds → predict\nheld-out fold", '#dae8fc', '#4a6f9c', fs=8.5)
        arrow(ax, (50, 62), (x + 13.5, 54))
        arrow(ax, (x + 13.5, 44), (x + 13.5, 38))
        box(ax, x + 3.5, 31, 20, 6, "OOF probabilities\n(n × 1)", '#eef4fb', '#7a9cc0', fs=8)

    box(ax, 25, 20, 50, 7,
        "Out-of-fold probability matrix  Z  (n × 3)", '#fff2cc', '#d6b656',
        fs=9.5, bold=True)
    for _, x in names:
        arrow(ax, (x + 13.5, 31), (50, 27))

    ax.text(50, 16.8, "STAGE 2 — Meta-learner", ha='center', fontsize=9,
            fontweight='bold', color='#2b4c7e', zorder=5,
            bbox=dict(facecolor='white', edgecolor='none', pad=2.5))
    box(ax, 22, 8, 56, 7,
        "Logistic Regression:  p = σ(β₀ + β₁z₁ + β₂z₂ + β₃z₃)",
        '#d5e8d4', '#82b366', fs=9.5, bold=True)
    arrow(ax, (50, 20), (50, 15))
    box(ax, 28, 0.5, 44, 5.5,
        "Theft probability → decision threshold → NORMAL / THEFT",
        '#e1d5e7', '#9673a6', fs=9)
    arrow(ax, (50, 8), (50, 6))

    ax.text(80.5, 11.5, "coefficients β are\ndirectly interpretable",
            fontsize=8, style='italic', color='#4a7a3a', ha='left', va='center')

    ax.set_title("Two-Stage Stacked Ensemble Architecture",
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_stacking.png", dpi=220, bbox_inches='tight')
    plt.close(fig)


for fn in (fig_taxonomy, fig_stacking):
    fn(); print(f"  {fn.__name__} done")
print(f"Extra figures written to {OUT}")
