# -*- coding: utf-8 -*-
"""Report figures generated from the real pipeline artifacts."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, json, os
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, f1_score

BASE = r"C:\Users\HP\Desktop\final-project"
OUT = os.path.join(BASE, "figures")
plt.rcParams.update({'font.family': 'serif', 'font.size': 9})

art = np.load(os.path.join(BASE, "report_artifacts.npz"))
with open(os.path.join(BASE, "report_metrics.json")) as f:
    M = json.load(f)

y = art['y_test']
p_sm, p_wt = art['proba_smote'], art['proba_weights']
names = ['XGBoost', 'RandomForest', 'MLP']

# ── FIG: ROC curves ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
for ax, (proba, base, title) in zip(axes, [
        (p_sm, art['base_test_smote'], 'A: SMOTETomek'),
        (p_wt, art['base_test_weights'], 'B: Class-weighted')]):
    for i, n in enumerate(names):
        fpr, tpr, _ = roc_curve(y, base[:, i])
        ax.plot(fpr, tpr, lw=1.2, alpha=0.8,
                label=f"{n} ({roc_auc_score(y, base[:, i]):.3f})")
    fpr, tpr, _ = roc_curve(y, proba)
    ax.plot(fpr, tpr, lw=2.6, color='#2b4c7e',
            label=f"Stacked ensemble ({roc_auc_score(y, proba):.3f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=0.9, alpha=0.5)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontweight='bold'); ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.3)
fig.suptitle("ROC Curves on the Untouched Test Set (n = %d)" % len(y),
             fontweight='bold')
fig.tight_layout()
fig.savefig(f"{OUT}/fig_roc_curves.png", dpi=220, bbox_inches='tight'); plt.close(fig)

# ── FIG: OOF vs TEST gap (the key finding) ────────────────────
fig, ax = plt.subplots(figsize=(7.6, 4.4))
x = np.arange(len(names)); w = 0.2
sm, wt = M['smote'], M['weights']
ax.bar(x - 1.5*w, [sm['base_oof_auc'][n] for n in names], w,
       label='SMOTETomek — OOF (resampled)', color='#e8a0a0', edgecolor='#b85450')
ax.bar(x - 0.5*w, [sm['base_test_auc'][n] for n in names], w,
       label='SMOTETomek — TEST (honest)', color='#b85450')
ax.bar(x + 0.5*w, [wt['base_oof_auc'][n] for n in names], w,
       label='Class-weighted — OOF', color='#a8c4e0', edgecolor='#2b4c7e')
ax.bar(x + 1.5*w, [wt['base_test_auc'][n] for n in names], w,
       label='Class-weighted — TEST', color='#2b4c7e')
for i, n in enumerate(names):
    gap = sm['base_oof_auc'][n] - sm['base_test_auc'][n]
    ax.annotate(f"−{gap:.2f}", xy=(i - w, sm['base_oof_auc'][n] + 0.012),
                ha='center', fontsize=8.5, color='#b85450', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("AUC")
ax.set_ylim(0.5, 1.05); ax.axhline(1.0, color='#999', lw=0.6)
ax.set_title("Evaluation Inflation: Resampled OOF vs Honest Test AUC",
             fontweight='bold')
ax.legend(fontsize=8, loc='lower left'); ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_oof_vs_test.png", dpi=220, bbox_inches='tight'); plt.close(fig)

# ── FIG: Ablation — what actually moved performance ───────────
labels = ["Raw features, SMOTE (8k)", "Raw features, weights (8k)",
          "Engineered, SMOTE (8k)", "Engineered, weights (8k)",
          "Engineered, SMOTE (42k full)", "Engineered, weights (42k full)",
          "1D-CNN, raw sequence (42k)"]
vals = [0.7393, 0.7682, 0.7227, 0.7599, M['smote']['auc'], M['weights']['auc'], 0.7745]
cols = ['#c9c9c9', '#c9c9c9', '#c9c9c9', '#c9c9c9', '#b85450', '#2b4c7e', '#7a9a5b']
fig, ax = plt.subplots(figsize=(8.4, 4.2))
bars = ax.barh(range(len(labels)), vals, color=cols, edgecolor='#444', height=0.62)
for i, v in enumerate(vals):
    ax.text(v + 0.004, i, f"{v:.4f}", va='center', fontsize=8.5)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8.5)
ax.invert_yaxis(); ax.set_xlim(0.65, 0.88); ax.set_xlabel("Honest test-set AUC")
ax.set_title("Ablation: Contribution of Each Design Choice", fontweight='bold')
ax.grid(axis='x', alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_ablation.png", dpi=220, bbox_inches='tight'); plt.close(fig)

# ── FIG: Confusion matrix (best model) ────────────────────────
best = M[M['best_mode']]
tn, fp, fn, tp = best['cm']
cm = np.array([[tn, fp], [fn, tp]])
fig, ax = plt.subplots(figsize=(4.6, 4.0))
im = ax.imshow(cm, cmap='Blues')
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]:,}", ha='center', va='center', fontsize=15,
                color='white' if cm[i, j] > cm.max()/2 else '#222', fontweight='bold')
ax.set_xticks([0, 1]); ax.set_xticklabels(['Predicted\nNormal', 'Predicted\nTheft'])
ax.set_yticks([0, 1]); ax.set_yticklabels(['Actual\nNormal', 'Actual\nTheft'])
ax.set_title(f"Confusion Matrix — Class-weighted Ensemble\n"
             f"FPR = {best['fpr']:.4f},  TPR = {best['tpr']:.4f}", fontweight='bold',
             fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_confusion_matrix.png", dpi=220, bbox_inches='tight'); plt.close(fig)

# ── FIG: Threshold analysis ───────────────────────────────────
ths = np.linspace(0.02, 0.98, 193)
fprs, tprs, f1s = [], [], []
for t in ths:
    yp = (p_wt >= t).astype(int)
    a, b, c, d = confusion_matrix(y, yp).ravel()
    fprs.append(b/(b+a)); tprs.append(d/(d+c)); f1s.append(f1_score(y, yp, zero_division=0))
fprs, tprs, f1s = map(np.array, (fprs, tprs, f1s))
ok = np.where(fprs <= 0.0103)[0]
t_cap = ths[ok[np.argmax(tprs[ok])]] if len(ok) else None
t_f1 = ths[np.argmax(f1s)]

fig, ax = plt.subplots(figsize=(8.0, 4.4))
ax.plot(ths, tprs, color='#2b4c7e', lw=2, label='TPR (recall)')
ax.plot(ths, fprs, color='#b85450', lw=2, label='FPR')
ax.plot(ths, f1s, color='#7a9a5b', lw=2, ls='--', label='F1-score')
ax.axhline(0.0103, color='#b85450', ls=':', lw=1.4, label='FPR target (0.0103)')
if t_cap:
    ax.axvline(t_cap, color='#666', ls='-.', lw=1.4)
    ax.annotate(f"recommended\nthr = {t_cap:.3f}", xy=(t_cap, 0.55),
                xytext=(t_cap + 0.12, 0.66), fontsize=8.5,
                arrowprops=dict(arrowstyle='->', color='#444'))
ax.axvline(0.5, color='#aaa', ls=':', lw=1.2)
ax.text(0.5, 0.92, ' default 0.5', fontsize=8, color='#777')
ax.set_xlabel("Decision threshold"); ax.set_ylabel("Metric value")
ax.set_title("Threshold Selection: Recall / FPR Trade-off", fontweight='bold')
ax.legend(fontsize=8.5); ax.grid(alpha=0.3); ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_threshold.png", dpi=220, bbox_inches='tight'); plt.close(fig)

# ── FIG: Feature importance ───────────────────────────────────
A = ['mean','std','cv','max','min','median','skew','kurtosis','IQR','peak_to_mean','total']
B = ['zero_ratio','zero_count','max_zero_streak','low_ratio','num_drops','max_drop','volatility']
C = [f'FFT_{i}' for i in range(1, 6)]
D = ['weekend_weekday_ratio','seasonality','peak_trough_month']
featnames = A + B + C + D + [f'week_{i:03d}' for i in range(1, 148)]
imp = art['xgb_importance']
order = np.argsort(imp)[::-1][:15]
fig, ax = plt.subplots(figsize=(7.4, 4.6))
colmap = {'A': '#2b4c7e', 'B': '#b85450', 'C': '#7a9a5b', 'D': '#d79b00', 'E': '#999999'}
def grp(i):
    if i < len(A): return 'A'
    if i < len(A)+len(B): return 'B'
    if i < len(A)+len(B)+len(C): return 'C'
    if i < len(A)+len(B)+len(C)+len(D): return 'D'
    return 'E'
ax.barh(range(len(order)), imp[order],
        color=[colmap[grp(i)] for i in order], edgecolor='#333', height=0.68)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([featnames[i] for i in order], fontsize=8.5)
ax.invert_yaxis(); ax.set_xlabel("XGBoost gain importance")
ax.set_title("Top 15 Features (class-weighted model)", fontweight='bold')
handles = [plt.Rectangle((0,0),1,1, color=colmap[k]) for k in 'ABCDE']
ax.legend(handles, ['Statistical','Theft-specific','FFT','Calendar','Weekly profile'],
          fontsize=8, loc='lower right')
ax.grid(axis='x', alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_feature_importance.png", dpi=220, bbox_inches='tight'); plt.close(fig)

print("Data figures written.")
print(f"  threshold (FPR-capped) = {t_cap:.3f}   F1-optimal = {t_f1:.3f}")
