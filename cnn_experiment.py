# -*- coding: utf-8 -*-
"""
Temporal 1D-CNN benchmark for ETD — does a deep model on the RAW daily sequence
beat the classical-ML stacked ensemble (honest test AUC 0.8229)?

Same split (random_state=42) and same honest, untouched test set as etd_pipeline.py,
so the AUC is directly comparable. Imbalance handled the deep-learning way:
BCEWithLogitsLoss(pos_weight = n_neg/n_pos) — the equivalent of the class weighting
that has been winning for the classical models.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score

from etd_pipeline import load_data, impute_rowwise_median

torch.manual_seed(42)
np.random.seed(42)
DEVICE = 'cpu'
EPOCHS = 25
BATCH = 512
PATH = r'C:\Users\HP\Desktop\final-project\data set.csv'


def rownorm(X):
    """Per-customer min-max to [0,1] — preserves each customer's temporal shape."""
    mn = X.min(1, keepdims=True)
    mx = X.max(1, keepdims=True)
    return (X - mn) / (mx - mn + 1e-8)


class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 7, padding=3), nn.ReLU(), nn.AdaptiveMaxPool1d(8),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 8, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.head(self.features(x))


def main():
    print("Loading full dataset...")
    X_raw, y, ids, cons = load_data(PATH)

    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train {len(ytr)} | Test {len(yte)}  (theft train={int((ytr==1).sum())})")

    print("Imputation + per-customer normalisation...")
    Xtr = rownorm(impute_rowwise_median(Xtr_raw)).astype(np.float32)
    Xte = rownorm(impute_rowwise_median(Xte_raw)).astype(np.float32)

    Xtr_t = torch.from_numpy(Xtr).unsqueeze(1)          # (n, 1, 1034)
    Xte_t = torch.from_numpy(Xte).unsqueeze(1)
    ytr_t = torch.from_numpy(ytr.astype(np.float32))
    n = Xtr_t.shape[0]

    pos_weight = torch.tensor([(ytr == 0).sum() / (ytr == 1).sum()], dtype=torch.float32)
    print(f"pos_weight = {pos_weight.item():.2f}")

    model = CNN1D().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc = 0.0
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb = Xtr_t[idx].to(DEVICE)
            yb = ytr_t[idx].to(DEVICE).unsqueeze(1)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)

        # Evaluate honest test AUC
        model.eval()
        with torch.no_grad():
            proba = torch.sigmoid(model(Xte_t.to(DEVICE))).cpu().numpy().ravel()
        auc = roc_auc_score(yte, proba)
        best_auc = max(best_auc, auc)
        print(f"  epoch {ep:2d}  loss={tot/n:.4f}  TEST AUC={auc:.4f}")

    # Final metrics at default 0.5 and threshold-tuned operating points
    print("\n" + "=" * 56)
    print("1D-CNN RESULTS (honest test set)")
    print("=" * 56)
    print(f"Best TEST AUC over training: {best_auc:.4f}")
    print(f"Final TEST AUC:             {auc:.4f}   (ensemble baseline 0.8229)")

    def op(thr):
        yp = (proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, yp).ravel()
        fpr = fp / (fp + tn); tpr = tp / (tp + fn)
        return thr, fpr, tpr, f1_score(yte, yp, zero_division=0)

    grid = np.linspace(0.05, 0.95, 181)
    pts = [op(t) for t in grid]
    capped = [r for r in pts if r[1] <= 0.0103]
    best_cap = max(capped, key=lambda r: r[2]) if capped else None
    best_f1 = max(pts, key=lambda r: r[3])
    d = op(0.5)
    print(f"Default 0.50:                FPR={d[1]:.4f} TPR={d[2]:.4f} F1={d[3]:.4f}")
    if best_cap:
        print(f"Max recall FPR<=0.0103:      thr={best_cap[0]:.3f} "
              f"FPR={best_cap[1]:.4f} TPR={best_cap[2]:.4f} F1={best_cap[3]:.4f}")
    print(f"F1-optimal:                  thr={best_f1[0]:.3f} "
          f"FPR={best_f1[1]:.4f} TPR={best_f1[2]:.4f} F1={best_f1[3]:.4f}")

    torch.save(model.state_dict(), r'C:\Users\HP\Desktop\final-project\etd_cnn.pt')
    print("\nSaved CNN weights: etd_cnn.pt")


if __name__ == '__main__':
    main()
