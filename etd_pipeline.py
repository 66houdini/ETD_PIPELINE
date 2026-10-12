"""
Electricity Theft Detection — Two-Stage Stacked Ensemble Pipeline
Author: Tunji (190408030)
Supervisor: Dr. O.A. Alamu
Dataset: SGCC (25,863 customers, 26-day consumption readings)

Pipeline:
  1. Load & explore
  2. Preprocessing (imputation, normalisation, feature engineering)
  3. SMOTETomek balancing
  4. Stage 1 — base classifiers (XGBoost, RF, MLP)
  5. Stage 2 — Logistic Regression meta-learner
  6. Evaluation (AUC, F1, FPR)
  7. SHAP interpretability
  8. Save model for hardware integration
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import skew, kurtosis
from scipy.fft import fft

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, f1_score, confusion_matrix,
                              classification_report, roc_curve)

from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek
import shap
import pickle

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")

    # Separate consumption columns, ID, and label
    id_col   = 'CONS_NO'
    label_col = 'FLAG'
    cons_cols = [c for c in df.columns if c not in [id_col, label_col]]

    X_raw = df[cons_cols].values.astype(np.float32)   # float32 halves memory
    y     = df[label_col].values
    ids   = df[id_col].values
    del df                                            # free the 400 MB DataFrame

    print(f"Consumption days: {len(cons_cols)}")
    print(f"Class distribution — Normal: {(y==0).sum()}, Theft: {(y==1).sum()}")
    print(f"Imbalance ratio: {round((y==0).sum()/(y==1).sum(), 1)}:1")
    return X_raw, y, ids, cons_cols


# ─────────────────────────────────────────────
# 2. PREPROCESSING  (leakage-free: fit on TRAIN only)
# ─────────────────────────────────────────────

EPS = 1e-8


def _calendar_masks(dates):
    """Pre-compute date-derived index helpers (shared by all customers)."""
    dt = pd.to_datetime(pd.Series(dates), errors='coerce')
    dow = dt.dt.dayofweek.to_numpy()              # 0=Mon ... 6=Sun
    weekend = dow >= 5
    month_id = (dt.dt.year * 12 + dt.dt.month).to_numpy()
    return weekend, month_id


def _max_run(mask_row):
    """Longest run of True in a 1-D boolean array."""
    best = run = 0
    for v in mask_row:
        run = run + 1 if v else 0
        if run > best:
            best = run
    return best


def engineer_features(X, dates):
    """
    Overhauled feature engineering (leakage-free; all computations are per-customer
    row-wise on the RAW imputed consumption series, NOT column-scaled).

    Replaces the old 1,034 raw daily values (98.5% of importance, mostly noise)
    with a compact, interpretable set:
      A) Magnitude & statistical features        (on raw series)
      B) Theft-specific behavioural features      (zeros, drops, volatility)
      C) Periodicity features                     (FFT on per-customer shape)
      D) Calendar features                        (weekday/weekend, seasonality)
      E) Denoised weekly profile                  (per-customer normalised)
    """
    n, D = X.shape
    weekend, month_id = dates
    mean = X.mean(1); std = X.std(1); mx = X.max(1); mn = X.min(1)

    # ---- A) Magnitude & statistical ----
    cv = std / (mean + EPS)                                   # coeff. of variation
    q75 = np.percentile(X, 75, axis=1); q25 = np.percentile(X, 25, axis=1)
    A = np.column_stack([
        mean, std, cv, mx, mn, np.median(X, 1),
        skew(X, axis=1), kurtosis(X, axis=1),
        q75 - q25,                                           # IQR
        mx / (mean + EPS),                                   # peak-to-mean
        X.sum(1),                                            # total consumption
    ])

    # ---- B) Theft-specific behavioural ----
    is_zero = X <= 1e-6
    is_low = X < (0.1 * mean[:, None])                       # below 10% of own mean
    diffs = np.diff(X, axis=1)
    rel_drop = np.clip(-diffs / (X[:, :-1] + EPS), 0, None)  # day-over-day % drop
    B = np.column_stack([
        is_zero.mean(1),                                     # zero-day ratio
        is_zero.sum(1),                                      # zero-day count
        np.array([_max_run(r) for r in is_zero]),           # longest zero streak
        is_low.mean(1),                                      # low-consumption ratio
        (rel_drop > 0.5).sum(1),                             # # of >50% daily drops
        rel_drop.max(1),                                     # largest daily drop
        np.abs(diffs).mean(1),                               # volatility
    ])

    # ---- C) Periodicity (FFT on per-customer normalised SHAPE) ----
    Xn = (X - mn[:, None]) / (mx - mn + EPS)[:, None]
    C = np.abs(fft(Xn, axis=1))[:, 1:6]                      # top-5, DC excluded

    # ---- D) Calendar ----
    wkend_mean = X[:, weekend].mean(1)
    wkday_mean = X[:, ~weekend].mean(1)
    months = np.unique(month_id)
    monthly = np.column_stack([X[:, month_id == m].mean(1) for m in months])
    D_feats = np.column_stack([
        wkend_mean / (wkday_mean + EPS),                    # weekend/weekday ratio
        monthly.std(1) / (monthly.mean(1) + EPS),           # seasonality
        monthly.max(1) / (monthly.min(1) + EPS),            # peak/trough month
    ])

    # ---- E) Denoised weekly profile (per-customer min-max normalised) ----
    n_weeks = D // 7
    weekly = X[:, :n_weeks * 7].reshape(n, n_weeks, 7).mean(2)
    w_min = weekly.min(1)[:, None]; w_max = weekly.max(1)[:, None]
    weekly_norm = (weekly - w_min) / (w_max - w_min + EPS)

    feats = np.hstack([A, B, C, D_feats, weekly_norm])
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


def impute_rowwise_median(X, chunk=4000):
    """
    Memory-safe imputation: fill each customer's missing days with that customer's
    own median (leakage-free — every row is independent, so no train/test fitting).
    Done in row-chunks so the full 42k-row matrix never blows up memory. Replaces
    KNNImputer, which needs a ~14 GB distance matrix on the full dataset.
    """
    out = np.empty_like(X)
    for i in range(0, X.shape[0], chunk):
        blk = X[i:i + chunk].copy()
        med = np.nanmedian(blk, axis=1)
        med = np.where(np.isnan(med), 0.0, med)        # all-missing rows -> 0
        r, c = np.where(np.isnan(blk))
        blk[r, c] = med[r]
        out[i:i + chunk] = blk
    return out


def engineer_features_chunked(X, cal, chunk=4000):
    """Run engineer_features() in row-chunks so the FFT / weekly intermediates
    (which are ~n x 1034) never exceed a few hundred MB on the full dataset."""
    parts = [engineer_features(X[i:i + chunk], cal)
             for i in range(0, X.shape[0], chunk)]
    return np.vstack(parts)


def preprocess_train_test(X_raw_tr, X_raw_te, dates, impute='median'):
    """
    Leakage-free. impute='median' (memory-safe, for the full dataset) or 'knn'
    (KNNImputer fit on train, only feasible on small subsets). Features are
    engineered per-customer (row-wise); a StandardScaler over the engineered
    features is fit on TRAIN only.
    """
    if impute == 'knn':
        print("\n[1/3] KNN imputation (fit on train only)...")
        imputer = KNNImputer(n_neighbors=5)
        X_tr_imp = imputer.fit_transform(X_raw_tr)
        X_te_imp = imputer.transform(X_raw_te)
    else:
        print("\n[1/3] Row-wise median imputation (memory-safe, chunked)...")
        imputer = None
        X_tr_imp = impute_rowwise_median(X_raw_tr)
        X_te_imp = impute_rowwise_median(X_raw_te)

    print("[2/3] Feature engineering (chunked: stats, theft, FFT, calendar, weekly)...")
    cal = _calendar_masks(dates)
    X_tr_feat = engineer_features_chunked(X_tr_imp, cal)
    X_te_feat = engineer_features_chunked(X_te_imp, cal)
    del X_tr_imp, X_te_imp

    print("[3/3] Standardising engineered features (fit on train only)...")
    scaler = StandardScaler()
    X_tr_feat = scaler.fit_transform(X_tr_feat)
    X_te_feat = scaler.transform(X_te_feat)

    print(f"Feature matrix — train: {X_tr_feat.shape}, test: {X_te_feat.shape}")
    return X_tr_feat, X_te_feat, imputer, scaler


# ─────────────────────────────────────────────
# 3. BASE-CLASSIFIER FACTORY (imbalance strategy: 'smote' or 'weights')
# ─────────────────────────────────────────────

def make_base_classifiers(mode, y_train):
    """
    mode='smote'  : plain classifiers (imbalance handled upstream by SMOTETomek)
    mode='weights': cost-sensitive classifiers (no synthetic data)
                    - XGBoost: scale_pos_weight = n_neg / n_pos
                    - RandomForest: class_weight='balanced'
                    - MLP: sklearn's MLPClassifier has no class_weight, so it
                      trains unweighted (documented limitation).
    """
    spw = (y_train == 0).sum() / (y_train == 1).sum()  # ~10.7 for our data

    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
        random_state=42, n_jobs=-1,
        scale_pos_weight=(spw if mode == 'weights' else 1.0),
    )
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10, random_state=42, n_jobs=-1,
        class_weight=('balanced' if mode == 'weights' else None),
    )
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64), activation='relu',
        max_iter=300, random_state=42,
    )
    return {'XGBoost': xgb, 'RandomForest': rf, 'MLP': mlp}


# ─────────────────────────────────────────────
# 4. STAGE 1 — BASE CLASSIFIERS (OOF predictions)
# ─────────────────────────────────────────────

def train_base_classifiers(X_train, y_train, X_test, base_clfs):
    """
    5-fold cross-validation to generate out-of-fold (OOF) probability predictions.
    These OOF predictions become the input features for the meta-learner.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    n_train = X_train.shape[0]
    n_test  = X_test.shape[0]
    n_clfs  = len(base_clfs)

    oof_preds  = np.zeros((n_train, n_clfs))   # OOF for meta-learner training
    test_preds = np.zeros((n_test, n_clfs))    # Test predictions

    clf_names = list(base_clfs.keys())
    fold_aucs = {}                                  # for the report's fold tables

    print("\n--- Stage 1: Training base classifiers ---")
    for clf_idx, (name, clf) in enumerate(base_clfs.items()):
        print(f"\nTraining {name}...")
        fold_aucs[name] = []
        test_fold_preds = np.zeros((n_test, 5))  # accumulate across folds

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            clf.fit(X_tr, y_tr)
            oof_preds[val_idx, clf_idx] = clf.predict_proba(X_val)[:, 1]
            test_fold_preds[:, fold]    = clf.predict_proba(X_test)[:, 1]

            fold_auc = roc_auc_score(y_val, oof_preds[val_idx, clf_idx])
            fold_aucs[name].append(float(fold_auc))
            print(f"  Fold {fold+1} AUC: {fold_auc:.4f}")

        test_preds[:, clf_idx] = test_fold_preds.mean(axis=1)

        # Individual model performance on full OOF
        oof_auc = roc_auc_score(y_train, oof_preds[:, clf_idx])
        print(f"  {name} OOF AUC: {oof_auc:.4f}")

    return oof_preds, test_preds, clf_names, base_clfs, fold_aucs


# ─────────────────────────────────────────────
# 5. STAGE 2 — LOGISTIC REGRESSION META-LEARNER
# ─────────────────────────────────────────────

def train_meta_learner(oof_preds, test_preds, y_train, y_test, clf_names):
    print("\n--- Stage 2: Training Logistic Regression meta-learner ---")

    meta = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
    meta.fit(oof_preds, y_train)

    # Final predictions
    y_pred_proba = meta.predict_proba(test_preds)[:, 1]
    y_pred       = meta.predict(test_preds)

    return meta, y_pred_proba, y_pred


# ─────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────

def evaluate(y_test, y_pred, y_pred_proba, clf_names, meta, title):
    print("\n" + "="*55)
    print(f"RESULTS — Two-Stage Stacked Ensemble  [{title}]")
    print("="*55)

    auc = roc_auc_score(y_test, y_pred_proba)
    f1  = f1_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)   # False Positive Rate
    tpr = tp / (tp + fn)   # True Positive Rate (Recall)

    print(f"\nAUC:                  {auc:.4f}  (target >= 0.985)")
    print(f"F1-score:             {f1:.4f}")
    print(f"False Positive Rate:  {fpr:.4f}  (target <= 0.0103)")
    print(f"True Positive Rate:   {tpr:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={tn}  FP={fp}")
    print(f"  FN={fn}  TP={tp}")

    print(f"\nMeta-learner (LR) coefficients:")
    for name, coef in zip(clf_names, meta.coef_[0]):
        print(f"  {name}: {coef:.4f}")

    print("\nFull classification report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Theft']))

    return {'auc': auc, 'fpr': fpr, 'f1': f1, 'tpr': tpr,
            'cm': (tn, fp, fn, tp), 'coef': dict(zip(clf_names, meta.coef_[0]))}


def threshold_analysis(y_test, y_proba, fpr_cap=0.0103):
    """
    The default 0.5 cutoff is arbitrary. AUC measures ranking across ALL thresholds;
    here we pick operating points that matter for a utility:
      - F1-optimal threshold (best precision/recall balance)
      - Recall-maximising threshold subject to FPR <= 0.0103 (the paper's FPR target)
    """
    print("\n" + "="*60)
    print(f"THRESHOLD ANALYSIS  (FPR cap = {fpr_cap})")
    print("="*60)

    def point(thr):
        yp = (y_proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, yp).ravel()
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = f1_score(y_test, yp, zero_division=0)
        return thr, fpr, tpr, f1, (tn, fp, fn, tp)

    grid = np.linspace(0.01, 0.99, 197)
    pts = [point(th) for th in grid]

    best_f1 = max(pts, key=lambda r: r[3])
    capped = [r for r in pts if r[1] <= fpr_cap]
    best_cap = max(capped, key=lambda r: r[2]) if capped else None
    default = point(0.5)

    def show(tag, r):
        if r is None:
            print(f"  {tag:<34} (no threshold satisfies the FPR cap)")
            return
        thr, fpr, tpr, f1, cm = r
        print(f"  {tag:<34} thr={thr:.3f}  FPR={fpr:.4f}  TPR={tpr:.4f}  F1={f1:.4f}")

    show("Default (0.50):", default)
    show("F1-optimal:", best_f1)
    show(f"Max recall with FPR<={fpr_cap}:", best_cap)
    return {'default': default, 'best_f1': best_f1, 'best_capped': best_cap}


# ─────────────────────────────────────────────
# 4b. RUN ONE FULL EXPERIMENT (Stage 1 + Stage 2 + eval)
# ─────────────────────────────────────────────

def run_experiment(mode, X_train, X_test, y_train, y_test):
    """mode in {'smote', 'weights'}. Returns metrics dict + fitted objects."""
    title = 'SMOTETomek' if mode == 'smote' else 'Class-weighted'
    print("\n" + "#"*60)
    print(f"# EXPERIMENT: {title}")
    print("#"*60)

    if mode == 'smote':
        smt = SMOTETomek(random_state=42)
        X_tr, y_tr = smt.fit_resample(X_train, y_train)
        print(f"After SMOTETomek — Normal: {(y_tr==0).sum()}, Theft: {(y_tr==1).sum()}")
    else:
        X_tr, y_tr = X_train, y_train
        spw = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"No resampling. XGB scale_pos_weight={spw:.2f}, "
              f"RF class_weight='balanced' (MLP unweighted).")

    base_clfs = make_base_classifiers(mode, y_train)
    oof, testp, names, fitted, fold_aucs = train_base_classifiers(
        X_tr, y_tr, X_test, base_clfs)

    # Honest base-classifier AUC: OOF (training-side) vs real imbalanced TEST.
    print("\n--- Base classifier AUC: OOF vs real TEST ---")
    base_test_auc, base_oof_auc = {}, {}
    for i, name in enumerate(names):
        ta = roc_auc_score(y_test, testp[:, i])
        oa = roc_auc_score(y_tr, oof[:, i])
        base_test_auc[name], base_oof_auc[name] = ta, oa
        print(f"  {name:<14} OOF: {oa:.4f}  |  TEST: {ta:.4f}")

    meta, ypp, yp = train_meta_learner(oof, testp, y_tr, y_test, names)
    metrics = evaluate(y_test, yp, ypp, names, meta, title)
    metrics['base_test_auc'] = base_test_auc
    metrics['base_oof_auc'] = base_oof_auc
    metrics['fold_aucs'] = fold_aucs
    metrics['base_test_preds'] = testp
    metrics['fitted'] = fitted
    metrics['meta'] = meta
    metrics['names'] = names
    metrics['proba'] = ypp
    return metrics


# ─────────────────────────────────────────────
# 7. SHAP INTERPRETABILITY
# ─────────────────────────────────────────────

def run_shap(base_clfs, X_test, clf_names):
    print("\n--- SHAP Analysis (XGBoost base classifier) ---")
    xgb_model = base_clfs['XGBoost']

    # Use a sample for speed
    X_sample = X_test[:500]
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample)

    # Summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("SHAP summary plot saved: shap_summary.png")


# ─────────────────────────────────────────────
# 8. SAVE MODEL FOR HARDWARE INTEGRATION
# ─────────────────────────────────────────────

def save_model(meta, base_clfs, imputer, scaler):
    model_bundle = {
        'meta_learner': meta,
        'base_classifiers': base_clfs,
        'imputer': imputer,
        'scaler': scaler
    }
    with open('etd_model.pkl', 'wb') as f:
        pickle.dump(model_bundle, f)
    print("\nModel saved: etd_model.pkl")
    print("Use this file for hardware-software integration (pyserial → inference)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    # Full dataset + memory-safe median imputation (fits on 8 GB).
    # For a quick subset run, point at datasetsmall.csv and set IMPUTE='knn'.
    DATASET_PATH = r'C:\Users\HP\Desktop\final-project\data set.csv'
    IMPUTE = 'median'

    # 1. Load
    X_raw, y, ids, cons_cols = load_data(DATASET_PATH)

    # 2. Split RAW first (stratified), then preprocess fitting on TRAIN only.
    X_raw_tr, X_raw_te, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(y_train)} | Test: {len(y_test)}")
    print(f"Train class dist — Normal: {(y_train==0).sum()}, Theft: {(y_train==1).sum()}")
    del X_raw                                          # free the full raw matrix
    X_train, X_test, imputer, scaler = preprocess_train_test(
        X_raw_tr, X_raw_te, cons_cols, impute=IMPUTE)

    # 3. Run both imbalance-handling experiments on identical features.
    results = {}
    results['smote']   = run_experiment('smote',   X_train, X_test, y_train, y_test)
    results['weights'] = run_experiment('weights', X_train, X_test, y_train, y_test)

    # 4. Side-by-side comparison (the paper's ablation / tuning-log row).
    print("\n" + "="*64)
    print("EXPERIMENT COMPARISON  (real imbalanced test set)")
    print("="*64)
    print(f"{'Metric':<22}{'A: SMOTETomek':>20}{'B: Class-weighted':>22}")
    for key, label in [('auc', 'Ensemble AUC'), ('f1', 'F1-score'),
                       ('fpr', 'FPR'), ('tpr', 'TPR (recall)')]:
        print(f"{label:<22}{results['smote'][key]:>20.4f}{results['weights'][key]:>22.4f}")
    print(f"{'Targets':<22}{'AUC>=0.985':>20}{'FPR<=0.0103':>22}")

    # 5. Keep the better ensemble (by test AUC) for SHAP + hardware save.
    best_mode = max(results, key=lambda m: results[m]['auc'])
    best = results[best_mode]
    print(f"\nBest experiment by test AUC: {best_mode} (AUC={best['auc']:.4f})")

    # 5b. Threshold tuning on the best model (lift recall at acceptable FPR).
    threshold_analysis(y_test, best['proba'])

    # 5c. Save every artifact the report figures need, so plots can be
    #     regenerated without retraining (reproducibility + speed).
    import json
    np.savez_compressed(
        'report_artifacts.npz',
        y_test=y_test,
        proba_smote=results['smote']['proba'],
        proba_weights=results['weights']['proba'],
        base_test_smote=results['smote']['base_test_preds'],
        base_test_weights=results['weights']['base_test_preds'],
        xgb_importance=best['fitted']['XGBoost'].feature_importances_,
    )
    summary = {
        m: {k: results[m][k] for k in
            ('auc', 'fpr', 'f1', 'tpr', 'base_test_auc', 'base_oof_auc',
             'fold_aucs')}
        | {'cm': [int(v) for v in results[m]['cm']],
           'coef': {k: float(v) for k, v in results[m]['coef'].items()}}
        for m in results
    }
    summary['best_mode'] = best_mode
    summary['n_train'] = int(len(y_train))
    summary['n_test'] = int(len(y_test))
    with open('report_metrics.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("\nSaved report_artifacts.npz + report_metrics.json")

    run_shap(best['fitted'], X_test, best['names'])
    save_model(best['meta'], best['fitted'], imputer, scaler)
