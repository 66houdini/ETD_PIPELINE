# -*- coding: utf-8 -*-
"""
Multi-seed evaluation with validation-set threshold selection.

Addresses two limitations of the single-run results in etd_pipeline.py:
  1. No confidence intervals  -> repeat the whole experiment over several seeds
     (the seed changes the data split, the CV folds, SMOTETomek and every
     classifier) and report mean, standard deviation and a 95% t-interval.
  2. Threshold chosen on the test set -> split train/validation/test
     (70/10/20), choose the decision threshold on VALIDATION only, and report
     what that threshold does on the untouched TEST set.

Results are checkpointed after every (seed, mode) run, so an interrupted run
resumes where it stopped.

Usage:
    python multiseed_experiment.py            # full run, 5 seeds
    python multiseed_experiment.py --smoke    # 3,000-customer check, 2 seeds
"""
import argparse, json, os, time
import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from imblearn.combine import SMOTETomek

from etd_pipeline import (load_data, impute_rowwise_median, _calendar_masks,
                          engineer_features_chunked, make_base_classifiers)

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE, 'data set.csv')
SEEDS = [42, 1, 2, 3, 4]
MODES = ['weights', 'smote', 'smote_infold']
FPR_CAP = 0.0103
GRID = np.linspace(0.01, 0.99, 197)      # same grid as etd_pipeline.threshold_analysis


def load_features(smoke):
    """Imputation and feature engineering are row-wise (no fitted state), so
    they are computed once for every customer and cached. Only the scaler is
    fitted, and that happens per seed on the training partition."""
    cache = os.path.join(BASE, 'features_smoke.npz' if smoke else 'features_all.npz')
    if os.path.exists(cache):
        z = np.load(cache)
        return z['F'], z['y']
    X_raw, y, _, cons_cols = load_data(DATASET_PATH)
    if smoke:
        _, X_raw, _, y = train_test_split(X_raw, y, test_size=3000,
                                          random_state=0, stratify=y)
    F = engineer_features_chunked(impute_rowwise_median(X_raw),
                                  _calendar_masks(cons_cols))
    np.savez_compressed(cache, F=F, y=y)
    return F, y


def point(y, proba, thr):
    tn, fp, fn, tp = confusion_matrix(y, (proba >= thr).astype(int),
                                      labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {'thr': float(thr), 'fpr': float(fpr), 'tpr': float(tpr),
            'precision': float(prec), 'f1': float(f1)}


def select_thresholds(y_val, p_val):
    """Choose operating points on VALIDATION data only."""
    pts = [point(y_val, p_val, t) for t in GRID]
    capped = [p for p in pts if p['fpr'] <= FPR_CAP]
    thr_cap = max(capped, key=lambda p: p['tpr'])['thr'] if capped else 0.5
    thr_f1 = max(pts, key=lambda p: p['f1'])['thr']
    return thr_cap, thr_f1


def run_one(F, y, seed, mode):
    idx = np.arange(len(y))
    idx_trval, idx_te = train_test_split(idx, test_size=0.20, random_state=seed,
                                         stratify=y)
    idx_tr, idx_va = train_test_split(idx_trval, test_size=0.125,     # 10% of all
                                      random_state=seed, stratify=y[idx_trval])
    scaler = StandardScaler().fit(F[idx_tr])
    X_tr, X_va, X_te = (scaler.transform(F[i]) for i in (idx_tr, idx_va, idx_te))
    y_tr, y_va, y_te = y[idx_tr], y[idx_va], y[idx_te]

    # 'smote'        : resample the whole training set, THEN cross-validate
    #                  (held-out folds contain synthetic points).
    # 'smote_infold' : cross-validate the real training set and resample each
    #                  training fold only (held-out folds are real data).
    # 'weights'      : no resampling; cost-sensitive classifiers.
    if mode == 'smote':
        X_fit, y_fit = SMOTETomek(random_state=seed).fit_resample(X_tr, y_tr)
    else:
        X_fit, y_fit = X_tr, y_tr

    clfs = make_base_classifiers(mode, y_tr)
    for c in clfs.values():
        c.set_params(random_state=seed)
    names = list(clfs)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros((len(y_fit), len(names)))
    val_p = np.zeros((len(y_va), len(names)))
    test_p = np.zeros((len(y_te), len(names)))
    for tr, ho in skf.split(X_fit, y_fit):
        if mode == 'smote_infold':
            X_f, y_f = SMOTETomek(random_state=seed).fit_resample(X_fit[tr], y_fit[tr])
        else:
            X_f, y_f = X_fit[tr], y_fit[tr]
        for j, name in enumerate(names):
            clfs[name].fit(X_f, y_f)
            oof[ho, j] = clfs[name].predict_proba(X_fit[ho])[:, 1]
            val_p[:, j] += clfs[name].predict_proba(X_va)[:, 1] / 5
            test_p[:, j] += clfs[name].predict_proba(X_te)[:, 1] / 5

    meta = LogisticRegression(C=1.0, random_state=seed, max_iter=1000).fit(oof, y_fit)
    p_va = meta.predict_proba(val_p)[:, 1]
    p_te = meta.predict_proba(test_p)[:, 1]
    thr_cap, thr_f1 = select_thresholds(y_va, p_va)

    return {
        'seed': seed, 'mode': mode,
        'n_train': int(len(y_tr)), 'n_val': int(len(y_va)), 'n_test': int(len(y_te)),
        'n_fit': int(len(y_fit)),
        'base_oof_auc': {n: float(roc_auc_score(y_fit, oof[:, j]))
                         for j, n in enumerate(names)},
        'base_val_auc': {n: float(roc_auc_score(y_va, val_p[:, j]))
                         for j, n in enumerate(names)},
        'base_test_auc': {n: float(roc_auc_score(y_te, test_p[:, j]))
                          for j, n in enumerate(names)},
        'coef': {n: float(c) for n, c in zip(names, meta.coef_[0])},
        'ensemble_val_auc': float(roc_auc_score(y_va, p_va)),
        'ensemble_test_auc': float(roc_auc_score(y_te, p_te)),
        'test_at_default': point(y_te, p_te, 0.5),
        'val_at_fpr_cap_thr': point(y_va, p_va, thr_cap),
        'test_at_fpr_cap_thr': point(y_te, p_te, thr_cap),
        'test_at_val_f1_thr': point(y_te, p_te, thr_f1),
    }


def agg(values):
    v = np.asarray(values, dtype=float)
    n = len(v)
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else 0.0
    return {'mean': float(v.mean()), 'std': sd, 'ci95_half_width': half,
            'min': float(v.min()), 'max': float(v.max()), 'n': n}


def summarise(runs):
    out = {}
    for mode in MODES:
        rs = [r for r in runs if r['mode'] == mode]
        if not rs:
            continue
        s = {'seeds': [r['seed'] for r in rs],
             'ensemble_test_auc': agg([r['ensemble_test_auc'] for r in rs])}
        for n in rs[0]['base_oof_auc']:
            s[f'{n}_oof_auc'] = agg([r['base_oof_auc'][n] for r in rs])
            s[f'{n}_test_auc'] = agg([r['base_test_auc'][n] for r in rs])
            s[f'{n}_inflation'] = agg([r['base_oof_auc'][n] - r['base_test_auc'][n]
                                       for r in rs])
            s[f'{n}_coef'] = agg([r['coef'][n] for r in rs])
        for block in ('test_at_default', 'test_at_fpr_cap_thr', 'test_at_val_f1_thr'):
            for k in ('thr', 'fpr', 'tpr', 'precision', 'f1'):
                s[f'{block}.{k}'] = agg([r[block][k] for r in rs])
        out[mode] = s
    # Paired comparison across seeds (same split for both modes).
    w = {r['seed']: r for r in runs if r['mode'] == 'weights'}
    m = {r['seed']: r for r in runs if r['mode'] == 'smote'}
    common = sorted(set(w) & set(m))
    if len(common) > 1:
        d = [w[s]['ensemble_test_auc'] - m[s]['ensemble_test_auc'] for s in common]
        out['weights_minus_smote_test_auc'] = agg(d) | {
            'paired_t_p': float(stats.ttest_1samp(d, 0.0).pvalue)}
    f = {r['seed']: r for r in runs if r['mode'] == 'smote_infold'}
    for tag, other in (('weights_minus_smote_infold_test_auc', w),
                       ('smote_minus_smote_infold_test_auc', m)):
        common = sorted(set(other) & set(f))
        if len(common) > 1:
            d = [other[s]['ensemble_test_auc'] - f[s]['ensemble_test_auc'] for s in common]
            out[tag] = agg(d) | {'paired_t_p': float(stats.ttest_1samp(d, 0.0).pvalue)}
    return out


def print_summary(summ):
    for mode in MODES:
        if mode not in summ:
            continue
        s = summ[mode]
        print(f"\n=== {mode}  (seeds {s['seeds']}) ===")
        for k, a in s.items():
            if isinstance(a, dict):
                print(f"  {k:<32} {a['mean']:.4f} ± {a['std']:.4f}  "
                      f"(95% CI ±{a['ci95_half_width']:.4f}, "
                      f"range {a['min']:.4f}–{a['max']:.4f})")
    for tag in ('weights_minus_smote_test_auc', 'weights_minus_smote_infold_test_auc',
                'smote_minus_smote_infold_test_auc'):
        if tag in summ:
            a = summ[tag]
            print(f"{tag}: {a['mean']:+.4f} ± {a['std']:.4f} "
                  f"(paired t-test p = {a['paired_t_p']:.4f})")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    seeds = SEEDS[:2] if args.smoke else SEEDS
    res_path = os.path.join(BASE, 'multiseed_smoke.json' if args.smoke
                            else 'multiseed_results.json')
    runs = json.load(open(res_path)) if os.path.exists(res_path) else []
    done = {(r['seed'], r['mode']) for r in runs}

    F, y = load_features(args.smoke)
    print(f"Features: {F.shape}  theft: {int(y.sum())}", flush=True)
    for seed in seeds:
        for mode in MODES:
            if (seed, mode) in done:
                continue
            t0 = time.time()
            r = run_one(F, y, seed, mode)
            r['seconds'] = round(time.time() - t0, 1)
            runs.append(r)
            json.dump(runs, open(res_path, 'w'), indent=2)
            print(f"seed {seed:>3} {mode:<8} test AUC {r['ensemble_test_auc']:.4f}  "
                  f"val-thr {r['test_at_fpr_cap_thr']['thr']:.3f} -> test FPR "
                  f"{r['test_at_fpr_cap_thr']['fpr']:.4f} TPR "
                  f"{r['test_at_fpr_cap_thr']['tpr']:.4f}  [{r['seconds']}s]", flush=True)

    summ = summarise(runs)
    if not args.smoke:
        json.dump(summ, open(os.path.join(BASE, 'multiseed_summary.json'), 'w'), indent=2)
    print_summary(summ)
