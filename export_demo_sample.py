# -*- coding: utf-8 -*-
"""
Export a small sample of REAL held-out customers from the SGCC test set for the
hardware demo's accumulated-pattern classification.

Uses the SAME split as training (test_size=0.2, random_state=42, stratify=y), so
every exported row is genuinely unseen by the model — no cherry-picking, no leakage.

Output: demo_historical_sample.csv
  - 1,034 daily consumption columns, ORIGINAL DATE HEADERS PRESERVED
    (the feature engineering derives weekday/weekend and monthly seasonality
     from these headers, so they must not be renamed or dropped)
  - TRUE_FLAG column (ground truth) so the demo can show whether the model was
    right. The integration script drops it before inference.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SRC = r"C:\Users\HP\Desktop\final-project\data set.csv"
OUT = r"C:\Users\HP\Desktop\final-project\demo_historical_sample.csv"
N_PER_CLASS = 3
SEED = 42

print("Loading full dataset...")
df = pd.read_csv(SRC)
cons_cols = [c for c in df.columns if c not in ('CONS_NO', 'FLAG')]
y = df['FLAG'].values
print(f"Full: {df.shape}, theft={int((y==1).sum())}, normal={int((y==0).sum())}")

# Reproduce the TRAINING split exactly. Splitting an index array with the same
# y/random_state/test_size yields the identical partition as splitting X.
idx_train, idx_test = train_test_split(
    np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
print(f"Held-out test partition: {len(idx_test)} customers")

test_df = df.iloc[idx_test]
y_test = y[idx_test]

# Random (not cherry-picked) sample of each class from the held-out set.
rng = np.random.default_rng(SEED)
theft_idx = rng.choice(np.where(y_test == 1)[0], N_PER_CLASS, replace=False)
norm_idx = rng.choice(np.where(y_test == 0)[0], N_PER_CLASS, replace=False)
pick = np.concatenate([theft_idx, norm_idx])

sample = test_df.iloc[pick][cons_cols].copy()
sample['TRUE_FLAG'] = y_test[pick]
sample = sample.reset_index(drop=True)

sample.to_csv(OUT, index=False)
print(f"\nExported {len(sample)} customers -> demo_historical_sample.csv")
print(f"  columns: {len(cons_cols)} daily readings + TRUE_FLAG")
print(f"  first date col: {cons_cols[0]}   last: {cons_cols[-1]}")
print(f"  TRUE_FLAG values: {list(sample['TRUE_FLAG'])}  (1=theft, 0=normal)")
print(f"  missing values in sample: {int(sample[cons_cols].isna().sum().sum())} "
      f"(handled by row-wise median imputation at inference)")
