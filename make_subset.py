"""
Create a stratified subset of the full SGCC dataset for a memory-safe
validation run on an 8 GB machine.

Full data: data set.csv  (42,372 customers x 1,034 daily readings + CONS_NO + FLAG)
Output:    datasetsmall.csv (~8,000 customers, FLAG ratio preserved)
"""
import pandas as pd
import numpy as np

SRC = r"C:\Users\HP\Desktop\final-project\data set.csv"
OUT = r"C:\Users\HP\Desktop\final-project\datasetsmall.csv"
TARGET_N = 8000
SEED = 42

print("Loading full dataset (this reads 163 MB, please wait)...")
df = pd.read_csv(SRC)
print(f"Full shape: {df.shape}")

# --- Full-data diagnostics (for the paper / log) ---
n_normal = int((df['FLAG'] == 0).sum())
n_theft = int((df['FLAG'] == 1).sum())
ratio = n_normal / n_theft
cons_cols = [c for c in df.columns if c not in ('CONS_NO', 'FLAG')]
total_missing = int(df[cons_cols].isna().sum().sum())
cells = len(df) * len(cons_cols)
print(f"Full class dist  -> Normal: {n_normal}, Theft: {n_theft}, ratio: {ratio:.1f}:1")
print(f"Full missing values: {total_missing:,} of {cells:,} cells "
      f"({100*total_missing/cells:.2f}%)")

# --- Stratified sample preserving FLAG ratio ---
frac = TARGET_N / len(df)
sub = df.groupby('FLAG').sample(frac=frac, random_state=SEED)
sub = sub.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle

s_normal = int((sub['FLAG'] == 0).sum())
s_theft = int((sub['FLAG'] == 1).sum())
s_missing = int(sub[cons_cols].isna().sum().sum())
print(f"\nSubset shape: {sub.shape}")
print(f"Subset class dist -> Normal: {s_normal}, Theft: {s_theft}, "
      f"ratio: {s_normal/s_theft:.1f}:1")
print(f"Subset missing values: {s_missing:,}")

sub.to_csv(OUT, index=False)
print(f"\nSaved subset -> {OUT}")
