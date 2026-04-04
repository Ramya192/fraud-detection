# prepare_data.py
# Rebuilds transactions_balanced.csv from the original creditcard.csv
# Keeps ALL columns: Time, V1-V28, Amount, is_Fraud, hour
# Run this once before running main.py
#
# Usage: python data/prepare_data.py

import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
RAW_PATH      = Path("data/creditcard.csv")
OUTPUT_PATH   = Path("data/transactions_balanced.csv")

# ── Load original dataset ──────────────────────────────────────────────
print("Loading original creditcard.csv ...")
df = pd.read_csv(RAW_PATH)
print(f"  Original shape: {df.shape}")          # (284807, 31)
print(f"  Columns: {list(df.columns)}")

# ── Rename Class → is_Fraud if needed ─────────────────────────────────
if "Class" in df.columns and "is_Fraud" not in df.columns:
    df = df.rename(columns={"Class": "is_Fraud"})
    print("  Renamed 'Class' → 'is_Fraud'")

# ── Add hour feature from Time column ─────────────────────────────────
# Time = seconds elapsed since first transaction in dataset
# hour = which hour of the day (0-23) using modulo
df["hour"] = (df["Time"] // 3600 % 24).astype(int)
print("  Added 'hour' feature")

# ── Balance dataset: equal fraud and legit samples ─────────────────────
fraud_df = df[df["is_Fraud"] == 1]
legit_df = df[df["is_Fraud"] == 0]

n_fraud = len(fraud_df)                          # 492 in original dataset
print(f"  Fraud transactions:      {n_fraud}")
print(f"  Legitimate transactions: {len(legit_df)}")

# Sample equal number of legit transactions (random_state for reproducibility)
legit_sample = legit_df.sample(n=n_fraud, random_state=42)

# Combine and shuffle
balanced_df = pd.concat([fraud_df, legit_sample]).sample(
    frac=1, random_state=42
).reset_index(drop=True)

print(f"\n  Balanced shape: {balanced_df.shape}")
print(f"  Fraud:  {balanced_df['is_Fraud'].sum()}")
print(f"  Legit:  {(balanced_df['is_Fraud'] == 0).sum()}")
print(f"  Columns kept: {list(balanced_df.columns)}")

# ── Save ───────────────────────────────────────────────────────────────
balanced_df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✓ Saved to {OUTPUT_PATH}")
print("  Ready to run main.py")
