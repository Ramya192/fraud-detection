import pandas as pd

# Load Dataset
df = pd.read_csv('data/creditcard.csv')

# Basic Shape
print("Total transactions:",len(df))
print("Columns:",df.columns.tolist())
print()

# How many Frauds vs Legitimate?
fraud_count = df[df["Class"]== 1].shape[0]
legit_count = df[df["Class"]== 0].shape[0]
print("Legitimate Transactions:", legit_count)
print("Fraud Transactions:", fraud_count)
print()

# What % is fraud?
fraud_pct = (fraud_count/len(df)) * 100
print(f"Fraud percentage: {round(fraud_pct,2)}%")
print()

# What does a Fraud transaction look like?
print("Sample Fraud Transaction:")
print(df[df["Class"]== 1].iloc[0][["Time","Amount","Class"]])
print()

# What does a Legitimate transaction look like?
print("Sample Legitimate Transaction:")
print(df[df["Class"]== 0].iloc[0][["Time","Amount","Class"]])
print()

# Amount Statistics
print("Amount Statistics:")
print(df["Amount"].describe())