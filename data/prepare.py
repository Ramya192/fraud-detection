import pandas as pd

# Load full Dataset
df = pd.read_csv("data/creditcard.csv")

# Select only the columns our agent needs
# V1-V28 are too complex for now - we use Amount, Time, Class
agent_df = df[["Time", "Amount", "Class"]].copy()

# Rename Class to is_Fraud for Clarity
agent_df = agent_df.rename(columns= {"Class":"is_Fraud"})
agent_df["hour"] = ((agent_df["Time"]/3600) %24 ).astype(int)

# Separate Fraud and Legitimate transactions
fraud_df = agent_df[agent_df["is_Fraud"]== 1]
legit_df = agent_df[agent_df["is_Fraud"]== 0]

# Take all 492 frauds + 492 random legitimate transactions
# This gives us balanced sample to work with 
legit_sample = legit_df.sample(n=492, random_state=42)
balanced_df = pd.concat([fraud_df, legit_sample])

# Shuffle the rows
balanced_df = balanced_df.sample(frac=1,random_state=42).reset_index(drop=True)

# Save to new CSV
balanced_df.to_csv("data/transactions_balanced.csv", index=False)

print("Original dataset:", len(df), "transactions")
print("Balanced dataset:", len(balanced_df), "transactions")
print("Fraud in balanced:", len(balanced_df[balanced_df["is_Fraud"] == 1]))
print("Legit in balanced:", len(balanced_df[balanced_df["is_Fraud"] == 0]))
print()
print("Sample transactions:")
print(balanced_df.head())
print()
print("Saved to data/transactions_balanced.csv")

