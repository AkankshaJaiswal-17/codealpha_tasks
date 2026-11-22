import numpy as np
import pandas as pd

# NumPy practical
income = np.array([50000, 25000, 80000, 30000, 60000])

print("Average Income:", np.mean(income))
print("Maximum Income:", np.max(income))

# Creating dataset
data = {
    "Name": ["Riya", "Aman", "Akash", "Neha", "Raj"],
    "Income": [50000, 25000, 80000, 30000, 60000],
    "Debt": [5000, 20000, 10000, 15000, 8000],
    "Payment_History": ["Good", "Poor", "Good", "Poor", "Good"],
    "Credit_Label": ["Good", "Poor", "Good", "Poor", "Good"]
}

df = pd.DataFrame(data)

print("\nComplete Dataset:")
print(df)

print("\nFirst 3 Rows:")
print(df.head(3))

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

# Separating features and target
X = df[["Income", "Debt", "Payment_History"]]
y = df["Credit_Label"]

print("\nFeatures:")
print(X)

print("\nTarget:")
print(y)

# Saving dataset
df.to_csv("credit_data.csv", index=False)

print("\nDataset saved successfully!")