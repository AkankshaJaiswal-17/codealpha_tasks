import pandas as pd
from pathlib import Path

# Step 1: Locate the dataset
base_path = Path(__file__).parent

data_path = (
    base_path
    / "data"
    / "LT_Vehicle_Loan"
    / "train.csv"
)

# Step 2: Load the dataset
df = pd.read_csv(data_path)

# Step 3: Display dataset information
print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())