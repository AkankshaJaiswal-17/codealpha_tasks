import pandas as pd
from pathlib import Path

# Load dataset
base_path = Path(__file__).parent
data_path = base_path / "data" / "LT_Vehicle_Loan" / "train.csv"

df = pd.read_csv(data_path)

# 1. Dataset information
print("===== DATASET INFORMATION =====")
df.info()

# 2. Missing values
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# 3. Total missing values
print("\nTotal Missing Values:")
print(df.isnull().sum().sum())

# 4. Duplicate records
print("\n===== DUPLICATE RECORDS =====")
print(df.duplicated().sum())

# 5. Data types
print("\n===== DATA TYPES =====")
print(df.dtypes)

# 6. Target distribution
print("\n===== LOAN DEFAULT DISTRIBUTION =====")
print(df["loan_default"].value_counts())

print("\n===== TARGET PERCENTAGES =====")
print(df["loan_default"].value_counts(normalize=True) * 100)

# Missing value analysis
missing = df.isnull().sum()

missing_data = pd.DataFrame({
    "Missing_Count": missing,
    "Missing_Percentage": (missing / len(df)) * 100
})

missing_data = missing_data[missing_data["Missing_Count"] > 0]

missing_data = missing_data.sort_values(
    by="Missing_Count",
    ascending=False
)

print("\n===== MISSING VALUE ANALYSIS =====")
print(missing_data)