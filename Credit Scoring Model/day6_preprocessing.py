import pandas as pd
from pathlib import Path

# Step 1: Locate dataset
base_path = Path(__file__).parent

data_path = (
    base_path
    / "data"
    / "LT_Vehicle_Loan"
    / "train.csv"
)

# Step 2: Load original dataset
df = pd.read_csv(data_path)

print("===== BEFORE PREPROCESSING =====")
print("Missing values:")
print(df.isnull().sum())

print("\nMissing Employment Type:")
print(df["Employment.Type"].isnull().sum())

# Step 3: Handle missing values
df["Employment.Type"] = (
    df["Employment.Type"].fillna("Unknown")
)

# Step 4: Verify
print("\n===== AFTER PREPROCESSING =====")
print("Missing Employment Type:")
print(df["Employment.Type"].isnull().sum())

print("\nEmployment Type Distribution:")
print(df["Employment.Type"].value_counts())

print("\nDataset Shape:")
print(df.shape)

# Step 5: Convert date columns

df["Date.of.Birth"] = pd.to_datetime(
    df["Date.of.Birth"],
    format="%d-%m-%y"
)

df["DisbursalDate"] = pd.to_datetime(
    df["DisbursalDate"],
    format="%d-%m-%y"
)

# Step 6: Verify date conversion

print("\n===== DATE CONVERSION =====")
print(df[["Date.of.Birth", "DisbursalDate"]].head())

print("\nData Types:")
print(df[["Date.of.Birth", "DisbursalDate"]].dtypes)

# Correct DOB years that were interpreted as future dates
future_birth = df["Date.of.Birth"] > df["DisbursalDate"]

df.loc[future_birth, "Date.of.Birth"] = (
    df.loc[future_birth, "Date.of.Birth"]
    - pd.DateOffset(years=100)
)

print("Future DOB records after correction:")
print((df["Date.of.Birth"] > df["DisbursalDate"]).sum())

print("\nCorrected DOB range:")
print(df["Date.of.Birth"].min())
print(df["Date.of.Birth"].max())

df["Age"] = (
    df["DisbursalDate"].dt.year
    - df["Date.of.Birth"].dt.year
)

birthday_not_reached = (
    (df["DisbursalDate"].dt.month < df["Date.of.Birth"].dt.month)
    |
    (
        (df["DisbursalDate"].dt.month == df["Date.of.Birth"].dt.month)
        &
        (df["DisbursalDate"].dt.day < df["Date.of.Birth"].dt.day)
    )
)

df.loc[birthday_not_reached, "Age"] -= 1

print("\n===== CORRECTED AGE ANALYSIS =====")
print(df["Age"].describe())
print("\nFirst 10 Age Values:")
print(df["Age"].head(10))

# Step 8: Identify unnecessary columns

print("\n===== COLUMN ANALYSIS =====")

print("Total Columns:", len(df.columns))

# Identifier columns
identifier_columns = [
    "UniqueID",
    "Employee_code_ID"
]

print("\nIdentifier Columns:")
for col in identifier_columns:
    if col in df.columns:
        print(col, "->", df[col].nunique(), "unique values")

# Check remaining columns
print("\nAll Dataset Columns:")
print(df.columns.tolist())

# Step 9: Remove unnecessary identifier columns

columns_to_drop = [
    "UniqueID",
    "Employee_code_ID",
    "Date.of.Birth"
]

df = df.drop(columns=columns_to_drop)

print("\n===== AFTER REMOVING IDENTIFIERS =====")

print("Dataset Shape:", df.shape)

print("\nRemaining Columns:")
print(df.columns.tolist())

# Step 10: Check remaining missing values

print("\n===== MISSING VALUES AFTER PREPROCESSING =====")

missing_values = df.isnull().sum()

print(missing_values[missing_values > 0])

print("\nTotal Missing Values:")
print(df.isnull().sum().sum())

print("\nDataset Shape:")
print(df.shape)

# Step 11: Check duplicate records

print("\n===== DUPLICATE RECORD ANALYSIS =====")

duplicate_count = df.duplicated().sum()

print("Total Duplicate Rows:", duplicate_count)

if duplicate_count > 0:
    print("Duplicate records found.")
else:
    print("No duplicate records found.")

# Step 12: Inspect duplicate records

duplicates = df[df.duplicated(keep=False)]

print("\n===== DUPLICATE RECORDS =====")
print("Total rows involved:", len(duplicates))

print(duplicates.head(20))

print("\nDuplicate Target Distribution:")
print(duplicates["loan_default"].value_counts())

# Step 13: Target variable distribution

print("\n===== TARGET VARIABLE DISTRIBUTION =====")

print(df["loan_default"].value_counts())

print("\nPercentage Distribution:")
print(
    df["loan_default"].value_counts(normalize=True) * 100
)

# Step 14: Save preprocessed dataset

output_path = (
    base_path / "data" / "LT_Vehicle_Loan"
    / "train_preprocessed.csv"
)

df.to_csv(output_path, index=False)

print("\n===== DATASET SAVED =====")
print("File saved at:", output_path)
print("Final Dataset Shape:", df.shape)

# Step 15: Verify saved dataset

processed_df = pd.read_csv(output_path)

print("\n===== SAVED DATASET VERIFICATION =====")

print("Dataset Shape:", processed_df.shape)

print("\nMissing Values:")
print(processed_df.isnull().sum().sum())

print("\nDuplicate Rows:")
print(processed_df.duplicated().sum())

print("\nTarget Distribution:")
print(processed_df["loan_default"].value_counts())

print("\nFirst 5 Rows:")
print(processed_df.head())