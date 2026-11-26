import pandas as pd
from pathlib import Path

# Step 1: Load preprocessed dataset

base_path = Path(__file__).parent

data_path = (
    base_path / "data" / "LT_Vehicle_Loan"
    / "train_preprocessed.csv"
)

df = pd.read_csv(data_path)

print("===== DAY 7: FEATURE ENGINEERING =====")

print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Data Types:")
print(df.dtypes)

# Step 2: Identify feature types

print("\n===== CATEGORICAL COLUMNS =====")

categorical_columns = [
    "Employment.Type",
    "PERFORM_CNS.SCORE.DESCRIPTION",
    "branch_id",
    "supplier_id",
    "manufacturer_id",
    "Current_pincode_ID",
    "State_ID"
]

for col in categorical_columns:
    print(
        col,
        "-> Unique values:",
        df[col].nunique()
    )


print("\n===== DATE AND DURATION COLUMNS =====")

date_duration_columns = [
    "DisbursalDate",
    "AVERAGE.ACCT.AGE",
    "CREDIT.HISTORY.LENGTH"
]

for col in date_duration_columns:
    print("\n", col)
    print(df[col].head())

# Step 3: Convert duration columns into months

def convert_to_months(value):

    parts = value.split()

    years = int(parts[0].replace("yrs", ""))
    months = int(parts[1].replace("mon", ""))

    return years * 12 + months


df["AVERAGE.ACCT.AGE.MONTHS"] = (
    df["AVERAGE.ACCT.AGE"].apply(convert_to_months)
)

df["CREDIT.HISTORY.LENGTH.MONTHS"] = (
    df["CREDIT.HISTORY.LENGTH"].apply(convert_to_months)
)

print("\n===== DURATION CONVERSION =====")

print(df[
    [
        "AVERAGE.ACCT.AGE",
        "AVERAGE.ACCT.AGE.MONTHS",
        "CREDIT.HISTORY.LENGTH",
        "CREDIT.HISTORY.LENGTH.MONTHS"
    ]
].head(10))

print("\n===== FEATURE VERIFICATION =====")

print("Average Account Age (Months):")
print(df["AVERAGE.ACCT.AGE.MONTHS"].describe())

print("\nCredit History Length (Months):")
print(df["CREDIT.HISTORY.LENGTH.MONTHS"].describe())

print("\nMissing Values:")
print(
    df[
        [
            "AVERAGE.ACCT.AGE.MONTHS",
            "CREDIT.HISTORY.LENGTH.MONTHS"
        ]
    ].isnull().sum()
)

print("\nNew Dataset Shape:", df.shape)

# Step 5: Extract date-based features

df["DisbursalDate"] = pd.to_datetime(df["DisbursalDate"])

df["Disbursal_Month"] = df["DisbursalDate"].dt.month
df["Disbursal_Year"] = df["DisbursalDate"].dt.year

print("\n===== DATE FEATURE ENGINEERING =====")

print(df[
    ["DisbursalDate", "Disbursal_Month", "Disbursal_Year"]
].head(10))

print("\nMonth Distribution:")
print(df["Disbursal_Month"].value_counts().sort_index())

print("\nYear Distribution:")
print(df["Disbursal_Year"].value_counts().sort_index())

# Step 6: Inspect categorical features

categorical_columns = [
    "Employment.Type",
    "PERFORM_CNS.SCORE.DESCRIPTION",
    "manufacturer_id",
    "branch_id",
    "State_ID"
]

print("\n===== CATEGORICAL FEATURE INSPECTION =====")

for col in categorical_columns:
    print(f"\n{col}")
    print("Unique values:", df[col].nunique())
    print(df[col].value_counts().head(10))

# Step 7: Compare credit score and description

print("\n===== CREDIT SCORE AND DESCRIPTION =====")

print(
    df.groupby("PERFORM_CNS.SCORE.DESCRIPTION")[
        "PERFORM_CNS.SCORE"
    ].agg(["count", "min", "max", "nunique"])
    .sort_values("count", ascending=False)
)

# Step 8: Credit score vs loan default

print("\n===== CREDIT SCORE VS LOAN DEFAULT =====")

score_analysis = df.groupby("PERFORM_CNS.SCORE").agg(
    Total_Customers=("loan_default", "count"),
    Defaults=("loan_default", "sum")
)

score_analysis["Default_Rate"] = (
    score_analysis["Defaults"] /
    score_analysis["Total_Customers"] * 100
)

print(
    score_analysis.sort_values(
        "Total_Customers", ascending=False
    ).head(15)
)

# Step 9: Create bureau history indicator

special_scores = [0, 11, 14, 15, 16, 17, 18]

df["Has_Normal_Bureau_Score"] = (
    ~df["PERFORM_CNS.SCORE"].isin(special_scores)
).astype(int)

print("\n===== BUREAU HISTORY INDICATOR =====")

print(
    df["Has_Normal_Bureau_Score"].value_counts()
)

print("\nDefault Rate by Bureau Score Availability:")

print(
    df.groupby("Has_Normal_Bureau_Score")["loan_default"]
    .agg(["count", "sum", "mean"])
)

print("\n===== FINAL FEATURE CHECK =====")

print("Dataset Shape:", df.shape)

print("\nNew Features:")
print(
    df[
        [
            "AVERAGE.ACCT.AGE.MONTHS",
            "CREDIT.HISTORY.LENGTH.MONTHS",
            "Disbursal_Month",
            "Has_Normal_Bureau_Score"
        ]
    ].head()
)

print("\nMissing Values in New Features:")
print(
    df[
        [
            "AVERAGE.ACCT.AGE.MONTHS",
            "CREDIT.HISTORY.LENGTH.MONTHS",
            "Disbursal_Month",
            "Has_Normal_Bureau_Score"
        ]
    ].isnull().sum()
)

# Step 11: Save engineered dataset

output_path = (
    base_path / "data" / "LT_Vehicle_Loan"
    / "train_feature_engineered.csv"
)

df.to_csv(output_path, index=False)

print("\n===== DATASET SAVED =====")
print("Saved at:", output_path)
print("Final Shape:", df.shape)