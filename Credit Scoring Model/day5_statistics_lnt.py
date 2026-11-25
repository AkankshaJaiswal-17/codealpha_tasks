import pandas as pd
from pathlib import Path

base_path = Path(__file__).parent
data_path = base_path / "data" / "LT_Vehicle_Loan" / "train.csv"

df = pd.read_csv(data_path)

# Numerical summary
print("===== NUMERICAL STATISTICS =====")
print(df.describe())

# Categorical summary
print("\n===== CATEGORICAL STATISTICS =====")
print(df.describe(include="str"))

# Target distribution
print("\n===== TARGET DISTRIBUTION =====")
print(df["loan_default"].value_counts(normalize=True) * 100)

# DATA DICTIONARY
dictionary_path = (
    base_path / "data" / "LT_Vehicle_Loan" / "data_dictionary.csv"
)

data_dictionary = pd.read_csv(dictionary_path)

print("\n===== DATA DICTIONARY =====")
print(data_dictionary.to_string(index=False))

# IMPORTANT FEATURE ANALYSIS

print("\n===== CREDIT SCORE DISTRIBUTION =====")
print(df["PERFORM_CNS.SCORE"].value_counts().sort_index().head(20))

print("\n===== LOAN AMOUNT RANGE =====")
print(df["disbursed_amount"].agg(["min", "max", "mean", "median"]))

print("\n===== EMPLOYMENT TYPE =====")
print(df["Employment.Type"].value_counts(dropna=False))

print("\n===== DATE FORMATS =====")
print(df[["Date.of.Birth", "DisbursalDate"]].head(10))

print("\n===== CREDIT HISTORY EXAMPLES =====")
print(df[["AVERAGE.ACCT.AGE", "CREDIT.HISTORY.LENGTH"]].head(10))

print("\n===== CREDIT SCORE DESCRIPTION =====")
print(df["PERFORM_CNS.SCORE.DESCRIPTION"].value_counts())

print("\n===== CREDIT SCORE AND DESCRIPTION =====")
print(
    df.groupby(
        ["PERFORM_CNS.SCORE", "PERFORM_CNS.SCORE.DESCRIPTION"]
    ).size().head(25)
)

print("\n===== DEFAULT RATE BY CREDIT SCORE DESCRIPTION =====")

default_analysis = (
    df.groupby("PERFORM_CNS.SCORE.DESCRIPTION")["loan_default"]
    .agg(["count", "mean"])
    .sort_values(by="mean", ascending=False)
)

default_analysis["default_percentage"] = (
    default_analysis["mean"] * 100
)

print(default_analysis)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

default_analysis["default_percentage"].sort_values().plot(
    kind="barh"
)

plt.title("Default Rate by Credit Score Description")
plt.xlabel("Default Percentage")
plt.ylabel("Credit Score Category")
plt.tight_layout()

plt.savefig(
    base_path / "credit_score_default_analysis.png",
    dpi=300
)

plt.show()