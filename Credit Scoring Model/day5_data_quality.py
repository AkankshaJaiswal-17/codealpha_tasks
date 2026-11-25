import pandas as pd

df = pd.read_csv("Credit Scoring Model/data/LoanTapData.csv")

#Missing values
print("Missing Values:")
print(df.isnull().sum())

#Total missing values
print("\nTotal Missing Values:")
print(df.isnull().sum().sum())

#Duplicate records
print("\nDuplicate Rows:")
print(df.duplicated().sum())

#Data types
print("\nColumn Data Types:")
print(df.dtypes)

#Unique values in each column
print("\nUnique Values:")
print(df.nunique())

#Categorical columns
categorical_column = df.select_dtypes(include="object").columns

print("\nCategorical Columns:")
print(categorical_column.tolist())

#Unique values in categorical columns
for col in categorical_column:
    print(f"\n{col}:")

print(df[col].value_counts().head(10))

print("\nLoan Status Distribution:")
print(df["loan_status"].value_counts(dropna=False))

print("\nLoan Status Percentage:")
print(df["loan_status"].value_counts(normalize=True)*100)

# Loan status percentage
print("\nLoan Status Percentage:")
status_percentage = df["loan_status"].value_counts(normalize=True) * 100
print(status_percentage.round(2))

print("\nAll Dataset Columns:")
for i, column in enumerate(df.columns, start=1):
    print(i, column)

important_columns = [
    "loan_amnt",
    "annual_inc",
    "dti",
    "grade",
    "sub_grade",
    "emp_length",
    "loan_status"
]

print("\nImportant Column Missing Values:")
print(df[important_columns].isnull().sum())

# Missing value percentage
missing_percentage = (df.isnull().sum() / len(df)) * 100

print("\nMissing Value Percentage:")
print(missing_percentage[missing_percentage > 0].sort_values(ascending=False).round(2))

numerical_columns = [
    "mort_acc",
    "pub_rec_bankruptcies",
    "revol_util"
]

print("\nNumerical Column Statistics:")
print(df[numerical_columns].describe())

check_columns = [
    "loan_amnt",
    "int_rate",
    "annual_inc",
    "dti",
    "revol_util"
]

print("\nMinimum and Maximum Values:")

for col in check_columns:
    print(f"\n{col}")
    print("Minimum:", df[col].min())
    print("Maximum:", df[col].max())

check_columns = ["annual_inc" , "dti" , "revol_util"]

for col in check_columns:
    print(f"\nTop 10 values in {col}:")
    print(df[col].nlargest(10))

print("\nSuspicious DTI Records:")
print("DTI above 100:", (df["dti"] > 100).sum())

print("\nSuspicious Revolving Utilization:")
print("Revol Util above 100:", (df["revol_util"] > 100).sum())

print("\nDTI equal to 9999:")
print((df["dti"] == 9999).sum())