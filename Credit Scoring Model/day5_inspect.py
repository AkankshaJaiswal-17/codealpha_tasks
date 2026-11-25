import pandas as pd

#Load dataset
df = pd.read_csv("Credit Scoring Model/data/LoanTapData.csv")

#Basic Information
print("Dataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
df.info()