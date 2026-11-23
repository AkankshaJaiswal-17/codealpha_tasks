import pandas as pd

#Load dataset
data = pd.read_csv("Credit Scoring Model/credit_data.csv")

#Display first 5 rows 
print("First 5 Rows:")
print(data.head())

#Dataset shape
print("\nDataset Shape:")
print(data.shape)

#Column Information
print("\nDataset Information")
print(data.info())

#Missing Value
print("Missing Values:")
print(data.isnull().sum())

#Duplicate records
print("\nDuplicate Records:")
print(data.duplicated().sum())

#Statistical summary
print("\nStatistical Summary:")
print(data.describe())
