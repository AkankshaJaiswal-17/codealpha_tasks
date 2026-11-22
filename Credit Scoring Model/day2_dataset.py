import pandas as pd

# Creating a sample dataset
data = {
    'Income': [50000, 25000, 80000, 30000, 60000],
    'Debt': [5000, 20000, 10000, 15000, 8000],
    'Payment_History': ['Good', 'Poor', 'Good', 'Poor', 'Good'],
    'Credit_Label': ['Good', 'Poor', 'Good', 'Poor', 'Good']
}

# Convert data into a DataFrame
df = pd.DataFrame(data)

# Display dataset
print("Our Dataset:")
print(df)

# Display dataset information
print("\nDataset Shape:")
print(df.shape)

# Separate features and target
X = df[['Income', 'Debt', 'Payment_History']]
y = df['Credit_Label']

print("\nFeatures:")
print(X)

print("\nTarget:")
print(y)