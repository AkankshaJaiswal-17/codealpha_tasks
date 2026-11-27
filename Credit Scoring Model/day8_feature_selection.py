import pandas as pd
import joblib
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline


# Project path
base_path = Path(__file__).parent

data_path = (base_path / "data" / "LT_Vehicle_Loan" / "train_feature_engineered.csv")

#Load dataset
df = pd.read_csv(data_path)

print("===== DAY 8: FEATURE SELECTION =====")

print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nTarget Distribution:")
print(df["loan_default"].value_counts())

# Step 3: Separate features and target

columns_to_drop = [
    "loan_default",
    "DisbursalDate",
    "Disbursal_Year",
    "PERFORM_CNS.SCORE.DESCRIPTION",
    "AVERAGE.ACCT.AGE",
    "CREDIT.HISTORY.LENGTH",
    "Current_pincode_ID",
    "MobileNo_Avl_Flag",
    "Aadhar_flag",
    "PAN_flag",
    "VoterID_flag",
    "Driving_flag",
    "Passport_flag"
]

X = df.drop(columns=columns_to_drop)
y = df["loan_default"]

print("\n===== FEATURES AND TARGET =====")

print("X Shape:", X.shape)
print("y Shape:", y.shape)

print("\nFeature Columns:")
print(X.columns.tolist())

print("\nTarget Distribution:")
print(y.value_counts())

# Step 4: Train-Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n===== TRAIN-TEST SPLIT =====")

print("Training Features:", X_train.shape)
print("Testing Features:", X_test.shape)

print("Training Target:", y_train.shape)
print("Testing Target:", y_test.shape)

print("\nTraining Target Distribution:")
print(y_train.value_counts(normalize=True) * 100)

print("\nTesting Target Distribution:")
print(y_test.value_counts(normalize=True) * 100)

print("===== TRAIN-TEST SPLIT =====")

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

print("\nTraining Target Distribution (%):")
print((y_train.value_counts(normalize=True) * 100).round(2))

print("\nTesting Target Distribution (%):")
print((y_test.value_counts(normalize=True) * 100).round(2))

print("===== STEP 5: FEATURE TYPE IDENTIFICATION =====")

# Remove high-cardinality and location-related ID columns
columns_to_remove = [
    "branch_id",
    "supplier_id",
    "State_ID"
]

X_train_model = X_train.drop(
    columns=columns_to_remove
).copy()

X_test_model = X_test.drop(
    columns=columns_to_remove
).copy()

# Categorical columns
categorical_features = [
    "Employment.Type",
    "manufacturer_id",
    "Disbursal_Month"
]

# Numerical columns
numerical_features = [
    col for col in X_train_model.columns
    if col not in categorical_features
]

print("\nCategorical Features:")
print(categorical_features)

print("\nNumerical Features:")
print(numerical_features)

print("\nNumber of Categorical Features:",
      len(categorical_features))

print("Number of Numerical Features:",
      len(numerical_features))

print("\nFinal Training Shape:", X_train_model.shape)
print("Final Testing Shape:", X_test_model.shape)

# Numerical preprocessing
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine both transformers
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("Preprocessing pipeline created successfully!")

X_train_processed = preprocessor.fit_transform(X_train_model)

X_test_processed = preprocessor.transform(X_test_model)

print("\n===== PREPROCESSING RESULTS =====")

print("Training Data Shape:", X_train_processed.shape)
print("Testing Data Shape:", X_test_processed.shape)
print("Training Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)

# Save preprocessing pipeline
model_path = base_path / "preprocessor.pkl"

joblib.dump(preprocessor, model_path)

print("\nPreprocessor saved successfully!")
print("Saved at:", model_path)