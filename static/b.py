import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('../models/indian_student_dataset_simple.csv')

# Numerical columns to scale
numerical_cols = [
    'absence_days', 'weekly_self_study_hours',
    'math_score', 'history_score', 'physics_score',
    'chemistry_score', 'biology_score', 'english_score', 'geography_score'
]

# Check column existence
for col in numerical_cols:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in dataset!")

# Ensure all columns are numeric and handle missing values
df[numerical_cols] = df[numerical_cols].apply(pd.to_numeric, errors='coerce')  # force conversion
df[numerical_cols] = df[numerical_cols].fillna(0)  # or use df.mean() if preferred

# Fit scaler
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Save scaler
joblib.dump(scaler, '../models/scaler.pkl')

print("✅ Scaler saved as 'scaler.pkl'")

# Fit and save scaler
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
joblib.dump(scaler, '../models/scaler.pkl')

print("✅ Scaler saved as 'scaler.pkl'")
