import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import os

# ✅ Load dataset with absolute or relative path
df = pd.read_csv('../models/indian_student_dataset_simple.csv')  # Or use full path if needed

# ✅ Categorical columns to encode
categorical_cols = ['gender', 'part_time_job', 'extracurricular_activities', 'career_aspiration']

# ✅ Handle missing values if any
df[categorical_cols] = df[categorical_cols].fillna('Unknown')

# ✅ Encode and save label encoders
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    joblib.dump(le, f'label_encoder_{col}.pkl')

print("✅ Label encoders saved as 'label_encoder_*.pkl'")
