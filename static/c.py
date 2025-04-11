import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('../models/indian_student_dataset_simple.csv')

# Columns
categorical_cols = ['gender', 'part_time_job', 'extracurricular_activities', 'career_aspiration']
numerical_cols = [
    'absence_days', 'weekly_self_study_hours', 'math_score', 'history_score',
    'physics_score', 'chemistry_score', 'biology_score', 'english_score', 'geography_score'
]

# ✅ Handle missing values
df[categorical_cols] = df[categorical_cols].fillna('Unknown')
df[numerical_cols] = df[numerical_cols].fillna(0)

# ✅ Apply label encoders
for col in categorical_cols:
    le = joblib.load(f'label_encoder_{col}.pkl')
    df[col] = le.transform(df[col].astype(str))

# ✅ Apply scaler to numerical features
scaler = joblib.load('../models/scaler.pkl')
df[numerical_cols] = scaler.transform(df[numerical_cols])

# ✅ Prepare X and y
X = df.drop(columns=['name', 'email_id', 'email_password', 'career_aspiration'])
y = df['career_aspiration']  # Already encoded above

# ✅ Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ✅ Save model
joblib.dump(model, '../models/model.pkl')

print("✅ Model saved as 'model.pkl'")
