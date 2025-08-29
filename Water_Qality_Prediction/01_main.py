import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

MODEL_FILE = "model1.pkl"

df = pd.read_csv("water_quality_dataset_100k_preprocessed.csv")

num_features = df.select_dtypes(include=[np.number]).columns.tolist()
num_features.remove('Target')
non_feature_cols = ['Target', 'Month', 'Day', 'Time of Day']
num_features = [col for col in num_features if col not in non_feature_cols]

# Feature Engineering
df['water_tem_to_Air_ratio'] = df['Water Temperature'] / df['Air Temperature']
df['total_metals'] = df[['Iron', 'Lead', 'Copper', 'Zinc', 'Manganese']].sum(axis=1)
num_features.extend(['water_tem_to_Air_ratio', 'total_metals'])

X = df[num_features].copy()
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train or Load Model

if not os.path.exists(MODEL_FILE):
    print("Training model...")
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)
    joblib.dump(model_rf, MODEL_FILE)
    print("✅ Model trained and saved as 'model1.pkl'")
else:
    print("Loading saved model...")
    model_rf = joblib.load(MODEL_FILE)


# Save Input CSV

X_test_copy = X_test.copy()
X_test_copy.to_csv("input1.csv", index=False)
print("✅ 'input1.csv' saved (test features only).")

# Run Inference
print("Running inference...")
predictions = model_rf.predict(X_test)

X_test_with_preds = X_test.copy()
X_test_with_preds['Predicted_Target'] = predictions
X_test_with_preds['Actual_Target'] = y_test.values 

X_test_with_preds.to_csv("output1.csv", index=False)
print("✅ 'output1.csv' saved (test features + predictions).")
