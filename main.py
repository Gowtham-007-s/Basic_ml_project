# 1. Import libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Load dataset
data = pd.read_csv("attention_data.csv")

print("\nDataset Preview:")
print(data.head())

# 3. Separate features and target
X = data.drop("attention_level", axis=1)
y = data["attention_level"]

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 7. Predictions
y_pred = model.predict(X_test_scaled)

# 8. Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 9. New prediction
new_student = np.array([[40, 2, 90, 150, 1, 25]])
new_student_scaled = scaler.transform(new_student)
prediction = model.predict(new_student_scaled)

labels = ["Focused", "Distracted", "Fatigued"]

print("\nNew Student Data:", new_student)
print("Predicted Attention State:", labels[prediction[0]])

