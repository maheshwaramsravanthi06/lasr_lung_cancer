import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv("dataset.csv")

print("Columns in dataset:")
print(data.columns)

# Encode categorical column (GENDER)
data["GENDER"] = data["GENDER"].map({"M": 1, "F": 0})

# Encode target variable
data["LUNG_CANCER"] = data["LUNG_CANCER"].map({"YES": 1, "NO": 0})

# Features and target
X = data.drop("LUNG_CANCER", axis=1)
y = data["LUNG_CANCER"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test data
pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, pred)
print("Model Accuracy:", accuracy)

# Show feature importance
importance = model.feature_importances_
print("Feature Importance:", importance)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model trained successfully!")