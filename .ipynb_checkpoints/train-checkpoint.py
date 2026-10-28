"""Local training script (same logic as the Colab notebook).
Assignment ke liye Colab notebook use karo; ye script sirf backup hai."""
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/diabetes.csv")
X = df.drop("Outcome", axis=1).values
y = df["Outcome"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print("Accuracy:", acc)
print(classification_report(y_test, pred))

joblib.dump(model, "model.joblib")
with open("model_params.json", "w") as f:
    json.dump({"accuracy": acc, **model.get_params()}, f, indent=2)
