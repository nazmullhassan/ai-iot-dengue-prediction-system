"""
XGBoost Training Module
AI-IoT-Based Early Dengue Prediction and Prevention System
"""

from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_DATA = PROJECT_ROOT / "data" / "processed" / "train_dataset.csv"
TEST_DATA = PROJECT_ROOT / "data" / "processed" / "test_dataset.csv"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "Outcome"


def load_data():
    train_df = pd.read_csv(TRAIN_DATA)
    test_df = pd.read_csv(TEST_DATA)

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]

    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    return X_train, X_test, y_train, y_test


def train_model():

    X_train, X_test, y_train, y_test = load_data()

    model = XGBClassifier(
        random_state=42,
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\nAccuracy:", round(accuracy * 100, 2), "%")

    print("\nClassification Report\n")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix\n")
    print(confusion_matrix(y_test, predictions))

    joblib.dump(model, MODEL_DIR / "xgboost_model.pkl")

    print("\nModel saved successfully.")


if __name__ == "__main__":
    train_model()