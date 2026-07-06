"""
Hybrid XGBoost-LSTM Model
AI-IoT-Based Early Dengue Prediction and Prevention System
Author: Nazmul Hassan
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_DATA = PROJECT_ROOT / "data" / "processed" / "train_dataset.csv"
TEST_DATA = PROJECT_ROOT / "data" / "processed" / "test_dataset.csv"

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

TARGET = "Outcome"


def load_data():
    train_df = pd.read_csv(TRAIN_DATA)
    test_df = pd.read_csv(TEST_DATA)

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]

    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    return X_train, X_test, y_train, y_test


def train_xgboost(X_train, X_test, y_train):
    model = XGBClassifier(
        random_state=42,
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="logloss",
    )

    model.fit(X_train, y_train)

    train_prob = model.predict_proba(X_train)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    joblib.dump(model, MODEL_DIR / "hybrid_xgboost_component.pkl")

    return train_prob, test_prob


def prepare_lstm_input(X_train, X_test, xgb_train_prob, xgb_test_prob):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, MODEL_DIR / "hybrid_scaler.pkl")

    X_train_hybrid = np.column_stack((X_train_scaled, xgb_train_prob))
    X_test_hybrid = np.column_stack((X_test_scaled, xgb_test_prob))

    X_train_lstm = X_train_hybrid.reshape(
        X_train_hybrid.shape[0],
        X_train_hybrid.shape[1],
        1,
    )

    X_test_lstm = X_test_hybrid.reshape(
        X_test_hybrid.shape[0],
        X_test_hybrid.shape[1],
        1,
    )

    return X_train_lstm, X_test_lstm


def build_lstm_model(input_shape):
    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.30),
            LSTM(32),
            Dropout(0.30),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def save_training_curves(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Hybrid XGBoost-LSTM Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "hybrid_accuracy_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Hybrid XGBoost-LSTM Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "hybrid_loss_curve.png", dpi=300)
    plt.close()


def train_hybrid_model():
    X_train, X_test, y_train, y_test = load_data()

    xgb_train_prob, xgb_test_prob = train_xgboost(X_train, X_test, y_train)

    X_train_lstm, X_test_lstm = prepare_lstm_input(
        X_train,
        X_test,
        xgb_train_prob,
        xgb_test_prob,
    )

    model = build_lstm_model(
        input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        X_train_lstm,
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.20,
        callbacks=callbacks,
        verbose=1,
    )

    probabilities = model.predict(X_test_lstm)
    predictions = (probabilities >= 0.5).astype(int).flatten()

    accuracy = accuracy_score(y_test, predictions)

    print("\nHybrid XGBoost-LSTM Accuracy:", round(accuracy * 100, 2), "%")

    print("\nClassification Report\n")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix\n")
    print(confusion_matrix(y_test, predictions))

    save_training_curves(history)

    model.save(MODEL_DIR / "hybrid_xgboost_lstm_model.keras")

    print("\nHybrid model saved successfully.")
    print("Training curves saved successfully.")


if __name__ == "__main__":
    train_hybrid_model()