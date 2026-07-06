"""
IoT Risk Prediction and Alert Module
AI-IoT-Based Early Dengue Prediction and Prevention System
Author: Nazmul Hassan
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

IOT_DATA_FILE = PROJECT_ROOT / "results" / "iot_sensor_data.csv"
OUTPUT_FILE = PROJECT_ROOT / "results" / "iot_risk_predictions.csv"


def classify_risk(row):
    """
    Rule-based dengue risk classification from simulated IoT sensor values.
    """

    risk_score = 0

    if row["Temperature"] >= 102:
        risk_score += 35

    if row["Humidity"] >= 80:
        risk_score += 20

    if row["Platelet_Count"] < 150000:
        risk_score += 25

    if row["WBC_Count"] < 4500:
        risk_score += 10

    if row["HeartRate"] >= 110:
        risk_score += 10

    if risk_score >= 70:
        risk_level = "High"
        alert = "Immediate medical attention recommended"
    elif risk_score >= 40:
        risk_level = "Moderate"
        alert = "Monitor closely and repeat assessment"
    else:
        risk_level = "Low"
        alert = "Routine monitoring"

    return pd.Series([risk_score, risk_level, alert])


def main():
    if not IOT_DATA_FILE.exists():
        raise FileNotFoundError(
            f"IoT sensor data not found. Run iot_patient_simulator.py first.\n{IOT_DATA_FILE}"
        )

    df = pd.read_csv(IOT_DATA_FILE)

    df[["Risk_Score", "Risk_Level", "Alert"]] = df.apply(
        classify_risk,
        axis=1,
    )

    df.to_csv(OUTPUT_FILE, index=False)

    print("IoT risk prediction completed successfully.")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nRisk Summary:")
    print(df["Risk_Level"].value_counts())

    print("\nHigh Risk Alerts:")
    print(df[df["Risk_Level"] == "High"][["Timestamp", "Temperature", "Humidity", "Platelet_Count", "Risk_Score", "Alert"]])


if __name__ == "__main__":
    main()