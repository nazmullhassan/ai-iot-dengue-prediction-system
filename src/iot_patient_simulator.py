"""
IoT Patient Simulator
AI-IoT-Based Early Dengue Prediction and Prevention System

Author:
Nazmul Hassan
"""

import random
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = RESULTS_DIR / "iot_sensor_data.csv"


def generate_patient():

    return {

        "Temperature": round(random.uniform(97.0,104.5),2),

        "HeartRate": random.randint(60,140),

        "Humidity": random.randint(55,95),

        "Latitude": 23.8103,

        "Longitude": 90.4125,

        "Platelet_Count": random.randint(30000,300000),

        "WBC_Count": random.randint(2500,11000),

        "Timestamp": pd.Timestamp.now()

    }


records=[]

print("="*60)
print("IoT Patient Monitoring Started")
print("="*60)

for i in range(20):

    patient=generate_patient()

    records.append(patient)

    print(patient)

    time.sleep(1)

df=pd.DataFrame(records)

df.to_csv(OUTPUT_FILE,index=False)

print()

print("Simulation completed.")

print(f"Saved to: {OUTPUT_FILE}")