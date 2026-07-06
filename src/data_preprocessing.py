"""
=====================================================================
Project:
AI-IoT-Based Early Dengue Prediction and Prevention System

Author:
Nazmul Hassan

Degree:
Master of Science (M.Sc.) in Computer Science & Engineering
Major: Cybersecurity

University:
United International University (UIU), Dhaka, Bangladesh

Research Supervisor:
Prof. Khondaker A. Mamun, PhD
Professor, Department of Computer Science & Engineering
Director, MSCSE Program & Director, IRIIC
United International University (UIU)

Research Paper:
AI- and IoT-Based Early Dengue Prediction and Prevention System

Research Area:
Artificial Intelligence (AI), Internet of Things (IoT),
Machine Learning, Smart Healthcare, Dengue Prediction

Programming Language:
Python 3.12+

Dataset:
Dengue Dataset Bangladesh (Kaggle)

Repository:
https://github.com/nazmullhassan/ai-iot-dengue-prediction-system

Copyright © 2026 Nazmul Hassan
=====================================================================
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

DATASET_FILE = RAW_DATA_DIR / "dengue_dataset.csv"

TRAIN_DATASET_FILE = PROCESSED_DATA_DIR / "train_dataset.csv"
TEST_DATASET_FILE = PROCESSED_DATA_DIR / "test_dataset.csv"

TEST_SIZE = 0.20
RANDOM_STATE = 42


def load_dataset():
    """
    Load the Dengue dataset from the raw data folder.
    """

    print("=" * 70)
    print("Loading Dengue Dataset...")
    print("=" * 70)

    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"\nDataset not found!\nPlease place the dataset here:\n{DATASET_FILE}"
        )

    df = pd.read_csv(DATASET_FILE)

    print("Dataset loaded successfully.")
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nFirst Five Records:")
    print(df.head())

    return df


def main():
    """
    Main execution function.
    """

    dataset = load_dataset()

    print("\nDataset Information:")
    dataset.info()


if __name__ == "__main__":
    main()