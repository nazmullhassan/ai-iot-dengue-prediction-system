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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

DATASET_FILE = RAW_DATA_DIR / "dengue_dataset.csv"

TRAIN_DATASET_FILE = PROCESSED_DATA_DIR / "train_dataset.csv"
TEST_DATASET_FILE = PROCESSED_DATA_DIR / "test_dataset.csv"

TEST_SIZE = 0.20
RANDOM_STATE = 42
TARGET_COLUMN = "Outcome"


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


def inspect_missing_values(df):
    """
    Display missing value summary.
    """

    print("\n" + "=" * 70)
    print("Missing Value Summary")
    print("=" * 70)

    missing_summary = df.isnull().sum()
    missing_summary = missing_summary[missing_summary > 0]

    if missing_summary.empty:
        print("No missing values found.")
    else:
        print(missing_summary)

    return missing_summary


def clean_missing_values(df):
    """
    Handle missing values in the dataset.

    Joint_Pain contains many missing values. Since it is a categorical
    symptom feature, missing values are filled using the most frequent value.
    """

    print("\n" + "=" * 70)
    print("Cleaning Missing Values")
    print("=" * 70)

    df = df.copy()

    if "Joint_Pain" in df.columns:
        most_frequent_value = df["Joint_Pain"].mode()[0]
        df["Joint_Pain"] = df["Joint_Pain"].fillna(most_frequent_value)
        print(f"Joint_Pain missing values filled with: {most_frequent_value}")

    remaining_missing = df.isnull().sum().sum()
    print(f"Remaining missing values: {remaining_missing}")

    return df


def encode_categorical_features(df):
    """
    Encode categorical text columns into numeric values.
    """

    print("\n" + "=" * 70)
    print("Encoding Categorical Features")
    print("=" * 70)

    df = df.copy()

    categorical_columns = df.select_dtypes(include=["object", "str"]).columns

    for column in categorical_columns:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])
        print(f"Encoded column: {column}")

    return df


def split_dataset(df):
    """
    Split dataset into training and testing sets.
    """

    print("\n" + "=" * 70)
    print("Splitting Dataset")
    print("=" * 70)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )

    print(f"Training rows: {train_df.shape[0]}")
    print(f"Testing rows : {test_df.shape[0]}")

    return train_df, test_df


def save_processed_data(train_df, test_df):
    """
    Save processed train and test datasets.
    """

    print("\n" + "=" * 70)
    print("Saving Processed Datasets")
    print("=" * 70)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(TRAIN_DATASET_FILE, index=False)
    test_df.to_csv(TEST_DATASET_FILE, index=False)

    print(f"Training dataset saved to: {TRAIN_DATASET_FILE}")
    print(f"Testing dataset saved to : {TEST_DATASET_FILE}")


def main():
    """
    Main execution function.
    """

    dataset = load_dataset()

    print("\nDataset Information:")
    dataset.info()

    inspect_missing_values(dataset)

    cleaned_dataset = clean_missing_values(dataset)

    encoded_dataset = encode_categorical_features(cleaned_dataset)

    train_dataset, test_dataset = split_dataset(encoded_dataset)

    save_processed_data(train_dataset, test_dataset)

    print("\nData preprocessing completed successfully.")


if __name__ == "__main__":
    main()