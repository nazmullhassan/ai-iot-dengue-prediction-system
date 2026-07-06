"""
EDA Visualization Module
AI-IoT-Based Early Dengue Prediction and Prevention System
Author: Nazmul Hassan
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_FILE = PROJECT_ROOT / "data" / "raw" / "dengue_dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_dataset():
    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_DATA_FILE}")

    return pd.read_csv(RAW_DATA_FILE)


def save_bar_chart(data, title, xlabel, ylabel, filename):
    plt.figure(figsize=(10, 6))
    data.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=300)
    plt.close()


def generate_visualizations():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()

    # 1. Outcome distribution
    outcome_counts = df["Outcome"].value_counts()
    save_bar_chart(
        outcome_counts,
        "Distribution of Dengue Outcomes",
        "Outcome",
        "Count",
        "outcome_distribution.png",
    )

    # 2. Gender distribution by outcome
    gender_outcome = pd.crosstab(df["Gender"], df["Outcome"])
    save_bar_chart(
        gender_outcome,
        "Dengue Outcome by Gender",
        "Gender",
        "Count",
        "gender_outcome_distribution.png",
    )

    # 3. Age distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df["Age"], bins=20)
    plt.title("Age-wise Distribution of Patients")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "age_distribution.png", dpi=300)
    plt.close()

    # 4. Biomarker prevalence
    biomarkers = df[["NS1", "IgG", "IgM"]].sum()
    save_bar_chart(
        biomarkers,
        "Prevalence of NS1, IgG, and IgM Biomarkers",
        "Biomarker",
        "Positive Count",
        "biomarker_prevalence.png",
    )

    # 5. Area type analysis
    area_type = pd.crosstab(df["AreaType"], df["Outcome"])
    save_bar_chart(
        area_type,
        "Dengue Outcomes by Area Type",
        "Area Type",
        "Count",
        "area_type_outcome.png",
    )

    # 6. House type analysis
    house_type = pd.crosstab(df["HouseType"], df["Outcome"])
    save_bar_chart(
        house_type,
        "Dengue Outcomes by House Type",
        "House Type",
        "Count",
        "house_type_outcome.png",
    )

    # 7. Top dengue-positive areas
    positive_cases = df[df["Outcome"] == 1]
    top_areas = positive_cases["Area"].value_counts().head(10)
    save_bar_chart(
        top_areas,
        "Top Areas with Dengue-Positive Cases",
        "Area",
        "Positive Cases",
        "top_dengue_positive_areas.png",
    )

    print("EDA visualizations generated successfully.")
    print(f"Saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    generate_visualizations()