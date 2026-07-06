"""
SHAP Explainable AI Analysis
AI-IoT-Based Early Dengue Prediction and Prevention System
Author: Nazmul Hassan
"""

from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_DATA = PROJECT_ROOT / "data" / "processed" / "test_dataset.csv"
MODEL_FILE = PROJECT_ROOT / "models" / "xgboost_model.pkl"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)

TARGET = "Outcome"


def main():
    test_df = pd.read_csv(TEST_DATA)

    X_test = test_df.drop(columns=[TARGET])

    model = joblib.load(MODEL_FILE)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # SHAP Summary Plot
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_test,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "shap_summary_plot.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # SHAP Bar Plot
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_test,
        plot_type="bar",
        show=False,
    )
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "shap_bar_plot.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # SHAP Feature Importance CSV
    shap_importance = pd.DataFrame(
        {
            "Feature": X_test.columns,
            "Mean_ABS_SHAP": abs(shap_values).mean(axis=0),
        }
    ).sort_values(by="Mean_ABS_SHAP", ascending=False)

    shap_importance.to_csv(
        RESULTS_DIR / "shap_feature_importance.csv",
        index=False,
    )

    print("SHAP analysis completed successfully.")
    print(f"Saved results in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()