import os
import matplotlib

import args
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from load_data import load_data

sns.set(style="whitegrid")

CHARTS_DIR = os.path.join(os.path.dirname(__file__), "static", "charts")


def _chart_path(filename: str) -> str:
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return os.path.join(CHARTS_DIR, filename)


def _save(filename: str):
    plt.tight_layout()
    plt.savefig(_chart_path(filename), bbox_inches="tight", dpi=100)
    plt.close("all")


def run_eda() -> dict:
    data = load_data()
    charts = []
    # 1. LOAD DATA
    print("\n" + "=" * 80)
    print("1. LOAD DATA")
    print("=" * 80)

    print("Shape:", data.shape)
    print("\nFirst 5 Rows:")
    print(data.head())

    # 2. BASIC INFO / STRUCTURE
    print("\n" + "=" * 80)
    print("2. BASIC INFO / STRUCTURE")
    print("=" * 80)

    print("\nDataset Information:")
    data.info()

    print("\nColumn Data Types:")
    print(data.dtypes)

    print("\nDescriptive Statistics (Numeric Columns):")
    print(data.describe())

    print("\nDescriptive Statistics (Categorical Columns):")
    print(data.describe(include=["object"]))

    # 3. MISSING VALUES
    print("\n" + "=" * 80)
    print("3. MISSING VALUES")
    print("=" * 80)

    missing = data.isnull().sum()
    missing_pct = (missing / len(data)) * 100

    missing_df = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": missing_pct
    })

    missing_df = missing_df[
        missing_df["missing_count"] > 0
    ].sort_values(by="missing_count", ascending=False)

    print(missing_df)

    if not missing_df.empty:
        plt.figure(figsize=(10, 5), dpi=100)
        sns.barplot(x=missing_df.index, y=missing_df["missing_pct"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Missing %")
        plt.title("Missing values by column")
        _save("missing_values.png")
        charts.append("missing_values.png")

    # 4. DUPLICATES
    print("\n" + "=" * 80)
    print("4. DUPLICATE ROWS")
    print("=" * 80)
    print("Duplicate rows:", data.duplicated().sum())

    # 5. TARGET VARIABLE DISTRIBUTION
    print("\n" + "=" * 80)
    print("5. TARGET VARIABLE - Placementstatus")
    print("=" * 80)
    print(data["PlacementStatus"].value_counts())

    plt.figure(dpi=125)
    sns.countplot(x="PlacementStatus", data=data)
    plt.xlabel("PlacementStatus (0 = Not Placed, 1 = Placed)")
    plt.ylabel("Count")
    plt.title("Placement Status Distribution")
    _save("placement_status.png")
    charts.append("placement_status.png")

    return {
        "charts": charts,
        "missing": missing_df.to_dict(),
        "duplicates": int(data.duplicated().sum())
    }
