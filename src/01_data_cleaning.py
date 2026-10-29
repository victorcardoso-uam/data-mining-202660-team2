"""
01_data_cleaning.py
Cleaning and ingestion of the "Productivity Prediction of Garment Employees" dataset (UCI).

The script follows these steps.
  1. Loads the data with a relative path (portable across laptops).
  2. Fixes data entry errors in 'department' ("sweing" becomes "sewing", extra spaces removed).
  3. Imputes WIP = 0 only in Finishing (there is no work in progress at that station).
  4. Creates the target variable 'missed_target' (1 means the team did NOT reach its target).
  5. Validates data quality with assertions and saves data/processed/garments_clean.csv.

Run it with  python src/01_data_cleaning.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "garments_worker_productivity.csv"
OUT = ROOT / "data" / "processed" / "garments_clean.csv"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) Text columns, strip spaces and fix the data entry typo
    df["department"] = df["department"].str.strip().replace({"sweing": "sewing"})
    df["quarter"] = df["quarter"].str.strip()
    df["day"] = df["day"].str.strip()

    # 2) Convert date to datetime type (used only as a reference, not as a predictor)
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")

    # 3) Imputation based on a business rule (not on the median).
    #    The WIP nulls match exactly the Finishing records, and at Finishing
    #    the product is already assembled, so the real WIP is 0.
    nulls_by_dept = df.loc[df["wip"].isna(), "department"].value_counts()
    print("WIP nulls by department\n", nulls_by_dept.to_string(), "\n")
    df.loc[df["department"] == "finishing", "wip"] = df.loc[
        df["department"] == "finishing", "wip"
    ].fillna(0)

    # 4) Target variable (binary classification)
    df["missed_target"] = (
        df["actual_productivity"] < df["targeted_productivity"]
    ).astype(int)

    return df


def validate(df: pd.DataFrame) -> None:
    """Data quality assertions (Session 4, Data Integrity & Quality)."""
    assert df.isna().sum().sum() == 0, "Null values remain"
    assert set(df["department"]) == {"sewing", "finishing"}, "Unexpected departments"
    assert df["team"].between(1, 12).all(), "Team out of range 1 to 12"
    assert df["targeted_productivity"].between(0, 1).all(), "Target outside [0, 1]"
    assert (df[["smv", "wip", "over_time", "incentive", "no_of_workers"]] >= 0).all().all(), \
        "There are impossible negative values"
    print("Quality assertions passed")


if __name__ == "__main__":
    raw = pd.read_csv(RAW)
    print(f"Raw dataset with {raw.shape[0]} rows x {raw.shape[1]} columns")
    print("Raw departments found", raw["department"].unique().tolist(), "\n")

    clean_df = clean(raw)
    validate(clean_df)

    rate = clean_df["missed_target"].mean()
    print(f"\nRecords that did NOT reach the target, {clean_df['missed_target'].sum()} "
          f"({rate:.1%})")
    print("Clean departments found", clean_df["department"].unique().tolist())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(OUT, index=False)
    print(f"\nSaved to {OUT.relative_to(ROOT)} with shape {clean_df.shape}")
