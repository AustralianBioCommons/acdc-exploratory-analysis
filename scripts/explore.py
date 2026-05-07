"""Exploratory analysis on the flattened CDAH table produced by flatten.py."""

import pandas as pd

FLAT_CSV = "outputs/cdah_flat.csv"


def load():
    return pd.read_csv(FLAT_CSV)


if __name__ == "__main__":
    df = load()
    print(f"shape: {df.shape}")
    print("columns:", list(df.columns))
    print("\nfirst 10 rows:")
    print(df.head(10).to_string())
