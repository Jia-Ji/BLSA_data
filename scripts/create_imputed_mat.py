#!/usr/bin/env python3
"""
Read two imputed CSV files, sort by the first column, align by the first column,
and save both aligned datasets into a single MAT file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import savemat


def sort_and_align_by_first_column(
    activity_df: pd.DataFrame, heart_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort each DataFrame by first column and align rows by first-column key."""
    if activity_df.empty or heart_df.empty:
        return activity_df.iloc[0:0].copy(), heart_df.iloc[0:0].copy()

    a_col0 = activity_df.columns[0]
    h_col0 = heart_df.columns[0]

    a_sorted = activity_df.sort_values(by=a_col0, kind="mergesort").copy()
    h_sorted = heart_df.sort_values(by=h_col0, kind="mergesort").copy()

    a_sorted = a_sorted[a_sorted[a_col0].notna()].copy()
    h_sorted = h_sorted[h_sorted[h_col0].notna()].copy()

    # Pair duplicate keys one-by-one by adding occurrence index per key.
    a_sorted["__key"] = a_sorted[a_col0]
    h_sorted["__key"] = h_sorted[h_col0]
    a_sorted["__dup_idx"] = a_sorted.groupby("__key", sort=False).cumcount()
    h_sorted["__dup_idx"] = h_sorted.groupby("__key", sort=False).cumcount()

    merged = a_sorted.merge(
        h_sorted,
        on=["__key", "__dup_idx"],
        how="inner",
        suffixes=("_a", "_h"),
    )

    aligned_activity = pd.DataFrame(
        {col: merged[f"{col}_a"] for col in activity_df.columns}
    )
    aligned_heart = pd.DataFrame(
        {col: merged[f"{col}_h"] for col in heart_df.columns}
    )

    return aligned_activity, aligned_heart


def split_key_to_three_fields(key_series: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split key like '1012_18_2009-10-11' into idno, visit, visit_date.
    """
    parts = key_series.astype(str).str.split("_", n=2, expand=True)
    if parts.shape[1] != 3:
        raise ValueError(
            "First-column key must have 3 parts separated by underscores: "
            "'idno_visit_visit_date'."
        )

    idno = pd.to_numeric(parts[0], errors="coerce")
    visit = pd.to_numeric(parts[1], errors="coerce")
    visit_date = parts[2]

    if idno.isna().any() or visit.isna().any() or visit_date.isna().any():
        raise ValueError(
            "Found malformed first-column keys. Expected format like "
            "'1012_18_2009-10-11'."
        )

    return (
        idno.to_numpy(dtype=np.int64),
        visit.to_numpy(dtype=np.int64),
        visit_date.to_numpy(dtype=object),
    )









def main() -> None:
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent
    data_dir = workspace_root / "datarelease" / "BLSA_Actiheart_Summary_Data"

    activity_path = data_dir / "ActivityCountRaw_2016_Jun.csv"
    heart_path = data_dir / "HeartRateRaw_2016_Jun.csv"
    output_mat = data_dir / "blsa_hr_pa.mat"

    activity_df = pd.read_csv(activity_path)
    heart_df = pd.read_csv(heart_path)

    activity_aligned, heart_aligned = sort_and_align_by_first_column(activity_df, heart_df)

    first_col_activity = activity_aligned.columns[0]
    first_col_heart = heart_aligned.columns[0]

    key_activity = activity_aligned[first_col_activity]
    key_heart = heart_aligned[first_col_heart]
    if not key_activity.reset_index(drop=True).equals(key_heart.reset_index(drop=True)):
        raise ValueError("Aligned first-column keys differ between ActivityCount and HeartRate.")

    idno, visit, visit_date = split_key_to_three_fields(key_activity)

    # Remove first key column from both matrices, as requested.
    activity_data_only = activity_aligned.iloc[:, 1:].copy()
    heart_data_only = heart_aligned.iloc[:, 1:].copy()
    tab_df = build_tab(idno=idno, visit=visit, visit_date=visit_date, csv_dir=csv_dir)
    tab_with_header = np.vstack(
        [
            tab_df.columns.to_numpy(dtype=object),
            tab_df.to_numpy(dtype=object),
        ]
    )

    # Check the activity_data_only.coulumns and heart_data_only.columns are the same after alignment
    if not activity_data_only.columns.equals(heart_data_only.columns):
        raise ValueError("After alignment, the timestamp columns differ between ActivityCount and HeartRate.")
    else:
        print("Aligned timestamp columns match between ActivityCount and HeartRate.")

    savemat(
        output_mat,
        {
            "idno": idno,
            "visit": visit,
            "visit_date": visit_date,
            "ActivityCountRaw": activity_data_only.to_numpy(dtype=object),
            "HeartRateRaw": heart_data_only.to_numpy(dtype=object),
            "Timestamp": np.array(activity_data_only.columns, dtype=object),
            "Tab": tab_with_header,
        },
        do_compression=True,
    )

    print(f"Saved MAT file: {output_mat}")
    print(f"ActivityCountImputed shape: {activity_data_only.shape}")
    print(f"HeartRateImputed shape: {heart_data_only.shape}")
    print(f"Tab rows/cols: {tab_df.shape}")


if __name__ == "__main__":
    main()
