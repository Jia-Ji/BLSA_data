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


def _clean_cell_value(value: Any) -> Any:
    """Normalize common placeholder values to missing."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return np.nan
        if set(text) == {"."}:
            return np.nan
        return text
    return value


def _collapse_series(values: pd.Series) -> Any:
    """
    Collapse potentially repeated values into one value per key.
    - If no valid value, return NaN.
    - If one unique valid value, return it.
    - If multiple unique values, join with '; '.
    """
    unique_vals: list[Any] = []
    seen: set[str] = set()

    for raw in values:
        cleaned = _clean_cell_value(raw)
        if pd.isna(cleaned):
            continue
        key = str(cleaned)
        if key in seen:
            continue
        seen.add(key)
        unique_vals.append(cleaned)

    if not unique_vals:
        return np.nan
    if len(unique_vals) == 1:
        return unique_vals[0]
    return "; ".join(str(v) for v in unique_vals)


def load_and_prepare_lookup(
    csv_path: Path,
    wanted_columns: dict[str, str],
) -> pd.DataFrame:
    """
    Read lookup file, align key columns to idno/visit, select wanted columns,
    and collapse repeated rows per (idno, visit).
    wanted_columns maps input_name -> output_name.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    colmap = {c.lower(): c for c in df.columns}

    if "visit" not in colmap:
        raise ValueError(f"'visit' column not found in {csv_path}")

    id_key = "idno"
    if id_key is None:
        raise ValueError(f"'idno' not found in {csv_path}")

    prepared = pd.DataFrame()
    prepared["idno"]= pd.to_numeric(df[colmap[id_key]], errors="coerce")
    prepared["visit"] = pd.to_numeric(df[colmap["visit"]], errors="coerce")

    for src, dst in wanted_columns.items():
        src_lower = src.lower()
        if src_lower in colmap:
            prepared[dst] = df[colmap[src_lower]]
        else:
            prepared[dst] = np.nan

    prepared = prepared.dropna(subset=["idno", "visit"]).copy()
    prepared["idno"] = prepared["idno"].astype(np.int64)
    prepared["visit"] = prepared["visit"].astype(np.int64)

    agg_map = {col: _collapse_series for col in prepared.columns if col not in {"idno", "visit"}}
    return prepared.groupby(["idno", "visit"], as_index=False).agg(agg_map)


def load_and_prepare_medication_lookup(csv_path: Path) -> pd.DataFrame:
    """
    Medication-specific processing:
    keep act1(atc1) and length_of_use positionally aligned.
    Missing length_of_use is written as literal 'NA'.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    colmap = {c.lower(): c for c in df.columns}

    for needed in ("idno", "visit", "atc1", "length_of_use"):
        if needed not in colmap:
            raise ValueError(f"'{needed}' column not found in {csv_path}")

    prepared = pd.DataFrame(
        {
            "idno": pd.to_numeric(df[colmap["idno"]], errors="coerce"),
            "visit": pd.to_numeric(df[colmap["visit"]], errors="coerce"),
            "act1": df[colmap["atc1"]],
            "length_of_use": df[colmap["length_of_use"]],
        }
    ).dropna(subset=["idno", "visit"])

    prepared["idno"] = prepared["idno"].astype(np.int64)
    prepared["visit"] = prepared["visit"].astype(np.int64)

    def _collapse_pairs(group: pd.DataFrame) -> pd.Series:
        act_list: list[str] = []
        len_list: list[str] = []
        seen_pairs: set[tuple[str, str]] = set()

        for _, row in group.iterrows():
            act_raw = _clean_cell_value(row["act1"])
            if pd.isna(act_raw):
                continue
            act = str(act_raw)

            len_raw = _clean_cell_value(row["length_of_use"])
            length = "NA" if pd.isna(len_raw) else str(len_raw)

            pair = (act, length)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            act_list.append(act)
            len_list.append(length)

        if not act_list:
            return pd.Series({"act1": np.nan, "length_of_use": np.nan})

        return pd.Series(
            {
                "act1": "; ".join(act_list),
                "length_of_use": "; ".join(len_list),
            }
        )

    rows: list[dict[str, Any]] = []
    for (idno_val, visit_val), grp in prepared.groupby(["idno", "visit"], sort=False):
        collapsed = _collapse_pairs(grp)
        rows.append(
            {
                "idno": int(idno_val),
                "visit": int(visit_val),
                "act1": collapsed["act1"],
                "length_of_use": collapsed["length_of_use"],
            }
        )

    return pd.DataFrame(rows)


def build_tab(
    idno: np.ndarray,
    visit: np.ndarray,
    visit_date: np.ndarray,
    csv_dir: Path,
) -> pd.DataFrame:
    """Build Tab by left-joining requested variables on (idno, visit)."""
    tab = pd.DataFrame(
        {
            "idno": idno.astype(np.int64),
            "visit": visit.astype(np.int64),
            "visit_date": visit_date.astype(object),
        }
    )

    masterdemog = load_and_prepare_lookup(
        csv_dir / "crbsh_masterdemog.csv",
        {
            "FirstVisit_Age": "FirstVisit_Age",
            "LastVisit_Age": "LastVisit_Age",
            "sex": "sex",
        },
    )
    demographics = load_and_prepare_lookup(
        csv_dir / "der_demographics.csv",
        {"drinker": "drinker"},
    )
    smoke = load_and_prepare_lookup(
        csv_dir / "crbsh_blsasmoke.csv",
        {
            "smkq07": "smoke_yrs",
            "smkq02": "smoke",
        },
    )
    medication = load_and_prepare_medication_lookup(csv_dir / "crbsh_blsamedication.csv")

    for lookup in (masterdemog, demographics, smoke, medication):
        tab = tab.merge(lookup, on=["idno", "visit"], how="left")

    ordered_columns = [
        "idno",
        "visit",
        "visit_date",
        "FirstVisit_Age",
        "LastVisit_Age",
        "sex",
        "drinker",
        "smoke_yrs",
        "smoke",
        "act1",
        "length_of_use",
    ]
    for col in ordered_columns:
        if col not in tab.columns:
            tab[col] = np.nan
    ordered_tab = tab[ordered_columns]
    ordered_tab.columns = ordered_columns
    print(ordered_tab.head())
    return ordered_tab


def dataframe_to_mat_struct(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Convert DataFrame to MATLAB struct-compatible dict of column vectors."""
    mat_struct: dict[str, np.ndarray] = {}
    for col in df.columns:
        values = df[col].to_numpy(dtype=object).reshape(-1, 1)
        mat_struct[col] = values
    return mat_struct


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "datarelease" / "BLSA_Actiheart_Summary_Data"
    csv_dir = script_dir / "datarelease" / "data-csv"

    activity_path = data_dir / "ActivityCountImputed_2016_Jun.csv"
    heart_path = data_dir / "HeartRateImputed_2016_Jun.csv"
    output_mat = data_dir / "blsa_imputed_actiheart.mat"

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

    savemat(
        output_mat,
        {
            "idno": idno,
            "visit": visit,
            "visit_date": visit_date,
            "ActivityCountImputed": activity_data_only.to_numpy(dtype=object),
            "HeartRateImputed": heart_data_only.to_numpy(dtype=object),
            "ActivityCountImputed_timestamp": np.array(activity_data_only.columns, dtype=object),
            "HeartRateImputed_timestamp": np.array(heart_data_only.columns, dtype=object),
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
