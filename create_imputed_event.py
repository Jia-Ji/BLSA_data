from __future__ import annotations

from pathlib import Path
from typing import Any
import re

import numpy as np
import pandas as pd

'''
1. read in diagnoses_blsadx.csv
2. remove rows without first active year
3. filter for MACE events (MI, unstable angina, old MI, angina, CAD, heart failure, stroke)
4. for each subject, find the first MACE event date
5. save the resulting dataframe with subject_id and event_date to a new csv file
'''


def is_mace_icd(icd):
    """
    Identify MACE from ICD-9 code.
    """
    if pd.isna(icd):
        return False

    # Remove decimal point, e.g. 410.9 -> 4109
    icd = str(icd).replace(".", "").strip()

    # Broad MACE-related ICD-9 prefixes
    mace_prefix3 = {"410", "412", "428", "436"}   # MI, old MI, HF, stroke
    mace_exact4 = {"4275"}                        # cardiac arrest

    # Stroke with infarction
    stroke_codes = {
        "43301", "43311", "43321", "43331", "43381", "43391",
        "43401", "43411", "43491"
    }

    if icd[:3] in mace_prefix3:
        return True
    if icd[:4] in mace_exact4:
        return True
    if icd[:5] in stroke_codes:
        return True

    return False

def is_mace_text(text):
    if pd.isna(text):
        return False

    text = str(text).lower()

    patterns = [
        r"\bmyocardial infarction\b",
        r"\bheart attack\b",
        r"\bmi\b",
        r"\bstroke\b",
        r"\bcerebrovascular\b",
        r"\bheart failure\b",
        r"\bcardiac arrest\b",
    ]

    return any(re.search(pattern, text) for pattern in patterns)








# def build_icd_event_dates(icd_df, subject_col="subject_id"):
#     icd_df = icd_df.copy()
#     icd_df["record_date"] = pd.to_datetime(icd_df["record_date"])

#     def is_mace(code):
#         code = str(code)
#         return any(code.startswith(prefix) for prefix in MACE_ICD9_PREFIX)

#     icd_df["is_mace"] = icd_df["ICD9_1"].apply(is_mace)

#     mace_df = icd_df[icd_df["is_mace"]]

#     first_event = (
#         mace_df.groupby(subject_col)["record_date"]
#         .min()
#         .reset_index()
#         .rename(columns={"record_date": "event_date"})
#     )

#     return first_event

def main():
    script_dir = Path(__file__).resolve().parent
    diag_path = script_dir / "datarelease" / "data-csv"/ "diagnoses_blsadx.csv"
    diag_df = pd.read_csv(diag_path)
    diag_df = diag_df.dropna(subset=["year_1st_act"])
    # A row is MACE if any ICD-9 column matches OR diagnosis text matches
    diag_df["MACE_combined"] = (
        diag_df["icd9_1"].apply(is_mace_icd) |
        diag_df["icd9_2"].apply(is_mace_icd) |
        diag_df["icd9_3"].apply(is_mace_icd) |
        diag_df["diag_text"].apply(is_mace_text)
    )

    # Keep only MACE rows
    mace_df = diag_df[diag_df["MACE_combined"]].copy()

    # For each subject, find the first MACE event date
    first_mace = (
        mace_df.groupby("idno")["year_1st_act"]
        .min()
        .reset_index()
        .rename(columns={"year_1st_act": "event_date"})
    )   

    # Save result
    mace_df.to_csv("mace_filtered_combined.csv", index=False)
    first_mace.to_csv("first_mace_events.csv", index=False)

    print(f"Total rows: {len(diag_df)}")
    print(f"MACE rows: {len(mace_df)}")
    print(f"Unique subjects with MACE: {first_mace['idno'].nunique()}")

if __name__ == "__main__":
    main()