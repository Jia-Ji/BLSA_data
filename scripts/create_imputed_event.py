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
4. for each subject, find the first MACE or death event date
5. save the resulting dataframe with subject_id and event_date to a new csv file
'''


def is_mace_icd(icd):
    """
    Identify MACE-related diagnosis from ICD-9 code.
    Includes:
    410.x MI
    411.x acute/subacute ischemic heart disease
    412   old MI
    413.x angina
    414.x chronic ischemic heart disease / CAD
    428.x heart failure
    436   stroke
    427.5 cardiac arrest
    433.x1 / 434.x1 stroke with infarction
    """
    if pd.isna(icd):
        return False

    icd = str(icd).replace(".", "").strip()

    mace_prefix3 = {"410", "411", "412", "413", "414", "428", "436"}
    mace_exact4 = {"4275"}
    stroke_codes = {
        "43301", "43311", "43321", "43331", "43381", "43391",
        "43401", "43411", "43491"
    }

    return (
        icd[:3] in mace_prefix3 or
        icd[:4] in mace_exact4 or
        icd[:5] in stroke_codes
    )


def is_mace_text(text):
    if pd.isna(text):
        return False

    text = str(text).lower()

    patterns = [
        r"\bmyocardial infarction\b",
        r"\bheart attack\b",
        r"\bmi\b",
        r"\bunstable angina\b",
        r"\bangina\b",
        r"\bcoronary artery disease\b",
        r"\bcad\b",
        r"\bischemic heart disease\b",
        r"\bheart failure\b",
        r"\bstroke\b",
        r"\bcerebrovascular\b",
        r"\bcardiac arrest\b",
    ]

    return any(re.search(pattern, text) for pattern in patterns)

def parse_activity_id(id: pd.Series)-> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parts = id.astype(str).str.split("_", n=2, expand=True)
    if parts.shape[1] != 3:
        raise ValueError(
            "First-column key must have 3 parts separated by underscores: "
            "'idno_visit_visit_date'."
        )
 
    idno = pd.to_numeric(parts[0], errors="coerce")
    visit = pd.to_numeric(parts[1], errors="coerce")
    visit_date = parts[2]

    visit_year = pd.to_numeric(visit_date.str.slice(0, 4), errors="coerce")
 
    if idno.isna().any() or visit.isna().any() or visit_date.isna().any():
        raise ValueError(
            "Found malformed first-column keys. Expected format like "
            "'1012_18_2009-10-11'."
        )
 
    return (
        idno.to_numpy(dtype=np.int64),
        visit.to_numpy(dtype=np.int64),
        visit_year.to_numpy(dtype=np.int64),
        visit_date.to_numpy(dtype=object),
    )  

def parse_dateofdeath(death_df):
    death_df = death_df.copy()
    # Strip spaces first
    death_df["dateofdeath"] = death_df["dateofdeath"].astype(str).str.strip()

    # Convert blanks to missing
    death_df["dateofdeath"] = death_df["dateofdeath"].replace("", pd.NA)

    death_df["death_year"] = pd.to_datetime(death_df["dateofdeath"], errors="coerce").dt.year
    return death_df[["idno", "death_year"]]

def get_year_from_date(date_str):
    date_str = date_str.astype(str).str.strip()
    date_str = date_str.replace("", pd.NA)
    return pd.to_datetime(date_str, errors="coerce").dt.year

def build_actiheart_data_with_events(actiheart_id_df, first_mace_df, cohort_df, last_visit_df) -> pd.DataFrame:
    '''
    Merge Actiheart summary data with first MACE event dates.
    1. find the corresponding year of the idno and visit in the Actiheart summary data
    2. merge with first MACE event and death dates and state whether the event occurred before or after the visit date
    3. save the resulting dataframe to a new csv file
    '''
    # Extract visit year from cohort data
    cohort_df = cohort_df.copy()
    cohort_df["actiheart_year"] = get_year_from_date(cohort_df["visitdate"])

    # Add visit year to Actiheart ID data
    actiheart_id_df_with_years = actiheart_id_df.merge(
        cohort_df[['idno', 'visit', 'actiheart_year']],
        on=['idno', 'visit'],
        how='left'
    )

    # Extract death year from cohort data
    cohort_df['death_year'] = get_year_from_date(cohort_df['dateofdeath'])

    # Merge Actiheart data with MACE and death events
    merged_df_with_events = actiheart_id_df_with_years.merge(first_mace_df, on=['idno'], how='left')
    merged_df_with_events = merged_df_with_events.merge(
        cohort_df[['idno', 'visit', 'death_year']],
        on=['idno', 'visit'],
        how='left'
    )

    # eligible subjects: Remove the suvject-visit rows where 1st_MACE_year is before the visit year
    eligible_subjects_with_events = merged_df_with_events[merged_df_with_events['1st_MACE_year'].isna() | (merged_df_with_events['1st_MACE_year'] > merged_df_with_events['actiheart_year']) |
                                                           merged_df_with_events['death_year'].isna() & (merged_df_with_events['death_year'] > merged_df_with_events['actiheart_year'])].copy()
    
    print(f'The number of eligible subjects after filtering for no MACE before visit: {eligible_subjects_with_events["idno"].nunique()}')

    # Get the last visit year for each subject
    last_visit_df = last_visit_df.copy()
    last_visit_df['last_visit_year'] = get_year_from_date(last_visit_df['LastVisit_Date'])
    last_visit_years = last_visit_df.groupby('idno')['last_visit_year'].max().reset_index()
    eligible_subjects_with_events = eligible_subjects_with_events.merge(last_visit_years, 
                                        on='idno', how='left').rename(columns={'last_visit_year': 'last_followup_year'})
    '''
    if MACE occurs:
        end_time = 1st_MACE_year
        event = 1
    else:
        end_time = min(death_year, last_followup_year, study_end)
        event = 0   
    '''
    # Define censoring time (minimum of available times)
    eligible_subjects_with_events['censor_year'] = eligible_subjects_with_events[['death_year', 'last_followup_year']].min(axis=1)
    # Define event occurrence
    eligible_subjects_with_events['event_occurred'] = eligible_subjects_with_events['1st_MACE_year'].notna().astype(int)
    # End time logic
    eligible_subjects_with_events['end_year'] = np.where(
        eligible_subjects_with_events['event_occurred'] == 1,
        eligible_subjects_with_events['1st_MACE_year'],
        eligible_subjects_with_events['censor_year']
    )

    # # Create a column to indicate if the event occurred after the visit
    # eligible_subjects_with_events = eligible_subjects_with_events.copy()
    # eligible_subjects_with_events['event_after_visit'] = (
    #     (eligible_subjects_with_events['1st_MACE_year'] > eligible_subjects_with_events['visit_year']) |
    #     (eligible_subjects_with_events['death_year'] > eligible_subjects_with_events['visit_year'])
    # )
    print(f'The number of subjects with MACE after visit: {eligible_subjects_with_events[eligible_subjects_with_events["1st_MACE_year"] > eligible_subjects_with_events["actiheart_year"]]["idno"].nunique()}') 
    print(f'The number of subjects with death after visit: {eligible_subjects_with_events[eligible_subjects_with_events["death_year"] > eligible_subjects_with_events["actiheart_year"]]["idno"].nunique()}')
    print(f"The number of events: {eligible_subjects_with_events['event_occurred'].sum()}")
    # print(f'The number of subjects with MACE or death after visit: {eligible_subjects_with_events[eligible_subjects_with_events["event_after_visit"]]["idno"].nunique()}')

    eligible_subjects_with_events.to_csv("eligible_with_mace_events.csv", index=False)

    return eligible_subjects_with_events



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
    actiheart_id_path = script_dir / "datainfo" / "Actiheart_Summary_Data_IDs.csv" 
    cohort_data_path = script_dir / "datarelease" / "data-csv" / "cohort_cohort.csv"
    last_visit_data_path = script_dir / "datarelease" / "data-csv" / "crbsh_masterdemog.csv"

    # Read data
    diag_df = pd.read_csv(diag_path)
    actiheart_id_df = pd.read_csv(actiheart_id_path)
    cohort_df = pd.read_csv(cohort_data_path)
    last_visit_df = pd.read_csv(last_visit_data_path)

    # Remove rows without first active year
    diag_df = diag_df.dropna(subset=["year_1st_act"])
    # Remove row where subject is not in Actiheart summary data
    diag_df = diag_df[diag_df["idno"].isin(actiheart_id_df["idno"])]

    # A row is MACE if any ICD-9 column matches OR diagnosis text matches
    diag_df["MACE_combined"] = (
        diag_df["icd9_1"].apply(is_mace_icd) |
        diag_df["icd9_2"].apply(is_mace_icd) |
        diag_df["icd9_3"].apply(is_mace_icd) |
        diag_df["diag_text"].apply(is_mace_text)
    )

    # Keep only MACE rows
    mace_df = diag_df[diag_df["MACE_combined"]].copy()

    # Make sure year_1st_act is numeric
    mace_df["year_1st_act"] = pd.to_numeric(mace_df["year_1st_act"], errors="coerce")

    # For each subject, keep the row with the earliest MACE year
    idx = mace_df.groupby(["idno"])["year_1st_act"].idxmin()

    first_mace_df = (
        mace_df.loc[idx, ["idno","year_1st_act", "icd9_1","diag_text"]]
        .copy()
        .rename(columns={"year_1st_act": "1st_MACE_year"})
        .sort_values(["idno"])
        .reset_index(drop=True)
)

    # Build Actiheart data with event dates
    merged_df = build_actiheart_data_with_events(actiheart_id_df,  first_mace_df, cohort_df, last_visit_df)

    # Save result
    mace_df.to_csv("mace_filtered_combined.csv", index=False)
    first_mace_df.to_csv("first_mace_events.csv", index=False)

    print(f"Total rows: {len(diag_df)}")
    print(f"MACE rows: {len(mace_df)}")
    print(f"Unique subjects with actiheart data: {actiheart_id_df['idno'].nunique()}")
    print(f"Unique subjects with MACE: {first_mace_df['idno'].nunique()}")
    


if __name__ == "__main__":
    main()