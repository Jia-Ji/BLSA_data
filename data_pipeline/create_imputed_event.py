from __future__ import annotations

import csv
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

# def build_actiheart_data_with_events(actiheart_id_df, first_mace_df, cohort_df, last_visit_df) -> pd.DataFrame:
#     '''
#     Merge Actiheart summary data with first MACE event dates.
#     1. find the corresponding year of the idno and visit in the Actiheart summary data
#     2. merge with first MACE event and death dates and state whether the event occurred before or after the visit date
#     3. save the resulting dataframe to a new csv file
#     '''
#     # Extract visit year from cohort data
#     cohort_df = cohort_df.copy()
#     cohort_df["actiheart_year"] = get_year_from_date(cohort_df["visitdate"])

#     # Add visit year to Actiheart ID data
#     actiheart_id_df_with_years = actiheart_id_df.merge(
#         cohort_df[['idno', 'visit', 'actiheart_year']],
#         on=['idno', 'visit'],
#         how='left'
#     )

#     # Extract death year from cohort data
#     cohort_df['death_year'] = get_year_from_date(cohort_df['dateofdeath'])

#     # Merge Actiheart data with MACE and death events
#     merged_df_with_events = actiheart_id_df_with_years.merge(first_mace_df, on=['idno'], how='left')
#     merged_df_with_events = merged_df_with_events.merge(
#         cohort_df[['idno', 'visit', 'death_year']],
#         on=['idno', 'visit'],
#         how='left'
#     )

#     # eligible subjects: Remove the suvject-visit rows where 1st_MACE_year is before the visit year
#     eligible_subjects_with_events = merged_df_with_events[merged_df_with_events['1st_MACE_year'].isna() | (merged_df_with_events['1st_MACE_year'] > merged_df_with_events['actiheart_year']) |
#                                                            merged_df_with_events['death_year'].isna() & (merged_df_with_events['death_year'] > merged_df_with_events['actiheart_year'])].copy()
    
#     print(f'The number of eligible subjects after filtering for no MACE before visit: {eligible_subjects_with_events["idno"].nunique()}')

#     # Get the last visit year for each subject
#     last_visit_df = last_visit_df.copy()
#     last_visit_df['last_visit_year'] = get_year_from_date(last_visit_df['LastVisit_Date'])
#     last_visit_years = last_visit_df.groupby('idno')['last_visit_year'].max().reset_index()
#     eligible_subjects_with_events = eligible_subjects_with_events.merge(last_visit_years, 
#                                         on='idno', how='left').rename(columns={'last_visit_year': 'last_followup_year'})
#     '''
#     if MACE occurs:
#         end_time = 1st_MACE_year
#         event = 1
#     else:
#         end_time = min(death_year, last_followup_year, study_end)
#         event = 0   
#     '''
#     # Define censoring time (minimum of available times)
#     eligible_subjects_with_events['censor_year'] = eligible_subjects_with_events[['death_year', 'last_followup_year']].min(axis=1)
#     # Define event occurrence
#     eligible_subjects_with_events['event_occurred'] = eligible_subjects_with_events['1st_MACE_year'].notna().astype(int)
#     # End time logic
#     eligible_subjects_with_events['end_year'] = np.where(
#         eligible_subjects_with_events['event_occurred'] == 1,
#         eligible_subjects_with_events['1st_MACE_year'],
#         eligible_subjects_with_events['censor_year']
#     )

#     # # Create a column to indicate if the event occurred after the visit
#     # eligible_subjects_with_events = eligible_subjects_with_events.copy()
#     # eligible_subjects_with_events['event_after_visit'] = (
#     #     (eligible_subjects_with_events['1st_MACE_year'] > eligible_subjects_with_events['visit_year']) |
#     #     (eligible_subjects_with_events['death_year'] > eligible_subjects_with_events['visit_year'])
#     # )
#     print(f'The number of subjects with MACE after visit: {eligible_subjects_with_events[eligible_subjects_with_events["1st_MACE_year"] > eligible_subjects_with_events["actiheart_year"]]["idno"].nunique()}') 
#     print(f'The number of subjects with death after visit: {eligible_subjects_with_events[eligible_subjects_with_events["death_year"] > eligible_subjects_with_events["actiheart_year"]]["idno"].nunique()}')
#     print(f"The number of events: {eligible_subjects_with_events['event_occurred'].sum()}")
#     # print(f'The number of subjects with MACE or death after visit: {eligible_subjects_with_events[eligible_subjects_with_events["event_after_visit"]]["idno"].nunique()}')

#     eligible_subjects_with_events.to_csv("eligible_with_mace_events.csv", index=False)

#     return eligible_subjects_with_events

def get_visit_age(masterdemog_df, cohort_df):
    """Calculate the age at the time of visit based on the first visit age and year."""
    masterdemog_df = masterdemog_df.copy()
    cohort_df = cohort_df.copy()
    cohort_df['visit_year'] = get_year_from_date(cohort_df['visitdate'])
    masterdemog_df = masterdemog_df.merge(cohort_df[['idno', 'visit', 'visit_year']], on=['idno', 'visit'], how='left')
    masterdemog_df['FirstVisit_Date'] = pd.to_datetime(masterdemog_df['FirstVisit_Date'], errors='coerce')
    masterdemog_df['FirstVisit_Year'] = masterdemog_df['FirstVisit_Date'].dt.year
    masterdemog_df['visit_age'] = masterdemog_df['FirstVisit_Age'] + (masterdemog_df['visit_year'] - masterdemog_df['FirstVisit_Year'])
    return masterdemog_df[['idno', 'visit', 'visit_age', 'gender']]

def build_tab(actiheart_id_df, first_mace, last_visit, cohort_data, demographic, drinker, smoke, medication):
    actiheart_id_df = actiheart_id_df.copy()
    
    # add visit date to actiheart_id_df
    actiheart_id_df = actiheart_id_df.merge(cohort_data[['idno', 'visit', 'visitdate']], on=['idno', 'visit'], how='left').rename(columns={'visitdate': 'visit_date'})

    # merge with first MACE event date
    actiheart_id_df = actiheart_id_df.merge(first_mace[['idno', '1st_MACE_year', 'icd9_1','diag_text']], on='idno', how='left')

    # merge with death date
    actiheart_id_df = actiheart_id_df.merge(cohort_data[['idno', 'visit', 'dateofdeath']], on=['idno', 'visit'], how='left')

    # get the last visit date for each subject
    last_visit = last_visit.copy()
    last_visit['LastVisit_Date'] = pd.to_datetime(last_visit['LastVisit_Date'], errors='coerce')
    last_visit = last_visit.groupby('idno')['LastVisit_Date'].max().reset_index()
    # merge with last visit date
    actiheart_id_df = actiheart_id_df.merge(last_visit[['idno', 'LastVisit_Date']], on='idno', how='left').rename(columns={'LastVisit_Date': 'last_followup_date'})
    
    # check weight and height values for each subject-visit, if there are multiple values, print them out for manual review
    for idno, visit in demographic.groupby(['idno', 'visit']).size().index:
        subset = demographic[(demographic['idno'] == idno) & (demographic['visit'] == visit)]
        if subset['weight'].nunique(dropna=True) > 1:
            print(f"Subject {idno} visit {visit} has multiple weight values: {subset['weight'].unique()}")
        if subset['height'].nunique(dropna=True) > 1:
            print(f"Subject {idno} visit {visit} has multiple height values: {subset['height'].unique()}")   
    # merge with demographic data with age, sex
    demographic = demographic.drop_duplicates(subset=['idno', 'visit'], keep='first')
    actiheart_id_df = actiheart_id_df.merge(demographic, on=['idno', 'visit'], how='left') 

    # merge with drinker status
    actiheart_id_df = actiheart_id_df.merge(drinker, on=['idno', 'visit'], how='left')

    # merge with smoke status
    actiheart_id_df = actiheart_id_df.merge(smoke, on=['idno', 'visit'], how='left')

    # merge with medication data
    actiheart_id_df = actiheart_id_df.merge(medication, on=['idno', 'visit'], how='left')

    # add a column to indicate whether the subject has incident MACE event after the visit
    actiheart_id_df['MACE_after_visit'] = np.where(
        (actiheart_id_df['1st_MACE_year'].notna()) & 
        (get_year_from_date(actiheart_id_df['visit_date']) < actiheart_id_df['1st_MACE_year']),
        1, 0
    )
    
    # compute death follow-up days from visit date
    actiheart_id_df['visit_date'] = pd.to_datetime(actiheart_id_df['visit_date'], errors='coerce')
    actiheart_id_df['dateofdeath'] = pd.to_datetime(actiheart_id_df['dateofdeath'], errors='coerce')
    actiheart_id_df['death_followup_days'] = (actiheart_id_df['dateofdeath'] - actiheart_id_df['visit_date']).dt.days   

    # add death event indicator
    actiheart_id_df['death_event'] = np.where(
        (actiheart_id_df['dateofdeath'].notna()) & 
        (actiheart_id_df['visit_date'] < actiheart_id_df['dateofdeath']),
        1, 0)

    return actiheart_id_df



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
    workspace_root = script_dir.parent
    output_dir = workspace_root / "combined_data"
    output_dir.mkdir(exist_ok=True)
    print(f"Script directory: {script_dir}")
    print(f"Workspace root: {workspace_root}")
    csv_dir = workspace_root / "datarelease" / "data-csv"
    actiheart_id_path = workspace_root / "datainfo" / "Actiheart_Summary_Data_IDs.csv" 
    diagnosis_path = csv_dir / "diagnoses_blsadx.csv"
    demographic_path = workspace_root / "datarelease" / "BLSA_Actiheart_Summary_Data"/ "SubjectInfoRaw_2016_jun.csv"

    # Read data
    actiheart_id_df = pd.read_csv(actiheart_id_path)
    diagnosis = pd.read_csv(diagnosis_path)
    last_visit = load_and_prepare_lookup(
        csv_dir / "crbsh_masterdemog.csv",
        {
            "lastvisit_date": "LastVisit_Date"
        })
    cohort_data = load_and_prepare_lookup(
        csv_dir / "cohort_cohort.csv",
        {
            "visitdate": "visitdate",
            "dateofdeath": "dateofdeath"
        })
    
    drinker = load_and_prepare_lookup(
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

    demographic = pd.read_csv(demographic_path)[["BLSA_id", "visit_id", "age",
                                                  "weight", "height", "sex"
                                                  ]].rename(columns={"BLSA_id": "idno", "visit_id": "visit", "date": "actiheart_date"})
    
    # Clean year_1st_act to ensure it's numeric
    def clean_year(year_str):
        if pd.isna(year_str):
            return np.nan
        if isinstance(year_str, str):
            years = [pd.to_numeric(y.strip(), errors='coerce') for y in year_str.split(';')]
            years = [y for y in years if not pd.isna(y)]
            if years:
                return min(years)
            else:
                return np.nan
        else:
            return year_str

    diagnosis["year_1st_act"] = diagnosis["year_1st_act"].apply(clean_year)
    diagnosis = diagnosis.dropna(subset=["year_1st_act"])

    # A row is MACE if any ICD-9 column matches OR diagnosis text matches
    diagnosis["is_MACE"] = (
        diagnosis["icd9_1"].apply(is_mace_icd) |
        diagnosis["icd9_2"].apply(is_mace_icd) |
        diagnosis["icd9_3"].apply(is_mace_icd) |
        diagnosis["diag_text"].apply(is_mace_text)
    )

    # Filter for MACE events
    mace = diagnosis[diagnosis["is_MACE"]].copy()

    # For each subject, find the first MACE event date
    first_mace = mace.loc[mace.groupby("idno")["year_1st_act"].idxmin()].copy()
    first_mace = first_mace.rename(columns={"year_1st_act": "1st_MACE_year"})

    # Remove row where subject is not in Actiheart summary data
    actiheart_first_mace = first_mace[first_mace["idno"].isin(actiheart_id_df["idno"])]

    

    # Build Actiheart data with event dates and covariates
    actiheart_combined = build_tab(actiheart_id_df, actiheart_first_mace, last_visit, cohort_data, demographic, drinker, smoke, medication)
    
    # Save result
    actiheart_combined.to_csv(output_dir / "actiheart_with_events.csv", index=False)
   
    # Print summary statistics
    print(f"Total subjects in Actiheart data: {actiheart_id_df['idno'].nunique()}")
    print(f"Subjects with MACE events: {first_mace['idno'].nunique()}")
    print(f"Subjects with MACE events in Actiheart data: {actiheart_first_mace['idno'].nunique()}")
    print(f"Subjects with MACE events after visit in Actiheart data: {actiheart_combined[actiheart_combined['MACE_after_visit'] == 1]['idno'].nunique()}")
    print(f"Subjects with death dates in Actiheart data: {actiheart_combined[actiheart_combined['dateofdeath'].notna()]['idno'].nunique()}")


if __name__ == "__main__":
    main()