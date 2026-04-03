
from __future__ import annotations


from datetime import datetime

import numpy as np
import pandas as pd

# =============================================================================
# Dataset Information Recording Functions for Time-to-Event Analysis
# =============================================================================

def record_sample_size_info(df: pd.DataFrame, id_col: str = "idno") -> dict:
    """
    Record basic sample size information for survival analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with subject-level or visit-level data
    id_col : str
        Column name for subject identifier
        
    Returns
    -------
    dict
        Dictionary containing sample size metrics
    """
    info = {
        "total_rows": len(df),
        "unique_subjects": df[id_col].nunique(),
        "rows_per_subject_mean": len(df) / df[id_col].nunique() if df[id_col].nunique() > 0 else 0,
        "rows_per_subject_median": df.groupby(id_col).size().median(),
    }
    return info


def record_event_info(
    df: pd.DataFrame,
    mace_year_col: str = "1st_MACE_year",
    death_year_col: str = "death_year",
    baseline_year_col: str = "visit_year",
    id_col: str = "idno"
) -> dict:
    """
    Record event occurrence information for MACE and death endpoints.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with event columns
    mace_year_col : str
        Column for MACE event year
    death_year_col : str
        Column for death year
    baseline_year_col : str
        Column for baseline/visit year
    id_col : str
        Column for subject identifier
        
    Returns
    -------
    dict
        Dictionary with event statistics
    """
    # Get unique subjects for subject-level analysis
    subject_df = df.drop_duplicates(subset=[id_col])
    n_subjects = len(subject_df)
    
    # MACE events
    has_mace = subject_df[mace_year_col].notna()
    mace_after_baseline = (
        has_mace & 
        (subject_df[mace_year_col] > subject_df[baseline_year_col])
    )
    
    # Death events
    has_death = subject_df[death_year_col].notna()
    death_after_baseline = (
        has_death & 
        (subject_df[death_year_col] > subject_df[baseline_year_col])
    )
    
    # Composite endpoint (MACE or death)
    composite_event = mace_after_baseline | death_after_baseline
    
    # Competing risks: death before MACE (for MACE analysis)
    death_before_mace = (
        death_after_baseline & 
        (~mace_after_baseline | (subject_df[death_year_col] < subject_df[mace_year_col]))
    )
    
    info = {
        "n_subjects": n_subjects,
        # MACE events
        "n_mace_events": mace_after_baseline.sum(),
        "mace_event_rate_pct": (mace_after_baseline.sum() / n_subjects * 100) if n_subjects > 0 else 0,
        "n_mace_censored": n_subjects - mace_after_baseline.sum(),
        # Death events
        "n_death_events": death_after_baseline.sum(),
        "death_event_rate_pct": (death_after_baseline.sum() / n_subjects * 100) if n_subjects > 0 else 0,
        "n_death_censored": n_subjects - death_after_baseline.sum(),
        # Composite endpoint
        "n_composite_events (MACE or death)": composite_event.sum(),
        "composite_event_rate_pct": (composite_event.sum() / n_subjects * 100) if n_subjects > 0 else 0,
        # Competing risks
        "n_death_before_mace (Competing risks)": death_before_mace.sum(),
        "death_before_mace_pct": (death_before_mace.sum() / n_subjects * 100) if n_subjects > 0 else 0,
    }
    return info


def record_followup_time_info(
    df: pd.DataFrame,
    mace_year_col: str = "1st_MACE_year",
    death_year_col: str = "death_year",
    baseline_year_col: str = "visit_year",
    id_col: str = "idno",
    censor_year: int = None
) -> dict:
    """
    Record follow-up time distribution for survival analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with time columns
    mace_year_col : str
        Column for MACE event year
    death_year_col : str
        Column for death year
    baseline_year_col : str
        Column for baseline year
    id_col : str
        Column for subject identifier
    censor_year : int, optional
        Administrative censoring year (e.g., end of study)
        
    Returns
    -------
    dict
        Dictionary with follow-up time statistics
    """
    if censor_year is None:
        censor_year = datetime.now().year
        
    subject_df = df.drop_duplicates(subset=[id_col]).copy()
    
    # Calculate time to MACE (or censoring)
    def calc_time_to_mace(row):
        baseline = row[baseline_year_col]
        mace = row[mace_year_col]
        death = row[death_year_col]
        
        if pd.notna(mace) and mace > baseline:
            return mace - baseline
        elif pd.notna(death) and death > baseline:
            return death - baseline  # censored at death
        else:
            return censor_year - baseline  # administrative censoring
    
    # Calculate time to death (or censoring)
    def calc_time_to_death(row):
        baseline = row[baseline_year_col]
        death = row[death_year_col]
        
        if pd.notna(death) and death > baseline:
            return death - baseline
        else:
            return censor_year - baseline
    
    # Calculate time to composite event
    def calc_time_to_composite(row):
        baseline = row[baseline_year_col]
        mace = row[mace_year_col]
        death = row[death_year_col]
        
        event_times = []
        if pd.notna(mace) and mace > baseline:
            event_times.append(mace)
        if pd.notna(death) and death > baseline:
            event_times.append(death)
        
        if event_times:
            return min(event_times) - baseline
        else:
            return censor_year - baseline
    
    subject_df["time_to_mace"] = subject_df.apply(calc_time_to_mace, axis=1)
    subject_df["time_to_death"] = subject_df.apply(calc_time_to_death, axis=1)
    subject_df["time_to_composite"] = subject_df.apply(calc_time_to_composite, axis=1)
    
    info = {
        "censor_year": censor_year,
        # MACE follow-up
        "mace_followup_median_years": subject_df["time_to_mace"].median(),
        "mace_followup_mean_years": subject_df["time_to_mace"].mean(),
        "mace_followup_q25_years": subject_df["time_to_mace"].quantile(0.25),
        "mace_followup_q75_years": subject_df["time_to_mace"].quantile(0.75),
        "mace_followup_min_years": subject_df["time_to_mace"].min(),
        "mace_followup_max_years": subject_df["time_to_mace"].max(),
        # Death follow-up
        "death_followup_median_years": subject_df["time_to_death"].median(),
        "death_followup_mean_years": subject_df["time_to_death"].mean(),
        "death_followup_q25_years": subject_df["time_to_death"].quantile(0.25),
        "death_followup_q75_years": subject_df["time_to_death"].quantile(0.75),
        "death_followup_min_years": subject_df["time_to_death"].min(),
        "death_followup_max_years": subject_df["time_to_death"].max(),
        # Composite follow-up
        "composite_followup_median_years": subject_df["time_to_composite"].median(),
        "composite_followup_mean_years": subject_df["time_to_composite"].mean(),
        "composite_followup_q25_years": subject_df["time_to_composite"].quantile(0.25),
        "composite_followup_q75_years": subject_df["time_to_composite"].quantile(0.75),
        # Total person-years
        "total_person_years_mace": subject_df["time_to_mace"].sum(),
        "total_person_years_death": subject_df["time_to_death"].sum(),
    }
    return info


def record_baseline_characteristics(
    df: pd.DataFrame,
    baseline_year_col: str = "visit_year",
    id_col: str = "idno",
    age_col: str = None,
    sex_col: str = None
) -> dict:
    """
    Record baseline demographic characteristics of the cohort.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with baseline characteristics
    baseline_year_col : str
        Column for baseline year
    id_col : str
        Column for subject identifier
    age_col : str, optional
        Column for age at baseline
    sex_col : str, optional
        Column for sex/gender
        
    Returns
    -------
    dict
        Dictionary with baseline characteristics
    """
    subject_df = df.drop_duplicates(subset=[id_col])
    
    info = {
        "baseline_year_median": subject_df[baseline_year_col].median(),
        "baseline_year_range": (
            subject_df[baseline_year_col].min(),
            subject_df[baseline_year_col].max()
        ),
        "enrollment_period_years": (
            subject_df[baseline_year_col].max() - subject_df[baseline_year_col].min()
        ),
    }
    
    if age_col and age_col in df.columns:
        info.update({
            "age_mean": subject_df[age_col].mean(),
            "age_std": subject_df[age_col].std(),
            "age_median": subject_df[age_col].median(),
            "age_q25": subject_df[age_col].quantile(0.25),
            "age_q75": subject_df[age_col].quantile(0.75),
            "age_min": subject_df[age_col].min(),
            "age_max": subject_df[age_col].max(),
        })
    
    if sex_col and sex_col in df.columns:
        sex_counts = subject_df[sex_col].value_counts()
        info.update({
            "sex_distribution": sex_counts.to_dict(),
            "sex_distribution_pct": (sex_counts / len(subject_df) * 100).to_dict(),
        })
    
    return info


def record_missing_data_info(
    df: pd.DataFrame,
    key_columns: list = None,
    id_col: str = "idno"
) -> dict:
    """
    Record missing data information for key survival analysis variables.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze
    key_columns : list, optional
        List of key columns to check. If None, checks all columns.
    id_col : str
        Column for subject identifier
        
    Returns
    -------
    dict
        Dictionary with missing data statistics
    """
    if key_columns is None:
        key_columns = df.columns.tolist()
    
    missing_info = {}
    for col in key_columns:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            pct_missing = (n_missing / len(df) * 100) if len(df) > 0 else 0
            missing_info[col] = {
                "n_missing": n_missing,
                "pct_missing": round(pct_missing, 2)
            }
    
    # Overall completeness
    complete_cases = df[key_columns].dropna().shape[0] if key_columns else len(df)
    
    info = {
        "column_missing_info": missing_info,
        "n_complete_cases": complete_cases,
        "pct_complete_cases": (complete_cases / len(df) * 100) if len(df) > 0 else 0,
    }
    return info


def record_event_type_breakdown(
    df: pd.DataFrame,
    mace_icd_col: str = "icd9_1",
    mace_text_col: str = "diag_text",
    id_col: str = "idno"
) -> dict:
    """
    Record breakdown of MACE event types (MI, stroke, heart failure, etc.).
    
    Parameters
    ----------
    df : pd.DataFrame
        MACE events dataset
    mace_icd_col : str
        Column containing ICD codes
    mace_text_col : str
        Column containing diagnosis text
    id_col : str
        Column for subject identifier
        
    Returns
    -------
    dict
        Dictionary with MACE type breakdown
    """
    def classify_mace_type(row):
        icd = str(row.get(mace_icd_col, "")).replace(".", "").strip() if pd.notna(row.get(mace_icd_col)) else ""
        text = str(row.get(mace_text_col, "")).lower() if pd.notna(row.get(mace_text_col)) else ""
        
        # Classify by ICD code primarily
        if icd.startswith(("410", "411", "412")):
            return "myocardial_infarction"
        elif icd.startswith("413"):
            return "angina"
        elif icd.startswith("414"):
            return "chronic_ischemic_heart_disease"
        elif icd.startswith("428"):
            return "heart_failure"
        elif icd.startswith(("433", "434", "436")):
            return "stroke"
        elif icd.startswith("4275"):
            return "cardiac_arrest"
        # Fall back to text classification
        elif "stroke" in text or "cerebrovascular" in text:
            return "stroke"
        elif "heart failure" in text:
            return "heart_failure"
        elif "myocardial infarction" in text or "heart attack" in text or "mi" in text:
            return "myocardial_infarction"
        elif "angina" in text:
            return "angina"
        elif "coronary" in text or "cad" in text or "ischemic heart" in text:
            return "chronic_ischemic_heart_disease"
        else:
            return "other_mace"
    
    if len(df) == 0:
        return {"mace_type_counts": {}, "mace_type_pct": {}}
    
    df = df.copy()
    df["mace_type"] = df.apply(classify_mace_type, axis=1)
    
    type_counts = df["mace_type"].value_counts()
    type_pct = (type_counts / len(df) * 100).round(2)
    
    info = {
        "mace_type_counts": type_counts.to_dict(),
        "mace_type_pct": type_pct.to_dict(),
        "n_unique_subjects_with_mace": df[id_col].nunique(),
    }
    return info


def generate_survival_analysis_report(
    eligible_df: pd.DataFrame,
    mace_df: pd.DataFrame = None,
    id_col: str = "idno",
    mace_year_col: str = "1st_MACE_year",
    death_year_col: str = "death_year",
    baseline_year_col: str = "visit_year",
    censor_year: int = None,
    output_path: str = None
) -> dict:
    """
    Generate a comprehensive report of dataset information for time-to-event analysis.
    
    Parameters
    ----------
    eligible_df : pd.DataFrame
        Dataset of eligible subjects with event information
    mace_df : pd.DataFrame, optional
        Dataset with detailed MACE event information
    id_col : str
        Column for subject identifier
    mace_year_col : str
        Column for MACE event year
    death_year_col : str
        Column for death year
    baseline_year_col : str
        Column for baseline year
    censor_year : int, optional
        Administrative censoring year
    output_path : str, optional
        Path to save the report as text file
        
    Returns
    -------
    dict
        Comprehensive report dictionary
    """
    report = {
        "report_generated_at": datetime.now().isoformat(),
        "sample_size": record_sample_size_info(eligible_df, id_col),
        "event_info": record_event_info(
            eligible_df, mace_year_col, death_year_col, baseline_year_col, id_col
        ),
        "followup_time": record_followup_time_info(
            eligible_df, mace_year_col, death_year_col, baseline_year_col, id_col, censor_year
        ),
        "baseline_characteristics": record_baseline_characteristics(
            eligible_df, baseline_year_col, id_col
        ),
        "missing_data": record_missing_data_info(
            eligible_df,
            [id_col, mace_year_col, death_year_col, baseline_year_col],
            id_col
        ),
    }
    
    if mace_df is not None and len(mace_df) > 0:
        report["mace_type_breakdown"] = record_event_type_breakdown(mace_df, id_col=id_col)
    
    # Print formatted report
    print("\n" + "=" * 70)
    print("SURVIVAL ANALYSIS DATASET REPORT")
    print("=" * 70)
    
    print("\n--- SAMPLE SIZE ---")
    for k, v in report["sample_size"].items():
        print(f"  {k}: {v}")
    
    print("\n--- EVENT INFORMATION ---")
    for k, v in report["event_info"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
    
    print("\n--- FOLLOW-UP TIME (Years) ---")
    for k, v in report["followup_time"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
    
    print("\n--- BASELINE CHARACTERISTICS ---")
    for k, v in report["baseline_characteristics"].items():
        print(f"  {k}: {v}")
    
    print("\n--- MISSING DATA ---")
    missing = report["missing_data"]
    print(f"  Complete cases: {missing['n_complete_cases']} ({missing['pct_complete_cases']:.1f}%)")
    for col, info in missing["column_missing_info"].items():
        if info["n_missing"] > 0:
            print(f"  {col}: {info['n_missing']} missing ({info['pct_missing']}%)")
    
    if "mace_type_breakdown" in report:
        print("\n--- MACE EVENT TYPE BREAKDOWN ---")
        for mace_type, count in report["mace_type_breakdown"]["mace_type_counts"].items():
            pct = report["mace_type_breakdown"]["mace_type_pct"].get(mace_type, 0)
            print(f"  {mace_type}: {count} ({pct}%)")
    
    print("\n" + "=" * 70)
    
    # Save report to file if path provided
    if output_path:
        import json
        # Convert non-serializable items
        def serialize(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, tuple):
                return list(obj)
            return obj
        
        serializable_report = json.loads(
            json.dumps(report, default=serialize)
        )
        with open(output_path, 'w') as f:
            json.dump(serializable_report, f, indent=2)
        print(f"\nReport saved to: {output_path}")
    
    return report


if __name__ == "__main__":
    eligible_df = pd.read_csv("eligiable_with_mace_events.csv")
    mace_df = pd.read_csv("mace_filtered_combined.csv")
    
    report = generate_survival_analysis_report(
        eligible_df=eligible_df,
        mace_df=mace_df,
        id_col="idno",
        mace_year_col="1st_MACE_year",
        death_year_col="death_year",
        baseline_year_col="visit_year",
        censor_year=2024,
        output_path="survival_analysis_report.json"
    )