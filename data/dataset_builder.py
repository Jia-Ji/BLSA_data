import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from scipy import stats


from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ActiheartDatasetBuilder:
    """
    Build minute-level and subject-level datasets from Actiheart HR + PA data.

    Expected minute-level columns:
        idno, visit, date, minute_index, hr, pa

    Expected covariate/outcome columns may include:
        idno, sex, age, smoke, weight, height,
        death_event, death_followup_days
    """

    path: dict
    output: dict
    min_valid_minutes_per_day: int = 600
    min_valid_minutes_per_subject: int = 1000
    hr_min: float = 20
    hr_max: float = 200
    pa_min: float = 0

    def __post_init__(self) -> None:
        # data loading
        hr_pa_df = pd.read_csv(self.path['hr_pa_path'])
        event_df = pd.read_csv(self.path['event_path'])

        self.minute_df = hr_pa_df
        self.covariate_df = event_df

        # output paths
        self.output_minute_path = self.output['processed_minute_data_path']
        self.output_subject_path = self.output['subject_feature_data_path']
        self.output_analysis_path = self.output['analysis_data_path']
        self.output_data_quality_path = self.output.get('data_quality_path')

        # outputs
        self.processed_minute_df: pd.DataFrame = pd.DataFrame()
        self.subject_feature_df: pd.DataFrame = pd.DataFrame()
        self.analysis_df: pd.DataFrame = pd.DataFrame()
        self.data_quality_df: pd.DataFrame = pd.DataFrame()

    def build_data_quality_table(
        self,
        cleaned_minute_df: pd.DataFrame,
        processed_minute_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        required_cols = ['idno', 'visit', 'day', 'valid_row']
        missing_cols = [c for c in required_cols if c not in cleaned_minute_df.columns]
        if missing_cols:
            raise ValueError(f"cleaned_minute_df missing required columns: {missing_cols}")

        raw_day = (
            cleaned_minute_df
            .groupby(['idno', 'visit', 'day'], dropna=False)
            .agg(
                n_rows_day=('valid_row', 'size'),
                valid_minutes_day=('valid_row', 'sum'),
            )
            .reset_index()
        )
        raw_day['date_valid'] = raw_day['valid_minutes_day'] >= self.min_valid_minutes_per_day

        if processed_minute_df is None or processed_minute_df.empty:
            raw_day['retained_prop'] = np.nan
        else:
            proc_required = ['idno', 'visit', 'day']
            proc_missing = [c for c in proc_required if c not in processed_minute_df.columns]
            if proc_missing:
                raise ValueError(f"processed_minute_df missing required columns: {proc_missing}")

            processed_day = (
                processed_minute_df
                .groupby(['idno', 'visit', 'day'], dropna=False)
                .size()
                .reset_index(name='n_rows_processed_day')
            )

            raw_day = raw_day.merge(processed_day, on=['idno', 'visit', 'day'], how='left')
            raw_day['n_rows_processed_day'] = raw_day['n_rows_processed_day'].fillna(0).astype(int)
            raw_day['retained_prop'] = raw_day['n_rows_processed_day'] / raw_day['n_rows_day']

        raw_day['date'] = pd.to_datetime(raw_day['day']).dt.strftime('%Y-%m-%d')

        out_cols = ['idno', 'visit', 'date', 'date_valid', 'retained_prop']
        data_quality = raw_day[out_cols].sort_values(['idno', 'visit', 'date']).reset_index(drop=True)
        self.data_quality_df = data_quality
        return data_quality

    def preprocess_minute_data(self) -> pd.DataFrame:
        df = self.minute_df.copy()

        required_cols = ['idno', 'visit', 'date', 'minute_index', 'hr', 'pa']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in minute_df: {missing_cols}")

        df['visit_date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values(['idno', 'visit', 'visit_date', 'minute_index']).reset_index(drop=True)

        # remove impossible values
        df.loc[(df['hr'] < self.hr_min) | (df['hr'] > self.hr_max), 'hr'] = np.nan
        df.loc[df['pa'] < self.pa_min, 'pa'] = np.nan

        # day variable
        df['day'] = df['visit_date'].dt.date

        # row validity
        df['valid_row'] = df['hr'].notna() & df['pa'].notna()

        cleaned_unfiltered_df = df[['idno', 'visit', 'day', 'valid_row']].copy()

        # keep days with enough valid data
        day_summary = (
            df.groupby(['idno', 'visit', 'day'])['valid_row']
            .sum()
            .reset_index(name='valid_minutes_day')
        )

        keep_days = day_summary[day_summary['valid_minutes_day'] >= self.min_valid_minutes_per_day]
        df = df.merge(
            keep_days[['idno', 'visit', 'day']],
            on=['idno', 'visit', 'day'],
            how='inner'
        )

        # keep subjects with enough total valid minutes across all retained rows
        subj_summary = (
            df.groupby('idno')['valid_row']
            .sum()
            .reset_index(name='valid_minutes_subject')
        )
        keep_subjects = subj_summary[
            subj_summary['valid_minutes_subject'] >= self.min_valid_minutes_per_subject
        ]
        df = df.merge(keep_subjects[['idno']], on='idno', how='inner')

        # time features
        df['minute_of_day'] = df['minute_index'] % 1440
        df['hour_of_day'] = df['minute_of_day'] / 60.0

        # cyclical time features
        df['tod_sin'] = np.sin(2 * np.pi * df['minute_of_day'] / 1440.0)
        df['tod_cos'] = np.cos(2 * np.pi * df['minute_of_day'] / 1440.0)

        # rolling PA features per day
        group_cols = ['idno', 'visit', 'day']
        df['pa_5min_mean'] = (
            df.groupby(group_cols)['pa']
            .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
        )
        df['pa_15min_mean'] = (
            df.groupby(group_cols)['pa']
            .transform(lambda x: x.rolling(window=15, min_periods=1).mean())
        )

        self.processed_minute_df = df.reset_index(drop=True)

        self.build_data_quality_table(
            cleaned_minute_df=cleaned_unfiltered_df,
            processed_minute_df=self.processed_minute_df
        )

        if self.output_data_quality_path:
            self.data_quality_df.to_csv(self.output_data_quality_path, index=False)

        self.processed_minute_df.to_csv(self.output_minute_path, index=False)
        return self.processed_minute_df

    def keep_first_actiheart_visit(self) -> pd.DataFrame:
        """
        Keep only the earliest retained Actiheart visit for each subject.

        A visit is considered eligible if it remains in processed_minute_df after
        preprocessing and filtering.
        """
        if self.processed_minute_df.empty:
            raise ValueError("Run preprocess_minute_data() first.")

        df = self.processed_minute_df.copy()

        # get one date per idno-visit
        visit_level = (
            df.groupby(['idno', 'visit'], as_index=False)
            .agg(
                visit_date=('visit_date', 'min'),
                n_valid_rows=('valid_row', 'sum')
            )
            .sort_values(['idno', 'visit_date', 'visit'])
        )

        # keep first eligible visit per subject
        first_visit = (
            visit_level
            .groupby('idno', as_index=False)
            .first()[['idno', 'visit', 'visit_date']]
        )

        df_first = df.merge(
            first_visit[['idno', 'visit']],
            on=['idno', 'visit'],
            how='inner'
        ).copy()

        self.processed_minute_df = df_first.sort_values(
            ['idno', 'visit', 'visit_date', 'minute_index']
        ).reset_index(drop=True)

        return self.processed_minute_df

    def build_subject_features(self) -> pd.DataFrame:
        """
        Create subject-level features for later modeling and comparison.
        Assumes processed_minute_df already contains only the rows you want to use
        for subject-level analysis, e.g. first Actiheart visit only.
        """
        if self.processed_minute_df.empty:
            raise ValueError("Run preprocess_minute_data() first.")

        df = self.processed_minute_df.copy()

        def safe_corr(x, y):
            valid = x.notna() & y.notna()
            if valid.sum() < 3:
                return np.nan
            return np.corrcoef(x[valid], y[valid])[0, 1]

        features = []
        for (idno, visit), sub in df.groupby(['idno', 'visit']):
            valid = sub[sub['valid_row']].copy()
            if len(valid) == 0:
                continue

            feat = {
                'idno': idno,
                'visit': visit,
                'visit_date': sub['visit_date'].min(),
                'n_rows': len(sub),
                'n_valid_rows': len(valid),
                'n_days': sub['day'].nunique(),

                'mean_hr': valid['hr'].mean(),
                'std_hr': valid['hr'].std(),
                'median_hr': valid['hr'].median(),
                'min_hr': valid['hr'].min(),
                'max_hr': valid['hr'].max(),

                'mean_pa': valid['pa'].mean(),
                'std_pa': valid['pa'].std(),
                'median_pa': valid['pa'].median(),
                'min_pa': valid['pa'].min(),
                'max_pa': valid['pa'].max(),

                'mean_pa_5min': valid['pa_5min_mean'].mean(),
                'mean_pa_15min': valid['pa_15min_mean'].mean(),

                'corr_hr_pa': safe_corr(valid['hr'], valid['pa']),
                'wear_time_ratio': len(valid) / len(sub) if len(sub) > 0 else np.nan,
                'mean_hour_of_day': valid['hour_of_day'].mean(),
            }

            feat['prop_sedentary'] = (valid['pa'] == 0).mean()
            feat['prop_active'] = (valid['pa'] > 0).mean()

            sedentary = valid[valid['pa'] == 0]
            active = valid[valid['pa'] > 0]

            feat['mean_hr_sedentary'] = sedentary['hr'].mean() if len(sedentary) > 0 else np.nan
            feat['mean_hr_active'] = active['hr'].mean() if len(active) > 0 else np.nan
            feat['delta_hr_active_sedentary'] = (
                feat['mean_hr_active'] - feat['mean_hr_sedentary']
                if pd.notna(feat['mean_hr_active']) and pd.notna(feat['mean_hr_sedentary'])
                else np.nan
            )

            features.append(feat)

        self.subject_feature_df = pd.DataFrame(features).sort_values(['idno']).reset_index(drop=True)
        self.subject_feature_df.to_csv(self.output_subject_path, index=False)
        return self.subject_feature_df

    def merge_with_covariates(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        if self.covariate_df is None:
            return feature_df.copy()

        out = feature_df.merge(self.covariate_df, on=["idno", "visit"], how="left")

        if {"weight", "height"}.issubset(out.columns):
            height = out["height"].copy()
            if height.median(skipna=True) > 3:
                height = height / 100.0
            out["bmi"] = out["weight"] / (height ** 2)

        self.analysis_df = out.copy()
        self.analysis_df.to_csv(self.output_analysis_path, index=False)
        return out

    @staticmethod
    def add_age_group(df: pd.DataFrame, age_col: str = "age") -> pd.DataFrame:
        out = df.copy()
        out["age_group"] = pd.cut(
            out[age_col],
            bins=[0, 60, 70, 80, np.inf],
            labels=["<60", "60-69", "70-79", "80+"],
            right=False,
        )
        return out

    def summary(self) -> Dict[str, int]:
        return {
            'n_processed_rows': len(self.processed_minute_df),
            'n_subject_features': len(self.subject_feature_df),
            'n_analysis_subjects': len(self.analysis_df),
        }