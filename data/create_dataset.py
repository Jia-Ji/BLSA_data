import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from scipy import stats


@dataclass
class ActiheartDatasetBuilder:
    """
    Build minute-level and subject-level datasets from Actiheart HR + PA data.

    Expected minute-level columns:
        idno, visit, date, minute_index, hr, pa

    Expected covariate/outcome columns may include:
        idno, sex, age, smoke, weight, height, death, followup_time
    """

    def __init__(self, path, output,
                 min_valid_minutes_per_day: int = 600, min_valid_minutes_per_subject: int = 1000,
                hr_min: float = 20, hr_max: float = 200, pa_min: float = 0
                ) -> None:
        
        # data loading
        hr_pa_df = pd.read_csv(path['hr_pa_path']) 
        event_df = pd.read_csv(path['event_path'])
        self.minute_df = hr_pa_df
        self.covariate_df = event_df

        # output paths
        self.output_minute_path = output['processed_minute_data_path']
        self.output_subject_path = output['subject_feature_data_path']  
        self.output_analysis_path = output['analysis_data_path']
        self.output_data_quality_path = output.get('data_quality_path')

        # configuration
        self.min_valid_minutes_per_day = min_valid_minutes_per_day
        self.min_valid_minutes_per_subject = min_valid_minutes_per_subject
        self.hr_min = hr_min
        self.hr_max = hr_max
        self.pa_min = pa_min

        self.processed_minute_df: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
        self.subject_feature_df: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
        self.analysis_df: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
        self.data_quality_df: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)

    def build_data_quality_table(
        self,
        cleaned_minute_df: pd.DataFrame,
        processed_minute_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Build a per-day table with idno/visit/date and quality/retention metrics.

        Output columns:
            - idno, visit, date: day-level identifiers (date is YYYY-MM-DD)
            - date_valid: whether the day meets min_valid_minutes_per_day
            - retained_prop: proportion of rows retained in processed dataset for that day
        """
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

        # basic checks
        required_cols = ['idno', 'visit', 'date', 'minute_index', 'hr', 'pa']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in minute_df: {missing_cols}")

        # types
        df['visit_date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['idno', 'visit', 'date', 'minute_index']).reset_index(drop=True)

        # remove impossible values
        df.loc[(df['hr'] < self.hr_min) | (df['hr'] > self.hr_max), 'hr'] = np.nan
        df.loc[df['pa'] < self.pa_min, 'pa'] = np.nan

        # day variable
        df['day'] = df['visit_date'].dt.date

        # row validity
        df['valid_row'] = df['hr'].notna() & df['pa'].notna()

        # build day-quality table on cleaned-but-unfiltered data
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

        # keep subjects with enough total valid minutes
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

        # # optional log transform if PA is skewed
        # df['pa_log'] = np.log1p(df['pa'])

        self.processed_minute_df = df.reset_index(drop=True)

        # finalize day-quality table with retention proportion
        self.build_data_quality_table(
            cleaned_minute_df=cleaned_unfiltered_df,
            processed_minute_df=self.processed_minute_df
        )
        if self.output_data_quality_path:
            self.data_quality_df.to_csv(self.output_data_quality_path, index=False)

        # save processed minute-level data
        self.processed_minute_df.to_csv(self.output_minute_path, index=False)
        return self.processed_minute_df

    def build_subject_features(self) -> pd.DataFrame:
        """
        Create subject-level features for later modeling and comparison.
        """
        if self.processed_minute_df.empty:
            raise ValueError("Run preprocess_minute_data() first.")

        df = self.processed_minute_df.copy()

        # pa_col = 'pa_log' if use_log_pa else 'pa'

        def safe_corr(x, y):
            valid = x.notna() & y.notna()
            if valid.sum() < 3:
                return np.nan
            return np.corrcoef(x[valid], y[valid])[0, 1]

        features = []
        for idno, sub in df.groupby('idno'):
            valid = sub[sub['valid_row']].copy()
            if len(valid) == 0:
                continue

            feat = {
                'idno': idno,
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

            # simple activity zone summaries
            feat['prop_sedentary'] = (valid['pa'] == 0).mean()
            feat['prop_active'] = (valid['pa'] > 0).mean()

            # heart-rate conditional on activity
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

        self.subject_feature_df = pd.DataFrame(features)

        # save subject-level features
        self.subject_feature_df.to_csv(self.output_subject_path, index=False)
        return self.subject_feature_df

    def merge_covariates(self) -> pd.DataFrame:
        """
        Merge subject features with covariates/outcomes.
        """
        if self.subject_feature_df.empty:
            raise ValueError("Run build_subject_features() first.")

        df = self.subject_feature_df.copy()

        if self.covariate_df is not None:
            if 'idno' not in self.covariate_df.columns:
                raise ValueError("covariate_df must contain column 'idno'")
            df = df.merge(self.covariate_df, on='idno', how='left')

        # BMI
        if {'weight', 'height'}.issubset(df.columns):
            df['bmi'] = df['weight'] / (df['height'] ** 2)

        self.analysis_df = df

        # save analysis dataset
        self.analysis_df.to_csv(self.output_analysis_path, index=False)
        return self.analysis_df

    def create_age_group(self,
                         age_col: str = 'age',
                         bins: Optional[List[float]] = None,
                         labels: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Add age group column for comparison analyses.
        """
        if self.analysis_df.empty:
            raise ValueError("Run merge_covariates() first.")

        df = self.analysis_df.copy()

        if age_col not in df.columns:
            raise ValueError(f"Column '{age_col}' not found in analysis_df")

        if bins is None:
            bins = [0, 60, 70, 80, np.inf]
        if labels is None:
            labels = ['<60', '60-69', '70-79', '80+']

        df['age_group'] = pd.cut(df[age_col], bins=bins, labels=labels, right=False)
        self.analysis_df = df

        return self.analysis_df

    def compare_two_groups(self,
                           group_col: str,
                           group1,
                           group2,
                           feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare two groups (e.g., sex=M vs F, death=1 vs 0) using t-test and Mann-Whitney U.
        """
        if self.analysis_df.empty:
            raise ValueError("Run merge_covariates() first.")

        df = self.analysis_df.copy()

        if group_col not in df.columns:
            raise ValueError(f"Column '{group_col}' not found in analysis_df")

        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in feature_cols if c not in ['idno']]

        g1 = df[df[group_col] == group1]
        g2 = df[df[group_col] == group2]

        results = []
        for col in feature_cols:
            x1 = g1[col].dropna()
            x2 = g2[col].dropna()

            if len(x1) < 2 or len(x2) < 2:
                continue

            try:
                t_p = stats.ttest_ind(x1, x2, equal_var=False, nan_policy='omit').pvalue
            except Exception:
                t_p = np.nan

            try:
                u_p = stats.mannwhitneyu(x1, x2, alternative='two-sided').pvalue
            except Exception:
                u_p = np.nan

            results.append({
                'feature': col,
                'group1': group1,
                'group2': group2,
                'mean_group1': x1.mean(),
                'mean_group2': x2.mean(),
                'median_group1': x1.median(),
                'median_group2': x2.median(),
                't_test_p': t_p,
                'mannwhitney_p': u_p
            })

        return pd.DataFrame(results).sort_values('mannwhitney_p')

    def compare_multiple_groups(self,
                                group_col: str,
                                feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare more than two groups (e.g., age groups) using ANOVA and Kruskal-Wallis.
        """
        if self.analysis_df.empty:
            raise ValueError("Run merge_covariates() first.")

        df = self.analysis_df.copy()

        if group_col not in df.columns:
            raise ValueError(f"Column '{group_col}' not found in analysis_df")

        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in feature_cols if c not in ['idno']]

        results = []
        for col in feature_cols:
            groups = [g[col].dropna().values for _, g in df.groupby(group_col)]
            groups = [g for g in groups if len(g) >= 2]

            if len(groups) < 2:
                continue

            try:
                anova_p = stats.f_oneway(*groups).pvalue
            except Exception:
                anova_p = np.nan

            try:
                kw_p = stats.kruskal(*groups).pvalue
            except Exception:
                kw_p = np.nan

            results.append({
                'feature': col,
                'anova_p': anova_p,
                'kruskal_p': kw_p
            })

        return pd.DataFrame(results).sort_values('kruskal_p')

    def get_modeling_dataset(self,
                             feature_cols: Optional[List[str]] = None,
                             target_col: str = 'dateofdeath',
                             dropna_target: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Return X, y for downstream modeling.
        """
        if self.analysis_df.empty:
            raise ValueError("Run merge_covariates() first.")

        df = self.analysis_df.copy()

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in analysis_df")

        if feature_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in numeric_cols if c not in ['idno', target_col]]

        model_df = df[feature_cols + [target_col]].copy()

        if dropna_target:
            model_df = model_df.dropna(subset=[target_col])

        X = model_df[feature_cols]
        y = model_df[target_col]

        return X, y

    def summary(self) -> Dict[str, int]:
        out = {
            'n_processed_rows': len(self.processed_minute_df),
            'n_subject_features': len(self.subject_feature_df),
            'n_analysis_subjects': len(self.analysis_df),
        }
        return out