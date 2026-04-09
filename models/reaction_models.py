import os
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm

from models.base_model import BaseModel


class LinearReactionModel(BaseModel):
    """
    Per-subject HR ~ PA regression.
    Input: minute-level dataframe
    Output: subject-level feature dataframe
    """

    def __init__(self, config):
        super().__init__(config)
        self.feature_df_ = None

        self.name = config.get("name", "LinearReactionModel")
        self.min_rows = config.get("min_rows_per_subject", 500)
        self.use_pa_log = config.get("use_pa_log", False)
        self.required_cols = config.get("required_cols")
        self.output_features = config.get("output_features")

        print(f"Initialized {self.name}")


    def fit(self, X: pd.DataFrame, y=None):
        missing = [c for c in self.required_cols if c not in X.columns]
        if missing:
            raise ValueError(f"Missing columns for LinearReactionModel.fit: {missing}")


        results = []

        for idno, sub_df in X.groupby("idno"):
            sub_df = sub_df.copy()

            if self.use_pa_log:
                sub_df["pa_used"] = np.log1p(sub_df["pa"])
                sub_df["pa_5min_used"] = np.log1p(sub_df["pa_5min_mean"])
                sub_df["pa_15min_used"] = np.log1p(sub_df["pa_15min_mean"])
            else:
                sub_df["pa_used"] = sub_df["pa"]
                sub_df["pa_5min_used"] = sub_df["pa_5min_mean"]
                sub_df["pa_15min_used"] = sub_df["pa_15min_mean"]

            sub_df = sub_df.dropna(
                subset=["hr", "pa_used", "pa_5min_used", "pa_15min_used", "tod_sin", "tod_cos"]
            )

            if len(sub_df) < self.min_rows:
                continue

            sub_df["pa_sq"] = sub_df["pa_used"] ** 2

            X_reg = sub_df[[
                "pa_used", "pa_sq", "pa_5min_used", "pa_15min_used", "tod_sin", "tod_cos"
            ]]
            y_reg = sub_df["hr"]
            X_reg = sm.add_constant(X_reg)

            try:
                model = sm.OLS(y_reg, X_reg).fit()
            except Exception:
                continue

            feat = {
                "idno": idno,
                "n_minutes_used": len(sub_df),
            }
            for out_feat in self.output_features:
                if out_feat == "hr_pa_intercept":
                    feat.update({"hr_pa_intercept": model.params.get("const", np.nan)})
                if out_feat == "hr_pa_slope":
                    feat.update({"hr_pa_slope": model.params.get("pa_used", np.nan)})
                if out_feat == "hr_pa_quad":
                    feat.update({"hr_pa_quad": model.params.get("pa_sq", np.nan)})
                if out_feat == "hr_pa_5min_effect":
                    feat.update({"hr_pa_5min_effect": model.params.get("pa_5min_used", np.nan)})
                    model.params.get("pa_5min_used", np.nan)
                if out_feat == "hr_pa_15min_effect":
                    feat.update({"hr_pa_15min_effect": model.params.get("pa_15min_used", np.nan)})
                if out_feat == "hr_pa_tod_sin":
                    feat.update({"hr_pa_tod_sin": model.params.get("tod_sin", np.nan)})
                if out_feat == "hr_pa_tod_cos":
                    feat.update({"hr_pa_tod_cos": model.params.get("tod_cos", np.nan)})
                if out_feat == "hr_pa_r2":
                    feat.update({"hr_pa_r2": model.rsquared})
                if out_feat == "hr_pa_resid_std":
                    feat.update({"hr_pa_resid_std": np.std(model.resid)})
                if out_feat == "mean_hr":
                    feat.update({"mean_hr": sub_df["hr"].mean()})
                if out_feat == "std_hr":
                    feat.update({"std_hr": sub_df["hr"].std()})
                if out_feat == "mean_pa":
                    feat.update({"mean_pa": sub_df["pa"].mean()})
                if out_feat == "std_pa":
                    feat.update({"std_pa": sub_df["pa"].std()})
                if out_feat == "corr_hr_pa":
                    feat.update({"corr_hr_pa": sub_df[["hr", "pa"]].corr().iloc[0, 1]})

            results.append(feat)

        self.feature_df_ = pd.DataFrame(results)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        For step 1, predict() returns extracted subject-level reaction features.
        """
        if self.feature_df_ is None:
            raise ValueError("Model has not been fitted.")
        return self.feature_df_.copy()

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)