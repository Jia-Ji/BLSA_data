import pickle
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from lifelines import CoxPHFitter

from models.base_model import BaseModel


class CoxOutcomeModel(BaseModel):
    """
    Predict time-to-death outcome from subject-level features using Cox regression.
    """

    def __init__(self, config):
        super().__init__(config)

        self.name = config.get("name", "CoxModel")
        self.duration_col = config.get("duration_col")
        self.event_col = config.get("event_col")
        self.categorical_cols = config.get("categorical_cols", [])
        self.include_cols = config.get("include_cols", [])
        self.model_params = config.get("model_params", {})

        self.preprocessor_ = None
        self.model_ = None
        self.feature_cols_ = None
        self.categorical_cols_ = None
        self.transformed_feature_names_ = None

        print(f"Initialized {self.name}.")

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit Cox model.

        Option 1:
            fit(df)
            where df already contains duration_col and event_col

        Option 2:
            fit(X_features, y_event)
            where y_event contains duration_col and event_col
        """
        if y is None:
            if self.duration_col not in X.columns:
                raise ValueError(f"Duration column '{self.duration_col}' not found.")
            if self.event_col not in X.columns:
                raise ValueError(f"Event column '{self.event_col}' not found.")

            df = X.copy().dropna(subset=[self.duration_col, self.event_col])

        else:
            if self.duration_col not in y.columns:
                raise ValueError(f"Duration column '{self.duration_col}' not found in y.")
            if self.event_col not in y.columns:
                raise ValueError(f"Event column '{self.event_col}' not found in y.")

            df = X.copy()
            df[self.duration_col] = y[self.duration_col].values
            df[self.event_col] = y[self.event_col].values
            df = df.dropna(subset=[self.event_col])

        df = df[df[self.duration_col] > 0].copy()
        df[self.event_col] = df[self.event_col].astype(int)

        usable_cat = [c for c in self.categorical_cols if c in df.columns]

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [
            c for c in numeric_cols
            if c in self.include_cols or c not in [self.duration_col, self.event_col]
        ]

        self.feature_cols_ = numeric_cols
        self.categorical_cols_ = usable_cat

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())
                    ]),
                    self.feature_cols_,
                ),
                (
                    "cat",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore"))
                    ]),
                    self.categorical_cols_,
                ),
            ],
            remainder="drop"
        )

        feature_input = df[self.feature_cols_ + self.categorical_cols_]

        X_processed = preprocessor.fit_transform(feature_input)

        transformed_feature_names = preprocessor.get_feature_names_out()
        X_processed_df = pd.DataFrame(
            X_processed,
            columns=transformed_feature_names,
            index=df.index
        )

        cox_df = X_processed_df.copy()
        cox_df[self.duration_col] = df[self.duration_col].values
        cox_df[self.event_col] = df[self.event_col].values

        cph = CoxPHFitter(**self.model_params)
        cph.fit(
            cox_df,
            duration_col=self.duration_col,
            event_col=self.event_col
        )

        self.preprocessor_ = preprocessor
        self.model_ = cph
        self.transformed_feature_names_ = list(transformed_feature_names)

        return self

    def predict(self, X: pd.DataFrame):
        """
        Returns partial hazard (relative risk).
        """
        if self.preprocessor_ is None or self.model_ is None:
            raise ValueError("Model has not been fitted.")

        X_input = X[self.feature_cols_ + self.categorical_cols_]
        X_processed = self.preprocessor_.transform(X_input)

        X_processed_df = pd.DataFrame(
            X_processed,
            columns=self.transformed_feature_names_,
            index=X.index
        )

        return self.model_.predict_partial_hazard(X_processed_df)

    def predict_risk(self, X: pd.DataFrame):
        """
        Alias of predict() for Cox model.
        """
        return self.predict(X)

    def predict_survival_function(self, X: pd.DataFrame):
        if self.preprocessor_ is None or self.model_ is None:
            raise ValueError("Model has not been fitted.")

        X_input = X[self.feature_cols_ + self.categorical_cols_]
        X_processed = self.preprocessor_.transform(X_input)

        X_processed_df = pd.DataFrame(
            X_processed,
            columns=self.transformed_feature_names_,
            index=X.index
        )

        return self.model_.predict_survival_function(X_processed_df)

    def print_summary(self):
        if self.model_ is None:
            raise ValueError("Model has not been fitted.")
        self.model_.print_summary()

    def summary(self):
        if self.model_ is None:
            raise ValueError("Model has not been fitted.")
        return self.model_.summary

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)