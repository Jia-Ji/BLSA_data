import pickle
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

from models.base_model import BaseModel


class LogisticOutcomeModel(BaseModel):
    """
    Predict binary death outcome from subject-level features.
    """

    def __init__(self, config):
        super().__init__(config)

        self.name = config.get("name", "LogisticOutcomeModel")
        self.target_col = config.get("target_col", "death")
        self.categorical_cols = config.get("categorical_cols")
        self.exclude_cols = config.get("exclude_cols")
        self.model_params = config.get("model_params", {})

        self.pipeline_ = None
        self.feature_cols_ = None
        self.categorical_cols_ = None

        print(f"Initialized {self.name}.")

    def fit(self, X: pd.DataFrame, y=None):
        
        if y is None:
            if self.target_col not in X.columns:
                raise ValueError(f"Target column '{self.target_col}' not found.")
            df = X.copy().dropna(subset=[self.target_col])
            y = df[self.target_col].astype(int)
        else:
            df = X.copy()

        usable_cat = [c for c in self.categorical_cols if c in df.columns]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in self.exclude_cols + [self.target_col]]

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
            ]
        )

        clf = Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(**self.model_params))
        ])

        clf.fit(df[self.feature_cols_ + self.categorical_cols_], y)
        self.pipeline_ = clf
        return self

    def predict(self, X: pd.DataFrame):
        if self.pipeline_ is None:
            raise ValueError("Model has not been fitted.")
        return self.pipeline_.predict(X[self.feature_cols_ + self.categorical_cols_])

    def predict_risk(self, X: pd.DataFrame):
        if self.pipeline_ is None:
            raise ValueError("Model has not been fitted.")
        return self.pipeline_.predict_proba(X[self.feature_cols_ + self.categorical_cols_])[:, 1]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)
        

