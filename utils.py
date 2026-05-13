import sys
from contextlib import redirect_stdout
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class _Tee:
    """Write to multiple file-like objects (e.g. console + main_log.txt)."""

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

def create_preprocessor(feature_cols, categorical_cols):
    
    preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "num",
                        Pipeline([
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler())
                        ]),
                        feature_cols,
                    ),
                    (
                        "cat",
                        Pipeline([
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore"))
                        ]),
                        categorical_cols,
                    ),
                ],
                remainder="drop"
            )
    return preprocessor