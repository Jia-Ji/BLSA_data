import pandas as pd
from typing import Optional

from models.base_model import BaseModel


class TwoStepPipeline:
    """
    Step 1: fit reaction model on minute-level data -> subject-level features
    Step 2: merge covariates/outcomes -> fit outcome model
    """

    def __init__(self, reaction_model: BaseModel, outcome_model: BaseModel, dataset_builder):
        self.reaction_model = reaction_model
        self.outcome_model = outcome_model
        self.dataset_builder = dataset_builder

        self.minute_df_ = None
        self.reaction_feature_df_ = None
        self.analysis_df_ = None

    def fit(self, minute_df: Optional[pd.DataFrame] = None):
        if minute_df is not None:
            self.dataset_builder.minute_df = minute_df.copy()

        # preprocess
        self.minute_df_ = self.dataset_builder.preprocess_minute_data()

        # step 1
        self.reaction_model.fit(self.minute_df_)
        self.reaction_feature_df_ = self.reaction_model.predict(self.minute_df_)

        # merge with covariates / outcomes
        self.analysis_df_ = self.dataset_builder.merge_with_covariates(self.reaction_feature_df_)

        # step 2
        self.outcome_model.fit(self.analysis_df_)

        return self

    def predict(self, minute_df: pd.DataFrame) -> pd.Series:
        temp_builder = self.dataset_builder
        temp_builder.minute_df = minute_df.copy()
        minute_processed = temp_builder.preprocess_minute_data()

        reaction_features = self.reaction_model.predict(minute_processed)
        analysis_df = temp_builder.merge_with_covariates(reaction_features)

        return self.outcome_model.predict(analysis_df)

    def predict_risk(self, minute_df: pd.DataFrame):
        temp_builder = self.dataset_builder
        temp_builder.minute_df = minute_df.copy()
        minute_processed = temp_builder.preprocess_minute_data()

        reaction_features = self.reaction_model.predict(minute_processed)
        analysis_df = temp_builder.merge_with_covariates(reaction_features)

        return self.outcome_model.predict_risk(analysis_df)

    def get_analysis_dataset(self) -> pd.DataFrame:
        if self.analysis_df_ is None:
            raise ValueError("Pipeline has not been fitted.")
        return self.analysis_df_.copy()