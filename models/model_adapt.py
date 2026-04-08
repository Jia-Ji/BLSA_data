import torch
from torch import nn, Tensor
from omegaconf import DictConfig
import pandas as pd
from .linear_reg import fit_subject_hr_pa_model

class CompeleteModel(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.__initialize_modules(config)
    
    def __initialize_modules(self, config: DictConfig):





class SubjectHRPAFeaturesExtractor:
    def __init__(self, min_rows=500):
        self.min_rows = min_rows

    def fit_transform(self, df):
        """
        Fit subject-level HR~PA models and return summary features for all subjects.
        """
        features_list = []
        for sub_id, sub_df in df.groupby('idno'):
            features = fit_subject_hr_pa_model(sub_df, self.min_rows)
            if features is not None:
                features_list.append(features)

        features_df = pd.DataFrame(features_list)
        return features_df