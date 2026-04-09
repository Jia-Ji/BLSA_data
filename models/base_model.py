from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd


class BaseModel(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def fit(self, X, y=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def predict_risk(self, X):
        """Optional for survival models."""
        raise NotImplementedError

    def save(self, path: str):
        raise NotImplementedError

    @classmethod
    def load(cls, path: str):
        raise NotImplementedError