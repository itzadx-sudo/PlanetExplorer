from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class TabularPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_cols: list[str] = []

    def fit(self, df: pd.DataFrame, target_col: str, drop_cols: list[str] | None = None) -> "TabularPreprocessor":
        drop_cols = drop_cols or []
        self.feature_cols = [c for c in df.columns if c != target_col and c not in drop_cols]
        self.scaler.fit(df[self.feature_cols])
        return self

    def transform_X(self, df: pd.DataFrame) -> np.ndarray:
        df = df.reindex(columns=self.feature_cols, fill_value=0.0)
        return self.scaler.transform(df)
