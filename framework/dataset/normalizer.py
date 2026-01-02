import warnings
from typing import Any, Dict, Literal, Tuple

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

NormMethod = Literal[
    "z-score",  # StandardScaler: (x - mean) / std
    "min-max",  # MinMaxScaler: (x - min) / (max - min)
    "robust",  # RobustScaler: uses median/IQR, robust to outliers
    "log",  # log1p then z-score
    "log-minmax",  # log1p then min-max
    "power",  # PowerTransformer (Yeo-Johnson), handles negative values
    "quantile",  # QuantileTransformer, maps to uniform/normal distribution
    "none",  # no transformation
]


class Normalizer:
    def __init__(self, columns: Tuple[str], method: NormMethod) -> None:
        self._columns = columns

        # enforce lowercase for method
        self._method = method.lower()
        self._is_fitted = False
        self._scaler = self._init_scaler()

    @property
    def method(self) -> str:
        return self._method

    @property
    def columns(self) -> Tuple[str]:
        return self._columns

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def _init_scaler(self):
        match self._method:
            case "z-score":
                return StandardScaler()
            case "min-max":
                return MinMaxScaler()
            case "robust":
                return RobustScaler()
            case "log":
                return Pipeline(
                    [
                        ("log", FunctionTransformer(np.log1p, np.expm1, validate=True)),
                        ("scale", StandardScaler()),
                    ]
                )
            case "log-minmax":
                return Pipeline(
                    [
                        ("log", FunctionTransformer(np.log1p, np.expm1, validate=True)),
                        ("scale", MinMaxScaler()),
                    ]
                )
            case "power":
                return PowerTransformer(method="yeo-johnson", standardize=True)
            case "quantile":
                return QuantileTransformer(
                    output_distribution="normal", random_state=42
                )
            case _:
                warnings.warn(
                    f"Unknown normalization method '{self._method}'. "
                    "No transformation will be applied."
                )
                return FunctionTransformer()

    def fit(self, data: np.ndarray) -> "Normalizer":
        self._scaler.fit(data)
        self._is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError(
                "Normalizer not fitted yet. Call `fit()` before `transform()`."
            )
        return self._scaler.transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError(
                "Normalizer not fitted yet. Call `fit()` before `inverse_transform()`."
            )
        return self._scaler.inverse_transform(data)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.fit(data).transform(data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "columns": self._columns,
            "method": self._method,
            "scaler": self._scaler if self._is_fitted else None,
            "is_fitted": self._is_fitted,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Normalizer":
        instance = cls(columns=d["columns"], method=d["method"])
        if d["is_fitted"] and d["scaler"] is not None:
            instance._scaler = d["scaler"]
            instance._is_fitted = True
        return instance

    def save(self, path: str) -> None:
        joblib.dump(self.to_dict(), path)

    @classmethod
    def load(cls, path: str) -> "Normalizer":
        return cls.from_dict(joblib.load(path))

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return f"Normalizer({self._columns}, '{self._method}', {status})"
