import warnings
from typing import Any, Dict, Literal, Sequence, Tuple

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
SUPPORTED_METHODS = frozenset(
    [
        "z-score",
        "min-max",
        "robust",
        "log",
        "log-minmax",
        "power",
        "quantile",
        "none",
    ]
)


class Normalizer:
    def __init__(self, columns: Sequence[str], method: NormMethod) -> None:
        self._columns = tuple(columns)
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

    @property
    def supported_methods(self) -> Tuple[str]:
        return tuple(SUPPORTED_METHODS)

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
                        (
                            "log",
                            FunctionTransformer(np.log1p, np.expm1, validate=True),
                        ),
                        ("scale", StandardScaler()),
                    ]
                )
            case "log-minmax":
                return Pipeline(
                    [
                        (
                            "log",
                            FunctionTransformer(np.log1p, np.expm1, validate=True),
                        ),
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
                return FunctionTransformer(
                    func=lambda x: x, inverse_func=lambda x: x, validate=True
                )

    def fit(self, data: np.ndarray) -> "Normalizer":
        if data.size == 0:
            raise ValueError("Cannot fit on empty data.")

        self._fit_shape = data.shape
        n_features = len(self._columns)
        data_2d = data.reshape(-1, n_features)
        self._scaler.fit(data_2d)
        self._is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError(
                "Normalizer not fitted yet. Call `fit()` before `transform()`."
            )

        original_shape = data.shape
        n_features = len(self._columns)
        data_2d = data.reshape(-1, n_features)
        result_2d = self._scaler.transform(data_2d)
        return result_2d.reshape(original_shape)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError(
                "Normalizer not fitted yet. Call `fit()` before `inverse_transform()`."
            )
        if not hasattr(self._scaler, "inverse_transform"):
            raise RuntimeError(
                f"Scaler '{self._method}' does not support inverse transformation."
            )

        original_shape = data.shape
        n_features = len(self._columns)
        data_2d = data.reshape(-1, n_features)
        result_2d = self._scaler.inverse_transform(data_2d)
        return result_2d.reshape(original_shape)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.fit(data).transform(data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "columns": self._columns,
            "method": self._method,
            "scaler": self._scaler if self._is_fitted else None,
            "is_fitted": self._is_fitted,
        }

    def to_joblib(self, path: str) -> None:
        joblib.dump(self.to_dict(), path)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Normalizer":
        method = d["method"].lower()

        # Validate method; fall back to "none" if unsupported
        if method not in SUPPORTED_METHODS:
            warnings.warn(f"Method '{method}' not supported. Falling back to 'none'.")
            method = "none"

        instance = cls(columns=d["columns"], method=method)
        if d["is_fitted"]:
            if d["scaler"] is None:
                raise ValueError(
                    "Inconsistent state: is_fitted=True but scaler is None."
                )
            instance._scaler = d["scaler"]
            instance._is_fitted = True
        return instance

    @classmethod
    def from_joblib(cls, path: str) -> "Normalizer":
        return cls.from_dict(joblib.load(path))

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return f"Normalizer({self._columns}, '{self._method}', {status})"


if __name__ == "__main__":
    # Simple test
    np.random.seed(42)
    x_train = np.random.randn(int(1e7), 3).astype(np.float32)
    x_test = np.random.randn(int(1e3), 3).astype(np.float32)

    columns = ("feat1", "feat2", "feat3")

    normalizer = Normalizer(columns=columns, method="z-score")
    print(f"Created: {normalizer}")

    normalizer.fit(x_train)
    print(f"After fit: {normalizer}")

    x_train_norm = normalizer.transform(x_train)
    print(f"Transform shape: {x_train_norm.shape}")
    print(
        f"Normalized stats - mean: {x_train_norm.mean():.4f}, std: {x_train_norm.std():.4f}"
    )

    x_test_norm = normalizer.transform(x_test)
    x_test_recovered = normalizer.inverse_transform(x_test_norm)
    recovery_error = np.mean((x_test - x_test_recovered) ** 2)
    print(f"Inverse transform MSE: {recovery_error:.2e}")

    # Test save/load
    normalizer.to_joblib("/tmp/test_norm.joblib")
    normalizer_loaded = Normalizer.from_joblib("/tmp/test_norm.joblib")
    print(f"Loaded: {normalizer_loaded}")

    # Verify loaded normalizer
    x_test_norm_loaded = normalizer_loaded.transform(x_test)
    diff = np.mean((x_test_norm - x_test_norm_loaded) ** 2)
    print(f"Loaded normalizer transform difference MSE: {diff:.2e}")
