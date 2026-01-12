import warnings
from pathlib import Path
from typing import Literal

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
    "standard",  # StandardScaler: (x - mean) / std
    "minmax",  # MinMaxScaler: (x - min) / (max - min)
    "robust",  # RobustScaler: uses median/IQR, robust to outliers
    "log1p_standard",  # log1p then standard
    "log1p_minmax",  # log1p then min-max
    "log10_standard",  # log10 then standard
    "log10_minmax",  # log10 then min-max
    "power",  # PowerTransformer (Yeo-Johnson), handles negative values
    "quantile",  # QuantileTransformer, maps to uniform/normal distribution
    "identity",  # no transformation
]

NORM_METHODS: tuple[str, ...] = (
    "standard",
    "minmax",
    "robust",
    "log1p_standard",
    "log1p_minmax",
    "log10_standard",
    "log10_minmax",
    "power",
    "quantile",
    "identity",
)
NORM_METHODS_SET = frozenset(NORM_METHODS)

EPS = 1e-10


def _identity_fn(x):
    return x


def _identity_inv(x):
    return x


def _log10_transform(x):
    return np.log10(x + EPS)


def _log10_inverse(x):
    return np.clip(np.power(10.0, x) - EPS, 0, None)


class ArrayScaler:
    def __init__(self, method: NormMethod) -> None:
        self._method, self._scaler = self._init_scaler(method)
        self._is_fitted = False

    def __repr__(self):
        return f"{self.__class__.__name__}(method='{self._method}', is_fitted={self._is_fitted})"

    @staticmethod
    def _init_scaler(
        method: NormMethod,
    ) -> tuple[NormMethod, Pipeline | FunctionTransformer]:
        method = method.lower().replace("-", "_")

        # assign scaler
        identity = FunctionTransformer(
            func=_identity_fn, inverse_func=_identity_inv, validate=True
        )
        match method:
            case "identity":
                return "identity", identity
            case "standard":
                return "standard", StandardScaler()
            case "minmax":
                return "minmax", MinMaxScaler()
            case "robust":
                return "robust", RobustScaler()
            case "power":
                return "power", PowerTransformer(method="yeo-johnson")
            case "quantile":
                return "quantile", QuantileTransformer(
                    output_distribution="normal", random_state=42
                )
            case _ if method.startswith("log"):
                is_log1p = "log1p" in method
                log_fn = np.log1p if is_log1p else _log10_transform
                inv_fn = np.expm1 if is_log1p else _log10_inverse
                scaler = StandardScaler() if "standard" in method else MinMaxScaler()

                return method, Pipeline(
                    [
                        (
                            "log",
                            FunctionTransformer(
                                func=log_fn,
                                inverse_func=inv_fn,
                                validate=True,
                            ),
                        ),
                        ("scale", scaler),
                    ]
                )
            case _:
                warnings.warn(
                    f"Unknown normalization method '{method}'. Using 'identity'."
                )
                return "identity", identity

    def fit(self, data: np.ndarray) -> None:
        # normalize across ALL points and features simultaneously
        self._scaler.fit(np.asarray(data).reshape(-1, 1))
        self._is_fitted = True

    def transform(self, data: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError(
                f"{__class__.__name__} not fitted yet. Call `fit()` first."
            )
        data = np.asarray(data)
        return self._scaler.transform(data.reshape(-1, 1)).reshape(data.shape)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError(
                f"{__class__.__name__} not fitted yet. "
                "Call `fit()` before `inverse_transform()`."
            )

        data = np.asarray(data)
        return self._scaler.inverse_transform(data.reshape(-1, 1)).reshape(data.shape)

    def to_dict(self) -> dict[str, str]:
        return {"method": self._method}


class ArrayScalerBundle:
    def __init__(self, scaler_config: dict[tuple[str, ...], NormMethod]) -> None:
        self._scalars: dict[tuple[str, ...], ArrayScaler] = {
            keys: ArrayScaler(method) for keys, method in scaler_config.items()
        }
        self._is_fitted = False
        self._fitted_keys: tuple[str, ...] | None = None
        self._validate_config()

    def __repr__(self):
        return f"{self.__class__.__name__}(scalers={list(self._scalars.keys())}, is_fitted={self._is_fitted})"

    def _validate_config(self) -> None:
        if not self._scalars:
            warnings.warn("ArrayScalerBundle initialized with empty configuration.")
            return

        all_keys = [k for keys in self._scalars.keys() for k in keys]
        duplicates = {k for k in all_keys if all_keys.count(k) > 1}
        if duplicates:
            raise ValueError(f"Overlapping keys in config: {duplicates}")

    def _get_indices_map(
        self, keys: tuple[str, ...]
    ) -> dict[tuple[str, ...], list[int]]:
        key_to_idx = {key: idx for idx, key in enumerate(keys)}
        return {
            key_group: [key_to_idx[k] for k in key_group if k in key_to_idx]
            for key_group in self._scalars.keys()
        }

    def _validate_data(self, data: np.ndarray, keys: tuple[str, ...], op: str) -> None:
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {data.shape}")
        if data.shape[1] != len(keys):
            raise ValueError(
                f"Data has {data.shape[1]} columns but {len(keys)} keys provided"
            )

        configured_keys = {k for keys in self._scalars.keys() for k in keys}
        provided_keys = set(keys)
        missing = provided_keys - configured_keys
        if missing:
            raise ValueError(f"Keys {missing} not in config during {op}")

        unused = configured_keys - provided_keys
        if unused:
            warnings.warn(f"Configured keys {unused} not provided during {op}")

    def fit(self, data: np.ndarray, keys: tuple[str, ...]) -> None:
        data = np.asarray(data)
        self._validate_data(data, keys, "fit")

        indices_map = self._get_indices_map(keys)
        for key_group, scaler in self._scalars.items():
            if indices := indices_map.get(key_group):
                scaler.fit(data[:, indices])

        self._is_fitted = True
        self._fitted_keys = keys

    def transform(self, data: np.ndarray, keys: tuple[str, ...]) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} not fitted. Call fit() first"
            )

        data = np.asarray(data)
        self._validate_data(data, keys, "transform")

        if self._fitted_keys != keys:
            warnings.warn(
                f"Transform keys {keys} differ from fitted keys {self._fitted_keys}"
            )

        result = data.copy()
        indices_map = self._get_indices_map(keys)
        for key_group, scaler in self._scalars.items():
            if indices := indices_map.get(key_group):
                result[:, indices] = scaler.transform(data[:, indices])

        return result

    def inverse_transform(self, data: np.ndarray, keys: tuple[str, ...]) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} not fitted. Call fit() first"
            )

        data = np.asarray(data)
        self._validate_data(data, keys, "inverse_transform")

        if self._fitted_keys != keys:
            warnings.warn(
                f"Inverse transform keys {keys} differ from fitted keys {self._fitted_keys}"
            )

        result = data.copy()
        indices_map = self._get_indices_map(keys)
        for key_group, scaler in self._scalars.items():
            if indices := indices_map.get(key_group):
                result[:, indices] = scaler.inverse_transform(data[:, indices])

        return result

    def fit_transform(self, data: np.ndarray, keys: tuple[str, ...]) -> np.ndarray:
        self.fit(data, keys)
        return self.transform(data, keys)

    def to_dict(self) -> dict[str, dict[str, str]]:
        return {
            ",".join(keys): scaler.to_dict() for keys, scaler in self._scalars.items()
        }

    @classmethod
    def from_dict(cls, config: dict[str, dict[str, str]]) -> "ArrayScalerBundle":
        scaler_config = {
            tuple(keys.split(",")): v["method"] for keys, v in config.items()
        }
        return cls(scaler_config)

    def to_joblib(self, filepath: Path | str) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def from_joblib(cls, filepath: Path | str) -> "ArrayScalerBundle":
        filepath = Path(filepath)
        return joblib.load(filepath)


if __name__ == "__main__":
    import tempfile

    # 500 samples with 6 features
    keys = ("x", "y", "z", "vx", "vy", "vz")
    data = np.random.rand(500, len(keys)) * 10

    bundle = ArrayScalerBundle(
        {
            ("x", "y", "z"): "standard",
            ("vx", "vy", "vz"): "robust",
        }
    )
    print(bundle)

    bundle.fit(data, keys)
    print(bundle)

    rec_data = bundle.inverse_transform(bundle.transform(data, keys), keys)
    assert np.allclose(data, rec_data)
    print("Reconstruction successful!")

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "scaler_bundle.joblib"
        bundle.to_joblib(config_path)
        print(f"Saved bundle to {config_path}")

        loaded_bundle = ArrayScalerBundle.from_joblib(config_path)
        print(loaded_bundle)

        rec_loaded = loaded_bundle.inverse_transform(bundle.transform(data, keys), keys)
        assert np.allclose(data, rec_loaded)
        print("Loaded bundle works correctly!")

    print("All tests passed.")
