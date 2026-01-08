from pathlib import Path
from typing import Dict, Generator, List, Optional, Sequence, Union

import joblib
import numpy as np
import pandas as pd

Metadata = Dict[str, Union[int, float]]
Snapshot = Dict[str, Union[Metadata, pd.DataFrame]]


class SnapshotCollector:
    def __init__(self, snapshots: List[Snapshot]) -> None:
        self._snapshots = snapshots
        self._data_filepath: Optional[str] = None

    @property
    def snapshots(self) -> List[Snapshot]:
        return self._snapshots

    def __len__(self) -> int:
        return len(self._snapshots)

    def __repr__(self):
        meta = (
            f"num_snapshots={len(self)}"
            if self._data_filepath is None
            else f"data_filepath='{self._data_filepath}', num_snapshots={len(self)}"
        )
        return f"{self.__class__.__name__}({meta})"

    def __getitem__(self, idx: int) -> Snapshot:
        return self._snapshots[idx]

    @classmethod
    def from_joblib(
        cls,
        filepath: str,
        min_stars: int = 10,
    ) -> "SnapshotCollector":
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        raw_data = joblib.load(filepath)
        snapshots = []
        for coord, series in raw_data.items():
            for timestamp, snapshot in series.items():
                stars_df = pd.DataFrame(snapshot["stars"])

                # Optimization: Use boolean indexing instead of .query() for better performance
                if "is_within_2x_r_tidal" in stars_df.columns:
                    stars_df = stars_df[stars_df["is_within_2x_r_tidal"]]

                stars_df = stars_df.reset_index(drop=True)

                if len(stars_df) >= min_stars:
                    snapshots.append(
                        {
                            "header": {"pseudo_coord": coord, **snapshot["header"]},
                            "stars": stars_df,
                        }
                    )

        instance = cls(snapshots=snapshots)
        instance._data_filepath = str(filepath)
        return instance

    def sample_iter(
        self,
        feature_keys: Sequence[str],
        target_keys: Sequence[str],
        n_sample_per_snapshot: int = 16,
        n_star_per_sample: int = 128,
        drop_ratio_range: tuple = (0.15, 0.25),
        drop_probability: float = 0.2,
        seed: int = 42,
    ) -> Generator[Dict[str, np.ndarray], None, None]:
        if isinstance(feature_keys, str) or isinstance(target_keys, str):
            raise TypeError(
                "Expected feature_keys and target_keys to be sequences of strings, "
                f"got type(feature_keys)={type(feature_keys)}, "
                f"type(target_keys)={type(target_keys)} instead."
            )

        feature_keys_arr = np.array(tuple(feature_keys), dtype="<U64")
        target_keys_arr = np.array(tuple(target_keys), dtype="<U64")

        for snapshot_idx, snapshot in enumerate(self.snapshots):
            star_df = snapshot["stars"]
            n_stars = len(star_df)
            header_dict = snapshot.get("header", {})

            if not all(k in header_dict for k in target_keys):
                missing = [k for k in target_keys if k not in header_dict]
                raise KeyError(f"Missing target keys in header: {missing}")

            targets = np.array([header_dict[k] for k in target_keys], dtype=np.float32)

            if not all(k in star_df.columns for k in feature_keys):
                missing = [k for k in feature_keys if k not in star_df.columns]
                raise KeyError(f"Missing feature keys in stars dataframe: {missing}")

            full_features = star_df[feature_keys].to_numpy(dtype=np.float32)

            for sample_idx in range(n_sample_per_snapshot):
                ss = np.random.SeedSequence([seed, snapshot_idx, sample_idx])
                rng = np.random.default_rng(ss)

                # sample snapshots with sufficient stars
                if n_stars >= n_star_per_sample:
                    indices = rng.choice(n_stars, size=n_star_per_sample, replace=False)
                    features = full_features[indices]
                    valid_mask = np.ones(n_star_per_sample, dtype=bool)

                    # randomly drop some stars to simulate snapshots with insufficient stars
                    if rng.random() < drop_probability:
                        n_drop = int(n_star_per_sample * rng.uniform(*drop_ratio_range))
                        drop_indices = rng.choice(
                            n_star_per_sample, size=n_drop, replace=False
                        )
                        valid_mask[drop_indices] = False
                # pad snapshots with insufficient stars
                else:
                    features = np.zeros(
                        (n_star_per_sample, len(feature_keys)), dtype=np.float32
                    )
                    valid_mask = np.zeros(n_star_per_sample, dtype=bool)

                    # shuffle and fill existing stars
                    indices = rng.permutation(n_stars)
                    features[:n_stars] = full_features[indices]
                    valid_mask[:n_stars] = True

                yield {
                    "features": features,
                    "feature_keys": feature_keys_arr,
                    "valid_mask": valid_mask,
                    "targets": targets,
                    "target_keys": target_keys_arr,
                }

    def sample(
        self,
        feature_keys: Sequence[str],
        target_keys: Sequence[str],
        n_sample_per_snapshot: int = 16,
        n_star_per_sample: int = 128,
        drop_ratio_range: tuple = (0.15, 0.25),
        drop_probability: float = 0.2,
        seed: int = 42,
    ) -> List[Dict[str, np.ndarray]]:
        return list(
            self.sample_iter(
                n_sample_per_snapshot=n_sample_per_snapshot,
                n_star_per_sample=n_star_per_sample,
                feature_keys=feature_keys,
                target_keys=target_keys,
                drop_ratio_range=drop_ratio_range,
                drop_probability=drop_probability,
                seed=seed,
            )
        )


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    load_dotenv()

    empty_collector = SnapshotCollector([])
    print(empty_collector)

    collector = SnapshotCollector.from_joblib(
        (Path(os.getenv("OBS_DIR")) / "Rad12-zmet0014-M8-0509-obs.joblib").resolve()
    )
    print(collector)

    sampled_data = collector.sample(
        n_sample_per_snapshot=4,
        n_star_per_sample=10,
        feature_keys=[
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
            "lon_deg",
            "lat_deg",
            "pm_lon_coslat_mas_yr",
            "pm_lat_mas_yr",
            "log_L_L_sol",
        ],
        target_keys=["time", "total_mass_within_2x_r_tidal"],
    )
    for idx, sample in enumerate(sampled_data):
        print(f" sample {idx} ".center(40, "-"))
        print(sample.keys())
        print(sample)
        if idx >= 3:
            break
