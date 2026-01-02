from pathlib import Path
from typing import Dict, Generator, List, Optional, Union

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
        snapshots = [
            {"header": {"pseudo_coord": coord, **snapshot["header"]}, "stars": stars_df}
            for coord, series in raw_data.items()
            for timestamp, snapshot in series.items()
            if len(
                stars_df := pd.DataFrame(snapshot["stars"])
                .query("is_within_2x_r_tidal")
                .reset_index(drop=True)
            )
            >= min_stars
        ]
        instance = cls(snapshots=snapshots)
        instance._data_filepath = str(filepath)
        return instance

    def sample_iter(
        self,
        feature_keys: List[str],
        target_keys: List[str],
        n_sample_per_snapshot: int = 16,
        n_star_per_snapshot: int = 128,
        drop_ratio_range: tuple = (0.15, 0.25),
        drop_probability: float = 0.2,
        seed: int = 42,
    ) -> Generator[Dict[str, np.ndarray], None, None]:
        feature_keys_arr = np.array(feature_keys, dtype="<U64")
        target_keys_arr = np.array(target_keys, dtype="<U64")

        for snapshot_idx, snapshot in enumerate(self.snapshots):
            star_df = snapshot["stars"]
            n_stars = len(star_df)
            header_dict = snapshot.get("header", {})

            if not all(k in header_dict for k in target_keys):
                missing = [k for k in target_keys if k not in header_dict]
                raise KeyError(f"Missing target keys in header: {missing}")

            targets = np.array([header_dict[k] for k in target_keys], dtype=np.float32)

            for sample_idx in range(n_sample_per_snapshot):
                rng = np.random.default_rng(snapshot_idx * 100 + sample_idx * 10 + seed)
                # sample snapshots with sufficient stars
                if n_stars >= n_star_per_snapshot:
                    sampled_df = star_df.sample(
                        n=n_star_per_snapshot, replace=False, random_state=rng
                    ).reset_index(drop=True)
                    valid_mask = np.ones(n_star_per_snapshot, dtype=bool)
                    # randomly drop some stars to simulate snapshots with insufficient stars
                    if rng.random() < drop_probability:
                        n_drop = int(
                            n_star_per_snapshot * rng.uniform(*drop_ratio_range)
                        )
                        valid_mask[
                            rng.choice(n_star_per_snapshot, size=n_drop, replace=False)
                        ] = False
                # pad snapshots with insufficient stars
                else:
                    s = star_df.sample(n=n_stars, replace=False, random_state=rng)
                    n_pad = n_star_per_snapshot - n_stars
                    sampled_df = pd.concat(
                        [
                            s,
                            pd.DataFrame({col: [0] * n_pad for col in star_df.columns}),
                        ],
                        ignore_index=True,
                    )
                    valid_mask = np.array([True] * n_stars + [False] * n_pad)

                yield {
                    "features": sampled_df[feature_keys].to_numpy(dtype=np.float32),
                    "feature_keys": feature_keys_arr,
                    "valid_mask": np.asarray(valid_mask, dtype=bool),
                    "targets": targets,
                    "target_keys": target_keys_arr,
                }

    def sample(
        self,
        n_sample_per_snapshot: int = 16,
        n_star_per_snapshot: int = 128,
        feature_keys: Optional[List[str]] = None,
        target_keys: Optional[List[str]] = None,
        drop_ratio_range: tuple = (0.15, 0.25),
        drop_probability: float = 0.2,
        seed: int = 42,
    ) -> List[Dict[str, np.ndarray]]:
        return list(
            self.sample_iter(
                n_sample_per_snapshot=n_sample_per_snapshot,
                n_star_per_snapshot=n_star_per_snapshot,
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
        n_star_per_snapshot=10,
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
