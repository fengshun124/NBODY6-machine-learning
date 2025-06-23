import logging
import os
import re
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
from joblib import delayed

from utils.parallel import ParallelTqdm

SNAPSHOT_FIELD_KEY_DICT = {
    "001 X1": "x",
    "002 X2": "y",
    "003 X3": "z",
    "004 V1": "u",
    "005 V2": "v",
    "006 V3": "w",
    # '023 M': 'mass',
    "032 Name": "uid",
}
SNAPSHOT_FILENAME_PATTERN = re.compile(r"^snap\.40_(\d+)\.h5part$")
SIM_DIRNAME_PATTERN = re.compile(r".*z(\d+)_w(\d+)$")
SIM_ATTR_EXTRACT_CONFIG_DICT = {
    "z": {
        "pattern": re.compile(r".*z(\d+)"),
        "type": int,
    },
    "w": {
        "pattern": re.compile(r".*w(\d+)"),
        "type": int,
    },
}

logger = logging.getLogger(__name__)


class NBodyH5SnapshotCollector:
    def __init__(
        self,
        nbody_simulation_root: Optional[str],
        attr_extract_config_dict: Optional[
            Dict[str, Dict[str, Union[re.Pattern, type]]]
        ] = None,
    ) -> None:
        if attr_extract_config_dict is None:
            attr_extract_config_dict = SIM_ATTR_EXTRACT_CONFIG_DICT
        self._nbody_sim_root = nbody_simulation_root
        self._attr_labels: Tuple[str, ...] = tuple(
            list(attr_extract_config_dict.keys()) + ["age"]
        )
        self._raw_collection_dict: Optional[
            Dict[str, Dict[Tuple[Union[int, float, str], ...], pd.DataFrame]]
        ] = None
        self._uid_aligned_collection_dict: Optional[
            Dict[str, Dict[Tuple[Union[int, float, str], ...], pd.DataFrame]]
        ] = None
        self._snapshot_pairs: Optional[
            List[Tuple[Tuple[Union[int, float, str], ...], pd.DataFrame]]
        ] = None
        self._attr_value_dict: Optional[Dict[str, List[Union[int, float, str]]]] = None

    @property
    def nbody_simulation_root(self) -> str:
        return self._nbody_sim_root

    @property
    def attr_labels(self) -> Tuple[str, ...]:
        return self._attr_labels

    @property
    def attr_value_dict(self) -> Dict[str, List[Union[int, float, str]]]:
        if self._attr_value_dict is not None:
            return self._attr_value_dict
        if self._raw_collection_dict is None:
            logger.warning(
                "Raw collection dictionary is not set. Running collect() first."
            )
            return {}

        attr_value_dict = {
            attr_label: list(
                {
                    attr_value
                    for sim_dict in self._raw_collection_dict.values()
                    for attr_tuple in sim_dict.keys()
                    for label, attr_value in zip(self._attr_labels, attr_tuple)
                    if label == attr_label
                }
            )
            for attr_label in self._attr_labels
        }
        self._attr_value_dict = attr_value_dict
        return self._attr_value_dict

    @property
    def raw_collection(
        self,
    ) -> Dict[str, Dict[Tuple[Union[int, float, str], ...], pd.DataFrame]]:
        if self._raw_collection_dict is None:
            logger.warning(
                "Raw collection dictionary is not set. Running collect() first."
            )
            return {}
        return self._raw_collection_dict

    @property
    def uid_aligned_collection(
        self,
    ) -> Dict[str, Dict[Tuple[Union[int, float, str], ...], pd.DataFrame]]:
        if self._raw_collection_dict is None:
            logger.warning(
                "Raw collection dictionary is not set. Running collect() first."
            )
            return {}

        if self._uid_aligned_collection_dict is not None:
            return self._uid_aligned_collection_dict

        self._uid_aligned_collection_dict = {}
        for sim_name, sim_dict in self._raw_collection_dict.items():
            unique_uids = set().union(
                *(snapshot.index for snapshot in sim_dict.values())
            )

            self._uid_aligned_collection_dict[sim_name] = {}
            for attr_tuple, snapshot_df in sim_dict.items():
                self._uid_aligned_collection_dict[sim_name][attr_tuple] = (
                    snapshot_df.reindex(unique_uids, fill_value=np.nan).sort_index()
                )

        return self._uid_aligned_collection_dict

    @property
    def snapshot_pairs(
        self,
    ) -> List[Tuple[Tuple[Union[int, float, str], ...], pd.DataFrame]]:
        if self._raw_collection_dict is None:
            logger.warning(
                "Raw collection dictionary is not set. Running collect() first."
            )
            return []

        if self._snapshot_pairs is not None:
            return self._snapshot_pairs

        self._snapshot_pairs = [
            (attr_tuple, snapshot_df)
            for sim_dict in self._raw_collection_dict.values()
            for attr_tuple, snapshot_df in sim_dict.items()
        ]
        return self._snapshot_pairs

    @staticmethod
    def _load_snapshot(
        snapshot_h5file: str, attr_dict: dict[str, Union[int, float, str]]
    ) -> Optional[Tuple[pd.DataFrame, dict[str, Union[int, float, str]]]]:
        try:
            with h5py.File(snapshot_h5file, "r") as h5f:
                snapshot_df = (
                    pd.DataFrame(
                        {
                            df_key: h5f["Step#0"][field_key][()]
                            for field_key, df_key in SNAPSHOT_FIELD_KEY_DICT.items()
                        }
                    )
                    .set_index("uid")
                    .sort_index()
                )
                return snapshot_df, attr_dict
        except Exception as e:
            logger.error(f"Failed to collect snapshot {snapshot_h5file}: {e}")
            return None

    def _load_simulation(
        self,
        simulation_dir: str,
        num_jobs: int = -1,
    ) -> Dict[Tuple[Union[int, float, str]], pd.DataFrame]:
        sim_version_str = os.path.basename(simulation_dir)
        sim_base_attr_dict = {}

        for attr_key, config in SIM_ATTR_EXTRACT_CONFIG_DICT.items():
            match = config["pattern"].match(sim_version_str)
            if match:
                sim_base_attr_dict[attr_key] = config["type"](match.group(1))
            else:
                logger.warning(
                    f"Could not extract attribute '{attr_key}' "
                    f"from simulation directory name '{sim_version_str}'."
                )
                return {}

        snapshot_tuples = [
            (
                entry.path,
                {
                    **sim_base_attr_dict,
                    "age": int(SNAPSHOT_FILENAME_PATTERN.match(entry.name).group(1)),
                },
            )
            for entry in os.scandir(simulation_dir)
            if entry.is_file()
        ]
        if not snapshot_tuples:
            logger.warning(f"No valid snapshot files found in {simulation_dir}.")
            return {}

        snapshot_pairs = ParallelTqdm(
            n_jobs=num_jobs,
            desc=f'Loading "{os.path.basename(simulation_dir)}"',
            batch_size="auto",
            backend="loky",
        )(
            delayed(self._load_snapshot)(h5f, attr_dict)
            for h5f, attr_dict in sorted(snapshot_tuples, key=lambda x: x[1]["age"])
        )
        snapshot_pairs = [pair for pair in snapshot_pairs if pair is not None]

        if not snapshot_pairs:
            logger.warning(f"No valid snapshots loaded from {simulation_dir}.")
            return {}

        return {
            tuple(attr_dict[k] for k in self._attr_labels): snapshot_df
            for snapshot_df, attr_dict in snapshot_pairs
        }

    def collect(
        self,
        num_jobs: int = -1,
    ) -> Tuple[
        Dict[str, Dict[Tuple[Union[int, float, str], ...], pd.DataFrame]],
        Tuple[str, ...],
    ]:
        if self._raw_collection_dict is not None:
            return self._raw_collection_dict, self._attr_labels

        sim_dirs = [
            entry.path
            for entry in os.scandir(self._nbody_sim_root)
            if entry.is_dir() and SIM_DIRNAME_PATTERN.match(entry.name)
        ]
        if not sim_dirs:
            logger.warning(
                f"No valid simulation directories found in {self._nbody_sim_root}."
            )
            return {}, self._attr_labels

        logger.info(f"Collecting snapshots from {len(sim_dirs)} simulations...")
        self._raw_collection_dict = {
            os.path.basename(sim_dir): self._load_simulation(sim_dir, num_jobs)
            for sim_dir in sorted(sim_dirs)
        }
        self._raw_collection_dict = {
            k: v for k, v in self._raw_collection_dict.items() if v
        }

        # invalidating cached collections
        self._uid_aligned_collection_dict = None
        self._snapshot_pairs = None
        self._attr_value_dict = None

        return self._raw_collection_dict, self._attr_labels

    def __repr__(self):
        repr_str = (
            f"{self.__class__.__name__}(\n"
            f'  nbody_simulation_root="{os.path.abspath(self._nbody_sim_root)}",\n'
            f"  attribute_labels={self._attr_labels}\n"
        )
        if self._raw_collection_dict:
            repr_str += f"  n_simulations: {len(self._raw_collection_dict)}\n"
            repr_str += f"  n_total_snapshots: {sum(len(v) for v in self._raw_collection_dict.values())}\n"

            repr_str += "  raw_collection_dict:\n   {\n"
            for sim_label, sim_dict in self._raw_collection_dict.items():
                repr_str += f'      "{sim_label}": {len(sim_dict)} snapshot(s)\n'
            repr_str += "   }\n"

        if self._attr_value_dict:
            repr_str += "  attribute_values:\n   {\n"
            for attr_label, values in self._attr_value_dict.items():
                repr_str += f'      "{attr_label}": {len(values)} unique value(s)\n'
            repr_str += "   }\n"

        repr_str += ")"
        return repr_str


def test():
    root_dir = "../../data/N10k"

    collector = NBodyH5SnapshotCollector(nbody_simulation_root=root_dir)
    collector.collect(num_jobs=-1)
    print(collector)

    # print(collector.attr_value_dict)
    collector.attr_value_dict
    print(collector)

    # for collection_dict_label, collection_dict in [
    #     ("raw_collection", collector.raw_collection),
    #     ("uid_aligned_collection", collector.uid_aligned_collection),
    # ]:
    #     print("\n" + f" {collection_dict_label} START ".center(68, "x") + "\n")

    #     for sim_name, sim_data_dict in collection_dict.items():
    #         print("\n" + f" {sim_name} ".center(70, "=") + "\n")
    #         for idx, (attr_tuple, snapshot_df) in enumerate(sim_data_dict.items()):
    #             print(
    #                 " | ".join(
    #                     [
    #                         f" {k}={v}".ljust(8 if k != "age" else 10)
    #                         for k, v in zip(collector.attr_labels, attr_tuple)
    #                     ]
    #                 )
    #                 + "&"
    #                 + " / ".join(
    #                     [
    #                         f"{len(snapshot_df)} row(s)".rjust(14),
    #                         f"{len(snapshot_df.dropna())} Non-NaN row(s)".rjust(18),
    #                     ]
    #                 )
    #             )

    #             if idx % 400 == 0 and idx > 0:
    #                 print(
    #                     f" {idx}th pd.DataFrame ".center(50, "-"),
    #                     snapshot_df.head(4),
    #                     "...",
    #                     snapshot_df.tail(3),
    #                     "-" * 50,
    #                     sep="\n",
    #                 )

    #     print("\n" + f" {collection_dict_label} END ".center(48, "x") + "\n")


if __name__ == "__main__":
    test()
