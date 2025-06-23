import logging
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from joblib import delayed
from utils.parallel import ParallelTqdm

from module.collector import SIM_ATTR_EXTRACT_CONFIG_DICT, NBodyH5SnapshotCollector

RANDOM_SEED = 42
logger = logging.getLogger(__name__)


class NBodySnapshotSamplerBase(ABC):
    def __init__(
        self,
        nbody_simulation_root: Optional[str] = None,
        attr_extract_config_dict: Optional[
            Dict[str, Dict[str, Union[re.Pattern, type]]]
        ] = None,
    ) -> None:
        self._collector = (
            NBodyH5SnapshotCollector(
                nbody_simulation_root=nbody_simulation_root,
                attr_extract_config_dict=attr_extract_config_dict
                or SIM_ATTR_EXTRACT_CONFIG_DICT,
            )
            if nbody_simulation_root
            else None
        )

        self._sampled_collection_dict: Optional[
            Dict[str, Dict[Tuple[Union[int, float, str], ...], List[pd.DataFrame]]]
        ] = None
        self._sampled_snapshots_pairs: Optional[
            List[Tuple[Union[int, float, str], ...], pd.DataFrame]
        ] = None
        self._sample_strategy_dict: Optional[Dict] = None

    @staticmethod
    @abstractmethod
    def _sample_snapshot(
        snapshot_df: pd.DataFrame, attr_tuple: Tuple[Union[int, float, str]], **kwargs
    ) -> Tuple[Tuple[Union[int, float, str], ...], List[pd.DataFrame]]:
        pass

    def _sample_simulation(
        self,
        simulation_label: str,
        simulation_dict: Dict[Tuple[Union[int, float, str], ...], pd.DataFrame],
        n_jobs: int = -1,
    ) -> Tuple[Tuple[Union[int, float, str], ...], List[pd.DataFrame]]:
        sampled_snapshot_pairs = ParallelTqdm(
            n_jobs=n_jobs,
            desc=f'Sampling "{simulation_label}"',
        )(
            delayed(self._sample_snapshot)(
                snapshot_df, attr_dict, self._sample_strategy_dict
            )
            for attr_dict, snapshot_df in simulation_dict.items()
        )
        sampled_snapshot_pairs = [
            pair for pair in sampled_snapshot_pairs if pair is not None
        ]
        if not sampled_snapshot_pairs:
            logger.warning(
                f"No snapshots sampled for simulation '{simulation_label}'. "
            )
            return {}

        return {
            attr_dict: snapshot_dfs
            for attr_dict, snapshot_dfs in sampled_snapshot_pairs
        }

    def sample(
        self,
        sample_strategy_dict: Dict,
        n_jobs: int = -1,
    ) -> Tuple[
        Dict[str, Dict[Tuple[Union[int, float, str], ...], List[pd.DataFrame]]],
        Tuple[str, ...],
    ]:
        if (
            self._sample_strategy_dict is not None
            and self._sample_strategy_dict == sample_strategy_dict
        ):
            logger.info(
                "Sample strategy is the same as the previous one. "
                "Returning the previously sampled collection."
            )
            return self._sampled_collection_dict, self._collector.attr_labels
        self._sample_strategy_dict = sample_strategy_dict

        if self._sampled_collection_dict is not None:
            return self._sampled_collection_dict, self._collector.attr_labels

        if self._collector is None:
            raise ValueError(
                "Collector is not set. Please construct or set a collector first."
            )

        if self._collector.raw_collection == {}:
            logger.warning(
                "Raw collection dictionary is not set. "
                "Running collect() first to initialize it."
            )
            return {}, self._collector.attr_labels

        logger.info("Sampling snapshots from the raw collection. ")
        self._sampled_collection_dict = {
            simulation_label: self._sample_simulation(
                simulation_label,
                simulation_dict,
                n_jobs=n_jobs,
            )
            for simulation_label, simulation_dict in self._collector.raw_collection.items()
        }
        self._sampled_collection_dict = {
            k: v for k, v in self._sampled_collection_dict.items() if v
        }

        return (
            self._sampled_collection_dict,
            self._collector.attr_labels,
        )

    @property
    def sampled_snapshot_pairs(
        self,
    ) -> Optional[List[Tuple[Tuple[Union[int, float, str], ...], pd.DataFrame]]]:
        if self._sampled_snapshots_pairs is not None:
            return self._sampled_snapshots_pairs

        self._sampled_snapshots_pairs = [
            (attr_dict, snapshot_dfs)
            for snapshot_dict in self._sampled_collection_dict.values()
            for attr_dict, snapshot_dfs in snapshot_dict.items()
        ]
        return self._sampled_snapshots_pairs

    # delegate methods to collector
    def collect(self, n_jobs: int = -1):
        return self._collector.collect(n_jobs)

    @property
    def nbody_simulation_root(self) -> str:
        return self._collector.nbody_simulation_root

    @property
    def attr_labels(self) -> Tuple[str]:
        return self._collector.attr_labels

    @property
    def attr_value_dict(self) -> Dict[str, List[Union[int, float, str]]]:
        return self._collector.attr_value_dict

    @property
    def collector(self) -> NBodyH5SnapshotCollector:
        return self._collector

    @collector.setter
    def collector(self, collector: NBodyH5SnapshotCollector):
        if not isinstance(collector, NBodyH5SnapshotCollector):
            raise TypeError("collector must be an instance of NBodyH5SnapshotCollector")
        self._collector = collector

    def __repr__(self) -> str:
        repr_str = f"{self.__class__.__name__}("

        if self._collector:
            collector_repr = (
                repr(self._collector)
                .replace("\n", "\n  ")
                .replace(
                    "NBodyH5SnapshotCollector", "  collector=NBodyH5SnapshotCollector"
                )
            )
            repr_str += f"\n{collector_repr},\n"

        if self._sample_strategy_dict:
            repr_str += "  sample_strategy_dict={\n"
            for key, value in self._sample_strategy_dict.items():
                repr_str += f"    {key}: {value},\n"
            repr_str += "  },\n"

        if self._sampled_collection_dict:
            repr_str += "  sampled_collection_dict:\n  {\n"
            for sim_label, sim_dict in self._sampled_collection_dict.items():
                repr_str += f'      "{sim_label}": {len(sim_dict)} snapshot(s)\n'
            repr_str += "   }\n"

        repr_str += ")"
        return repr_str


class NBodySnapshotNonEmptySampler(NBodySnapshotSamplerBase):
    @staticmethod
    def _sample_snapshot(snapshot_df, attr_tuple, sample_strategy_dict):
        n_star_per_sample = sample_strategy_dict.get("n_star_per_sample", 256)
        n_sample_per_snapshot = sample_strategy_dict.get("n_sample_per_snapshot", 24)
        random_seed = sample_strategy_dict.get("random_seed", RANDOM_SEED)
        
        # age filter
        if attr_tuple[-1] > 1900 or attr_tuple[-1] == 0:
            return None

        if snapshot_df.empty:
            logger.debug(
                f"Snapshot DataFrame for attributes {attr_tuple} is empty, skipping..."
            )
            return None

        if len(snapshot_df) < n_star_per_sample:
            logger.debug(
                f"Snapshot DataFrame for attributes {attr_tuple} has insufficient stars: "
                f"{len(snapshot_df)} < {n_star_per_sample}, skipping..."
            )
            return None

        if len(snapshot_df) * 1.6 < n_star_per_sample * n_sample_per_snapshot:
            logger.debug(
                f"Insufficient stars (1.6x buffer) in snapshot {attr_tuple} ({len(snapshot_df)}) "
                f"to avoid excessive sampling with {n_sample_per_snapshot} samples of "
                f"{n_star_per_sample} stars each. Skipping..."
            )
            return None

        return (
            attr_tuple,
            [
                snapshot_df.sample(
                    n=n_star_per_sample,
                    replace=False,
                    random_state=(random_seed + i) ** 3,
                )
                for i in range(n_sample_per_snapshot)
            ],
        )

    def sample(
        self,
        n_star_per_sample: int = 256,
        n_sample_per_snapshot: int = 24,
        random_seed: Optional[int] = RANDOM_SEED,
        n_jobs: int = -1,
    ) -> Tuple[
        Dict[str, Dict[Tuple[Union[int, float, str], ...], List[pd.DataFrame]]],
        Tuple[str, ...],
    ]:
        return super().sample(
            sample_strategy_dict={
                "n_star_per_sample": n_star_per_sample,
                "n_sample_per_snapshot": n_sample_per_snapshot,
                "random_seed": random_seed,
            },
            n_jobs=n_jobs,
        )


def test():
    sampler = NBodySnapshotNonEmptySampler(nbody_simulation_root="../../data/N15k")
    sampler.collect()
    print(sampler)

    # collector = NBodyH5SnapshotCollector(nbody_simulation_root="../../data/N15k")
    # collector.collect(-1)

    # sampler.collector = collector
    # print(sampler)

    sampler.sample(
        n_star_per_sample=256,
        n_sample_per_snapshot=24,
        n_jobs=-1,
    )
    sampler.attr_value_dict
    print(sampler)

    # for attr_tuple, snapshot_dfs in sampler.sampled_snapshot_pairs:
    #     print(
    #         f"{attr_tuple}: {len(snapshot_dfs)} sampled snapshots",
    #         " - sample size:".ljust(24)
    #         + " / ".join([f"{len(df)}" for df in snapshot_dfs]),
    #         " - non-empty entries:".ljust(24)
    #         + " / ".join([f"{len(df.dropna())}" for df in snapshot_dfs]),
    #         sep="\n",
    #     )


if __name__ == "__main__":
    test()
