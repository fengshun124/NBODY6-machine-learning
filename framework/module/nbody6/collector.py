import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from module.nbody6.calc.binary import (
    calc_equivalent_radius,
    calc_log_total_luminosity,
    calc_photocentric,
    calc_total_mass,
)
from module.nbody6.calc.misc import (
    calc_log_effective_temperature,
    calc_log_surface_flux,
)
from module.nbody6.loader import NBody6OutputLoader, NBody6Snapshot


class NBody6SnapshotCollector:
    def __init__(self, root: Union[str, Path]) -> None:
        self._root = Path(root).resolve()
        self._loader = NBody6OutputLoader(self._root)

    # proxy methods for NBody6OutputLoader
    def prepare_snapshot(
        self,
        is_strict_load: bool = True,
        is_allow_trim: bool = True,
        is_strict_merge: bool = False,
        is_verbose: bool = False,
    ) -> Optional[Dict[float, NBody6Snapshot]]:
        self._loader.load(is_strict=is_strict_load, is_allow_trim=is_allow_trim)
        self._loader.merge(is_strict=is_strict_merge, is_verbose=is_verbose)

        return self._loader.snapshot_dict

    @property
    def snapshot_dict(self) -> Dict[float, NBody6Snapshot]:
        return self._loader.snapshot_dict

    def build_observed_snapshot(
        self, distance_pc: Union[int, List[int]], is_verbose: bool = False
    ) -> Dict[int, Dict[float, pd.DataFrame]]:
        distance_pc = (
            sorted(set(distance_pc))
            if not isinstance(distance_pc, (int, float))
            else [distance_pc]
        )

        snapshots_by_dist = {}

        for dist_pc in (
            dist_pbar := tqdm(
                distance_pc,
                position=0,
                leave=False,
                disable=not is_verbose,
                dynamic_ncols=True,
            )
        ):
            dist_pbar.set_description(f"Simulating Observation@{dist_pc} pc")
            obs_snapshot_dict = {}
            for timestamp in (
                ts_pbar := tqdm(
                    self._loader.timestamps,
                    position=1,
                    leave=False,
                    disable=not is_verbose,
                    dynamic_ncols=True,
                )
            ):
                ts_pbar.set_description(f"Processing Snapshot@{timestamp:.2f}Myr")

                obs_snapshot_dict[timestamp] = self._build_observed_snapshot(
                    snapshot=self._loader.snapshot_dict[timestamp],
                    semi_threshold=0.6 * dist_pc,
                )
            snapshots_by_dist[dist_pc] = obs_snapshot_dict

        return snapshots_by_dist

    def _build_observed_snapshot(self, snapshot: NBody6Snapshot, semi_threshold: float):
        header_dict = snapshot.header
        main_df = snapshot.data
        pair_df = snapshot.binary_pair

        name_attr_map = main_df.set_index("name").to_dict(orient="index")

        # collect single stars
        single_star_df = main_df[~main_df["is_binary"]].assign(
            is_unresolved_binary=False
        )

        # collect resolved binaries and unresolved binaries
        resolved_pair_df = pair_df[pair_df["semi"] >= semi_threshold]
        resolved_binary_df = main_df[
            main_df["name"].isin(
                resolved_pair_df[["obj1_ids", "obj2_ids"]].sum(axis=1).sum()
            )
        ].assign(is_unresolved_binary=False)
        unresolved_pair_df = pair_df[~pair_df.index.isin(resolved_pair_df.index)]

        def _compose_label(obj1_ids: List[int], obj2_ids: List[int]) -> str:
            def _format(ids: Iterable[int]) -> str:
                return (
                    "(" + "+".join(map(str, sorted(ids))) + ")"
                    if len(ids) != 1
                    else f"{ids[0]}"
                )

            def _key_order(ids: List[int]) -> Tuple[int, int]:
                return (len(ids), min(ids))

            left, right = sorted([list(obj1_ids), list(obj2_ids)], key=_key_order)
            fmt_left, fmt_right = _format(left), _format(right)
            return f"{fmt_left}+{fmt_right}"

        def _merge_attr(
            star1_attr: Dict[str, float], star2_attr: Dict[str, float]
        ) -> Dict[str, float]:
            # luminosity
            log_L_sol = calc_log_total_luminosity(
                log_L1_sol=star1_attr["log_L_sol"],
                log_L2_sol=star2_attr["log_L_sol"],
            )
            # equivalent radius
            log_eq_R_sol = np.log10(
                calc_equivalent_radius(
                    np.power(10, star1_attr["log_R_sol"]),
                    np.power(10, star2_attr["log_R_sol"]),
                )
            )
            # effective temperature
            log_T_eff = calc_log_effective_temperature(
                log_L_sol=log_L_sol, log_R_sol=log_eq_R_sol
            )
            T_eff = np.power(10, log_T_eff)
            # surface flux
            log_F_sol = calc_log_surface_flux(log_T_eff=log_T_eff)
            # total mass
            total_mass = calc_total_mass(star1_attr["mass"], star2_attr["mass"])

            pos = calc_photocentric(
                L1_sol=np.power(10, star1_attr["log_L_sol"]),
                L2_sol=np.power(10, star2_attr["log_L_sol"]),
                vec1=[star1_attr["x"], star1_attr["y"], star1_attr["z"]],
                vec2=[star2_attr["x"], star2_attr["y"], star2_attr["z"]],
            )
            vel = calc_photocentric(
                L1_sol=np.power(10, star1_attr["log_L_sol"]),
                L2_sol=np.power(10, star2_attr["log_L_sol"]),
                vec1=[star1_attr["vx"], star1_attr["vy"], star1_attr["vz"]],
                vec2=[star2_attr["vx"], star2_attr["vy"], star2_attr["vz"]],
            )

            r_dc = np.linalg.norm(pos - header_dict["cluster_density_center"])
            r_dc_r_tidal = r_dc / header_dict["r_tidal"]

            return {
                "mass": total_mass,
                **dict(zip(["x", "y", "z"], pos)),
                **dict(zip(["vx", "vy", "vz"], vel)),
                "r_dc": r_dc,
                "r_dc_r_tidal": r_dc_r_tidal,
                "T_eff": T_eff,
                "log_T_eff": log_T_eff,
                "log_L_sol": log_L_sol,
                "log_R_sol": log_eq_R_sol,
                "log_F_sol": log_F_sol,
                "is_binary": True,
                "is_unresolved_binary": True,
            }

        memo: Dict[Tuple[int, ...], Dict[str, float]] = {}

        def _resolve_group(ids: List[int]) -> Dict[str, float]:
            sorted_ids = tuple(sorted(ids))
            if sorted_ids in memo:
                return memo[sorted_ids]
            if len(sorted_ids) == 1:
                record = name_attr_map[sorted_ids[0]]
            elif len(sorted_ids) == 2:
                record = _merge_attr(
                    star1_attr=_resolve_group([sorted_ids[0]]),
                    star2_attr=_resolve_group([sorted_ids[1]]),
                )
            else:
                raise NotImplementedError(
                    f"Merging more than 2 stars ({sorted_ids}) is not implemented yet."
                )
            memo[sorted_ids] = record
            return record

        merged_unresolved_binary_df = (
            pd.DataFrame(
                {
                    _compose_label(pair["obj1_ids"], pair["obj2_ids"]): _merge_attr(
                        star1_attr=_resolve_group(pair["obj1_ids"]),
                        star2_attr=_resolve_group(pair["obj2_ids"]),
                    )
                    for _, pair in unresolved_pair_df.iterrows()
                }
            )
            .T.reset_index()
            .rename(columns={"index": "name"})
        )

        return (
            pd.concat(
                [
                    single_star_df,
                    resolved_binary_df,
                    merged_unresolved_binary_df,
                ],
                ignore_index=True,
            )
            .sort_values(
                by="name",
                key=lambda s: s.map(
                    lambda x: (0, int(x))
                    if isinstance(x, int)
                    else (1, int(re.search(r"\d+", str(x)).group()))
                ),
            )
            .reset_index(drop=True)
        )
