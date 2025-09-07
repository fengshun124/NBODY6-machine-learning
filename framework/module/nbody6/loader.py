import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import joblib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from module.nbody6.calc.binary import calc_semi_major_axis
from module.nbody6.calc.property import calc_log_surface_flux
from module.nbody6.file.base import NBody6FileParserBase, NBody6FileSnapshot
from module.nbody6.file.fort19 import NBody6Fort19
from module.nbody6.file.fort82 import NBody6Fort82
from module.nbody6.file.fort83 import NBody6Fort83
from module.nbody6.file.misc import NBody6DensityCenterFile
from module.nbody6.file.out9 import NBody6OUT9
from module.nbody6.file.out34 import NBody6OUT34


def _emit_exception(
    exception: Type[BaseException],
    message: str,
    note: Optional[str] = None,
    is_strict: bool = True,
    warn_category: Type[Warning] = UserWarning,
    stacklevel: int = 2,
) -> None:
    if is_strict:
        err = exception(message)
        if note:
            if hasattr(err, "add_note"):
                err.add_note(note)
            else:
                try:
                    raise Exception(note)
                except Exception as detail:
                    raise err from detail
        raise err

    warnings.warn(f"{message} {note}", category=warn_category, stacklevel=stacklevel)


@dataclass
class NBody6Snapshot:
    header: Dict[str, Any]
    data: pd.DataFrame
    binary_pair: pd.DataFrame
    raw_snapshot_dict: Dict[str, NBody6FileSnapshot]

    def __repr__(self):
        return (
            f"{type(self).__name__}({self.header['time']:.2f} Myr, n={len(self.data)})"
        )


class NBody6OutputLoader:
    def __init__(self, root: Union[str, Path]) -> None:
        self._root = Path(root).resolve()
        self._nbody6_file_dict: Dict[str, NBody6FileParserBase] = {}
        self._init_file_dict()
        self._is_nbody6_file_dict_loaded = False
        self._snapshot_dict: Optional[Dict[float, NBody6Snapshot]] = None
        self._raw_timestamps: Optional[List[float]] = None
        self._timestamps: Optional[List[float]] = None
        self._stat_summary: Optional[pd.DataFrame] = None

    def _init_file_dict(self) -> None:
        self._nbody6_file_dict = {
            "densCentre.txt": NBody6DensityCenterFile(self._root / "densCentre.txt"),
            "OUT34": NBody6OUT34(self._root / "OUT34"),
            "OUT9": NBody6OUT9(self._root / "OUT9"),
            "fort.19": NBody6Fort19(self._root / "fort.19"),
            "fort.82": NBody6Fort82(self._root / "fort.82"),
            "fort.83": NBody6Fort83(self._root / "fort.83"),
        }

    def __repr__(self) -> str:
        repr_str = f"NBody6OutputLoader(root={self._root}"
        if self._raw_timestamps:
            repr_str += (
                f", raw_timestamps: {len(self._raw_timestamps)} "
                f"({min(self._raw_timestamps):.2f} Myr - {max(self._raw_timestamps):.2f} Myr)"
            )
        if self._timestamps:
            repr_str += (
                f", timestamps: {len(self._timestamps)} "
                f"({min(self._timestamps):.2f} Myr - {max(self._timestamps):.2f} Myr)"
            )
        return repr_str + ")"

    @property
    def raw_timestamps(self) -> Optional[List[float]]:
        if not self._raw_timestamps:
            warnings.warn("Timestamps not loaded. Call `load()` first.", UserWarning)
        return self._raw_timestamps

    @property
    def timestamps(self) -> Optional[List[float]]:
        if self._timestamps is None:
            warnings.warn("Snapshots not merged. Call `merge()` first.", UserWarning)
            return None
        return self._timestamps

    @property
    def nbody6_file_dict(self) -> Dict[str, NBody6FileParserBase]:
        if not self._is_nbody6_file_dict_loaded:
            warnings.warn(
                "Parsers not loaded. Call `load()` to read files.", UserWarning
            )
        return self._nbody6_file_dict

    @property
    def summary(self) -> Optional[pd.DataFrame]:
        if self._stat_summary is None:
            warnings.warn(
                "Summary not available. Call `summarize()` after `merge()`.",
                UserWarning,
            )
        return self._stat_summary

    def load(
        self,
        is_strict: bool = True,
        is_allow_trim: bool = False,
        timestamp_tolerance: float = 2e-2,
    ) -> Dict[str, NBody6FileParserBase]:
        # check if files are already loaded. If so, reset them.
        if self._is_nbody6_file_dict_loaded:
            warnings.warn(
                "Reloading files: ALL previous data will be discarded.", UserWarning
            )
            self._init_file_dict()
            self._is_nbody6_file_dict_loaded = False

        if not is_strict:
            warnings.warn("Non-strict load: Data may contain NaN values.", UserWarning)

        if is_allow_trim:
            warnings.warn(
                "Trim enabled: Loaded data may NOT cover full time range.", UserWarning
            )

        for filename, nbody6_file in self._nbody6_file_dict.items():
            try:
                nbody6_file.load(is_strict=is_strict)
            except Exception as e:
                raise ValueError(f"Error loading {filename}: {e}") from e

        # check if all files share the same timestamps with tolerance of 1e-2
        timestamp_dict = {
            name: nbody6_file.timestamps
            for name, nbody6_file in self._nbody6_file_dict.items()
        }

        timestamp_df = pd.DataFrame.from_dict(timestamp_dict, orient="index").T
        # check if all files have the same number of timestamps
        if timestamp_df.notnull().sum(axis=0).nunique() > 1:
            msg = (
                "Expected ALL files to have the same number of timestamps, "
                f"but got {timestamp_df.notnull().sum(axis=0).to_dict()}."
            )
            if not is_allow_trim:
                raise ValueError(f"Error aligning {self._root}: {msg}")
            else:
                warnings.warn(
                    f"Consistency check warning for {self._root}: {msg} "
                    "Trimming timestamps to those shared by ALL files.",
                    UserWarning,
                )

        # check if timestamps match within tolerance
        ref_timestamp_file = "OUT34"
        if not is_allow_trim:
            mismatched_timestamp_df = timestamp_df[
                timestamp_df.max(axis=1) - timestamp_df.min(axis=1)
                > timestamp_tolerance
            ]
            mismatched_indices = mismatched_timestamp_df.index.tolist()
            if mismatched_indices:
                _emit_exception(
                    ValueError,
                    f"Error aligning {self._root}: "
                    f"Timestamp mismatch at index/indices {mismatched_indices}",
                    note=(
                        f"\n{mismatched_timestamp_df}\n"
                        "Use `is_allow_trim=True` to keep only common timestamps."
                    ),
                )

            # unify timestamps to OUT34
            ref_timestamp_indices = timestamp_df.index.to_list()
            ref_timestamps = timestamp_df.loc[
                ref_timestamp_indices, ref_timestamp_file
            ].values.tolist()

        else:
            trimmed_timestamp_df = timestamp_df.dropna()
            trimmed_timestamp_df = trimmed_timestamp_df[
                (trimmed_timestamp_df.max(axis=1) - trimmed_timestamp_df.min(axis=1))
                <= timestamp_tolerance
            ]
            # stop if no common timestamps found
            if trimmed_timestamp_df.empty:
                timestamp_stats = {
                    name: (
                        lambda arr: {
                            "min": round(float(np.nanmin(arr)), 3),
                            "max": round(float(np.nanmax(arr)), 3),
                            "step": round(
                                float(
                                    np.nanmean(np.diff(pd.Series(arr).dropna().values))
                                ),
                                4,
                            )
                            if np.sum(~np.isnan(arr)) > 1
                            else None,
                            "count": int(np.sum(~np.isnan(arr))),
                        }
                    )(timestamp_df[name].values.astype(float))
                    for name in timestamp_df.columns
                }
                raise ValueError(
                    f"Error aligning {self._root}: No common timestamps found "
                    f"within tolerance {timestamp_tolerance} Myr, "
                    "which may indicate corrupted or incomplete simulation output. "
                    f"Timestamps stats: {timestamp_stats}"
                )

            # unify timestamps to OUT34 in the trimmed set
            ref_timestamp_indices = trimmed_timestamp_df.index.to_list()
            ref_timestamps = trimmed_timestamp_df.loc[
                ref_timestamp_indices, ref_timestamp_file
            ].values.tolist()

        # check if reference timestamps start from 0
        if ref_timestamps[0] != 0:
            warnings.warn(
                f"Reference timestamps start from {ref_timestamps[0]:.2f} Myr instead of 0 Myr. "
                "Possibly corrupted or incomplete simulation output.",
                UserWarning,
            )

        for name, nbody6_file in self._nbody6_file_dict.items():
            if name == ref_timestamp_file:
                continue
            diff_indices = (
                timestamp_df.loc[ref_timestamp_indices, name] != ref_timestamps
            )
            if diff_indices.any():
                for ts, ref_ts in zip(
                    timestamp_df.loc[ref_timestamp_indices, name][diff_indices],
                    np.array(ref_timestamps)[diff_indices],
                ):
                    nbody6_file.update_timestamp(ts, ref_ts)

        self._raw_timestamps = ref_timestamps
        self._is_nbody6_file_dict_loaded = True
        return self._nbody6_file_dict

    def get_snapshot(self, timestamp: float) -> Optional[NBody6Snapshot]:
        if not self._is_nbody6_file_dict_loaded:
            warnings.warn("Parsers not loaded. Call `load()` first.")
            return None
        if self._snapshot_dict is None:
            warnings.warn("Snapshots not merged. Call `merge()` first.")
            return None
        if timestamp in self._snapshot_dict:
            return self._snapshot_dict.get(timestamp)
        else:
            merged_timestamps = self.timestamps
            if not merged_timestamps:
                warnings.warn(
                    "No merged snapshots available to select a closest timestamp."
                )
                return None
            new_timestamp = merged_timestamps[
                np.argmin(np.abs(np.array(merged_timestamps) - timestamp))
            ]
            warnings.warn(
                f"Timestamp {timestamp} not found. "
                f"Returning closest snapshot at {new_timestamp:.2f} Myr.",
                UserWarning,
            )
            return self._snapshot_dict.get(new_timestamp)

    @property
    def snapshot_dict(self) -> Optional[Dict[float, NBody6Snapshot]]:
        if self._snapshot_dict is None:
            warnings.warn("Snapshots not merged. Call `merge()` first.", UserWarning)
            return None
        if not self._is_nbody6_file_dict_loaded:
            warnings.warn("Parsers not loaded. Call `load()` first.", UserWarning)
        return self._snapshot_dict

    @property
    def snapshot_list(self) -> Optional[List[Tuple[float, NBody6Snapshot]]]:
        return list(self.snapshot_dict.items()) if self.snapshot_dict else None

    def merge(
        self, is_strict: bool = True, is_verbose: bool = True
    ) -> Dict[float, NBody6Snapshot]:
        self._snapshot_dict = {}
        if not self._raw_timestamps:
            warnings.warn("Raw data not loaded. Call load() first.", UserWarning)
            return self._snapshot_dict
        if self._timestamps:
            warnings.warn(
                "Remerging snapshots: ALL previous merged-snapshots will be discarded.",
                UserWarning,
            )
            self._timestamps = None

        for timestamp in (
            ts_pbar := tqdm(
                self._raw_timestamps,
                position=0,
                leave=False,
                disable=not is_verbose,
                dynamic_ncols=True,
            )
        ):
            ts_pbar.set_description(f"Merging Snapshot@{timestamp:.2f} Myr")
            snapshot = self._merge(timestamp, is_strict=is_strict)
            if snapshot is None:
                break
            self._snapshot_dict[timestamp] = snapshot

        if not self._snapshot_dict:
            warnings.warn(
                "No valid snapshots found. Verify input files and timestamps.",
                UserWarning,
            )

        self._timestamps = sorted(self._snapshot_dict.keys())
        return self._snapshot_dict

    def _merge(self, timestamp: float, is_strict: bool) -> Optional[NBody6Snapshot]:
        # key definitions
        _POS_VEL_KEYS = ["x", "y", "z", "vx", "vy", "vz"]
        _DENSITY_CENTER_DIST_KEYS = ["r_dc", "r_dc_r_tidal"]
        _ATTR_KEYS = ["mass", "zlum", "rad", "tempe"]
        _BINARY_ATTR_KEYS = ["ecc", "semi", "log_period_days"]
        _BINARY_PAIR_KEYS = [
            "obj1_ids",
            "obj2_ids",
            "obj1_mass",
            "obj2_mass",
            "obj1_total_mass",
            "obj2_total_mass",
            "obj1_r_dc",
            "obj2_r_dc",
            "obj1_r_dc_r_tidal",
            "obj2_r_dc_r_tidal",
            *_BINARY_ATTR_KEYS,
        ]

        def _build_header_dict(
            density_info_snap: NBody6FileSnapshot,
            o34_snap: NBody6FileSnapshot,
        ) -> Dict[str, Any]:
            return {
                "time": o34_snap.header["time"],
                "r_tidal": density_info_snap.header["r_tidal"],
                "cluster_density_center": (
                    density_info_snap.header["density_center_x"],
                    density_info_snap.header["density_center_y"],
                    density_info_snap.header["density_center_z"],
                ),
                "r_tidal_out34": o34_snap.header["rtide"],
                "cluster_density_center_out34": tuple(
                    map(float, o34_snap.header["rd"])
                ),
                "cluster_mass_center_out34": tuple(map(float, o34_snap.header["rcm"])),
                "cluster_gal_position_out34": (
                    o34_snap.header["rg"] * o34_snap.header["rbar"]
                ),
                "cluster_gal_velocity_out34": (
                    o34_snap.header["vg"] * o34_snap.header["vstar"]
                ),
            }

        def _build_pos_vel_df(
            o34_snap: NBody6FileSnapshot,
            o9_snap: NBody6FileSnapshot,
            density_info_snap: NBody6FileSnapshot,
        ) -> Tuple[pd.DataFrame, Dict[int, List[int]]]:
            atomic_pos_vel_df = o34_snap.data.copy()[["name"] + _POS_VEL_KEYS]
            # distance to `densCentre.txt`-obtained density center in parsecs
            atomic_pos_vel_df["r_dc"] = np.linalg.norm(
                atomic_pos_vel_df[["x", "y", "z"]].values
                - np.array(
                    [
                        density_info_snap.header["density_center_x"],
                        density_info_snap.header["density_center_y"],
                        density_info_snap.header["density_center_z"],
                    ]
                ),
                axis=1,
            )
            # normalized distance to tidal radius
            atomic_pos_vel_df["r_dc_r_tidal"] = (
                atomic_pos_vel_df["r_dc"] / density_info_snap.header["r_tidal"]
            )

            # construct regularized binary map from OUT9
            reg_bin_map = (
                o9_snap.data.copy()
                .melt(
                    id_vars=["cmName"], value_vars=["name1", "name2"], value_name="name"
                )
                .drop(columns="variable")
                .groupby("cmName")["name"]
                .apply(list)
                .to_dict()
            )

            # extend pos / vel information to full table
            pos_vel_df = pd.DataFrame(
                [
                    row
                    for rows in atomic_pos_vel_df.copy().apply(
                        lambda r: (
                            [
                                (
                                    r["name"],
                                    *r[_POS_VEL_KEYS + _DENSITY_CENTER_DIST_KEYS],
                                )
                            ]
                            if (r["name"] not in reg_bin_map)
                            else [
                                (
                                    n,
                                    *r[_POS_VEL_KEYS + _DENSITY_CENTER_DIST_KEYS],
                                )
                                for n in reg_bin_map[r["name"]]
                            ]
                        ),
                        axis=1,
                    )
                    for row in rows
                ],
                columns=["name"] + _POS_VEL_KEYS + _DENSITY_CENTER_DIST_KEYS,
            ).astype({"name": int})

            return pos_vel_df, reg_bin_map

        def _build_attr_df(
            f82_snap: NBody6FileSnapshot,
            f83_snap: NBody6FileSnapshot,
        ) -> Tuple[pd.DataFrame, Dict[int, float]]:
            reg_bin_attr_df = pd.concat(
                [
                    f82_snap.data.copy().rename(
                        columns={f"{attr}{i}": attr for attr in ["name"] + _ATTR_KEYS}
                    )[["name"] + _ATTR_KEYS]
                    for i in (1, 2)
                ],
                ignore_index=True,
            ).drop_duplicates(subset=["name"])

            attr_df = pd.concat(
                [
                    reg_bin_attr_df,
                    f83_snap.data.copy()[["name"] + _ATTR_KEYS],
                ]
            )

            if attr_df["name"].duplicated().any():
                duplicate_names = attr_df[attr_df["name"].duplicated()]["name"].unique()
                warnings.warn(
                    f"Duplicate names found in attributes at {timestamp} Myr: {duplicate_names}. "
                    "Keeping the first occurrence.",
                    UserWarning,
                )
                attr_df = attr_df.drop_duplicates(subset=["name"], keep="first")

            mass_map = attr_df.set_index("name")["mass"].to_dict()

            return attr_df.reset_index(drop=True), mass_map

        def _build_binary_pair_df(
            o9_snapshot: NBody6FileSnapshot,
            f19_snapshot: NBody6FileSnapshot,
            reg_bin_map: Dict[int, List[int]],
            mass_map: Dict[int, float],
            r_dc_map: Dict[int, Dict[str, float]],
        ) -> pd.DataFrame:
            reg_bin_pair_df = (
                (
                    o9_snapshot.data.copy()
                    .assign(
                        semi=lambda df: df.apply(
                            lambda r: calc_semi_major_axis(
                                m1_sol=r["mass1"],
                                m2_sol=r["mass2"],
                                period_days=np.power(10, r["p"]),
                            ),
                            axis=1,
                        ),
                    )
                    .rename(columns={"cmName": "pair", "p": "log_period_days"})
                )
                if not o9_snapshot.data.empty
                else pd.DataFrame(
                    columns=["name1", "name2", "mass1", "mass2", "pair"]
                    + self._BINARY_ATTR_KEYS
                )
            )

            non_reg_bin_pair_df = (
                f19_snapshot.data.copy()[
                    ["name1", "name2", "mass1", "mass2", "ecc", "p"]
                ]
                .assign(
                    pair=lambda df: df.apply(
                        lambda r: f"b-{int(r['name1'])}-{int(r['name2'])}"
                        if r["mass1"] >= r["mass2"]
                        else f"b-{int(r['name2'])}-{int(r['name1'])}",
                        axis=1,
                    ),
                    semi=lambda df: df.apply(
                        lambda r: calc_semi_major_axis(
                            m1_sol=r["mass1"],
                            m2_sol=r["mass2"],
                            period_days=np.power(10, r["p"]),
                        ),
                        axis=1,
                    ),
                )
                .rename(columns={"p": "log_period_days"})
                if not f19_snapshot.data.empty
                else pd.DataFrame(
                    columns=["name1", "name2", "mass1", "mass2", "pair"]
                    + self._BINARY_ATTR_KEYS
                )
            )

            pair_dfs = [
                pair_df
                for pair_df in [reg_bin_pair_df, non_reg_bin_pair_df]
                if not pair_df.empty
            ]

            if not pair_dfs:
                pair_df = pd.DataFrame(columns=["name1"] + _BINARY_PAIR_KEYS)
            else:
                pair_df = pd.concat(pair_dfs, ignore_index=True)
                # warn if duplicate pairs found
                if pair_df["pair"].duplicated().any():
                    duplicate_pairs = pair_df[pair_df["pair"].duplicated()][
                        "pair"
                    ].unique()
                    warnings.warn(
                        f"Duplicate pairs found in binary pairs at {timestamp} Myr: {duplicate_pairs}. "
                        "Keeping the first occurrence.",
                        UserWarning,
                    )
                    pair_df = pair_df.drop_duplicates(subset=["pair"], keep="first")

                pair_df = (
                    pair_df.assign(
                        obj1_ids=pair_df["name1"].map(
                            lambda name: reg_bin_map.get(name, [name]),
                        ),
                        obj2_ids=pair_df["name2"].map(
                            lambda name: reg_bin_map.get(name, [name]),
                        ),
                    )
                    .assign(
                        obj1_mass=lambda df: df["obj1_ids"].map(
                            lambda ids: [mass_map.get(i) for i in ids]
                            if set(ids).issubset(mass_map)
                            else None
                        ),
                        obj2_mass=lambda df: df["obj2_ids"].map(
                            lambda ids: [mass_map.get(i) for i in ids]
                            if set(ids).issubset(mass_map)
                            else None
                        ),
                    )
                    .assign(
                        obj1_total_mass=lambda df: df["obj1_mass"].map(np.sum),
                        obj2_total_mass=lambda df: df["obj2_mass"].map(np.sum),
                        obj1_r_dc=lambda df: df["name1"].map(
                            lambda n: r_dc_map.get(n, {}).get("r_dc", np.nan)
                        ),
                        obj2_r_dc=lambda df: df["name2"].map(
                            lambda n: r_dc_map.get(n, {}).get("r_dc", np.nan)
                        ),
                        obj1_r_dc_r_tidal=lambda df: df["name1"].map(
                            lambda n: r_dc_map.get(n, {}).get("r_dc_r_tidal", np.nan)
                        ),
                        obj2_r_dc_r_tidal=lambda df: df["name2"].map(
                            lambda n: r_dc_map.get(n, {}).get("r_dc_r_tidal", np.nan)
                        ),
                    )
                )

            return pair_df.reset_index(drop=True)

        # skip if cluster has dissolved
        density_info_snapshot = self._nbody6_file_dict["densCentre.txt"][timestamp]
        if density_info_snapshot.header["r_tidal"] < 0:
            warnings.warn(
                f"Cluster dissolved at {timestamp} Myr. Stopping merge.", UserWarning
            )
            return None

        # extract rest of the snapshots of corresponding timestamp
        out34_snapshot = self._nbody6_file_dict["OUT34"][timestamp]
        out9_snapshot = self._nbody6_file_dict["OUT9"][timestamp]
        fort19_snapshot = self._nbody6_file_dict["fort.19"][timestamp]
        fort82_snapshot = self._nbody6_file_dict["fort.82"][timestamp]
        fort83_snapshot = self._nbody6_file_dict["fort.83"][timestamp]

        # collect header information
        header_dict = _build_header_dict(
            o34_snap=out34_snapshot, density_info_snap=density_info_snapshot
        )

        # collect position and velocity information
        position_velocity_df, regularized_binary_map = _build_pos_vel_df(
            o34_snap=out34_snapshot,
            o9_snap=out9_snapshot,
            density_info_snap=density_info_snapshot,
        )

        # collect attribute information
        attribute_df, star_mass_map = _build_attr_df(
            f82_snap=fort82_snapshot,
            f83_snap=fort83_snapshot,
        )

        main_data_df = pd.merge(
            attribute_df, position_velocity_df, on="name", how="inner"
        )
        # check if any names in pos_vel_df and attr_df are missing after merge
        missing_pos_vel_names = set(position_velocity_df["name"]) - set(
            main_data_df["name"]
        )
        if missing_pos_vel_names:
            _emit_exception(
                exception=ValueError,
                warn_category=UserWarning,
                message=f"Name(s) {missing_pos_vel_names} present in OUT34/OUT9 "
                f"but missing in fort.82/fort.83 at {timestamp} Myr.",
                note="Set `is_strict=False` to ignore this check."
                if is_strict
                else "Dropping affected rows.",
                is_strict=is_strict,
            )

        missing_attr_names = set(attribute_df["name"]) - set(main_data_df["name"])
        if missing_attr_names:
            _emit_exception(
                exception=ValueError,
                warn_category=UserWarning,
                message=f"Name(s) {missing_attr_names} present in fort.82/fort.83 "
                f"but missing in OUT34/OUT9 at {timestamp} Myr.",
                note="Set `is_strict=False` to ignore this check."
                if is_strict
                else "Dropping affected rows.",
                is_strict=is_strict,
            )

        # convert T_eff, L, R, F to log scale
        main_data_df.rename(
            columns={
                # effective temperature in Kelvin
                "tempe": "log_T_eff",
                # luminosity in L_sun
                "zlum": "log_L_sol",
                # radius in R_sun
                "rad": "log_R_sol",
            },
            inplace=True,
        )
        main_data_df["T_eff"] = np.power(10, main_data_df["log_T_eff"])
        main_data_df["L_sol"] = np.power(10, main_data_df["log_L_sol"])
        main_data_df["R_sol"] = np.power(10, main_data_df["log_R_sol"])
        # flux in solar flux units
        main_data_df["log_F_sol"] = calc_log_surface_flux(main_data_df["log_T_eff"])

        binary_pair_df = _build_binary_pair_df(
            out9_snapshot,
            fort19_snapshot,
            regularized_binary_map,
            star_mass_map,
            main_data_df.set_index("name")[_DENSITY_CENTER_DIST_KEYS].to_dict(
                orient="index"
            ),
        )

        main_data_df["is_binary"] = main_data_df["name"].isin(
            set(binary_pair_df[["name1", "name2"]].values.flatten())
        )

        return NBody6Snapshot(
            header=header_dict,
            data=main_data_df.sort_values(by="name").reset_index(drop=True)[
                [
                    "name",
                    *_POS_VEL_KEYS,
                    "mass",
                    *_DENSITY_CENTER_DIST_KEYS,
                    "T_eff",
                    "log_T_eff",
                    "log_L_sol",
                    "log_R_sol",
                    "log_F_sol",
                    "is_binary",
                ]
            ],
            binary_pair=binary_pair_df.sort_values(by="name1").reset_index(drop=True)[
                _BINARY_PAIR_KEYS
            ],
            raw_snapshot_dict={
                "densCentre.txt": density_info_snapshot,
                "OUT34": out34_snapshot,
                "OUT9": out9_snapshot,
                "fort.82": fort82_snapshot,
                "fort.83": fort83_snapshot,
                "fort.19": fort19_snapshot,
            },
        )

    def export_snapshot_dict(
        self, output_path: Union[str, Path], overwrite: bool = False
    ) -> None:
        if self._snapshot_dict is None:
            raise ValueError("Snapshots not merged. Call `merge()` first.")
        output_path = Path(output_path).resolve()
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"File `{output_path}` already exists. Use `overwrite=True` to enforce overwrite."
            )
        joblib.dump(self._snapshot_dict, output_path)
        print(f"Snapshot dictionary exported to `{output_path}`.")

    def summarize(self, is_verbose: bool = False) -> Optional[pd.DataFrame]:
        if self._snapshot_dict is None:
            warnings.warn("Snapshots not merged. Call `merge()` first.", UserWarning)
            return None
        if self._stat_summary is not None:
            warnings.warn(
                "Re-summarizing statistics: ALL previous summaries will be discarded.",
                UserWarning,
            )
            self._stat_summary = None

        def _summarize_descriptive(
            data_df: pd.DataFrame, key: str, prefix: Optional[str] = None
        ) -> Dict[str, Any]:
            prefix = f"{prefix}_" if prefix else ""
            return {
                f"{prefix}{key}_mean": data_df[key].mean(),
                f"{prefix}{key}_std": data_df[key].std(),
                f"{prefix}{key}_min": data_df[key].min(),
                f"{prefix}{key}_q1": data_df[key].quantile(0.25),
                f"{prefix}{key}_median": data_df[key].median(),
                f"{prefix}{key}_q3": data_df[key].quantile(0.75),
                f"{prefix}{key}_max": data_df[key].max(),
            }

        def _summarize(
            main_df: pd.DataFrame,
            binary_pair_df: pd.DataFrame,
        ) -> Dict[str, Any]:
            stats = {
                "n_star": len(main_df),
                "n_binaries": int(main_df["is_binary"].sum()),
                "n_binary_systems": len(binary_pair_df),
                **_summarize_descriptive(main_df, "mass"),
                "total_mass": main_df["mass"].sum(),
            }
            if not binary_pair_df.empty:
                stats.update(**_summarize_descriptive(binary_pair_df, "ecc"))
                stats.update(**_summarize_descriptive(binary_pair_df, "semi"))
                stats.update(
                    **_summarize_descriptive(binary_pair_df, "log_period_days")
                )
            return stats

        if self._snapshot_dict is None:
            warnings.warn("Snapshots not merged. Call `merge()` first.", UserWarning)
            return None

        simulation_stats_dicts = []
        for timestamp, snapshot in (
            pbar := tqdm(
                self._snapshot_dict.items(), leave=False, disable=not is_verbose
            )
        ):
            if is_verbose:
                pbar.set_description(f"Processing Snapshot@{timestamp:.2f} Myr")

            header = snapshot.header
            snapshot_df = snapshot.data
            binary_pair_df = snapshot.binary_pair

            simulation_stats_dicts.extend(
                [
                    {
                        "time": header["time"],
                        "r_tidal": header["r_tidal"],
                        "cluster_density_center": header["cluster_density_center"],
                        # overall
                        **_summarize(snapshot_df, binary_pair_df),
                        # within tidal radius
                        **{
                            f"within_r_tidal_{k}": v
                            for k, v in _summarize(
                                snapshot_df[snapshot_df["r_dc_r_tidal"] <= 1],
                                binary_pair_df[
                                    (binary_pair_df["obj1_r_dc_r_tidal"] <= 1)
                                    & (binary_pair_df["obj2_r_dc_r_tidal"] <= 1)
                                ],
                            ).items()
                        },
                        # within 2 times tidal radius
                        **{
                            f"within_2x_r_tidal_{k}": v
                            for k, v in _summarize(
                                snapshot_df[snapshot_df["r_dc_r_tidal"] <= 2],
                                binary_pair_df[
                                    (binary_pair_df["obj1_r_dc_r_tidal"] <= 2)
                                    & (binary_pair_df["obj2_r_dc_r_tidal"] <= 2)
                                ],
                            ).items()
                        },
                    }
                ]
            )

        self._stat_summary = (
            pd.DataFrame(simulation_stats_dicts)
            .sort_values(by="time")
            .reset_index(drop=True)
        )

        return self._stat_summary
