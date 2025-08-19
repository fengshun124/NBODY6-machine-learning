import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm.auto import tqdm

from module.nbody6.calc.binary import calc_semi_major_axis
from module.nbody6.file.base import NBody6FileParserBase, NBody6FileSnapshot
from module.nbody6.file.fort19 import NBody6Fort19
from module.nbody6.file.fort82 import NBody6Fort82
from module.nbody6.file.fort83 import NBody6Fort83
from module.nbody6.file.misc import NBody6DensityCenterFile
from module.nbody6.file.out9 import NBody6OUT9
from module.nbody6.file.out34 import NBody6OUT34
from module.nbody6.plot.animate import animate_nbody6_snapshots


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
        self._file_dict: Dict[str, NBody6FileParserBase] = {}
        self._init_file_dict()
        self._is_file_dict_loaded = False
        self._snapshot_dict: Optional[Dict[float, NBody6Snapshot]] = None
        self._raw_timestamps: Optional[List[float]] = None
        self._timestamps: Optional[List[float]] = None
        self._stat_summary: Optional[pd.DataFrame] = None

    def _init_file_dict(self) -> None:
        self._file_dict = {
            "densCentre.txt": NBody6DensityCenterFile(self._root / "densCentre.txt"),
            "OUT34": NBody6OUT34(self._root / "OUT34"),
            "OUT9": NBody6OUT9(self._root / "OUT9"),
            "fort.19": NBody6Fort19(self._root / "fort.19"),
            "fort.82": NBody6Fort82(self._root / "fort.82"),
            "fort.83": NBody6Fort83(self._root / "fort.83"),
        }

    def __repr__(self):
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
            warnings.warn("Timestamps are not loaded. Call load() first.")
        return self._raw_timestamps

    @property
    def timestamps(self) -> Optional[List[float]]:
        if self._timestamps is None:
            warnings.warn("Snapshot dictionary is not initialized. Call merge() first.")
            return None
        return self._timestamps

    @property
    def file_dict(self) -> Dict[str, NBody6FileParserBase]:
        if not self._is_file_dict_loaded:
            warnings.warn(
                "Parsers are created but not loaded yet. Call load() to read files."
            )
        return self._file_dict

    @property
    def summary(self) -> Optional[pd.DataFrame]:
        if not self._stat_summary:
            warnings.warn(
                "Statistics summary is not available. Call summarize() first."
            )
        return self._stat_summary

    def load(self, is_strict: bool = True) -> Dict[str, NBody6FileParserBase]:
        # check if files are already loaded. If so, reset them.
        if self._is_file_dict_loaded:
            warnings.warn(
                "Files are already loaded. Existing data will be cleared before reloading."
            )
            self._init_file_dict()

        if not is_strict:
            warnings.warn(
                "Loading files in non-strict mode. `NaN` values may be present in the data."
            )

        for filename, nbody6_file in self._file_dict.items():
            try:
                nbody6_file.load(is_strict=is_strict)
            except Exception as e:
                raise ValueError(f"Error loading {filename}: {e}") from e

        # check if all files share the same timestamps with tolerance of 1e-2
        timestamp_dict = {
            name: nbody6_file.timestamps
            for name, nbody6_file in self._file_dict.items()
        }

        # check if all files have the same number of timestamps
        timestamp_len_dict = {name: len(ts) for name, ts in timestamp_dict.items()}
        if len(set(timestamp_len_dict.values())) > 1:
            raise ValueError(
                f"Error merging {self._root}: "
                f"Timestamp lengths mismatch: {timestamp_len_dict}. "
                "All files must have the same number of timestamps."
            )

        for i, timestamps in enumerate(zip(*timestamp_dict.values())):
            if max(timestamps) - min(timestamps) > 2e-2:
                mismatched_timestamps = {
                    name: ts[i] for name, ts in timestamp_dict.items()
                }
                raise ValueError(
                    f"Error merging {self._root}: "
                    f"Timestamp mismatch at index {i}: {mismatched_timestamps}"
                )

        # set standard timestamps to OUT34
        ref_timestamps = list(timestamp_dict["OUT34"])
        for name, nbody6_file in self._file_dict.items():
            if name == "OUT34":
                continue
            for ts, ref_ts in zip(nbody6_file.timestamps, ref_timestamps):
                if ts != ref_ts:
                    nbody6_file.update_timestamp(ts, ref_ts)

        self._raw_timestamps = ref_timestamps
        self._is_file_dict_loaded = True
        return self._file_dict

    def get_snapshot(self, timestamp: float) -> Optional[NBody6Snapshot]:
        if not self._is_file_dict_loaded:
            warnings.warn("File dictionary is not loaded. Call load() first.")
            return None
        if self._snapshot_dict is None:
            warnings.warn("Snapshot dictionary is not initialized. Call merge() first.")
            return None
        if timestamp in self._snapshot_dict:
            return self._snapshot_dict.get(timestamp)
        else:
            merged_timestamps = self.timestamps
            if not merged_timestamps:
                warnings.warn(
                    "No merged snapshots available to find the closest timestamp."
                )
                return None
            new_timestamp = merged_timestamps[
                np.argmin(np.abs(np.array(merged_timestamps) - timestamp))
            ]
            warnings.warn(
                f"Timestamp {timestamp} not found in snapshot dictionary. "
                f"Returning snapshot for closest timestamp {new_timestamp}."
            )
            return self._snapshot_dict.get(new_timestamp)

    @property
    def snapshot_dict(self) -> Optional[Dict[float, NBody6Snapshot]]:
        if self._snapshot_dict is None:
            warnings.warn("Snapshot dictionary is not initialized. Call merge() first.")
            return None
        if not self._is_file_dict_loaded:
            warnings.warn("File dictionary is not loaded. Call load() first.")
        return self._snapshot_dict

    @property
    def snapshot_list(self) -> Optional[List[Tuple[float, NBody6Snapshot]]]:
        return list(self.snapshot_dict.items()) if self.snapshot_dict else None

    def merge(
        self, is_strict: bool = True, is_verbose: bool = True
    ) -> Dict[float, NBody6Snapshot]:
        self._snapshot_dict = {}
        if not self._raw_timestamps:
            warnings.warn("Raw data not loaded. Call load() first.")
            return self._snapshot_dict
        if self._timestamps:
            warnings.warn("Snapshot dictionary already initialized. Overwriting.")
            self._timestamps = None

        for timestamp in (
            pbar := tqdm(self._raw_timestamps, leave=False)
            if is_verbose
            else self._raw_timestamps
        ):
            snapshot = self._merge(timestamp)
            if snapshot is None:
                break

            if is_verbose:
                pbar.set_description(f"Merging Snapshot@{timestamp:.2f} Myr")

            # validation
            for attr_name, data_label, id_col in [
                ("data", "Snapshot", "name"),
                ("binary_pair", "Binary Pairing", "pair"),
            ]:
                data_df = getattr(snapshot, attr_name)
                if data_df.isnull().values.any():
                    nan_rows = data_df[data_df.isnull().any(axis=1)]
                    nan_ids = nan_rows[id_col].tolist()
                    msg = (
                        f"{data_label} at {timestamp} contains NaN values. "
                        "Possibly due to misaligned `name` values across files. "
                        f"IDs with NaN values: {nan_ids}. "
                    )
                    if is_strict:
                        raise ValueError(
                            msg + "\nSet `is_strict=False` to ignore this check."
                        )
                    else:
                        warnings.warn(msg + "\nRows with NaN values will be dropped.")
                        setattr(
                            snapshot, attr_name, data_df.dropna().reset_index(drop=True)
                        )
            self._snapshot_dict[timestamp] = snapshot

        if not self._snapshot_dict:
            warnings.warn(
                "No valid snapshots found. Check if the input files are correct."
            )

        self._timestamps = sorted(self._snapshot_dict.keys())
        return self._snapshot_dict

    def _merge(self, timestamp: float) -> Optional[NBody6Snapshot]:
        # extract snapshots with header and data from corresponding timestamp
        out34_snapshot = self._file_dict["OUT34"][timestamp]
        out9_snapshot = self._file_dict["OUT9"][timestamp]
        fort19_snapshot = self._file_dict["fort.19"][timestamp]
        fort82_snapshot = self._file_dict["fort.82"][timestamp]
        fort83_snapshot = self._file_dict["fort.83"][timestamp]
        density_info_snapshot = self._file_dict["densCentre.txt"][timestamp]

        # skip if cluster has dissolved
        if density_info_snapshot.header["r_tidal"] < 0:
            warnings.warn(
                f"Cluster has dissolved at timestamp {timestamp}. Terminating merging."
            )
            return None

        # collect header information
        header_dict = {
            "time": out34_snapshot.header["time"],
            "r_tidal": density_info_snapshot.header["r_tidal"],
            "cluster_density_center": (
                density_info_snapshot.header["density_center_x"],
                density_info_snapshot.header["density_center_y"],
                density_info_snapshot.header["density_center_z"],
            ),
            "r_tidal_out34": out34_snapshot.header["rtide"],
            "cluster_density_center_out34": tuple(
                map(float, out34_snapshot.header["rd"])
            ),
            "cluster_mass_center_out34": tuple(
                map(float, out34_snapshot.header["rcm"])
            ),
            "cluster_gal_position_out34": (
                out34_snapshot.header["rg"] * out34_snapshot.header["rbar"]
            ),
            "cluster_gal_velocity_out34": (
                out34_snapshot.header["vg"] * out34_snapshot.header["vstar"]
            ),
        }

        pos_vel_keys = ["x", "y", "z", "vx", "vy", "vz"]
        attr_keys = ["mass", "zlum", "rad", "tempe"]
        density_center_dist_keys = ["r_dc", "r_dc_r_tidal"]

        # calculate distance to the revised density center from `densCentre.txt`
        # in both parsecs and normalized by tidal radius
        atomic_pos_vel_df = out34_snapshot.data.copy()[["name"] + pos_vel_keys]
        atomic_pos_vel_df["r_dc"] = np.linalg.norm(
            atomic_pos_vel_df[["x", "y", "z"]].values
            - np.array(
                [
                    density_info_snapshot.header["density_center_x"],
                    density_info_snapshot.header["density_center_y"],
                    density_info_snapshot.header["density_center_z"],
                ]
            ),
            axis=1,
        )
        atomic_pos_vel_df["r_dc_r_tidal"] = (
            atomic_pos_vel_df["r_dc"] / density_info_snapshot.header["r_tidal"]
        )

        # collect attributes from fort.82 (regularized bin.) and fort.83 (single + non-reg. bin.)
        attr_df = (
            pd.concat(
                [
                    pd.concat(
                        [
                            fort82_snapshot.data.copy().rename(
                                columns={
                                    f"{attr}{i}": attr for attr in ["name"] + attr_keys
                                }
                            )[["name", *attr_keys]]
                            for i in (1, 2)
                        ],
                        ignore_index=True,
                    ).drop_duplicates(subset=["name"]),
                    fort83_snapshot.data.copy()[["name"] + attr_keys],
                ]
            )
            .drop_duplicates(subset=["name"])
            .reset_index(drop=True)
        )

        # map regularized binaries from OUT9
        reg_bin_map = (
            out9_snapshot.data.copy()
            .melt(id_vars=["cmName"], value_vars=["name1", "name2"], value_name="name")
            .drop(columns="variable")
            .groupby("cmName")["name"]
            .apply(list)
            .to_dict()
        )

        # extend pos / vel for regularized binaries
        pos_vel_df = pd.DataFrame(
            [
                i
                for s in atomic_pos_vel_df.copy().apply(
                    lambda r: (
                        [(r["name"], *r[pos_vel_keys + density_center_dist_keys])]
                        if (r["name"] not in reg_bin_map)
                        else [
                            (n, *r[pos_vel_keys + density_center_dist_keys])
                            for n in reg_bin_map[r["name"]]
                        ]
                    ),
                    axis=1,
                )
                for i in s
            ],
            columns=["name"] + pos_vel_keys + density_center_dist_keys,
        ).astype({"name": int})

        # merge attributes and positions / velocities
        main_df = pd.merge(attr_df, pos_vel_df, on="name", how="left").reset_index(
            drop=True
        )
        # keep only stellar names (no cmNames that > `nzero` in OUT34 header)
        main_df = main_df[main_df["name"] <= out34_snapshot.header["nzero"]]
        # convert T_eff, L, R, F to log scale
        main_df.rename(
            columns={
                # effective temperature in Kelvin
                "tempe": "log_T_eff",
                # lumonosity in L_sun
                "zlum": "log_L_sol",
                # radius in R_sun
                "rad": "log_R_sol",
            },
            inplace=True,
        )
        main_df["T_eff"] = np.power(10, main_df["log_T_eff"])
        main_df["L_sol"] = np.power(10, main_df["log_L_sol"])
        main_df["R_sol"] = np.power(10, main_df["log_R_sol"])
        # flux in L_sun / R_sun^2
        main_df["F_sol"] = main_df["L_sol"] / main_df["R_sol"] ** 2
        main_df["log_F_sol"] = np.log10(main_df["F_sol"])

        # is_binary flag for non-regularized binaries & regularized binaries
        binary_names = set(
            fort82_snapshot.data[["name1", "name2"]].values.flatten()
        ) | set(fort19_snapshot.data[["name1", "name2"]].values.flatten())
        main_df["is_binary"] = main_df["name"].isin(binary_names)

        # construct binary pairs and calculate semi-major axis
        reg_bin_pair_df = (
            out9_snapshot.data.copy()[
                ["name1", "name2", "mass1", "mass2", "ecc", "p", "cmName"]
            ]
            .rename(columns={"cmName": "pair"})
            .assign(
                semi=lambda df: df.apply(
                    lambda r: calc_semi_major_axis(
                        m1=r["mass1"],
                        m2=r["mass2"],
                        period_days=np.power(10, r["p"]),
                    ),
                    axis=1,
                ),
            )
        )
        f19_df = (
            fort19_snapshot.data.copy()[
                ["name1", "name2", "mass1", "mass2", "ecc", "p"]
            ]
            if not fort19_snapshot.data.empty
            else pd.DataFrame(columns=["name1", "name2", "mass1", "mass2", "ecc", "p"])
        )
        non_reg_bin_pair_df = (
            f19_df.assign(
                pair=lambda df: df.apply(
                    lambda r: f"b-{int(r['name1'])}-{int(r['name2'])}"
                    if r["mass1"] >= r["mass2"]
                    else f"b-{int(r['name2'])}-{int(r['name1'])}",
                    axis=1,
                ),
                semi=lambda df: df.apply(
                    lambda r: calc_semi_major_axis(
                        m1=r["mass1"], m2=r["mass2"], period_days=np.power(10, r["p"])
                    ),
                    axis=1,
                ),
            )
            if not f19_df.empty
            else pd.DataFrame(
                columns=["name1", "name2", "mass1", "mass2", "ecc", "p", "pair", "semi"]
            )
        )

        mass_map = attr_df.set_index("name")["mass"].to_dict()

        def _extend_id(n):
            return (
                [int(n)]
                if (
                    int(n) <= out34_snapshot.header["nzero"]
                    and int(n) not in reg_bin_map
                )
                else [
                    int(x)
                    for x in reg_bin_map.get(int(n), [int(n)])
                    if int(x) <= out34_snapshot.header["nzero"]
                ]
            )

        # construct binary pairs DataFrame for both regularized and non-regularized binaries
        pair_info_keys = [
            "obj1_ids",
            "obj2_ids",
            "obj1_total_mass",
            "obj2_total_mass",
            "obj1_mass",
            "obj2_mass",
            "obj1_r_dc",
            "obj2_r_dc",
            "obj1_r_dc_r_tidal",
            "obj2_r_dc_r_tidal",
            "pair",
            "semi",
            "ecc",
            "log_period_days",
        ]

        pair_dfs = [
            pair_df
            for pair_df in [reg_bin_pair_df, non_reg_bin_pair_df]
            if not pair_df.empty
        ]

        if not pair_dfs:
            pair_df = pd.DataFrame(columns=["name1", *pair_info_keys])
        else:
            pair_df = (
                pd.concat(
                    pair_dfs,
                    ignore_index=True,
                )
                .drop_duplicates(subset=["pair"])
                .reset_index(drop=True)
            ).rename(columns={"p": "log_period_days"})
            # drop whose name1 / name2 NOT in main_df or cmName
            pair_df = pair_df[
                pair_df["name1"].isin(main_df["name"])
                & pair_df["name2"].isin(main_df["name"])
            ].copy()

            # map r_dc and r_dc_r_tidal for binary pairs
            r_dc_df = (
                pd.concat(
                    [
                        pos_vel_df[["name", "r_dc", "r_dc_r_tidal"]],
                        atomic_pos_vel_df[["name", "r_dc", "r_dc_r_tidal"]],
                    ]
                )
                .drop_duplicates(subset=["name"])
                .set_index("name")
            )

            # extend obj1_ids and obj2_ids for regularized binaries
            # and calculate information for obj1 and obj2
            pair_df = (
                pair_df.assign(
                    obj1_ids=pair_df["name1"].map(_extend_id),
                    obj2_ids=pair_df["name2"].map(_extend_id),
                )
                .assign(
                    obj1_mass=lambda df: df["obj1_ids"].map(
                        lambda ids: [mass_map.get(i) for i in ids]
                    ),
                    obj2_mass=lambda df: df["obj2_ids"].map(
                        lambda ids: [mass_map.get(i) for i in ids]
                    ),
                )
                .assign(
                    obj1_total_mass=lambda df: df["obj1_mass"].map(sum),
                    obj2_total_mass=lambda df: df["obj2_mass"].map(sum),
                    obj1_r_dc=lambda df: df["name1"].map(
                        lambda n: r_dc_df.loc[int(n), "r_dc"]
                    ),
                    obj2_r_dc=lambda df: df["name2"].map(
                        lambda n: r_dc_df.loc[int(n), "r_dc"]
                    ),
                    obj1_r_dc_r_tidal=lambda df: df["name1"].map(
                        lambda n: r_dc_df.loc[int(n), "r_dc_r_tidal"]
                    ),
                    obj2_r_dc_r_tidal=lambda df: df["name2"].map(
                        lambda n: r_dc_df.loc[int(n), "r_dc_r_tidal"]
                    ),
                )
            )

        return NBody6Snapshot(
            header=header_dict,
            data=main_df.sort_values(by="name").reset_index(drop=True)[
                [
                    "name",
                    "x",
                    "y",
                    "z",
                    "vx",
                    "vy",
                    "vz",
                    "mass",
                    "r_dc",
                    "r_dc_r_tidal",
                    "T_eff",
                    "log_T_eff",
                    "log_L_sol",
                    "log_R_sol",
                    "log_F_sol",
                    "is_binary",
                ]
            ],
            binary_pair=pair_df.sort_values(by="name1").reset_index(drop=True)[
                pair_info_keys
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
            raise ValueError(
                "Snapshot dictionary is not initialized. Call merge() first."
            )
        output_path = Path(output_path).resolve()
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"File `{output_path}` already exists. Use overwrite=True to force overwrite."
            )
        joblib.dump(self._snapshot_dict, output_path)
        print(f"Snapshot dictionary exported to `{output_path}`.")

    def animate_snapshots(
        self,
        output_path: Union[str, Path],
        fig_title_text: Optional[str] = None,
        animation_fps: int = 10,
        animation_dpi: int = 300,
    ) -> FuncAnimation:
        snapshot_list = self.snapshot_list
        if not snapshot_list:
            raise ValueError("No snapshots available to animate.")
        animation = animate_nbody6_snapshots(
            snapshot_list=snapshot_list,
            output_path=output_path,
            fig_title_text=fig_title_text,
            animation_fps=animation_fps,
            animation_dpi=animation_dpi,
        )
        if not output_path:
            plt.show()
        return animation

    def summarize(self, is_verbose: bool = False) -> Optional[pd.DataFrame]:
        if self._snapshot_dict is None:
            warnings.warn("Snapshot dictionary is not initialized. Call merge() first.")
            return None
        if self._stat_summary is not None:
            warnings.warn(
                "Statistics summary already exists. Overwriting the existing summary."
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
            warnings.warn("Snapshot dictionary is not initialized. Call merge() first.")
            return None

        simulation_stats_dicts = []
        for timestamp, snapshot in (
            pbar := tqdm(self._snapshot_dict.items(), leave=False)
            if is_verbose
            else self._snapshot_dict.items()
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
