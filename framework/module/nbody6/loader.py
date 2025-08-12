import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from module.nbody6.file.base import NBody6OutputFile
from module.nbody6.file.fort19 import NBody6Fort19
from module.nbody6.file.fort82 import NBody6Fort82
from module.nbody6.file.fort83 import NBody6Fort83
from module.nbody6.file.misc import NBody6DensityCenterFile
from module.nbody6.file.out9 import NBody6OUT9
from module.nbody6.file.out34 import NBody6OUT34
from module.nbody6.plot.animate import animate_nbody6_snapshots


class NBody6OutputLoader:
    def __init__(self, root: Union[str, Path]) -> None:
        self._root = Path(root).resolve()

        self._file_dict: Dict[str, NBody6OutputFile] = {}
        self._init_file_dict()
        self._is_file_dict_loaded = False

        self._snapshot_dict: Optional[
            Dict[float, Dict[str, Union[Dict[str, any], pd.DataFrame]]]
        ] = None

        self._raw_timestamps: Optional[List[float]] = None
        self._timestamps: Optional[List[float]] = None

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
    def file_dict(self) -> Dict[str, NBody6OutputFile]:
        if not self._file_dict:
            warnings.warn("File dictionary is empty. Call load() first.")
        return self._file_dict

    def load(self, is_strict: bool = True) -> Dict[str, NBody6OutputFile]:
        # check if files are already loaded. If so, reset them.
        if self._is_file_dict_loaded:
            warnings.warn(
                "Files are already loaded. Existing data will be cleared before reloading."
            )
            self._init_file_dict()

        if not is_strict:
            warnings.warn(
                "Loading files in non-strict mode. `NaN` values may presented in the data."
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

    def get_snapshot(
        self, timestamp: float
    ) -> Optional[Dict[str, Union[Dict[str, any], pd.DataFrame]]]:
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
    def snapshot_dict(
        self,
    ) -> Optional[Dict[float, Dict[str, Union[Dict[str, any], pd.DataFrame]]]]:
        if self._snapshot_dict is None:
            warnings.warn("Snapshot dictionary is not initialized. Call merge() first.")
            return None
        if not self._is_file_dict_loaded:
            warnings.warn("File dictionary is not loaded. Call load() first.")
        return self._snapshot_dict

    @property
    def snapshot_list(
        self,
    ) -> Optional[List[Tuple[float, Dict[str, Union[Dict[str, any], pd.DataFrame]]]]]:
        return list(self.snapshot_dict.items()) if self.snapshot_dict else None

    def merge(
        self, is_strict: bool = True
    ) -> Dict[float, Dict[str, Union[Dict[str, any], pd.DataFrame]]]:
        self._snapshot_dict = {}
        if not self._raw_timestamps:
            warnings.warn("Raw data not loaded. Call load() first.")
            return self._snapshot_dict
        if self._timestamps:
            warnings.warn("Snapshot dictionary already initialized. Overwriting.")
            self._timestamps = None

        for timestamp in self._raw_timestamps:
            snapshot_data = self._merge(timestamp)

            # skip if snapshot_data is None (e.g., cluster has dissolved)
            if snapshot_data is None:
                break

            # check if snapshot_df contains NaN values
            snapshot_df = snapshot_data["data"]
            if snapshot_df.isnull().values.any():
                nan_rows = snapshot_df[snapshot_df.isnull().any(axis=1)]
                nan_names = nan_rows["name"].tolist()

                msg_text = (
                    f"Snapshot at {timestamp} contains NaN values. "
                    "Possibly due to misaligned `name` values across files. "
                    f"Names with NaN values: {nan_names}. "
                )

                if is_strict:
                    raise ValueError(
                        msg_text + "\nSet `is_strict=False` to ignore this check."
                    )
                else:
                    warnings.warn(msg_text + "\nRows with NaN values will be dropped.")
                    snapshot_data["data"] = snapshot_df.dropna()

            self._snapshot_dict[timestamp] = snapshot_data

        # construct timestamps from snapshot_dict keys
        self._timestamps = list(self._snapshot_dict.keys())
        return self._snapshot_dict

    def _merge(
        self, timestamp: float
    ) -> Dict[str, Union[Dict[str, Any], pd.DataFrame, Dict[str, NBody6OutputFile]]]:
        # extract snapshots with header and data from corresponding timestamp
        out34_snapshot = self._file_dict["OUT34"][timestamp]
        out9_snapshot = self._file_dict["OUT9"][timestamp]
        fort19_snapshot = self._file_dict["fort.19"][timestamp]
        fort82_snapshot = self._file_dict["fort.82"][timestamp]
        fort83_snapshot = self._file_dict["fort.83"][timestamp]
        density_center_info = self._file_dict["densCentre.txt"][timestamp]

        # skip if cluster has dissolved
        if density_center_info["header"]["r_tidal"] < 0:
            warnings.warn(
                f"Cluster has dissolved at timestamp {timestamp}. "
                "Terminating merging."
            )
            return None

        # calculate distance between stars and density center, in multiple of tidal radius
        cluster_gal_position = (
            out34_snapshot["header"]["rg"] * out34_snapshot["header"]["rbar"]
        )
        cluster_gal_velocity = (
            out34_snapshot["header"]["vg"] * out34_snapshot["header"]["vstar"]
        )

        # collect header information
        header_dict = {
            "time": out34_snapshot["header"]["time"],
            "r_tidal": density_center_info["header"]["r_tidal"],
            "cluster_density_center": (
                density_center_info["header"]["density_center_x"],
                density_center_info["header"]["density_center_y"],
                density_center_info["header"]["density_center_z"],
            ),
            "r_tidal_out34": out34_snapshot["header"]["rtide"],
            "cluster_density_center_out34": tuple(
                map(float, out34_snapshot["header"]["rd"])
            ),
            "cluster_mass_center_out34": tuple(
                map(float, out34_snapshot["header"]["rcm"])
            ),
            "cluster_gal_position_out34": cluster_gal_position,
            "cluster_gal_velocity_out34": cluster_gal_velocity,
        }

        pos_vel_keys = ["x", "y", "z", "vx", "vy", "vz"]
        attr_keys = ["mass", "zlum", "rad", "tempe"]
        out34_snapshot_df = out34_snapshot["data"][["name"] + pos_vel_keys]

        # non regularized binaries
        non_reg_bin_df = (
            fort19_snapshot["data"]
            .assign(pair=lambda df: df.index.map(lambda x: f"b{x + 1}"))
            .melt(
                id_vars=["pair", "semi"],
                value_vars=["name1", "name2"],
                value_name="name",
            )
            .loc[:, ["pair", "name", "semi"]]
            .merge(fort83_snapshot["data"][["name"] + attr_keys], on="name", how="left")
            .merge(out34_snapshot_df[["name"] + pos_vel_keys], on="name", how="left")
        )

        # regularized binaries
        raw_reg_bin_df = (
            out9_snapshot["data"][["name1", "name2", "semi", "cmName"]]
            .rename(columns={"cmName": "pair"})
            .merge(fort82_snapshot["data"], on=["name1", "name2"], how="left")
        )
        reg_bin_df = (
            pd.concat(
                [
                    raw_reg_bin_df.assign(
                        name=lambda df, i=i: df[f"name{i}"],
                        **{
                            lbl: (lambda df, lbl=lbl, i=i: df[f"{lbl}{i}"])
                            for lbl in attr_keys
                        },
                    )
                    for i in (1, 2)
                ],
                ignore_index=True,
            )
            .loc[:, ["pair", "name", "semi"] + attr_keys]
            .merge(
                out34_snapshot_df.rename(columns={"name": "pair"}).loc[
                    :, ["pair"] + pos_vel_keys
                ],
                on="pair",
                how="left",
            )
        )

        # single stars
        binary_names = set(non_reg_bin_df["name"]) | set(reg_bin_df["name"])
        singles_df = (
            fort83_snapshot["data"]
            .loc[lambda df: ~df["name"].isin(binary_names)]
            .assign(pair="s", semi=-1)
            .loc[:, ["pair", "name", "semi"] + attr_keys]
            .merge(out34_snapshot_df[["name"] + pos_vel_keys], on="name", how="left")
        )

        merged_df = pd.concat(
            [reg_bin_df, non_reg_bin_df, singles_df], ignore_index=True
        )
        # calculate distance to revised density center from `densCentre.txt`
        merged_df["r_dc"] = np.linalg.norm(
            merged_df[["x", "y", "z"]].values
            - np.array(
                [
                    density_center_info["header"]["density_center_x"],
                    density_center_info["header"]["density_center_y"],
                    density_center_info["header"]["density_center_z"],
                ]
            ),
            axis=1,
        )
        merged_df["r_dc_r_tidal"] = (
            merged_df["r_dc"] / density_center_info["header"]["r_tidal"]
        )

        # convert T_eff, L, R, F to log scale
        merged_df.rename(
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
        merged_df["T_eff"] = np.power(10, merged_df["log_T_eff"])
        merged_df["L_sol"] = np.power(10, merged_df["log_L_sol"])
        merged_df["R_sol"] = np.power(10, merged_df["log_R_sol"])
        # flux in L_sun / R_sun^2
        merged_df["F_sol"] = merged_df["L_sol"] / merged_df["R_sol"] ** 2
        merged_df["log_F_sol"] = np.log10(merged_df["F_sol"])

        # is_binary flag for non-regularized binaries & regularized binaries
        merged_df["is_binary"] = merged_df["name"].isin(binary_names)

        # skip binaries ID (`name` <= `nzero`) in the merged table
        merged_df = merged_df[merged_df["name"] <= out34_snapshot["header"]["nzero"]]

        return {
            "header": header_dict,
            "data": merged_df[
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
                    "L_sol",
                    "R_sol",
                    "F_sol",
                    "log_T_eff",
                    "log_L_sol",
                    "log_R_sol",
                    "log_F_sol",
                    "is_binary",
                ]
            ],
            "raw": {
                "densCentre.txt": density_center_info["data"],
                "OUT34": out34_snapshot["data"],
                "OUT9": out9_snapshot["data"],
                "fort.82": fort82_snapshot["data"],
                "fort.83": fort83_snapshot["data"],
                "fort.19": fort19_snapshot["data"],
            },
        }

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
    ):
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

    def get_statistics(self) -> Optional[pd.DataFrame]:
        def _describe_stats(
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

        if self._snapshot_dict is None:
            warnings.warn("Snapshot dictionary is not initialized. Call merge() first.")
            return None

        simulation_stats = []
        for timestamp, snapshot_data in self._snapshot_dict.items():
            header = snapshot_data["header"]
            snapshot_df = snapshot_data["data"]

            snapshot_within_r_tidal_df = snapshot_df[snapshot_df["r_dc_r_tidal"] < 1]
            snapshot_within_2x_r_tidal_df = snapshot_df[snapshot_df["r_dc_r_tidal"] < 2]

            simulation_stats.extend(
                [
                    {
                        "time": header["time"],
                        "r_tidal": header["r_tidal"],
                        "cluster_density_center": header["cluster_density_center"],
                        "n_star": len(snapshot_df),
                        "n_reg_binary": snapshot_df["is_binary"].sum(),
                        **_describe_stats(
                            snapshot_df,
                            "mass",
                        ),
                        "total_mass": snapshot_df["mass"].sum(),
                        "n_star_within_r_tidal": len(snapshot_within_r_tidal_df),
                        "n_reg_binary_within_r_tidal": (
                            snapshot_within_r_tidal_df["is_binary"].sum()
                        ),
                        **_describe_stats(
                            snapshot_within_r_tidal_df,
                            "mass",
                            prefix="within_r_tidal",
                        ),
                        "total_mass_within_r_tidal": snapshot_within_r_tidal_df[
                            "mass"
                        ].sum(),
                        "n_star_within_2x_r_tidal": len(snapshot_within_2x_r_tidal_df),
                        "n_reg_binary_within_2x_r_tidal": (
                            snapshot_within_2x_r_tidal_df["is_binary"].sum()
                        ),
                        **_describe_stats(
                            snapshot_within_2x_r_tidal_df,
                            "mass",
                            prefix="within_2x_r_tidal",
                        ),
                        "total_mass_within_2x_r_tidal": snapshot_within_2x_r_tidal_df[
                            "mass"
                        ].sum(),
                    }
                ]
            )

        return (
            pd.DataFrame(simulation_stats).sort_values(by="time").reset_index(drop=True)
        )
