from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.stats import sigma_clip
from matplotlib.animation import FFMpegWriter, FuncAnimation
from tqdm.auto import tqdm

from module.nbody6.plot.plot import visualize_nbody6_snapshot


def animate_nbody6_snapshots(
    snapshot_list: List[Tuple[float, Dict[str, Union[Dict[str, Any], pd.DataFrame]]]],
    fig_title_text: Optional[str] = None,
    animation_fps: int = 10,
    animation_dpi: int = 300,
    output_path: Union[str, Path] = None,
) -> FuncAnimation:
    # pre-compute plot range
    all_T_eff = np.concatenate(
        [snapshot[1]["data"]["log_T_eff"].values for snapshot in snapshot_list]
    )
    T_eff_major_locator, T_eff_minor_locator = 0.4, 0.2
    T_eff_plot_range = (
        np.ceil((all_T_eff.min() / T_eff_minor_locator) - 2) * T_eff_minor_locator,
        np.floor((all_T_eff.max() / T_eff_minor_locator) + 2) * T_eff_minor_locator,
    )

    all_L = np.concatenate(
        [snapshot[1]["data"]["log_L_sol"].values for snapshot in snapshot_list]
    )
    L_major_locator, L_minor_locator = 1, 0.2
    L_plot_range = (
        np.ceil((all_L.min() / L_minor_locator) - 2) * L_minor_locator,
        np.floor((all_L.max() / L_minor_locator) + 2) * L_minor_locator,
    )

    all_v_xyz = np.concatenate(
        [
            snapshot[1]["data"][["vx", "vy", "vz"]].values.flatten()
            for snapshot in snapshot_list
        ]
    )
    v_xyz_plot_limit = np.ceil(np.max(sigma_clip(all_v_xyz, sigma=3)))

    figure = visualize_nbody6_snapshot(
        snapshot=snapshot_list[0],
        title_text=fig_title_text,
        hr_T_eff_plot_range=T_eff_plot_range,
        hr_T_eff_major_locator=T_eff_major_locator,
        hr_T_eff_minor_locator=T_eff_minor_locator,
        hr_L_plot_range=L_plot_range,
        hr_L_major_locator=L_major_locator,
        hr_L_minor_locator=L_minor_locator,
        xyz_plot_limit=60,
        xyz_major_locator=25,
        xyz_minor_locator=5,
        v_xyz_plot_limit=v_xyz_plot_limit,
    )

    def update(frame: int):
        return visualize_nbody6_snapshot(
            snapshot=snapshot_list[frame],
            figure=figure,
            title_text=fig_title_text,
            hr_T_eff_plot_range=T_eff_plot_range,
            hr_T_eff_major_locator=T_eff_major_locator,
            hr_T_eff_minor_locator=T_eff_minor_locator,
            hr_L_plot_range=L_plot_range,
            hr_L_major_locator=L_major_locator,
            hr_L_minor_locator=L_minor_locator,
            xyz_plot_limit=70,
            xyz_major_locator=30,
            xyz_minor_locator=5,
            v_xyz_plot_limit=v_xyz_plot_limit,
        )

    animation = FuncAnimation(figure, update, frames=len(snapshot_list), blit=False)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tqdm(total=len(snapshot_list), desc="Saving Animation") as pbar:
            animation.save(
                output_path,
                writer=FFMpegWriter(
                    fps=animation_fps,
                    metadata={"title": fig_title_text or "NBody6 Snapshot Animation"},
                    codec="libx264",
                ),
                dpi=animation_dpi,
                progress_callback=lambda i, n: pbar.update(1),
            )
    else:
        plt.show()

    return animation
