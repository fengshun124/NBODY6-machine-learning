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
    T_eff_major_locator, T_eff_minor_locator = 0.6, 0.2
    L_major_locator, L_minor_locator = 2, 0.2

    all_v_xyz = np.concatenate(
        [
            snapshot[1]["data"][["vx", "vy", "vz"]].values.flatten()
            for snapshot in snapshot_list
        ]
    )
    v_xyz_plot_limit = np.ceil(np.max(sigma_clip(all_v_xyz, sigma=3) / 0.8)) * 0.8

    figure = plt.figure(figsize=(16.2, 6.3), dpi=300)

    def update(frame: int):
        return visualize_nbody6_snapshot(
            snapshot=snapshot_list[frame],
            figure=figure,
            title_text=fig_title_text,
            hr_T_eff_plot_range=(2.8, 5.2),
            hr_T_eff_major_locator=T_eff_major_locator,
            hr_T_eff_minor_locator=T_eff_minor_locator,
            hr_L_plot_range=(-4.4, 4.4),
            hr_L_major_locator=L_major_locator,
            hr_L_minor_locator=L_minor_locator,
            xyz_plot_limit=75,
            xyz_major_locator=50,
            xyz_minor_locator=5,
            v_xyz_plot_limit=v_xyz_plot_limit,
        )

    animation = FuncAnimation(figure, update, frames=len(snapshot_list), blit=False)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tqdm(
            total=len(snapshot_list),
            desc=f"Animating {fig_title_text or ''}",
        ) as pbar:
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
