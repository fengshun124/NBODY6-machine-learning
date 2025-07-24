from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.stats import sigma_clip
from matplotlib.legend_handler import HandlerTuple

plt.style.use("./module/nbody6/plot/plot_style.mlpstyle")


def visualize_nbody6_snapshot(
    snapshot: Tuple[float, Dict[str, Union[Dict[str, Any], pd.DataFrame]]],
    figure: Optional[plt.Figure] = None,
    title_text: Optional[str] = None,
    hr_T_eff_plot_range: Optional[Tuple[float, float]] = None,
    hr_T_eff_major_locator: float = 0.4,
    hr_T_eff_minor_locator: float = 0.2,
    hr_L_plot_range: Optional[Tuple[float, float]] = None,
    hr_L_major_locator: float = 1,
    hr_L_minor_locator: float = 0.2,
    hr_mass_reference: Optional[List[float]] = [0.05, 0.1, 0.5, 1, 2, 4, 8],
    xyz_plot_limit: Optional[float] = None,
    xyz_major_locator: Optional[float] = None,
    xyz_minor_locator: Optional[float] = None,
    v_xyz_plot_limit: Optional[float] = None,
) -> plt.Figure:
    timestamp, snapshot_data = snapshot
    snapshot_header = snapshot_data["header"]
    snapshot_df = snapshot_data["data"]

    if not figure:
        figure = plt.figure(figsize=(16.2, 6.3), dpi=300)
    else:
        figure.clear()

    if title_text:
        figure.suptitle(title_text)

    grid = figure.add_gridspec(
        nrows=3,
        ncols=7,
        hspace=0,
        wspace=0,
        width_ratios=[1.3, 3, 1.3, 2, 2, 1, 0.2],
        height_ratios=[0.2, 1, 1],
    )
    figure.tight_layout()

    # info ax
    info_ax = figure.add_subplot(grid[1, 4])
    info_ax.axis("off")
    info_ax.text(
        0.05,
        0.05,
        f"$T={snapshot_header['time']:.2f}$ Myr\n"
        rf"$N_\mathrm{{reg.\;bin.}}={int(snapshot_df['is_binary'].sum())}$"
        "\n"
        rf"$N_\mathrm{{single+non-reg.\;bin.}}={len(snapshot_df) - snapshot_df['is_binary'].sum()}$"
        "\n"
        rf"$M_\mathrm{{total}}={snapshot_df['mass'].sum():.2f}\;M_\odot$"
        "\n"
        rf"$R_\mathrm{{tidal}}={snapshot_header['rtide']:.2f}\;\mathrm{{pc}}$",
        transform=info_ax.transAxes,
        va="bottom",
        ha="left",
    )
    # dummy points for binary/single legend
    for is_binary, plot_marker in [(True, "d"), (False, "o")]:
        info_ax.scatter(
            [],
            [],
            marker=plot_marker,
            s=72,
            fc="lightgrey",
            ec="k",
            label="reg. binary" if is_binary else "single+non-reg. bin.",
        )
    info_ax.legend(
        loc="upper left",
        frameon=True,
        scatterpoints=True,
        handletextpad=0.2,
        labelspacing=1,
        fontsize=12,
        bbox_to_anchor=(0, 1.05),
    )

    # HR-diagram
    T_eff_key, L_key, mass_key = "log_T_eff", "log_L_sol", "mass"
    hr_ax = figure.add_subplot(grid[1:, 1])

    hr_ax.set_xlabel(r"$\log_{10}\left(T_{{\rm eff}} [K]\right)$")
    hr_ax.set_ylabel(r"$\log_{10}\left(L [L_\odot]\right)$")
    hr_ax.invert_xaxis()

    if hr_T_eff_plot_range:
        T_eff_min, T_eff_max = hr_T_eff_plot_range
    else:
        T_eff_min = (
            np.ceil((snapshot_df[T_eff_key].min() / hr_T_eff_minor_locator) - 2)
            * hr_T_eff_minor_locator
        )
        T_eff_max = (
            np.floor((snapshot_df[T_eff_key].max() / hr_T_eff_minor_locator) + 2)
            * hr_T_eff_minor_locator
        )
    hr_ax.set_xlim(T_eff_max, T_eff_min)
    hr_ax.xaxis.set_major_locator(plt.MultipleLocator(hr_T_eff_major_locator))
    hr_ax.xaxis.set_minor_locator(plt.MultipleLocator(hr_T_eff_minor_locator))
    hr_ax.tick_params(
        axis="x", which="both", bottom=True, top=True, labelbottom=True, labeltop=True
    )
    hr_ax.xaxis.set_label_position("bottom")

    if hr_L_plot_range:
        L_min, L_max = hr_L_plot_range
    else:
        L_min = (
            np.ceil((snapshot_df[L_key].min() / hr_L_minor_locator) - 2)
            * hr_L_minor_locator
        )
        L_max = (
            np.floor((snapshot_df[L_key].max() / hr_L_minor_locator) + 2)
            * hr_L_minor_locator
        )
    hr_ax.set_ylim(L_min, L_max)
    hr_ax.yaxis.set_major_locator(plt.MultipleLocator(hr_L_major_locator))
    hr_ax.yaxis.set_minor_locator(plt.MultipleLocator(hr_L_minor_locator))
    hr_ax.tick_params(
        axis="y", which="both", left=True, right=True, labelleft=True, labelright=True
    )
    hr_ax.yaxis.set_label_position("left")

    hr_ax.grid(ls=":", lw=0.5, c="k", alpha=0.4)

    def mass2size(mass: float) -> float:
        return 12 + 64 * np.log1p(mass)

    for is_binary, (plot_marker, plot_color) in [
        (True, ("d", "darkorange")),
        (False, ("o", "tab:green")),
    ]:
        subset_df = snapshot_df[snapshot_df["is_binary"] == is_binary].sort_values(
            by=[mass_key], ascending=False
        )
        hr_ax.scatter(
            subset_df[T_eff_key],
            subset_df[L_key],
            marker=plot_marker,
            s=mass2size(subset_df[mass_key]),
            c="None",
            edgecolors=plot_color,
            linewidths=0.8,
        )

    # dummy points for mass legend
    handles, labels = zip(
        *[
            (
                (
                    hr_ax.scatter(
                        [],
                        [],
                        marker="o",
                        s=mass2size(mass),
                        c="None",
                        edgecolors="tab:green",
                        linewidths=0.8,
                    ),
                    hr_ax.scatter(
                        [],
                        [],
                        marker="d",
                        s=mass2size(mass),
                        c="None",
                        edgecolors="darkorange",
                        linewidths=0.8,
                    ),
                ),
                f"{mass:.2f} $M_\\odot$",
            )
            for mass in hr_mass_reference
        ]
    )
    hr_ax.legend(
        handles,
        labels,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.2)},
        loc="center right",
        title="Mass",
        frameon=True,
        scatterpoints=True,
        handletextpad=0.4,
        labelspacing=1.2,
        bbox_to_anchor=(-0.25, 0.5),
    )

    # spatial / velocity distribution
    xyz_ax_dict = {
        "xy": (grid[1, 3], ("X [pc]", "Y [pc]"), ("x", "y", "vz")),
        "xz": (grid[2, 3], ("X [pc]", "Z [pc]"), ("x", "z", "vy")),
        "yz": (grid[2, 4], ("Y [pc]", "Z [pc]"), ("y", "z", "vx")),
    }

    v_xyz_plot_limit = v_xyz_plot_limit or np.ceil(
        np.max(
            np.abs(
                sigma_clip(snapshot_df[["vx", "vy", "vz"]].values.flatten(), sigma=2)
            )
        )
    )
    cbar_norm = plt.Normalize(-v_xyz_plot_limit, v_xyz_plot_limit)

    if xyz_plot_limit is None:
        xyz_plot_limit = np.ceil(
            np.max(
                sigma_clip(np.ptp(snapshot_df[["x", "y", "z"]].values, axis=0), sigma=2)
            )
            / 2
        )
    xyz_major_locator = xyz_major_locator or xyz_plot_limit
    xyz_minor_locator = xyz_minor_locator or xyz_major_locator / 4

    for ax_label, (
        ax,
        (x_label, y_label),
        (x_key, y_key, v_z_key),
    ) in xyz_ax_dict.items():
        ax = figure.add_subplot(ax)
        for is_binary, plot_marker in [(True, "d"), (False, "o")]:
            subset_df = snapshot_df[snapshot_df["is_binary"] == is_binary].sort_values(
                by=[mass_key], ascending=False
            )
            ax.scatter(
                subset_df[x_key],
                subset_df[y_key],
                marker=plot_marker,
                s=mass2size(subset_df[mass_key]),
                c=subset_df[v_z_key],
                cmap="RdBu_r",
                norm=cbar_norm,
                edgecolors="k",
                linewidths=0.6,
            )

        ax.set_xlabel(x_label)
        ax.xaxis.set_major_locator(plt.MultipleLocator(xyz_major_locator))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(xyz_minor_locator))
        ax.set_xlim(
            -xyz_plot_limit - 1 * xyz_minor_locator,
            xyz_plot_limit + 1 * xyz_minor_locator,
        )

        ax.set_ylabel(y_label)
        ax.yaxis.set_major_locator(plt.MultipleLocator(xyz_major_locator))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(xyz_minor_locator))
        ax.set_ylim(
            -xyz_plot_limit - 1 * xyz_minor_locator,
            xyz_plot_limit + 1 * xyz_minor_locator,
        )

        # ax.set_aspect("equal", adjustable="box")
        ax.grid(ls=":", lw=0.5, c="k", alpha=0.4)

        match ax_label:
            case "xy":
                ax.tick_params(
                    axis="both",
                    which="both",
                    bottom=True,
                    top=True,
                    labelbottom=False,
                    labeltop=True,
                )
                ax.xaxis.set_label_position("top")
            case "yz":
                ax.tick_params(
                    axis="both",
                    which="both",
                    left=True,
                    right=True,
                    labelleft=False,
                    labelright=True,
                )
                ax.yaxis.set_label_position("right")

    # colorbar
    cax = figure.add_subplot(grid[1:, -1])
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=cbar_norm, cmap="RdBu_r"),
        cax=cax,
        orientation="vertical",
    )
    cbar.set_label(r"$v$ $[\mathrm{km\,s}^{-1}]$")
    cbar.ax.tick_params(direction="in", which="both")

    cbar_ticks = np.linspace(-cbar_norm.vmax, cbar_norm.vmax, 5)
    cbar.ax.set_yticks(cbar_ticks)
    cbar.ax.set_yticklabels(
        [rf"$\leq{cbar_ticks[0]:.1f}$"]
        + [f"{tick:.1f}" for tick in cbar_ticks[1:-1]]
        + [rf"$\geq{cbar_ticks[-1]:.1f}$"]
    )

    return figure
