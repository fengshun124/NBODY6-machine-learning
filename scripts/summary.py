import os
import re
from pathlib import Path
from string import ascii_lowercase

import click
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm.auto import tqdm

load_dotenv()

# filename patterns
FEATURE_SET_PATTERNS = {
    "lon_deg+lat_deg+pm_lon_coslat_mas_yr+pm_lat_mas_yr": "sky",
    "lon_deg+lat_deg+pm_lon_coslat_mas_yr+pm_lat_mas_yr+log_L_L_sol": "sky+L",
    "x+y+z+vx+vy+vz": "cartesian",
    "x+y+z+vx+vy+vz+log_L_L_sol": "cartesian+L",
}

TARGET_PATTERNS = {
    "time": "age",
    "total_mass_within_2x_r_tidal": "total_mass",
}

# hard-coded param sizes for each experiment configuration
# based on the model architectures and feature dimensions
MODEL_PARAM_SIZES = {
    "SS[4]": {"sky": 125, "sky+L": 153, "cartesian": 181, "cartesian+L": 209},
    "SS[8]": {"sky": 249, "sky+L": 305, "cartesian": 361, "cartesian+L": 417},
    "SS[12]": {"sky": 373, "sky+L": 457, "cartesian": 541, "cartesian+L": 625},
    "SS[24]": {"sky": 745, "sky+L": 913, "cartesian": 1081, "cartesian+L": 1249},
    "DS[4,4]": {"sky": 45, "sky+L": 49, "cartesian": 53, "cartesian+L": 57},
    "DS[8,8]": {"sky": 121, "sky+L": 129, "cartesian": 137, "cartesian+L": 145},
    "DS[12,12]": {"sky": 229, "sky+L": 241, "cartesian": 253, "cartesian+L": 265},
    "DS[24,24]": {"sky": 745, "sky+L": 769, "cartesian": 793, "cartesian+L": 817},
    "ST[4,2,1]": {"sky": 401, "sky+L": 413, "cartesian": 425, "cartesian+L": 437},
    "ST[4,4,1]": {"sky": 401, "sky+L": 413, "cartesian": 425, "cartesian+L": 437},
    "ST[4,2,2]": {"sky": 597, "sky+L": 609, "cartesian": 621, "cartesian+L": 633},
    "ST[4,4,2]": {"sky": 597, "sky+L": 609, "cartesian": 621, "cartesian+L": 633},
    "ST[6,2,1]": {"sky": 781, "sky+L": 799, "cartesian": 817, "cartesian+L": 835},
    "ST[6,2,2]": {"sky": 1183, "sky+L": 1201, "cartesian": 1219, "cartesian+L": 1237},
    "ST[6,3,1]": {"sky": 781, "sky+L": 799, "cartesian": 817, "cartesian+L": 835},
    "ST[6,3,2]": {"sky": 1183, "sky+L": 1201, "cartesian": 1219, "cartesian+L": 1237},
    "ST[6,6,1]": {"sky": 781, "sky+L": 799, "cartesian": 817, "cartesian+L": 835},
}

PARQUET_FILENAME_PATTERN = re.compile(
    r"^"
    r"(?P<experiment_name>"
    r"from\+(?P<feature_set_key>[^-]+)"
    r"-to\+(?P<target_key>[^-]+)"
    r"-nstar(?P<n_star_per_sample>\d+)"
    r"-nsnap(?P<n_sample_per_snapshot>\d+)"
    r"-dp(?P<drop_probability>[^-]+)"
    r"-dr(?P<drop_ratio_min>[^_]+)_(?P<drop_ratio_max>[^-]+)"
    r"-(?P<model_name>summary_stats|set_transformer|deep_sets)"
    r"-(?P<model_hparams>.+?)"
    r"-bs(?P<batch_size>\d+)"
    r"-lr(?P<learning_rate>[^-]+)"
    r"-wd(?P<weight_decay>[^-]+)"
    r")"
    r"(?:-(?P<auxiliary_label>[^-]+)-seed(?P<seed>\d+)-test)?"
    r"(?:\.parquet)?"
    r"$"
)


def _parse_fmt_float(value: str) -> float:
    """Decode formatted float: 'p' -> '.', 'n' -> '-' and convert to float."""
    return float(value.translate(str.maketrans({"p": ".", "n": "-"})))


def _lookup_model_param_size(model_label: str, feature_set_label: str) -> int:
    try:
        return MODEL_PARAM_SIZES[model_label][feature_set_label]
    except KeyError as exc:
        raise ValueError(
            "Missing model param size for "
            f"model={model_label!r}, feature_set={feature_set_label!r}"
        ) from exc


def _parse_experiment_name(experiment_name: str) -> dict[str, object]:
    if not (match := PARQUET_FILENAME_PATTERN.match(experiment_name)):
        raise ValueError(f"Unexpected experiment/parquet name: {experiment_name!r}")

    parts = match.groupdict()
    feature_set_key = parts["feature_set_key"]
    target_key = parts["target_key"]
    if feature_set_key not in FEATURE_SET_PATTERNS:
        raise ValueError(f"Unknown feature set key: {feature_set_key!r}")
    if target_key not in TARGET_PATTERNS:
        raise ValueError(f"Unknown target key: {target_key!r}")

    model_name = parts["model_name"]
    model_hparams = parts["model_hparams"]
    model_family = model_name
    feature_set_label = FEATURE_SET_PATTERNS[feature_set_key]
    target_label = TARGET_PATTERNS[target_key]

    match model_name:
        case "summary_stats":
            if not (model_match := re.search(r"^h(?P<h>[^-]+)(?:-|$)", model_hparams)):
                raise ValueError(f"Invalid summary_stats params: {model_hparams!r}")
            model_label = f"SS[{model_match.group('h')}]"
        case "set_transformer":
            if not (
                model_match := re.search(
                    r"^hd(?P<hd>[^-]+)-nh(?P<nh>[^-]+)-ns(?P<ns>[^-]+)(?:-|$)",
                    model_hparams,
                )
            ):
                raise ValueError(f"Invalid set_transformer params: {model_hparams!r}")
            model_label = (
                f"ST[{model_match.group('hd')},"
                f"{model_match.group('nh')},"
                f"{model_match.group('ns')}]"
            )
        case "deep_sets":
            if not (
                model_match := re.search(
                    r"^phi(?P<phi>[^-]+)-rho(?P<rho>[^-]+)(?:-|$)",
                    model_hparams,
                )
            ):
                raise ValueError(f"Invalid deep_sets params: {model_hparams!r}")
            model_label = f"DS[{model_match.group('phi')},{model_match.group('rho')}]"
        case _:
            raise ValueError(f"Unknown model name: {model_name!r}")

    model_param_size = _lookup_model_param_size(model_label, feature_set_label)

    return {
        "experiment_name": parts["experiment_name"],
        "feature_set_label": feature_set_label,
        "target_label": target_label,
        "model_name": model_name,
        "model_hparams": model_hparams,
        "model_family": model_family,
        "model_label": model_label,
        "model_param_size": model_param_size,
        "n_star_per_sample": int(parts["n_star_per_sample"]),
        "n_sample_per_snapshot": int(parts["n_sample_per_snapshot"]),
        "drop_probability": _parse_fmt_float(parts["drop_probability"]),
        "drop_ratio_min": _parse_fmt_float(parts["drop_ratio_min"]),
        "drop_ratio_max": _parse_fmt_float(parts["drop_ratio_max"]),
        "batch_size": int(parts["batch_size"]),
        "learning_rate": _parse_fmt_float(parts["learning_rate"]),
        "weight_decay": _parse_fmt_float(parts["weight_decay"]),
        "auxiliary_label": parts.get("auxiliary_label"),
        "seed": int(parts["seed"]) if parts.get("seed") else None,
    }


def _calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:

    pearson_r, pearson_p = pearsonr(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
    }


def _calc_sample_balanced_metrics(
    result_df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    sample_id_col: str = "sample_id",
) -> dict[str, float]:
    """Summarize per-sample error distributions with equal sample weighting."""
    required_columns = {sample_id_col, y_true_col, y_pred_col}
    missing_columns = required_columns - set(result_df.columns)
    if missing_columns:
        raise ValueError(
            "Missing required columns for samplewise metrics: "
            f"{sorted(missing_columns)}"
        )

    error = result_df[y_pred_col] - result_df[y_true_col]
    per_sample_df = (
        pd.DataFrame(
            {
                sample_id_col: result_df[sample_id_col].to_numpy(),
                "error": error.to_numpy(),
                "abs_error": np.abs(error.to_numpy()),
                "sq_error": np.square(error.to_numpy()),
            }
        )
        .groupby(sample_id_col, sort=False)
        .agg(
            sample_medae=("abs_error", "median"),
            sample_mse=("sq_error", "mean"),
            sample_bias=("error", "median"),
        )
    )
    per_sample_df["sample_rmse"] = np.sqrt(per_sample_df["sample_mse"])

    sample_medae = per_sample_df["sample_medae"].to_numpy()
    sample_rmse = per_sample_df["sample_rmse"].to_numpy()
    sample_bias = per_sample_df["sample_bias"].to_numpy()

    medae_q25, medae_q50, medae_q75 = np.quantile(sample_medae, [0.25, 0.50, 0.75])
    medae_iqr = medae_q75 - medae_q25
    medae_inlier = sample_medae[
        (sample_medae >= medae_q25 - 1.5 * medae_iqr)
        & (sample_medae <= medae_q75 + 1.5 * medae_iqr)
    ]
    medae_whisker_low, medae_whisker_high = (
        (float(np.min(medae_inlier)), float(np.max(medae_inlier)))
        if medae_inlier.size > 0
        else (medae_q25, medae_q75)
    )

    return {
        "sb_medae_med": float(medae_q50),
        "sb_rmse_med": float(np.median(sample_rmse)),
        "sb_bias_med": float(np.median(sample_bias)),
        "sb_medae_p90": float(np.quantile(sample_medae, 0.90)),
        "sb_medae_p95": float(np.quantile(sample_medae, 0.95)),
        "sb_medae_q25": float(medae_q25),
        "sb_medae_q75": float(medae_q75),
        "sb_medae_iqr": float(medae_iqr),
        "sb_medae_wlow": float(medae_whisker_low),
        "sb_medae_whigh": float(medae_whisker_high),
        "n_sample_id": int(len(per_sample_df)),
    }


def _format_stats_annotation(metric: dict[str, float], y_unit: str) -> str:
    p_val_str = (
        f"{metric['pearson_p']:.2e}"
        if metric["pearson_p"] < 0.001
        else f"{metric['pearson_p']:.3f}"
    )
    unit_str = f" {y_unit}" if y_unit else ""
    return (
        f"Pearson $r={metric['pearson_r']:.4f}$\n($p={p_val_str}$)\n"
        f"$R^2={metric['r2']:.4f}$\n"
        f"MAE$={metric['mae']:.2f}${unit_str}\n"
        f"RMSE$={metric['rmse']:.2f}${unit_str}"
    )


def _plot_distribution(
    result_df: pd.DataFrame,
    meta: dict[str, object],
    figure_path: Path,
) -> None:
    plt.style.use(
        os.environ.get(
            "MPL_STYLE",
            "seaborn-v0_8-whitegrid",
        )
    )

    _TARGET_PLOT_LABEL = {
        "age": "$t$",
        "total_mass": "$M_\\mathrm{total}$",
    }

    # validate required columns
    required_columns = {
        "target_scaled",
        "prediction_scaled",
        "target_original",
        "prediction_original",
    }
    missing_columns = required_columns - set(result_df.columns)
    if missing_columns:
        raise ValueError(
            "Missing required columns for plotting distribution: "
            f"{sorted(missing_columns)}"
        )

    # setup plot configurations
    scaled_limit = (-4, 4)
    scaled_major_locator = mpl.ticker.MultipleLocator(base=3, offset=0)
    scaled_minor_locator = mpl.ticker.MultipleLocator(base=0.25)

    target_key = str(meta.get("target_key", ""))
    is_age_target = target_key == "time"
    if is_age_target:
        y_label = "age"
        y_unit = "Myr"
        original_limit = (-30, 330)
        original_major_locator = mpl.ticker.MultipleLocator(base=60)
        original_minor_locator = mpl.ticker.MultipleLocator(base=30)
    else:
        y_label = "$M_\\mathrm{tot.}$"
        y_unit = "$M_\\odot$"
        original_limit = (-25, 250)
        original_major_locator = mpl.ticker.MultipleLocator(base=100)
        original_minor_locator = mpl.ticker.MultipleLocator(base=25)

    fig, (scaled_ax, original_ax) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    fig.suptitle(
        f"{meta['model_label']}: {meta['feature_set_label']}"
        f"-> {_TARGET_PLOT_LABEL[meta['target_label']]}",
        fontsize=20,
        y=0.98,
    )

    metrics_by_space = meta.get("metrics", {})
    if not isinstance(metrics_by_space, dict):
        metrics_by_space = {}
    scaled_metrics = metrics_by_space.get(
        "scaled",
        _calc_metrics(
            result_df["target_scaled"].to_numpy(),
            result_df["prediction_scaled"].to_numpy(),
        ),
    )
    original_metrics = metrics_by_space.get(
        "original",
        _calc_metrics(
            result_df["target_original"].to_numpy(),
            result_df["prediction_original"].to_numpy(),
        ),
    )
    if not isinstance(scaled_metrics, dict) or not isinstance(original_metrics, dict):
        raise ValueError(
            "Invalid meta['metrics'] format. Expected "
            "{'scaled': dict[str, float], 'original': dict[str, float]}."
        )

    plot_configs = [
        {
            "ax": scaled_ax,
            "y_true": result_df["target_scaled"].to_numpy(),
            "y_pred": result_df["prediction_scaled"].to_numpy(),
            "metrics": scaled_metrics,
            "limit": scaled_limit,
            "major_locator": scaled_major_locator,
            "minor_locator": scaled_minor_locator,
            "unit": "",
            "xlabel": f"True {y_label} (Scaled)",
            "ylabel": f"Predicted {y_label} (Scaled)",
        },
        {
            "ax": original_ax,
            "y_true": result_df["target_original"].to_numpy(),
            "y_pred": result_df["prediction_original"].to_numpy(),
            "metrics": original_metrics,
            "limit": original_limit,
            "major_locator": original_major_locator,
            "minor_locator": original_minor_locator,
            "unit": y_unit,
            "xlabel": f"True {y_label} [{y_unit}]",
            "ylabel": f"Predicted {y_label} [{y_unit}]",
        },
    ]

    cmap = mpl.cm.inferno
    cmin = 100
    norm = mpl.colors.LogNorm(vmin=cmin, vmax=2e5)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for ax_id, config in enumerate(plot_configs):
        ax = config["ax"]
        limit = config["limit"]

        ax.hist2d(
            config["y_true"],
            config["y_pred"],
            bins=100,
            range=[limit, limit],
            cmap=cmap,
            norm=norm,
            cmin=cmin,
        )[3]
        ax.axline((0, 0), slope=1, color="k", ls=":", lw=1)

        ax.text(
            0.05,
            0.95,
            _format_stats_annotation(config["metrics"], config["unit"]),
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor="gray",
                alpha=0.8,
            ),
        )
        ax.text(
            0.95,
            0.95,
            f"({ascii_lowercase[ax_id]})",
            transform=ax.transAxes,
            fontsize=20,
            verticalalignment="top",
            horizontalalignment="right",
        )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(config["xlabel"])
        ax.set_ylabel(config["ylabel"])
        ax.set_xlim(limit)
        ax.set_ylim(limit)
        ax.xaxis.set_major_locator(config["major_locator"])
        ax.xaxis.set_minor_locator(config["minor_locator"])
        ax.yaxis.set_major_locator(config["major_locator"])
        ax.yaxis.set_minor_locator(config["minor_locator"])

    cbar = fig.colorbar(sm, ax=[scaled_ax, original_ax], fraction=0.05, pad=0.04)
    cbar.set_label("N")
    cbar.ax.tick_params(direction="in")

    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def _process_parquet_file(
    parquet_file: Path,
    figures_dir: Path,
) -> dict | None:
    """Parse and evaluate a single parquet result file. Returns None on error."""
    try:
        experiment_info = _parse_experiment_name(parquet_file.stem)
    except ValueError as e:
        print(f"Skipping file {parquet_file} due to parsing error: {e}")
        return None

    test_result_df = pd.read_parquet(parquet_file)
    # raw metrics
    original_metrics = _calc_metrics(
        test_result_df["target_original"].values,
        test_result_df["prediction_original"].values,
    )
    scaled_metrics = _calc_metrics(
        test_result_df["target_scaled"].values,
        test_result_df["prediction_scaled"].values,
    )
    # sample-balanced metrics
    original_sample_metrics = _calc_sample_balanced_metrics(
        test_result_df,
        y_true_col="target_original",
        y_pred_col="prediction_original",
    )
    scaled_sample_metrics = _calc_sample_balanced_metrics(
        test_result_df,
        y_true_col="target_scaled",
        y_pred_col="prediction_scaled",
    )

    _plot_distribution(
        test_result_df,
        meta={
            **experiment_info,
            "metrics": {
                "scaled": scaled_metrics,
                "original": original_metrics,
            },
        },
        figure_path=figures_dir / Path(parquet_file.name).with_suffix(".png"),
    )

    return {
        **experiment_info,
        "n_sample_id": original_sample_metrics["n_sample_id"],
        **{f"physical_{k}": v for k, v in original_metrics.items()},
        **{
            f"physical_{k}": v
            for k, v in original_sample_metrics.items()
            if k != "n_sample_id"
        },
        **{f"scaled_{k}": v for k, v in scaled_metrics.items()},
        **{
            f"scaled_{k}": v
            for k, v in scaled_sample_metrics.items()
            if k != "n_sample_id"
        },
    }


def _calc_pctl_robust_floor_metric(
    summary_df: pd.DataFrame,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> pd.DataFrame:
    # compute scores and aggregate separately
    condition_groups = summary_df.groupby(
        ["feature_set_label", "target_label"], sort=False
    )
    condition_count = condition_groups["physical_sb_medae_med"].transform("count")
    rank_span = (condition_count - 1).clip(lower=1)

    # rank-normalised score for median MedAE
    rank_medae_med = condition_groups["physical_sb_medae_med"].rank(
        ascending=True, method="average"
    )
    score_medae_med = 1.0 - (rank_medae_med - 1.0) / rank_span

    # rank-normalised score for 95th-percentile MedAE
    rank_medae_p95 = condition_groups["physical_sb_medae_p95"].rank(
        ascending=True, method="average"
    )
    score_medae_p95 = 1.0 - (rank_medae_p95 - 1.0) / rank_span

    condition_score = alpha * score_medae_med + (1.0 - alpha) * score_medae_p95
    model_scores = (
        pd.DataFrame(
            {
                "model_label": summary_df["model_label"],
                "condition_score": condition_score,
            }
        )
        .groupby("model_label")["condition_score"]
        .agg(score_mean="mean", score_floor="min", score_std="std")
    )
    model_scores["pctl_robust_floor"] = (
        beta * model_scores["score_mean"] + (1.0 - beta) * model_scores["score_floor"]
    )

    return model_scores.sort_values("pctl_robust_floor", ascending=False).reset_index()


@click.command()
@click.option(
    "--result-root",
    "result_root",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Root directory containing the experiment result",
)
@click.option(
    "--export-dir",
    "exp_dir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    default="summary",
    show_default=True,
    help="Summary output directory for saving summary CSV and figures. "
    "Will be created if it doesn't exist.",
)
@click.option(
    "--workers",
    "workers",
    type=click.IntRange(1, os.cpu_count()),
    default=1,
    show_default=True,
    help="Number of parallel workers for processing parquet files",
)
def main(
    result_root: Path | str,
    exp_dir: Path | str,
    workers: int,
) -> None:
    result_root = Path(result_root).resolve(strict=True)

    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    fig_export_dir = exp_dir / "figures"
    fig_export_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(result_root.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under: {result_root}")

    # gather summary records
    summary_records = [
        r
        for r in tqdm(
            Parallel(n_jobs=workers, return_as="generator")(
                delayed(_process_parquet_file)(f, fig_export_dir) for f in parquet_files
            ),
            total=len(parquet_files),
            desc="Processing PARQUET files",
        )
        if r is not None
    ]
    summary_df = (
        pd.DataFrame(summary_records)
        .sort_values(
            by=[
                "model_family",
                "model_label",
                "feature_set_label",
                "target_label",
            ]
        )
        .reset_index(drop=True)
    )
    summary_csv_path = exp_dir / "summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    click.echo(f"Saved summary CSV: {summary_csv_path}")
    click.echo(f"Saved figure directory: {fig_export_dir}")

    # calculate percentile robust floor metric
    prf_metric = _calc_pctl_robust_floor_metric(summary_df)
    prf_csv_path = exp_dir / "pctl_robust_floor_scores.csv"
    prf_metric.to_csv(prf_csv_path, index=False)

    click.echo(f"Saved percentile robust floor metric CSV: {prf_csv_path}")


if __name__ == "__main__":
    main()
