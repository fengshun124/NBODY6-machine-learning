import os
import re
from pathlib import Path

import click
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from tqdm import tqdm

load_dotenv()
OUTPUT_BASE_ENV = os.getenv("OUTPUT_BASE")
if OUTPUT_BASE_ENV is None:
    raise EnvironmentError("OUTPUT_BASE environment variable is not set.")
OUTPUT_BASE = Path(OUTPUT_BASE_ENV)

FEATURE_SET_PATTERNS = {
    "lon_deg+lat_deg+pm_lon_coslat_mas_yr+pm_lat_mas_yr": "Sky",
    "lon_deg+lat_deg+pm_lon_coslat_mas_yr+pm_lat_mas_yr+log_L_L_sol": "Sky+L",
    "x+y+z+vx+vy+vz": "Cartesian",
    "x+y+z+vx+vy+vz+log_L_L_sol": "Cartesian+L",
}

TARGET_PATTERNS = {
    "time": "age",
    "total_mass_within_2x_r_tidal": "total_mass",
}

MODEL_PARAM_SIZES = {
    "SS[4]": {"Sky": 125, "Sky+L": 153, "Cartesian": 181, "Cartesian+L": 209},
    "SS[8]": {"Sky": 249, "Sky+L": 305, "Cartesian": 361, "Cartesian+L": 417},
    "SS[12]": {"Sky": 373, "Sky+L": 457, "Cartesian": 541, "Cartesian+L": 625},
    "SS[24]": {"Sky": 745, "Sky+L": 913, "Cartesian": 1081, "Cartesian+L": 1249},
    "DS[4,4]": {"Sky": 45, "Sky+L": 49, "Cartesian": 53, "Cartesian+L": 57},
    "DS[8,8]": {"Sky": 121, "Sky+L": 129, "Cartesian": 137, "Cartesian+L": 145},
    "DS[12,12]": {"Sky": 229, "Sky+L": 241, "Cartesian": 253, "Cartesian+L": 265},
    "DS[24,24]": {"Sky": 745, "Sky+L": 769, "Cartesian": 793, "Cartesian+L": 817},
    "ST[4,2,1]": {"Sky": 401, "Sky+L": 413, "Cartesian": 425, "Cartesian+L": 437},
    "ST[4,4,1]": {"Sky": 401, "Sky+L": 413, "Cartesian": 425, "Cartesian+L": 437},
    "ST[4,2,2]": {"Sky": 597, "Sky+L": 609, "Cartesian": 621, "Cartesian+L": 633},
    "ST[4,4,2]": {"Sky": 597, "Sky+L": 609, "Cartesian": 621, "Cartesian+L": 633},
    "ST[6,2,1]": {"Sky": 781, "Sky+L": 799, "Cartesian": 817, "Cartesian+L": 835},
    "ST[6,2,2]": {"Sky": 1183, "Sky+L": 1201, "Cartesian": 1219, "Cartesian+L": 1237},
    "ST[6,3,1]": {"Sky": 781, "Sky+L": 799, "Cartesian": 817, "Cartesian+L": 835},
    "ST[6,3,2]": {"Sky": 1183, "Sky+L": 1201, "Cartesian": 1219, "Cartesian+L": 1237},
    "ST[6,6,1]": {"Sky": 781, "Sky+L": 799, "Cartesian": 817, "Cartesian+L": 835},
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
    r"-(?P<model_family>summary_stats|set_transformer|deep_sets)"
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


def _parse_experiment_config(experiment_name: str):
    if not (match := PARQUET_FILENAME_PATTERN.match(experiment_name)):
        raise ValueError(f"Unexpected experiment/parquet name: {experiment_name!r}")

    parts = match.groupdict()
    if (feature_set_key := parts["feature_set_key"]) not in FEATURE_SET_PATTERNS:
        raise ValueError(f"Unknown feature set key: {feature_set_key!r}")
    if (target_key := parts["target_key"]) not in TARGET_PATTERNS:
        raise ValueError(f"Unknown target key: {target_key!r}")

    model_hparams = parts["model_hparams"]
    match model_family := parts["model_family"]:
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
            raise ValueError(f"Unknown model: {model_family!r}")

    return {
        "model_family": model_family,
        "model_label": model_label,
        "feature_set": FEATURE_SET_PATTERNS[feature_set_key],
        "target": TARGET_PATTERNS[target_key],
        "n_star_per_sample": int(parts["n_star_per_sample"]),
        "n_sample_per_snapshot": int(parts["n_sample_per_snapshot"]),
        "drop_probability": _parse_fmt_float(parts["drop_probability"]),
        "drop_ratio_min": _parse_fmt_float(parts["drop_ratio_min"]),
        "drop_ratio_max": _parse_fmt_float(parts["drop_ratio_max"]),
        "batch_size": int(parts["batch_size"]),
        "learning_rate": _parse_fmt_float(parts["learning_rate"]),
        "weight_decay": _parse_fmt_float(parts["weight_decay"]),
        "seed": int(parts["seed"]) if parts.get("seed") else None,
        "auxiliary_label": parts.get("auxiliary_label", ""),
        "experiment_name": parts["experiment_name"],
    }


def _calc_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    pearson_r, pearson_p = pearsonr(y_true, y_pred)
    spearman_r, spearman_p = spearmanr(y_true, y_pred)

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "ev": float(explained_variance_score(y_true, y_pred)),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
    }


def _process_parquet_file(parquet_file: Path):
    try:
        experiment_config = _parse_experiment_config(parquet_file.stem)
    except ValueError as e:
        click.echo(f"Skipping {parquet_file!r}: {e}", err=True)
        return None

    result_df = pd.read_parquet(
        parquet_file,
        columns=[
            "snapshot_id",
            "target_scaled",
            "prediction_scaled",
            "target_physical",
            "prediction_physical",
        ],
    )

    target_scaled = result_df["target_scaled"].to_numpy(dtype=float, copy=False)
    pred_scaled = result_df["prediction_scaled"].to_numpy(dtype=float, copy=False)
    row_scaled_metrics = _calc_regression_metrics(
        y_true=target_scaled,
        y_pred=pred_scaled,
    )

    target_phys = result_df["target_physical"].to_numpy(dtype=float, copy=False)
    pred_phys = result_df["prediction_physical"].to_numpy(dtype=float, copy=False)
    row_phys_metrics = _calc_regression_metrics(
        y_true=target_phys,
        y_pred=pred_phys,
    )

    snapshot_grouped = result_df.groupby("snapshot_id", sort=False, observed=True)
    snapshot_summary_df = snapshot_grouped.agg(
        target_phys=("target_physical", "first"),
        pred_phys_mean=("prediction_physical", "mean"),
    )
    snapshot_summary_df["pred_phys_std"] = (
        snapshot_grouped["prediction_physical"]
        .std(ddof=0)
        .to_numpy(dtype=float, copy=False)
    )

    snapshot_target_phys = snapshot_summary_df["target_phys"].to_numpy(
        dtype=float, copy=False
    )
    snapshot_pred_phys_mean = snapshot_summary_df["pred_phys_mean"].to_numpy(
        dtype=float, copy=False
    )
    snapshot_pred_phys_std = snapshot_summary_df["pred_phys_std"].to_numpy(
        dtype=float, copy=False
    )
    snapshot_error_signed = snapshot_pred_phys_mean - snapshot_target_phys
    snapshot_error_abs = np.abs(snapshot_pred_phys_mean - snapshot_target_phys)

    snapshot_mean_phys_metrics = _calc_regression_metrics(
        y_true=snapshot_target_phys,
        y_pred=snapshot_pred_phys_mean,
    )
    snapshot_error_signed_mean = float(snapshot_error_signed.mean())
    snapshot_error_signed_median = float(np.median(snapshot_error_signed))
    snapshot_error_abs_med = float(np.median(snapshot_error_abs))
    snapshot_error_abs_q90 = float(np.quantile(snapshot_error_abs, 0.9))

    within_snapshot_phys_metrics = {
        "within_snapshot_phys_pred_std_mean": float(snapshot_pred_phys_std.mean()),
        "within_snapshot_phys_pred_std_q90": float(
            np.quantile(snapshot_pred_phys_std, 0.9)
        ),
        "within_snapshot_phys_std_ae_corr": float(
            pearsonr(snapshot_pred_phys_std, snapshot_error_abs).statistic
        ),
    }

    return {
        **experiment_config,
        "n_total_samples": int(len(result_df)),
        "n_snapshots": int(len(snapshot_summary_df)),
        **{f"row_scaled_{k}": v for k, v in row_scaled_metrics.items()},
        **{f"row_phys_{k}": v for k, v in row_phys_metrics.items()},
        **{f"snapshot_mean_phys_{k}": v for k, v in snapshot_mean_phys_metrics.items()},
        "snapshot_mean_phys_se_mean": snapshot_error_signed_mean,
        "snapshot_mean_phys_se_median": snapshot_error_signed_median,
        "snapshot_mean_phys_ae_median": snapshot_error_abs_med,
        "snapshot_mean_phys_ae_q90": snapshot_error_abs_q90,
        **within_snapshot_phys_metrics,
    }


def _calc_model_score(
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    _METRIC_WEIGHTS: dict[str, float] = {
        "snapshot_mean_phys_ae_median": 0.6,
        "snapshot_mean_phys_ae_q90": 0.4,
    }
    if (weight_sum := sum(_METRIC_WEIGHTS.values())) <= 0:
        raise ValueError("Metric weights must sum to a positive number.")
    norm_weights = {m: w / weight_sum for m, w in _METRIC_WEIGHTS.items()}

    base_df = (
        summary_df.groupby(
            [
                "model_label",
                summary_df["feature_set"] + "+" + summary_df["target"],
            ]
        )[list(norm_weights.keys())]
        .median()
        .rename_axis(index=["model_label", "task"])
        .reset_index(level="task")
    )

    task_dfs = [
        task_df.assign(
            **{
                f"{task}_{m}_score": (task_df[m].rank(pct=True, ascending=False))
                for m in norm_weights
            },
            **{
                f"{task}_score": sum(
                    task_df[m].rank(pct=True, ascending=False) * w
                    for m, w in norm_weights.items()
                )
            },
        ).drop(columns=["task", *norm_weights.keys()])
        for task, task_df in base_df.groupby("task")
    ]

    return (
        pd.concat(task_dfs, axis=1)
        .assign(
            overall_score=lambda df: df.filter(regex=r"^[^+]+\+[^+]+_score$").mean(
                axis=1
            )
        )
        .sort_values("overall_score", ascending=False)
        .reset_index()
    )


@click.command()
@click.option(
    "--result-root",
    "result_root",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Root directory containing experiment results.",
)
@click.option(
    "--workers",
    "n_workers",
    type=click.IntRange(min=1),
    default=4,
    show_default=True,
    help="Number of parallel workers to use for processing parquet files.",
)
@click.option(
    "--subfolder",
    type=str,
    default="",
    help="Subfolder within summary output directory.",
    show_default=True,
)
@click.option(
    "--skip-confirm",
    is_flag=True,
    default=False,
    help="Skip confirmation prompt if summary file already exists.",
)
def main(
    result_root: Path | str,
    n_workers: int,
    subfolder: str,
    skip_confirm: bool,
):
    output_dir = (
        (OUTPUT_BASE / "summary" / subfolder)
        if subfolder
        else (OUTPUT_BASE / "summary")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (parquet_files := sorted(Path(result_root).rglob("*.parquet"))):
        raise FileNotFoundError(f"No parquet files found in {result_root!r}")

    summary_csv_path = output_dir / "summary.csv"
    if summary_csv_path.exists():
        if not skip_confirm:
            click.confirm(
                f"Summary file {summary_csv_path} already exists. Overwrite?",
                abort=True,
            )
        click.secho(
            f"Warning: Summary file {summary_csv_path} will be overwritten.",
            fg="yellow",
            err=True,
        )
    records = [
        record
        for record in tqdm(
            Parallel(n_jobs=n_workers, return_as="generator")(
                delayed(_process_parquet_file)(file) for file in parquet_files
            ),
            total=len(parquet_files),
            desc="Processing PARQUET files",
        )
        if record is not None
    ]
    summary_df = (
        pd.DataFrame(records)
        .sort_values(
            by=[
                "model_family",
                "model_label",
                "feature_set",
                "target",
                "seed",
            ]
        )
        .reset_index(drop=True)
    )
    summary_df.to_csv(summary_csv_path, index=False)
    click.echo(f"Summary saved to {output_dir / 'summary.csv'}")

    if (score_csv_path := output_dir / "ranking.csv").exists():
        if not skip_confirm:
            click.confirm(
                f"Score file {score_csv_path} already exists. Overwrite?",
                abort=True,
            )
        click.secho(
            f"Warning: Score file {score_csv_path} will be overwritten.",
            fg="yellow",
            err=True,
        )
    score_df = _calc_model_score(summary_df)
    score_df.to_csv(score_csv_path, index=False)
    click.echo(f"Score saved to {score_csv_path}")


if __name__ == "__main__":
    main()
