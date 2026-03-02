"""
collect_results.py
------------------
Scan a folder for experiment subdirectories, detect the algorithm subfolder
automatically, and export every metric from analysis.py to its own CSV.

Usage
-----
  python collect_results.py <folder> [options]

  # All metrics for all experiments in 260302_full
  python collect_results.py _experiments/260302_full

  # Only two metrics, custom output dir
  python collect_results.py _experiments/260302_full \\
      --metrics completion_time solution_count \\
      --output results/260302

  # Just list what was detected, don't compute anything
  python collect_results.py _experiments/260302_full --list
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

# analysis.py lives in the same directory as this script
sys.path.insert(0, str(Path(__file__).parent))
from analysis import Experiment, _METRICS, inst_list


# ---------------------------------------------------------------------------
# Experiment detection
# ---------------------------------------------------------------------------

def _find_alg_subdir(exp_dir: str) -> str | None:
    """
    Return the name of the subdirectory inside *exp_dir* that holds instance
    result folders (S1, M1, L1, …).  Returns None if nothing is found.
    """
    try:
        entries = [e for e in os.scandir(exp_dir) if e.is_dir()]
    except PermissionError:
        return None
    for entry in entries:
        for inst in inst_list:
            if os.path.isdir(os.path.join(entry.path, inst)):
                return entry.name
    return None


def scan_experiments(folder: str) -> list[tuple[str, str, str]]:
    """
    Return a sorted list of (exp_name, exp_dir, alg) for every detected
    experiment subfolder inside *folder*.
    """
    folder = os.path.abspath(folder)
    results = []
    try:
        entries = sorted(
            (e for e in os.scandir(folder) if e.is_dir()),
            key=lambda e: e.name,
        )
    except FileNotFoundError:
        print(f"Error: folder not found: {folder}", file=sys.stderr)
        sys.exit(1)

    for entry in entries:
        alg = _find_alg_subdir(entry.path)
        if alg is not None:
            results.append((entry.name, entry.path, alg))
    return results


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def read_alg_config(exp_dir: str) -> dict:
    """Load alg_config.json from *exp_dir*, returning {} on any error."""
    config_path = os.path.join(exp_dir, "alg_config.json")
    try:
        with open(config_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


# ---------------------------------------------------------------------------
# Metric collection
# ---------------------------------------------------------------------------

def collect_metric(
    experiments: list[tuple[str, str, str]],
    metric: str,
) -> pd.DataFrame | None:
    """
    Compute *metric* for every experiment and return a concatenated DataFrame
    with 'experiment' and 'max_length' label columns, or None if all failed.
    """
    frames = []
    for exp_name, exp_dir, alg in experiments:
        try:
            config = read_alg_config(exp_dir)
            exp = Experiment(exp_dir, alg, ["experiment"], [exp_name])
            df = exp.get_agg_table(metric)
            df["max_length"] = config.get("max_length", None)
            frames.append(df)
        except Exception as exc:
            print(
                f"  [skip] {exp_name}/{alg}/{metric}: {exc}",
                file=sys.stderr,
            )
    if not frames:
        return None
    return pd.concat(frames, axis=0, ignore_index=True)


# ---------------------------------------------------------------------------
# Summary tables (horizontal join across metrics)
# ---------------------------------------------------------------------------

def _str_inst(df: pd.DataFrame) -> pd.DataFrame:
    """Cast 'inst' to str so Categorical dtype doesn't break merges."""
    df = df.copy()
    df["inst"] = df["inst"].astype(str)
    return df


def build_summary_table(metric_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    """
    Join metric DataFrames into a wide per-(experiment, inst, level) table.

    Level-aware metrics — joined on (experiment, inst, level):
      completion_time   → completion_time
      solution_count    → sol
      count_cuts        → pivoted by cut_type  →  count_cuts_<TYPE>

    Per-inst metrics — joined on (experiment, inst), broadcast across levels:
      build_time        → build_time  (+ max_length)
      max_level         → max_level
      avg_cuts          → pivoted by cut_type  →  avg_cuts_<TYPE>
      computation_time  → pivoted by step       →  time_<STEP>
    """
    if not metric_dfs:
        return None

    _LEVEL_KEY = ["experiment", "inst", "level"]
    _INST_KEY  = ["experiment", "inst"]

    result: pd.DataFrame | None = None

    def _merge(left, right, key):
        if left is None:
            return right.copy()
        return left.merge(right, on=key, how="outer")

    # -- Level-aware metrics (join key includes level) -----------------------

    # completion_time: (experiment, inst, level, completion_time)
    if "completion_time" in metric_dfs:
        df = _str_inst(metric_dfs["completion_time"])[_LEVEL_KEY + ["completion_time"]]
        result = _merge(result, df, _LEVEL_KEY)

    # solution_count: (experiment, inst, level, sol)
    if "solution_count" in metric_dfs:
        df = _str_inst(metric_dfs["solution_count"])[_LEVEL_KEY + ["sol"]]
        result = _merge(result, df, _LEVEL_KEY)

    # count_cuts: (experiment, inst, level, cut_type, count_cuts)
    # → pivot by cut_type → count_cuts_<TYPE>
    if "count_cuts" in metric_dfs:
        df = _str_inst(metric_dfs["count_cuts"])
        if not df.empty:
            df = (
                df.pivot_table(index=_LEVEL_KEY, columns="cut_type", values="count_cuts")
                .rename(columns=lambda c: f"count_cuts_{c}")
                .reset_index()
            )
            df.columns.name = None
            result = _merge(result, df, _LEVEL_KEY)

    # -- Per-inst metrics (broadcast across levels) --------------------------

    # build_time: (experiment, inst, build_time, max_length)
    if "build_time" in metric_dfs:
        df = _str_inst(metric_dfs["build_time"])
        cols = [c for c in [*_INST_KEY, "max_length", "build_time"] if c in df.columns]
        result = _merge(result, df[cols], _INST_KEY)

    # max_level: (experiment, inst, max_level)
    if "max_level" in metric_dfs:
        df = _str_inst(metric_dfs["max_level"])[_INST_KEY + ["max_level"]]
        result = _merge(result, df, _INST_KEY)

    # avg_cuts: pivot by cut_type → avg_cuts_<TYPE>
    if "avg_cuts" in metric_dfs:
        df = _str_inst(metric_dfs["avg_cuts"])
        if not df.empty:
            df = (
                df.pivot_table(index=_INST_KEY, columns="cut_type", values="num_literals")
                .rename(columns=lambda c: f"avg_cuts_{c}")
                .reset_index()
            )
            df.columns.name = None
            result = _merge(result, df, _INST_KEY)

    # computation_time: pivot by step → time_<STEP>
    if "computation_time" in metric_dfs:
        df = _str_inst(metric_dfs["computation_time"])
        if not df.empty:
            df = (
                df.pivot_table(
                    index=_INST_KEY, columns="step", values="solve_time", aggfunc="sum"
                )
                .rename(columns=lambda c: f"time_{c}")
                .reset_index()
            )
            df.columns.name = None
            result = _merge(result, df, _INST_KEY)

    if result is None:
        return None

    # Backfill max_length from any metric that carries it (in case build_time
    # was not collected).
    if "max_length" not in result.columns:
        for df in metric_dfs.values():
            if "max_length" in df.columns:
                ml = _str_inst(df)[_INST_KEY + ["max_length"]].drop_duplicates()
                result = result.merge(ml, on=_INST_KEY, how="left")
                break

    return result


def build_per_inst_table(metric_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    """
    Join metric DataFrames into a wide per-(experiment, inst) table by
    aggregating away the level dimension.

    Aggregations applied:
      completion_time   → value at the highest level (total wall-clock time)
      solution_count    → sum over levels  →  total_solutions
      count_cuts        → sum over levels, pivoted by cut_type  →  count_cuts_<TYPE>
      build_time        → direct  (+ max_length)
      max_level         → direct
      avg_cuts          → pivoted by cut_type  →  avg_cuts_<TYPE>
      computation_time  → pivoted by step      →  time_<STEP>
    """
    if not metric_dfs:
        return None

    _INST_KEY = ["experiment", "inst"]
    result: pd.DataFrame | None = None

    def _merge(left, right):
        if left is None:
            return right.copy()
        return left.merge(right, on=_INST_KEY, how="outer")

    # build_time: (experiment, inst, build_time, max_length)
    if "build_time" in metric_dfs:
        df = _str_inst(metric_dfs["build_time"])
        cols = [c for c in [*_INST_KEY, "max_length", "build_time"] if c in df.columns]
        result = _merge(result, df[cols])

    # max_level: (experiment, inst, max_level)
    if "max_level" in metric_dfs:
        df = _str_inst(metric_dfs["max_level"])[_INST_KEY + ["max_level"]]
        result = _merge(result, df)

    # completion_time: value at the highest level per inst
    if "completion_time" in metric_dfs:
        df = _str_inst(metric_dfs["completion_time"])
        df = (
            df.loc[df.groupby(_INST_KEY)["level"].idxmax()]
            [_INST_KEY + ["completion_time"]]
        )
        result = _merge(result, df)

    # solution_count: sum sol across all levels
    if "solution_count" in metric_dfs:
        df = (
            _str_inst(metric_dfs["solution_count"])
            .groupby(_INST_KEY, as_index=False)["sol"]
            .sum()
            .rename(columns={"sol": "total_solutions"})
        )
        result = _merge(result, df)

    # avg_cuts: pivot by cut_type → avg_cuts_<TYPE>
    if "avg_cuts" in metric_dfs:
        df = _str_inst(metric_dfs["avg_cuts"])
        if not df.empty:
            df = (
                df.pivot_table(index=_INST_KEY, columns="cut_type", values="num_literals")
                .rename(columns=lambda c: f"avg_cuts_{c}")
                .reset_index()
            )
            df.columns.name = None
            result = _merge(result, df)

    # count_cuts: sum over levels, then pivot by cut_type → count_cuts_<TYPE>
    if "count_cuts" in metric_dfs:
        df = _str_inst(metric_dfs["count_cuts"])
        if not df.empty:
            df = (
                df.groupby(_INST_KEY + ["cut_type"], as_index=False)["count_cuts"]
                .sum()
                .pivot_table(index=_INST_KEY, columns="cut_type", values="count_cuts")
                .rename(columns=lambda c: f"count_cuts_{c}")
                .reset_index()
            )
            df.columns.name = None
            result = _merge(result, df)

    # computation_time: pivot by step → time_<STEP>
    if "computation_time" in metric_dfs:
        df = _str_inst(metric_dfs["computation_time"])
        if not df.empty:
            df = (
                df.pivot_table(
                    index=_INST_KEY, columns="step", values="solve_time", aggfunc="sum"
                )
                .rename(columns=lambda c: f"time_{c}")
                .reset_index()
            )
            df.columns.name = None
            result = _merge(result, df)

    if result is None:
        return None

    if "max_length" not in result.columns:
        for df in metric_dfs.values():
            if "max_length" in df.columns:
                ml = _str_inst(df)[_INST_KEY + ["max_length"]].drop_duplicates()
                result = result.merge(ml, on=_INST_KEY, how="left")
                break

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a folder for experiment subdirectories and export all "
            "metrics from analysis.py to CSV files in an output directory."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "folder",
        help="Folder that contains experiment subdirectories.",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        metavar="DIR",
        help="Output directory for CSV files (default: <folder>/results).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=_METRICS,
        choices=_METRICS,
        metavar="METRIC",
        help=(
            "Metrics to compute. Choices: "
            + ", ".join(_METRICS)
            + ". Default: all."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print detected experiments and exit without computing anything.",
    )

    args = parser.parse_args()

    # --- Detect experiments -------------------------------------------------
    experiments = scan_experiments(args.folder)
    if not experiments:
        print(
            f"No experiments detected in '{args.folder}'.\n"
            "A subdirectory is considered an experiment when it contains an "
            "algorithm subfolder (e.g. 'benders', 'MibS') that itself holds "
            "instance directories (S1, M1, L1, …).",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Detected {len(experiments)} experiment(s) in '{args.folder}':")
    for exp_name, exp_dir, alg in experiments:
        cfg = read_alg_config(exp_dir)
        max_length = cfg.get("max_length", "?")
        print(f"  {exp_name:<25}  alg: {alg:<10}  max_length: {max_length}")

    if args.list:
        return

    # --- Compute and write metrics ------------------------------------------
    output_dir = args.output or os.path.join(args.folder, "_results")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    print(f"Computing {len(args.metrics)} metric(s):\n")

    written = []
    metric_dfs: dict[str, pd.DataFrame] = {}

    for metric in args.metrics:
        print(f"  {metric:<30}", end="", flush=True)
        df = collect_metric(experiments, metric)
        if df is not None and not df.empty:
            metric_dfs[metric] = df
            out_path = os.path.join(output_dir, f"{metric}.csv")
            df.to_csv(out_path, index=False)
            print(f"{len(df):>6} rows  →  {out_path}")
            written.append(out_path)
        else:
            print("  (no data)")

    # --- Build and write summary tables -------------------------------------
    for label, builder, fname in [
        ("summary (per level)",    build_summary_table,   "summary.csv"),
        ("summary (per inst)",     build_per_inst_table,  "summary_per_inst.csv"),
    ]:
        print(f"\n  {label:<30}", end="", flush=True)
        df = builder(metric_dfs)
        if df is not None and not df.empty:
            out_path = os.path.join(output_dir, fname)
            df.to_csv(out_path, index=False)
            print(f"{len(df):>6} rows  →  {out_path}")
            written.append(out_path)
        else:
            print("  (no data)")

    print(f"\nDone. {len(written)} CSV file(s) written to '{output_dir}'.")


if __name__ == "__main__":
    main()
