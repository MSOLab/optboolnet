"""
collect_results.py
------------------
Scan a folder for experiment subdirectories, detect the algorithm subfolder
automatically, and export summary + aggregate tables.

Usage
-----
  python collect_results.py <folder> [options]

  # Build summaries + aggregate tables for all metrics in 260302_full
  python collect_results.py _experiments/260302_full

  # Build summaries from only two metrics, custom output dir
  python collect_results.py _experiments/260302_full \\
      --metrics completion_time solution_count \\
      --output results/260302

  # Build aggregate tables only from existing summary CSVs
  python collect_results.py _experiments/260302_full/_results --from-summary

  # Just list what was detected, don't compute anything
  python collect_results.py _experiments/260302_full --list
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# analysis.py lives in the same directory as this script
sys.path.insert(0, str(Path(__file__).parent))
from analysis import Experiment, _METRICS, inst_list

INST_ORDER = ["S1", "S2", "S3", "S4", "M1", "M2", "M3", "L1", "L2", "L3", "L4"]
LABEL_MAP: dict[str, str] = {}
_CUT_LABEL_MAP: dict[str, str] = {
    "ATTRACTOR_CUT": "AT cut",
    "TRAP_SPACE_CUT": "TS cut",
    "MINIMALITY": "MIN cut",
    "NO_GOOD_MASTER": "No-good(master)",
    "NO_GOOD_LOWER_LEVEL": "No-good(lower)",
}


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
        try:
            child_names = {c.name for c in os.scandir(entry.path) if c.is_dir()}
        except PermissionError:
            continue
        if any(inst in child_names for inst in inst_list):
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


@dataclass
class ExperimentContext:
    exp_name: str
    exp_dir: str
    alg: str
    config: dict
    exp: Experiment


def build_experiment_contexts(experiments: list[tuple[str, str, str]]) -> list[ExperimentContext]:
    """
    Build Experiment objects once and keep config metadata cached for reuse
    across all metrics.
    """
    contexts: list[ExperimentContext] = []
    for exp_name, exp_dir, alg in experiments:
        config = read_alg_config(exp_dir)
        try:
            exp = Experiment(exp_dir, alg, ["experiment"], [exp_name])
        except Exception as exc:
            print(f"  [skip] {exp_name}/{alg}: {exc}", file=sys.stderr)
            continue
        contexts.append(
            ExperimentContext(
                exp_name=exp_name,
                exp_dir=exp_dir,
                alg=alg,
                config=config,
                exp=exp,
            )
        )
    return contexts


# ---------------------------------------------------------------------------
# Metric collection
# ---------------------------------------------------------------------------

def collect_metrics(
    contexts: list[ExperimentContext],
    metrics: list[str],
) -> dict[str, pd.DataFrame]:
    """
    Compute all requested metrics in one pass per experiment context.
    """
    frames_by_metric: dict[str, list[pd.DataFrame]] = {metric: [] for metric in metrics}
    for ctx in contexts:
        try:
            metric_tables = ctx.exp.get_agg_tables(metrics)
        except Exception as exc:
            print(f"  [skip] {ctx.exp_name}/{ctx.alg}: {exc}", file=sys.stderr)
            continue

        for metric in metrics:
            df = metric_tables.get(metric)
            if df is None or df.empty:
                continue
            df = df.copy()
            df["max_length"] = ctx.config.get("max_length", None)
            df["max_control_size"] = ctx.config.get("max_control_size", None)
            frames_by_metric[metric].append(df)

    return {
        metric: pd.concat(frames, axis=0, ignore_index=True)
        for metric, frames in frames_by_metric.items()
        if frames
    }


# ---------------------------------------------------------------------------
# Summary tables (horizontal join across metrics)
# ---------------------------------------------------------------------------

def _str_inst(df: pd.DataFrame) -> pd.DataFrame:
    """Cast 'inst' to str so Categorical dtype doesn't break merges."""
    df = df.copy()
    df["inst"] = df["inst"].astype(str)
    return df


def _to_indexed(df: pd.DataFrame, key_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    cols = key_cols + [c for c in value_cols if c in df.columns]
    out = df[cols].drop_duplicates(subset=key_cols).set_index(key_cols)
    out.index = out.index.set_names(key_cols)
    return out


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

    level_parts: list[pd.DataFrame] = []
    inst_parts: list[pd.DataFrame] = []

    # -- Level-aware metrics (join key includes level) -----------------------

    # completion_time: (experiment, inst, level, completion_time, level_finished)
    if "completion_time" in metric_dfs:
        df = _str_inst(metric_dfs["completion_time"])
        level_parts.append(_to_indexed(df, _LEVEL_KEY, ["completion_time", "level_finished"]))

    # solution_count: (experiment, inst, level, sol)
    if "solution_count" in metric_dfs:
        df = _str_inst(metric_dfs["solution_count"])
        level_parts.append(_to_indexed(df, _LEVEL_KEY, ["sol"]))

    # count_cuts: (experiment, inst, level, cut_type, count_cuts)
    # → pivot by cut_type → count_cuts_<TYPE>
    if "count_cuts" in metric_dfs:
        df = _str_inst(metric_dfs["count_cuts"])
        if not df.empty:
            df = (
                df.pivot_table(index=_LEVEL_KEY, columns="cut_type", values="count_cuts")
                .rename(columns=lambda c: f"count_cuts_{c}")
            )
            df.columns.name = None
            level_parts.append(df)

    # -- Per-inst metrics (broadcast across levels) --------------------------

    # build_time: (experiment, inst, build_time, max_length, max_control_size)
    if "build_time" in metric_dfs:
        df = _str_inst(metric_dfs["build_time"])
        inst_parts.append(_to_indexed(df, _INST_KEY, ["max_length", "max_control_size", "build_time"]))

    # max_level: (experiment, inst, max_level)
    if "max_level" in metric_dfs:
        df = _str_inst(metric_dfs["max_level"])
        inst_parts.append(_to_indexed(df, _INST_KEY, ["max_level"]))

    # avg_cuts: pivot by cut_type → avg_cuts_<TYPE>
    if "avg_cuts" in metric_dfs:
        df = _str_inst(metric_dfs["avg_cuts"])
        if not df.empty:
            df = (
                df.pivot_table(index=_INST_KEY, columns="cut_type", values="num_literals")
                .rename(columns=lambda c: f"avg_cuts_{c}")
            )
            df.columns.name = None
            inst_parts.append(df)

    # computation_time: pivot by step → time_<STEP>
    if "computation_time" in metric_dfs:
        df = _str_inst(metric_dfs["computation_time"])
        if not df.empty:
            df = (
                df.pivot_table(
                    index=_INST_KEY, columns="step", values="solve_time", aggfunc="sum"
                )
                .rename(columns=lambda c: f"time_{c}")
            )
            df.columns.name = None
            inst_parts.append(df)
    if not level_parts and not inst_parts:
        return None

    result_idx: pd.DataFrame | None = None
    if level_parts:
        result_idx = pd.concat(level_parts, axis=1, join="outer")
    if inst_parts:
        inst_idx = pd.concat(inst_parts, axis=1, join="outer")
        if result_idx is None:
            result_idx = inst_idx
        else:
            result_idx = result_idx.join(inst_idx, on=_INST_KEY, how="left")

    result = result_idx.reset_index()

    # Backfill max_length / max_control_size from any metric that carries them
    # (in case build_time was not collected).
    if "max_length" not in result.columns:
        for df in metric_dfs.values():
            if "max_length" in df.columns:
                ml = _to_indexed(_str_inst(df), _INST_KEY, ["max_length"])
                result = result.join(ml, on=_INST_KEY, how="left")
                break
    if "max_control_size" not in result.columns:
        for df in metric_dfs.values():
            if "max_control_size" in df.columns:
                mcs = _to_indexed(_str_inst(df), _INST_KEY, ["max_control_size"])
                result = result.join(mcs, on=_INST_KEY, how="left")
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
    inst_parts: list[pd.DataFrame] = []

    # build_time: (experiment, inst, build_time, max_length, max_control_size)
    if "build_time" in metric_dfs:
        df = _str_inst(metric_dfs["build_time"])
        inst_parts.append(_to_indexed(df, _INST_KEY, ["max_length", "max_control_size", "build_time"]))

    # max_level: (experiment, inst, max_level)
    if "max_level" in metric_dfs:
        df = _str_inst(metric_dfs["max_level"])
        inst_parts.append(_to_indexed(df, _INST_KEY, ["max_level"]))

    # completion_time: value at the highest *finished* level per inst
    if "completion_time" in metric_dfs:
        df = _str_inst(metric_dfs["completion_time"])
        if "level_finished" in df.columns:
            df = df[df["level_finished"]]
        if not df.empty:
            df = (
                df.loc[df.groupby(_INST_KEY)["level"].idxmax()]
                [_INST_KEY + ["completion_time"]]
            )
            inst_parts.append(_to_indexed(df, _INST_KEY, ["completion_time"]))

    # solution_count: sum sol across all levels
    if "solution_count" in metric_dfs:
        df = (
            _str_inst(metric_dfs["solution_count"])
            .groupby(_INST_KEY, as_index=False)["sol"]
            .sum()
            .rename(columns={"sol": "total_solutions"})
        )
        inst_parts.append(_to_indexed(df, _INST_KEY, ["total_solutions"]))

    # avg_cuts: pivot by cut_type → avg_cuts_<TYPE>
    if "avg_cuts" in metric_dfs:
        df = _str_inst(metric_dfs["avg_cuts"])
        if not df.empty:
            df = (
                df.pivot_table(index=_INST_KEY, columns="cut_type", values="num_literals")
                .rename(columns=lambda c: f"avg_cuts_{c}")
            )
            df.columns.name = None
            inst_parts.append(df)

    # count_cuts: sum over levels, then pivot by cut_type → count_cuts_<TYPE>
    if "count_cuts" in metric_dfs:
        df = _str_inst(metric_dfs["count_cuts"])
        if not df.empty:
            df = (
                df.groupby(_INST_KEY + ["cut_type"], as_index=False)["count_cuts"]
                .sum()
                .pivot_table(index=_INST_KEY, columns="cut_type", values="count_cuts")
                .rename(columns=lambda c: f"count_cuts_{c}")
            )
            df.columns.name = None
            inst_parts.append(df)

    # computation_time: pivot by step → time_<STEP>
    if "computation_time" in metric_dfs:
        df = _str_inst(metric_dfs["computation_time"])
        if not df.empty:
            df = (
                df.pivot_table(
                    index=_INST_KEY, columns="step", values="solve_time", aggfunc="sum"
                )
                .rename(columns=lambda c: f"time_{c}")
            )
            df.columns.name = None
            inst_parts.append(df)
    if not inst_parts:
        return None

    result = pd.concat(inst_parts, axis=1, join="outer").reset_index()

    if "max_length" not in result.columns:
        for df in metric_dfs.values():
            if "max_length" in df.columns:
                ml = _to_indexed(_str_inst(df), _INST_KEY, ["max_length"])
                result = result.join(ml, on=_INST_KEY, how="left")
                break
    if "max_control_size" not in result.columns:
        for df in metric_dfs.values():
            if "max_control_size" in df.columns:
                mcs = _to_indexed(_str_inst(df), _INST_KEY, ["max_control_size"])
                result = result.join(mcs, on=_INST_KEY, how="left")
                break

    return result


# ---------------------------------------------------------------------------
# Aggregate tables
# ---------------------------------------------------------------------------

def make_solution_count(df: pd.DataFrame) -> pd.DataFrame:
    inst_present = [c for c in INST_ORDER if c in df["inst"].values]
    tbl = (
        df.pivot_table(
            index=["max_length", "experiment"],
            columns="inst",
            values="total_solutions",
            aggfunc="first",
        )
        .reindex(columns=inst_present)
        .sort_index()
    )
    tbl.columns.name = None
    return tbl


def make_solution_count_bold(df: pd.DataFrame, ct: pd.DataFrame) -> pd.DataFrame:
    base = make_solution_count(df)
    if "level_finished" not in ct.columns:
        return base

    keys = ["experiment", "inst"]
    finished = ct[ct["level_finished"] == True].copy()
    if finished.empty:
        finished_pairs = set()
    else:
        finished_max = (
            finished.groupby(keys, as_index=False)["level"].max()
            .rename(columns={"level": "finished_level"})
        )
        finished_max["finished_level"] = pd.to_numeric(
            finished_max["finished_level"], errors="coerce"
        )

        if "max_control_size" in df.columns:
            req = (
                df[keys + ["max_control_size"]]
                .drop_duplicates()
                .rename(columns={"max_control_size": "required_level"})
            )
            req["required_level"] = pd.to_numeric(req["required_level"], errors="coerce")
        else:
            req = (
                ct.groupby("experiment", as_index=False)["level"].max()
                .rename(columns={"level": "required_level"})
            )
            req = finished_max[["experiment", "inst"]].merge(req, on="experiment", how="left")

        chk = finished_max.merge(req, on=keys, how="left")
        chk = chk[chk["required_level"].notna()]
        finished_pairs = set(
            zip(
                chk.loc[chk["finished_level"] >= chk["required_level"], "experiment"],
                chk.loc[chk["finished_level"] >= chk["required_level"], "inst"],
            )
        )

    out = base.copy().astype(object)
    for (_, experiment), row in base.iterrows():
        for inst in base.columns:
            val = row[inst]
            if pd.isna(val):
                out.loc[(_, experiment), inst] = ""
            else:
                cell = str(int(round(val)))
                if (experiment, inst) in finished_pairs:
                    cell = f"{cell}*"
                out.loc[(_, experiment), inst] = cell
    return out


def make_completion_time(ct: pd.DataFrame, variant: str | None) -> pd.DataFrame:
    ct = ct.copy().dropna(subset=["experiment", "inst", "level"])
    if variant:
        ct = ct[ct["experiment"].str.endswith(variant)]
    ct["label"] = ct["experiment"].map(lambda e: LABEL_MAP.get(e, e))
    if "level_finished" in ct.columns:
        ct["completion_time"] = ct["completion_time"].where(ct["level_finished"])

    inst_present = [i for i in INST_ORDER if i in ct["inst"].values]
    labels_ordered = sorted(ct["label"].unique())
    ml_values = sorted(ct["max_length"].unique())
    tbl = ct.pivot_table(
        index="level",
        columns=["max_length", "inst", "label"],
        values="completion_time",
    )
    ordered_cols = [
        (ml, inst, label)
        for ml in ml_values
        for inst in inst_present
        for label in labels_ordered
        if (ml, inst, label) in tbl.columns
    ]
    tbl = tbl.reindex(columns=ordered_cols).round(1)
    tbl.columns.names = ["T_max", "", ""]
    tbl.index.name = "lambda"
    return tbl


def _cut_columns(spi: pd.DataFrame, col_prefix: str) -> list[str]:
    return sorted(c for c in spi.columns if c.startswith(f"{col_prefix}_"))


def _cut_display_label(col_name: str, col_prefix: str) -> str:
    raw = col_name[len(col_prefix) + 1 :]
    return _CUT_LABEL_MAP.get(raw, raw)


def _cuts_for_ml(
    spi_ml: pd.DataFrame, cut_cols: list[str], section: str, col_prefix: str
) -> pd.DataFrame | None:
    inst_present = [i for i in INST_ORDER if i in spi_ml["inst"].values]
    rows = []
    for alg in sorted(spi_ml["alg"].dropna().unique()):
        spi_alg = spi_ml[spi_ml["alg"] == alg]
        for col in cut_cols:
            if col not in spi_alg.columns:
                continue
            sub = spi_alg.groupby("inst")[col].mean().reindex(inst_present)
            if sub.notna().sum() == 0:
                continue
            sub.name = (section, alg, _cut_display_label(col, col_prefix))
            rows.append(sub)
    if not rows:
        return None
    tbl = pd.concat(rows, axis=1).T
    tbl.index = pd.MultiIndex.from_tuples(tbl.index, names=["Section", "Algorithm", "Cuts"])
    return tbl


def make_cuts_table(spi: pd.DataFrame, variant: str | None, col_prefix: str) -> pd.DataFrame:
    spi = spi.copy()
    if variant:
        spi = spi[spi["experiment"].str.endswith(variant)]
    spi["alg"] = spi["experiment"].str.split("_").str[0]

    cut_cols = _cut_columns(spi, col_prefix)
    if not cut_cols:
        return pd.DataFrame()
    section = "Total # of cuts" if col_prefix == "count_cuts" else "Avg. literals in a cut"
    parts: dict[int, pd.DataFrame] = {}
    for ml in sorted(spi["max_length"].unique()):
        tbl_ml = _cuts_for_ml(spi[spi["max_length"] == ml], cut_cols, section, col_prefix)
        if tbl_ml is not None:
            parts[ml] = tbl_ml
    if not parts:
        return pd.DataFrame()
    tbl = pd.concat(parts, axis=1)
    tbl.columns.names = ["T_max", "inst"]
    return tbl.apply(pd.to_numeric, errors="coerce").round(1)


def build_aggregate_tables(
    summary_level: pd.DataFrame | None,
    summary_per_inst: pd.DataFrame | None,
    variant: str | None,
) -> dict[str, pd.DataFrame]:
    if summary_level is None or summary_per_inst is None:
        return {}
    if summary_level.empty or summary_per_inst.empty:
        return {}
    return {
        "agg_solution_count": make_solution_count(summary_per_inst),
        "agg_solution_count_bold": make_solution_count_bold(summary_per_inst, summary_level),
        "agg_completion_time": make_completion_time(summary_level, variant),
        "agg_cuts_total": make_cuts_table(summary_per_inst, variant, "count_cuts"),
        "agg_cuts_avg": make_cuts_table(summary_per_inst, variant, "avg_cuts"),
    }


def write_aggregate_tables(
    output_dir: str,
    summary_level: pd.DataFrame | None,
    summary_per_inst: pd.DataFrame | None,
    variant: str | None,
) -> list[str]:
    written: list[str] = []
    agg_tables = build_aggregate_tables(summary_level, summary_per_inst, variant)
    for name, tbl in agg_tables.items():
        print(f"\n  {name:<30}", end="", flush=True)
        if tbl is None or tbl.empty:
            print("  (no data)")
            continue
        out_path = os.path.join(output_dir, f"{name}.csv")
        tbl.to_csv(out_path)
        print(f"{str(tbl.shape):>12}  →  {out_path}")
        written.append(out_path)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build summary and aggregate CSV tables from experiment folders "
            "or from existing summary CSV files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "folder",
        help=(
            "Input folder. Default mode: experiment root folder. "
            "With --from-summary: folder containing summary.csv and summary_per_inst.csv."
        ),
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
            "Metrics to compute and use when building summaries. Choices: "
            + ", ".join(_METRICS)
            + ". Default: all."
        ),
    )
    parser.add_argument(
        "--variant",
        default=None,
        choices=["agg", "decomp"],
        help="Restrict completion-time and cuts aggregate tables to one variant.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print detected experiments and exit without computing anything.",
    )
    parser.add_argument(
        "--from-summary",
        action="store_true",
        help="Read summary.csv and summary_per_inst.csv from <folder> and only write aggregate tables.",
    )

    args = parser.parse_args()

    # --- Aggregate-only mode (from existing summary CSV files) --------------
    if args.from_summary:
        summary_path = os.path.join(args.folder, "summary.csv")
        summary_per_inst_path = os.path.join(args.folder, "summary_per_inst.csv")
        if not os.path.exists(summary_path) or not os.path.exists(summary_per_inst_path):
            print(
                "Missing summary files. Expected both:\n"
                f"  {summary_path}\n"
                f"  {summary_per_inst_path}",
                file=sys.stderr,
            )
            sys.exit(1)

        output_dir = args.output or args.folder
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
        print(f"Loading summaries from '{args.folder}'...")
        summary = pd.read_csv(summary_path)
        summary_per_inst = pd.read_csv(summary_per_inst_path)
        written = write_aggregate_tables(
            output_dir=output_dir,
            summary_level=summary,
            summary_per_inst=summary_per_inst,
            variant=args.variant,
        )
        print(f"\nDone. {len(written)} CSV file(s) written to '{output_dir}'.")
        return

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
        max_control_size = cfg.get("max_control_size", "?")
        print(
            f"  {exp_name:<25}  alg: {alg:<10}  "
            f"max_length: {max_length:<4}  max_control_size: {max_control_size}"
        )

    if args.list:
        return

    contexts = build_experiment_contexts(experiments)
    if not contexts:
        print("No loadable experiments after initialization.", file=sys.stderr)
        sys.exit(1)

    # --- Compute and write metrics ------------------------------------------
    output_dir = args.output or os.path.join(args.folder, "_results")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    print(f"Computing {len(args.metrics)} metric(s) for summary generation...")
    metric_dfs = collect_metrics(contexts, args.metrics)
    print(f"Computed metric tables: {', '.join(sorted(metric_dfs.keys())) or '(none)'}")

    # --- Build and write summary tables -------------------------------------
    written = []
    summary_outputs: dict[str, pd.DataFrame] = {}
    for key, label, builder, fname in [
        ("summary", "summary (per level)", build_summary_table, "summary.csv"),
        ("summary_per_inst", "summary (per inst)", build_per_inst_table, "summary_per_inst.csv"),
    ]:
        print(f"\n  {label:<30}", end="", flush=True)
        df = builder(metric_dfs)
        if df is not None and not df.empty:
            out_path = os.path.join(output_dir, fname)
            df.to_csv(out_path, index=False)
            print(f"{len(df):>6} rows  →  {out_path}")
            written.append(out_path)
            summary_outputs[key] = df
        else:
            print("  (no data)")

    written.extend(
        write_aggregate_tables(
            output_dir=output_dir,
            summary_level=summary_outputs.get("summary"),
            summary_per_inst=summary_outputs.get("summary_per_inst"),
            variant=args.variant,
        )
    )

    print(f"\nDone. {len(written)} CSV file(s) written to '{output_dir}'.")


if __name__ == "__main__":
    main()
