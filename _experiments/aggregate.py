"""
aggregate.py
------------
Export summary pivot tables to CSV files inside RESULTS_DIR.

Files produced:
  agg_solution_count.csv   (max_length, experiment) × inst  →  total_solutions
  agg_completion_time.csv  level × (T_max, inst, label)     →  completion_time
  agg_cuts_total.csv       (section, alg, cut) × (T_max, inst)  →  # cuts
  agg_cuts_avg.csv         (section, alg, cut) × (T_max, inst)  →  avg literals

Usage:
  python aggregate.py _experiments/260302_fixed/_results
  python aggregate.py _experiments/260302_fixed/_results --variant decomp
"""

import argparse
import os

import pandas as pd

INST_ORDER = ["S1", "S2", "S3", "S4", "M1", "M2", "M3", "L1", "L2", "L3", "L4"]

# Fill in to rename experiment labels in the completion-time table.
# Unlisted experiments keep their raw name.
LABEL_MAP: dict[str, str] = {
    # 'BEN_45_agg':    'BEN (agg)',
    # 'BEN_45_decomp': 'BEN (decomp)',
    # 'SEP_45_agg':    'SEP (agg)',
    # 'SEP_45_decomp': 'SEP (decomp)',
    # 'MibS_45':       'PBN',
}

# (section_label, alg_prefix, csv_column, display_label)
_CUT_SPECS = [
    ("Total # of cuts", "SEP", "count_cuts_TRAP_SPACE_CUT", "TS cut"),
    ("Total # of cuts", "SEP", "count_cuts_ATTRACTOR_CUT", "AT cut"),
    ("Total # of cuts", "BEN", "count_cuts_ATTRACTOR_CUT", "AT cut"),
    ("Avg. literals in a cut", "SEP", "avg_cuts_TRAP_SPACE_CUT", "TS cut"),
    ("Avg. literals in a cut", "SEP", "avg_cuts_ATTRACTOR_CUT", "AT cut"),
    ("Avg. literals in a cut", "BEN", "avg_cuts_ATTRACTOR_CUT", "AT cut"),
]


# ---------------------------------------------------------------------------
# Table builders
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
    """
    Same pivot as make_solution_count, but cells where the top level was
    finished are marked (*) (e.g. 12*).  Unfinished or missing cells
    are plain strings.  Requires 'level_finished' column in ct.
    """
    base = make_solution_count(df)

    if "level_finished" not in ct.columns:
        return base

    # For each (experiment, inst): was the highest-numbered level finished?
    top = (
        ct.sort_values("level")
        .groupby(["experiment", "inst"])
        .last()
        .reset_index()
        [["experiment", "inst", "level_finished"]]
    )
    finished_pairs = set(
        zip(
            top.loc[top["level_finished"], "experiment"],
            top.loc[top["level_finished"], "inst"],
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
    ct = ct.copy()
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

    # Canonical column order: T_max (asc) → inst (INST_ORDER) → label (alpha)
    ordered_cols = [
        (ml, inst, label)
        for ml in ml_values
        for inst in inst_present
        for label in labels_ordered
        if (ml, inst, label) in tbl.columns
    ]
    tbl = tbl.reindex(columns=ordered_cols).round(1)
    tbl.columns.names = ["T_max", "", ""]
    tbl.index.name = "λ"
    return tbl


def _cuts_for_ml(spi_ml: pd.DataFrame, specs: list) -> pd.DataFrame | None:
    """Build one (row-index, inst) slice for a single max_length value."""
    inst_present = [i for i in INST_ORDER if i in spi_ml["inst"].values]
    rows = []
    for section, alg, col, cut_label in specs:
        if col not in spi_ml.columns:
            continue
        sub = (
            spi_ml[spi_ml["alg"] == alg]
            .groupby("inst")[col]
            .mean()  # averages agg/decomp variants if both present
            .reindex(inst_present)
        )
        sub.name = (section, alg, cut_label)
        rows.append(sub)
    if not rows:
        return None
    tbl = pd.concat(rows, axis=1).T
    tbl.index = pd.MultiIndex.from_tuples(
        tbl.index, names=["Section", "Algorithm", "Cuts"]
    )
    return tbl


def make_cuts_table(
    spi: pd.DataFrame, variant: str | None, col_prefix: str
) -> pd.DataFrame:
    spi = spi.copy()
    if variant:
        spi = spi[spi["experiment"].str.endswith(variant)]
    spi["alg"] = spi["experiment"].str.split("_").str[0]

    specs = [(s, a, c, l) for s, a, c, l in _CUT_SPECS if col_prefix in c]
    ml_values = sorted(spi["max_length"].unique())

    parts: dict[int, pd.DataFrame] = {}
    for ml in ml_values:
        tbl_ml = _cuts_for_ml(spi[spi["max_length"] == ml], specs)
        if tbl_ml is not None:
            parts[ml] = tbl_ml

    if not parts:
        return pd.DataFrame()

    tbl = pd.concat(parts, axis=1)  # T_max becomes top column level
    tbl.columns.names = ["T_max", "inst"]

    tbl = tbl.apply(pd.to_numeric, errors="coerce").round(1)
    return tbl


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export summary pivot tables to CSV files in RESULTS_DIR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "results_dir",
        help="Folder produced by collect_results.py (contains summary_per_inst.csv etc.).",
    )
    parser.add_argument(
        "--variant",
        default=None,
        choices=["agg", "decomp"],
        help="Restrict completion-time and cuts tables to one experiment variant "
        "(default: include all variants, averaging where duplicates exist).",
    )
    args = parser.parse_args()

    rd = args.results_dir
    df = pd.read_csv(os.path.join(rd, "summary_per_inst.csv"))
    ct = pd.read_csv(os.path.join(rd, "completion_time.csv"))

    tables = {
        "agg_solution_count":      make_solution_count(df),
        "agg_solution_count_bold": make_solution_count_bold(df, ct),
        "agg_completion_time":     make_completion_time(ct, args.variant),
        "agg_cuts_total":          make_cuts_table(df, args.variant, "count_cuts"),
        "agg_cuts_avg":            make_cuts_table(df, args.variant, "avg_cuts"),
    }

    print(f"Writing tables to '{rd}':")
    for name, tbl in tables.items():
        if tbl is None or tbl.empty:
            print(f"  {name:<30} (no data, skipped)")
            continue
        out = os.path.join(rd, f"{name}.csv")
        tbl.to_csv(out)
        print(f"  {name:<30} {str(tbl.shape):>12}  →  {out}")


if __name__ == "__main__":
    main()
