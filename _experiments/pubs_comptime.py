import argparse
from pathlib import Path
import re

import pandas as pd


EXPERIMENT_RE = re.compile(r"^(?P<algorithm>[^_]+)_(?P<option>[^_]+)_(?P<tmax>\d+)$")
ALG_ORDER = {"SEP": 0, "BEN": 1, "MibS": 2, "PBN": 3}
OPTION_ORDER = {"DEC": 0, "AGG": 1, "PBN": 2}
ALG_MACROS = {
    "SEP": r"\SEP{}",
    "BEN": r"\BEN{}",
    "MibS": r"\MibS{}",
    "PBN": r"\PBN{}",
}

SMALL_INST = ["S1", "S2", "S3", "S4"]
MEDIUM_INST = ["M1", "M2", "M3"]
LARGE_INST = ["L1", "L2", "L3", "L4"]


def parse_experiment_name(name: str) -> tuple[str, str, int]:
    match = EXPERIMENT_RE.match(str(name))
    if not match:
        raise ValueError(f"Experiment name does not match '<Algorithm>_<Option>_<T_max>': {name}")
    return match.group("algorithm"), match.group("option"), int(match.group("tmax"))


def load_summary(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    parsed = df["experiment"].map(parse_experiment_name)
    df[["algorithm", "option", "parsed_tmax"]] = pd.DataFrame(parsed.tolist(), index=df.index)
    df["parsed_tmax"] = pd.to_numeric(df["parsed_tmax"], errors="raise")
    df["level"] = pd.to_numeric(df["level"], errors="coerce")
    df["completion_time"] = pd.to_numeric(df["completion_time"], errors="coerce")
    df["level_finished"] = df["level_finished"].astype(str).str.lower().map({"true": True, "false": False})
    return df


def filter_summary(
    df: pd.DataFrame,
    tmax_values: list[int] | None,
    include_patterns: list[str] | None,
    filter_patterns: list[str] | None,
) -> pd.DataFrame:
    out = df.copy()

    def _matches_any(name: object, patterns: list[re.Pattern[str]]) -> bool:
        text = "" if pd.isna(name) else str(name)
        return any(p.search(text) is not None for p in patterns)

    if tmax_values:
        out = out[out["parsed_tmax"].isin(tmax_values)]
    if include_patterns:
        include_compiled = [re.compile(pattern) for pattern in include_patterns]
        out = out[out["experiment"].map(lambda x: _matches_any(x, include_compiled))]
    if filter_patterns:
        filter_compiled = [re.compile(pattern) for pattern in filter_patterns]
        out = out[~out["experiment"].map(lambda x: _matches_any(x, filter_compiled))]
    return out


def pick_single_run(df: pd.DataFrame) -> pd.DataFrame:
    # Keep one run per (tmax, algorithm, option, inst, level). If duplicates exist, keep the
    # first by experiment name after deterministic sorting.
    out = df.copy()
    out["alg_rank"] = out["algorithm"].map(ALG_ORDER).fillna(len(ALG_ORDER))
    out["opt_rank"] = out["option"].map(OPTION_ORDER).fillna(len(OPTION_ORDER))
    out = out.sort_values(
        by=["parsed_tmax", "alg_rank", "opt_rank", "experiment", "inst", "level"],
        ascending=[True, True, True, True, True, True],
        kind="stable",
    )
    return out.drop_duplicates(
        subset=["parsed_tmax", "algorithm", "option", "inst", "level"], keep="first"
    )


def _fmt_time(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.1f}"


def _ordered_alg_options(df: pd.DataFrame) -> list[tuple[str, str]]:
    combos = (
        df[["algorithm", "option"]]
        .drop_duplicates()
        .assign(
            alg_rank=lambda x: x["algorithm"].map(ALG_ORDER).fillna(len(ALG_ORDER)),
            opt_rank=lambda x: x["option"].map(OPTION_ORDER).fillna(len(OPTION_ORDER)),
        )
        .sort_values(["alg_rank", "opt_rank", "algorithm", "option"], kind="stable")
    )
    return list(combos[["algorithm", "option"]].itertuples(index=False, name=None))


def _build_header_macro(
    macro_name: str,
    instances: list[str],
    alg_options: list[tuple[str, str]],
    include_alg_opt_rows: bool = True,
    include_cmidrules: bool = True,
) -> str:
    n = len(alg_options)
    line1_cells: list[str] = []
    cmidrules: list[str] = []
    col_start = 2
    for inst in instances:
        col_end = col_start + n - 1
        line1_cells.append(rf"\Hlabel{{{n}}}{{{inst}}}")
        cmidrules.append(rf"\cmidrule(lr){{{col_start}-{col_end}}}")
        col_start = col_end + 1
    line1 = r"$\TargetSize$ & " + " & ".join(line1_cells) + r" \\"
    if include_cmidrules:
        line1 = line1 + " " + "".join(cmidrules)
    section_rule_line = r"\cmidrule{1-1}" + "".join(cmidrules)

    lines = [rf"\newcommand{{\{macro_name}}}{{", r"\midrule", line1]
    if include_alg_opt_rows:
        alg_cells = [r"$\TargetSize$"]
        opt_cells = [" "]
        for _ in instances:
            for alg, opt in alg_options:
                alg_cells.append(rf"\multicolumn{{1}}{{c}}{{{ALG_MACROS.get(alg, alg)}}}")
                opt_cells.append(rf"\multicolumn{{1}}{{c}}{{{opt}}}")
        lines.append(" & ".join(alg_cells) + r" \\")
        lines.append(" & ".join(opt_cells) + r" \\ \midrule")
    else:
        lines = [rf"\newcommand{{\{macro_name}}}{{", r"\midrule", line1, section_rule_line]
    lines.append(r"}")
    return "\n".join(lines)

def _build_alg_header_macro(
    macro_name: str, instances: list[str], alg_options: list[tuple[str, str]]
) -> str:
    # One-time header that shows algorithm/option rows only.
    alg_cells = [" "]
    opt_cells = [" "]
    for _ in instances:
        for alg, opt in alg_options:
            alg_cells.append(rf"\multicolumn{{1}}{{c}}{{{ALG_MACROS.get(alg, alg)}}}")
            opt_cells.append(rf"\multicolumn{{1}}{{c}}{{{opt}}}")

    return "\n".join(
        [
            rf"\newcommand{{\{macro_name}}}{{",
            r"\toprule",
            " & ".join(alg_cells) + r" \\",
            " & ".join(opt_cells) + r" \\",
            r"}",
        ]
    )


def _build_section_rows(
    section_df: pd.DataFrame,
    instances: list[str],
    levels: list[int],
    tmax: int,
    alg_options: list[tuple[str, str]],
) -> list[str]:
    rows: list[str] = []
    for level in levels:
        cells = [str(int(level))]
        for inst in instances:
            for alg, opt in alg_options:
                sub = section_df[
                    (section_df["parsed_tmax"] == tmax)
                    & (section_df["inst"] == inst)
                    & (section_df["algorithm"] == alg)
                    & (section_df["option"] == opt)
                    & (section_df["level"] == level)
                ]
                if sub.empty:
                    cells.append("-")
                    continue
                row = sub.iloc[0]
                val = row["completion_time"] if bool(row.get("level_finished", False)) else None
                cells.append(_fmt_time(val))
        rows.append(" & ".join(cells) + r" \\")
    return rows


def build_table_block(df: pd.DataFrame, tmax: int, levels: list[int]) -> str:
    alg_options = _ordered_alg_options(df)
    ncols = 1 + len(SMALL_INST) * len(alg_options)
    tabular_spec = "l " + "r" * (ncols - 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\TABLE{{Computation time to find all minimal controls up to size $\TargetSize$ with $\Tmax={tmax}$ (sec.)\label{{tab:computation-time-tmax-{tmax}}}}}{{",
        r"\small",
        rf"\begin{{tabular}}{{{tabular_spec}}}",
    ]

    lines.append(r"\makealgheader")
    lines.append(r"\makesmallheader")
    lines.extend(_build_section_rows(df, SMALL_INST, levels, tmax, alg_options))
    lines.append(r"\makemediumheader")
    lines.extend(_build_section_rows(df, MEDIUM_INST, levels, tmax, alg_options))
    lines.append(r"\makelargeheader")
    lines.extend(_build_section_rows(df, LARGE_INST, levels, tmax, alg_options))
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}{}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def build_latex(df: pd.DataFrame, selected_levels: list[int] | None = None) -> str:
    tmax_values = sorted(df["parsed_tmax"].dropna().astype(int).unique().tolist())
    if len(tmax_values) != 1:
        raise ValueError("Expected exactly one T_max after filtering. Use --tmax with a single value.")
    alg_options = _ordered_alg_options(df)
    if not alg_options:
        raise ValueError("No algorithm/option combinations remain after filtering.")

    macros = "\n".join(
        [
            r"\newcommand{\Hlabel}[2]{\multicolumn{#1}{c}{\textbf{#2}}}",
            _build_header_macro(
                "makesmallheader",
                SMALL_INST,
                alg_options,
                include_alg_opt_rows=False,
                include_cmidrules=False,
            ),
            _build_alg_header_macro("makealgheader", SMALL_INST, alg_options),
            _build_header_macro(
                "makemediumheader",
                MEDIUM_INST,
                alg_options,
                include_alg_opt_rows=False,
                include_cmidrules=False,
            ),
            _build_header_macro(
                "makelargeheader",
                LARGE_INST,
                alg_options,
                include_alg_opt_rows=False,
                include_cmidrules=False,
            ),
        ]
    )
    lines = [macros, ""]

    all_levels = sorted(df["level"].dropna().astype(int).unique().tolist())
    if selected_levels:
        selected_set = set(selected_levels)
        levels = [lvl for lvl in all_levels if lvl in selected_set]
        if not levels:
            raise ValueError("No levels remain after applying --control-size.")
    else:
        levels = all_levels
    tmax = tmax_values[0]
    lines.append(build_table_block(df, tmax, levels))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create LaTeX computation-time tables from summary.csv.",
    )
    parser.add_argument("root_dir", help="Experiment root directory, e.g. _experiments/260306_full_no_good")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output tex path (default: <root_dir>/_results/agg_comptime_T<tmax>.tex)",
    )
    parser.add_argument(
        "--tmax",
        type=int,
        required=True,
        metavar="TMAX",
        help="Restrict output to exactly one T_max value, e.g. --tmax 45",
    )
    parser.add_argument(
        "--filter-exp",
        nargs="+",
        default=None,
        metavar="REGEX",
        help="Exclude experiments matching regex, e.g. --filter-exp '^BEN_AGG_.*$'",
    )
    parser.add_argument(
        "--include-exp",
        nargs="+",
        default=None,
        metavar="REGEX",
        help="Include only experiments matching regex, e.g. --include-exp '^(SEP_DEC|BEN_DEC|PBN)_.*$'",
    )
    parser.add_argument(
        "--control-size",
        nargs="+",
        type=int,
        default=None,
        metavar="LEVEL",
        help="Restrict output to selected control sizes (levels), e.g. --control-size 0 1 2. Default: all.",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    csv_path = root_dir / "_results" / "summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing summary file: {csv_path}")

    df = load_summary(csv_path)
    df = filter_summary(df, [args.tmax], args.include_exp, args.filter_exp)
    if df.empty:
        raise ValueError("No rows remain after filtering.")
    df = pick_single_run(df)

    latex = build_latex(df, selected_levels=args.control_size)

    output_path = (
        Path(args.output)
        if args.output
        else root_dir / "_results" / f"agg_comptime_T{args.tmax}.tex"
    )
    output_path.write_text(latex + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
