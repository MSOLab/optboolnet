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
CUT_MACROS = {
    "TS cut": r"\tscut{}",
    "AT cut": r"\attrcut{}",
}
CUT_ORDER = {"TS cut": 0, "AT cut": 1}
INST_COLUMNS = ["S1", "S2", "S3", "S4", "M1", "M2", "M3", "L1", "L2", "L3", "L4"]
TABULAR_SPEC = "cc rrrr rrr rrrr"


def parse_experiment_name(name: str) -> tuple[str, str, int]:
    match = EXPERIMENT_RE.match(str(name))
    if not match:
        raise ValueError(f"Experiment name does not match '<Algorithm>_<Option>_<T_max>': {name}")
    return match.group("algorithm"), match.group("option"), int(match.group("tmax"))


def load_cut_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    parsed = df["experiment"].map(parse_experiment_name)
    df[["algorithm", "option", "parsed_tmax"]] = pd.DataFrame(parsed.tolist(), index=df.index)
    df["max_length"] = pd.to_numeric(df["max_length"], errors="coerce")
    return df


def filter_rows(
    df: pd.DataFrame,
    tmax: int,
    include_patterns: list[str] | None,
    filter_patterns: list[str] | None,
) -> pd.DataFrame:
    out = df[df["max_length"] == tmax].copy()

    def _matches_any(name: object, patterns: list[re.Pattern[str]]) -> bool:
        text = "" if pd.isna(name) else str(name)
        return any(p.search(text) is not None for p in patterns)

    if include_patterns:
        include_compiled = [re.compile(pattern) for pattern in include_patterns]
        out = out[out["experiment"].map(lambda x: _matches_any(x, include_compiled))]
    if filter_patterns:
        filter_compiled = [re.compile(pattern) for pattern in filter_patterns]
        out = out[~out["experiment"].map(lambda x: _matches_any(x, filter_compiled))]
    return out


def sort_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["alg_rank"] = out["algorithm"].map(ALG_ORDER).fillna(len(ALG_ORDER))
    out["opt_rank"] = out["option"].map(OPTION_ORDER).fillna(len(OPTION_ORDER))
    out["cut_rank"] = out["Cuts"].map(CUT_ORDER).fillna(len(CUT_ORDER))
    return out.sort_values(
        by=["alg_rank", "opt_rank", "algorithm", "option", "cut_rank", "Cuts"],
        ascending=[True, True, True, True, True, True],
        kind="stable",
    )


def format_total_cell(value: object) -> str:
    if pd.isna(value):
        return "-"
    num = float(value)
    if abs(num - round(num)) < 1e-9:
        return str(int(round(num)))
    return f"{num:.1f}"


def format_avg_cell(value: object) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.1f}"


def _cut_display(cut_name: str) -> str:
    return CUT_MACROS.get(cut_name, cut_name)


def _build_section_rows(df: pd.DataFrame, formatter) -> list[str]:
    rows: list[str] = []
    ordered = sort_rows(df)
    for (alg, opt), grp in ordered.groupby(["algorithm", "option"], sort=False):
        first = True
        for _, row in grp.iterrows():
            alg_label = ALG_MACROS.get(alg, alg)
            if first:
                if opt != "PBN":
                    alg_label = f"{alg_label} ({opt})"
                first = False
            else:
                alg_label = ""

            cut_label = _cut_display(str(row["Cuts"]))
            vals = [formatter(row.get(inst)) for inst in INST_COLUMNS]
            rows.append(" & ".join([alg_label, cut_label] + vals) + r" \\")
    return rows


def build_latex(total_df: pd.DataFrame, avg_df: pd.DataFrame, tmax: int) -> str:
    lines = [
        r"\newcommand{\newcutstrengthheader}{",
        r"\multicolumn{2}{@{}c}{\textbf{Settings}} & \multicolumn{11}{c@{}}{\textbf{Instances}} \\ \cmidrule(r){1-2} \cmidrule(l){3-13}",
        r"Algorithm & Cuts & S1 & S2 & S3 & S4 & M1 & M2 & M3 & L1 & L2 & L3 & L4",
        r"}",
        "",
        r"\begin{table}[t]",
        r"\small",
        r"\centering",
        rf"\TABLE{{The statistics of the Benders cuts  $(\Tmax={tmax})$\label{{tab:result-benders-statistics-tmax-{tmax}}}}}{{",
        rf"\begin{{tabular}}{{{TABULAR_SPEC}}} \toprule",
        r"\newcutstrengthheader \\ \midrule",
        r"\multicolumn{13}{@{}l}{\textit{\textbf{Total \# of cuts}}} \\",
    ]

    lines.extend(_build_section_rows(total_df, format_total_cell))
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{13}{@{}l}{\textit{\textbf{Avg. literals in a cut}}} \\")
    lines.extend(_build_section_rows(avg_df, format_avg_cell))
    lines.extend([r"\bottomrule", r"\end{tabular}}{}", r"\end{table}"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create LaTeX cut statistics table from agg_cuts_total.csv and agg_cuts_avg.csv.",
    )
    parser.add_argument("root_dir", help="Experiment root directory, e.g. _experiments/260306_full_no_good")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output tex path (default: <root_dir>/_results/agg_cut_strength_T<tmax>.tex)",
    )
    parser.add_argument(
        "--tmax",
        type=int,
        required=True,
        metavar="TMAX",
        help="Use exactly one T_max value, e.g. --tmax 45",
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
        help="Include only experiments matching regex, e.g. --include-exp '^(SEP_DEC|BEN_DEC)_[0-9]+$'",
    )
    parser.add_argument(
        "--control-size",
        nargs="+",
        type=int,
        default=None,
        metavar="LEVEL",
        help="Accepted for CLI compatibility with pubs_comptime.py (unused for cuts tables).",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    total_path = root_dir / "_results" / "agg_cuts_total.csv"
    avg_path = root_dir / "_results" / "agg_cuts_avg.csv"
    if not total_path.exists():
        raise FileNotFoundError(f"Missing file: {total_path}")
    if not avg_path.exists():
        raise FileNotFoundError(f"Missing file: {avg_path}")

    total_df = load_cut_table(total_path)
    avg_df = load_cut_table(avg_path)

    total_df = filter_rows(total_df, args.tmax, args.include_exp, args.filter_exp)
    avg_df = filter_rows(avg_df, args.tmax, args.include_exp, args.filter_exp)
    if total_df.empty and avg_df.empty:
        raise ValueError("No rows remain after filtering for both cuts tables.")

    latex = build_latex(total_df, avg_df, args.tmax)

    output_path = (
        Path(args.output)
        if args.output
        else root_dir / "_results" / f"agg_cut_strength_T{args.tmax}.tex"
    )
    output_path.write_text(latex + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
