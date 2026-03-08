import argparse
import json
from pathlib import Path
import re

import pandas as pd


INST_COLUMNS = ["S1", "S2", "S3", "S4", "M1", "M2", "M3", "L1", "L2", "L3", "L4"]
ALG_ORDER = {"SEP": 0, "BEN": 1, "MibS": 2, "PBN": 3}
OPTION_ORDER = {"DEC": 0, "AGG": 1}
ALG_MACROS = {
    "SEP": r"\SEP{}",
    "BEN": r"\BEN{}",
    "MibS": r"\MibS{}",
    "PBN": r"\PBN{}",
}
TABULAR_SPEC = "cc cccc ccc cccc"
EXPERIMENT_RE = re.compile(r"^(?P<algorithm>[^_]+)_(?P<option>[^_]+)_(?P<tmax>\d+)$")


def parse_experiment_name(name: str) -> tuple[str, str, int]:
    match = EXPERIMENT_RE.match(str(name))
    if not match:
        raise ValueError(f"Experiment name does not match '<Algorithm>_<Option>_<T_max>': {name}")
    return (
        match.group("algorithm"),
        match.group("option"),
        int(match.group("tmax")),
    )


def load_verification_counts(json_path: Path) -> dict[tuple[str, str], int]:
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)

    incorrect_counts: dict[tuple[str, str], int] = {}
    by_instance = data.get("experiments_by_instance", {})
    for experiment, inst_map in by_instance.items():
        if not isinstance(inst_map, dict):
            continue
        for inst, metrics in inst_map.items():
            if not isinstance(metrics, dict):
                continue
            incorrect_counts[(experiment, inst)] = int(metrics.get("incorrect", 0) or 0)
    return incorrect_counts


def latex_cell(value: object, incorrect_count: int) -> str:
    if pd.isna(value):
        return r"\expna"

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return r"\expna"

    is_bold = text.endswith("*")
    if is_bold:
        text = text[:-1]
    rendered = text if incorrect_count == 0 else rf"{text} $\langle {incorrect_count} \rangle$"
    return rf"\textbf{{{rendered}}}" if is_bold and incorrect_count == 0 else rendered


def load_solution_counts(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str)
    parsed = df["experiment"].map(parse_experiment_name)
    df[["algorithm", "option", "parsed_tmax"]] = pd.DataFrame(parsed.tolist(), index=df.index)
    df["parsed_tmax"] = pd.to_numeric(df["parsed_tmax"], errors="raise")
    return df


def filter_solution_counts(
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


def sort_solution_counts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["alg_rank"] = out["algorithm"].map(ALG_ORDER).fillna(len(ALG_ORDER))
    out["opt_rank"] = out["option"].map(OPTION_ORDER).fillna(len(OPTION_ORDER))
    return out.sort_values(
        by=["parsed_tmax", "alg_rank", "opt_rank", "experiment"],
        ascending=[True, True, True, True],
        kind="stable",
    )


def make_row(row: pd.Series, incorrect_counts: dict[tuple[str, str], int]) -> str:
    algo = ALG_MACROS.get(row["algorithm"], row["algorithm"])
    option = str(row["option"])
    if option in OPTION_ORDER:
        algo = f"{algo} ({option})"
    experiment = str(row["experiment"])
    cells = [
        latex_cell(row.get(inst, ""), incorrect_counts.get((experiment, inst), 0))
        for inst in INST_COLUMNS
    ]
    entries = [algo, str(int(row["parsed_tmax"]))] + cells
    return " & ".join(entries) + r" \\"


def build_tabular(df: pd.DataFrame, incorrect_counts: dict[tuple[str, str], int]) -> str:
    lines = [
        rf"\begin{{tabular}}{{{TABULAR_SPEC}}}",
        r"\toprule",
        r"\ctrlnote \\",
        r"\cmidrule(r){1-2}\cmidrule(l){3-13}",
        r"Name & $\Tmax$ & S1 & S2 & S3 & S4 & M1 & M2 & M3 & L1 & L2 & L3 & L4 \\",
        r"\midrule",
    ]

    ordered = sort_solution_counts(df)
    tmax_values = ordered["parsed_tmax"].drop_duplicates().tolist()
    for _, tmax in enumerate(tmax_values):
        block = ordered[ordered["parsed_tmax"] == tmax]
        for _, row in block.iterrows():
            lines.append(make_row(row, incorrect_counts))

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a LaTeX tabular from agg_solution_count_bold.csv.",
    )
    parser.add_argument(
        "root_dir",
        help="Experiment root directory, e.g. _experiments/260306_full_no_good",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Write LaTeX output to this file. Defaults to <root_dir>/_results/agg_solution_count_bold.tex",
    )
    parser.add_argument(
        "--tmax",
        nargs="+",
        type=int,
        default=None,
        metavar="TMAX",
        help="Restrict output to these T_max values, e.g. --tmax 1 3 15 45",
    )
    parser.add_argument(
        "--filter-exp",
        nargs="+",
        default=None,
        metavar="REGEX",
        help=(
            "Exclude experiments whose names match any regex pattern. "
            r"Example: --filter-exp '^BEN_AGG_.*$'"
        ),
    )
    parser.add_argument(
        "--include-exp",
        nargs="+",
        default=None,
        metavar="REGEX",
        help=(
            "Keep only experiments whose names match at least one regex pattern. "
            r"Example: --include-exp '^(SEP|BEN)_.*$'"
        ),
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    csv_path = root_dir / "_results" / "agg_solution_count_bold.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV file: {csv_path}")

    df = load_solution_counts(csv_path)
    df = filter_solution_counts(df, args.tmax, args.include_exp, args.filter_exp)
    verify_path = root_dir / "_results" / "verify_control_Tinf.json"
    if not verify_path.exists():
        raise FileNotFoundError(f"Missing verification file: {verify_path}")
    incorrect_counts = load_verification_counts(verify_path)

    latex = build_tabular(df, incorrect_counts)

    output_path = (
        Path(args.output)
        if args.output
        else root_dir / "_results" / "agg_solution_count_bold.tex"
    )
    output_path.write_text(latex + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
