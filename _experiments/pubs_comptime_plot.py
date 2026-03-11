import argparse
import bisect
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import pandas as pd


EXPERIMENT_RE = re.compile(r"^(?P<algorithm>[^_]+)_(?P<option>[^_]+)_(?P<tmax>\d+)$")
ALG_ORDER = {"SEP": 0, "BEN": 1, "MibS": 2, "PBN": 3}
OPTION_ORDER = {"DEC": 0, "AGG": 1, "PBN": 2}
SMALL_INST = ["S1", "S2", "S3", "S4"]
MEDIUM_INST = ["M1", "M2", "M3"]
LARGE_INST = ["L1", "L2", "L3", "L4"]
INST_GRID = [
    ["S1", "S2", "S3", "S4"],
    ["M1", "M2", "M3", None],
    ["L1", "L2", "L3", "L4"],
]
OPTION_STYLES = {"DEC": "-", "AGG": "--", "PBN": "-."}
FINISHED_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8"]
AXIS_TITLE_FONTSIZE = 17
AXIS_LABEL_FONTSIZE = 15
TICK_LABEL_FONTSIZE = 13
LEGEND_FONTSIZE = 14
LINE_WIDTH = 2.8


def parse_experiment_name(name: str) -> tuple[str, str, int]:
    match = EXPERIMENT_RE.match(str(name))
    if not match:
        raise ValueError(f"Experiment name does not match '<Algorithm>_<Option>_<T_max>': {name}")
    return match.group("algorithm"), match.group("option"), int(match.group("tmax"))


def load_timestamps(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "experiment" not in df.columns:
        raise ValueError(f"Missing 'experiment' column in {csv_path}")

    if not {"alg_name", "option", "max_length"}.issubset(df.columns):
        parsed = df["experiment"].map(parse_experiment_name)
        parsed_df = pd.DataFrame(parsed.tolist(), columns=["alg_name", "option", "parsed_tmax"], index=df.index)
        if "alg_name" not in df.columns:
            df["alg_name"] = parsed_df["alg_name"]
        if "option" not in df.columns:
            df["option"] = parsed_df["option"]
        if "max_length" not in df.columns:
            df["max_length"] = parsed_df["parsed_tmax"]

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["level"] = pd.to_numeric(df["level"], errors="coerce")
    df["max_length"] = pd.to_numeric(df["max_length"], errors="coerce")
    return df


def load_optional_timestamps(csv_path: Path | None) -> pd.DataFrame | None:
    if csv_path is None or not csv_path.exists():
        return None
    return load_timestamps(csv_path)


def filter_timestamps(
    df: pd.DataFrame,
    tmax: int,
    include_patterns: list[str] | None,
    filter_patterns: list[str] | None,
    control_sizes: list[int] | None,
) -> pd.DataFrame:
    out = df.copy()

    def _matches_any(name: object, patterns: list[re.Pattern[str]]) -> bool:
        text = "" if pd.isna(name) else str(name)
        return any(p.search(text) is not None for p in patterns)

    out = out[out["max_length"] == tmax]
    if control_sizes:
        out = out[out["level"].isin(control_sizes)]
    if include_patterns:
        include_compiled = [re.compile(pattern) for pattern in include_patterns]
        out = out[out["experiment"].map(lambda x: _matches_any(x, include_compiled))]
    if filter_patterns:
        filter_compiled = [re.compile(pattern) for pattern in filter_patterns]
        out = out[~out["experiment"].map(lambda x: _matches_any(x, filter_compiled))]
    return out


def ordered_alg_options(df: pd.DataFrame) -> list[tuple[str, str]]:
    combos = (
        df[["alg_name", "option"]]
        .drop_duplicates()
        .assign(
            alg_rank=lambda x: x["alg_name"].map(ALG_ORDER).fillna(len(ALG_ORDER)),
            opt_rank=lambda x: x["option"].map(OPTION_ORDER).fillna(len(OPTION_ORDER)),
        )
        .sort_values(["alg_rank", "opt_rank", "alg_name", "option"], kind="stable")
    )
    return list(combos[["alg_name", "option"]].itertuples(index=False, name=None))


def color_map_for_alg_options(alg_options: list[tuple[str, str]]) -> dict[tuple[str, str], str]:
    palette = list(plt.cm.tab10.colors) + list(plt.cm.Set2.colors) + list(plt.cm.Dark2.colors)
    palette_hex = [mcolors.to_hex(color) for color in palette]
    return {
        alg_option: palette_hex[idx % len(palette_hex)]
        for idx, alg_option in enumerate(alg_options)
    }


def build_series(df: pd.DataFrame, inst: str, alg_name: str, option: str, time_limit: float) -> tuple[list[float], list[int]]:
    sub = df[
        (df["inst"] == inst)
        & (df["alg_name"] == alg_name)
        & (df["option"] == option)
    ].copy()
    sub = sub.dropna(subset=["timestamp"])
    sub = sub[sub["timestamp"] <= time_limit]
    if sub.empty:
        return [0.0], [0]
    sub = sub.sort_values(["timestamp", "level"], kind="stable")
    times = sub["timestamp"].astype(float).tolist()
    counts = list(range(1, len(times) + 1))
    return [0.0] + times, [0] + counts


def build_finished_points(
    finished_df: pd.DataFrame | None,
    inst: str,
    alg_name: str,
    option: str,
    time_limit: float,
    finished_levels: list[int] | None,
    series_times: list[float],
) -> dict[int, tuple[list[float], list[int]]]:
    if finished_df is None or finished_df.empty or not finished_levels:
        return {}
    sub = finished_df[
        (finished_df["inst"] == inst)
        & (finished_df["alg_name"] == alg_name)
        & (finished_df["option"] == option)
    ].copy()
    if finished_levels:
        sub = sub[sub["level"].isin(finished_levels)]
    sub = sub.dropna(subset=["timestamp"])
    sub = sub[sub["timestamp"] <= time_limit]
    if sub.empty:
        return {}
    sub = sub.sort_values(["level", "timestamp"], kind="stable")
    base_times = series_times[1:] if series_times and series_times[0] == 0.0 else series_times
    points_by_level: dict[int, tuple[list[float], list[int]]] = {}
    for level, level_df in sub.groupby("level", observed=True):
        point_x = level_df["timestamp"].astype(float).tolist()
        point_y = [bisect.bisect_right(base_times, x) for x in point_x]
        points_by_level[int(level)] = (point_x, point_y)
    return points_by_level


def build_finished_summary_lines(
    finished_df: pd.DataFrame | None,
    inst: str,
    alg_options: list[tuple[str, str]],
    finished_level: int,
    time_limit: float,
) -> list[str]:
    if finished_df is None or finished_df.empty:
        return []

    lines: list[str] = []
    for alg_name, option in alg_options:
        sub = finished_df[
            (finished_df["inst"] == inst)
            & (finished_df["alg_name"] == alg_name)
            & (finished_df["option"] == option)
            & (finished_df["level"] == finished_level)
        ].copy()
        sub = sub.dropna(subset=["timestamp"])
        sub = sub[sub["timestamp"] <= time_limit]
        if sub.empty:
            continue
        tmax = float(sub["timestamp"].max())
        lines.append(f"{alg_name}-{option}: {tmax:.1f}s")
    return lines


def plot_grid(
    df: pd.DataFrame,
    finished_df: pd.DataFrame | None,
    tmax: int,
    time_limit: float,
    time_limit_min: float,
    finished_levels: list[int] | None,
    title: str | None = None,
) -> plt.Figure:
    alg_options = ordered_alg_options(df)
    if not alg_options:
        raise ValueError("No algorithm/option combinations remain after filtering.")

    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=False)
    legend_handles: list[Line2D] = []
    seen_labels: set[str] = set()
    finished_legend_handles: list[Line2D] = []
    seen_finished_levels: set[int] = set()
    color_map = color_map_for_alg_options(alg_options)
    finished_marker_map = {
        level: FINISHED_MARKERS[idx % len(FINISHED_MARKERS)]
        for idx, level in enumerate(finished_levels or [])
    }
    single_finished_level = (
        int(finished_levels[0]) if finished_levels is not None and len(finished_levels) == 1 else None
    )
    legend_ax = axes[1, 3]
    legend_ax.axis("off")

    for row_idx, row in enumerate(INST_GRID):
        for col_idx, inst in enumerate(row):
            ax = axes[row_idx, col_idx]
            if inst is None:
                continue

            inst_df = df[df["inst"] == inst]
            inst_max = inst_df["timestamp"].dropna()
            inst_max_x = float(inst_max.max()) if not inst_max.empty else 0.0
            inst_max_x = min(inst_max_x, float(time_limit))
            inst_max_x = max(inst_max_x, float(time_limit_min), 1.0)
            for alg_name, option in alg_options:
                x, y = build_series(inst_df, inst, alg_name, option, time_limit)
                if x[-1] < inst_max_x:
                    x = x + [inst_max_x]
                    y = y + [y[-1]]
                label = f"{alg_name}-{option}"
                color = color_map[(alg_name, option)]
                linestyle = OPTION_STYLES.get(option, "-")
                ax.step(
                    x,
                    y,
                    where="post",
                    label=label,
                    color=color,
                    linestyle=linestyle,
                    linewidth=LINE_WIDTH,
                    alpha=0.7,
                )
                points_by_level = build_finished_points(
                    finished_df,
                    inst,
                    alg_name,
                    option,
                    time_limit,
                    finished_levels,
                    x,
                )
                for level, (point_x, point_y) in sorted(points_by_level.items()):
                    marker = finished_marker_map.get(level, FINISHED_MARKERS[0])
                    ax.scatter(
                        point_x,
                        point_y,
                        s=56,
                        color=color,
                        alpha=0.6,
                        marker=marker,
                        edgecolors="black",
                        linewidths=0.4,
                        zorder=3,
                    )
                    if level not in seen_finished_levels:
                        finished_legend_handles.append(
                            Line2D(
                                [0],
                                [0],
                                linestyle="None",
                                marker=marker,
                                markerfacecolor="#666666",
                                markeredgecolor="black",
                                markeredgewidth=0.4,
                                markersize=8,
                                alpha=0.6,
                                label=rf"Finished $\lambda={level}$",
                            )
                        )
                        seen_finished_levels.add(level)
                if label not in seen_labels:
                    legend_handles.append(
                        Line2D(
                            [0],
                            [0],
                            color=color,
                            linestyle=linestyle,
                            linewidth=LINE_WIDTH,
                            alpha=0.7,
                            label=label,
                        )
                    )
                    seen_labels.add(label)

            ax.set_title(inst, fontsize=AXIS_TITLE_FONTSIZE)
            xpad = max(0.02 * inst_max_x, 0.1)
            ax.set_xlim(-xpad, inst_max_x)
            ymax = ax.get_ylim()[1]
            ypad = max(0.02 * ymax, 0.1)
            ax.set_ylim(bottom=-ypad)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
            ax.grid(True, alpha=0.25)
            ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
            if col_idx == 0:
                ax.set_ylabel("Cumulative # of solutions", fontsize=AXIS_LABEL_FONTSIZE)
            if row_idx == 2:
                ax.set_xlabel("Time (s)", fontsize=AXIS_LABEL_FONTSIZE)
            if single_finished_level is not None:
                summary_lines = build_finished_summary_lines(
                    finished_df,
                    inst,
                    alg_options,
                    single_finished_level,
                    time_limit,
                )
                if summary_lines:
                    ax.text(
                        0.98,
                        0.03,
                        "$\\bf{[Finished]}$\n" + "\n".join(summary_lines),
                        transform=ax.transAxes,
                        ha="right",
                        va="bottom",
                        multialignment="left",
                        fontsize=TICK_LABEL_FONTSIZE - 1,
                        color="#444444",
                        alpha=0.8,
                        bbox=dict(facecolor="white", alpha=0.35, edgecolor="none"),
                        zorder=0,
                    )

    combined_handles = legend_handles + finished_legend_handles
    if combined_handles:
        legend_ax.legend(
            handles=combined_handles,
            loc="center",
            ncol=1,
            frameon=False,
            fontsize=LEGEND_FONTSIZE,
        )
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot cumulative solution counts from timestamp.csv.",
    )
    parser.add_argument("root_dir", help="Experiment root directory, e.g. _experiments/260306_full_no_good")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output image path (default: <root_dir>/_results/agg_comptime_plot_T<tmax>.<ext>)",
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
        help="Include only experiments matching regex.",
    )
    parser.add_argument(
        "--control-size",
        nargs="+",
        type=int,
        default=None,
        metavar="LEVEL",
        help="Restrict output to selected control sizes (levels). Default: all.",
    )
    parser.add_argument(
        "--finished-levels",
        nargs="+",
        type=int,
        default=None,
        metavar="LEVEL",
        help="Overlay FINISHED timestamps for selected levels as points.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=600.0,
        metavar="SECONDS",
        help="Time limit shown on the x-axis and used to clip timestamps. Default: 600.",
    )
    parser.add_argument(
        "--time-limit-min",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="Minimum x-axis length per instance. If all algorithms finish earlier, extend to this value. Default: 30.",
    )
    parser.add_argument(
        "--output-ext",
        default="png",
        metavar="EXT",
        help="Default output extension when --output is omitted, e.g. png or pdf. Default: png.",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    csv_path = root_dir / "_results" / "timestamp.csv"
    finished_csv_path = root_dir / "_results" / "timestamp_finished.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing timestamp file: {csv_path}")

    df = load_timestamps(csv_path)
    df = filter_timestamps(df, args.tmax, args.include_exp, args.filter_exp, args.control_size)
    finished_df = load_optional_timestamps(finished_csv_path)
    if finished_df is not None:
        finished_df = filter_timestamps(
            finished_df,
            args.tmax,
            args.include_exp,
            args.filter_exp,
            args.control_size,
        )
    if df.empty:
        raise ValueError("No rows remain after filtering.")

    fig = plot_grid(
        df,
        finished_df=finished_df,
        tmax=args.tmax,
        time_limit=args.time_limit,
        time_limit_min=args.time_limit_min,
        finished_levels=args.finished_levels,
        title=None,
    )

    output_ext = str(args.output_ext).lstrip(".") or "png"
    output_path = (
        Path(args.output)
        if args.output
        else root_dir / "_results" / f"agg_comptime_plot_T{args.tmax}.{output_ext}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
