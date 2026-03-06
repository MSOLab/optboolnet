from typing import Any, List, Optional
import os
import numpy as np
from optboolnet.log import EnumBendersStep, EnumCutType, BendersLogger
from optboolnet.instances import iter_bn_in_repo
import pandas as pd

_build_log_columns = ["log_level"] + BendersLogger.build_log_columns
_solve_log_columns = ["log_level"] + BendersLogger.solve_log_columns
_cut_log_columns = ["log_level"] + BendersLogger.cut_log_columns
inst_list = ["S1", "S2", "S3", "S4", "M1", "M2", "M3", "L1", "L2", "L3", "L4"]


def load_pbn_solution(fpath: str, option_names, options):
    def get_sec(time_str: str, shift=0.0):
        """Get seconds from time."""
        hhmmss, milisec = time_str.split(".")
        h, m, s = hhmmss.split(":")
        return round(
            int(h) * 3600
            + int(m) * 60
            + int(s)
            + float("0" + milisec) / (10**6)
            + shift,
            6,
        )

    data_list = list()
    for inst in inst_list:
        inst_data_list = list()
        with open(f"{fpath}/{inst}.csv", "r") as _f:
            lines = iter(_f.readlines())
            for _ in range(3):
                next(lines)  # skip the first 3 lines
            target_core_size = 0
            count = 0
            for line in lines:
                contents = line[15:]
                if contents[:2] == "-1":
                    continue
                if contents[:8] == "Checking":
                    target_core_size = int(contents[-2])
                    if target_core_size > 0:
                        inst_data_list.append(
                            [inst, target_core_size - 1, get_sec(line[:14]), count]
                        )
                    count = 0
                elif contents[:12] == "Intervention":
                    count += 1
                else:
                    Exception()
        if len(inst_data_list) == 0:
            print(inst)
            inst_data_list = [[inst, 0, None, 0]]
        data_list.extend(inst_data_list)

    df = pd.DataFrame(data_list, columns=["inst", "level", "completion_time", "sol"])

    df["inst"] = pd.Categorical(df["inst"], categories=inst_list)
    for opt_name, opt in zip(option_names, options):
        df[opt_name] = opt
    return df


class BendersAnalysis:
    def __init__(
        self,
        build_log_fname: str,
        solve_log_fname: str,
        cut_log_fname: str,
        option_names: List[str],
        options: List[Any],
        inst: str,
    ) -> None:
        self.option_names = option_names
        self.options = options
        self.inst = inst
        self._key_list = self.option_names + ["inst"]
        self.build_log = self.rename_exp(self._read_log_csv(build_log_fname, _build_log_columns))
        self.solve_log = self.rename_exp(self._read_log_csv(solve_log_fname, _solve_log_columns))
        self.cut_log = self.rename_exp(self._read_log_csv(cut_log_fname, _cut_log_columns))

        self._prepare_log_types()
        self._prepare_metric_tables()

    @property
    def key_list(self):
        return self._key_list

    @property
    def key_len(self):
        return len(self.key_list)

    @staticmethod
    def _read_log_csv(path: str, default_columns: list[str]) -> pd.DataFrame:
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            return pd.DataFrame(columns=default_columns[1:])

    def rename_exp(self, df: pd.DataFrame):
        if "experiment" in df.columns:
            df = df.drop(columns=["experiment"])
        else:
            df = df.copy()
        df["inst"] = self.inst
        for opt_name, opt in zip(self.option_names, self.options):
            df[opt_name] = opt
        return df

    def _prepare_log_types(self):
        for df in (self.build_log, self.solve_log, self.cut_log):
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
            if "level" in df.columns:
                df["level"] = pd.to_numeric(df["level"], errors="coerce")

        if "step" in self.solve_log.columns:
            self.solve_log["step"] = self.solve_log["step"].astype("category")
        if "solve_time" in self.solve_log.columns:
            self.solve_log["solve_time"] = pd.to_numeric(self.solve_log["solve_time"], errors="coerce")
        if "feasible" in self.solve_log.columns:
            self.solve_log["feasible"] = self.solve_log["feasible"].astype(str).str.strip().str.lower().map(
                {"true": True, "false": False}
            )

        if "model" in self.solve_log.columns:
            model_clean = self.solve_log["model"].astype(str).str.replace("'", "", regex=False)
            self.solve_log["model"] = model_clean
            self.solve_log["model_num"] = pd.to_numeric(model_clean, errors="coerce")

        if "cut_type" in self.cut_log.columns:
            self.cut_log["cut_type"] = self.cut_log["cut_type"].astype("category")
        if "cut_strength" in self.cut_log.columns:
            self.cut_log["cut_strength"] = pd.to_numeric(self.cut_log["cut_strength"], errors="coerce")

    def _prepare_metric_tables(self):
        key_list = self.key_list
        key_level = key_list + ["level"]
        metrics: dict[str, pd.DataFrame] = {}

        # solution_count
        solution_count = (
            self.cut_log.loc[
                self.cut_log.get("cut_type", pd.Series(index=self.cut_log.index, dtype=object))
                == EnumCutType.MINIMALITY.name,
                key_level + ["cut_strength"],
            ]
            .groupby(key_level, observed=True)["cut_strength"]
            .count()
            .rename("sol")
            .reset_index()
        )
        if solution_count.empty:
            solution_count = pd.DataFrame(
                [[*self.options, self.inst, 0, 0]],
                columns=key_level + ["sol"],
            )
        metrics["solution_count"] = solution_count

        # build_time
        metrics["build_time"] = (
            self.build_log[key_list + ["timestamp"]]
            .groupby(key_list, observed=True)["timestamp"]
            .max()
            .rename("build_time")
            .reset_index()
        )

        # completion_time
        if self.solve_log.empty:
            metrics["completion_time"] = pd.DataFrame(
                [[*self.options, self.inst, 0, np.nan, False]],
                columns=key_level + ["completion_time", "level_finished"],
            )
        else:
            completion_time = (
                self.solve_log[key_level + ["timestamp"]]
                .groupby(key_level, observed=True)["timestamp"]
                .max()
                .rename("completion_time")
                .reset_index()
            )
            finished_levels = set(
                self.solve_log.loc[
                    self.solve_log["step"] == EnumBendersStep.FINISHED.name, "level"
                ].dropna()
            )
            completion_time["level_finished"] = completion_time["level"].isin(finished_levels)
            metrics["completion_time"] = completion_time

        # computation_time
        metrics["computation_time"] = (
            self.solve_log.loc[
                self.solve_log["step"] != EnumBendersStep.FINISHED.name,
                key_list + ["step", "solve_time"],
            ]
            .groupby(key_list + ["step"], observed=True)["solve_time"]
            .sum()
            .reset_index()
        )

        # count_cuts
        metrics["count_cuts"] = (
            self.cut_log.loc[
                self.cut_log["cut_type"] != EnumCutType.MINIMALITY.name,
                key_level + ["cut_type"],
            ]
            .groupby(key_level + ["cut_type"], observed=True)
            .size()
            .rename("count_cuts")
            .reset_index()
        )

        # avg_cuts
        metrics["avg_cuts"] = (
            self.cut_log.loc[
                self.cut_log["cut_type"] != EnumCutType.MINIMALITY.name,
                key_list + ["cut_type", "cut_strength"],
            ]
            .groupby(key_list + ["cut_type"], observed=True)["cut_strength"]
            .mean()
            .rename("num_literals")
            .reset_index()
        )

        # count_attractor_size
        if "model_num" not in self.solve_log.columns:
            metrics["count_attractor_size"] = pd.DataFrame(columns=key_list + ["model", "attractors"])
        else:
            count_df = (
                self.solve_log.loc[
                    (self.solve_log["step"] == EnumBendersStep.LOWER_LEVEL_PROBLEM.name)
                    & (self.solve_log["model_num"].notna()),
                    key_list + ["model_num", "timestamp"],
                ]
                .groupby(key_list + ["model_num"], observed=True)["timestamp"]
                .count()
                .rename("attractors")
                .reset_index()
            )
            if count_df.empty:
                metrics["count_attractor_size"] = pd.DataFrame(columns=key_list + ["model", "attractors"])
            else:
                sol_sum = (
                    metrics["solution_count"]
                    .groupby(key_list, observed=True)["sol"]
                    .sum()
                    .reset_index()
                )
                merged_df = count_df.merge(sol_sum, on=key_list, how="left")
                merged_df["sol"] = merged_df["sol"].fillna(0)
                merged_df["attractors"] = merged_df["attractors"] - merged_df["sol"]
                merged_df["attractors"] = (
                    merged_df["attractors"] - merged_df["attractors"].shift(-1).fillna(0)
                ).astype(int)
                merged_df = merged_df[merged_df["attractors"] > 0].copy()
                merged_df = merged_df.rename(columns={"model_num": "model"})
                merged_df["model"] = merged_df["model"].astype(int)
                metrics["count_attractor_size"] = merged_df[key_list + ["model", "attractors"]]

        # separation_success
        metrics["separation_success"] = (
            self.solve_log.loc[
                self.solve_log["step"] == EnumBendersStep.SEPARATION_PROBLEM.name,
                key_list + ["step", "feasible"],
            ]
            .groupby(key_list + ["step", "feasible"], observed=True)
            .size()
            .rename("success")
            .reset_index()
        )

        # max_level
        metrics["max_level"] = (
            self.solve_log.loc[
                self.solve_log["step"] == EnumBendersStep.FINISHED.name,
                key_list + ["level"],
            ]
            .groupby(key_list, observed=True)["level"]
            .max()
            .rename("max_level")
            .reset_index()
        )

        self._metric_tables = metrics

    def get_metric_tables(self, metric_names: Optional[list[str]] = None) -> dict[str, pd.DataFrame]:
        names = metric_names or list(self._metric_tables.keys())
        return {name: self._metric_tables[name] for name in names}

    @property
    def solution_count(self):
        return self._metric_tables["solution_count"]

    @property
    def build_time(self):
        return self._metric_tables["build_time"]

    @property
    def completion_time(self):
        return self._metric_tables["completion_time"]

    @property
    def computation_time(self):
        return self._metric_tables["computation_time"]

    @property
    def count_cuts(self):
        return self._metric_tables["count_cuts"]

    @property
    def avg_cuts(self):
        return self._metric_tables["avg_cuts"]

    @property
    def count_attractor_size(self):
        return self._metric_tables["count_attractor_size"]

    @property
    def separation_success(self):
        return self._metric_tables["separation_success"]

    @property
    def max_level(self):
        return self._metric_tables["max_level"]


class Experiment:
    def __init__(
        self, work_dir: str, alg: str, option_names: List[str], options: List[Any]
    ) -> None:
        self.work_dir = work_dir
        self.alg = alg
        self.option_names = option_names
        self.options = options
        self.log_list = [
            BendersAnalysis(
                f"{self.exp_path}\\{inst}\\log__build.txt",
                f"{self.exp_path}\\{inst}\\log__solve.txt",
                f"{self.exp_path}\\{inst}\\log__cut.txt",
                option_names,
                options,
                inst,
            )
            for inst, bn in iter_bn_in_repo()
            if os.path.exists(f"{self.exp_path}\\{inst}")
        ]

    @property
    def exp_path(self):
        return f"{self.work_dir}\\{self.alg}"

    def get_agg_table(self, attr_name: str) -> pd.DataFrame:
        """
        solution_count,
        completion_time,
        build_time,
        count_cuts,
        avg_cuts,
        computation_time,
        separation_success,
        count_attractor_size,
        max_level
        """

        if not self.log_list:
            return pd.DataFrame()
        return pd.concat(
            (getattr(log_analysis, attr_name) for log_analysis in self.log_list),
            axis=0,
            ignore_index=True,
        )

    def get_agg_tables(self, attr_names: List[str]) -> dict[str, pd.DataFrame]:
        if not self.log_list:
            return {name: pd.DataFrame() for name in attr_names}
        frames_by_metric = {name: [] for name in attr_names}
        for log_analysis in self.log_list:
            metric_tables = log_analysis.get_metric_tables(attr_names)
            for name in attr_names:
                frames_by_metric[name].append(metric_tables[name])
        return {
            name: pd.concat(frames_by_metric[name], axis=0, ignore_index=True)
            for name in attr_names
        }


_METRICS = [
    "solution_count",
    "completion_time",
    "build_time",
    "count_cuts",
    "avg_cuts",
    "computation_time",
    "separation_success",
    "count_attractor_size",
    "max_level",
]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Aggregate experiment logs into a table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # BEN Benders experiment, print completion_time to stdout
  python analysis.py _experiments/260302_full/BEN_1_agg benders \\
      --option-names alg --options BEN

  # MibS experiment, save solution_count to CSV
  python analysis.py _experiments/260302_full/MibS_1 MibS \\
      --option-names alg --options MibS \\
      --metric solution_count --output results/mibs_sols.csv
""",
    )
    parser.add_argument(
        "work_dir",
        help="Path to the experiment folder (parent of the alg subfolder).",
    )
    parser.add_argument(
        "alg",
        help="Algorithm subfolder name inside work_dir (e.g. 'benders' or 'MibS').",
    )
    parser.add_argument(
        "--metric",
        default="completion_time",
        choices=_METRICS,
        help="Metric to compute (default: completion_time).",
    )
    parser.add_argument(
        "--option-names",
        nargs="+",
        default=[],
        metavar="NAME",
        help="Column names for experiment labels (e.g. --option-names alg T_max).",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        default=[],
        metavar="VALUE",
        help="Values matching --option-names (e.g. --options BEN 1).",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Write results to this CSV file instead of printing to stdout.",
    )

    args = parser.parse_args()

    if len(args.option_names) != len(args.options):
        parser.error("--option-names and --options must have the same number of items.")

    exp = Experiment(args.work_dir, args.alg, args.option_names, args.options)
    df = exp.get_agg_table(args.metric)

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True) if os.path.dirname(args.output) else None
        df.to_csv(args.output, index=False)
        print(f"Saved to {args.output}")
    else:
        print(df.to_string(index=False))
