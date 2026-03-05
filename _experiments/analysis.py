from typing import Any, List
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
        self.build_log = self.rename_exp(pd.read_csv(build_log_fname))
        self.solve_log = self.rename_exp(pd.read_csv(solve_log_fname))
        if "model" in self.solve_log.columns:
            model_clean = self.solve_log["model"].astype(str).str.replace("'", "", regex=False)
            self.solve_log["model"] = model_clean
            self.solve_log["model_num"] = pd.to_numeric(model_clean, errors="coerce")
        self.cut_log = self.rename_exp(pd.read_csv(cut_log_fname))

    @property
    def key_list(self):
        return self.option_names + ["inst"]

    @property
    def key_len(self):
        return len(self.key_list)

    def rename_exp(self, df: pd.DataFrame):
        df.drop(columns=["experiment"], inplace=True)
        df["inst"] = self.inst
        for opt_name, opt in zip(self.option_names, self.options):
            df[opt_name] = opt
        return df

    @property
    def solution_count(self):
        _df = (
            self.cut_log.loc[
                self.cut_log["cut_type"] == EnumCutType.MINIMALITY.name,
                self.key_list + ["level", "cut_strength"],
            ]
            .groupby(self.key_list + ["level"])
            .count()
            .rename(columns={"cut_strength": "sol"})
            .reset_index()
        )
        if len(_df) == 0:
            return pd.DataFrame(
                [[*self.options, self.inst, 0, 0]], columns=self.key_list + ["level", "sol"]
            )
        else:
            return _df

    @property
    def build_time(self):
        return (
            self.build_log[self.key_list + ["timestamp"]]
            .groupby(self.key_list)
            .max()
            .rename(columns={"timestamp": "build_time"})
            .reset_index()
        )

    @property
    def completion_time(self):
        if len(self.solve_log) == 0:
            return pd.DataFrame(
                [[*self.options, self.inst, 0, np.nan, False]],
                columns=self.key_list + ["level", "completion_time", "level_finished"],
            )
        time_df = (
            self.solve_log[self.key_list + ["level", "timestamp"]]
            .groupby(self.key_list + ["level"])
            .max()
            .rename(columns={"timestamp": "completion_time"})
            .reset_index()
        )
        finished_levels = self.solve_log.loc[
            self.solve_log["step"] == EnumBendersStep.FINISHED.name, "level"
        ].unique()
        time_df["level_finished"] = time_df["level"].isin(finished_levels)
        return time_df

    @property
    def computation_time(self):
        return (
            self.solve_log.loc[
                self.solve_log["step"] != EnumBendersStep.FINISHED.name,
                self.key_list + ["step", "solve_time"],
            ]
            .groupby(self.key_list + ["step"])
            .sum()
            .rename(columns={"timestamp": "completion_time"})
            .reset_index()
        )

    @property
    def count_cuts(self):
        return (
            self.cut_log.loc[
                self.cut_log["cut_type"] != EnumCutType.MINIMALITY.name,
                self.key_list + ["level", "cut_type", "timestamp"],
            ]
            .groupby(self.key_list + ["level", "cut_type"])
            .count()
            .rename(columns={"timestamp": "count_cuts"})
            .reset_index()
        )

    @property
    def avg_cuts(self):
        return (
            self.cut_log.loc[
                self.cut_log["cut_type"] != EnumCutType.MINIMALITY.name,
                self.key_list + ["level", "cut_type", "cut_strength"],
            ]
            .groupby(self.key_list + ["cut_type"])
            # .agg(["mean", "count"])
            .mean()
            .rename(columns={"cut_strength": "num_literals"})
            .reset_index()
            .drop("level", axis=1)
        )

    @property
    def count_attractor_size(self):
        if "model_num" not in self.solve_log.columns:
            return pd.DataFrame(
                columns=self.key_list + ["model", "attractors"]
            )

        count_df = (
            self.solve_log.loc[
                (self.solve_log["step"] == EnumBendersStep.LOWER_LEVEL_PROBLEM.name)
                & (self.solve_log["model_num"].notna()),
                self.key_list + ["model_num", "timestamp"],
            ]
            .groupby(self.key_list + ["model_num"])
            .count()
            .rename(columns={"timestamp": "attractors"})
        )
        if len(count_df) == 0:
            return pd.DataFrame(columns=self.key_list + ["model", "attractors"])
        merged_df = (
            count_df.reset_index()
            .set_index(self.key_list)
            .merge(
                self.solution_count.groupby(self.key_list)
                .sum()
                .reset_index()
                .set_index(self.key_list),
                left_index=True,
                right_index=True,
                how="left"
            ).copy()
        )
        merged_df["attractors"] = merged_df["attractors"] - merged_df["sol"]
        merged_df["attractors"] = (merged_df["attractors"]  - merged_df["attractors"].shift(-1).fillna(0)).astype(int)
        merged_df = merged_df[merged_df["attractors"] > 0]
        merged_df.drop(["sol","level"], axis=1, inplace=True)
        merged_df = merged_df.rename(columns={"model_num": "model"})
        merged_df["model"] = merged_df["model"].astype(int)
        return merged_df[merged_df["attractors"] > 0]

    @property
    def separation_success(self):
        return (
            self.solve_log.loc[
                self.solve_log["step"] == EnumBendersStep.SEPARATION_PROBLEM.name,
                self.key_list + ["step", "feasible", "solve_time"],
            ]
            .groupby(self.key_list + ["step", "feasible"])
            .count()
            .reset_index()
            .rename(columns={"solve_time": "success"})
        )

    @property
    def max_level(self):
        return (
            self.solve_log.loc[
                self.solve_log["step"] == EnumBendersStep.FINISHED.name,
                self.key_list + ["level"],
            ]
            .groupby(self.key_list)
            .max()
            .rename(columns={"level": "max_level"})
            .reset_index()
        )


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

        return pd.concat(
            (getattr(log_analysis, attr_name) for log_analysis in self.log_list), axis=0
        )


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
