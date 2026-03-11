import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List

_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
# Avoid shadowing third-party modules with files from this directory.
sys.path = [p for p in sys.path if os.path.abspath(p) != _THIS_DIR]
_SRC_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from optboolnet.algorithm import LongestAttractorControl
from optboolnet.config import SolverConfig
from optboolnet.instances import _INSTANCE_LIST_FULL, load_bn_in_repo


def parse_args():
    parser = argparse.ArgumentParser(
        description="Solve longest joint control-attractor single-level MILP for instances."
    )
    parser.add_argument(
        "--tmax",
        type=int,
        required=True,
        help="Maximum attractor length considered by the model.",
    )
    parser.add_argument(
        "--instances",
        nargs="+",
        default=list(_INSTANCE_LIST_FULL),
        help="Instance names to run (default: S1..S4, M1..M3, L1..L4).",
    )
    parser.add_argument(
        "--phenotype-mode",
        choices=["violating", "none"],
        default="violating",
        help="violating: enforce p=0, none: no phenotype constraint.",
    )
    parser.add_argument(
        "--solver-name",
        default="gurobi_persistent",
        help="Pyomo solver name.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=600.0,
        help="Solver time limit in seconds (default: 600).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Solver thread count.",
    )
    parser.add_argument(
        "--tee",
        action="store_true",
        help="Print solver logs.",
    )
    parser.add_argument(
        "--max-control-size",
        type=int,
        default=None,
        help="Maximum number of controlled variables allowed (optional).",
    )
    parser.add_argument(
        "--mip-progress-log",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write per-instance MIP progress logs (default: enabled).",
    )
    parser.add_argument(
        "--state-periodicity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Optional tightening: if period t is selected, enforce x[i,1]=x[i,t'] "
            "for t' in {1+t,1+2t,...}."
        ),
    )
    return parser.parse_args()


def make_solver_config(args) -> SolverConfig:
    solver_config = SolverConfig()
    solver_config.solver_name = args.solver_name
    solver_config.time_limit = args.time_limit
    solver_config.threads = args.threads
    solver_config.tee = args.tee
    return solver_config


def clone_solver_config(base: SolverConfig) -> SolverConfig:
    cloned = SolverConfig()
    for key, value in base.__dict__.items():
        setattr(cloned, key, value)
    return cloned


def build_output_dir(tmax: int) -> str:
    base_dir = os.path.abspath(os.path.dirname(__file__))
    stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    output_dir = os.path.join(
        base_dir,
        "results",
        "longest_attractor",
        f"Tmax_{tmax}",
        stamp,
    )
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def validate_instances(instances: List[str]) -> None:
    invalid = sorted(set(instances) - set(_INSTANCE_LIST_FULL))
    if invalid:
        raise ValueError(
            f"Invalid instances: {invalid}. Available instances: {_INSTANCE_LIST_FULL}"
        )


def write_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    args = parse_args()
    if args.tmax < 1:
        raise ValueError("--tmax must be >= 1")
    if args.time_limit is not None and args.time_limit <= 0:
        raise ValueError("--time-limit must be > 0")
    if args.threads is not None and args.threads < 1:
        raise ValueError("--threads must be >= 1")
    if args.max_control_size is not None and args.max_control_size < 0:
        raise ValueError("--max-control-size must be >= 0")

    validate_instances(args.instances)
    solver_config_base = make_solver_config(args)
    output_dir = build_output_dir(args.tmax)
    mip_logs_dir = os.path.join(output_dir, "mip_logs")
    if args.mip_progress_log:
        os.makedirs(mip_logs_dir, exist_ok=True)

    summary = {
        "meta": {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "tmax": args.tmax,
            "instances": args.instances,
            "phenotype_mode": args.phenotype_mode,
            "solver_name": args.solver_name,
            "time_limit": args.time_limit,
            "threads": args.threads,
            "tee": args.tee,
            "max_control_size": args.max_control_size,
            "mip_progress_log": args.mip_progress_log,
            "state_periodicity": args.state_periodicity,
            "mip_logs_dir": mip_logs_dir if args.mip_progress_log else None,
            "output_dir": output_dir,
        },
        "results": {},
    }

    for inst in args.instances:
        print(f"[RUN] {inst}", flush=True)
        bn = load_bn_in_repo(inst)
        manager = LongestAttractorControl(inst, bn)
        solver_config = clone_solver_config(solver_config_base)
        mip_log_path = None
        if args.mip_progress_log and "gurobi" in solver_config.solver_name.lower():
            mip_log_path = os.path.join(mip_logs_dir, f"{inst}.log")
            # Option key follows Gurobi parameter naming in Pyomo solver options.
            setattr(solver_config, "LogFile", mip_log_path)
        result = manager.solve(
            tmax=args.tmax,
            phenotype_mode=args.phenotype_mode,
            solver_config=solver_config,
            max_control_size=args.max_control_size,
            use_state_periodicity=args.state_periodicity,
        )
        result["mip_log_file"] = mip_log_path
        summary["results"][inst] = result
        write_json(os.path.join(output_dir, f"{inst}.json"), result)

    write_json(os.path.join(output_dir, "summary.json"), summary)
    print(f"[DONE] Wrote results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
