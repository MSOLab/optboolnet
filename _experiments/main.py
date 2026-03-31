import argparse
import concurrent.futures
import json
import os
import sys
import traceback
from datetime import datetime
from typing import Optional, Tuple

_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
# Avoid shadowing third-party modules with files from this directory.
sys.path = [p for p in sys.path if os.path.abspath(p) != _THIS_DIR]

from optboolnet.config import BendersConfig, MibSBilevelConfig
# from optboolnet.config import PyBoolNetConfig
from optboolnet.instances import load_bn_in_repo

CONFIG_CLASS = {
    "benders": BendersConfig,
    "mibs": MibSBilevelConfig,
    # "pbn": PyBoolNetConfig,
}

ALGORITHM_WORKDIR = {
    "benders": "benders",
    "mibs": "MibS",
    "pbn": "pbn",
}

CONFIG_NAME_TO_ALGORITHM = {
    "BendersConfig": "benders",
    "MibSBilevelConfig": "mibs",
    "PyBoolNetConfig": "pbn",
}


def run_benders(inst: str, exp_name: str, config: BendersConfig) -> None:
    from optboolnet.algorithm import BendersAttractorControl

    bn = load_bn_in_repo(inst)
    attr_ctrl_manager = BendersAttractorControl(exp_name, bn, config.logging_config)
    attr_ctrl_manager.allow_empty_attractor = config.allow_empty_attractor
    attr_ctrl_manager.solve_separation = config.solve_separation
    attr_ctrl_manager.preprocess_max_forbidden_trap_space = config.preprocess_max_forbidden_trap_space
    attr_ctrl_manager.separation_heuristic = config.separation_heuristic
    attr_ctrl_manager.use_high_point_relaxation = config.use_high_point_relaxation
    attr_ctrl_manager.total_time_limit = config.total_time_limit
    attr_ctrl_manager.use_aggregated_LLP = config.use_aggregated_LLP
    attr_ctrl_manager.get_control_strategies(
        max_control_size=config.max_control_size,
        max_length=config.max_length,
        master_solver_config=config.master_solver_config,
        LLP_solver_config=config.LLP_solver_config,
        separation_solver_config=config.separation_solver_config,
    )


def run_mibs(inst: str, config: MibSBilevelConfig) -> None:
    try:
        from optboolnet.mibs import MibSAttractorControl
    except ModuleNotFoundError as exc:
        if exc.name == "imp":
            raise RuntimeError(
                "MibS dependencies require Python <= 3.11 (module 'imp' removed in Python 3.12)."
            ) from exc
        raise

    model_name = "INTERDICTION" if config.use_interdiction else "BILEVEL"
    bn = load_bn_in_repo(inst)
    attr_ctrl_manager = MibSAttractorControl(model_name, bn, config)
    attr_ctrl_manager.total_time_limit = config.total_time_limit
    attr_ctrl_manager.get_control_strategies()


# def run_pbn(inst: str, exp_name: str, config: PyBoolNetConfig) -> None:
#     from optboolnet.pyboolnet import PyBoolNetAttractorControl

#     bn = load_bn_in_repo(inst)
#     attr_ctrl_manager = PyBoolNetAttractorControl(exp_name, bn, config.logging_config)
#     attr_ctrl_manager.total_time_limit = config.total_time_limit
#     attr_ctrl_manager.update = config.update
#     attr_ctrl_manager.max_output_trapspaces = config.max_output_trapspaces
#     attr_ctrl_manager.get_control_strategies(
#         max_control_size=config.max_control_size,
#         starting_length=config.starting_length,
#     )


def infer_algorithm(config_payload: dict, exp_dir: str) -> str:
    config_name = config_payload.get("__name__")
    if config_name in CONFIG_NAME_TO_ALGORITHM:
        return CONFIG_NAME_TO_ALGORITHM[config_name]

    # Fallback for legacy config files with no __name__ metadata.
    if "update" in config_payload or "max_output_trapspaces" in config_payload:
        return "pbn"
    if "use_interdiction" in config_payload or "use_valid_cuts" in config_payload:
        return "mibs"
    if "solve_separation" in config_payload:
        return "benders"

    raise ValueError(f"Cannot infer algorithm from config in: {exp_dir}")


def validate_explicit_time_limit(config_payload: dict, exp_dir: str) -> float:
    if "total_time_limit" not in config_payload:
        raise ValueError(
            f"Missing explicit 'total_time_limit' in {exp_dir}/alg_config.json. "
            "Refusing to use default time limits for safety."
        )
    time_limit = config_payload["total_time_limit"]
    if not isinstance(time_limit, (int, float)) or time_limit <= 0:
        raise ValueError(
            f"Invalid 'total_time_limit' in {exp_dir}/alg_config.json: {time_limit!r}. "
            "Expected a positive number."
        )
    return float(time_limit)


def load_algorithm_config(exp_dir: str):
    config_path = os.path.join(exp_dir, "alg_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config file: {config_path}")

    with open(config_path, "r") as fp:
        payload = json.load(fp)
    algorithm = infer_algorithm(payload, exp_dir)
    json_time_limit = validate_explicit_time_limit(payload, exp_dir)
    return algorithm, CONFIG_CLASS[algorithm].from_dict(payload), json_time_limit


def apply_time_limit_override(config, time_limit: Optional[float]) -> None:
    if time_limit is None:
        return

    config.total_time_limit = time_limit
    for solver_attr in ("solver_config", "master_solver_config", "LLP_solver_config", "separation_solver_config"):
        solver_cfg = getattr(config, solver_attr, None)
        if solver_cfg is not None and hasattr(solver_cfg, "time_limit"):
            solver_cfg.time_limit = time_limit


def run_task(task: Tuple[str, str, str, str, Optional[float]]) -> str:
    algorithm, exp_dir, exp_name, inst, time_limit_override = task
    _, config, _ = load_algorithm_config(exp_dir)
    apply_time_limit_override(config, time_limit_override)

    work_dir = os.path.join(exp_dir, ALGORITHM_WORKDIR[algorithm], inst)
    os.makedirs(work_dir, exist_ok=True)
    config.logging_config.fpath = work_dir

    if algorithm == "benders":
        run_benders(inst, exp_name, config)
    elif algorithm == "mibs":
        run_mibs(inst, config)
    elif algorithm == "pbn":
        run_pbn(inst, exp_name, config)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    return f"{exp_name}/{inst}"


def parse_args():
    ap = argparse.ArgumentParser(description="Unified benchmark entry point for benders, mibs, and pbn.")
    ap.add_argument("--root-dir", required=True, help="Root directory that contains experiment subdirectories.")
    ap.add_argument("--subdirs", nargs="+", required=True, help="Experiment subdirectory names.")
    ap.add_argument("--instances", nargs="+", required=True, help="Instance names to run.")
    ap.add_argument("--workers", type=int, default=1, help="Number of worker processes. Use 1 for sequential mode.")
    ap.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Override total_time_limit (seconds) from alg_config.json for all runs.",
    )
    return ap.parse_args()


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.time_limit is not None and args.time_limit <= 0:
        raise ValueError("--time-limit must be > 0")

    cur_dir = os.path.abspath(os.path.dirname(__file__))
    root_dir = args.root_dir if os.path.isabs(args.root_dir) else os.path.join(cur_dir, args.root_dir)
    if not os.path.isdir(root_dir):
        raise NotADirectoryError(f"Root directory not found: {root_dir}")

    tasks = []
    detected_algorithms = set()
    subdir_time_limits = {}
    for subdir in args.subdirs:
        exp_dir = os.path.join(root_dir, subdir)
        if not os.path.isdir(exp_dir):
            raise NotADirectoryError(f"Experiment subdirectory not found: {exp_dir}")
        if not os.path.exists(os.path.join(exp_dir, "alg_config.json")):
            raise FileNotFoundError(f"Missing alg_config.json in: {exp_dir}")
        algorithm, _, json_time_limit = load_algorithm_config(exp_dir)
        detected_algorithms.add(algorithm)
        subdir_time_limits[subdir] = json_time_limit
        for inst in args.instances:
            tasks.append((algorithm, exp_dir, subdir, inst, args.time_limit))

    alg_list = ", ".join(sorted(detected_algorithms))
    run_tag = datetime.now().strftime("%y%m%d_%H%M%S")
    run_log_path = os.path.join(root_dir, f"run_times__{run_tag}.log")
    run_args_path = os.path.join(root_dir, f"run_args__{run_tag}.json")

    run_args_payload = {
        "timestamp": timestamp(),
        "argv": sys.argv[1:],
        "parsed_args": {
            "root_dir": args.root_dir,
            "resolved_root_dir": root_dir,
            "subdirs": args.subdirs,
            "instances": args.instances,
            "workers": args.workers,
            "time_limit": args.time_limit,
        },
        "subdir_time_limits_json": subdir_time_limits,
    }
    with open(run_args_path, "w", encoding="utf-8") as run_args_fp:
        json.dump(run_args_payload, run_args_fp, indent=2)

    with open(run_log_path, "a", encoding="utf-8") as run_log_fp:
        def log_line(msg: str) -> None:
            lines = msg.splitlines() or [""]
            for line in lines:
                stamped = f"{timestamp()} {line}"
                print(stamped, flush=True)
                run_log_fp.write(stamped + "\n")
            run_log_fp.flush()

        log_line("[RUN START]")
        log_line(f"[RUN LOG] {run_log_path}")
        log_line(f"[RUN ARGS] {run_args_path}")
        limit_msg = "per-subdir-json (strict)" if args.time_limit is None else f"{args.time_limit}s (runtime override)"
        log_line(
            f"Launching {len(tasks)} runs with algorithms={alg_list}, workers={args.workers}, "
            f"time_limit={limit_msg}"
        )
        if args.time_limit is None:
            subdir_limits_str = ", ".join(
                f"{subdir}:{subdir_time_limits[subdir]}s" for subdir in args.subdirs
            )
            log_line(f"[TIME LIMITS][JSON] {subdir_limits_str}")
        else:
            log_line(f"[TIME LIMITS][OVERRIDE] all subdirs -> {args.time_limit}s (json unchanged)")

        failures = []
        if args.workers == 1:
            for task in tasks:
                task_name = f"{task[2]}/{task[3]}"
                log_line(f"[TASK START] {task_name}")
                try:
                    done = run_task(task)
                    log_line(f"[TASK END][OK] {done}")
                except Exception:
                    log_line(f"[TASK END][FAIL] {task_name}")
                    failures.append((task[2], task[3], traceback.format_exc()))
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
                task_iter = iter(tasks)
                in_flight = {}

                # Keep up to `workers` instances running; submit the next one as soon as any finishes.
                for _ in range(min(args.workers, len(tasks))):
                    task = next(task_iter, None)
                    if task is None:
                        break
                    task_name = f"{task[2]}/{task[3]}"
                    log_line(f"[TASK START] {task_name}")
                    in_flight[executor.submit(run_task, task)] = task

                while in_flight:
                    done_futures, _ = concurrent.futures.wait(
                        in_flight,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in done_futures:
                        task = in_flight.pop(future)
                        task_name = f"{task[2]}/{task[3]}"
                        try:
                            done = future.result()
                            log_line(f"[TASK END][OK] {done}")
                        except Exception:
                            log_line(f"[TASK END][FAIL] {task_name}")
                            failures.append((task[2], task[3], traceback.format_exc()))

                        next_task = next(task_iter, None)
                        if next_task is not None:
                            next_name = f"{next_task[2]}/{next_task[3]}"
                            log_line(f"[TASK START] {next_name}")
                            in_flight[executor.submit(run_task, next_task)] = next_task

        log_line("[RUN END]")

        if failures:
            log_line("\nFailures:")
            for subdir, inst, err in failures:
                log_line(f"[FAIL] {subdir}/{inst}")
                log_line(err)
            raise SystemExit(1)

        log_line("All runs completed.")


if __name__ == "__main__":
    main()
