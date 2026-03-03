import argparse
import datetime
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

from optboolnet.boolnet import Control
from optboolnet.checking import nusmv_check_phenotype
from optboolnet.instances import load_bn_in_repo, _INSTANCE_LIST_FULL


_ALGO_SUBDIRS = ["benders", "MibS"]

# Per-worker global: loaded once per process via _init_worker
_worker_bn = None


def _init_worker(inst_name: str):
    global _worker_bn
    _worker_bn = load_bn_in_repo(inst_name)


def _check_ctrl(ctrl: Control):
    ok = nusmv_check_phenotype(_worker_bn, control=ctrl)
    return ctrl, ok


def _find_sol_path(work_dir: str, inst: str):
    for subdir in _ALGO_SUBDIRS:
        path = os.path.join(work_dir, subdir, inst, "sol.json")
        if os.path.exists(path):
            return path
    return None


def verify_work_dir(work_dir: str, output_file: str, instances: List[str], workers: int):
    print(work_dir)
    for inst in instances:
        sol_path = _find_sol_path(work_dir, inst)
        if sol_path is None:
            print(f"\t{inst}: sol.json not found, skipping")
            with open(output_file, "a", encoding="utf-8") as _f:
                _f.write(f"{work_dir},{inst},MISSING\n")
            continue
        print(inst)
        ctrl_list: List[Control] = []
        with open(sol_path, "r") as _f:
            for sol_list in json.load(_f).values():
                for sol in sol_list:
                    ctrl_list.append(Control(sol))

        n = len(ctrl_list)
        print(f"\t{n} controls, {workers} workers")

        incorrect = []
        completed = 0
        last_pct = 0

        # Each worker process loads bn once via _init_worker; ctrl items are
        # distributed across workers and checked concurrently.
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(inst,),
        ) as executor:
            futures = {executor.submit(_check_ctrl, ctrl): ctrl for ctrl in ctrl_list}
            for future in as_completed(futures):
                ctrl, ok = future.result()
                completed += 1
                if not ok:
                    incorrect.append(ctrl)
                    print(f"\tincorrect: {ctrl}")
                pct = completed * 100 // n
                milestone = pct // 10 * 10
                if milestone > last_pct:
                    print(f"\t{milestone}% ({completed}/{n})")
                    last_pct = milestone

        if incorrect:
            with open(output_file, "a", encoding="utf-8") as _f:
                for ctrl in incorrect:
                    _f.write(f"{work_dir},{inst},{ctrl}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=(
            "Verify controls from sol.json in parallel: "
            "check all attractors satisfy the phenotype."
        )
    )
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--work_dirs",
        nargs="+",
        metavar="DIR",
        help="One or more explicit work directories to verify.",
    )
    group.add_argument(
        "--root_dir",
        metavar="DIR",
        help=(
            "Root directory; every subdirectory that contains a 'benders/' or 'MibS/' folder "
            "is treated as a work directory."
        ),
    )
    ap.add_argument(
        "--instances",
        nargs="+",
        metavar="INST",
        default=_INSTANCE_LIST_FULL,
        help=f"Instances to verify (default: all). Available: {', '.join(_INSTANCE_LIST_FULL)}",
    )
    ap.add_argument(
        "--output",
        metavar="FILE",
        default="_experiments/verify_control_log.txt",
        help="Output file to write incorrect controls to (default: _experiments/verify_control_log.txt)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        metavar="N",
        help=f"Number of parallel NuSMV processes (default: cpu_count={os.cpu_count()})",
    )
    args = ap.parse_args()

    _timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _dirs_str = ", ".join(args.work_dirs) if args.work_dirs else f"root={args.root_dir}"
    _insts_str = ", ".join(args.instances)
    with open(args.output, "a", encoding="utf-8") as _f:
        _f.write(
            f"# [{_timestamp}] work_dirs=[{_dirs_str}] instances=[{_insts_str}]"
            f" workers={args.workers}\n"
        )

    if args.work_dirs:
        work_dir_list = args.work_dirs
    else:
        root_dir = args.root_dir
        work_dir_list = [
            os.path.join(root_dir, sub)
            for sub in sorted(os.listdir(root_dir))
            if any(
                os.path.isdir(os.path.join(root_dir, sub, algo))
                for algo in _ALGO_SUBDIRS
            )
        ]
        print(f"Found {len(work_dir_list)} work dir(s) under {root_dir}:")
        for d in work_dir_list:
            print(f"  {d}")

    for work_dir in work_dir_list:
        verify_work_dir(work_dir, args.output, args.instances, args.workers)
