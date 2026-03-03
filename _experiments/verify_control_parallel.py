import argparse
import datetime
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

from optboolnet.boolnet import Control
from optboolnet.checking import nusmv_check_phenotype
from optboolnet.instances import load_bn_in_repo, _INSTANCE_LIST_FULL


_ALGO_SUBDIRS = ["benders", "MibS"]

# Per-worker BN cache: {inst_name: bn}. Persists for the lifetime of each
# worker process so each BN is loaded at most once per worker.
_worker_bns = {}


def _check_ctrl_for_inst(inst_name: str, ctrl: Control):
    if inst_name not in _worker_bns:
        _worker_bns[inst_name] = load_bn_in_repo(inst_name)
    bn = _worker_bns[inst_name]
    t0 = time.perf_counter()
    ok = nusmv_check_phenotype(bn, control=ctrl)
    elapsed = time.perf_counter() - t0
    return inst_name, ctrl, ok, elapsed


def _find_sol_path(work_dir: str, inst: str):
    for subdir in _ALGO_SUBDIRS:
        path = os.path.join(work_dir, subdir, inst, "sol.json")
        if os.path.exists(path):
            return path
    return None


def verify_work_dir(work_dir: str, output_file: str, instances: List[str], workers: int):
    print(work_dir)

    # Collect all (inst, ctrl) pairs across every instance up front so that a
    # single pool can draw from all of them and CPU stays fully utilised even
    # when one instance has fewer controls than the number of workers.
    all_pairs: List[tuple] = []
    inst_counts = {}
    for inst in instances:
        sol_path = _find_sol_path(work_dir, inst)
        if sol_path is None:
            print(f"\t{inst}: sol.json not found, skipping")
            with open(output_file, "a", encoding="utf-8") as _f:
                _f.write(f"{work_dir},{inst},MISSING\n")
            continue
        ctrl_list: List[Control] = []
        with open(sol_path, "r") as _f:
            for sol_list in json.load(_f).values():
                for sol in sol_list:
                    ctrl_list.append(Control(sol))
        inst_counts[inst] = len(ctrl_list)
        all_pairs.extend((inst, ctrl) for ctrl in ctrl_list)
        print(f"\t{inst}: {len(ctrl_list)} controls")

    total = len(all_pairs)
    if total == 0:
        return
    print(f"\tTotal: {total} checks across {len(inst_counts)} instance(s), {workers} workers")

    completed_by_inst = {inst: 0 for inst in inst_counts}
    last_pct_by_inst = {inst: 0 for inst in inst_counts}

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_check_ctrl_for_inst, inst, ctrl): (inst, ctrl)
            for inst, ctrl in all_pairs
        }
        for future in as_completed(futures):
            inst, ctrl, ok, elapsed = future.result()
            completed_by_inst[inst] += 1

            if not ok:
                print(f"\tincorrect [{inst}] {ctrl} ({elapsed:.1f}s)")
                with open(output_file, "a", encoding="utf-8") as _f:
                    _f.write(f"{work_dir},{inst},{ctrl},{elapsed:.1f}s\n")

            # Per-instance progress milestones
            n = inst_counts[inst]
            c = completed_by_inst[inst]
            pct = c * 100 // n
            milestone = pct // 10 * 10
            if milestone > last_pct_by_inst[inst]:
                print(f"\t[{inst}] {milestone}% ({c}/{n})")
                last_pct_by_inst[inst] = milestone


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
