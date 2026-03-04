import argparse
import datetime
import json
import os
from typing import List

from optboolnet.boolnet import Control
from optboolnet.checking import nusmv_check_phenotype
from optboolnet.instances import load_bn_in_repo, _INSTANCE_LIST_FULL


_ALGO_SUBDIRS = ["benders", "MibS"]


def _find_sol_path(work_dir: str, inst: str):
    for subdir in _ALGO_SUBDIRS:
        path = os.path.join(work_dir, subdir, inst, "sol.json")
        if os.path.exists(path):
            return path
    return None


def verify_work_dir(work_dir: str, output_file: str, instances: List[str]):
    print(work_dir)
    for inst in instances:
        sol_path = _find_sol_path(work_dir, inst)
        if sol_path is None:
            print(f"\t{inst}: sol.json not found, skipping")
            continue
        print(inst)
        bn = load_bn_in_repo(inst)
        ctrl_list: List[Control] = []
        with open(sol_path, "r") as _f:
            for sol_list in json.load(_f).values():
                for sol in sol_list:
                    ctrl_list.append(Control(sol))
        print("\t", len(ctrl_list))

        n = len(ctrl_list)
        last_pct = 0
        for i, ctrl in enumerate(ctrl_list):
            if not nusmv_check_phenotype(
                bn,
                control=ctrl,
                property_variant="ctl_ef_ag",
                constrain_controlled_vars=True,
            ):
                print("incorrect", ctrl)
                with open(output_file, "a", encoding="utf-8") as _f:
                    _f.write(f"{work_dir},{inst},{ctrl}\n")
            pct = (i + 1) * 100 // n
            milestone = pct // 10 * 10
            if milestone > last_pct:
                print(f"\t{milestone}% ({i + 1}/{n})")
                last_pct = milestone


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Verify controls from sol.json: check all attractors satisfy the phenotype."
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
    args = ap.parse_args()

    # Append a run-header line to the output file
    _timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _dirs_str = ", ".join(args.work_dirs) if args.work_dirs else f"root={args.root_dir}"
    _insts_str = ", ".join(args.instances)
    with open(args.output, "a", encoding="utf-8") as _f:
        _f.write(
            f"# [{_timestamp}] work_dirs=[{_dirs_str}] instances=[{_insts_str}]\n"
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
        verify_work_dir(work_dir, args.output, args.instances)
