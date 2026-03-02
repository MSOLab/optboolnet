import argparse
import json
import os
from typing import List

from optboolnet.boolnet import Control
from optboolnet.config import SolverConfig
from optboolnet.instances import load_bn_in_repo, _INSTANCE_LIST_FULL
from optboolnet.model import ExtendedAttractorDetectionIP


LLP_solver_config = {
    "solver_name": "gurobi_persistent",
    "save_results": False,
    "tee": False,
    "threads": None,
    "warmstart": False,
    "time_limit": None,
}


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

        length_break = False
        for length in range(4, 101):
            if length_break:
                break
            print(length)
            model_LLP = ExtendedAttractorDetectionIP(
                f"{length}", bn, length, SolverConfig(**LLP_solver_config)
            )
            model_LLP.fix_var(model_LLP.v, 0)
            model_LLP.make_constr_stability_condition()
            model_LLP.make_constr_phenotype_at_all_t()
            model_LLP.set_phenotype_obj()

            for ctrl in ctrl_list:
                model_LLP.fix_control(ctrl)
                if model_LLP.optimize():
                    if model_LLP.p.value == 0:
                        print("incorrect", ctrl)
                        with open(output_file, "a", encoding="utf-8") as _f:
                            _f.write(f"{work_dir},{inst},{length},{ctrl}\n")
                        length_break = True
                        break


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
