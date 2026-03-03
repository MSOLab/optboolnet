import json
import os

import optboolnet
from optboolnet.config import BendersConfig
from optboolnet.instances import load_bn_in_repo
from optboolnet.algorithm import BendersAttractorControl


def main_benders(inst: str, work_dir: str, param_str: str, config: BendersConfig):
    bn = load_bn_in_repo(inst)
    attr_ctrl_manager = BendersAttractorControl(param_str, bn, config.logging_config)
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


if __name__ == "__main__":
    from argparse import ArgumentParser
    import shutil

    cur_dir = os.path.dirname(__file__)
    cur_fname = os.path.basename(__file__).split("/")[-1][:-3]
    package_dir = os.path.dirname(optboolnet.__file__)

    ap = ArgumentParser()
    ap.add_argument("-n", help="experiment_name", required=False)
    ap.add_argument("-f", help="experiment_folder", required=False)
    ap.add_argument("-i", help="instances", default="")
    ap.add_argument("--time_limit", help="time limit in seconds", default=600, type=int)
    args = ap.parse_args()

    try:
        root_dir = ""
        exp_name_list = args.n.split(",")
    except Exception:
        root_dir = f"{args.f}\\"
        exp_name_list = [sub_dir for sub_dir in os.listdir(f"{cur_dir}\\{args.f}")]

    _base_config = BendersConfig.from_json(f"{cur_dir}\\benders_config.json")

    for exp_name in exp_name_list:
        print(exp_name)
        rel_path = f"{root_dir}{exp_name}"
        if not os.path.exists(f"{cur_dir}\\{rel_path}"):
            os.makedirs(f"{cur_dir}\\{exp_name}")
        shutil.copy(__file__, f"{cur_dir}\\{rel_path}\\{cur_fname}.py.backup")

        if os.path.exists(f"{cur_dir}\\{rel_path}\\alg_config.json"):
            with open(f"{cur_dir}\\{rel_path}\\alg_config.json", "r") as _f:
                alg_config = BendersConfig.from_dict(json.load(_f))
        else:
            alg_config = _base_config
            with open(f"{cur_dir}\\{rel_path}\\alg_config.json", "w") as _f:
                json.dump(alg_config.to_dict(), _f)

        for inst in args.i.split(","):
            print(inst)
            work_dir = f"{cur_dir}\\{rel_path}\\benders\\{inst}"
            alg_config.logging_config.fpath = work_dir

            if not os.path.exists(work_dir):
                os.makedirs(work_dir)
            main_benders(inst, work_dir, exp_name, alg_config)
