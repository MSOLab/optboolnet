import json
import multiprocessing as mp
import os, sys

import psutil
import optboolnet
from optboolnet.mibs import MibSAttractorControl
from optboolnet.config import BendersConfig, Config, MibSBilevelConfig
from optboolnet.instances import iter_bn_in_repo, load_bn_in_repo, _INSTANCE_LIST_FULL
from optboolnet.algorithm import BendersAttractorControl

TIME_MARGIN = 5

def controller(func, stoptime, _args, **_kwargs):
    thr = mp.Process(target=func, args=_args, kwargs=_kwargs)
    thr.daemon = True
    thr.start()
    thr.join(stoptime+TIME_MARGIN)
    if thr.is_alive():
        thr.terminate()
        [x.kill() for x in psutil.process_iter() if "mibs" in x.name().lower()]


def save_bnet_settings(inst: str, work_dir: str):
    bn = load_bn_in_repo(inst)
    with open(f"{work_dir}\\transition_formula.bnet", "w") as _f:
        _f.writelines(bn.to_bnet())
    with open(f"{work_dir}\\transition_formula_neg.bnet", "w") as _f:
        _f.writelines(bn.to_neg_CNF().to_bnet())
    with open(f"{work_dir}\\control_settings.json", "w") as _f:
        json.dump(bn._control_config.to_dict(), _f, indent=4)


def main_benders(inst: str, work_dir: str, param_str: str, config: Config):
    bn = load_bn_in_repo(inst)
    attr_ctrl_manager = BendersAttractorControl(param_str, bn, config.logging_config)
    attr_ctrl_manager.allow_empty_attractor = config.allow_empty_attractor
    attr_ctrl_manager.solve_separation = config.solve_separation    
    attr_ctrl_manager.preprocess_max_forbidden_trap_space = config.preprocess_max_forbidden_trap_space
    attr_ctrl_manager.separation_heuristic = config.separation_heuristic
    attr_ctrl_manager.use_high_point_relaxation = config.use_high_point_relaxation
    attr_ctrl_manager.total_time_limit = config.total_time_limit
    attr_ctrl_manager.use_aggregated_LLP = config.use_aggregated_LLP
    attr_ctrl_manager.get_control_strategies(max_control_size=config.max_control_size, max_length=config.max_length,master_solver_config=config.master_solver_config, LLP_solver_config=config.LLP_solver_config, separation_solver_config=config.separation_solver_config)


def main_MibS(inst: str, work_dir: str, param_str: str, config: MibSBilevelConfig):
    bn = load_bn_in_repo(inst)
    attr_ctrl_manager = MibSAttractorControl(param_str, bn, config)
    attr_ctrl_manager.total_time_limit = config.total_time_limit
    attr_ctrl_manager.get_control_strategies()


if __name__ == "__main__":
    from argparse import ArgumentParser
    import shutil

    cur_dir = os.path.dirname(__file__)
    cur_fname = os.path.basename(__file__).split("/")[-1][:-3]
    package_dir = os.path.dirname(optboolnet.__file__)

    ap = ArgumentParser()
    ap.add_argument("algorithm")
    ap.add_argument("-n", help="experiment_name", required=False)
    ap.add_argument("-f", help="experiment_folder", required=False)
    ap.add_argument("-g", help="instance_group", default="")
    ap.add_argument("-i", help="instances", default="")
    ap.add_argument("--time_limit", help="time limit in seconds", default=600, type=int)
    args = ap.parse_args()
    try:
        root_dir = ""
        exp_name_list = args.n.split(",")
    except:
        root_dir = f"{args.f}\\"
        exp_name_list = [sub_dir for sub_dir in os.listdir(f"{cur_dir}\\{args.f}")]
    alg = args.algorithm
    if alg == "benders":
        _base_config = BendersConfig.from_json(f"{cur_dir}\\benders_config.json")
        _Config = BendersConfig
    elif alg == "MibS":
        _base_config = MibSBilevelConfig.from_json(f"{cur_dir}\\MibS_config.json")
        _Config = MibSBilevelConfig
    else:
        raise Exception()
        
    for exp_name in exp_name_list:
        print(exp_name)
        rel_path = f"{root_dir}{exp_name}"
        if not os.path.exists(f"{cur_dir}\\{rel_path}"):
            os.makedirs(f"{cur_dir}\\{exp_name}")
        shutil.copy(__file__, f"{cur_dir}\\{rel_path}\\{cur_fname}.py.backup")
        if os.path.exists(f"{cur_dir}\\{rel_path}\\alg_config.json"):
            with open(f"{cur_dir}\\{rel_path}\\alg_config.json", "r") as _f:
                alg_config = _Config.from_dict(json.load(_f))
        else:
            alg_config = _base_config
            with open(f"{cur_dir}\\{rel_path}\\alg_config.json", "w") as _f:
                json.dump(alg_config.to_dict(), _f)

        for inst in args.i.split(","):
            bn = load_bn_in_repo(inst)
            print(inst)
            work_dir = f"{cur_dir}\\{rel_path}\\{alg}\\{inst}"
            alg_config.logging_config.fpath = work_dir

            if not os.path.exists(work_dir):
                os.makedirs(work_dir)
            # save_bnet_settings(inst, work_dir)
            if alg == "benders":
                # controller(main_benders, args.time_limit, (inst, work_dir, exp_name, alg_config))
                main_benders(inst, work_dir, exp_name, alg_config)
            elif alg == "MibS":
                # controller(main_MibS, alg_config.total_time_limit, (inst, work_dir, exp_name, alg_config))
                main_MibS(inst, work_dir, exp_name, alg_config)
