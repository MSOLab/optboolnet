import json
import os
import sys

_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
# Avoid shadowing the third-party `pyboolnet` package with `_experiments/pyboolnet.py`.
sys.path = [p for p in sys.path if os.path.abspath(p) != _THIS_DIR]

import optboolnet
from optboolnet.config import PyBoolNetConfig
from optboolnet.instances import load_bn_in_repo
from optboolnet.pyboolnet import PyBoolNetAttractorControl


def main_pbn(inst: str, work_dir: str, param_str: str, config: PyBoolNetConfig):
    bn = load_bn_in_repo(inst)
    attr_ctrl_manager = PyBoolNetAttractorControl(param_str, bn, config.logging_config)
    attr_ctrl_manager.total_time_limit = config.total_time_limit
    attr_ctrl_manager.update = config.update
    attr_ctrl_manager.max_output_trapspaces = config.max_output_trapspaces
    attr_ctrl_manager.get_control_strategies(
        max_control_size=config.max_control_size,
        starting_length=config.starting_length,
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
    ap.add_argument("--time_limit", help="time limit in seconds", default=None, type=int)
    ap.add_argument(
        "--update",
        choices=["synchronous", "asynchronous", "mixed"],
        default=None,
        help="PyBoolNet update mode override",
    )
    args = ap.parse_args()

    try:
        root_dir = ""
        exp_name_list = args.n.split(",")
    except Exception:
        root_dir = f"{args.f}\\"
        exp_name_list = [sub_dir for sub_dir in os.listdir(f"{cur_dir}\\{args.f}")]

    _base_config_path = f"{cur_dir}\\pbn_config.json"
    if os.path.exists(_base_config_path):
        _base_config = PyBoolNetConfig.from_json(_base_config_path)
    else:
        _base_config = PyBoolNetConfig()

    for exp_name in exp_name_list:
        print(exp_name)
        rel_path = f"{root_dir}{exp_name}"
        if not os.path.exists(f"{cur_dir}\\{rel_path}"):
            os.makedirs(f"{cur_dir}\\{exp_name}")
        shutil.copy(__file__, f"{cur_dir}\\{rel_path}\\{cur_fname}.py.backup")

        if os.path.exists(f"{cur_dir}\\{rel_path}\\alg_config.json"):
            with open(f"{cur_dir}\\{rel_path}\\alg_config.json", "r") as _f:
                alg_config = PyBoolNetConfig.from_dict(json.load(_f))
        else:
            alg_config = _base_config
            with open(f"{cur_dir}\\{rel_path}\\alg_config.json", "w") as _f:
                json.dump(alg_config.to_dict(), _f)

        if args.time_limit is not None:
            alg_config.total_time_limit = args.time_limit
        if args.update is not None:
            alg_config.update = args.update

        for inst in args.i.split(","):
            print(inst)
            work_dir = f"{cur_dir}\\{rel_path}\\pbn\\{inst}"
            alg_config.logging_config.fpath = work_dir

            if not os.path.exists(work_dir):
                os.makedirs(work_dir)
            main_pbn(inst, work_dir, exp_name, alg_config)
