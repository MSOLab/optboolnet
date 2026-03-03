import json
import os

import optboolnet
from optboolnet.mibs import MibSAttractorControl
from optboolnet.config import MibSBilevelConfig
from optboolnet.instances import load_bn_in_repo


def main_MibS(inst: str, work_dir: str, param_str: str, config: MibSBilevelConfig):
    model_name = "INTERDICTION" if config.use_interdiction else "BILEVEL"
    bn = load_bn_in_repo(inst)
    attr_ctrl_manager = MibSAttractorControl(model_name, bn, config)
    attr_ctrl_manager.total_time_limit = config.total_time_limit
    attr_ctrl_manager.get_control_strategies()


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
    ap.add_argument(
        "--interdiction",
        help="use the explicit interdiction formulation (MibSInterdictBilevelIP); "
             "automatically passes MibS_bilevelProblemType=1, "
             "MibS_objBoundStrategy=1, MibS_useBendersInterdictionCut=1 to MibS",
        action="store_true",
        default=False,
    )
    ap.add_argument(
        "--tee",
        help="print MibS solver progress to the console",
        action="store_true",
        default=False,
    )
    ap.add_argument(
        "--save-model",
        metavar="FILENAME",
        help="write the Pyomo model as a human-readable LP file with symbolic variable "
             "names (e.g. LLP.p, LLP.delta[gene,0]) before the PAO/MibS conversion; "
             "the file is written per-instance inside the work_dir with the given name "
             "(use a .lp extension, e.g. debug.lp)",
        default=None,
    )
    ap.add_argument(
        "--keep-temp",
        help="keep mibs_temp.mps and mibs_temp.aux after each solve call "
             "(useful for inspecting the raw bilevel MPS passed to MibS)",
        action="store_true",
        default=False,
    )
    args = ap.parse_args()

    try:
        root_dir = ""
        exp_name_list = args.n.split(",")
    except Exception:
        root_dir = f"{args.f}\\"
        exp_name_list = [sub_dir for sub_dir in os.listdir(f"{cur_dir}\\{args.f}")]

    _base_config = MibSBilevelConfig.from_json(f"{cur_dir}\\MibS_config.json")

    for exp_name in exp_name_list:
        print(exp_name)
        rel_path = f"{root_dir}{exp_name}"
        if not os.path.exists(f"{cur_dir}\\{rel_path}"):
            os.makedirs(f"{cur_dir}\\{exp_name}")
        shutil.copy(__file__, f"{cur_dir}\\{rel_path}\\{cur_fname}.py.backup")

        if os.path.exists(f"{cur_dir}\\{rel_path}\\alg_config.json"):
            with open(f"{cur_dir}\\{rel_path}\\alg_config.json", "r") as _f:
                alg_config = MibSBilevelConfig.from_dict(json.load(_f))
        else:
            alg_config = _base_config
            with open(f"{cur_dir}\\{rel_path}\\alg_config.json", "w") as _f:
                json.dump(alg_config.to_dict(), _f)

        # --interdiction overrides the config field regardless of what the JSON says
        if args.interdiction:
            alg_config.use_interdiction = True
            alg_config.use_aggregated_LLP = False

        # --tee overrides solver_config.tee so MibS output appears on stdout
        if args.tee:
            alg_config.solver_config.tee = True

        # --keep-temp: stop _run_mibs from deleting mibs_temp.mps / mibs_temp.aux
        if args.keep_temp:
            alg_config.solver_config.keep_temp = True

        for inst in args.i.split(","):
            print(inst)
            work_dir = f"{cur_dir}\\{rel_path}\\MibS\\{inst}"
            alg_config.logging_config.fpath = work_dir

            # --save-model: write Pyomo LP with symbolic names into work_dir
            if args.save_model:
                alg_config.solver_config.save_lp_path = f"{work_dir}\\{args.save_model}"

            if not os.path.exists(work_dir):
                os.makedirs(work_dir)
            main_MibS(inst, work_dir, exp_name, alg_config)
