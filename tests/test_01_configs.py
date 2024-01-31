import pytest
import json, sys, os
import optboolnet
from optboolnet.config import ControlConfig, BendersConfig, SolverConfig
from optboolnet.model import CoreIP

_FPATH = os.path.dirname(__file__)

stdout_copy = 0
os.dup2(sys.stdout.fileno(), stdout_copy)

# Restore stdout
sys.stdout = os.fdopen(stdout_copy, "w")


def teardown_function(function):
    if os.path.exists(f"{os.path.dirname(__file__)}/save_benders_config.json"):
        os.remove(f"{os.path.dirname(__file__)}/save_benders_config.json")


def test_control_config():
    # by a json file name
    # bn = optboolnet.CNFBooleanNetwork(
    #     data=_FPATH + "/test_instance/transition_formula.bnet",
    #     control_config=_FPATH + "/test_instance/control_setting.json",
    # )
    # # by a dictionary
    # with open(_FPATH + "/test_instance/control_setting.json") as _f:
    #     bn = optboolnet.CNFBooleanNetwork(
    #         data=_FPATH + "/test_instance/transition_formula.bnet",
    #         control_config=json.load(_f),
    #     )
    # by explicit construction
    with open(_FPATH + "/test_instance/control_setting.json") as _f:
        bn = optboolnet.CNFBooleanNetwork(
            data=_FPATH + "/test_instance/transition_formula.bnet",
            control_config=ControlConfig.from_dict(json.load(_f)),
        )


def test_solver_config():
    # by a json file name
    with open(_FPATH + "/test_instance/control_setting.json") as _f:
        bn = optboolnet.CNFBooleanNetwork(
            data=_FPATH + "/test_instance/transition_formula.bnet",
            control_config=ControlConfig.from_dict(json.load(_f)),
        )
    _config = SolverConfig(
        **{
            "solver_name": "gurobi_persistent",
            "save_results": False,
            "tee": False,
            "warmstart": False,
        }
    )
    _config.tee = True
    CoreIP("test_json", bn, _config).optimize()

    # by a dictionary
    with open(_FPATH + "/test_instance/benders_config.json") as _f:
        _config = SolverConfig.from_dict(json.load(_f)["master_solver_config"])
    _config.tee = True
    CoreIP("test_dict", bn, _config).optimize()


def test_benders_config_and_save():
    _benders_config = BendersConfig.from_json(
        os.path.dirname(__file__) + "/test_instance/benders_config.json"
    )
    with open(os.path.dirname(__file__) + "/save_benders_config.json", "w") as _f:
        json.dump(_benders_config.to_dict(), _f, indent=4)
    recovered_config = BendersConfig.from_json(
        os.path.dirname(__file__) + "/save_benders_config.json"
    )


if __name__ == "__main__":
    # test_control_config()
    test_solver_config()
    test_benders_config_and_save()
