import pytest
from optboolnet.exception import EmptyAttractorError
from optboolnet.instances import load_bn_in_repo
from optboolnet.config import SolverConfig, LoggingConfig
from optboolnet.exception import InvalidConfigError
from optboolnet.algorithm import BendersAttractorControl, BendersFixPointControl
from itertools import product
import os, sys

from optboolnet.model import AttractorDetectionIP

stdout_copy = 0
os.dup2(sys.stdout.fileno(), stdout_copy)

# Restore stdout
sys.stdout = os.fdopen(stdout_copy, "w")

_FPATH = os.path.dirname(__file__)


_solver_config = SolverConfig(
    **{
        "solver_name": "gurobi_persistent",
        "save_results": False,
        "tee": False,
        "warmstart": False,
        # "mip_display": 0,
        "threads": 1,
        "time_limit": 3.0,
    }
)
_logging_config = LoggingConfig(**{"to_stream": False, "fpath": "", "fname": ""})


def setup_function(function):
    pass


def teardown_function(function):
    pass


def test_fixed_point_inconsistency():
    _benders_config_dict = {
        "enforce": False,
        "max_control_size": 0,
        "max_length": 1,
        "allow_empty_attractor": False,
        "master_solver_config": _solver_config,
        "LLP_solver_config": _solver_config,
        "separation_solver_config": _solver_config,
        "logging_config": _logging_config,
        "solve_separation": False,  # inconsistent value
        "preprocess_max_forbidden_trap_space": True,
        "separation_heuristic": True,
        "use_high_point_relaxation": True,
        "total_time_limit": None,
    }
    bn = load_bn_in_repo("S1")
    detected = False
    try:
        # _incorrect_benders_config = BendersConfig.from_dict(_benders_config_dict)
        alg = BendersFixPointControl("test", bn)

        alg.get_control_strategies(**_benders_config_dict)
    except InvalidConfigError as e:
        detected = True
        print("Succesfully handled the exception:\n", e)
    assert detected


def test_fixed_point_control():
    _benders_config_dict = {
        "enforce": True,
        "max_control_size": 2,
        "max_length": 1,
        "allow_empty_attractor": False,
        "solve_separation": False,
        "preprocess_max_forbidden_trap_space": False,
        "separation_heuristic": False,
        "use_high_point_relaxation": True,
        "total_time_limit": None,
    }

    for inst, answer in zip(["S2", "S4"], [2, 9]):
        bn = load_bn_in_repo(inst)
        print(inst)
        for [
            _benders_config_dict["use_high_point_relaxation"],
        ] in product([True, False]):
            print(
                inst,
                _benders_config_dict["use_high_point_relaxation"],
            )
            alg = BendersFixPointControl(inst, bn)

            try:
                s = alg.get_control_strategies(**_benders_config_dict)
            except EmptyAttractorError:
                continue
            assert _benders_config_dict["use_high_point_relaxation"] == isinstance(
                alg.model_master, AttractorDetectionIP
            )

            assert alg.solution_count == answer


def test_attractor_control():
    _benders_config_dict = {
        "enforce": False,
        "max_control_size": 2,
        "max_length": 4,
        "allow_empty_attractor": False,
        "solve_separation": False,
        "preprocess_max_forbidden_trap_space": False,
        "separation_heuristic": False,
        "use_high_point_relaxation": False,
        "total_time_limit": None,
    }
    for inst, answer in zip(["S2", "S4"], [9, 9]):
        bn = load_bn_in_repo(inst)
        print(inst)
        detected = 0
        for (
            _benders_config_dict["solve_separation"],
            _benders_config_dict["separation_heuristic"],
        ) in product([True, False], [True, False]):
            try:
                alg = BendersAttractorControl(inst, bn)
                s = alg.get_control_strategies(**_benders_config_dict)
                assert alg.solution_count == answer
            except InvalidConfigError:
                detected += 1
        assert detected == 1  # (F,T,T), (F,T,F), (F,F,T)


if __name__ == "__main__":
    test_fixed_point_inconsistency()
    test_fixed_point_control()
    test_attractor_control()
