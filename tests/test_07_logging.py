import pytest
from optboolnet.instances import load_bn_in_repo
from optboolnet.config import LoggingConfig
from optboolnet.algorithm import BendersFixPointControl
from itertools import product
import os, sys

stdout_copy = 0
os.dup2(sys.stdout.fileno(), stdout_copy)

# Restore stdout
sys.stdout = os.fdopen(stdout_copy, "w")


def setup_function(function):
    pass


def teardown_function(function):
    import logging

    logging.shutdown()
    for suffix in ["build", "cut", "solve"]:
        if os.path.exists(f"tests/log__{suffix}.txt"):
            os.remove(f"tests/log__{suffix}.txt")


def test_logging_1():
    _logging_config = LoggingConfig(**{"to_stream": False, "fpath": "", "fname": ""})
    _benders_config_dict = {
        "max_control_size": 0,
        "max_length": 1,
        "allow_empty_attractor": False,
        "solve_separation": False,
        "preprocess_max_forbidden_trap_space": False,
        "separation_heuristic": False,
        "use_high_point_relaxation": True,
        "total_time_limit": None,
    }
    inst = "S1"
    bn = load_bn_in_repo(inst)
    for _logging_config.fpath, _logging_config.to_stream in product(
        [os.path.dirname(__file__), ""], [True, False]
    ):
        alg = BendersFixPointControl(inst, bn, _logging_config)
        s = alg.get_control_strategies(**_benders_config_dict)
    with open(f"{os.path.dirname(__file__)}/log__cut.txt", "r") as _f:
        assert len(_f.readlines()) > 2


if __name__ == "__main__":
    test_logging_1()
