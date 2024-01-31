import pytest
from optboolnet.config import SolverConfig
from optboolnet.instances import load_bn_in_repo, iter_bn_in_repo
from optboolnet.model import TrapSpaceDetectionIP
import os, sys

_solver_name = "gurobi_persistent"

stdout_copy = 0
os.dup2(sys.stdout.fileno(), stdout_copy)

# Restore stdout
sys.stdout = os.fdopen(stdout_copy, "w")

_config = SolverConfig(
    **{
        "solver_name": "gurobi_persistent",
        "save_results": False,
        "tee": False,
        "warmstart": False,
        "threads": 1,
        "time_limit": 3.0,
    }
)


def setup_function(function):
    pass


def teardown_function(function):
    pass


def test_forbidden_trap_space_enumeration():
    for (inst, bn), answer in zip(iter_bn_in_repo(["small"]), [8, 5, 23, 3]):
        bn = load_bn_in_repo(inst)
        print(inst)
        forbidden_ts_ip = TrapSpaceDetectionIP("Forbidden_TS_test", bn, _config)
        forbidden_ts_ip.set_constr_target_size(0)
        forbidden_ts_ip.fix_phenotype(0)
        forbidden_ts_ip.set_objective_sparse_cut()
        while forbidden_ts_ip.optimize():
            ctrl = forbidden_ts_ip.get_control()
            ts = forbidden_ts_ip.get_trap_space()
            forbidden_ts_ip.add_trap_space_maximality_cut(ctrl, ts)
        assert len(forbidden_ts_ip.constrs_benders) == answer


if __name__ == "__main__":
    test_forbidden_trap_space_enumeration()
