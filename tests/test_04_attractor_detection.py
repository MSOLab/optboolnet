import pytest
from optboolnet.instances import load_bn_in_repo
from optboolnet.model import AttractorDetectionIP
from optboolnet.config import SolverConfig
import os, sys

stdout_copy = 0
os.dup2(sys.stdout.fileno(), stdout_copy)

# Restore stdout
sys.stdout = os.fdopen(stdout_copy, "w")

_solver_config = {
    "solver_name": "gurobi_persistent",
    "save_results": False,
    "tee": False,
    "warmstart": False,
    "time_limit": 3.0,
}


def setup_function(function):
    pass


def teardown_function(function):
    pass


def test_attractor_dection():
    for inst in ["S1", "S2", "M3", "L2"]:
        bn = load_bn_in_repo(inst)
        print("Check attractors with both phenotypes exist", inst)
        attr_ip = AttractorDetectionIP(
            "test_attractor_enumeration", bn, 5, SolverConfig(**_solver_config)
        )
        attr_ip.make_constr_stability_condition()
        attr_ip.make_constr_phenotype_at_all_t()
        attr_ip.set_constr_target_size(0)
        attr_ip.set_phenotype_obj(_minimize=True)
        attr_ip.optimize()
        assert attr_ip.p.value == 0

        attr_ip.set_phenotype_obj(_minimize=False)
        attr_ip.optimize()
        assert attr_ip.p.value == 1


def find_all_attractors():
    inst = "M1"
    bn = load_bn_in_repo(inst)

    attractor_list = list()
    print("Enumerate all attractors of the length up to 6:", inst),
    for length, answer in zip(range(1, 7), [27, 19, 6, 6, 0, 27]):
        attr_ip = AttractorDetectionIP(bn, length, SolverConfig(_solver_config))
        attr_ip.make_constr_stability_condition()
        for attractor in attractor_list:
            attr_ip.add_no_good_x(attractor)
        attr_ip.set_constr_target_size(0)
        count = 0
        while attr_ip.optimize():
            attractor = attr_ip.get_attractor()
            attractor_list += [attractor]
            attr_ip.add_no_good_x(attractor)
            count += 1
        assert count == answer


if __name__ == "__main__":
    test_attractor_dection()
    find_all_attractors()
