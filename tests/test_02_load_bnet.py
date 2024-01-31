import pytest
import os
from optboolnet.instances import load_bn, load_bn_in_repo, iter_bn_in_repo
from optboolnet.boolnet import Hypercube

_FPATH = os.path.dirname(__file__)


def setup_function(function):
    pass


def teardown_function(function):
    pass


def test_load_bn():
    load_bn_in_repo("S1")
    load_bn_in_repo("M1")
    load_bn_in_repo("L1")
    load_bn(f"{_FPATH}/test_instance")
    try:
        load_bn_in_repo("RANDOM_FAKE_INSTANCE")
    except FileNotFoundError as e:
        pass

    # iterate through the repo
    for inst, bn in iter_bn_in_repo():
        print(bn)
        vars_list = sorted(list(bn.keys()))
        second_vars_list = sorted(
            bn.controllable_vars.copy() + bn.uncontrollable_vars.copy()
        )
        assert vars_list == second_vars_list


def test_cnf_parsing():
    # BN with constant transition formulas
    bn = load_bn(f"{_FPATH}/test_instance")

    for var_name, clause_count in zip(bn.keys(), [1, 0, 2]):
        print([clause.to_dict() for clause in bn.items_clause(var_name)])
        assert (
            len([clause.to_dict() for clause in bn.items_clause(var_name)])
            == clause_count
        )


def test_hypercube():
    bn = load_bn(f"{_FPATH}/test_instance")
    hc = Hypercube()
    hc["x2"] = 1

    print("Fixed: ", set(hc.keys()))
    print("Unfixed: ", hc.unfixed_vars(bn.vars_list))
    print("Unfixed, controllable: ", hc.unfixed_vars(bn.controllable_vars))
    print("Unfixed, uncontrollable: ", hc.unfixed_vars(bn.uncontrollable_vars))

    assert set(hc.keys()) == set(["x2"])
    assert set(hc.unfixed_vars(bn.vars_list)) == set(["x1", "x3"])
    assert set(hc.unfixed_vars(bn.controllable_vars)) == set(["x1"])
    assert set(hc.unfixed_vars(bn.uncontrollable_vars)) == set(["x3"])


if __name__ == "__main__":
    test_load_bn()
    test_cnf_parsing()
    test_hypercube()
