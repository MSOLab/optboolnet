from optboolnet.instances import load_bn_in_repo
from optboolnet.model import AttractorDetectionIP
from optboolnet.config import SolverConfig
from optboolnet import checking

def test_nusmv_check_attractor():
    bn = load_bn_in_repo("S1")
    _solver_config = {
        "solver_name": "gurobi_persistent",
        "save_results": False,
        "tee": False,
        "warmstart": False,
        "time_limit": 3.0,
    }
    length = 6
    attr_ip = AttractorDetectionIP("M1", bn, length, SolverConfig(**_solver_config))
    attr_ip.make_constr_stability_condition()
    attr_ip.set_constr_target_size(0)
    attr_ip.optimize()
    attractor = attr_ip.get_attractor()
    assert checking.nusmv_check_attractor(bn, attractor)

def test_nusmv_check_phenotype():
    bn = load_bn_in_repo("S1")
    assert not checking.nusmv_check_phenotype(bn)
    assert not checking.nusmv_check_phenotype(bn, control={"EGF": 0})
    assert checking.nusmv_check_phenotype(bn, control={"EGF": 0, "ER_a": 0})

if __name__ == "__main__":
    test_nusmv_check_attractor()
    test_nusmv_check_phenotype()
