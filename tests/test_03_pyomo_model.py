import pytest
import os
from pyomo.kernel import NonNegativeReals
import pyomo as pmo
import pyomo.environ as pmoenv
from pyomo.environ import *

from pyomo.opt.base import check_available_solvers

_solver_list = [
    "gurobi",
    "glpk",
    "cbc",
    "cplex",
    "scip",
    "clp",
]

_persistent_solver_list = ["gurobi_persistent"]


class BasicLP:
    def __init__(self) -> None:
        # https://pyomo.readthedocs.io/en/stable/pyomo_overview/simple_examples.html
        self.model = pmoenv.ConcreteModel()
        self.model.x = pmoenv.Var([1, 2], domain=NonNegativeReals)
        self.model.OBJ = pmoenv.Objective(
            expr=2 * self.model.x[1] + 3 * self.model.x[2]
        )
        self.model.Constraint1 = pmoenv.Constraint(
            expr=3 * self.model.x[1] + 4 * self.model.x[2] >= 1
        )


class BasicLP2(pmoenv.ConcreteModel):
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        # https://pyomo.readthedocs.io/en/stable/pyomo_overview/simple_examples.html
        self.x = pmoenv.Var([1, 2], domain=NonNegativeReals)
        self.OBJ = pmoenv.Objective(expr=2 * self.x[1] + 3 * self.x[2])
        self.Constraint1 = pmoenv.Constraint(expr=3 * self.x[1] + 4 * self.x[2] >= 1)


def setup_function(function):
    pass


def teardown_function(function):
    pass


def test_build_model():
    knapsack = BasicLP()
    for solver_name in _solver_list:
        if check_available_solvers(solver_name):
            print(solver_name)
            solver = pmoenv.SolverFactory(solver_name)
            solver.solve(knapsack.model)


def test_persistent_model():
    knapsack = BasicLP()
    for solver_name in _persistent_solver_list:
        if check_available_solvers(solver_name):
            print(solver_name)
            solver = pmoenv.SolverFactory(solver_name)
            solver.set_instance(knapsack.model)
            solver.solve()
            knapsack.model.c2 = pmoenv.Constraint(
                expr=knapsack.model.x[1] + knapsack.model.x[2] <= 2
            )
            solver.add_constraint(knapsack.model.c2)
            solver.solve()


def test_inherited_model():
    knapsack = BasicLP2()
    for solver_name in _persistent_solver_list:
        if check_available_solvers(solver_name):
            print(solver_name)
            solver = pmoenv.SolverFactory(solver_name)
            solver.set_instance(knapsack)
            solver.solve()
            knapsack.c2 = pmoenv.Constraint(expr=knapsack.x[1] + knapsack.x[2] <= 2)
            solver.add_constraint(knapsack.c2)
            solver.solve()


if __name__ == "__main__":
    test_build_model()
    test_persistent_model()
    test_inherited_model()
