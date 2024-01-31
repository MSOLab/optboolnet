import sys
from typing import Dict, Generator, Iterator, List, Optional, TypeVar
from optboolnet import CNFBooleanNetwork, Attractor, Control, Hypercube
from optboolnet.config import SolverConfig
from optboolnet.log import EnumCutType
from pyomo.solvers.plugins.solvers.direct_or_persistent_solver import (
    DirectOrPersistentSolver,
)
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.opt import TerminationCondition, SolverResults
import pyomo.environ as pmoenv

# TODO: lazy cut implementation for persistent solvers


class LiteralCounter(Iterator):
    def __init__(self, gen: Generator):
        self.gen = iter(gen)
        self.count = 0

    def __iter__(self):
        return self

    def next(self):
        nxt = next(self.gen)
        self.count += 1
        return nxt

    __next__ = next


class CoreIP(pmoenv.ConcreteModel):

    """The basic extension for handling a Boolean network
    and iterative problem solving with both direct and persitent solver"""

    def __init__(
        self,
        name: str,
        bn: CNFBooleanNetwork,
        solver_config: SolverConfig,
        *args,
        **kwds
    ):
        super().__init__(*args, **kwds)
        self.name = name
        self.bn = bn
        self.solver_config = solver_config

        ### ======== index sets
        def C_init(model: AttractorDetectionIP):
            return ((i, c) for i in model.I for c in model.C_i[i])

        def pos_lit_init(model: AttractorDetectionIP):
            return (
                (i, c, i_)
                for (i, c), clause in model.bn.iter_clauses()
                for i_ in clause.pos_literals
            )

        def neg_lit_init(model: AttractorDetectionIP):
            return (
                (i, c, i_)
                for (i, c), clause in model.bn.iter_clauses()
                for i_ in clause.neg_literals
            )

        self.B = pmoenv.Set(initialize=[0, 1])
        """The Boolean domain"""
        self.I = pmoenv.Set(initialize=list(bn.keys()))
        """The set of variables"""
        self.J = pmoenv.Set(initialize=bn.controllable_vars)
        """The set of controllable variables"""
        self.J_c = pmoenv.Set(initialize=bn.uncontrollable_vars)
        """The set of uncontrollable variables"""
        self.C_i = pmoenv.Set(self.I, initialize=self.bn.get_clause_idx_dict())
        """The set of clauses for each variable"""
        self.C = pmoenv.Set(dimen=2, initialize=C_init)
        """The set of all clauses"""
        self.pos_lit = pmoenv.Set(dimen=3, initialize=pos_lit_init)
        """"""
        self.neg_lit = pmoenv.Set(dimen=3, initialize=neg_lit_init)
        """"""
        ### ======== solver
        self.solver: DirectOrPersistentSolver = pmoenv.SolverFactory(
            self.solver_config.solver_name
        )
        self.solver.options = solver_config.options
        # self.solver._max_constraint_degree = 1  # for efficienct parsing
        if isinstance(self.solver, PersistentSolver):
            self.solver.set_instance(self)

        ### ======== objective
        self.obj = pmoenv.Objective(expr=1)
        """The objective function, initialized as 1"""
        self.set_objective(1)

    def update_options_time_limit(self, time_limit: Optional[float]):
        self.solver.options["time_limit"] = time_limit

    def fix_var(self, var: pmoenv.ScalarVar, value: int):
        # var.fix(value)
        var.setlb(value)
        var.setub(value)
        if isinstance(self.solver, PersistentSolver):
            self.solver.update_var(var)

    def relax_var(self, var: pmoenv.ScalarVar):
        var.setlb(0)
        var.setub(1)
        if isinstance(self.solver, PersistentSolver):
            self.solver.update_var(var)

    def set_objective(self, expr: pmoenv.Expression, _minimize: bool = True):
        _sense = pmoenv.minimize if _minimize else pmoenv.maximize
        del self.obj
        self.obj = pmoenv.Objective(expr=expr, sense=_sense)
        if isinstance(self.solver, PersistentSolver):
            self.solver.set_objective(self.obj)

    def append_vars_to_solvers(self, pmo_vars_list: List[pmoenv.Var]):
        if not isinstance(self.solver, PersistentSolver):
            return
        for var in pmo_vars_list:
            if var.is_indexed():
                for single_var in var.values():
                    self.solver._add_var(single_var)
            else:
                self.solver._add_var(var)

    def add_constr_to_list(
        self, expr: pmoenv.Expression, target_list: pmoenv.ConstraintList
    ):
        """Appends a constraint to the target list

        Args:
            expr (pmoenv.Expression): the expression used to generate a constraint
            target_list (pmoenv.ConstraintList): the list to append the new constraint
        """
        new_constr = target_list.add(expr)
        if isinstance(self.solver, PersistentSolver):
            self.solver.add_constraint(new_constr)

    def clear_constr_list(self, target_list: pmoenv.ConstraintList):
        for _, constr in target_list.items():
            if isinstance(self.solver, PersistentSolver):
                self.solver.remove_constraint(constr)
        target_list.clear()

    def optimize(self, to_optimum: bool = True) -> bool:
        """Find an attractor by optimization

        Args:
            to_optimum (bool, optional):
            Whether to check the solution is optimal or feasible.
            Defaults to True.

        Returns:
            bool: the indicator for the termination condition
        """
        if isinstance(self.solver, PersistentSolver):
            results: SolverResults = self.solver.solve(**self.solver_config.kwgs)
        else:
            results = self.solver.solve(self, **self.solver_config.kwgs)

        if to_optimum:  # check the optimality
            return results.solver.termination_condition == TerminationCondition.optimal
        else:  # check the feasibility
            return results.solver.termination_condition in [
                TerminationCondition.feasible,
                TerminationCondition.optimal,
            ]


class MasterControlIP(CoreIP):
    """The extension of IP model for handling control variables d"""

    def __init__(
        self,
        name: str,
        bn: CNFBooleanNetwork,
        solver_setting: SolverConfig,
        *args,
        **kwds
    ):
        super().__init__(name, bn, solver_setting, *args, **kwds)

        ### ======== variables
        self.d = pmoenv.Var(self.J * self.B, domain=pmoenv.Binary)
        """d[j,k]=1 iff variable j is controlled to be k for all j in J, k in [0,1]"""
        self.append_vars_to_solvers([self.d])

        ### ======== constraints

        self.constrs_target_size = pmoenv.ConstraintList()
        """"""
        self.constrs_exclusivity = pmoenv.ConstraintList()
        """"""
        self.constrs_minimality = pmoenv.ConstraintList()
        """"""
        self.constrs_benders = pmoenv.ConstraintList()
        """"""

        self.make_constr_exclusivity()

    def set_constr_target_size(self, control_size: int):
        self.clear_constr_list(self.constrs_target_size)
        if control_size == None:
            return
        else:
            self.add_constr_to_list(
                pmoenv.summation(self.d) == control_size,
                self.constrs_target_size,
            )

    def get_control(self) -> Control:
        ctrl_dict = dict()
        for j in self.J:
            if pmoenv.value(self.d[j, 0]) == 1:
                ctrl_dict[j] = 0
            elif pmoenv.value(self.d[j, 1]) == 1:
                ctrl_dict[j] = 1
            # else: ctrl[j] = Hypercube.FREE
        return Control(ctrl_dict)

    def fix_control(self, ctrl: Control):
        """Fix the control d as the given object

        Args:
            ctrl (Control):
        """
        for j, k in ctrl.items():
            self.fix_var(self.d[j, k], 1)
            self.fix_var(self.d[j, 1 - k], 0)  # may be dropped
        for j in ctrl.unfixed_vars(self.bn.controllable_vars):
            self.fix_var(self.d[j, 0], 0)
            self.fix_var(self.d[j, 1], 0)

    def make_constr_exclusivity(self):
        """A variable cannot be fixed both 0 and 1"""
        self.clear_constr_list(self.constrs_exclusivity)
        for j in self.J:
            self.add_constr_to_list(
                self.d[j, 0] + self.d[j, 1] <= 1, self.constrs_exclusivity
            )

    def append_no_good_cut_d(self, ctrl: Control):
        self.add_constr_to_list(
            pmoenv.quicksum(
                self.d[j, 0] + self.d[j, 1]
                for j in ctrl.unfixed_vars(self.bn.controllable_vars)
            )
            + pmoenv.quicksum(
                (1 - self.d[j, k] + self.d[j, 1 - k]) for j, k in ctrl.items()
            )
            >= 1,
            self.constrs_benders,
        )
        return (EnumCutType.NO_GOOD_MASTER, 2 * len(self.J))

    def append_minimality_cut(self, ctrl: Control):
        if len(ctrl) == 0:  # the resulting cut is 0 >= 1, infeasible
            self.constr_minimality_infeasible = pmoenv.Constraint.Infeasible()
        else:
            self.add_constr_to_list(
                pmoenv.quicksum((1 - self.d[j, k]) for j, k in ctrl.items()) >= 1,
                self.constrs_minimality,
            )
        return (EnumCutType.MINIMALITY, len(ctrl))

    def append_logical_benders_cut(self, attr: Attractor):
        def _gen_terms():
            for j, k, alpha_j, beta_j in zip(
                self.J, attr.get_first_state(), attr.alpha, attr.beta
            ):
                if beta_j == 1:
                    yield self.d[j, 1 - k]
                else:
                    yield 1 - self.d[j, k]
                if alpha_j == 0:
                    yield self.d[j, k]

        iter_terms = LiteralCounter(_gen_terms())
        self.add_constr_to_list(pmoenv.quicksum(iter_terms) >= 1, self.constrs_benders)
        return (EnumCutType.ATTRACTOR_CUT, iter_terms.count)

    def append_forbidden_trap_space_cut(
        self, forbidden_ctrl: Control, forbidden_trap_space: Hypercube
    ):
        def _gen_terms():
            for j, k in forbidden_ctrl.items():
                yield (1 - self.d[j, k])
            for j in forbidden_ctrl.unfixed_vars(self.bn.controllable_vars):
                try:
                    yield self.d[j, 1 - forbidden_trap_space[j]]
                except:
                    pass

        iter_terms = LiteralCounter(_gen_terms())
        self.add_constr_to_list(pmoenv.quicksum(iter_terms) >= 1, self.constrs_benders)
        return (EnumCutType.TRAP_SPACE_CUT, iter_terms.count)

    def set_objective_min_control(self):
        return super().set_objective(sum(self.d.values()), True)


class AttractorDetectionIP(MasterControlIP):
    """The Pyomo integer programming model for finding an attractor of a given length under a control."""

    def __init__(
        self,
        name: str,
        bn: CNFBooleanNetwork,
        length: int,
        solver_setting: SolverConfig,
        *args,
        **kwds
    ):
        """

        Args:
            bn (CNFBooleanNetwork): CNF Boolean network with control settings
            length (int): the length of the target attractor
        """
        super().__init__(name, bn, solver_setting, *args, **kwds)
        self.length = length

        ### ======== index sets

        self.T_range = pmoenv.Set(
            initialize=range(1, 1 + self.length),
        )
        """The list of all positions of an attractor"""

        ### ======== variables

        self.x = pmoenv.Var(self.I * self.T_range, domain=pmoenv.Binary)
        """x[i,t] denotes the value of variable i at position t for all i in I,  t in [T]"""
        self.y = pmoenv.Var(self.C * self.T_range, domain=pmoenv.Binary)
        """y[i,c,t] denotes the value of c-th clause of variable i at position t for all i in I, k in [0,1], t in [T]"""
        self.p = pmoenv.ScalarVar(domain=pmoenv.Binary)
        """p = 1 iff the desired property is satisfied"""
        self.append_vars_to_solvers([self.x, self.y, self.p])

        ### ======== constraints

        self.constrs_stability = pmoenv.ConstraintList()
        """"""
        self.constrs_phenotype = pmoenv.ConstraintList()
        """"""
        self.constrs_no_good_x = pmoenv.ConstraintList()
        """"""
        self._constrs_stability: List[pmoenv.Constraint] = list()

    def prev(self, t: int):
        if t == 1:
            return self.length
        else:
            return t - 1

    def make_constr_phenotype_at_all_t(self):
        """The phenotype indicates 1 iff the phenotype is satisfied at all states"""

        self.clear_constr_list(self.constrs_phenotype)
        for t in self.T_range:
            self.add_constr_to_list(
                expr=self.p <= self.x[self.bn.phenotype, t],
                target_list=self.constrs_phenotype,
            )
        self.add_constr_to_list(
            expr=self.p
            >= 1
            + sum(self.x[self.bn.phenotype, t] for t in self.T_range)
            - self.length,
            target_list=self.constrs_phenotype,
        )

    def make_constr_stability_condition(self):
        """A variable must be fixed if the control is active.
        Otherwise, transition formulas must be satisfied
        """
        self.clear_constr_list(self.constrs_stability)
        for j, t in self.J * self.T_range:
            self.add_constr_to_list(
                self.d[j, 1] <= self.x[j, t],
                self.constrs_stability,
            )
            self.add_constr_to_list(
                self.d[j, 0] <= 1 - self.x[j, t],
                self.constrs_stability,
            )

        for i in self.I:
            (d_0, d_1) = (self.d[i, 0], self.d[i, 1]) if i in self.J else (0, 0)
            for t in self.T_range:
                x_i_t = self.x[i, t]
                for c in self.C_i[i]:
                    self.add_constr_to_list(
                        x_i_t <= self.y[i, c, self.prev(t)] + (d_0 + d_1),
                        self.constrs_stability,
                    )
                self.add_constr_to_list(
                    x_i_t
                    >= (1 - len(self.C_i[i]))
                    + sum(self.y[i, c, self.prev(t)] for c in self.C_i[i])
                    - (d_0 + d_1),
                    self.constrs_stability,
                )

        for (i, c), clause in self.bn.iter_clauses():
            for t in self.T_range:
                x_lit_list = [self.x[i_, t] for i_ in clause.pos_literals] + [
                    1 - self.x[i_, t] for i_ in clause.neg_literals
                ]

                for x_lit in x_lit_list:
                    self.add_constr_to_list(
                        self.y[i, c, t] >= x_lit, self.constrs_stability
                    )
                self.add_constr_to_list(
                    self.y[i, c, t] <= sum(x_lit_list),
                    self.constrs_stability,
                )

    def set_phenotype_obj(self, _minimize: bool = True):
        self.set_objective(expr=self.p, _minimize=_minimize)

    def add_no_good_x(self, attractor: Attractor):
        """Adds a constraints that removes the current attractor"""
        for t, x_del in attractor.iter_states():
            self.add_constr_to_list(
                sum(
                    x_i_1 if value == 0 else (1 - x_i_1)
                    for x_i_1, value in zip(self.x[:, 1], x_del)
                )
                >= 1,
                self.constrs_no_good_x,
            )

    def get_attractor(self) -> Attractor:
        """Extract the states of the discovered attractors with no repetition

        Returns:
            Attractor: the compact representation of the attractor
        """
        unique_state_seq: List[List[int]] = list()
        for t in self.T_range:
            new_state = [int(self.x[i, t].value) for i in self.I]
            if all(new_state != _state for _state in unique_state_seq):
                unique_state_seq.append(new_state)
            else:
                break
        x_1 = [self.x[j, 1].value for j in self.J]
        alpha = [
            all(self.x[j, 1].value == self.x[j, t].value for t in self.T_range)
            for j in self.J
        ]
        beta = [
            all(
                (self.x[j, t].value == 1)
                == all(self.y[j, c, self.prev(t)].value == 1 for c in self.C_i[j])
                for t in self.T_range
            )
            for j in self.J
        ]

        return Attractor(self.bn, unique_state_seq, x_1, alpha, beta)

    def fix_phenotype(self, value: int):
        """fix the phenotype indicator p to be either 0 or 1

        Args:
            value (int): the value to fix
        """
        self.fix_var(self.p, value)


class ExtendedAttractorDetectionIP(AttractorDetectionIP):
    def __init__(
        self,
        name: str,
        bn: CNFBooleanNetwork,
        length: int,
        solver_setting: SolverConfig,
        *args,
        **kwds
    ):
        """v = 1 iff there's no attractor"""

        super().__init__(name, bn, length, solver_setting, *args, **kwds)
        self.v = pmoenv.ScalarVar(domain=pmoenv.Binary)
        self.append_vars_to_solvers([self.v])

    def make_constr_stability_condition(self):
        """A variable must be fixed if the control is active.
        Otherwise, transition formulas must be satisfied
        """
        self.clear_constr_list(self.constrs_stability)
        for j, t in self.J * self.T_range:
            self.add_constr_to_list(
                self.d[j, 1] <= self.x[j, t],
                self.constrs_stability,
            )
            self.add_constr_to_list(
                -self.v + self.d[j, 0] <= 1 - self.x[j, t],
                self.constrs_stability,
            )

        for i in self.I:
            (d_0, d_1) = (self.d[i, 0], self.d[i, 1]) if i in self.J else (0, 0)
            for t in self.T_range:
                x_i_t = self.x[i, t]
                for c in self.C_i[i]:
                    self.add_constr_to_list(
                        x_i_t <= self.y[i, c, self.prev(t)] + (d_0 + d_1),
                        self.constrs_stability,
                    )
                self.add_constr_to_list(
                    x_i_t
                    >= (1 - len(self.C_i[i]))
                    + sum(self.y[i, c, self.prev(t)] for c in self.C_i[i])
                    - (d_0 + d_1),
                    self.constrs_stability,
                )

        for (i, c), clause in self.bn.iter_clauses():
            for t in self.T_range:
                x_lit_list = [self.x[i_, t] for i_ in clause.pos_literals] + [
                    1 - self.x[i_, t] for i_ in clause.neg_literals
                ]

                for i_ in clause.pos_literals:
                    self.add_constr_to_list(
                        self.y[i, c, t] >= self.x[i_, t],
                        self.constrs_stability,
                    )
                for i_ in clause.neg_literals:
                    self.add_constr_to_list(
                        self.y[i, c, t] >= 1 - self.x[i_, t] - self.v,
                        self.constrs_stability,
                    )
                self.add_constr_to_list(
                    self.y[i, c, t] <= sum(x_lit_list),
                    self.constrs_stability,
                )

    def set_phenotype_obj(self, _minimize: bool = True):
        self.set_objective(expr=self.p + 2 * self.v, _minimize=_minimize)


class TrapSpaceDetectionIP(MasterControlIP):
    """The Pyomo integer programming model for finding an attractor of a given length under a control."""

    def __init__(
        self,
        name: str,
        bn: CNFBooleanNetwork,
        solver_setting: SolverConfig,
        *args,
        **kwds
    ):
        """

        Args:
            bn (CNFBooleanNetwork): CNF Boolean network with control settings
            length (int): the length of the target attractor
        """
        super().__init__(name, bn, solver_setting, *args, **kwds)
        self.neg_bn = bn.to_neg_CNF()
        ### ======== index sets
        self.neg_C_i = pmoenv.Set(self.I, initialize=self.neg_bn.get_clause_idx_dict())
        ### ======== variables

        self.h = pmoenv.Var(self.I * self.B, domain=pmoenv.Binary)
        """h[i,k] denotes the value of variable i is fixed to be k in the selected trap space for all i in I, k in [0,1]"""
        self.append_vars_to_solvers([self.h])

        ### ======== constraints

        self.constrs_stability = pmoenv.ConstraintList()
        """"""
        self.constrs_phenotype = pmoenv.ConstraintList()
        """"""
        self.constrs_no_good_x = pmoenv.ConstraintList()
        """"""
        self.constrs_separation = pmoenv.ConstraintList()
        """"""

        self.make_constr_stability_condition()

    def make_constr_stability_condition(self):
        self.clear_constr_list(self.constrs_stability)

        for i in self.I:
            self.add_constr_to_list(
                self.h[i, 0] + self.h[i, 1] <= 1,
                self.constrs_stability,
            )

        for j, k in self.J * self.B:
            self.add_constr_to_list(
                self.d[j, k] <= self.h[j, k],
                self.constrs_stability,
            )

        for i, k in self.I * self.B:
            d_i = (self.d[i, 0], self.d[i, 1]) if i in self.J else (0, 0)
            for k, clauses in enumerate(
                [self.neg_bn.items_clause(i), self.bn.items_clause(i)]
            ):
                for clause in clauses:
                    self.add_constr_to_list(
                        self.h[i, k] - d_i[k]
                        <= pmoenv.quicksum(self.h[i_, 1] for i_ in clause.pos_literals)
                        + pmoenv.quicksum(self.h[i_, 0] for i_ in clause.neg_literals),
                        self.constrs_stability,
                    )

    def fix_phenotype(self, value: int):
        """fix the phenotype of the trap space to be either 0 or 1

        Args:
            value (int): the value to fix
        """
        self.fix_var(self.h[self.bn.phenotype, value], 1)

    def add_constr_separation(self, ctrl: Control):
        for j, k in ctrl.items():
            self.fix_var(self.d[j, 1 - k], 0)
            self.relax_var(self.d[j, k])
            self.fix_var(self.h[j, 1 - k], 0)
            self.relax_var(self.h[j, k])

        for j in ctrl.unfixed_vars(self.bn.controllable_vars):
            self.fix_var(self.d[j, 0], 0)
            self.fix_var(self.d[j, 1], 0)
            self.relax_var(self.h[j, 0])
            self.relax_var(self.h[j, 1])

    def set_objective_sparse_cut(self):
        self.set_objective(
            expr=sum(self.h[i, k] for i, k in self.J * self.B),
            _minimize=True,
        )

    def get_trap_space(self) -> Hypercube:
        trap_space = Hypercube()
        for i in self.I:
            if self.h[i, 0].value == 1:
                trap_space[i] = 0
            elif self.h[i, 1].value == 1:
                trap_space[i] = 1
        return trap_space

    def add_trap_space_maximality_cut(self, ctrl: Control, trap_space: Hypercube):
        self.add_constr_to_list(
            pmoenv.quicksum(1 - self.d[j, k] for j, k in ctrl.items())
            + pmoenv.quicksum(1 - self.h[i, k] for i, k in trap_space.items())
            >= 1,
            self.constrs_benders,
        )


Model = TypeVar(
    "Model",
    CoreIP,
    MasterControlIP,
    AttractorDetectionIP,
    TrapSpaceDetectionIP,
    ExtendedAttractorDetectionIP,
)
