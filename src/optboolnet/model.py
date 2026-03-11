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


def _as_binary(expr) -> int:
    """Convert a Pyomo numeric expression/value to a binary int via 0.5 threshold."""
    return 1 if pmoenv.value(expr) > 0.5 else 0


def _is_true(expr) -> bool:
    return _as_binary(expr) == 1


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
        self.K_star = pmoenv.Set(initialize=[0, 1, 2])
        """Three-way control domain: 0=fix-to-0, 1=fix-to-1, 2=uncontrolled"""
        
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
        self.dummy_zero = pmoenv.ScalarVar(domain=[0, 0])
        self.append_vars_to_solvers([self.dummy_zero])
        self.last_termination_condition = None
        self.last_solver_status = None

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
        self.last_termination_condition = results.solver.termination_condition
        self.last_solver_status = results.solver.status

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
            _sum = pmoenv.summation(self.d)
            if isinstance(_sum, int) and (_sum == 0):
                _sum = self.dummy_zero
            self.add_constr_to_list(
                _sum == control_size,
                self.constrs_target_size,
            )

    def get_control(self) -> Control:
        ctrl_dict = dict()
        for j in self.J:
            if _is_true(self.d[j, 0]):
                ctrl_dict[j] = 0
            elif _is_true(self.d[j, 1]):
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
        _sum = pmoenv.quicksum(
            self.d[j, 0] + self.d[j, 1]
            for j in ctrl.unfixed_vars(self.bn.controllable_vars)
        ) + pmoenv.quicksum(
            (1 - self.d[j, k] + self.d[j, 1 - k]) for j, k in ctrl.items()
        )
        if isinstance(_sum, int) and (_sum == 0):
            _sum = self.dummy_zero
        self.add_constr_to_list(
            _sum >= 1,
            self.constrs_benders,
        )
        return (EnumCutType.NO_GOOD_MASTER, 2 * len(self.J))

    def append_minimality_cut(self, ctrl: Control):
        _sum = pmoenv.quicksum((1 - self.d[j, k]) for j, k in ctrl.items())
        if isinstance(_sum, int) and (_sum == 0):
            _sum = self.dummy_zero
        self.add_constr_to_list(
            _sum >= 1,
            self.constrs_minimality,
        )
        return (EnumCutType.MINIMALITY, len(ctrl))

    def append_logical_benders_cut(self, attr: Attractor):
        # Build the list of cut‐terms in two passes:
        #  1) for each j,k,alpha_j,beta_j yield (d[j,1-k] if beta_j==1 else 1 - d[j,k])
        #  2) for each where alpha_j==0 yield d[j,k]
        state0 = attr.get_first_state()
        terms = [
            (self.d[j, 1 - k] if beta_j == 1 else 1 - self.d[j, k])
            for j, k, alpha_j, beta_j in zip(self.J, state0, attr.alpha, attr.beta)
        ] + [
            self.d[j, k]
            for j, k, alpha_j, beta_j in zip(self.J, state0, attr.alpha, attr.beta)
            if alpha_j == 0
        ]

        # Sum them into a single Pyomo expression
        expr = pmoenv.quicksum(terms)

        # Add the Benders cut
        self.add_constr_to_list(expr >= 1, self.constrs_benders)
        return (EnumCutType.ATTRACTOR_CUT, len(terms))

    def append_forbidden_trap_space_cut(
        self, forbidden_ctrl: Control, forbidden_trap_space: Hypercube
    ):
        # build a flat list of all the terms you want to forbid
        terms = list()
        for j, k in forbidden_ctrl.items():
            terms.append (1 - self.d[j, k])
        for j in forbidden_ctrl.unfixed_vars(self.bn.controllable_vars):
            try:
                terms.append( self.d[j, 1 - forbidden_trap_space[j]])
            except:
                pass

        expr = pmoenv.quicksum(terms)
        if isinstance(expr, int) and expr == 0:
            expr = self.dummy_zero
        self.add_constr_to_list(expr >= 1, self.constrs_benders)

        return (EnumCutType.TRAP_SPACE_CUT, len(terms))

    def set_objective_min_control(self):
        return super().set_objective(sum(self.d.values()), True)


class InterdictMasterControlIP(CoreIP):
    """CoreIP with explicit three-way control variables d[j,k], k in {0,1,2}.

    k=0: gene j is fixed to 0  (d^0_j = 1)
    k=1: gene j is fixed to 1  (d^1_j = 1)
    k=2: gene j is uncontrolled (d^*_j = 1)

    Exactly one of {d[j,0], d[j,1], d[j,2]} equals 1 for each j in J
    (equality exclusivity).  Control size counts only k in {0,1}.

    Intended for interdiction bilevel models where the coupling constraints
    delta[j,k] <= 1 - d[j,k] must live in the LLP so that MibS detects the
    non-zero E matrix (ULP vars appearing in LLP constraints).
    """

    def __init__(
        self,
        name: str,
        bn: CNFBooleanNetwork,
        solver_setting: SolverConfig,
        *args,
        **kwds
    ):
        super().__init__(name, bn, solver_setting, *args, **kwds)

        self.d = pmoenv.Var(self.J * self.K_star, domain=pmoenv.Binary)
        """d[j,k]=1 iff variable j is in control state k for all j in J, k in {0,1,2}"""
        self.append_vars_to_solvers([self.d])

        self.constrs_target_size = pmoenv.ConstraintList()
        self.constrs_exclusivity = pmoenv.ConstraintList()
        self.constrs_minimality = pmoenv.ConstraintList()
        self.constrs_benders = pmoenv.ConstraintList()

        self.make_constr_exclusivity()

    def set_constr_target_size(self, control_size: int):
        self.clear_constr_list(self.constrs_target_size)
        if control_size is None:
            return
        # Only d[j,0] and d[j,1] count as "controlled"; d[j,2] means uncontrolled
        _sum = pmoenv.quicksum(self.d[j, k] for j in self.J for k in [0, 1])
        if isinstance(_sum, int) and _sum == 0:
            _sum = self.dummy_zero
        self.add_constr_to_list(_sum == control_size, self.constrs_target_size)

    def get_control(self) -> Control:
        ctrl_dict = dict()
        for j in self.J:
            if _is_true(self.d[j, 0]):
                ctrl_dict[j] = 0
            elif _is_true(self.d[j, 1]):
                ctrl_dict[j] = 1
        return Control(ctrl_dict)

    def make_constr_exclusivity(self):
        """Exactly one of {d[j,0], d[j,1], d[j,2]} equals 1 for each j."""
        self.clear_constr_list(self.constrs_exclusivity)
        for j in self.J:
            self.add_constr_to_list(
                self.d[j, 0] + self.d[j, 1] + self.d[j, 2] == 1,
                self.constrs_exclusivity,
            )

    def append_minimality_cut(self, ctrl: Control):
        """Forbid this exact control: at least one currently-fixed gene must change."""
        _sum = pmoenv.quicksum((1 - self.d[j, k]) for j, k in ctrl.items())
        if isinstance(_sum, int) and _sum == 0:
            _sum = self.dummy_zero
        self.add_constr_to_list(_sum >= 1, self.constrs_minimality)
        return (EnumCutType.MINIMALITY, len(ctrl))


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
            new_state = [_as_binary(self.x[i, t]) for i in self.I]
            if all(new_state != _state for _state in unique_state_seq):
                unique_state_seq.append(new_state)
            else:
                break
        x_1 = [_as_binary(self.x[j, 1]) for j in self.J]
        alpha = [
            all(
                _as_binary(self.x[j, 1]) == _as_binary(self.x[j, t])
                for t in self.T_range
            )
            for j in self.J
        ]
        beta = [
            all(
                _is_true(self.x[j, t])
                == all(_is_true(self.y[j, c, self.prev(t)]) for c in self.C_i[j])
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
                self.d[j, 1] - self.v <= self.x[j, t],  # x >= d[j,1] - v (eq:llp-controllable-4)
                self.constrs_stability,
            )
            self.add_constr_to_list(
                self.d[j, 0] <= 1 - self.x[j, t],  # x <= 1 - d[j,0] (eq:llp-controllable-3)
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


class AggregatedAttractorDetectionIP(MasterControlIP):
    """The aggregated LLP that finds attractors of any length up to max_length in a single model.

    Instead of solving max_length separate T-th LLPs, this model uses binary variable
    w[t]=1 to select the attractor length t and enforces the periodicity condition
    y[i,c,0] = y[i,c,T*] through the w selection. This corresponds to the aggregated
    bilevel formulation in the appendix of the IJOC paper.
    """

    def __init__(
        self,
        name: str,
        bn: CNFBooleanNetwork,
        max_length: int,
        solver_setting: SolverConfig,
        *args,
        **kwds,
    ):
        super().__init__(name, bn, solver_setting, *args, **kwds)
        self.max_length = max_length

        ### ======== index sets

        self.T_range = pmoenv.Set(initialize=range(1, 1 + max_length))
        """Time positions 1 to max_length"""
        self.T_range_0 = pmoenv.Set(initialize=range(0, 1 + max_length))
        """Time positions 0 to max_length; t=0 is the periodicity reference for y"""

        ### ======== variables

        self.x = pmoenv.Var(self.I * self.T_range, domain=pmoenv.Binary)
        """x[i,t] denotes the value of variable i at position t"""
        self.y = pmoenv.Var(self.C * self.T_range_0, domain=pmoenv.Binary)
        """y[i,c,t] is the truth value of clause c of variable i at time t.
        y[i,c,0] is the periodicity reference: y[i,c,0] = y[i,c,T*] where T* is the attractor length."""
        self.p = pmoenv.ScalarVar(domain=pmoenv.Binary)
        """p = 1 iff the phenotype is satisfied at all times t in [1..T_max]"""
        self.w = pmoenv.Var(self.T_range, domain=pmoenv.Binary)
        """w[t] = 1 iff the attractor length is exactly t (eq:llp-w-sum)"""
        self.append_vars_to_solvers(
            [self.x, self.y, self.p, self.w]
        )

        ### ======== constraints

        self.constrs_stability = pmoenv.ConstraintList()
        """"""
        self.constrs_phenotype = pmoenv.ConstraintList()
        """"""
        self.constrs_periodicity = pmoenv.ConstraintList()
        """"""
        self.constrs_no_good_x = pmoenv.ConstraintList()
        """"""

    def make_constr_phenotype_and_length(self):
        """Constraints for length selection (w) and phenotype indicator p.

        Implements eq:llp-w-sum and eq:llp-ph-ub/eq:llp-ph from ijoc_formulation.tex.
        """
        self.clear_constr_list(self.constrs_phenotype)

        # sum(w) = 1: exactly one attractor length is selected (eq:llp-w-sum)
        self.add_constr_to_list(
            pmoenv.summation(self.w) == 1,
            self.constrs_phenotype,
        )

        # p <= x_phi,t for all t (eq:llp-ph-ub)
        for t in self.T_range:
            self.add_constr_to_list(
                self.p <= self.x[self.bn.phenotype, t],
                self.constrs_phenotype,
            )

        # p >= 1 - sum_t (1 - x_phi,t) (eq:llp-ph)
        self.add_constr_to_list(
            self.p
            >= 1
            - pmoenv.quicksum(1 - self.x[self.bn.phenotype, t] for t in self.T_range),
            self.constrs_phenotype,
        )

    def make_constr_periodicity(self):
        """Enforce y[i,c,0] = y[i,c,T*] via the w selection (eq:agg-llp-bary-1).

        -(1 - w[t]) <= y[c,0] - y[c,t] <= (1 - w[t]) for all c, t.
        When w[t]=1: y[c,0] = y[c,t] (the period-t boundary condition).
        """
        self.clear_constr_list(self.constrs_periodicity)
        for (i, c) in self.C:
            for t in self.T_range:
                self.add_constr_to_list(
                    self.y[i, c, 0] - self.y[i, c, t] <= 1 - self.w[t],
                    self.constrs_periodicity,
                )
                self.add_constr_to_list(
                    self.y[i, c, t] - self.y[i, c, 0] <= 1 - self.w[t],
                    self.constrs_periodicity,
                )

    def make_constr_stability_condition(self):
        """Transition and literal constraints (no v variable).

        Uses y[i,c,t-1] for the transition at time t; at t=1 this is y[i,c,0],
        the periodicity reference (eq:agg-llp-uncon-1 through eq:agg-llp-lit-3).
        """
        self.clear_constr_list(self.constrs_stability)

        # Fix constraints for controlled genes (eq:agg-llp-fix-0, eq:agg-llp-fix-1)
        for j, t in self.J * self.T_range:
            self.add_constr_to_list(
                self.d[j, 1] <= self.x[j, t],
                self.constrs_stability,
            )
            self.add_constr_to_list(
                self.d[j, 0] <= 1 - self.x[j, t],
                self.constrs_stability,
            )

        # Transition formulas; t_prev = t-1, so at t=1 uses y[i,c,0] (periodicity ref)
        for i in self.I:
            (d_0, d_1) = (self.d[i, 0], self.d[i, 1]) if i in self.J else (0, 0)
            for t in self.T_range:
                t_prev = t - 1
                for c in self.C_i[i]:
                    self.add_constr_to_list(
                        self.x[i, t] <= self.y[i, c, t_prev] + (d_0 + d_1),
                        self.constrs_stability,
                    )
                self.add_constr_to_list(
                    self.x[i, t]
                    >= (1 - len(self.C_i[i]))
                    + sum(self.y[i, c, t_prev] for c in self.C_i[i])
                    - (d_0 + d_1),
                    self.constrs_stability,
                )

        # Clause-literal synchronization for t in T_range (eq:agg-llp-lit-1 through eq:agg-llp-lit-3)
        for (i, c), clause in self.bn.iter_clauses():
            for t in self.T_range:
                x_lit_list = [self.x[i_, t] for i_ in clause.pos_literals] + [
                    1 - self.x[i_, t] for i_ in clause.neg_literals
                ]
                for x_lit in x_lit_list:
                    self.add_constr_to_list(
                        self.y[i, c, t] >= x_lit,
                        self.constrs_stability,
                    )
                self.add_constr_to_list(
                    self.y[i, c, t] <= sum(x_lit_list),
                    self.constrs_stability,
                )

    def fix_length(self, T: int):
        """Parameterize this model as the T-th LLP by fixing w_T=1 and w_t=0 for t != T.

        Assumes this model was built with max_length == T so T_range = [1..T].
        Then w[T]=1 and periodicity constraints enforce y[c,0] = y[c,T], making
        this equivalent to the fixed-length subproblem \LLPModelAtT.
        """
        for t in self.T_range:
            self.fix_var(self.w[t], 1 if t == T else 0)

    def set_phenotype_obj(self, _minimize: bool = True):
        self.set_objective(expr=self.p, _minimize=_minimize)

    def get_attractor(self) -> Attractor:
        """Extract the attractor determined by the w selection."""
        T_star = next(t for t in self.T_range if _is_true(self.w[t]))
        unique_state_seq: List[List[int]] = list()
        for t in range(1, T_star + 1):
            new_state = [_as_binary(self.x[i, t]) for i in self.I]
            if all(new_state != _state for _state in unique_state_seq):
                unique_state_seq.append(new_state)
            else:
                break
        x_1 = [_as_binary(self.x[j, 1]) for j in self.J]
        alpha = [
            all(
                _as_binary(self.x[j, 1]) == _as_binary(self.x[j, t])
                for t in range(1, T_star + 1)
            )
            for j in self.J
        ]
        beta = [
            all(
                _is_true(self.x[j, t])
                == all(_is_true(self.y[j, c, t - 1]) for c in self.C_i[j])
                for t in range(1, T_star + 1)
            )
            for j in self.J
        ]
        return Attractor(self.bn, unique_state_seq, x_1, alpha, beta)


class LongestAttractorDetectionIP(AggregatedAttractorDetectionIP):
    """Single-level MILP for jointly selecting control and attractor with maximum length.

    This extends AggregatedAttractorDetectionIP by:
    - maximizing selected attractor length via w
    - enforcing anti-subcycle constraints with XOR indicators
    - supporting phenotype mode: "violating" (p=0) or "none"
    """

    PHENOTYPE_MODES = {"violating", "none"}

    def __init__(
        self,
        name: str,
        bn: CNFBooleanNetwork,
        max_length: int,
        solver_setting: SolverConfig,
        phenotype_mode: str = "violating",
        *args,
        **kwds,
    ):
        super().__init__(name, bn, max_length, solver_setting, *args, **kwds)
        self.phenotype_mode = phenotype_mode
        self.T_sub_prev = pmoenv.Set(initialize=range(1, max_length))
        """Indices t in [1..Tmax-1] used for q[t] = sum_{r=t+1..Tmax} w[r]."""
        self.T_sub = pmoenv.Set(initialize=range(2, 1 + max_length))
        """Indices t in [2..Tmax] used for no-repetition against state at t=1."""

        self.q = pmoenv.Var(self.T_sub_prev, domain=pmoenv.Binary)
        """q[t]=1 iff selected attractor length is strictly greater than t."""
        self.delta = pmoenv.Var(self.I * self.T_sub, domain=pmoenv.Binary)
        """delta[i,t]=1 iff x[i,t] differs from x[i,1] (XOR linearization)."""
        self.append_vars_to_solvers([self.q, self.delta])

        self.constrs_subcycle = pmoenv.ConstraintList()
        """"""
        self.constrs_max_control_size = pmoenv.ConstraintList()
        """"""
        self.constrs_state_periodicity = pmoenv.ConstraintList()
        """"""

    def make_constr_subcycle_prevention(self):
        """Prevent repeating the first state before the selected cycle length.

        Strong equivalent chain for q:
            q[t] - q[t+1] = w[t+1] for t in [1..Tmax-2]
            q[Tmax-1] = w[Tmax]

        and for time t+1, enforce at least one i differs from state 1 when q[t]=1.
        Also force delta to zero when q[t]=0.
        """
        self.clear_constr_list(self.constrs_subcycle)

        if self.max_length <= 1:
            return

        for t in range(1, self.max_length - 1):
            self.add_constr_to_list(
                self.q[t] - self.q[t + 1] == self.w[t + 1],
                self.constrs_subcycle,
            )

        self.add_constr_to_list(
            self.q[self.max_length - 1] == self.w[self.max_length],
            self.constrs_subcycle,
        )

        for t in self.T_sub:
            t_prev = t - 1
            for i in self.I:
                x_it = self.x[i, t]
                x_i1 = self.x[i, 1]
                d_it = self.delta[i, t]
                self.add_constr_to_list(d_it >= x_it - x_i1, self.constrs_subcycle)
                self.add_constr_to_list(d_it >= x_i1 - x_it, self.constrs_subcycle)
                self.add_constr_to_list(d_it <= x_it + x_i1, self.constrs_subcycle)
                self.add_constr_to_list(d_it <= 2 - x_it - x_i1, self.constrs_subcycle)

            self.add_constr_to_list(
                pmoenv.quicksum(self.delta[i, t] for i in self.I) >= self.q[t_prev],
                self.constrs_subcycle,
            )

    def set_length_objective(self):
        self.set_objective(
            expr=pmoenv.quicksum(t * self.w[t] for t in self.T_range),
            _minimize=False,
        )

    def set_phenotype_mode(self, phenotype_mode: str = "violating"):
        if phenotype_mode not in self.PHENOTYPE_MODES:
            raise ValueError(
                f"Invalid phenotype_mode='{phenotype_mode}'. "
                f"Expected one of: {sorted(self.PHENOTYPE_MODES)}"
            )
        self.phenotype_mode = phenotype_mode
        if phenotype_mode == "violating":
            self.fix_var(self.p, 0)
        else:
            self.relax_var(self.p)

    def set_constr_max_control_size(self, max_control_size: Optional[int]):
        self.clear_constr_list(self.constrs_max_control_size)
        if max_control_size is None:
            return
        if max_control_size < 0:
            raise ValueError(f"max_control_size must be >= 0, got {max_control_size}")
        self.add_constr_to_list(
            pmoenv.quicksum(self.d[j, k] for j in self.J for k in self.B)
            <= max_control_size,
            self.constrs_max_control_size,
        )

    def make_constr_state_periodicity(self, enabled: bool = False):
        """Optional tightening: when w[t]=1, force x[i,1]=x[i,t'] for t' = 1 (mod t).

        For each selected period t and each t' in {1+t, 1+2t, ...} within [1..Tmax]:
            -(1-w[t]) <= x[i,1] - x[i,t'] <= (1-w[t])
        """
        self.clear_constr_list(self.constrs_state_periodicity)
        if not enabled:
            return

        for t in self.T_range:
            for t_prime in range(1 + t, self.max_length + 1, t):
                for i in self.I:
                    self.add_constr_to_list(
                        self.x[i, 1] - self.x[i, t_prime] <= 1 - self.w[t],
                        self.constrs_state_periodicity,
                    )
                    self.add_constr_to_list(
                        self.x[i, t_prime] - self.x[i, 1] <= 1 - self.w[t],
                        self.constrs_state_periodicity,
                    )

    def get_selected_length(self) -> Optional[int]:
        for t in self.T_range:
            if self.w[t].value is not None and _is_true(self.w[t]):
                return int(t)
        return None

    def get_result(self) -> Dict:
        def _safe_obj_value():
            try:
                return float(pmoenv.value(self.obj.expr))
            except Exception:
                return None

        term = (
            str(self.last_termination_condition)
            if self.last_termination_condition is not None
            else None
        )
        status = str(self.last_solver_status) if self.last_solver_status is not None else None
        feasible = self.last_termination_condition in [
            TerminationCondition.feasible,
            TerminationCondition.optimal,
        ]
        result = {
            "termination_condition": term,
            "solver_status": status,
            "objective_value": _safe_obj_value(),
            "phenotype_mode": self.phenotype_mode,
            "selected_length": None,
            "control": None,
            "attractor_states": None,
            "attractor_length": None,
            "phenotype_indicator_p": None,
        }
        if not feasible:
            return result

        T_star = self.get_selected_length()
        attr = self.get_attractor() if T_star is not None else None
        result["selected_length"] = T_star
        result["control"] = dict(sorted(self.get_control().items()))
        result["attractor_states"] = attr.to_str_list() if attr is not None else None
        result["attractor_length"] = attr.get_length() if attr is not None else None
        result["phenotype_indicator_p"] = _as_binary(self.p)
        return result


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
        # self.fix_var(self.h[self.bn.phenotype, value], 1)
        self.add_constr_to_list(
            self.h[self.bn.phenotype, value] == 1,
            self.constrs_phenotype,
        )

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
            if _is_true(self.h[i, 0]):
                trap_space[i] = 0
            elif _is_true(self.h[i, 1]):
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
    InterdictMasterControlIP,
    AttractorDetectionIP,
    ExtendedAttractorDetectionIP,
    AggregatedAttractorDetectionIP,
    LongestAttractorDetectionIP,
    TrapSpaceDetectionIP,
)
