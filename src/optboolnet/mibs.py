from typing import Dict, Iterator, List, Optional, Union
import pyomo.environ as pmoenv
from pao.common import Solver, run_shellcmd, Results
from pao.mpr.solvers.mibs import LinearMultilevelSolver_MIBS
from pao.pyomo.solver import convert_pyomo2MultilevelProblem, PyomoSubmodelResults
from pao.pyomo import SubModel
import os
from optboolnet import CNFBooleanNetwork

from optboolnet.boolnet import CNFBooleanNetwork, Control
from optboolnet.config import MibSBilevelConfig, SolverMibSConfig
from optboolnet.log import BendersLogger, EnumBendersStep
from optboolnet.model import MasterControlIP
from optboolnet.algorithm import AttractorControl


class PrintIter(Iterator):
    def __init__(self, old_iter):
        self.old_iter = old_iter

    def __next__(self):
        value = next(self.old_iter, None)
        if value == None:
            raise StopIteration()
        else:
            # print(value)
            return value


def _run_mibs(model: MasterControlIP) -> bool:
    """Shared MibS solve routine for bilevel models.

    Reads the pre-computed time limit from model.solver.options (set by
    AttractorControl._optimize → update_options_time_limit) and passes it
    as -Alps_timeLimit so MibS stops cleanly within the wall-clock budget.
    """
    pyomo_solver = Solver("pao.pyomo.MIBS")
    with Solver("pao.mpr.MIBS") as mpr_solver:
        mpr_solver: LinearMultilevelSolver_MIBS
        lmp, soln_manager = convert_pyomo2MultilevelProblem(model)
        results = PyomoSubmodelResults(solution_manager=soln_manager)

        temp_mps, temp_aux = "mibs_temp.mps", "mibs_temp.aux"
        mpr_solver.create_mibs_model(lmp, temp_mps, temp_aux)

        # Build MibS command; pass remaining wall-clock time as an internal limit
        # so MibS can report partial results before the process is killed externally.
        cmd = [model.solver_config.executable, "-Alps_instance", temp_mps]
        time_limit = model.solver.options.get("time_limit", None)
        if time_limit is not None:
            cmd += ["-Alps_timeLimit", str(int(max(1, time_limit)))]

        ans = run_shellcmd(cmd, tee=model.solver_config.tee, time_limit=time_limit)
        os.remove(temp_mps)
        os.remove(temp_aux)

        line_iter = PrintIter(iter(ans["log"].split("\r\n")))
        _line = next(line_iter)

        lmp_results = Results()
        lmp_results.solver.rc = ans.rc

        lmp.U.x.values = [0.0] * len(lmp.U.x)
        lmp.U.LL.x.values = [0.0] * len(lmp.U.LL.x)

        while _line != "Optimal solution:":
            try:
                _line = next(line_iter)
            except StopIteration:
                return False

        lmp_results.solver.best_feasible_objective = float(
            next(line_iter).split(" = ")[1]
        )
        while _line != "First stage (upper level) variable values:":
            _line = next(line_iter)
        _line = next(line_iter)
        while _line != "Second stage (lower level) variable values:":
            var_name, value = _line.split(" = ")
            lmp.U.x.values[int(var_name[1:]) - 1] = float(value)
            _line = next(line_iter)
        _line = next(line_iter)
        while not _line.startswith("Number"):
            var_name, value = _line.split(" = ")
            lmp.U.LL.x.values[int(var_name[1:]) - len(lmp.U.x) - 1] = float(value)
            _line = next(line_iter)

        for _line in line_iter:
            pass
        pyomo_solver._initialize_results(results, lmp_results, model, lmp, None)
        results.solver.rc = lmp_results.solver.rc
        results.copy(From=lmp, To=model)
        return True


class MibSBilevelIP(MasterControlIP):
    def __init__(
        self,
        name: str,
        bn: CNFBooleanNetwork,
        length: int,
        solver_config: SolverMibSConfig,  # TODO: Currently ignored.
        *args,
        **kwds
    ):
        """The bilevel model in an extensive form

        Args:
            name (str): _description_
            bn (CNFBooleanNetwork): _description_
            solver_config (SolverConfig): _description_
        """
        super().__init__(name, bn, solver_config, *args, **kwds)
        self.solver_config = solver_config
        # TODO: Fix the misleading solver_config
        self.length = length

        ### ======== index sets
        def t_range_init(model: MibSBilevelIP, T: int):
            return [t for t in range(1, 1 + T)]

        def t_T_range_init(model: MibSBilevelIP):
            return ((t, T) for T in range(1, 1 + model.length) for t in range(1, 1 + T))

        self.T_range = pmoenv.Set(initialize=range(1, 1 + self.length))
        self.t_range = pmoenv.Set(self.T_range, initialize=t_range_init)
        self.t_T_range = pmoenv.Set(initialize=t_T_range_init)

        ### ======== the upper-level objective
        self.obj = pmoenv.Objective(expr=sum(d_var for d_var in self.d.values()))

        ### ======== the lower-level submodel (must be created before LLP variables)
        self.LLP = SubModel(fixed=[self.d])

        ### ======== LLP variables (declared on self.LLP so PAO classifies them correctly)
        self.LLP.p_T = pmoenv.Var(self.T_range, domain=pmoenv.Binary)
        self.LLP.v_T = pmoenv.Var(self.T_range, domain=pmoenv.Binary)
        self.LLP.x_T = pmoenv.Var(self.I * self.t_T_range, domain=pmoenv.Binary)
        self.LLP.y_T = pmoenv.Var(self.C * self.t_T_range, domain=pmoenv.Binary)

        ### ======== the upper-level constraints (linking ULP to LLP variables)
        self.constrs_phenotype = pmoenv.ConstraintList()
        for T in self.T_range:
            self.constrs_phenotype.add(expr=self.LLP.p_T[T] + self.LLP.v_T[T] - 1 >= 0)

        ### ======== the lower-level constraints
        self.LLP.constrs_phenotype = pmoenv.ConstraintList()

        # phenotype condition
        for t, T in self.t_T_range:
            self.LLP.constrs_phenotype.add(
                expr=self.LLP.p_T[T] <= self.LLP.x_T[self.bn.phenotype, t, T]
            )
        for T in self.T_range:
            self.LLP.constrs_phenotype.add(
                expr=self.LLP.p_T[T]
                >= 1
                + sum(self.LLP.x_T[self.bn.phenotype, t, T] for t in self.t_range[T])
                - T
            )

        # stability condition
        self.LLP.constrs_stability = pmoenv.ConstraintList()
        for j, t, T in self.J * self.t_T_range:
            self.LLP.constrs_stability.add(
                -self.LLP.v_T[T] + self.d[j, 1] <= self.LLP.x_T[j, t, T],
            )
            self.LLP.constrs_stability.add(
                self.d[j, 0] <= 1 - self.LLP.x_T[j, t, T],
            )

        for i in self.I:
            (d_0, d_1) = (self.d[i, 0], self.d[i, 1]) if i in self.J else (0, 0)
            for t, T in self.t_T_range:
                x_i_t = self.LLP.x_T[i, t, T]
                for c in self.C_i[i]:
                    self.LLP.constrs_stability.add(
                        x_i_t <= self.LLP.y_T[i, c, self.prev(t, T), T] + (d_0 + d_1),
                    )
                self.LLP.constrs_stability.add(
                    x_i_t
                    >= (1 - len(self.C_i[i]))
                    + sum(self.LLP.y_T[i, c, self.prev(t, T), T] for c in self.C_i[i])
                    - (d_0 + d_1)
                )

        for (i, c), clause in self.bn.iter_clauses():
            for t, T in self.t_T_range:
                x_lit_list = [self.LLP.x_T[i_, t, T] for i_ in clause.pos_literals] + [
                    1 - self.LLP.x_T[i_, t, T] for i_ in clause.neg_literals
                ]

                for i_ in clause.pos_literals:
                    self.LLP.constrs_stability.add(
                        self.LLP.y_T[i, c, t, T] >= self.LLP.x_T[i_, t, T]
                    )
                for i_ in clause.neg_literals:
                    self.LLP.constrs_stability.add(
                        self.LLP.y_T[i, c, t, T] >= 1 - self.LLP.x_T[i_, t, T] - self.LLP.v_T[T]
                    )
                self.LLP.constrs_stability.add(self.LLP.y_T[i, c, t, T] <= sum(x_lit_list))

        self.constrs_no_good_x = pmoenv.ConstraintList()

        ### ======== the lower-level objective
        self.LLP.obj = pmoenv.Objective(
            expr=sum(p for p in self.LLP.p_T.values())
            + sum(2 * v for v in self.LLP.v_T.values())
        )

    def prev(self, t: int, T: int):
        if t == 1:
            return T
        else:
            return t - 1

    def add_valid_cut(self):
        for T in self.T_range:
            for T_ in range(1, T):
                if T % T_ == 0:
                    self.constrs_benders.add(self.LLP.v_T[T] <= self.LLP.v_T[T_])
                    self.constrs_benders.add(self.LLP.p_T[T] >= self.LLP.p_T[T_])

    def not_allow_empty_attractor(self):
        self.constrs_phenotype.add(
            sum(self.LLP.v_T[T] for T in self.T_range) <= self.length - 1
        )

    # TODO: communicate solutions through a file stream (.sol or mps style)

    def optimize(self):
        return _run_mibs(self)


class MibSAggBilevelIP(MasterControlIP):
    """Aggregated bilevel model using a single LLP with w[t] length selector.

    Replaces the extensive-form T separate LLPs in MibSBilevelIP with one
    aggregated LLP that selects attractor length via w[t]. A scalar v variable
    handles the case where no valid attractor exists (v=1 → infeasibility relaxation),
    making the LLP always feasible regardless of the control d.

    ULP: min sum(d), s.t. p + v >= 1  [rational reaction constraint]
    LLP: min p + 2v, s.t. aggregated stability/phenotype/periodicity constraints
    """

    def __init__(
        self,
        name: str,
        bn: CNFBooleanNetwork,
        length: int,
        solver_config: SolverMibSConfig,
        *args,
        **kwds,
    ):
        super().__init__(name, bn, solver_config, *args, **kwds)
        self.solver_config = solver_config
        self.length = length

        ### ======== index sets
        self.T_range = pmoenv.Set(initialize=range(1, 1 + length))
        self.T_range_0 = pmoenv.Set(initialize=range(0, 1 + length))

        ### ======== ULP objective
        self.obj = pmoenv.Objective(expr=sum(d_var for d_var in self.d.values()))

        ### ======== LLP submodel
        self.LLP = SubModel(fixed=[self.d])

        ### ======== LLP variables
        self.LLP.x = pmoenv.Var(self.I * self.T_range, domain=pmoenv.Binary)
        """x[i,t] = state of gene i at time step t"""
        self.LLP.y = pmoenv.Var(self.C * self.T_range_0, domain=pmoenv.Binary)
        """y[i,c,t] = clause c of gene i satisfied at t; y[i,c,0] is periodicity reference"""
        self.LLP.p = pmoenv.ScalarVar(domain=pmoenv.Binary)
        """p=1 iff phenotype satisfied at every active time step"""
        self.LLP.v = pmoenv.ScalarVar(domain=pmoenv.Binary)
        """v=1 iff no valid attractor exists (infeasibility slack)"""
        self.LLP.w = pmoenv.Var(self.T_range, domain=pmoenv.Binary)
        """w[t]=1 iff attractor length is exactly t"""
        self.LLP.o = pmoenv.Var(self.T_range, domain=pmoenv.Binary)
        """o[t]=1 iff time step t is active (t <= selected length)"""
        self.LLP.p_bar = pmoenv.Var(self.T_range, domain=pmoenv.Binary)
        """p_bar[t]=1 iff phenotype violated at active time t"""

        ### ======== ULP constraint: rational reaction (linking)
        self.constrs_phenotype = pmoenv.ConstraintList()
        self.constrs_phenotype.add(self.LLP.p + self.LLP.v >= 1)

        ### ======== LLP constraints
        self.LLP.constrs_phenotype = pmoenv.ConstraintList()

        # sum(w) + v = 1: exactly one length selected OR infeasibility (v=1)
        self.LLP.constrs_phenotype.add(
            pmoenv.summation(self.LLP.w) + self.LLP.v == 1
        )

        # o[t] = sum(w[t'] for t' >= t): cumulative sum from right
        for t in self.T_range:
            self.LLP.constrs_phenotype.add(
                self.LLP.o[t]
                == pmoenv.quicksum(self.LLP.w[t_] for t_ in self.T_range if t_ >= t)
            )

        # p_bar[t] = o[t] * (1 - x[phi,t]): phenotype violated at active time t
        for t in self.T_range:
            self.LLP.constrs_phenotype.add(self.LLP.p_bar[t] <= self.LLP.o[t])
            self.LLP.constrs_phenotype.add(
                self.LLP.p_bar[t] <= 1 - self.LLP.x[self.bn.phenotype, t]
            )
            self.LLP.constrs_phenotype.add(
                self.LLP.p_bar[t] >= self.LLP.o[t] - self.LLP.x[self.bn.phenotype, t]
            )

        # p <= 1 - p_bar[t]; p >= 1 - sum(p_bar)
        for t in self.T_range:
            self.LLP.constrs_phenotype.add(self.LLP.p <= 1 - self.LLP.p_bar[t])
        self.LLP.constrs_phenotype.add(
            self.LLP.p >= 1 - pmoenv.summation(self.LLP.p_bar)
        )

        # periodicity: y[i,c,0] = y[i,c,T*] enforced by w[T*]=1
        self.LLP.constrs_periodicity = pmoenv.ConstraintList()
        for (i, c) in self.C:
            for t in self.T_range:
                self.LLP.constrs_periodicity.add(
                    self.LLP.y[i, c, 0] - self.LLP.y[i, c, t] <= 1 - self.LLP.w[t]
                )
                self.LLP.constrs_periodicity.add(
                    self.LLP.y[i, c, t] - self.LLP.y[i, c, 0] <= 1 - self.LLP.w[t]
                )

        # stability: controllability, transitions, literals
        self.LLP.constrs_stability = pmoenv.ConstraintList()

        # controllability: d[j,1] - v <= x[j,t]; d[j,0] <= 1 - x[j,t]
        for j in self.J:
            for t in self.T_range:
                self.LLP.constrs_stability.add(
                    self.d[j, 1] - self.LLP.v <= self.LLP.x[j, t]
                )
                self.LLP.constrs_stability.add(
                    self.d[j, 0] <= 1 - self.LLP.x[j, t]
                )

        # transition formulas: x[i,t] follows f_i(y[i,c,t-1]); at t=1 uses y[i,c,0]
        for i in self.I:
            (d_0, d_1) = (self.d[i, 0], self.d[i, 1]) if i in self.J else (0, 0)
            for t in self.T_range:
                t_prev = t - 1  # y[i,c,0] at t=1 is the periodicity reference
                for c in self.C_i[i]:
                    self.LLP.constrs_stability.add(
                        self.LLP.x[i, t] <= self.LLP.y[i, c, t_prev] + (d_0 + d_1)
                    )
                self.LLP.constrs_stability.add(
                    self.LLP.x[i, t]
                    >= (1 - len(self.C_i[i]))
                    + sum(self.LLP.y[i, c, t_prev] for c in self.C_i[i])
                    - (d_0 + d_1)
                )

        # literal constraints: positive literals unchanged; negative literals relaxed by v
        for (i, c), clause in self.bn.iter_clauses():
            for t in self.T_range:
                x_lit_list = [self.LLP.x[i_, t] for i_ in clause.pos_literals] + [
                    1 - self.LLP.x[i_, t] for i_ in clause.neg_literals
                ]
                for i_ in clause.pos_literals:
                    self.LLP.constrs_stability.add(
                        self.LLP.y[i, c, t] >= self.LLP.x[i_, t]
                    )
                for i_ in clause.neg_literals:
                    self.LLP.constrs_stability.add(
                        self.LLP.y[i, c, t] >= 1 - self.LLP.x[i_, t] - self.LLP.v
                    )
                self.LLP.constrs_stability.add(self.LLP.y[i, c, t] <= sum(x_lit_list))

        self.constrs_no_good_x = pmoenv.ConstraintList()

        ### ======== LLP objective
        self.LLP.obj = pmoenv.Objective(
            expr=self.LLP.p + 2 * self.LLP.v
        )

    def not_allow_empty_attractor(self):
        """Force v=0 so the LLP must find a real attractor (no infeasibility slack)."""
        self.constrs_phenotype.add(self.LLP.v == 0)

    def optimize(self):
        return _run_mibs(self)


class MibSAttractorControl(AttractorControl):
    def __init__(
        self,
        name: str,
        bn: CNFBooleanNetwork,
        _config: Union[str, dict, MibSBilevelConfig],
    ) -> None:
        self._config: MibSBilevelConfig = _config

        super().__init__(name, bn, _config.logging_config)
        self.name = name
        self.bn = bn
        self.step = EnumBendersStep.BUILD_MODEL
        self.target_size: int = 0
        self.solution_dict: Dict[int, List[Control]] = dict()

        # Fix: pull max_control_size and max_length from config
        self.max_control_size = _config.max_control_size
        self.max_length = _config.max_length

        _model_cls = MibSAggBilevelIP if _config.use_aggregated_LLP else MibSBilevelIP
        self.model_bilevel = self._build_model(
            _model_cls, name, bn, _config.max_length, _config.solver_config
        )

    def get_control_strategies(self):
        # TODO: merge with a new parent class
        # preprocessing
        if self._config.use_valid_cuts and isinstance(self.model_bilevel, MibSBilevelIP):
            self.model_bilevel.add_valid_cut()
        if not self._config.allow_empty_attractor:
            self.model_bilevel.not_allow_empty_attractor()
        # main step
        for self.target_size in self.iter_target_size(self.max_control_size):
            print(self.target_size)
            self.step = EnumBendersStep.FULL_BILEVEL
            _solution_list = list()
            self.model_bilevel.set_constr_target_size(self.target_size)
            while not self.is_timeout and self._optimize(self.model_bilevel):
                ctrl = self.model_bilevel.get_control()
                _solution_list.append(ctrl)
                self._append_cut(self.model_bilevel.append_minimality_cut, ctrl)
            self.solution_dict[self.target_size] = _solution_list
            if self.is_timeout:
                break
            self.step = EnumBendersStep.FINISHED
            self.logger.solve_logger_info(self.log_signature)
        self.logger.write_controls_to_json(self.solution_dict)
        return self.solution_dict
