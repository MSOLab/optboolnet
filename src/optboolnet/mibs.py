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

        ### ======== variables
        self.p_T = pmoenv.Var(self.T_range, domain=pmoenv.Binary)
        self.v_T = pmoenv.Var(self.T_range, domain=pmoenv.Binary)
        self.x_T = pmoenv.Var(self.I * self.t_T_range, domain=pmoenv.Binary)
        self.y_T = pmoenv.Var(self.C * self.t_T_range, domain=pmoenv.Binary)

        ### ========  the upper-level constraints
        self.constrs_phenotype = pmoenv.ConstraintList()
        """"""
        for T in self.T_range:
            self.constrs_phenotype.add(expr=self.p_T[T] + self.v_T[T] - 1 >= 0)

        ### ========  the upper-level objective
        self.obj = pmoenv.Objective(expr=sum(d_var for d_var in self.d.values()))

        ### ======== the lower-level constraints
        self.LLP = SubModel(fixed=[self.d])

        self.LLP.constrs_phenotype = pmoenv.ConstraintList()

        # phenotype condition
        for t, T in self.t_T_range:
            self.LLP.constrs_phenotype.add(
                expr=self.p_T[T] <= self.x_T[self.bn.phenotype, t, T]
            )
        for T in self.T_range:
            self.LLP.constrs_phenotype.add(
                expr=self.p_T[T]
                >= 1
                + sum(self.x_T[self.bn.phenotype, t, T] for t in self.t_range[T])
                - T
            )

        # stability condition
        self.LLP.constrs_stability = pmoenv.ConstraintList()
        for j, t, T in self.J * self.t_T_range:
            self.LLP.constrs_stability.add(
                -self.v_T[T] + self.d[j, 1] <= self.x_T[j, t, T],
            )
            self.LLP.constrs_stability.add(
                self.d[j, 0] <= 1 - self.x_T[j, t, T],
            )

        for i in self.I:
            (d_0, d_1) = (self.d[i, 0], self.d[i, 1]) if i in self.J else (0, 0)
            for t, T in self.t_T_range:
                x_i_t = self.x_T[i, t, T]
                for c in self.C_i[i]:
                    self.LLP.constrs_stability.add(
                        x_i_t <= self.y_T[i, c, self.prev(t, T), T] + (d_0 + d_1),
                    )
                self.LLP.constrs_stability.add(
                    x_i_t
                    >= (1 - len(self.C_i[i]))
                    + sum(self.y_T[i, c, self.prev(t, T), T] for c in self.C_i[i])
                    - (d_0 + d_1)
                )

        for (i, c), clause in self.bn.iter_clauses():
            for t, T in self.t_T_range:
                x_lit_list = [self.x_T[i_, t, T] for i_ in clause.pos_literals] + [
                    1 - self.x_T[i_, t, T] for i_ in clause.neg_literals
                ]

                for i_ in clause.pos_literals:
                    self.LLP.constrs_stability.add(
                        self.y_T[i, c, t, T] >= self.x_T[i_, t, T]
                    )
                for i_ in clause.neg_literals:
                    self.LLP.constrs_stability.add(
                        self.y_T[i, c, t, T] >= 1 - self.x_T[i_, t, T] - self.v_T[T]
                    )
                self.LLP.constrs_stability.add(self.y_T[i, c, t, T] <= sum(x_lit_list))

        self.constrs_no_good_x = pmoenv.ConstraintList()

        ### ======== the lower-level objective

        self.LLP.obj = pmoenv.Objective(
            expr=sum(p for p in self.p_T.values())
            + sum(2 * v for v in self.v_T.values())
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
                    self.constrs_benders.add(self.v_T[T] <= self.v_T[T_])
                    self.constrs_benders.add(self.p_T[T] >= self.p_T[T_])

    def not_allow_empty_attractor(self):
        self.constrs_phenotype.add(
            sum(self.v_T[T] for T in self.T_range) <= self.length - 1
        )

    # TODO: communicate solutions through a file stream (.sol or mps style)

    def optimize(self):
        pyomo_solver = Solver("pao.pyomo.MIBS")
        with Solver("pao.mpr.MIBS") as mpr_solver:
            mpr_solver: LinearMultilevelSolver_MIBS
            lmp, soln_manager = convert_pyomo2MultilevelProblem(self)
            results = PyomoSubmodelResults(solution_manager=soln_manager)

            temp_mps, temp_aux = "mibs_temp.mps", "mibs_temp.aux"
            mpr_solver.create_mibs_model(lmp, temp_mps, temp_aux)
            cmd = [self.solver_config.executable, "-Alps_instance", temp_mps]
            ans = run_shellcmd(
                cmd,
                tee=self.solver_config.tee,
                time_limit=self.solver.options["time_limit"],
            )
            os.remove(temp_mps)
            os.remove(temp_aux)
            line_iter = PrintIter(iter(ans["log"].split("\r\n")))
            _line = next(line_iter)

            lmp_results = Results()
            lmp_results.solver.rc = ans.rc

            lmp.U.x.values = [0.0] * len(lmp.U.x)
            lmp.U.LL.x.values = [0.0] * len(lmp.U.LL.x)

            # skip irrelevant lines
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
            pyomo_solver._initialize_results(results, lmp_results, self, lmp, None)
            results.solver.rc = lmp_results.solver.rc
            results.copy(From=lmp, To=self)
            return True


class MibSAttractorControl(AttractorControl):
    def __init__(
        self,
        name: str,
        bn: CNFBooleanNetwork,
        _config: Union[str, dict, MibSBilevelConfig],
    ) -> None:
        self._config: MibSBilevelConfig = MibSBilevelConfig.instantiate(_config)

        super().__init__(name, bn, _config.logging_config)
        self.name = name
        self.bn = bn
        self.step = EnumBendersStep.BUILD_MODEL
        self.target_size: int = 0
        self.solution_dict: Dict[int, List[Control]] = dict()

        self.model_bilevel: MibSBilevelIP = self.build_model(
            MibSBilevelIP, name, bn, _config.max_length, _config.solver_config
        )

    def get_control_strategies(self):
        # TODO: merge with a new parent class
        # preprocessing
        if self._config.use_valid_cuts:
            self.model_bilevel.add_valid_cut()
        if not self._config.allow_empty_attractor:
            self.model_bilevel.not_allow_empty_attractor()
        # main step
        for self.target_size in self.iter_target_size():
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
