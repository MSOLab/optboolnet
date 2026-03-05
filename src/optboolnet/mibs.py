from typing import Dict, Iterator, List, Optional, Union
import pyomo.environ as pmoenv
from pao.common import Solver, run_shellcmd, Results
from pao.mpr.solvers.mibs import LinearMultilevelSolver_MIBS
from pao.pyomo.solver import convert_pyomo2MultilevelProblem, PyomoSubmodelResults
from pao.pyomo import SubModel
import os
import shutil
import tempfile
from optboolnet import CNFBooleanNetwork

from optboolnet.boolnet import CNFBooleanNetwork, Control
from optboolnet.config import MibSBilevelConfig, SolverMibSConfig
from optboolnet.log import BendersLogger, EnumBendersStep
from optboolnet.model import MasterControlIP, InterdictMasterControlIP
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


def _save_bilevel_model(model: MasterControlIP, path: str) -> None:
    """Write a human-readable dump of the bilevel model.

    Produces two files:
      <path>         – ULP written as an LP file with symbolic variable names.
      <stem>_llp.txt – LLP variables and constraints as plain text.

    The LLP is dumped separately because Pyomo's LP writer does not support
    PAO SubModel blocks.
    """
    stem = path.rsplit(".", 1)[0] if "." in path else path

    # --- ULP as LP (deactivate LLP so the LP writer doesn't see SubModel) ---
    model.LLP.deactivate()
    try:
        model.write(path, io_options={"symbolic_solver_labels": True})
    finally:
        model.LLP.activate()

    # --- LLP as text dump ---
    llp_path = stem + "_llp.txt"
    with open(llp_path, "w") as f:
        f.write("=== LLP VARIABLES ===\n")
        for var in model.LLP.component_data_objects(pmoenv.Var, active=True):
            lb = var.lb if var.lb is not None else "-inf"
            ub = var.ub if var.ub is not None else "+inf"
            f.write(f"  {var.name}  [{lb}, {ub}]\n")

        f.write("\n=== LLP CONSTRAINTS ===\n")
        for constr in model.LLP.component_data_objects(pmoenv.Constraint, active=True):
            f.write(f"  {constr.name}:  {constr.expr}\n")


def _run_mibs(model: MasterControlIP, extra_options: Optional[Dict[str, int]] = None) -> bool:
    """Shared MibS solve routine for bilevel models.

    Reads the pre-computed time limit from model.solver.options (set by
    AttractorControl._optimize → update_options_time_limit) and passes it
    as -Alps_timeLimit so MibS stops cleanly within the wall-clock budget.

    Args:
        extra_options: Additional MibS parameters passed as command-line flags,
            e.g. {"MibS_bilevelProblemType": 1, "MibS_useBendersInterdictionCut": 1}.
    """
    pyomo_solver = Solver("pao.pyomo.MIBS")
    with Solver("pao.mpr.MIBS") as mpr_solver:
        mpr_solver: LinearMultilevelSolver_MIBS
        lmp, soln_manager = convert_pyomo2MultilevelProblem(model)
        results = PyomoSubmodelResults(solution_manager=soln_manager)

        # Optional: write a human-readable model dump before PAO strips variable names.
        # Activated by setting solver_config.save_lp_path to a file path (any extension).
        # Produces two files:
        #   <path>           – ULP written as LP with symbolic names
        #   <stem>_llp.txt   – LLP variables and constraints as a text dump
        save_lp_path = getattr(model.solver_config, "save_lp_path", None)
        if save_lp_path:
            _save_bilevel_model(model, save_lp_path)

        keep_temp = getattr(model.solver_config, "keep_temp", False)
        mibs_work_dir = tempfile.mkdtemp(prefix="mibs_work_")
        mps_fd, temp_mps = tempfile.mkstemp(prefix="mibs_temp_", suffix=".mps", dir=mibs_work_dir)
        aux_fd, temp_aux = tempfile.mkstemp(prefix="mibs_temp_", suffix=".aux", dir=mibs_work_dir)
        os.close(mps_fd)
        os.close(aux_fd)
        prev_cwd = os.getcwd()
        executable = model.solver_config.executable
        if not os.path.isabs(executable):
            executable = os.path.abspath(os.path.join(prev_cwd, executable))
        try:
            # PAO's create_mibs_model uses hard-coded tmp file names in CWD.
            # Use a per-run temp directory to avoid collisions across processes.
            os.chdir(mibs_work_dir)
            mpr_solver.create_mibs_model(lmp, temp_mps, temp_aux)

            # Build MibS command; pass remaining wall-clock time as an internal limit
            # so MibS can report partial results before the process is killed externally.
            cmd = [
                executable,
                "-Alps_instance",
                temp_mps,
                "-MibS_auxiliaryInfoFile",
                temp_aux,
            ]
            if extra_options:
                for key, val in extra_options.items():
                    cmd += [f"-{key}", str(val)]
            time_limit = model.solver.options.get("time_limit", None)
            if time_limit is not None:
                cmd += ["-Alps_timeLimit", str(int(max(1, time_limit)))]
            ans = run_shellcmd(cmd, tee=model.solver_config.tee, time_limit=time_limit)
        finally:
            os.chdir(prev_cwd)
            # Optional: keep temp files for post-mortem inspection.
            # Activated by setting solver_config.keep_temp = True.
            if not keep_temp:
                if os.path.exists(temp_mps):
                    os.remove(temp_mps)
                if os.path.exists(temp_aux):
                    os.remove(temp_aux)
                shutil.rmtree(mibs_work_dir, ignore_errors=True)

        log_lines = ans["log"].splitlines()
        if "Optimal solution:" not in ans["log"]:
            error_line = next((line.strip() for line in log_lines if line.strip().startswith("Error:")), None)
            if error_line is not None:
                raise RuntimeError(f"MibS execution error: {error_line}")

        line_iter = PrintIter(iter(log_lines))
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
    aggregated LLP that selects attractor length via w[t].

    This class follows the final bilevel model in ijoc_formulation.tex:
      ULP: min sum(d), s.t. LLP.p == 1
      LLP: min p, s.t. aggregated stability/phenotype/periodicity constraints
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
        """p=1 iff phenotype satisfied at all times t in [1..T_max]"""
        self.LLP.w = pmoenv.Var(self.T_range, domain=pmoenv.Binary)
        """w[t]=1 iff attractor length is exactly t"""

        ### ======== ULP coupling: enforce p = 1 at bilevel optimum
        self.constrs_phenotype = pmoenv.ConstraintList()
        self.constrs_phenotype.add(self.LLP.p == 1)

        ### ======== LLP constraints
        self.LLP.constrs_phenotype = pmoenv.ConstraintList()

        # sum_t w[t] = 1 (eq:llp-w-sum)
        self.LLP.constrs_phenotype.add(pmoenv.summation(self.LLP.w) == 1)

        # p <= x_phi,t for all t (eq:llp-ph-ub)
        for t in self.T_range:
            self.LLP.constrs_phenotype.add(self.LLP.p <= self.LLP.x[self.bn.phenotype, t])
        # p >= 1 - sum_t (1 - x_phi,t) (eq:llp-ph)
        self.LLP.constrs_phenotype.add(
            self.LLP.p
            >= 1
            - pmoenv.quicksum(1 - self.LLP.x[self.bn.phenotype, t] for t in self.T_range)
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

        # controllability: d[j,1] <= x[j,t]; d[j,0] <= 1 - x[j,t]
        for j in self.J:
            for t in self.T_range:
                self.LLP.constrs_stability.add(
                    self.d[j, 1] <= self.LLP.x[j, t]
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

        # literal constraints: clause-literal synchronization (eq:llp-literal-1~3)
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
                        self.LLP.y[i, c, t] >= 1 - self.LLP.x[i_, t]
                    )
                self.LLP.constrs_stability.add(self.LLP.y[i, c, t] <= sum(x_lit_list))

        self.constrs_no_good_x = pmoenv.ConstraintList()

        ### ======== LLP objective (eq:llp-obj)
        self.LLP.obj = pmoenv.Objective(expr=self.LLP.p)

    def not_allow_empty_attractor(self):
        """No-op: this formulation has no v-relaxation variable."""
        pass

    def optimize(self):
        return _run_mibs(self)


class MibSInterdictBilevelIP(InterdictMasterControlIP):
    """Explicit interdiction bilevel model (max-min formulation).

    Follows the interdiction formulation in the IJOC paper:

        max_{d}  min_{delta, x, y, w, phi, p}  p

    The upper level selects a control d (which genes to fix and to what value).
    The lower level finds the attractor (of length up to T_max) that minimises
    the phenotype indicator p.  Coupling between levels is achieved through
    auxiliary interdiction variables delta^k_j satisfying delta^k_j <= 1 - d^k_j,
    so d^k_j = 1 forces delta^k_j = 0 and activates the corresponding gene constraint.

    For MibS, pass the extra options:
        MibS_bilevelProblemType       = 1
        MibS_objBoundStrategy         = 1
        MibS_useBendersInterdictionCut = 1
    """

    #: Extra MibS command-line options required for the interdiction problem type.
    MIBS_INTERDICTION_OPTIONS: Dict[str, int] = {
        "MibS_bilevelProblemType": 1,
        "MibS_objBoundStrategy": 1,
        "MibS_useBendersInterdictionCut": 1,
    }

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

        # ---- index sets ----        
        self.T_range = pmoenv.Set(initialize=range(1, 1 + length))
        """Time positions 1..T_max"""
        self.T_range_0 = pmoenv.Set(initialize=range(0, 1 + length))
        """Time positions 0..T_max; t=0 is the periodicity reference for y"""

        # ---- ULP objective: maximise p (minimise -p where p is an LLP var) ----
        # For MibS interdiction (bilevelProblemType=1) the ULP objective must equal
        # the LLP objective.  We declare the LLP first so we can reference LLP.p.
        self.LLP = SubModel(fixed=[self.d])

        # ---- LLP variables ----
        self.LLP.delta = pmoenv.Var(self.J * self.K_star, domain=pmoenv.Binary)
        """delta[j,k]=1 auxiliary interdiction variable for k in {0,1,2}"""
        self.LLP.x = pmoenv.Var(self.I * self.T_range, domain=pmoenv.Binary)
        """x[i,t] = state of gene i at time t"""
        self.LLP.y = pmoenv.Var(self.C * self.T_range_0, domain=pmoenv.Binary)
        """y[i,c,t] = truth value of clause c of gene i at time t;
        y[i,c,0] is the periodicity reference (= y[i,c,T*])"""
        self.LLP.p = pmoenv.ScalarVar(domain=pmoenv.Binary)
        """p = 1 iff every active time step satisfies the phenotype"""
        self.LLP.w = pmoenv.Var(self.T_range, domain=pmoenv.Binary)
        """w[t] = 1 iff the selected attractor length is exactly t"""
        self.LLP.phi = pmoenv.Var(self.T_range, domain=pmoenv.Binary)
        """phi[t] = 1 iff time t is active AND the phenotype is violated at t"""

        # ULP objective: minimise -p  ≡  maximise p  (interdiction structure)
        self.obj = pmoenv.Objective(expr=-self.LLP.p, sense=pmoenv.minimize)

        # ---- LLP interdiction coupling: delta^k_j <= 1 - d^k_j ----
        # These MUST be LLP constraints so that d (a ULP variable) appears in the
        # LLP constraint matrix (E matrix in MibS notation).  A zero E matrix causes
        # MibS to report infeasible even when the problem has solutions.
        #
        # Semantics:
        #   d[j,0]=1 (fix to 0)  → delta[j,0]=0 → x[j,t]<=0, i.e. x[j,t]=0
        #   d[j,1]=1 (fix to 1)  → delta[j,1]=0 → x[j,t]>=1, i.e. x[j,t]=1
        #   d[j,2]=1 (free)      → delta[j,2]=0 → dynamics constraints bind
        self.LLP.constrs_interdict = pmoenv.ConstraintList()
        for j in self.J:
            # delta^0_j <= 1 - d^0_j
            self.LLP.constrs_interdict.add(self.LLP.delta[j, 0] <= 1 - self.d[j, 0])
            # delta^1_j <= 1 - d^1_j
            self.LLP.constrs_interdict.add(self.LLP.delta[j, 1] <= 1 - self.d[j, 1])
            # delta^*_j <= 1 - d^*_j  (explicit d[j,2] avoids needing d^0+d^1 on RHS)
            self.LLP.constrs_interdict.add(
                self.LLP.delta[j, 2] <= 1 - self.d[j, 2]
            )

        # ---- LLP constraints ----
        self.LLP.constrs_phenotype = pmoenv.ConstraintList()
        self.LLP.constrs_stability = pmoenv.ConstraintList()
        self.LLP.constrs_periodicity = pmoenv.ConstraintList()

        # (w-sum) sum_t w[t] = 1  →  exactly one attractor length is selected
        self.LLP.constrs_phenotype.add(pmoenv.summation(self.LLP.w) == 1)

        # (wrap) -(1-w[t]) <= y[c,0] - y[c,t] <= (1-w[t])  for all c, t
        # When w[T]=1 this enforces y[c,0] = y[c,T] (periodicity).
        for (i, c) in self.C:
            for t in self.T_range:
                self.LLP.constrs_periodicity.add(
                    self.LLP.y[i, c, 0] - self.LLP.y[i, c, t] <= 1 - self.LLP.w[t]
                )
                self.LLP.constrs_periodicity.add(
                    self.LLP.y[i, c, t] - self.LLP.y[i, c, 0] <= 1 - self.LLP.w[t]
                )

        # (ph-lin) phi[t] = (sum_{t'>=t} w[t']) AND (1 - x[phi, t])
        # Linearised as three inequalities; sum_{t'>=t} w[t'] acts as activity mask.
        for t in self.T_range:
            o_t = pmoenv.quicksum(self.LLP.w[t_] for t_ in self.T_range if t_ >= t)
            self.LLP.constrs_phenotype.add(self.LLP.phi[t] <= o_t)
            self.LLP.constrs_phenotype.add(
                self.LLP.phi[t] <= 1 - self.LLP.x[self.bn.phenotype, t]
            )
            self.LLP.constrs_phenotype.add(
                self.LLP.phi[t] >= o_t - self.LLP.x[self.bn.phenotype, t]
            )

        # (ph) p constraints
        # p <= (1/T_max) * sum_t (1 - phi[t])   (upper bound; strengthens LP relaxation)
        self.LLP.constrs_phenotype.add(
            self.LLP.p
            <= pmoenv.quicksum(1 - self.LLP.phi[t] for t in self.T_range) / self.length
        )
        # p >= 1 - sum_t phi[t]   (p=0 whenever any active time violates phenotype)
        self.LLP.constrs_phenotype.add(
            self.LLP.p >= 1 - pmoenv.summation(self.LLP.phi)
        )

        # (con-1), (con-2), (fix) for controllable genes j in J
        for j in self.J:
            for t in self.T_range:
                t_prev = t - 1  # y[j,c,0] at t=1 is the periodicity reference

                # (fix) x[j,t] <= delta^0_j   AND   x[j,t] >= 1 - delta^1_j
                self.LLP.constrs_stability.add(
                    self.LLP.x[j, t] <= self.LLP.delta[j, 0]
                )
                self.LLP.constrs_stability.add(
                    self.LLP.x[j, t] >= 1 - self.LLP.delta[j, 1]
                )

                # (con-1) x[j,t] <= y[c,t-1] + delta^*_j  for each c in C_j
                for c in self.C_i[j]:
                    self.LLP.constrs_stability.add(
                        self.LLP.x[j, t]
                        <= self.LLP.y[j, c, t_prev] + self.LLP.delta[j, 2]
                    )

                # (con-2) x[j,t] >= 1 - sum_c (1-y[c,t-1]) - delta^*_j
                self.LLP.constrs_stability.add(
                    self.LLP.x[j, t]
                    >= 1
                    - pmoenv.quicksum(
                        1 - self.LLP.y[j, c, t_prev] for c in self.C_i[j]
                    )
                    - self.LLP.delta[j, 2]
                )

        # (uncon-1), (uncon-2) for uncontrollable genes i in J_c
        for i in self.J_c:
            for t in self.T_range:
                t_prev = t - 1

                # (uncon-1) x[i,t] <= y[c,t-1]  for each c in C_i
                for c in self.C_i[i]:
                    self.LLP.constrs_stability.add(
                        self.LLP.x[i, t] <= self.LLP.y[i, c, t_prev]
                    )

                # (uncon-2) x[i,t] >= 1 - sum_c (1-y[c,t-1])
                self.LLP.constrs_stability.add(
                    self.LLP.x[i, t]
                    >= 1
                    - pmoenv.quicksum(
                        1 - self.LLP.y[i, c, t_prev] for c in self.C_i[i]
                    )
                )

        # (lit-1), (lit-2), (lit-3) clause-literal synchronisation for t in T_range
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
                        self.LLP.y[i, c, t] >= 1 - self.LLP.x[i_, t]
                    )
                self.LLP.constrs_stability.add(
                    self.LLP.y[i, c, t] <= sum(x_lit_list)
                )

        # Placeholder required by MibSAttractorControl (minimality cuts go here)
        self.constrs_no_good_x = pmoenv.ConstraintList()

        # LLP objective: minimise p  (leader maximises this worst-case value)
        self.LLP.obj = pmoenv.Objective(expr=self.LLP.p, sense=pmoenv.minimize)

    def not_allow_empty_attractor(self):
        """No-op: the interdiction LLP always selects an attractor via sum(w)=1."""
        pass

    def optimize(self):
        return _run_mibs(self, self.MIBS_INTERDICTION_OPTIONS)


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

        if _config.use_interdiction:
            _model_cls = MibSInterdictBilevelIP
        elif _config.use_aggregated_LLP:
            _model_cls = MibSAggBilevelIP
        else:
            _model_cls = MibSBilevelIP
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
            self.step = EnumBendersStep.FULL_BILEVEL
            _solution_list = list()
            self.model_bilevel.set_constr_target_size(self.target_size)
            while not self.is_timeout and self._optimize(self.model_bilevel) and self.is_phenotype_satisfied():
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

    def is_phenotype_satisfied(self):
        # In the interdiction model, the LLP objective is exactly p, so we can read p directly. Otherwise, don't check
        if self._config.use_interdiction:
            return pmoenv.value(self.model_bilevel.LLP.p) > 0.5
        else:
            return True  
        
