import time
from typing import Dict, List, Callable, Optional, Union
from optboolnet import CNFBooleanNetwork, Control
from optboolnet.boolnet import Attractor
from optboolnet.model import (
    Model,
    CoreIP,
    AttractorDetectionIP,
    ExtendedAttractorDetectionIP,
    MasterControlIP,
    TrapSpaceDetectionIP,
)
from optboolnet.config import (
    AttractorControlConfig,
    BendersConfig,
    LoggingConfig,
    SolverConfig,
)
from optboolnet.log import EnumBendersStep, BendersLogger


class AttractorControl:
    _config: AttractorControlConfig

    def __init__(
        self, name: str, bn: CNFBooleanNetwork, logging_config: LoggingConfig
    ) -> None:
        self.name = name
        self.bn = bn
        self.step = EnumBendersStep.BUILD_MODEL
        self.target_size: int = 0
        self.solution_dict: Dict[int, List[Control]] = dict()
        """(key) core size (value) list of minimal controls"""
        self.start_time = time.time()
        self.logger = BendersLogger(logging_config)

    @property
    def elapsed_time(self):
        return time.time() - self.start_time

    @property
    def log_signature(self) -> str:
        return (
            f"{self.elapsed_time:.3f},{self.name},{self.target_size},{self.step.name}"
        )

    @property
    def is_timeout(self) -> bool:
        if self._config.total_time_limit is None:
            return False
        else:
            return self.elapsed_time > self._config.total_time_limit

    @property
    def remaining_time(self) -> Optional[float]:
        if self._config.total_time_limit == None:
            return None
        else:
            return self._config.total_time_limit - self.elapsed_time

    def iter_target_size(self):
        for target_size in range(self._config.max_control_size + 1):
            yield target_size

    @BendersLogger.wrap_model_build
    def build_model(self, cls: type[Model], *args):
        return cls(*args)

    @BendersLogger.wrap_model_optimize
    def _optimize(self, problem: CoreIP):
        """Solves a problem (an auxiliary method for logging)

        Args:
            problem (CoreIP): A problem to solve
        """
        _time_limit = min(
            [
                tl
                for tl in [self.remaining_time, problem.solver_config.time_limit]
                if tl != None
            ],
            default=None,
        )
        # update_options_time_limit
        if (_time_limit != None):
            if (_time_limit != None) and (_time_limit < 0):
                return False
            else:
                problem.update_options_time_limit(_time_limit)
        return problem.optimize()

    @BendersLogger.wrap_cut
    def _append_cut(self, func: Callable, *args):
        """Appends a cut (an auxiliary method for logging)

        Args:
            func (Callable): a method of CoreIP that returns Tuple[BendersCutType,int]
        """
        return func(*args)


class BendersAttractorControl(AttractorControl):
    def __init__(
        self,
        name: str,
        bn: CNFBooleanNetwork,
        _config: Union[str, dict, BendersConfig],
    ) -> None:
        super().__init__(name, bn, _config.logging_config)
        self._config: BendersConfig = BendersConfig.instantiate(_config)

        if self._config.use_high_point_relaxation:
            self.model_master = self.build_model(
                AttractorDetectionIP,
                "0",
                self.bn,
                1,
                self._config.master_solver_config,
            )
            self.model_master.make_constr_phenotype_at_all_t()
            self.model_master.fix_phenotype(1)
        else:
            self.model_master = self.build_model(
                MasterControlIP,
                "0",
                self.bn,
                self._config.master_solver_config,
            )
        if self._config.solve_separation:
            self.model_separation = self.build_model(
                TrapSpaceDetectionIP,
                f"0",
                self.bn,
                self._config.separation_solver_config,
            )
            self.model_separation.fix_phenotype(0)
            self.model_separation.set_objective_sparse_cut()
        else:
            self.model_separation = None
        self.model_LLP_list: List[ExtendedAttractorDetectionIP] = list()
        for length in range(1, self._config.max_length + 1):
            model_LLP = self.build_model(
                ExtendedAttractorDetectionIP,
                f"{length}",
                self.bn,
                length,
                self._config.LLP_solver_config,
            )
            model_LLP.fix_var(model_LLP.v, 0)
            model_LLP.make_constr_stability_condition()
            model_LLP.make_constr_phenotype_at_all_t()
            model_LLP.set_phenotype_obj()
            self.model_LLP_list.append(model_LLP)

    def run_exhaustive_search(self):
        """Finds all minimal controls that induces the given phenotype at all states of every attractor

        Returns:
            _type_: _description_
        """
        # TODO: merge with a new parent class
        # preprocessing
        if self._config.preprocess_max_forbidden_trap_space:
            self.add_all_max_forbidden_trap_space_cuts()
        # main step
        for self.target_size in self.iter_target_size():
            _solution_list = list()
            self.model_master.set_constr_target_size(self.target_size)
            while not self.is_timeout and self.find_candidate():
                ctrl = self.model_master.get_control()
                if not self.is_separation_violated(ctrl) and not self.is_LLP_violated(
                    ctrl
                ):
                    _solution_list.append(ctrl)
                    self._append_cut(self.model_master.append_minimality_cut, ctrl)
            self.solution_dict[self.target_size] = _solution_list
            if self.is_timeout:
                break
            self.step = EnumBendersStep.FINISHED
            self.logger.solve_logger_info(self.log_signature)
        self.logger.write_controls_to_json(self.solution_dict)
        return self.solution_dict

    def find_candidate(self) -> bool:
        """Solves the master problem to find a solution candidate

        Returns:
            bool: _description_
        """
        self.step = EnumBendersStep.BENDERS_MASTER
        return self._optimize(self.model_master)

    def is_separation_violated(self, ctrl: Control) -> bool:
        """Finds a forbidden trap space and adds a constraint that cuts off the candidate if one exists

        Args:
            ctrl (Control): the control candidate discovered by a master

        Returns:
            bool: whether a forbidden trap space is found
        """

        if not self._config.solve_separation:
            return False
        self.step = EnumBendersStep.SEPARATION_PROBLEM

        # applying the heuristic may save the time but weaken the cut
        if self._config.separation_heuristic:
            self.model_separation.fix_control(ctrl)
        else:
            self.model_separation.add_constr_separation(ctrl)

        # solve the separation closure and possibly add a forbidden trap space cut
        if self._optimize(self.model_separation):
            forbidden_ctrl = self.model_separation.get_control()
            forbidden_ts = self.model_separation.get_trap_space()
            self._append_cut(
                self.model_master.append_forbidden_trap_space_cut,
                forbidden_ctrl,
                forbidden_ts,
            )
            return True
        else:
            return False

    def is_LLP_violated(self, ctrl: Control) -> bool:
        """Finds a forbidden attractor and adds a constraint that cuts off the candidate if one exists

        Args:
            ctrl (Control): the control candidate discovered by a master

        Returns:
            bool: True if a forbidden attractor is found
        """

        self.step = EnumBendersStep.LOWER_LEVEL_PROBLEM
        is_feasible = False
        for LLP_model in self.model_LLP_list:
            LLP_model.fix_control(ctrl)
            if self._optimize(LLP_model):
                is_feasible = True
                if LLP_model.p.value == 0:
                    attr = LLP_model.get_attractor()
                    self._append_cut(self.model_master.append_logical_benders_cut, attr)
                    return True
        if is_feasible or self._config.allow_empty_attractor:
            return False
        else:
            self._append_cut(self.model_master.append_no_good_cut_d, ctrl)
            return True

    def add_all_max_forbidden_trap_space_cuts(self):
        """Preprocess maximal forbidden trap spaces and add cuts to the master problem"""
        if not self._config.solve_separation:
            return
        self.step = EnumBendersStep.FORBIDDEN_TRAP_SPACE_CUT
        self.model_separation.set_constr_target_size(0)
        while self._optimize(self.model_separation):
            ctrl = self.model_separation.get_control()
            ts = self.model_separation.get_trap_space()
            self.model_separation.add_trap_space_maximality_cut(ctrl, ts)
            self._append_cut(
                self.model_master.append_forbidden_trap_space_cut, ctrl, ts
            )
        self.model_separation.set_constr_target_size(None)
        self.model_separation.clear_constr_list(self.model_separation.constrs_benders)

    @property
    def solution_count(self) -> int:
        return sum(len(_sol_list) for _sol_list in self.solution_dict.values())


class BendersFixPointControl(BendersAttractorControl):
    _config: BendersConfig

    def __init__(
        self, name: str, bn: CNFBooleanNetwork, _config: BendersConfig
    ) -> None:
        _config = BendersConfig.instantiate(_config)
        _config.check_fixed_point_configs()
        super().__init__(name, bn, _config)


def enumerate_attractors(
    bn: CNFBooleanNetwork,
    max_length: int,
    ctrl: Control,
    solver_config: SolverConfig,
):
    attractor_list: List[Attractor] = list()
    for length in range(1, 1 + max_length):
        print(length)
        attr_ip = AttractorDetectionIP(f"{length}", bn, length, solver_config)
        attr_ip.make_constr_stability_condition()
        attr_ip.make_constr_phenotype_at_all_t()
        attr_ip.set_phenotype_obj()
        for attractor in attractor_list:
            attr_ip.add_no_good_x(attractor)
        attr_ip.fix_control(ctrl)
        while attr_ip.optimize():
            attractor = attr_ip.get_attractor()
            yield attractor
            attractor_list += [attractor]
            attr_ip.add_no_good_x(attractor)
