from typing import Any, Dict, List, Optional, TypeVar, Union
from optboolnet.exception import InvalidConfigError, InvalidConfigWarning
import json, warnings


class Config:
    """The base class for environment settings and control parameters.
    By default, an instance is constructed from a dict object.
    One can be also instantiated by the class method instantiate"""

    subclasses = []

    __hidden__: List[str] = list()
    enforce: bool = True
    """If true, any incosistent values will be fixed as the developer's default values"""

    def __init__(self, data: Dict) -> None:
        for _key, _value in data.items():
            if isinstance(_value, dict) and "__name__" in _value:
                for _cls in Config.subclasses:
                    if _cls.__name__ == _value["__name__"]:
                        self.__setattr__(_key, _cls(_value))
                        break
            else:
                self.__setattr__(_key, _value)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses.append(cls)

    def force_argument(self, data: Dict):
        for _key, _value in data.items():
            if _key in self.__dict__:
                assert _value == self.__getattribute__(_key)
            self.__setattr__(_key, _value)

    @classmethod
    def from_json(cls, fname: str):
        """Converts a json file into the config

        Args:
            fname (str): the path of a json file

        Returns:
            _type_: an instance
        """
        with open(fname, "r") as _f:
            return cls(json.load(_f))

    @classmethod
    def instantiate(cls, _data: Union[str, dict, "Config"]):
        """Instantiates the data as an object

        Args:
            _data (type[T]): str (the path of a file), dict, or Config

        Returns:
            _type_: a Config object
        """
        if isinstance(_data, str):
            return cls.from_json(_data)
        elif isinstance(_data, dict):
            return cls(_data)
        elif isinstance(_data, cls):
            return _data
        else:
            raise TypeError(f"Config {cls} does not support type {type(_data)}")

    def to_dict(self):
        _config_dict = dict()
        _config_dict["__name__"] = self.__name__
        for _key, _value in self.__dict__.items():
            if _key in self.__hidden__:
                continue
            if isinstance(_value, Config):
                _config_dict[_key] = _value.to_dict()
            else:
                _config_dict[_key] = _value

        return _config_dict


class ControlConfig(Config):
    controllable_vars: List[str]
    """The list of controllable variables"""
    uncontrollable_vars: List[str]
    """The list of uncontrollable variables"""
    fixed_values: Dict[str, int]
    """(key) the name of a variable (value) a value to fix (0 or 1)"""
    phenotype: str
    """The variable that determines the phenotype"""


class LoggingConfig(Config):
    to_stream: bool
    """The list of controllable variables"""
    fpath: str
    """The list of uncontrollable variables"""
    fname: str


class SolverConfig(Config):
    __hidden__ = ["options"]
    """The configuration for a pyomo solver"""

    solver_name: str
    """The name of a model"""

    # default options when calling 'solve' of pyomo solver
    save_results: bool
    """If true, pyomo generates a report in solver.results (not recommended for performance)"""
    tee: bool
    """If true, the solver's progress is shown"""
    warmstart: bool
    """If true, warmstart is given"""

    ### Additional options
    threads: Optional[int] = None

    time_limit: Optional[float]
    """Time limit given to the solver"""

    mip_display: bool
    """(for cplex) If true, unwanted logs may be ignored"""

    def __init__(self, data: Dict) -> None:
        super().__init__(data)

    @property
    def options(self):
        """The dictionary for additional options (auto-generated)
        The keys with None values will be ignored
        """
        return {
            _key: self.__getattribute__(_key)
            for _key in (
                self.__dict__.keys()
                - set(("solver_name", "save_results", "tee", "warmstart"))
            )
            if self.__getattribute__(_key) != None
        }

    @property
    def kwgs(self):
        return {
            "save_results": self.save_results,
            "tee": self.tee,
            "warmstart": self.warmstart,
        }


class SolverMibSConfig(SolverConfig):
    executable: str


class AttractorControlConfig(Config):
    max_control_size: int
    """The upper limit to the size of controls"""
    max_length: int
    """The upper limit of the length of attractors to discover"""

    allow_empty_attractor: bool
    """If true, a no good cut removes the controls that induces no attractor
    Otherwise, an Exception is raised"""

    total_time_limit: Optional[float]
    logging_config: LoggingConfig


class BendersConfig(AttractorControlConfig):  # TODO: inherit AttractorControlConfig
    solve_separation: bool
    """If true, a separation problem is solved to boost the performance.
    Only valid when the max_length is sufficiently large so that every attractor can be found
    by one of the lower level problems."""
    preprocess_max_forbidden_trap_space: bool
    """If true, all maximal forbidden trap spaces are found and add the cuts removing them.
    This can boost performance if max_control_size is sufficiently large"""
    separation_heuristic: bool
    """If true, many variables in the separation problem are fixed. 
    This may slightly reduce the computation time for separation problems.
    However, seperation may fail or the resulting constraints may be weak."""

    use_high_point_relaxation: bool

    ### Config objects for solvers
    master_solver_config: SolverConfig
    LLP_solver_config: SolverConfig
    separation_solver_config: SolverConfig

    def __init__(self, data: Dict) -> None:
        super().__init__(data)

        # validation
        if (self.max_length != 1) and self.use_high_point_relaxation:
            raise InvalidConfigError(
                "The high point relaxation can be only used for is when the length of attractor is 1"
            )
        if (not self.solve_separation) and (
            self.preprocess_max_forbidden_trap_space or self.separation_heuristic
        ):
            raise InvalidConfigError(
                "preprocess_max_forbidden_trap_space and separation_heuristic can be used only when solve_separation is on"
            )

    def check_fixed_point_configs(self):
        _message = """The following settings for the fixed point control problem are enforced:
                \t self.max_length == 1
                \t self.preprocess_max_forbidden_trap_space == False
                \t self.solve_separation == False
                \t self.separation_heuristic == False
                """

        if not (
            (self.max_length == 1)
            and (self.preprocess_max_forbidden_trap_space == False)
            and (self.solve_separation == False)
            and (self.separation_heuristic == False)
        ):
            if self.enforce:
                warnings.warn(InvalidConfigWarning(_message))
                self.max_length = 1
                self.preprocess_max_forbidden_trap_space = False
                self.solve_separation = False
                self.separation_heuristic = False
            else:
                raise InvalidConfigError(_message)


class MibSBilevelConfig(AttractorControlConfig):
    use_valid_cuts: bool
    solver_config: SolverConfig

    def __init__(self, data: Dict) -> None:
        super().__init__(data)
        assert (
            self.solver_config.time_limit == None
        ), "MibS does not allow time limit in SolverConfig"
