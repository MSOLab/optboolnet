from __future__ import annotations
from typing import Dict, Generic, List, Optional, TypeVar
import json

_DEFAULT_SOLVER = "gurobi_persistent"
_C = TypeVar("_C", bound="AttractorControlConfig")


class Config:
    """The base class for environment settings and control parameters.
    By default, an instance is constructed from a dict object.
    One can be also instantiated by the class method instantiate"""

    subclasses: list[type[Config]] = []

    __hidden__: List[str] = list()
    enforce: bool = True
    """If true, any inconsistent values will be fixed as the developer's default values"""

    def __init__(self, **kwargs) -> None:
        for _key, _value in kwargs.items():
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
    def from_dict(cls, data: Dict):
        kwargs = dict()
        data.pop("__name__", None)
        for _key, _value in data.items():
            if isinstance(_value, dict) and "__name__" in _value:
                for _cls in Config.subclasses:
                    if _cls.__name__ == _value["__name__"]:
                        _value.pop("__name__")
                        kwargs[_key] = _cls.from_dict(_value)
                        break
            else:
                kwargs[_key] = _value
        return cls(**kwargs)

    @classmethod
    def from_json(cls, fname: str):
        """Converts a json file into the config

        Args:
            fname (str): the path of a json file

        Returns:
            _type_: an instance
        """
        with open(fname, "r") as _f:
            return cls.from_dict(json.load(_f))

    # @classmethod
    # def instantiate(cls, _data: Union[str, dict, "Config"]):
    #     """Instantiates the data as an object

    #     Args:
    #         _data (type[T]): str (the path of a file), dict, or Config

    #     Returns:
    #         _type_: a Config object
    #     """
    #     if isinstance(_data, str):
    #         return cls.from_json(_data)
    #     elif isinstance(_data, dict):
    #         return cls.from_dict(_data)
    #     elif isinstance(_data, cls):
    #         return _data
    #     else:
    #         raise TypeError(f"Config {cls} does not support type {type(_data)}")

    def to_dict(self):
        _config_dict = dict()
        _config_dict["__name__"] = self.__class__.__name__
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
    to_stream: bool = False
    """The list of controllable variables"""
    fpath: str = ""
    """The list of uncontrollable variables"""
    fname: str = ""


class SolverConfig(Config):
    __hidden__ = ["options"]
    """The configuration for a pyomo solver"""

    solver_name: str = _DEFAULT_SOLVER
    """The name of a model"""

    # default options when calling 'solve' of pyomo solver
    save_results: bool = False
    """If true, pyomo generates a report in solver.results (not recommended for performance)"""
    tee: bool = False
    """If true, the solver's progress is shown"""
    warmstart: bool = False
    """If true, warmstart is given"""

    ### Additional options
    threads: Optional[int] = None

    time_limit: Optional[float] = None
    """Time limit given to the solver"""

    mip_display: bool
    """(for cplex) If true, unwanted logs may be ignored"""

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
    max_control_size: int = 0
    max_length: int = 1
    allow_empty_attractor: bool = True
    total_time_limit: Optional[float] = 600


class BendersConfig(AttractorControlConfig):
    solve_separation: bool = False
    preprocess_max_forbidden_trap_space: bool = False
    separation_heuristic: bool = False
    use_high_point_relaxation: bool = False


class MibSBilevelConfig(AttractorControlConfig):
    use_valid_cuts: bool
    solver_config: SolverConfig

    def __init__(self, data: Dict) -> None:
        super().__init__(data)
        assert (
            self.solver_config.time_limit == None
        ), "MibS does not allow time limit in SolverConfig"


class TotalConfig(Config, Generic[_C]):
    alg_config: _C
    master_solver_config: SolverConfig
    LLP_solver_config: SolverConfig
    separation_solver_config: SolverConfig
    logging_config: LoggingConfig
