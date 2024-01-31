from typing import Dict, List, Tuple
import boolean
from colomoto.minibn import _TRUE, _FALSE
from colomoto import minibn
from colomoto.types import Hypercube as _Hypercube

from optboolnet.config import ControlConfig
from algorecell_types import PermanentPerturbation


class ORClause(boolean.Expression):
    """

    Args:
        boolean (_type_): _description_
    """

    def __init__(self, args: boolean.Expression):
        super().__init__()
        self.args: Tuple[boolean.Expression] = args
        self.pos_literals: List[str] = list()
        self.neg_literals: List[str] = list()
        for expr in self.args:
            if isinstance(expr, boolean.NOT):
                self.neg_literals.append(str(expr.get_symbols()[0]))
            elif isinstance(expr, boolean.Expression):
                self.pos_literals.append(str(expr.get_symbols()[0]))
            else:
                raise Exception()

    def to_dict(self):
        return {
            str(expr.get_symbols()[0]): 0 if isinstance(expr, boolean.NOT) else 1
            for expr in self.args
        }  # TODO: if the function is constant?

    def __repr__(self) -> str:
        return self.args.__repr__()


class CNFBooleanNetwork(minibn.BooleanNetwork):
    def __init__(
        self,
        data,
        control_config: ControlConfig,
        Symbol_class=boolean.Symbol,
        allowed_in_name=(".", "_", ":", "-"),
        **kwargs,
    ):
        super().__init__(data, Symbol_class, allowed_in_name)

        _control_config = control_config
        self._control_config = _control_config
        self.vars_list = list(self.keys())
        self.controllable_vars = [
            var for var in _control_config.controllable_vars if var in self.vars_list
        ]
        self.uncontrollable_vars = [
            var for var in _control_config.uncontrollable_vars if var in self.vars_list
        ]

        self.fixed_values = _control_config.fixed_values
        self.phenotype = _control_config.phenotype

        if (self.phenotype not in self) and ("phenotype_formula" in kwargs):
            self[self.phenotype] = kwargs.pop("phenotype_formula")
        for var_name, value in self.fixed_values.items():
            self[var_name] = value
        # TODO: assert when this formula is not a CNF
        # TODO: convert to CNF if needed
        # TODO: check if union of controllable/uncontrollable vars covers the whole vars
        # TODO: check if control is applied twice

        self.__clause_dict: Dict[str, List[ORClause]] = dict()
        for var_name, CNF_formula in self.items():
            if isinstance(CNF_formula, _FALSE):
                self.__clause_dict[var_name] = list()
            elif CNF_formula.isliteral or isinstance(
                CNF_formula, (boolean.OR, _TRUE)
            ):  # single clause
                self.__clause_dict[var_name] = [ORClause(CNF_formula.literals)]
            elif isinstance(CNF_formula, boolean.AND):  # multiple clauses
                self.__clause_dict[var_name] = [
                    ORClause(clause.literals) for clause in CNF_formula.args
                ]
            else:
                raise TypeError()

    def items(self) -> Tuple[str, boolean.Expression]:
        return super().items()

    def iter_clauses(self, vars_list: List[str] = list(), keyonly: bool = False):
        _vars_list = self.keys() if not vars_list else vars_list
        for var_name in _vars_list:
            for clause_idx, or_clause in enumerate(self.__clause_dict[var_name]):
                if keyonly:
                    yield (var_name, clause_idx)
                else:
                    yield (var_name, clause_idx), or_clause

    def items_clause(self, var_name: str):
        return self.__clause_dict[var_name]

    def get_clause(self, var_name: str, clause_num: int):
        return self.__clause_dict[var_name][clause_num]

    def get_clause_idx_dict(self):
        return {
            i: [idx for idx, _ in enumerate(self.__clause_dict[i])] for i in self.keys()
        }

    def get_summary(self):
        return {
            "num_vars": len(self),
            "num_controllable_vars": len(self.controllable_vars),
            "num_clauses": len(list(self.iter_clauses(keyonly=True))),
        }

    def to_bnet(self, sort: bool = True):
        line_list = [
            f"{var_name}, {self.ba.cnf(self.ba.NOT(transition_formula))}"
            for var_name, transition_formula in self.items()
        ]
        line_list = sorted(line_list) if sort else line_list
        return "\n".join(line_list)

    def to_neg_CNF(self, sort: bool = True):
        line_list = [
            f"{var_name}, {self.ba.cnf(self.ba.NOT(transition_formula))}"
            for var_name, transition_formula in self.items()
        ]
        line_list = sorted(line_list) if sort else line_list
        return CNFBooleanNetwork("\n".join(line_list), self._control_config)


class Attractor:
    def __init__(
        self,
        bn: CNFBooleanNetwork,
        value_list: List[List[int]],
        first_state: List[int],
        alpha: List[bool],
        beta: List[bool],
    ) -> None:
        self.bn = bn
        # TODO: lazy operation of these components
        self.first_state = first_state
        self.value_list = value_list
        self.alpha = alpha
        self.beta = beta

    def get_first_state(self):
        return self.first_state

    def iter_states(self):
        for t, state in enumerate(self.value_list):
            yield t, state

    def to_str_list(self):
        return ["".join([str(value) for value in state]) for state in self.value_list]


class Hypercube(_Hypercube):
    def unfixed_vars(self, vars_list: List[str]):
        return set(vars_list) - self.keys()


class Control(Hypercube, PermanentPerturbation):
    def __init__(self, *args, **kwargs):
        super(Hypercube, self).__init__(*args, **kwargs)
        super(PermanentPerturbation, self).__init__(*args, **kwargs)

    def __hash__(self) -> int:
        return super(PermanentPerturbation).__hash__()

    def unfixed_vars(self, vars_list: List[str]):
        return super().unfixed_vars(vars_list)
