from typing import Dict, List, Optional, Set, Tuple
import boolean
from colomoto.minibn import _TRUE, _FALSE
from colomoto import minibn
from colomoto.types import Hypercube as _Hypercube

from optboolnet.config import ControlConfig
from algorecell_types import PermanentPerturbation


def contains_and(expr) -> bool:
    """
    Recursively check whether `expr` or any sub-expression is an AND.
    """
    # is this node itself an AND?
    if isinstance(expr, boolean.AND):
        return True
    # otherwise, recurse into children
    for arg in expr.args:
        if contains_and(arg):
            return True
    return False


def simplify_cnf(ba, f):
    """
    Simplify a CNF formula by detecting suspect literals (variables that
    appear both positive and negative across clauses) and roundtripping
    through DNF→CNF. Analogous to ``simplify_dnf`` in ``colomoto.minibn``.
    """
    def is_wellformed_cnf(f):
        pos, neg = set(), set()
        def is_lit(f):
            if isinstance(f, ba.Symbol):
                pos.add(f.obj)
                return True
            elif isinstance(f, ba.NOT) and isinstance(f.args[0], ba.Symbol):
                neg.add(f.args[0].obj)
                return True
            return False

        def is_clause(f):
            if is_lit(f):
                return True
            if isinstance(f, ba.OR):
                for g in f.args:
                    if not is_lit(g):
                        return False
                return True
            return False

        if f is ba.TRUE or f is ba.FALSE:
            return True, set()
        if is_clause(f):
            return True, pos.intersection(neg)
        if isinstance(f, ba.AND):
            for g in f.args:
                if not is_clause(g):
                    return False, None
            return True, pos.intersection(neg)
        return False, None

    is_cnf, suspects = is_wellformed_cnf(f)
    if is_cnf and suspects:
        return ba.cnf(ba.dnf(f))
    return f


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

    PHENOTYPE_VAR: str = "__PHENOTYPE__"

    def __init__(
        self,
        data,
        control_config: ControlConfig,
        Symbol_class=boolean.Symbol,
        allowed_in_name=(".", "_", ":", "-"),
        to_cnf: bool = False,
        simplify: bool = False,
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
        assert set(self.controllable_vars).union(set(self.uncontrollable_vars)) == set(
            self.vars_list
        )

        self.__bn_cnf: Dict[str, boolean.Expression] = dict()
        self.__clause_dict: Dict[str, List[ORClause]] = dict()
        for var_name, formula in self.items():
            cnf_formula = self.ba.cnf(formula) if to_cnf else formula
            if simplify:
                cnf_formula = simplify_cnf(self.ba, cnf_formula)
            self.__bn_cnf[var_name] = cnf_formula
            self.__clause_dict[var_name] = self._parse_cnf_to_clauses(
                cnf_formula, var_name
            )

        # Hybrid encoding state (populated by compute_hybrid_partition)
        self.__neg_clause_dict: Optional[Dict[str, List[ORClause]]] = None
        self.__cnf_genes: Optional[Set[str]] = None
        self.__dnf_genes: Optional[Set[str]] = None

    @staticmethod
    def _parse_cnf_to_clauses(
        cnf_formula: boolean.Expression, var_name: str
    ) -> List[ORClause]:
        if isinstance(cnf_formula, _FALSE):
            return list()
        elif cnf_formula.isliteral or isinstance(
            cnf_formula, (boolean.OR, _TRUE)
        ):  # single clause
            assert not contains_and(
                cnf_formula
            ), f"{var_name}, {cnf_formula} is not a CNF"
            return [ORClause(cnf_formula.literals)]
        elif isinstance(cnf_formula, boolean.AND):  # multiple clauses
            return [ORClause(clause.literals) for clause in cnf_formula.args]
        else:
            raise TypeError()

    def compute_hybrid_partition(self, dnf_genes: Optional[Set[str]] = None):
        """Compute double CNF and partition genes into CNF/DNF sets.

        Args:
            dnf_genes: If provided, force these genes to use DNF encoding.
                All other genes use CNF. If None, uses the heuristic
                (gene i in CNF if |C^1_i| <= |C^0_i|).
        """
        self.__neg_clause_dict = {}
        self.__cnf_genes = set()
        self.__dnf_genes = set()
        for var_name, formula in self.__bn_cnf.items():
            neg_cnf = self.ba.cnf(self.ba.NOT(formula))
            neg_cnf = simplify_cnf(self.ba, neg_cnf)
            self.__neg_clause_dict[var_name] = self._parse_cnf_to_clauses(
                neg_cnf, var_name
            )
            if dnf_genes is not None:
                if var_name in dnf_genes:
                    self.__dnf_genes.add(var_name)
                else:
                    self.__cnf_genes.add(var_name)
            else:
                if len(self.__clause_dict[var_name]) <= len(
                    self.__neg_clause_dict[var_name]
                ):
                    self.__cnf_genes.add(var_name)
                else:
                    self.__dnf_genes.add(var_name)

    @property
    def is_hybrid_enabled(self) -> bool:
        return self.__cnf_genes is not None

    def is_cnf_gene(self, gene: str) -> bool:
        return gene in self.__cnf_genes

    def is_dnf_gene(self, gene: str) -> bool:
        return gene in self.__dnf_genes

    @property
    def cnf_genes(self) -> List[str]:
        return [v for v in self.keys() if v in self.__cnf_genes]

    @property
    def dnf_genes(self) -> List[str]:
        return [v for v in self.keys() if v in self.__dnf_genes]

    def items_neg_clause(self, var_name: str) -> List[ORClause]:
        return self.__neg_clause_dict[var_name]

    def get_neg_clause_idx_dict(self) -> Dict[str, List[int]]:
        return {
            i: [idx for idx, _ in enumerate(self.__neg_clause_dict[i])]
            for i in self.keys()
        }

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

    def items_cnf(self):
        return self.__bn_cnf.items()

    def get_cnf(self, var_name: str) -> boolean.Expression:
        return self.__bn_cnf[var_name]

    def to_bnet(self, sort: bool = False):
        line_list = [
            f"{var_name}, {cnf_formula}"
            for var_name, cnf_formula in self.__bn_cnf.items()
        ]
        line_list = sorted(line_list) if sort else line_list
        return "\n".join(line_list)

    def to_neg_CNF(self, sort: bool = True):
        line_list = [
            f"{var_name}, {self.ba.cnf(self.ba.NOT(cnf_formula))}"
            for var_name, cnf_formula in self.__bn_cnf.items()
        ]
        line_list = sorted(line_list) if sort else line_list
        return CNFBooleanNetwork("\n".join(line_list), self._control_config)

    @staticmethod
    def from_bnet(
        bn: minibn.BooleanNetwork,
        inputs: dict = dict(),
        target: dict = dict(),
        exclude: list = list(),
        simplify: bool = False,
    ):
        new_bn = minibn.BooleanNetwork(bn)
        config = ControlConfig()
        config.fixed_values = inputs.copy()
        assert len(target) > 0
        config.uncontrollable_vars = list(set(inputs.keys()).union(set(exclude)))
        config.uncontrollable_vars.append(CNFBooleanNetwork.PHENOTYPE_VAR)
        config.controllable_vars = list(set(new_bn.keys()).difference(set(config.uncontrollable_vars)))

        config.phenotype = CNFBooleanNetwork.PHENOTYPE_VAR
        cnf_clauses = []
        for var, value in target.items():
            if value == 1:
                cnf_clauses.append(f"{var}")
            else:
                cnf_clauses.append(f"!{var}")
        cnf_formula = " & ".join(cnf_clauses)
        new_bn[config.phenotype] = cnf_formula
        return CNFBooleanNetwork(new_bn, config, to_cnf=True, simplify=simplify)


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
        Hypercube.__init__(self, *args, **kwargs)
        PermanentPerturbation.__init__(self, *args, **kwargs)

    def __hash__(self) -> int:
        return hash(repr(self))

    def unfixed_vars(self, vars_list: List[str]):
        return super().unfixed_vars(vars_list)
