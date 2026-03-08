from __future__ import annotations

import logging
import os
import re
import tempfile
import time
from functools import lru_cache
from itertools import combinations, product
from typing import Callable, Dict, List, Optional, Tuple

from optboolnet.boolnet import CNFBooleanNetwork, Control
from optboolnet.config import LoggingConfig
from optboolnet.log import BendersLogger, EnumCutType

_PYBOOLNET_IMPORT_ERROR: Optional[Exception] = None
try:
    from pyboolnet.attractors import completeness
    from pyboolnet.file_exchange import bnet2primes
    from pyboolnet.helpers import dicts_are_consistent
    from pyboolnet.model_checking import model_checking
    from pyboolnet.prime_implicants import (
        find_constants,
        find_inputs,
        percolate,
        remove_variables,
    )
    from pyboolnet.temporal_logic import subspace2proposition
    from pyboolnet.trap_spaces import compute_trap_spaces
except Exception as exc:  # pragma: no cover - import guard only
    _PYBOOLNET_IMPORT_ERROR = exc

log = logging.getLogger(__name__)

# NuSMV reserved keywords; if a BN variable collides, PyBoolNet model_checking
# fails while generating SMV. Keep this local to avoid importing checking.py.
_NUSMV_RESERVED_LOWER = frozenset(
    x.lower()
    for x in [
        "MODULE", "DEFINE", "MDEFINE", "CONSTANTS", "VAR", "IVAR", "FROZENVAR",
        "ASSIGN", "TRANS", "INIT", "INVAR", "SPEC", "CTLSPEC", "LTLSPEC",
        "PSLSPEC", "COMPUTE", "INVARSPEC", "FAIRNESS", "JUSTICE", "COMPASSION",
        "ISA", "CONSTRAINT", "SIMPWFF", "CTLWFF", "LTLWFF", "PSLWFF", "COMPWFF",
        "MAX", "MIN",
        "IN", "UNION",
        "BOOLEAN", "INTEGER", "REAL", "WORD", "WORD1", "BOOL",
        "SIGNED", "UNSIGNED", "ARRAY", "OF",
        "COUNT", "EXTEND", "RESIZE", "SIZEOF", "TOINT", "SWCONST",
        "CASE", "ESAC",
        "NEXT", "SELF", "PROCESS",
        "TRUE", "FALSE",
        "EBF", "EBG", "ABF", "ABG",
        "mod", "union", "in", "xor", "xnor", "case", "esac", "next", "init",
        "process", "array", "of", "boolean", "integer", "real", "word", "self",
        "count", "extend", "resize", "sizeof", "toint", "signed", "unsigned",
    ]
)

_SAFE_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def _require_pyboolnet() -> None:
    if _PYBOOLNET_IMPORT_ERROR is not None:
        raise RuntimeError(
            "PyBoolNet is not available. Activate the conda environment with "
            "PyBoolNet installed (e.g. `buildtool`) and retry."
        ) from _PYBOOLNET_IMPORT_ERROR


def _subspace_key(subspace: Dict[str, int]) -> Tuple[Tuple[str, int], ...]:
    return tuple(sorted((k, int(v)) for k, v in subspace.items()))


def _normalize_subspace(subspace: Dict[str, int]) -> Dict[str, int]:
    return {k: int(v) for k, v in subspace.items()}


def is_included_in_subspace(subspace1: Dict[str, int], subspace2: Dict[str, int]) -> bool:
    return all(x in subspace1 and subspace1[x] == subspace2[x] for x in subspace2.keys())


def efag_set_of_subspaces(primes: dict, subspaces: List[Dict[str, int]]) -> str:
    return "EF(AG(" + " | ".join(subspace2proposition(primes, x) for x in subspaces) + "))"


def _to_nusmv_safe_name(name: str, used: set[str]) -> str:
    candidate = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not candidate or not candidate[0].isalpha():
        candidate = f"v_{candidate}"
    if len(candidate) < 2:
        candidate = f"v_{candidate}"
    if candidate.lower() in _NUSMV_RESERVED_LOWER:
        candidate = f"v_{candidate}"
    base = candidate
    suffix = 1
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


@lru_cache(maxsize=4096)
def _cached_rename_mapping(
    var_names: Tuple[str, ...]
) -> Tuple[Tuple[Tuple[str, str], ...], bool]:
    # Fast path: already safe, unique, and non-reserved.
    seen = set()
    safe_without_rename = True
    for var in var_names:
        if (
            len(var) < 2
            or (var.lower() in _NUSMV_RESERVED_LOWER)
            or (_SAFE_NAME_PATTERN.fullmatch(var) is None)
            or (var in seen)
        ):
            safe_without_rename = False
            break
        seen.add(var)
    if safe_without_rename:
        return tuple((var, var) for var in var_names), False

    rename_map: Dict[str, str] = {}
    used: set[str] = set()
    requires_rename = False
    for var in var_names:
        safe = _to_nusmv_safe_name(var, used)
        rename_map[var] = safe
        if safe != var:
            requires_rename = True
    return tuple(rename_map.items()), requires_rename


def _rename_primes_and_target_for_nusmv(
    primes: dict, target: List[Dict[str, int]]
) -> Tuple[dict, List[Dict[str, int]]]:
    # NuSMV used by PyBoolNet rejects 1-char variable names (e.g., "p").
    # Keep the original names everywhere else and only rename in model checking.
    mapping_items, requires_rename = _cached_rename_mapping(tuple(primes.keys()))
    rename_map = dict(mapping_items)

    if not requires_rename:
        return primes, target

    renamed_primes = {}
    for var, val in primes.items():
        zero_implicants, one_implicants = val
        renamed_zero = [{rename_map[k]: v for k, v in imp.items()} for imp in zero_implicants]
        renamed_one = [{rename_map[k]: v for k, v in imp.items()} for imp in one_implicants]
        renamed_primes[rename_map[var]] = [renamed_zero, renamed_one]

    renamed_target = [{rename_map[k]: v for k, v in sub.items()} for sub in target]
    return renamed_primes, renamed_target


def fix_components_and_reduce(
    primes: dict, subspace: Dict[str, int], keep_vars: List[str]
) -> dict:
    new_primes = percolate(primes, add_constants=subspace, copy=True)
    removable_vars = [k for k in find_constants(new_primes) if k not in keep_vars]
    return remove_variables(new_primes, removable_vars, copy=True)


def run_control_query(primes: dict, target: List[Dict[str, int]], update: str) -> bool:
    mc_primes, mc_target = _rename_primes_and_target_for_nusmv(primes, target)
    spec = "CTLSPEC " + efag_set_of_subspaces(mc_primes, mc_target)
    return bool(model_checking(mc_primes, update, "INIT TRUE", spec))


def reduce_and_run_control_query(
    primes: dict, subspace: Dict[str, int], target: List[Dict[str, int]], update: str
) -> bool:
    target_vars = list({item for subs in target for item in subs})
    new_primes = fix_components_and_reduce(primes, subspace, keep_vars=target_vars)
    return run_control_query(new_primes, target, update)


def control_is_valid_in_trap_spaces(
    primes: dict, trap_spaces: List[Dict[str, int]], target: List[Dict[str, int]], update: str
) -> bool:
    if not all(any(dicts_are_consistent(ts, subs) for subs in target) for ts in trap_spaces):
        return False

    half_ts = [ts for ts in trap_spaces if not any(is_included_in_subspace(ts, subs) for subs in target)]
    for ts in half_ts:
        if not reduce_and_run_control_query(primes, ts, target, update):
            return False
    return True


def control_direct_percolation(primes: dict, candidate: Dict[str, int], target: List[Dict[str, int]]) -> bool:
    perc = find_constants(primes=percolate(primes=primes, add_constants=candidate, copy=True))
    return any(is_included_in_subspace(perc, subs) for subs in target)


def control_model_checking(
    primes: dict,
    candidate: Dict[str, int],
    target: List[Dict[str, int]],
    update: str,
    max_output_trapspaces: int,
    perc: Optional[Dict[str, int]] = None,
) -> bool:
    if perc is None:
        perc = find_constants(primes=percolate(primes=primes, add_constants=candidate, copy=True))
    target_vars = list({item for subs in target for item in subs})
    new_primes = fix_components_and_reduce(primes, perc, keep_vars=target_vars)
    minimal_trap_spaces = compute_trap_spaces(new_primes, "min", max_output=max_output_trapspaces)

    if not control_is_valid_in_trap_spaces(new_primes, minimal_trap_spaces, target, update):
        return False
    return run_control_query(new_primes, target, update)


def control_completeness(
    primes: dict, candidate: Dict[str, int], target: Dict[str, int], update: str
) -> bool:
    perc = find_constants(primes=percolate(primes=primes, add_constants=candidate, copy=True))
    new_primes = fix_components_and_reduce(primes, perc, keep_vars=list(target.keys()))
    minimal_trap_spaces = compute_trap_spaces(new_primes, "min")
    if not all(is_included_in_subspace(ts, target) for ts in minimal_trap_spaces):
        return False
    return bool(completeness(new_primes, update))


def _percolated_constants(primes: dict, candidate: Dict[str, int]) -> Dict[str, int]:
    return find_constants(primes=percolate(primes=primes, add_constants=candidate, copy=True))


def _direct_percolation_from_constants(perc: Dict[str, int], target: List[Dict[str, int]]) -> bool:
    return any(is_included_in_subspace(perc, subs) for subs in target)


def find_necessary_interventions(primes: dict, target: List[Dict[str, int]]) -> Dict[str, int]:
    selected_vars: Dict[str, int] = {}
    candidates = find_inputs(primes) + list(find_constants(primes).keys())
    for var in candidates:
        if all(var in sub.keys() for sub in target):
            if all(sub_a[var] == sub_b[var] for sub_a in target for sub_b in target):
                selected_vars[var] = target[0][var]
    return selected_vars


def find_common_variables_in_control_strategies(
    primes: dict, target: List[Dict[str, int]]
) -> Dict[str, int]:
    common_inputs_and_constants = find_necessary_interventions(primes, target)
    constants = find_constants(primes)
    right_constants = [
        k
        for k in constants
        if k in common_inputs_and_constants and common_inputs_and_constants[k] == constants[k]
    ]
    return {
        k: common_inputs_and_constants[k]
        for k in common_inputs_and_constants
        if k not in right_constants
    }


def _timed_out(start_time: float, time_limit: Optional[float]) -> bool:
    if time_limit is None:
        return False
    return (time.time() - start_time) >= time_limit


def compute_control_strategies_with_model_checking(
    primes: dict,
    target: List[Dict[str, int]],
    update: str = "asynchronous",
    limit: int = 3,
    avoid_nodes: Optional[List[str]] = None,
    max_output_trapspaces: int = 1000000,
    starting_length: int = 0,
    known_strategies: Optional[List[Dict[str, int]]] = None,
    time_limit: Optional[float] = None,
    on_control_found: Optional[Callable[[Dict[str, int], int, str, float], None]] = None,
    on_size_finished: Optional[Callable[[int, bool], None]] = None,
) -> List[Dict[str, int]]:
    if not isinstance(target, list):
        raise TypeError("target must be a list of subspaces.")

    search_start = time.time()
    avoid_set = set(avoid_nodes or [])
    list_strategies = [_normalize_subspace(x) for x in (known_strategies or [])]

    perc_true_keys = set()
    for known in list_strategies:
        perc = _percolated_constants(primes, known)
        perc_true_keys.add(_subspace_key(perc))
    perc_false_keys = set()

    common_vars_in_cs = find_common_variables_in_control_strategies(primes, target)
    candidate_variables = [x for x in primes.keys() if x not in common_vars_in_cs and x not in avoid_set]
    log.info("Number of common variables in the CS: %d", len(common_vars_in_cs))
    log.info("Number of candidate variables: %d", len(candidate_variables))

    start_k = max(0, starting_length - len(common_vars_in_cs))
    end_k = limit + 1 - len(common_vars_in_cs)
    for i in range(start_k, end_k):
        if _timed_out(search_start, time_limit):
            log.warning("PyBoolNet control search reached time limit.")
            break
        target_size = i + len(common_vars_in_cs)
        log.info("Checking control strategies of size %d", target_size)
        completed_size = True
        for var_subset in combinations(candidate_variables, i):
            if _timed_out(search_start, time_limit):
                completed_size = False
                break
            for values in product((0, 1), repeat=i):
                if _timed_out(search_start, time_limit):
                    completed_size = False
                    break
                candidate = dict(zip(var_subset, values))
                candidate.update(common_vars_in_cs)

                if any(is_included_in_subspace(candidate, x) for x in list_strategies):
                    continue

                perc = _percolated_constants(primes, candidate)
                perc_key = _subspace_key(perc)
                check_st = time.time()

                if perc_key in perc_true_keys:
                    list_strategies.append(candidate)
                    if on_control_found is not None:
                        on_control_found(candidate, target_size, "CACHE_TRUE", time.time() - check_st)
                    continue
                if perc_key in perc_false_keys:
                    continue

                if _direct_percolation_from_constants(perc, target):
                    perc_true_keys.add(perc_key)
                    list_strategies.append(candidate)
                    if on_control_found is not None:
                        on_control_found(candidate, target_size, "PERCOLATION", time.time() - check_st)
                elif control_model_checking(
                    primes, candidate, target, update, max_output_trapspaces=max_output_trapspaces, perc=perc
                ):
                    perc_true_keys.add(perc_key)
                    list_strategies.append(candidate)
                    if on_control_found is not None:
                        on_control_found(candidate, target_size, "MODEL_CHECKING", time.time() - check_st)
                else:
                    perc_false_keys.add(perc_key)

        if on_size_finished is not None:
            on_size_finished(target_size, completed_size)
        if not completed_size:
            log.warning("PyBoolNet control search reached time limit.")
            break

    return list_strategies


def _bn_to_bnet_text(bn: CNFBooleanNetwork) -> str:
    return "\n".join(f"{var}, {formula}" for var, formula in bn.items())


def make_primes_from_bn(bn: CNFBooleanNetwork) -> dict:
    _require_pyboolnet()
    bnet_text = _bn_to_bnet_text(bn)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".bnet", delete=False, encoding="utf-8") as tmp:
        tmp.write(bnet_text)
        tmp_path = tmp.name
    try:
        return bnet2primes(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


class PyBoolNetAttractorControl:
    def __init__(
        self,
        name: str,
        bn: CNFBooleanNetwork,
        logging_config: LoggingConfig = LoggingConfig(),
    ) -> None:
        self.name = name
        self.bn = bn
        self.logger = BendersLogger(logging_config)
        self.start_time = time.time()
        self.total_time_limit: Optional[float] = None
        self.update: str = "synchronous"
        self.max_output_trapspaces: int = 1000000
        self.solution_dict: Dict[int, List[Control]] = {}
        self._found_keys: set[Tuple[Tuple[str, int], ...]] = set()

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time

    def _log_build(self, model_name: str, build_time: float) -> None:
        if self.logger.is_on:
            self.logger.build_logger.info(
                f"{self.elapsed_time:.3f},{self.name},0,BUILD_MODEL,{model_name},{build_time:.3f}"
            )

    def _log_found_control(
        self, candidate: Dict[str, int], target_size: int, method: str, solve_time: float
    ) -> None:
        ctrl = Control({k: int(v) for k, v in candidate.items() if k in self.bn.controllable_vars})
        key = tuple(sorted(ctrl.items()))
        if key in self._found_keys:
            return
        self._found_keys.add(key)
        if self.logger.is_on:
            self.logger.solve_logger.info(
                f"{self.elapsed_time:.3f},{self.name},{target_size},LOWER_LEVEL_PROBLEM,"
                f"'PBN_{method}',{solve_time:.3f},True"
            )
            self.logger.cut_logger.info(
                f"{self.elapsed_time:.3f},{self.name},{target_size},LOWER_LEVEL_PROBLEM,"
                f"{EnumCutType.MINIMALITY.name},{len(ctrl)}"
            )

    def _log_size_finished(self, target_size: int, completed: bool) -> None:
        if (not completed) or (not self.logger.is_on):
            return
        self.logger.solve_logger_info(f"{self.elapsed_time:.3f},{self.name},{target_size},FINISHED")

    def _group_controls(
        self, strategies: List[Dict[str, int]], max_control_size: int
    ) -> Dict[int, List[Control]]:
        grouped: Dict[int, List[Control]] = {k: [] for k in range(max_control_size + 1)}
        seen = set()
        controllable = set(self.bn.controllable_vars)
        for strategy in strategies:
            filtered = {k: int(v) for k, v in strategy.items() if k in controllable}
            ctrl = Control(filtered)
            if len(ctrl) > max_control_size:
                continue
            key = tuple(sorted(ctrl.items()))
            if key in seen:
                continue
            seen.add(key)
            grouped[len(ctrl)].append(ctrl)
        return grouped

    def get_control_strategies(
        self,
        max_control_size: int,
        target: Optional[List[Dict[str, int]]] = None,
        avoid_nodes: Optional[List[str]] = None,
        starting_length: int = 0,
        known_strategies: Optional[List[Dict[str, int]]] = None,
    ) -> Dict[int, List[Control]]:
        _require_pyboolnet()
        target_subspaces = target or [{self.bn.phenotype: 1}]
        if avoid_nodes is None:
            avoid_nodes = list(self.bn.uncontrollable_vars) + list(self.bn.fixed_values.keys())

        build_st = time.time()
        primes = make_primes_from_bn(self.bn)
        self._log_build("PBN_PRIMES", time.time() - build_st)
        strategies = compute_control_strategies_with_model_checking(
            primes=primes,
            target=target_subspaces,
            update=self.update,
            limit=max_control_size,
            avoid_nodes=avoid_nodes,
            max_output_trapspaces=self.max_output_trapspaces,
            starting_length=starting_length,
            known_strategies=known_strategies,
            time_limit=self.total_time_limit,
            on_control_found=self._log_found_control,
            on_size_finished=self._log_size_finished,
        )

        self.solution_dict = self._group_controls(strategies, max_control_size)
        self.logger.write_controls_to_json(self.solution_dict)
        return self.solution_dict
