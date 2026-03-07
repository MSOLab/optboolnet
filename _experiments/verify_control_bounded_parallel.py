import argparse
import ast
import datetime
import json
import multiprocessing as mp
from multiprocessing.pool import ApplyResult
import os
import re
import threading
import time
from typing import Dict, List, Optional, Tuple

from optboolnet.boolnet import Control
from optboolnet.config import SolverConfig
from optboolnet.instances import load_bn_in_repo, _INSTANCE_LIST_FULL
from optboolnet.model import AggregatedAttractorDetectionIP
from pyomo.opt import TerminationCondition


_ALGO_SUBDIRS = ["benders", "MibS"]

# ---------------------------------------------------------------------------
# Per-worker state
# ---------------------------------------------------------------------------

_worker_bns: Dict[str, object] = {}
_worker_models_viol: Dict[Tuple[str, int], AggregatedAttractorDetectionIP] = {}
_worker_models_any: Dict[Tuple[str, int], AggregatedAttractorDetectionIP] = {}


def _get_worker_bn(inst_name: str):
    if inst_name not in _worker_bns:
        _worker_bns[inst_name] = load_bn_in_repo(inst_name)
    return _worker_bns[inst_name]


def _solve_single_length_subproblem(
    inst_name: str,
    ctrl: Control,
    length: int,
) -> Tuple[int, bool, bool]:
    """Solve one fixed-length LLP and report attractor existence + violating existence."""
    key = (inst_name, length)
    if key not in _worker_models_any:
        bn = _get_worker_bn(inst_name)
        model_any = AggregatedAttractorDetectionIP(
            f"{inst_name}_{length}_any",
            bn,
            length,
            SolverConfig(),
        )
        model_any.make_constr_stability_condition()
        model_any.make_constr_periodicity()
        model_any.fix_length(length)
        _worker_models_any[key] = model_any
    model_any = _worker_models_any[key]
    model_any.fix_control(ctrl)

    if model_any.optimize():
        has_any = True
    else:
        term_any = getattr(model_any, "last_termination_condition", None)
        if term_any == TerminationCondition.infeasible:
            has_any = False
        elif term_any == TerminationCondition.maxTimeLimit:
            has_any = True
        else:
            has_any = True

    if not has_any:
        return length, False, False

    if key not in _worker_models_viol:
        bn = _get_worker_bn(inst_name)
        model_viol = AggregatedAttractorDetectionIP(
            f"{inst_name}_{length}_viol",
            bn,
            length,
            SolverConfig(),
        )
        model_viol.make_constr_stability_condition()
        model_viol.make_constr_periodicity()
        model_viol.make_constr_phenotype_and_length()
        model_viol.set_phenotype_obj()
        model_viol.fix_length(length)
        _worker_models_viol[key] = model_viol
    model_viol = _worker_models_viol[key]
    model_viol.fix_control(ctrl)

    if model_viol.optimize():
        return length, True, model_viol.p.value < 0.5

    term_viol = getattr(model_viol, "last_termination_condition", None)
    if term_viol == TerminationCondition.infeasible:
        return length, True, False
    if term_viol == TerminationCondition.maxTimeLimit:
        return length, True, True
    return length, True, True


def _solve_length_shard(
    inst_name: str,
    ctrl: Control,
    shard_lengths: List[int],
) -> Tuple[int, int]:
    """Return (min_viol_len, min_attr_len) in this shard, each -1 if absent."""
    min_viol_len = -1
    min_attr_len = -1
    for length in shard_lengths:
        _, has_any, is_violating = _solve_single_length_subproblem(inst_name, ctrl, length)
        if has_any and min_attr_len == -1:
            min_attr_len = length
        if is_violating and min_viol_len == -1:
            min_viol_len = length
        if min_attr_len > 0 and min_viol_len > 0:
            break
    return min_viol_len, min_attr_len


def _partition_lengths_mod(max_length: int, n_workers: int) -> List[List[int]]:
    shards: List[List[int]] = [[] for _ in range(n_workers)]
    for length in range(1, max_length + 1):
        shards[(length - 1) % n_workers].append(length)
    return shards


def _find_min_lens_for_inst(
    inst_name: str,
    ctrl: Control,
    length_shards: List[List[int]],
    worker_pools: List[mp.Pool],
) -> Tuple[int, int]:
    """Find (min_viol_len, min_attr_len) from statically assigned shards.

    Reuses worker-local models for (instance, length) across controls, and assigns
    lengths with round-robin mod worker_count.
    """
    if not length_shards:
        return -1, -1

    n_workers = len(length_shards)
    if n_workers == 1:
        return _solve_length_shard(inst_name, ctrl, length_shards[0])

    pending: List[ApplyResult] = []
    for worker_idx, shard_lengths in enumerate(length_shards):
        if not shard_lengths:
            continue
        pending.append(
            worker_pools[worker_idx].apply_async(
                _solve_length_shard,
                (inst_name, ctrl, shard_lengths),
            )
        )

    best_viol = -1
    best_attr = -1
    for async_result in pending:
        local_viol, local_attr = async_result.get()
        local_viol = _normalize_len(local_viol)
        local_attr = _normalize_len(local_attr)
        if local_viol > 0 and (best_viol == -1 or local_viol < best_viol):
            best_viol = local_viol
        if local_attr > 0 and (best_attr == -1 or local_attr < best_attr):
            best_attr = local_attr
    return best_viol, best_attr


# ---------------------------------------------------------------------------
# Main-process result cache and persistent cache
# Key: (inst_name, ctrl_key, T)  Value: (min_viol_len, min_attr_len)
# ---------------------------------------------------------------------------

_result_cache: Dict[tuple, Tuple[int, int]] = {}
_checker_data: Dict[str, Dict[str, Dict[str, int]]] = {}
_checker_lock = threading.Lock()


def _ctrl_key(ctrl: Control) -> tuple:
    return tuple(sorted(ctrl.items()))


def _ctrl_json_key(ctrl: Control) -> str:
    return json.dumps(sorted(ctrl.items()))


def _json_key_to_cache_key(inst: str, json_key: str, max_length: int) -> tuple:
    return (inst, tuple((k, v) for k, v in json.loads(json_key)), max_length)


def _normalize_len(value: object) -> int:
    try:
        iv = int(value)
    except (TypeError, ValueError):
        return -1
    return iv if iv > 0 else -1


def _parse_cached_len(value: object) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        iv = int(value)
    except (TypeError, ValueError):
        return None
    return iv if iv > 0 else -1


def _parse_cached_pair(value: object) -> Optional[Tuple[int, int]]:
    if not isinstance(value, dict):
        return None
    min_viol = _parse_cached_len(value.get("min_attr_len_viol"))
    min_attr = _parse_cached_len(value.get("min_attr_len"))
    if min_viol is None or min_attr is None:
        return None
    return min_viol, min_attr


def _flush_checker_json(json_path: str, data: Dict[str, Dict[str, Dict[str, int]]]) -> None:
    tmp = f"{json_path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    delays = [0.05, 0.1, 0.2, 0.5, 1.0]
    last_err: Optional[PermissionError] = None
    for delay in delays:
        try:
            os.replace(tmp, json_path)
            return
        except PermissionError as exc:
            last_err = exc
            time.sleep(delay)

    try:
        os.replace(tmp, json_path)
        return
    except PermissionError as exc:
        last_err = exc
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass

    if last_err is not None:
        raise last_err


def _load_single_cache(json_path: str) -> Dict[str, Dict[str, object]]:
    if not os.path.exists(json_path):
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return data


def _normalize_cache(
    data: Dict[str, Dict[str, object]]
) -> Tuple[Dict[str, Dict[str, Dict[str, int]]], bool]:
    normalized: Dict[str, Dict[str, Dict[str, int]]] = {}
    changed = False
    for inst, ctrl_map in data.items():
        if not isinstance(ctrl_map, dict):
            changed = True
            continue
        for jk, raw_pair in ctrl_map.items():
            norm_pair = _parse_cached_pair(raw_pair)
            if norm_pair is None:
                changed = True
                continue
            min_viol, min_attr = norm_pair
            normalized.setdefault(inst, {})[jk] = {
                "min_attr_len_viol": min_viol,
                "min_attr_len": min_attr,
            }
            if not isinstance(raw_pair, dict):
                changed = True
    return normalized, changed


def load_checker_json(json_path: str, max_length: int) -> int:
    global _checker_data
    _checker_data, normalized = _normalize_cache(_load_single_cache(json_path))
    if normalized:
        try:
            _flush_checker_json(json_path, _checker_data)
        except PermissionError as exc:
            print(f"[WARN] cache file is locked; skip normalize write now: {json_path} ({exc})")

    count = 0
    for inst, ctrl_map in _checker_data.items():
        for jk, pair_data in ctrl_map.items():
            min_viol_len = _normalize_len(pair_data.get("min_attr_len_viol"))
            min_attr_len = _normalize_len(pair_data.get("min_attr_len"))
            _result_cache[_json_key_to_cache_key(inst, jk, max_length)] = (
                min_viol_len,
                min_attr_len,
            )
            count += 1
    return count


def _should_update_cached_len(old_len: Optional[int], new_len: int) -> bool:
    if old_len is None:
        return True
    if old_len == -1 and new_len > 0:
        return True
    if old_len > 0 and new_len > 0 and new_len < old_len:
        return True
    return False


def _merge_cached_len(old_len: Optional[int], new_len: int) -> int:
    if _should_update_cached_len(old_len, new_len):
        return new_len
    return old_len if old_len is not None else new_len


def update_checker_json(
    json_path: str,
    inst: str,
    ctrl: Control,
    min_viol_len: int,
    min_attr_len: int,
) -> None:
    jk = _ctrl_json_key(ctrl)
    with _checker_lock:
        old_pair = _checker_data.get(inst, {}).get(jk, {})
        old_viol = _parse_cached_len(old_pair.get("min_attr_len_viol"))
        old_attr = _parse_cached_len(old_pair.get("min_attr_len"))

        new_viol = _merge_cached_len(old_viol, min_viol_len)
        new_attr = _merge_cached_len(old_attr, min_attr_len)
        if old_viol == new_viol and old_attr == new_attr:
            return
        _checker_data.setdefault(inst, {})[jk] = {
            "min_attr_len_viol": new_viol,
            "min_attr_len": new_attr,
        }
        try:
            _flush_checker_json(json_path, _checker_data)
        except PermissionError as exc:
            print(f"[WARN] cache write skipped (locked): {json_path} ({exc})")


def extract_from_log(log_path: str, cache_path: str, max_length: int) -> int:
    """Bootstrap cache from prior logs.

    Lines handled:
        work_dir,inst,{ctrl_dict},MIN_ATTR_LEN_VIOL,<n>,MIN_ATTR_LEN,<m>
    """
    if not os.path.exists(log_path):
        return 0
    count = 0
    changed = False
    with open(log_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",", 2)
            if len(parts) < 3:
                continue
            inst = parts[1]
            rest = parts[2]
            m = re.match(
                r"(\{[^}]*\}),MIN_ATTR_LEN_VIOL,(-?\d+),MIN_ATTR_LEN,(-?\d+)$",
                rest,
            )
            if not m:
                continue
            ctrl_str, viol_str, attr_str = m.group(1), m.group(2), m.group(3)
            try:
                ctrl_dict = ast.literal_eval(ctrl_str)
            except (ValueError, SyntaxError):
                continue
            min_viol_len = _normalize_len(viol_str)
            min_attr_len = _normalize_len(attr_str)
            jk = json.dumps(sorted(ctrl_dict.items()))
            old_pair = _checker_data.get(inst, {}).get(jk, {})
            old_viol = _parse_cached_len(old_pair.get("min_attr_len_viol"))
            old_attr = _parse_cached_len(old_pair.get("min_attr_len"))
            new_viol = _merge_cached_len(old_viol, min_viol_len)
            new_attr = _merge_cached_len(old_attr, min_attr_len)
            if old_viol != new_viol or old_attr != new_attr:
                _checker_data.setdefault(inst, {})[jk] = {
                    "min_attr_len_viol": new_viol,
                    "min_attr_len": new_attr,
                }
                _result_cache[_json_key_to_cache_key(inst, jk, max_length)] = (
                    new_viol,
                    new_attr,
                )
                if old_viol is None and old_attr is None:
                    count += 1
                changed = True
    if changed:
        try:
            _flush_checker_json(cache_path, _checker_data)
        except PermissionError as exc:
            print(f"[WARN] cache write skipped after bootstrap (locked): {cache_path} ({exc})")
    return count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_sol_path(work_dir: str, inst: str) -> Optional[str]:
    for subdir in _ALGO_SUBDIRS:
        path = os.path.join(work_dir, subdir, inst, "sol.json")
        if os.path.exists(path):
            return path
    return None


def _load_alg_max_length(work_dir: str) -> Optional[int]:
    cfg_path = os.path.join(work_dir, "alg_config.json")
    if not os.path.exists(cfg_path):
        return None
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        tmax = int(cfg["max_length"])
        return tmax if tmax >= 1 else None
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _is_valid_control(min_viol_len: int, min_attr_len: int, t_max: int) -> bool:
    return (1 <= min_attr_len <= t_max) and not (1 <= min_viol_len <= t_max)


def _log_result(
    output_file: str,
    work_dir: str,
    inst: str,
    ctrl: Control,
    min_viol_len: int,
    min_attr_len: int,
):
    line = (
        f"{work_dir},{inst},{ctrl},MIN_ATTR_LEN_VIOL,{min_viol_len},"
        f"MIN_ATTR_LEN,{min_attr_len}"
    )
    with open(output_file, "a", encoding="utf-8") as _f:
        _f.write(line + "\n")


def _log_error(
    output_file: str,
    work_dir: str,
    inst: str,
    ctrl: Control,
    err_type: str,
    detail: str,
) -> None:
    line = f"{work_dir},{inst},{ctrl},ERROR,{err_type},{detail}"
    with open(output_file, "a", encoding="utf-8") as _f:
        _f.write(line + "\n")



def _log_instance_done(output_file: str, work_dir: str, inst: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tag = os.path.basename(work_dir)
    print(f"\t[{tag}/{inst}] done ({ts})")
    with open(output_file, "a", encoding="utf-8") as _f:
        _f.write(f"{work_dir},{inst},DONE,{ts}\n")


def _collect_pairs(
    work_dir: str,
    instances: List[str],
    output_file: str,
    check_subset: bool,
) -> Tuple[
    List[Tuple[str, Control]],
    Dict[str, set],
    Dict[str, Dict[tuple, List[tuple]]],
]:
    def _all_subsets(ctrl: Control) -> List[Tuple[tuple, Control]]:
        items = sorted(ctrl.items())
        n = len(items)
        out: List[Tuple[tuple, Control]] = []
        for mask in range(1 << n):
            sub = {
                k: v
                for i, (k, v) in enumerate(items)
                if (mask >> i) & 1
            }
            sub_ctrl = Control(sub)
            out.append((_ctrl_key(sub_ctrl), sub_ctrl))
        return out

    pairs: List[Tuple[str, Control]] = []
    original_keys_by_inst: Dict[str, set] = {}
    strict_subset_keys_by_original_by_inst: Dict[str, Dict[tuple, List[tuple]]] = {}
    for inst in instances:
        sol_path = _find_sol_path(work_dir, inst)
        if sol_path is None:
            print(f"\t{os.path.basename(work_dir)}/{inst}: sol.json not found, skipping")
            with open(output_file, "a", encoding="utf-8") as _f:
                _f.write(f"{work_dir},{inst},MISSING\n")
            continue
        ctrl_list: List[Control] = []
        with open(sol_path, "r", encoding="utf-8") as _f:
            for sol_list in json.load(_f).values():
                for sol in sol_list:
                    ctrl_list.append(Control(sol))
        original_keys = {_ctrl_key(ctrl) for ctrl in ctrl_list}
        original_keys_by_inst[inst] = original_keys
        if check_subset:
            uniq_subset_ctrls: Dict[tuple, Control] = {}
            strict_subset_keys_by_original: Dict[tuple, set] = {}
            for ctrl in ctrl_list:
                orig_key = _ctrl_key(ctrl)
                strict_subset_keys_by_original.setdefault(orig_key, set())
                for sub_ctrl in _all_subsets(ctrl):
                    sub_key, sub_val = sub_ctrl
                    uniq_subset_ctrls.setdefault(sub_key, sub_val)
                    if sub_key != orig_key:
                        strict_subset_keys_by_original[orig_key].add(sub_key)
            expanded_ctrls = list(uniq_subset_ctrls.values())
            strict_subset_keys_by_original_by_inst[inst] = {
                k: sorted(list(v))
                for k, v in strict_subset_keys_by_original.items()
            }
            pairs.extend((inst, ctrl) for ctrl in expanded_ctrls)
            print(
                f"\t{os.path.basename(work_dir)}/{inst}: {len(ctrl_list)} controls "
                f"-> {len(expanded_ctrls)} unique subsets"
            )
        else:
            pairs.extend((inst, ctrl) for ctrl in ctrl_list)
            print(f"\t{os.path.basename(work_dir)}/{inst}: {len(ctrl_list)} controls")
    return pairs, original_keys_by_inst, strict_subset_keys_by_original_by_inst


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def verify_all(
    work_dir_list: List[str],
    output_file: str,
    instances: List[str],
    workers: int,
    max_length: int,
    cache_path: str,
    check_subset: bool,
) -> None:
    wdi_ctrls: Dict[Tuple[str, str], List[Control]] = {}
    wdi_original_keys: Dict[Tuple[str, str], set] = {}
    wdi_strict_subset_keys_by_original: Dict[Tuple[str, str], Dict[tuple, List[tuple]]] = {}
    t_max_by_work_dir: Dict[str, int] = {}
    for work_dir in work_dir_list:
        cfg_t_max = _load_alg_max_length(work_dir)
        if cfg_t_max is None:
            print(
                f"[WARN] {work_dir}: missing/invalid alg_config.json max_length; "
                f"fallback T_max={max_length}"
            )
            cfg_t_max = max_length
        elif cfg_t_max > max_length:
            print(
                f"[WARN] {work_dir}: alg_config max_length={cfg_t_max} > --T={max_length}; "
                "nonminimality checks may be conservative."
            )
        t_max_by_work_dir[work_dir] = cfg_t_max

    for work_dir in work_dir_list:
        pairs, original_keys_by_inst, strict_subset_keys_by_original_by_inst = _collect_pairs(
            work_dir,
            instances,
            output_file,
            check_subset,
        )
        for inst, ctrl in pairs:
            wdi = (work_dir, inst)
            wdi_ctrls.setdefault(wdi, []).append(ctrl)
        for inst, keys in original_keys_by_inst.items():
            wdi_original_keys[(work_dir, inst)] = keys
        for inst, subset_map in strict_subset_keys_by_original_by_inst.items():
            wdi_strict_subset_keys_by_original[(work_dir, inst)] = subset_map

    total = sum(len(ctrl_list) for ctrl_list in wdi_ctrls.values())
    if total == 0:
        return

    n_to_compute = 0
    for (_, inst), ctrl_list in wdi_ctrls.items():
        for ctrl in ctrl_list:
            if (inst, _ctrl_key(ctrl), max_length) not in _result_cache:
                n_to_compute += 1
    n_cached = total - n_to_compute
    n_wds = len({work_dir for work_dir, _ in wdi_ctrls})
    print(
        f"Total: {total} bounded checks ({n_cached} cached, {n_to_compute} to compute) "
        f"across {n_wds} experiment(s), {workers} subproblem workers [T={max_length}], "
        f"check_subset={check_subset}"
    )

    # Process by fixed instance first, then iterate experiments sequentially.
    # This keeps the same worker pool alive for that instance and lets workers
    # reuse their cached (instance, length) models across experiments.
    inst_to_wdis: Dict[str, List[Tuple[str, List[Control]]]] = {}
    for (work_dir, inst), ctrl_list in wdi_ctrls.items():
        inst_to_wdis.setdefault(inst, []).append((work_dir, ctrl_list))

    for inst in instances:
        wdis = inst_to_wdis.get(inst, [])
        if not wdis:
            continue
        wdis.sort(key=lambda x: x[0])

        total_for_fixed_inst = sum(len(ctrl_list) for _, ctrl_list in wdis)
        print(
            f"[{inst}] {len(wdis)} experiment(s), {total_for_fixed_inst} control(s) "
            f"with shared subproblem workers"
        )

        n_workers_inst = max(1, min(workers, max_length))
        length_shards = _partition_lengths_mod(max_length, n_workers_inst)
        worker_pools = [mp.Pool(processes=1) for _ in range(n_workers_inst)]
        try:
            for work_dir, ctrl_list in wdis:
                total_for_wdi = len(ctrl_list)
                completed_for_wdi = 0
                last_pct = 0
                wdi = (work_dir, inst)
                original_keys = wdi_original_keys.get(wdi, set())
                t_max = t_max_by_work_dir.get(work_dir, max_length)
                original_valid_by_key: Dict[tuple, bool] = {}

                for ctrl in ctrl_list:
                    ctrl_k = _ctrl_key(ctrl)
                    key = (inst, ctrl_k, max_length)
                    if key in _result_cache:
                        min_viol_len, min_attr_len = _result_cache[key]
                    else:
                        min_viol_len, min_attr_len = _find_min_lens_for_inst(
                            inst,
                            ctrl,
                            length_shards,
                            worker_pools,
                        )
                        _result_cache[key] = (min_viol_len, min_attr_len)
                        update_checker_json(
                            cache_path,
                            inst,
                            ctrl,
                            min_viol_len,
                            min_attr_len,
                        )

                    if ctrl_k in original_keys:
                        _log_result(
                            output_file,
                            work_dir,
                            inst,
                            ctrl,
                            min_viol_len,
                            min_attr_len,
                        )
                        is_valid = _is_valid_control(min_viol_len, min_attr_len, t_max)
                        original_valid_by_key[ctrl_k] = is_valid
                        if not is_valid:
                            tag = os.path.basename(work_dir)
                            print(
                                f"\tincorrect [{tag}/{inst}] {ctrl} "
                                f"(T_max={t_max}, min_viol={min_viol_len}, min_attr={min_attr_len})"
                            )
                            _log_error(
                                output_file=output_file,
                                work_dir=work_dir,
                                inst=inst,
                                ctrl=ctrl,
                                err_type="INCORRECT",
                                detail=(
                                    f"T_MAX,{t_max},MIN_ATTR_LEN_VIOL,{min_viol_len},"
                                    f"MIN_ATTR_LEN,{min_attr_len}"
                                ),
                            )

                    completed_for_wdi += 1
                    pct = completed_for_wdi * 100 // total_for_wdi
                    milestone = pct // 10 * 10
                    if milestone > last_pct:
                        tag = os.path.basename(work_dir)
                        print(
                            f"\t[{tag}/{inst}] {milestone}% "
                            f"({completed_for_wdi}/{total_for_wdi})"
                        )
                        last_pct = milestone

                if check_subset:
                    subset_map = wdi_strict_subset_keys_by_original.get(wdi, {})
                    for orig_key, subset_keys in subset_map.items():
                        # Only valid controls can be non-minimal under the new definition.
                        if not original_valid_by_key.get(orig_key, False):
                            continue
                        witness_key = None
                        witness_pair: Optional[Tuple[int, int]] = None
                        for sk in subset_keys:
                            pair = _result_cache.get((inst, sk, max_length))
                            if pair is None:
                                continue
                            sub_min_viol, sub_min_attr = pair
                            if _is_valid_control(sub_min_viol, sub_min_attr, t_max):
                                witness_key = sk
                                witness_pair = pair
                                break
                        if witness_key is None:
                            continue
                        orig_ctrl = Control(dict(orig_key))
                        witness_ctrl = Control(dict(witness_key))
                        witness_min_viol, witness_min_attr = witness_pair  # type: ignore[misc]
                        tag = os.path.basename(work_dir)
                        print(
                            f"\tnonminimal [{tag}/{inst}] {orig_ctrl} "
                            f"witness_subset={witness_ctrl} "
                            f"(T_max={t_max}, "
                            f"witness_min_viol={witness_min_viol}, witness_min_attr={witness_min_attr})"
                        )
                        _log_error(
                            output_file=output_file,
                            work_dir=work_dir,
                            inst=inst,
                            ctrl=orig_ctrl,
                            err_type="NONMINIMAL",
                            detail=(
                                f"WITNESS_SUBSET,{witness_ctrl},T_MAX,{t_max},"
                                f"WITNESS_MIN_ATTR_LEN_VIOL,{witness_min_viol},"
                                f"WITNESS_MIN_ATTR_LEN,{witness_min_attr}"
                            ),
                        )

                _log_instance_done(output_file, work_dir, inst)
        finally:
            for pool in worker_pools:
                pool.close()
            for pool in worker_pools:
                pool.join()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=(
            "Verify controls from sol.json in parallel using bounded LLP checks: "
            "for each control, compute min violating attractor length and min attractor length "
            "in [1..T] (or -1 if none exists within T), then classify INCORRECT/NONMINIMAL."
        )
    )
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--work_dirs",
        nargs="+",
        metavar="DIR",
        help="One or more explicit work directories to verify.",
    )
    group.add_argument(
        "--root_dir",
        metavar="DIR",
        help=(
            "Root directory; every subdirectory that contains a 'benders/' or 'MibS/' folder "
            "is treated as a work directory."
        ),
    )
    ap.add_argument(
        "--instances",
        nargs="+",
        metavar="INST",
        default=_INSTANCE_LIST_FULL,
        help=f"Instances to verify (default: all). Available: {', '.join(_INSTANCE_LIST_FULL)}",
    )
    ap.add_argument(
        "--T",
        type=int,
        required=True,
        metavar="N",
        help="Length bound T: check lengths 1..T.",
    )
    ap.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help=(
            "Output file to write bounded results to "
            "(default: _experiments/verify_control_bounded_log_T{T}.txt)."
        ),
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        metavar="N",
        help=(
            "Number of parallel fixed-length LLP subproblems per control "
            f"(default: cpu_count={os.cpu_count() or 1})"
        ),
    )
    ap.add_argument(
        "--json-cache",
        metavar="FILE",
        default=None,
        help=(
            "Persistent JSON cache for bounded checks "
            "(default: _experiments/checker_bounded_T{T}.json)."
        ),
    )
    ap.add_argument(
        "--bootstrap-from-log",
        action="store_true",
        help=(
            "If set, bootstrap cache entries from existing --output log "
            "(default: off; only JSON cache is used)."
        ),
    )
    ap.add_argument(
        "--check-subset",
        action="store_true",
        help=(
            "If set, expand each control to all subsets (2^k per control of size k) "
            "and verify/cache those subsets as well."
        ),
    )
    args = ap.parse_args()

    if args.T < 1:
        raise ValueError("--T must be >= 1")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    output_path = (
        args.output
        if args.output is not None
        else f"_experiments/verify_control_bounded_log_T{args.T}.txt"
    )
    cache_path = (
        args.json_cache
        if args.json_cache is not None
        else f"_experiments/checker_bounded_T{args.T}.json"
    )

    n_loaded = load_checker_json(cache_path, args.T)
    if n_loaded:
        print(f"Loaded {n_loaded} cached bounded result(s) from {cache_path}")
    if args.bootstrap_from_log:
        n_from_log = extract_from_log(output_path, cache_path, args.T)
        if n_from_log:
            print(f"Bootstrap-added {n_from_log} bounded result(s) from existing log")

    _timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _dirs_str = ", ".join(args.work_dirs) if args.work_dirs else f"root={args.root_dir}"
    _insts_str = ", ".join(args.instances)
    with open(output_path, "a", encoding="utf-8") as _f:
        _f.write(
            f"# [{_timestamp}] work_dirs=[{_dirs_str}] instances=[{_insts_str}] "
            f"workers={args.workers} T={args.T} check_subset={args.check_subset}\n"
        )

    if args.work_dirs:
        work_dir_list = args.work_dirs
    else:
        root_dir = args.root_dir
        work_dir_list = [
            os.path.join(root_dir, sub)
            for sub in sorted(os.listdir(root_dir))
            if any(
                os.path.isdir(os.path.join(root_dir, sub, algo))
                for algo in _ALGO_SUBDIRS
            )
        ]
        print(f"Found {len(work_dir_list)} experiment(s) under {root_dir}:")
        for d in work_dir_list:
            print(f"  {d}")

    verify_all(
        work_dir_list,
        output_path,
        args.instances,
        args.workers,
        args.T,
        cache_path,
        args.check_subset,
    )

    try:
        _flush_checker_json(cache_path, _checker_data)
    except PermissionError as exc:
        print(f"[WARN] final cache flush skipped (locked): {cache_path} ({exc})")
