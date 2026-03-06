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
_worker_models: Dict[Tuple[str, int], AggregatedAttractorDetectionIP] = {}


def _get_worker_bn(inst_name: str):
    if inst_name not in _worker_bns:
        _worker_bns[inst_name] = load_bn_in_repo(inst_name)
    return _worker_bns[inst_name]


def _solve_single_length_subproblem(
    inst_name: str,
    ctrl: Control,
    length: int,
) -> Tuple[int, bool]:
    """Solve one fixed-length LLP and report whether it yields a violating attractor."""
    key = (inst_name, length)
    if key not in _worker_models:
        bn = _get_worker_bn(inst_name)
        model_llp = AggregatedAttractorDetectionIP(
            f"{inst_name}_{length}",
            bn,
            length,
            SolverConfig(),
        )
        model_llp.make_constr_stability_condition()
        model_llp.make_constr_periodicity()
        model_llp.make_constr_phenotype_and_length()
        model_llp.set_phenotype_obj()
        model_llp.fix_length(length)
        _worker_models[key] = model_llp
    model_llp = _worker_models[key]
    model_llp.fix_control(ctrl)

    if model_llp.optimize():
        return length, model_llp.p.value < 0.5

    term = getattr(model_llp, "last_termination_condition", None)
    if term == TerminationCondition.infeasible:
        return length, False
    if term == TerminationCondition.maxTimeLimit:
        return length, True
    return length, True


def _solve_length_shard(
    inst_name: str,
    ctrl: Control,
    shard_lengths: List[int],
) -> int:
    """Solve one worker shard and return min violating length in that shard, else -1."""
    for length in shard_lengths:
        _, is_violating = _solve_single_length_subproblem(inst_name, ctrl, length)
        if is_violating:
            return length
    return -1


def _partition_lengths_mod(max_length: int, n_workers: int) -> List[List[int]]:
    shards: List[List[int]] = [[] for _ in range(n_workers)]
    for length in range(1, max_length + 1):
        shards[(length - 1) % n_workers].append(length)
    return shards


def _find_min_viol_len_for_inst(
    inst_name: str,
    ctrl: Control,
    length_shards: List[List[int]],
    worker_pools: List[mp.Pool],
) -> int:
    """Find min violating length from statically assigned shards.

    Reuses worker-local models for (instance, length) across controls, and assigns
    lengths with round-robin mod worker_count.
    """
    if not length_shards:
        return -1

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

    best_violation = -1
    for async_result in pending:
        local_min = _normalize_len(async_result.get())
        if local_min > 0 and (best_violation == -1 or local_min < best_violation):
            best_violation = local_min
    return best_violation


# ---------------------------------------------------------------------------
# Main-process result cache and persistent cache
# Key: (inst_name, ctrl_key, T)  Value: min_viol_len (-1 or >=1)
# ---------------------------------------------------------------------------

_result_cache: Dict[tuple, int] = {}
_checker_data: Dict[str, Dict[str, int]] = {}
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


def _flush_checker_json(json_path: str, data: Dict[str, Dict[str, int]]) -> None:
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
) -> Tuple[Dict[str, Dict[str, int]], bool]:
    normalized: Dict[str, Dict[str, int]] = {}
    changed = False
    for inst, ctrl_map in data.items():
        if not isinstance(ctrl_map, dict):
            changed = True
            continue
        for jk, raw_len in ctrl_map.items():
            norm_len = _parse_cached_len(raw_len)
            if norm_len is None:
                changed = True
                continue
            normalized.setdefault(inst, {})[jk] = norm_len
            if raw_len != norm_len:
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
        for jk, min_viol_len in ctrl_map.items():
            _result_cache[_json_key_to_cache_key(inst, jk, max_length)] = min_viol_len
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


def update_checker_json(
    json_path: str,
    inst: str,
    ctrl: Control,
    min_viol_len: int,
) -> None:
    jk = _ctrl_json_key(ctrl)
    with _checker_lock:
        old_len = _checker_data.get(inst, {}).get(jk)
        if not _should_update_cached_len(old_len, min_viol_len):
            return
        _checker_data.setdefault(inst, {})[jk] = min_viol_len
        try:
            _flush_checker_json(json_path, _checker_data)
        except PermissionError as exc:
            print(f"[WARN] cache write skipped (locked): {json_path} ({exc})")


def extract_from_log(log_path: str, cache_path: str, max_length: int) -> int:
    """Bootstrap cache from prior logs.

    Lines handled:
        work_dir,inst,{ctrl_dict},MIN_VIOL_LEN,<n>
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
            m = re.match(r"(\{[^}]*\}),MIN_VIOL_LEN,(-?\d+)$", rest)
            if not m:
                continue
            ctrl_str, len_str = m.group(1), m.group(2)
            try:
                ctrl_dict = ast.literal_eval(ctrl_str)
            except (ValueError, SyntaxError):
                continue
            min_viol_len = _normalize_len(len_str)
            jk = json.dumps(sorted(ctrl_dict.items()))
            old_len = _checker_data.get(inst, {}).get(jk)
            if _should_update_cached_len(old_len, min_viol_len):
                _checker_data.setdefault(inst, {})[jk] = min_viol_len
                _result_cache[_json_key_to_cache_key(inst, jk, max_length)] = min_viol_len
                if old_len is None:
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


def _log_result(
    output_file: str,
    work_dir: str,
    inst: str,
    ctrl: Control,
    min_viol_len: int,
):
    if min_viol_len > 0:
        tag = os.path.basename(work_dir)
        print(f"\tincorrect [{tag}/{inst}] {ctrl} min_len={min_viol_len}")
    line = f"{work_dir},{inst},{ctrl},MIN_VIOL_LEN,{min_viol_len}"
    with open(output_file, "a", encoding="utf-8") as _f:
        _f.write(line + "\n")


def _log_instance_done(output_file: str, work_dir: str, inst: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tag = os.path.basename(work_dir)
    print(f"\t[{tag}/{inst}] done ({ts})")
    with open(output_file, "a", encoding="utf-8") as _f:
        _f.write(f"{work_dir},{inst},DONE,{ts}\n")


def _collect_pairs(
    work_dir: str, instances: List[str], output_file: str
) -> List[Tuple[str, Control]]:
    pairs: List[Tuple[str, Control]] = []
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
        pairs.extend((inst, ctrl) for ctrl in ctrl_list)
        print(f"\t{os.path.basename(work_dir)}/{inst}: {len(ctrl_list)} controls")
    return pairs


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
) -> None:
    wdi_ctrls: Dict[Tuple[str, str], List[Control]] = {}
    for work_dir in work_dir_list:
        pairs = _collect_pairs(work_dir, instances, output_file)
        for inst, ctrl in pairs:
            wdi = (work_dir, inst)
            wdi_ctrls.setdefault(wdi, []).append(ctrl)

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
        f"across {n_wds} experiment(s), {workers} subproblem workers [T={max_length}]"
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

                for ctrl in ctrl_list:
                    key = (inst, _ctrl_key(ctrl), max_length)
                    if key in _result_cache:
                        min_viol_len = _result_cache[key]
                    else:
                        min_viol_len = _find_min_viol_len_for_inst(
                            inst,
                            ctrl,
                            length_shards,
                            worker_pools,
                        )
                        _result_cache[key] = min_viol_len
                        update_checker_json(cache_path, inst, ctrl, min_viol_len)

                    _log_result(output_file, work_dir, inst, ctrl, min_viol_len)

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
            "for each control, report the minimum attractor length in [1..T] that violates "
            "the phenotype, or -1 if none exists."
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
            f"workers={args.workers} T={args.T}\n"
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
    )

    try:
        _flush_checker_json(cache_path, _checker_data)
    except PermissionError as exc:
        print(f"[WARN] final cache flush skipped (locked): {cache_path} ({exc})")
