import argparse
import ast
import datetime
import json
import os
import re
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from optboolnet.boolnet import Control
from optboolnet.config import SolverConfig
from optboolnet.instances import load_bn_in_repo, _INSTANCE_LIST_FULL
from optboolnet.model import AggregatedAttractorDetectionIP
from pyomo.opt import TerminationCondition


_ALGO_SUBDIRS = ["benders", "MibS"]

# ---------------------------------------------------------------------------
# Per-worker LLP model cache
# ---------------------------------------------------------------------------

_worker_llp_models: Dict[Tuple[str, int], List[Tuple[int, AggregatedAttractorDetectionIP]]] = {}


def _build_decomposed_llp_models(
    inst_name: str,
    max_length: int,
) -> List[Tuple[int, AggregatedAttractorDetectionIP]]:
    bn = load_bn_in_repo(inst_name)
    models: List[Tuple[int, AggregatedAttractorDetectionIP]] = []
    for length in range(1, max_length + 1):
        model = AggregatedAttractorDetectionIP(
            f"{inst_name}_{length}",
            bn,
            length,
            SolverConfig(),
        )
        model.make_constr_stability_condition()
        model.make_constr_periodicity()
        model.make_constr_phenotype_and_length()
        model.set_phenotype_obj()
        model.fix_length(length)
        models.append((length, model))
    return models


def _get_worker_llp_models(
    inst_name: str,
    max_length: int,
) -> List[Tuple[int, AggregatedAttractorDetectionIP]]:
    key = (inst_name, max_length)
    if key not in _worker_llp_models:
        _worker_llp_models[key] = _build_decomposed_llp_models(inst_name, max_length)
    return _worker_llp_models[key]


def _find_min_viol_len_for_inst(inst_name: str, ctrl: Control, max_length: int) -> int:
    """Return -1 if no violating attractor exists up to max_length, else min violating length."""
    is_feasible = False
    for length, model_llp in _get_worker_llp_models(inst_name, max_length):
        model_llp.fix_control(ctrl)
        if model_llp.optimize():
            is_feasible = True
            if model_llp.p.value < 0.5:
                return length
            continue

        term = getattr(model_llp, "last_termination_condition", None)
        if term == TerminationCondition.infeasible:
            continue
        if term == TerminationCondition.maxTimeLimit:
            return length
        return length

    if is_feasible:
        return -1
    return -1


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
    tmp = json_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, json_path)


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
        _flush_checker_json(json_path, _checker_data)

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
        _flush_checker_json(json_path, _checker_data)


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
        _flush_checker_json(cache_path, _checker_data)
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
    all_triplets: List[Tuple[str, str, Control]] = []
    wdi_counts: Dict[Tuple[str, str], int] = {}

    for work_dir in work_dir_list:
        pairs = _collect_pairs(work_dir, instances, output_file)
        for inst, ctrl in pairs:
            wdi = (work_dir, inst)
            wdi_counts[wdi] = wdi_counts.get(wdi, 0) + 1
            all_triplets.append((work_dir, inst, ctrl))

    total = len(all_triplets)
    if total == 0:
        return

    to_run: List[Tuple[str, str, Control, tuple]] = []
    for work_dir, inst, ctrl in all_triplets:
        key = (inst, _ctrl_key(ctrl), max_length)
        if key not in _result_cache:
            to_run.append((work_dir, inst, ctrl, key))

    n_cached = total - len(to_run)
    n_wds = len({wd for wd, _, _ in all_triplets})
    print(
        f"Total: {total} bounded checks ({n_cached} cached, {len(to_run)} to compute) "
        f"across {n_wds} experiment(s), {workers} workers [T={max_length}]"
    )

    for work_dir, inst, ctrl in all_triplets:
        key = (inst, _ctrl_key(ctrl), max_length)
        if key in _result_cache:
            _log_result(output_file, work_dir, inst, ctrl, _result_cache[key])

    wdi_to_run_count: Dict[Tuple[str, str], int] = {}
    for work_dir, inst, ctrl, key in to_run:
        wdi = (work_dir, inst)
        wdi_to_run_count[wdi] = wdi_to_run_count.get(wdi, 0) + 1
    for wdi in wdi_counts:
        if wdi not in wdi_to_run_count:
            _log_instance_done(output_file, wdi[0], wdi[1])

    if not to_run:
        return

    wdi_completed: Dict[Tuple[str, str], int] = {wdi: 0 for wdi in wdi_to_run_count}
    last_pct_by_wdi: Dict[Tuple[str, str], int] = {wdi: 0 for wdi in wdi_to_run_count}

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_find_min_viol_len_for_inst, inst, ctrl, max_length): (
                work_dir,
                inst,
                ctrl,
                key,
            )
            for work_dir, inst, ctrl, key in to_run
        }
        for future in as_completed(futures):
            work_dir, inst, ctrl, key = futures[future]
            min_viol_len = _normalize_len(future.result())

            _result_cache[key] = min_viol_len
            update_checker_json(cache_path, inst, ctrl, min_viol_len)
            _log_result(output_file, work_dir, inst, ctrl, min_viol_len)

            wdi = (work_dir, inst)
            wdi_completed[wdi] += 1
            n = wdi_to_run_count[wdi]
            c = wdi_completed[wdi]

            pct = c * 100 // n
            milestone = pct // 10 * 10
            if milestone > last_pct_by_wdi[wdi]:
                tag = os.path.basename(work_dir)
                print(f"\t[{tag}/{inst}] {milestone}% ({c}/{n})")
                last_pct_by_wdi[wdi] = milestone

            if c == n:
                _log_instance_done(output_file, work_dir, inst)


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
        default=os.cpu_count(),
        metavar="N",
        help=f"Number of parallel LLP checks (default: cpu_count={os.cpu_count()})",
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
    args = ap.parse_args()

    if args.T < 1:
        raise ValueError("--T must be >= 1")

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
