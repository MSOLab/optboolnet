import argparse
import ast
import datetime
import json
import os
import re
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from optboolnet.boolnet import Control
from optboolnet.checking import nusmv_check_phenotype, nusmv_check_phenotype_full
from optboolnet.instances import load_bn_in_repo, _INSTANCE_LIST_FULL


_ALGO_SUBDIRS = ["benders", "MibS"]

# ---------------------------------------------------------------------------
# Per-worker state (each worker process has its own copy)
# ---------------------------------------------------------------------------

_worker_bns: dict = {}


def _check_ctrl_for_inst(inst_name: str, ctrl: Control) -> bool:
    """CTL phenotype check — fast, no counterexample trace."""
    if inst_name not in _worker_bns:
        _worker_bns[inst_name] = load_bn_in_repo(inst_name)
    bn = _worker_bns[inst_name]
    ok = nusmv_check_phenotype(
        bn,
        control=ctrl,
        property_variant="ctl_not_ef_ag",
        constrain_controlled_vars=True,
        preprocess_propagation=True,
    )
    return ok


def _get_loop_len_for_inst(inst_name: str, ctrl: Control) -> Optional[int]:
    """LTL phenotype check — slower, parses attractor cycle length from the trace."""
    if inst_name not in _worker_bns:
        _worker_bns[inst_name] = load_bn_in_repo(inst_name)
    bn = _worker_bns[inst_name]
    _, loop_len = nusmv_check_phenotype_full(
        bn,
        control=ctrl,
        preprocess_propagation=True,
    )
    return loop_len


def _check_ctrl_ltl_for_inst(inst_name: str, ctrl: Control) -> Tuple[bool, Optional[int]]:
    """LTL phenotype check with counterexample loop length."""
    if inst_name not in _worker_bns:
        _worker_bns[inst_name] = load_bn_in_repo(inst_name)
    bn = _worker_bns[inst_name]
    ok, loop_len = nusmv_check_phenotype_full(
        bn,
        control=ctrl,
        preprocess_propagation=True,
    )
    return ok, loop_len


# ---------------------------------------------------------------------------
# Main-process result cache (persists for the full run across all work_dirs)
# Key: (inst_name, ctrl_key)  Value: ok (bool only)
# ---------------------------------------------------------------------------

_result_cache: Dict[tuple, bool] = {}


def _ctrl_key(ctrl: Control) -> tuple:
    return tuple(sorted(ctrl.items()))


# ---------------------------------------------------------------------------
# Persistent JSON caches:
#   checker_positive.json: {inst -> {ctrl_json_key -> true}}
#   checker_negative.json: {inst -> {ctrl_json_key -> loop_len}}
#
# _checker_positive_data / _checker_negative_data mirror on-disk JSON so we
# never re-read the files on every update.  All writes go through
# _flush_checker_json which does an
# atomic rename, so a killed process cannot corrupt the file.
# _checker_lock serialises updates in the (single) main process.
# ---------------------------------------------------------------------------

_checker_positive_data: Dict[str, Dict[str, bool]] = {}
_checker_negative_data: Dict[str, Dict[str, int]] = {}
_checker_lock = threading.Lock()


def _ctrl_json_key(ctrl: Control) -> str:
    """Canonical JSON string key for a Control."""
    return json.dumps(sorted(ctrl.items()))


def _json_key_to_cache_key(inst: str, json_key: str) -> tuple:
    return (inst, tuple((k, v) for k, v in json.loads(json_key)))


def _flush_checker_json(json_path: str, data: Dict[str, Dict[str, object]]) -> None:
    """Best-effort atomic overwrite; warn on failure and keep running."""
    tmp = f"{json_path}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError as exc:
        print(f"[WARN] cache tmp write failed: {tmp} ({exc})")
        return

    delays = [0.05, 0.1, 0.2, 0.5, 1.0]
    last_err: Optional[OSError] = None
    for delay in delays:
        try:
            os.replace(tmp, json_path)
            return
        except OSError as exc:
            last_err = exc
            time.sleep(delay)

    print(
        f"[WARN] cache replace failed; skipping flush for now: {json_path} "
        f"(tmp={tmp}, err={last_err})"
    )
    if os.path.exists(tmp):
        try:
            os.remove(tmp)
        except OSError:
            pass


def _safe_flush_checker_json(json_path: str, data: Dict[str, Dict[str, object]]) -> None:
    """Never raise from cache flush; emit warning and continue."""
    try:
        _flush_checker_json(json_path, data)
    except Exception as exc:  # pragma: no cover - defensive guard
        print(f"[WARN] cache flush crashed; skipping write: {json_path} ({exc})")


def _merge_cache_into_result_cache(data: Dict[str, Dict[str, object]], ok_value: bool) -> int:
    count = 0
    for inst, ctrl_map in data.items():
        for jk in ctrl_map.keys():
            _result_cache[_json_key_to_cache_key(inst, jk)] = ok_value
            count += 1
    return count


def _load_single_cache(json_path: str) -> Dict[str, Dict[str, object]]:
    if not os.path.exists(json_path):
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return data


def _normalize_negative_len(value: object) -> int:
    """Normalize cached loop length to an integer (-1 = unknown)."""
    if isinstance(value, bool):
        return -1
    try:
        iv = int(value)
    except (TypeError, ValueError):
        return -1
    return iv if iv > 0 else -1


def _normalize_negative_cache(data: Dict[str, Dict[str, object]]) -> Tuple[Dict[str, Dict[str, int]], bool]:
    normalized: Dict[str, Dict[str, int]] = {}
    changed = False
    for inst, ctrl_map in data.items():
        if not isinstance(ctrl_map, dict):
            changed = True
            continue
        for jk, raw_len in ctrl_map.items():
            norm_len = _normalize_negative_len(raw_len)
            normalized.setdefault(inst, {})[jk] = norm_len
            if raw_len != norm_len:
                changed = True
    return normalized, changed


def _normalize_positive_cache(data: Dict[str, Dict[str, object]]) -> Tuple[Dict[str, Dict[str, bool]], bool]:
    normalized: Dict[str, Dict[str, bool]] = {}
    changed = False
    for inst, ctrl_map in data.items():
        if not isinstance(ctrl_map, dict):
            changed = True
            continue
        for jk, raw_ok in ctrl_map.items():
            if bool(raw_ok):
                normalized.setdefault(inst, {})[jk] = True
            else:
                changed = True
            if raw_ok is not True:
                changed = True
    return normalized, changed


def load_checker_jsons(
    positive_path: str,
    negative_path: str,
    legacy_path: Optional[str] = None,
) -> Tuple[int, int, int]:
    """Load positive/negative caches and optionally migrate legacy checker.json.

    Returns (positive_count, negative_count, migrated_from_legacy_count).
    """
    global _checker_positive_data, _checker_negative_data
    pos_exists = os.path.exists(positive_path)
    neg_exists = os.path.exists(negative_path)
    _checker_positive_data, pos_normalized = _normalize_positive_cache(
        _load_single_cache(positive_path)
    )
    _checker_negative_data, neg_normalized = _normalize_negative_cache(
        _load_single_cache(negative_path)
    )

    migrated = 0
    should_try_legacy_migration = (
        legacy_path
        and os.path.exists(legacy_path)
        and (not pos_exists or not neg_exists)
    )
    if should_try_legacy_migration:
        with open(legacy_path, "r", encoding="utf-8") as f:
            legacy = json.load(f)
        if isinstance(legacy, dict):
            for inst, ctrl_map in legacy.items():
                if not isinstance(ctrl_map, dict):
                    continue
                for jk, ok in ctrl_map.items():
                    if ok:
                        if jk not in _checker_positive_data.get(inst, {}):
                            _checker_positive_data.setdefault(inst, {})[jk] = True
                            _checker_negative_data.get(inst, {}).pop(jk, None)
                            migrated += 1
                    else:
                        if jk not in _checker_negative_data.get(inst, {}):
                            _checker_negative_data.setdefault(inst, {})[jk] = -1
                            _checker_positive_data.get(inst, {}).pop(jk, None)
                            migrated += 1
            if migrated > 0 or pos_normalized or neg_normalized:
                _safe_flush_checker_json(positive_path, _checker_positive_data)
                _safe_flush_checker_json(negative_path, _checker_negative_data)
    elif pos_normalized or neg_normalized:
        if pos_normalized:
            _safe_flush_checker_json(positive_path, _checker_positive_data)
        if neg_normalized:
            _safe_flush_checker_json(negative_path, _checker_negative_data)

    pos_count = _merge_cache_into_result_cache(_checker_positive_data, True)
    neg_count = _merge_cache_into_result_cache(_checker_negative_data, False)
    return pos_count, neg_count, migrated


def update_checker_jsons(
    positive_path: str,
    negative_path: str,
    inst: str,
    ctrl: Control,
    ok: bool,
    counterexample_len: Optional[int] = None,
) -> None:
    """Thread-safe: add one result to the correct cache and flush to disk."""
    jk = _ctrl_json_key(ctrl)
    with _checker_lock:
        if ok:
            removed = jk in _checker_negative_data.get(inst, {})
            _checker_positive_data.setdefault(inst, {})[jk] = True
            _checker_negative_data.get(inst, {}).pop(jk, None)
            _safe_flush_checker_json(positive_path, _checker_positive_data)
            if removed:
                _safe_flush_checker_json(negative_path, _checker_negative_data)
        else:
            neg_len = _normalize_negative_len(counterexample_len)
            existing = _checker_negative_data.get(inst, {}).get(jk)
            # Do not degrade known loop lengths to unknown (-1).
            if existing is None or (existing == -1 and neg_len > 0) or (existing > 0 and neg_len > 0 and existing != neg_len):
                _checker_negative_data.setdefault(inst, {})[jk] = neg_len
            removed = jk in _checker_positive_data.get(inst, {})
            _checker_positive_data.get(inst, {}).pop(jk, None)
            _safe_flush_checker_json(negative_path, _checker_negative_data)
            if removed:
                _safe_flush_checker_json(positive_path, _checker_positive_data)


def extract_from_log(log_path: str, positive_path: str, negative_path: str) -> int:
    """Parse an existing log file and bootstrap / update checker caches.

    Lines handled:
        work_dir,inst,{ctrl_dict},OK
        work_dir,inst,{ctrl_dict},INCORRECT

    All other lines (comments, MISSING, DONE, LOOP_LEN) are silently skipped.
    Returns the number of *new* entries added.
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
            # work_dir may contain backslashes but never commas, so split on
            # the first two commas to isolate inst and the rest of the line.
            parts = line.split(",", 2)
            if len(parts) < 3:
                continue
            inst = parts[1]
            rest = parts[2]
            m = re.match(r"(\{[^}]*\}),(OK|INCORRECT)(?:,(.*))?$", rest)
            if not m:
                continue
            ctrl_str, status, tail = m.group(1), m.group(2), (m.group(3) or "")
            try:
                ctrl_dict = ast.literal_eval(ctrl_str)
            except (ValueError, SyntaxError):
                continue
            jk = json.dumps(sorted(ctrl_dict.items()))
            if status == "OK":
                if jk not in _checker_positive_data.get(inst, {}):
                    _checker_positive_data.setdefault(inst, {})[jk] = True
                    _checker_negative_data.get(inst, {}).pop(jk, None)
                    _result_cache[_json_key_to_cache_key(inst, jk)] = True
                    count += 1
                    changed = True
            else:
                neg_len = -1
                tail_parts = tail.split(",") if tail else []
                if len(tail_parts) >= 2 and tail_parts[0] == "LOOP_LEN":
                    neg_len = _normalize_negative_len(tail_parts[1])
                old_len = _checker_negative_data.get(inst, {}).get(jk)
                should_update = (
                    old_len is None
                    or (old_len == -1 and neg_len > 0)
                    or (old_len > 0 and neg_len > 0 and old_len != neg_len)
                )
                if should_update:
                    _checker_negative_data.setdefault(inst, {})[jk] = neg_len
                    if old_len is None:
                        count += 1
                    changed = True
                if jk in _checker_positive_data.get(inst, {}):
                    _checker_positive_data[inst].pop(jk, None)
                    changed = True
                _result_cache[_json_key_to_cache_key(inst, jk)] = False
    if changed:
        _safe_flush_checker_json(positive_path, _checker_positive_data)
        _safe_flush_checker_json(negative_path, _checker_negative_data)
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
    ok: bool,
    loop_len: Optional[int] = None,
    logic: str = "ctl",
):
    if ok:
        if logic == "ltl":
            line = f"{work_dir},{inst},{ctrl},OK,LOOP_LEN,{loop_len}"
        else:
            line = f"{work_dir},{inst},{ctrl},OK"
    else:
        tag = os.path.basename(work_dir)
        if logic == "ltl":
            print(f"\tincorrect [{tag}/{inst}] {ctrl} loop={loop_len}")
            line = f"{work_dir},{inst},{ctrl},INCORRECT,LOOP_LEN,{loop_len}"
        else:
            print(f"\tincorrect [{tag}/{inst}] {ctrl}")
            line = f"{work_dir},{inst},{ctrl},INCORRECT"
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
    """Read sol.json for each instance and return (inst, ctrl) pairs."""
    pairs: List[Tuple[str, Control]] = []
    for inst in instances:
        sol_path = _find_sol_path(work_dir, inst)
        if sol_path is None:
            print(f"\t{os.path.basename(work_dir)}/{inst}: sol.json not found, skipping")
            with open(output_file, "a", encoding="utf-8") as _f:
                _f.write(f"{work_dir},{inst},MISSING\n")
            continue
        ctrl_list: List[Control] = []
        with open(sol_path, "r") as _f:
            for sol_list in json.load(_f).values():
                for sol in sol_list:
                    ctrl_list.append(Control(sol))
        pairs.extend((inst, ctrl) for ctrl in ctrl_list)
        print(f"\t{os.path.basename(work_dir)}/{inst}: {len(ctrl_list)} controls")
    return pairs


# ---------------------------------------------------------------------------
# Core — single pool across all work_dirs
# ---------------------------------------------------------------------------


def verify_all(
    work_dir_list: List[str],
    output_file: str,
    instances: List[str],
    workers: int,
    json_positive_path: str,
    json_negative_path: str,
    logic: str = "ctl",
) -> List[Tuple[str, str, Control]]:
    """
    Verify controls from all work_dirs using a shared worker pool.

    Controls are processed instance-by-instance across experiments to maximize
    worker-side reuse for the same instance before moving to the next one.

    Returns a list of (work_dir, inst, ctrl) for every control that failed
    the phenotype check — pass these to get_loop_lengths() for LTL analysis.
    """
    # Phase 1: collect all triplets upfront (sol.json reads are fast).
    # wdi = (work_dir, inst) key used for per-instance progress and DONE logging.
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
        return []

    # Phase 2: split cached vs to-run.
    # In LTL mode, recompute all controls so LOOP_LEN is always logged in output.
    to_run: List[Tuple[str, str, Control, tuple]] = []
    for work_dir, inst, ctrl in all_triplets:
        key = (inst, _ctrl_key(ctrl))
        if logic == "ltl" or key not in _result_cache:
            to_run.append((work_dir, inst, ctrl, key))

    n_cached = total - len(to_run)
    n_wds = len({wd for wd, _, _ in all_triplets})
    print(
        f"Total: {total} checks ({n_cached} cached, {len(to_run)} to compute)"
        f" across {n_wds} experiment(s), {workers} workers [{logic.upper()}]"
    )

    incorrect: List[Tuple[str, str, Control]] = []

    # Flush cached results.
    if logic == "ctl":
        for work_dir, inst, ctrl in all_triplets:
            key = (inst, _ctrl_key(ctrl))
            if key in _result_cache:
                ok = _result_cache[key]
                _log_result(output_file, work_dir, inst, ctrl, ok, logic=logic)
                if not ok:
                    incorrect.append((work_dir, inst, ctrl))

    # Mark (work_dir, inst) pairs that are fully covered by cache as done now.
    wdi_to_run_count: Dict[Tuple[str, str], int] = {}
    for work_dir, inst, ctrl, key in to_run:
        wdi = (work_dir, inst)
        wdi_to_run_count[wdi] = wdi_to_run_count.get(wdi, 0) + 1
    for wdi in wdi_counts:
        if wdi not in wdi_to_run_count:
            _log_instance_done(output_file, wdi[0], wdi[1])

    if not to_run:
        return incorrect

    # Phase 3: run non-cached triplets grouped by instance.
    inst_to_run: Dict[str, List[Tuple[str, str, Control, tuple]]] = {}
    for triplet in to_run:
        _, inst, _, _ = triplet
        inst_to_run.setdefault(inst, []).append(triplet)

    wdi_completed: Dict[Tuple[str, str], int] = {wdi: 0 for wdi in wdi_to_run_count}
    last_pct_by_wdi: Dict[Tuple[str, str], int] = {wdi: 0 for wdi in wdi_to_run_count}

    def _process_inst_triplets(
        executor: ProcessPoolExecutor,
        inst: str,
        inst_triplets: List[Tuple[str, str, Control, tuple]],
    ) -> None:
        if not inst_triplets:
            return
        print(
            f"[{inst}] processing {len(inst_triplets)} control(s) "
            f"across {len({wd for wd, _, _, _ in inst_triplets})} experiment(s)"
        )
        if logic == "ltl":
            futures = {
                executor.submit(_check_ctrl_ltl_for_inst, inst, ctrl): (
                    work_dir,
                    inst,
                    ctrl,
                    key,
                )
                for work_dir, inst, ctrl, key in inst_triplets
            }
        else:
            futures = {
                executor.submit(_check_ctrl_for_inst, inst, ctrl): (
                    work_dir,
                    inst,
                    ctrl,
                    key,
                )
                for work_dir, inst, ctrl, key in inst_triplets
            }
        for future in as_completed(futures):
            work_dir, inst, ctrl, key = futures[future]
            if logic == "ltl":
                ok, loop_len = future.result()
            else:
                ok = future.result()
                loop_len = None

            _result_cache[key] = ok
            update_checker_jsons(
                json_positive_path,
                json_negative_path,
                inst,
                ctrl,
                ok,
                counterexample_len=(loop_len if logic == "ltl" and not ok else None),
            )
            _log_result(
                output_file,
                work_dir,
                inst,
                ctrl,
                ok,
                loop_len=loop_len,
                logic=logic,
            )
            if not ok:
                incorrect.append((work_dir, inst, ctrl))

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

    with ProcessPoolExecutor(max_workers=workers) as executor:
        instance_set = set(instances)
        for inst in instances:
            _process_inst_triplets(executor, inst, inst_to_run.get(inst, []))

        # Handle any instances not listed in --instances but present in collected data.
        remaining_insts = sorted(inst for inst in inst_to_run.keys() if inst not in instance_set)
        for inst in remaining_insts:
            _process_inst_triplets(executor, inst, inst_to_run[inst])

    return incorrect


# ---------------------------------------------------------------------------
# LTL counterexample analysis (separate pass on incorrect controls)
# ---------------------------------------------------------------------------


def get_loop_lengths(
    incorrect_pairs: List[Tuple[str, str, Control]],
    output_file: str,
    workers: int,
) -> None:
    """
    Run LTL model checking on a list of incorrect (work_dir, inst, ctrl) pairs
    to determine the attractor cycle length from the counterexample trace.

    Results are appended to output_file as:
        work_dir,inst,ctrl,LOOP_LEN,<n>

    Intended to be called after verify_all() on the returned incorrect list.
    """
    if not incorrect_pairs:
        return
    print(f"LTL counterexample check: {len(incorrect_pairs)} controls, {workers} workers")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_get_loop_len_for_inst, inst, ctrl): (work_dir, inst, ctrl)
            for work_dir, inst, ctrl in incorrect_pairs
        }
        for future in as_completed(futures):
            work_dir, inst, ctrl = futures[future]
            loop_len = future.result()
            tag = os.path.basename(work_dir)
            print(f"\t[{tag}/{inst}] {ctrl} -> loop={loop_len}")
            with open(output_file, "a", encoding="utf-8") as _f:
                _f.write(f"{work_dir},{inst},{ctrl},LOOP_LEN,{loop_len}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=(
            "Verify controls from sol.json in parallel across all experiments: "
            "check all attractors satisfy the phenotype."
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
        "--output",
        metavar="FILE",
        default="_experiments/verify_control_log.txt",
        help="Output file to write results to (default: _experiments/verify_control_log.txt)",
    )
    ap.add_argument(
        "--summary-output",
        metavar="FILE",
        default=None,
        help=(
            "Output JSON path for summary report "
            "(default: {root_dir}/_results/verify_control_Tinf.json)."
        ),
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        metavar="N",
        help=f"Number of parallel NuSMV processes (default: cpu_count={os.cpu_count()})",
    )
    ap.add_argument(
        "--json-positive",
        metavar="FILE",
        default="_experiments/checker_positive.json",
        help=(
            "Persistent JSON cache for positive checks "
            "(default: _experiments/checker_positive.json)"
        ),
    )
    ap.add_argument(
        "--json-negative",
        metavar="FILE",
        default="_experiments/checker_negative.json",
        help=(
            "Persistent JSON cache for negative checks "
            "(value = counterexample loop length; CTL stores -1) "
            "(default: _experiments/checker_negative.json)"
        ),
    )
    ap.add_argument(
        "--json-legacy",
        metavar="FILE",
        default="_experiments/checker.json",
        help=(
            "Optional legacy checker cache to migrate from "
            "(default: _experiments/checker.json)"
        ),
    )
    ap.add_argument(
        "--logic",
        choices=["ctl", "ltl"],
        default="ctl",
        help="Model-checking logic to run (default: ctl). In ltl mode, LOOP_LEN is logged.",
    )
    args = ap.parse_args()

    n_pos, n_neg, n_migrated = load_checker_jsons(
        args.json_positive,
        args.json_negative,
        legacy_path=args.json_legacy,
    )
    if n_migrated:
        print(
            f"Migrated {n_migrated} cached result(s) from {args.json_legacy} "
            f"to {args.json_positive} / {args.json_negative}"
        )
    loaded_total = n_pos + n_neg
    if loaded_total:
        print(
            f"Loaded {loaded_total} cached result(s): "
            f"{n_pos} positive, {n_neg} negative"
        )

    _timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _dirs_str = ", ".join(args.work_dirs) if args.work_dirs else f"root={args.root_dir}"
    _insts_str = ", ".join(args.instances)
    with open(args.output, "a", encoding="utf-8") as _f:
        _f.write(
            f"# [{_timestamp}] work_dirs=[{_dirs_str}] instances=[{_insts_str}]"
            f" workers={args.workers} logic={args.logic}\n"
        )

    if args.work_dirs:
        work_dir_list = args.work_dirs
        parent_dirs = [
            os.path.abspath(os.path.dirname(os.path.normpath(wd)))
            for wd in work_dir_list
        ]
        results_root = os.path.commonpath(parent_dirs) if parent_dirs else "."
    else:
        root_dir = args.root_dir
        results_root = root_dir
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

    all_incorrect = verify_all(
        work_dir_list,
        args.output,
        args.instances,
        args.workers,
        args.json_positive,
        args.json_negative,
        logic=args.logic,
    )

    err_counts_by_exp: Dict[str, Dict[str, int]] = {}
    for work_dir in work_dir_list:
        exp = os.path.basename(os.path.normpath(work_dir))
        err_counts_by_exp.setdefault(exp, {"incorrect": 0, "nonminimal": 0})
    for work_dir, _, _ in all_incorrect:
        exp = os.path.basename(os.path.normpath(work_dir))
        err_counts_by_exp.setdefault(exp, {"incorrect": 0, "nonminimal": 0})
        err_counts_by_exp[exp]["incorrect"] += 1

    verify_summary_path = (
        args.summary_output
        if args.summary_output is not None
        else os.path.join(results_root, "_results", "verify_control_Tinf.json")
    )
    os.makedirs(os.path.dirname(verify_summary_path), exist_ok=True)
    sorted_counts = {
        exp: err_counts_by_exp[exp]
        for exp in sorted(err_counts_by_exp.keys())
    }
    summary_obj = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "root_dir": results_root,
        "T": "infinite",
        "logic": args.logic,
        "instances": args.instances,
        "experiments": sorted_counts,
        "summary": {
            "experiments": len(sorted_counts),
            "incorrect_total": sum(v["incorrect"] for v in sorted_counts.values()),
            "nonminimal_total": sum(v["nonminimal"] for v in sorted_counts.values()),
        },
    }
    with open(verify_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_obj, f, indent=2)
    print(f"Wrote verify summary: {verify_summary_path}")

    # Call get_loop_lengths(all_incorrect, args.output, args.workers) here
    # to run LTL counterexample analysis on the incorrect controls.
