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
# Persistent JSON cache  {inst -> {ctrl_json_key -> ok}}
#
# _checker_data mirrors the on-disk JSON so we never re-read the file on
# every update.  All writes go through _flush_checker_json which does an
# atomic rename, so a killed process cannot corrupt the file.
# _checker_lock serialises updates in the (single) main process.
# ---------------------------------------------------------------------------

_checker_data: Dict[str, Dict[str, bool]] = {}
_checker_lock = threading.Lock()


def _ctrl_json_key(ctrl: Control) -> str:
    """Canonical JSON string key for a Control."""
    return json.dumps(sorted(ctrl.items()))


def _json_key_to_cache_key(inst: str, json_key: str) -> tuple:
    return (inst, tuple((k, v) for k, v in json.loads(json_key)))


def _flush_checker_json(json_path: str) -> None:
    """Atomically overwrite json_path from _checker_data."""
    tmp = json_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_checker_data, f, indent=2)
    os.replace(tmp, json_path)


def load_checker_json(json_path: str) -> int:
    """Load checker.json into _checker_data and populate _result_cache.

    Returns the number of entries loaded.
    """
    global _checker_data
    if not os.path.exists(json_path):
        return 0
    with open(json_path, "r", encoding="utf-8") as f:
        _checker_data = json.load(f)
    count = 0
    for inst, ctrl_map in _checker_data.items():
        for jk, ok in ctrl_map.items():
            _result_cache[_json_key_to_cache_key(inst, jk)] = ok
            count += 1
    return count


def update_checker_json(json_path: str, inst: str, ctrl: Control, ok: bool) -> None:
    """Thread-safe: add one result to _checker_data and flush to disk."""
    jk = _ctrl_json_key(ctrl)
    with _checker_lock:
        _checker_data.setdefault(inst, {})[jk] = ok
        _flush_checker_json(json_path)


def extract_from_log(log_path: str, json_path: str) -> int:
    """Parse an existing log file and bootstrap / update checker.json.

    Lines handled:
        work_dir,inst,{ctrl_dict},OK
        work_dir,inst,{ctrl_dict},INCORRECT

    All other lines (comments, MISSING, DONE, LOOP_LEN) are silently skipped.
    Returns the number of *new* entries added.
    """
    if not os.path.exists(log_path):
        return 0
    count = 0
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
            m = re.match(r"(\{[^}]*\}),(OK|INCORRECT)(?:,|$)", rest)
            if not m:
                continue
            ctrl_str, status = m.group(1), m.group(2)
            try:
                ctrl_dict = ast.literal_eval(ctrl_str)
            except (ValueError, SyntaxError):
                continue
            jk = json.dumps(sorted(ctrl_dict.items()))
            if jk not in _checker_data.get(inst, {}):
                _checker_data.setdefault(inst, {})[jk] = (status == "OK")
                # also update _result_cache so duplicates within this run are skipped
                _result_cache[_json_key_to_cache_key(inst, jk)] = (status == "OK")
                count += 1
    if count:
        _flush_checker_json(json_path)
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
    json_path: str,
    logic: str = "ctl",
) -> List[Tuple[str, str, Control]]:
    """
    Verify controls from all work_dirs using a single shared worker pool.

    All (work_dir, inst, ctrl) triplets from every experiment are submitted at
    once so workers are never idle waiting for one slow experiment to finish
    before the next starts.

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
    # In LTL mode, recompute all controls because boolean cache does not store
    # loop lengths and we want LOOP_LEN logged in the output.
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

    # Phase 3: single pool for all non-cached triplets across all experiments.
    wdi_completed: Dict[Tuple[str, str], int] = {wdi: 0 for wdi in wdi_to_run_count}
    last_pct_by_wdi: Dict[Tuple[str, str], int] = {wdi: 0 for wdi in wdi_to_run_count}

    with ProcessPoolExecutor(max_workers=workers) as executor:
        if logic == "ltl":
            futures = {
                executor.submit(_check_ctrl_ltl_for_inst, inst, ctrl): (
                    work_dir,
                    inst,
                    ctrl,
                    key,
                )
                for work_dir, inst, ctrl, key in to_run
            }
        else:
            futures = {
                executor.submit(_check_ctrl_for_inst, inst, ctrl): (work_dir, inst, ctrl, key)
                for work_dir, inst, ctrl, key in to_run
            }
        for future in as_completed(futures):
            work_dir, inst, ctrl, key = futures[future]
            if logic == "ltl":
                ok, loop_len = future.result()
            else:
                ok = future.result()
                loop_len = None

            _result_cache[key] = ok
            update_checker_json(json_path, inst, ctrl, ok)
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
        "--workers",
        type=int,
        default=os.cpu_count(),
        metavar="N",
        help=f"Number of parallel NuSMV processes (default: cpu_count={os.cpu_count()})",
    )
    ap.add_argument(
        "--json",
        metavar="FILE",
        default="_experiments/checker.json",
        help="Persistent JSON cache for check results (default: _experiments/checker.json)",
    )
    ap.add_argument(
        "--logic",
        choices=["ctl", "ltl"],
        default="ctl",
        help="Model-checking logic to run (default: ctl). In ltl mode, LOOP_LEN is logged.",
    )
    args = ap.parse_args()

    n = load_checker_json(args.json)
    if n:
        print(f"Loaded {n} cached result(s) from {args.json}")

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

    all_incorrect = verify_all(
        work_dir_list,
        args.output,
        args.instances,
        args.workers,
        args.json,
        logic=args.logic,
    )

    # Call get_loop_lengths(all_incorrect, args.output, args.workers) here
    # to run LTL counterexample analysis on the incorrect controls.
