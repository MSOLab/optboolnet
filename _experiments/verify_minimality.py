import argparse
import datetime
import json
import os
import re
from typing import Dict, List, Optional, Tuple

from optboolnet.boolnet import Control
from optboolnet.config import SolverConfig
from optboolnet.instances import _INSTANCE_LIST_FULL, load_bn_in_repo
from optboolnet.model import AggregatedAttractorDetectionIP
from pyomo.opt import TerminationCondition


_ALGO_SUBDIRS = ["benders", "MibS", "pbn"]


def _ctrl_key(ctrl: Dict[str, int]) -> Tuple[Tuple[str, int], ...]:
    return tuple(sorted((k, int(v)) for k, v in ctrl.items()))


def _normalize_ctrl(ctrl: dict) -> Dict[str, int]:
    return {str(k): int(v) for k, v in ctrl.items()}


def _is_subset(sub: Dict[str, int], sup: Dict[str, int], strict: bool = False) -> bool:
    if strict and len(sub) >= len(sup):
        return False
    return all(sup.get(k, None) == v for k, v in sub.items())


def _drop_nonminimal(ctrls: List[Dict[str, int]]) -> List[Dict[str, int]]:
    """Same semantics as CtrlResult.drop_nonminimal in _experiments/control.py."""
    ordered = sorted((_normalize_ctrl(c) for c in ctrls), key=lambda d: (len(d), sorted(d.items())))
    out: List[Dict[str, int]] = []
    for ctrl in ordered:
        if not any(_is_subset(other, ctrl, strict=False) for other in out):
            out.append(ctrl)
    return out


def _parse_ctrl_json_key(jk: str) -> Optional[Dict[str, int]]:
    try:
        items = json.loads(jk)
    except json.JSONDecodeError:
        return None
    if not isinstance(items, list):
        return None
    ctrl: Dict[str, int] = {}
    for pair in items:
        if not (isinstance(pair, list) and len(pair) == 2):
            return None
        k, v = pair
        ctrl[str(k)] = int(v)
    return ctrl


def _load_bounded_cache(path: str) -> Dict[str, Dict[str, int]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        return {}

    out: Dict[str, Dict[str, int]] = {}
    for inst, ctrl_map in raw.items():
        if not isinstance(ctrl_map, dict):
            continue
        for jk, raw_len in ctrl_map.items():
            try:
                ln = int(raw_len)
            except (TypeError, ValueError):
                continue
            out.setdefault(str(inst), {})[jk] = ln if ln > 0 else -1
    return out


def _find_sol_path(work_dir: str, inst: str) -> Optional[str]:
    for subdir in _ALGO_SUBDIRS:
        path = os.path.join(work_dir, subdir, inst, "sol.json")
        if os.path.exists(path):
            return path
    return None


def _load_controls_from_sol(sol_path: str) -> List[Dict[str, int]]:
    with open(sol_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: List[Dict[str, int]] = []
    if not isinstance(data, dict):
        return out
    for sol_list in data.values():
        if not isinstance(sol_list, list):
            continue
        for sol in sol_list:
            if isinstance(sol, dict):
                out.append(_normalize_ctrl(sol))
    return out


def _load_max_control_size(work_dir: str) -> Optional[int]:
    config_path = os.path.join(work_dir, "alg_config.json")
    if not os.path.exists(config_path):
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    try:
        return int(cfg["max_control_size"])
    except (KeyError, TypeError, ValueError):
        return None


def _load_max_length(work_dir: str) -> Optional[int]:
    config_path = os.path.join(work_dir, "alg_config.json")
    if not os.path.exists(config_path):
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    try:
        return int(cfg["max_length"])
    except (KeyError, TypeError, ValueError):
        return None


def _parse_cache_horizon(path: str) -> Optional[int]:
    m = re.search(r"_T(\d+)\.json$", path)
    if m:
        return int(m.group(1))
    return None


def _is_valid_for_max_length(min_viol_len: int, max_length: int) -> bool:
    # min_viol_len is first violating attractor length.
    # Valid up to max_length iff no violation exists in [1..max_length].
    return min_viol_len == -1 or min_viol_len > max_length


def _build_minimal_controls_by_setting(
    cache_data: Dict[str, Dict[str, int]],
    settings: List[Tuple[str, int, int]],
) -> Tuple[
    Dict[Tuple[str, int, int], List[Dict[str, int]]],
    Dict[Tuple[str, int, int], Dict[Tuple[Tuple[str, int], ...], int]],
]:
    unique_settings = sorted(set(settings), key=lambda x: (x[0], x[1], x[2]))
    out: Dict[Tuple[str, int, int], List[Dict[str, int]]] = {}
    out_len_by_key: Dict[
        Tuple[str, int, int], Dict[Tuple[Tuple[str, int], ...], int]
    ] = {}
    for inst, max_control_size, max_length in unique_settings:
        candidates: List[Dict[str, int]] = []
        candidate_len_by_key: Dict[Tuple[Tuple[str, int], ...], int] = {}
        for jk, min_viol_len in cache_data.get(inst, {}).items():
            if _is_valid_for_max_length(min_viol_len, max_length):
                ctrl = _parse_ctrl_json_key(jk)
                if ctrl is not None:
                    candidates.append(ctrl)
                    candidate_len_by_key[_ctrl_key(ctrl)] = int(min_viol_len)
        minimal_ctrls = _drop_nonminimal(candidates)
        out[(inst, max_control_size, max_length)] = minimal_ctrls
        out_len_by_key[(inst, max_control_size, max_length)] = {
            _ctrl_key(ctrl): candidate_len_by_key[_ctrl_key(ctrl)]
            for ctrl in minimal_ctrls
            if _ctrl_key(ctrl) in candidate_len_by_key
        }
    return out, out_len_by_key


def _find_subset_witnesses(
    ctrl: Dict[str, int],
    minimal_set: List[Dict[str, int]],
) -> List[Dict[str, int]]:
    out = [m for m in minimal_set if _is_subset(m, ctrl, strict=True)]
    out.sort(key=lambda d: (len(d), sorted(d.items())))
    return out


def _detect_work_dirs_from_root(root_dir: str) -> List[str]:
    return [
        os.path.join(root_dir, sub)
        for sub in sorted(os.listdir(root_dir))
        if any(os.path.isdir(os.path.join(root_dir, sub, algo)) for algo in _ALGO_SUBDIRS)
    ]


def _default_nonminimal_output_path(
    root_dir: Optional[str],
    work_dirs: List[str],
) -> str:
    if root_dir:
        base_root = root_dir
    elif work_dirs:
        parent_dirs = [
            os.path.abspath(os.path.dirname(os.path.normpath(wd))) for wd in work_dirs
        ]
        base_root = os.path.commonpath(parent_dirs) if parent_dirs else "."
    else:
        base_root = "."
    return os.path.join(
        base_root,
        "_results",
        "verify_minimality_nonminimal.json",
    )


def _normalize_len(value: object) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        iv = int(value)
    except (TypeError, ValueError):
        return None
    return iv if iv > 0 else -1


def _is_strict_subset(sub: Dict[str, int], sup: Dict[str, int]) -> bool:
    if len(sub) >= len(sup):
        return False
    return all(sup.get(k, None) == v for k, v in sub.items())


def _iter_entry_witnesses(entry: Dict[str, object]) -> List[Tuple[Dict[str, int], Optional[int]]]:
    """Return [(witness_control, expected_min_viol_len), ...] from an entry."""
    out: List[Tuple[Dict[str, int], Optional[int]]] = []
    witness_list = entry.get("witness_minimals")
    if isinstance(witness_list, list):
        for item in witness_list:
            if not isinstance(item, dict):
                continue
            ctrl = item.get("control")
            if not isinstance(ctrl, dict):
                continue
            out.append(
                (
                    {str(k): int(v) for k, v in ctrl.items()},
                    _normalize_len(item.get("min_viol_len")),
                )
            )
    return out


class SequentialBoundedVerifier:
    def __init__(self, T: int) -> None:
        self.T = T
        self._bn_by_inst: Dict[str, object] = {}
        self._model_by_inst_len: Dict[Tuple[str, int], AggregatedAttractorDetectionIP] = {}

    def _get_model(self, inst: str, length: int) -> AggregatedAttractorDetectionIP:
        key = (inst, length)
        if key in self._model_by_inst_len:
            return self._model_by_inst_len[key]

        if inst not in self._bn_by_inst:
            self._bn_by_inst[inst] = load_bn_in_repo(inst)
        bn = self._bn_by_inst[inst]

        model = AggregatedAttractorDetectionIP(
            f"{inst}_{length}",
            bn,
            length,
            SolverConfig(),
        )
        model.make_constr_stability_condition()
        model.make_constr_periodicity()
        model.make_constr_phenotype_and_length()
        model.set_phenotype_obj()
        model.fix_length(length)
        self._model_by_inst_len[key] = model
        return model

    def _is_violating(self, model: AggregatedAttractorDetectionIP) -> bool:
        if model.optimize():
            return bool(model.p.value < 0.5)

        term = getattr(model, "last_termination_condition", None)
        if term == TerminationCondition.infeasible:
            return False
        if term == TerminationCondition.maxTimeLimit:
            return True
        return True

    def find_min_viol_len(self, inst: str, ctrl_dict: Dict[str, int]) -> int:
        ctrl = Control({str(k): int(v) for k, v in ctrl_dict.items()})
        for length in range(1, self.T + 1):
            model = self._get_model(inst, length)
            model.fix_control(ctrl)
            if self._is_violating(model):
                return model.get_attractor().get_length()
        return -1


def build_witness_report(data: Dict[str, object], T: int) -> Dict[str, object]:
    entries: List[Dict[str, object]] = list(data.get("non_minimal_controls", []))
    cache_path = str(data.get("json_cache", ""))

    # Deduplicate witness solves by (instance, witness control)
    unique: Dict[Tuple[str, Tuple[Tuple[str, int], ...]], Dict[str, object]] = {}
    for idx, e in enumerate(entries):
        inst = str(e.get("instance"))
        for witness_ctrl, expected in _iter_entry_witnesses(e):
            key = (inst, _ctrl_key(witness_ctrl))
            if key not in unique:
                unique[key] = {
                    "instance": inst,
                    "witness_minimal": witness_ctrl,
                    "expected_values": set(),
                    "entry_indices": [],
                }
            if expected is not None:
                unique[key]["expected_values"].add(expected)  # type: ignore[index]
            unique[key]["entry_indices"].append(idx)  # type: ignore[index]

    verifier = SequentialBoundedVerifier(T=T)
    n_unique = len(unique)
    print(f"Rechecking {n_unique} unique witness controls (single-thread), T={T}")

    solved: Dict[Tuple[str, Tuple[Tuple[str, int], ...]], int] = {}
    mismatches: List[Dict[str, object]] = []
    missing_expected: List[Dict[str, object]] = []
    inconsistent_expected: List[Dict[str, object]] = []
    entries_over_horizon: List[Dict[str, object]] = []
    witness_inconsistent_with_max_length: List[Dict[str, object]] = []
    witness_not_in_minimal_set: List[Dict[str, object]] = []
    witness_not_strict_subset_of_control: List[Dict[str, object]] = []
    entries_without_witness: List[Dict[str, object]] = []

    done = 0
    for key, meta in unique.items():
        inst = str(meta["instance"])
        witness = dict(meta["witness_minimal"])  # type: ignore[arg-type]
        expected_values = sorted(meta["expected_values"])  # type: ignore[arg-type]

        if len(expected_values) == 0:
            missing_expected.append(
                {
                    "instance": inst,
                    "witness_minimal": dict(sorted(witness.items())),
                    "entry_indices": list(meta["entry_indices"]),  # type: ignore[arg-type]
                }
            )
        if len(expected_values) > 1:
            inconsistent_expected.append(
                {
                    "instance": inst,
                    "witness_minimal": dict(sorted(witness.items())),
                    "expected_values": expected_values,
                    "entry_indices": list(meta["entry_indices"]),  # type: ignore[arg-type]
                }
            )

        computed = verifier.find_min_viol_len(inst, witness)
        solved[key] = computed

        if len(expected_values) == 1 and computed != expected_values[0]:
            mismatches.append(
                {
                    "instance": inst,
                    "witness_minimal": dict(sorted(witness.items())),
                    "expected": expected_values[0],
                    "computed": computed,
                    "entry_indices": list(meta["entry_indices"]),  # type: ignore[arg-type]
                }
            )

        done += 1
        if done % 10 == 0 or done == n_unique:
            print(f"  progress: {done}/{n_unique}")

    # Per-entry consistency with max_length and witness structural checks
    minimal_by_setting: Dict[str, List[Dict[str, int]]] = data.get("minimal_controls_by_setting", {})  # type: ignore[assignment]
    for idx, e in enumerate(entries):
        inst = str(e.get("instance"))
        control = e.get("control")
        max_control_size_raw = e.get("max_control_size")
        max_length_raw = e.get("max_length")
        if not isinstance(control, dict):
            continue
        try:
            max_control_size = int(max_control_size_raw)
        except (TypeError, ValueError):
            max_control_size = None
        max_length = _normalize_len(max_length_raw)
        if max_length is None or max_length < 1:
            continue
        control_norm = {str(k): int(v) for k, v in control.items()}

        witness_list = _iter_entry_witnesses(e)
        if len(witness_list) == 0:
            entries_without_witness.append(
                {
                    "entry_index": idx,
                    "instance": inst,
                    "control": dict(sorted(control_norm.items())),
                }
            )
            continue

        setting_key_new = (
            f"{inst}|{max_control_size}|{max_length}"
            if max_control_size is not None
            else None
        )
        setting_key_old = (
            f"{inst}|{max_control_size}" if max_control_size is not None else None
        )
        minimal_setting: List[Dict[str, int]] = []
        if setting_key_new and setting_key_new in minimal_by_setting:
            minimal_setting = minimal_by_setting[setting_key_new]
        elif setting_key_old and setting_key_old in minimal_by_setting:
            minimal_setting = minimal_by_setting[setting_key_old]
        minimal_keys = {
            _ctrl_key({str(k): int(v) for k, v in c.items()})
            for c in minimal_setting
            if isinstance(c, dict)
        }

        if max_length > T:
            entries_over_horizon.append(
                {
                    "entry_index": idx,
                    "instance": inst,
                    "max_length": max_length,
                    "T": T,
                }
            )
            continue

        for witness_idx, (witness_norm, _) in enumerate(witness_list):
            if not _is_strict_subset(witness_norm, control_norm):
                witness_not_strict_subset_of_control.append(
                    {
                        "entry_index": idx,
                        "witness_index": witness_idx,
                        "instance": inst,
                        "control": dict(sorted(control_norm.items())),
                        "witness_minimal": dict(sorted(witness_norm.items())),
                    }
                )

            if _ctrl_key(witness_norm) not in minimal_keys:
                witness_not_in_minimal_set.append(
                    {
                        "entry_index": idx,
                        "witness_index": witness_idx,
                        "instance": inst,
                        "max_control_size": max_control_size,
                        "max_length": max_length,
                        "witness_minimal": dict(sorted(witness_norm.items())),
                        "setting_key_checked": setting_key_new or setting_key_old,
                    }
                )

            solved_key = (inst, _ctrl_key(witness_norm))
            computed = solved.get(solved_key)
            if computed is None:
                continue
            if not _is_valid_for_max_length(computed, max_length):
                witness_inconsistent_with_max_length.append(
                    {
                        "entry_index": idx,
                        "witness_index": witness_idx,
                        "instance": inst,
                        "max_length": max_length,
                        "witness_minimal": dict(sorted(witness_norm.items())),
                        "computed_witness_min_viol_len": computed,
                        "expected_valid_condition": "min_viol_len == -1 or min_viol_len > max_length",
                    }
                )

    all_correct = (
        len(mismatches) == 0
        and len(missing_expected) == 0
        and len(inconsistent_expected) == 0
    )
    max_length_consistent = (
        len(witness_inconsistent_with_max_length) == 0 and len(entries_over_horizon) == 0
    )
    witness_correct = (
        len(witness_not_in_minimal_set) == 0
        and len(witness_not_strict_subset_of_control) == 0
        and len(entries_without_witness) == 0
    )

    report = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "input": None,  # filled by caller
        "json_cache": cache_path,
        "T": T,
        "summary": {
            "entries_total": len(entries),
            "unique_witness_controls": n_unique,
            "mismatch_count": len(mismatches),
            "missing_expected_count": len(missing_expected),
            "inconsistent_expected_count": len(inconsistent_expected),
            "all_witness_minimal_min_viol_len_correct": all_correct,
            "entries_over_horizon_count": len(entries_over_horizon),
            "witness_inconsistent_with_max_length_count": len(
                witness_inconsistent_with_max_length
            ),
            "all_witness_consistent_with_max_length": max_length_consistent,
            "witness_not_in_minimal_set_count": len(witness_not_in_minimal_set),
            "witness_not_strict_subset_count": len(witness_not_strict_subset_of_control),
            "entries_without_witness_count": len(entries_without_witness),
            "all_witness_correct": witness_correct,
        },
        "mismatches": mismatches,
        "missing_expected": missing_expected,
        "inconsistent_expected": inconsistent_expected,
        "entries_over_horizon": entries_over_horizon,
        "witness_inconsistent_with_max_length": witness_inconsistent_with_max_length,
        "witness_not_in_minimal_set": witness_not_in_minimal_set,
        "witness_not_strict_subset_of_control": witness_not_strict_subset_of_control,
        "entries_without_witness": entries_without_witness,
    }
    return report


def write_witness_report(
    verify_data: Dict[str, object],
    verify_output_path: str,
    T: int,
    input_path_label: str,
) -> Dict[str, object]:
    report = build_witness_report(verify_data, T=T)
    report["input"] = input_path_label
    os.makedirs(os.path.dirname(verify_output_path) or ".", exist_ok=True)
    with open(verify_output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    s = report["summary"]
    print(
        "Witness verify done. "
        f"all_correct={s['all_witness_minimal_min_viol_len_correct']}, "
        f"max_length_consistent={s['all_witness_consistent_with_max_length']}, "
        f"all_witness_correct={s['all_witness_correct']}, "
        f"mismatches={s['mismatch_count']}, "
        f"witness_not_in_set={s['witness_not_in_minimal_set_count']}, "
        f"witness_not_subset={s['witness_not_strict_subset_count']}"
    )
    print(f"Witness report: {verify_output_path}")
    return report


def verify_witness_main(input_path: str, T: Optional[int], output_path: Optional[str]) -> str:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cache_path = str(data.get("json_cache", ""))
    resolved_T = T if T is not None else (_parse_cache_horizon(cache_path) or 60)
    if resolved_T < 1:
        raise ValueError("verify T must be >= 1")
    out = output_path or os.path.join("_results", "verify_witness_report.json")
    write_witness_report(
        verify_data=data,
        verify_output_path=out,
        T=resolved_T,
        input_path_label=input_path,
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Merged tool: (1) detect non-minimal controls from sol.json via bounded cache, "
            "and (2) verify witness minimal controls by re-solving bounded LLP."
        )
    )
    group = ap.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--work_dirs",
        nargs="+",
        metavar="DIR",
        help="One or more explicit experiment directories.",
    )
    group.add_argument(
        "--root_dir",
        metavar="DIR",
        help=(
            "Root directory; every subdirectory that contains 'benders/', 'MibS/', or 'pbn/' "
            "is treated as a work directory."
        ),
    )
    ap.add_argument(
        "--verify-input",
        metavar="FILE",
        default=None,
        help=(
            "Run witness verification only on an existing verify_minimality JSON output "
            "(skip discovery stage)."
        ),
    )
    ap.add_argument(
        "--instances",
        nargs="+",
        metavar="INST",
        default=_INSTANCE_LIST_FULL,
        help=f"Instances to scan (default: all). Available: {', '.join(_INSTANCE_LIST_FULL)}",
    )
    ap.add_argument(
        "--json-cache",
        required=False,
        metavar="FILE",
        help="Bounded checker cache JSON (e.g., _experiments/checker_bounded_T60.json).",
    )
    ap.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help=(
            "Output JSON path for detected non-minimal controls "
            "(default: {root}/_results/verify_minimality_nonminimal.json)."
        ),
    )
    ap.add_argument(
        "--verify-witness",
        action="store_true",
        help="After discovery, run witness bounded recheck and write a verification report JSON.",
    )
    ap.add_argument(
        "--verify-T",
        type=int,
        default=None,
        metavar="N",
        help="Length bound for witness recheck (default: parse from cache file name, else 60).",
    )
    ap.add_argument(
        "--verify-output",
        default=None,
        metavar="FILE",
        help=(
            "Witness verification report path "
            "(default: _results/verify_witness_report.json)."
        ),
    )
    args = ap.parse_args()

    # Verify-only mode
    if args.verify_input:
        verify_witness_main(
            input_path=args.verify_input,
            T=args.verify_T,
            output_path=args.verify_output,
        )
        return

    if not args.work_dirs and not args.root_dir:
        raise ValueError("Either --work_dirs or --root_dir is required unless --verify-input is used.")
    if not args.json_cache:
        raise ValueError("--json-cache is required for discovery mode.")

    if args.work_dirs:
        work_dir_list = args.work_dirs
    else:
        work_dir_list = _detect_work_dirs_from_root(args.root_dir)
        print(f"Found {len(work_dir_list)} experiment(s) under {args.root_dir}:")
        for d in work_dir_list:
            print(f"  {d}")
    output_path = args.output or _default_nonminimal_output_path(args.root_dir, work_dir_list)

    cache_data = _load_bounded_cache(args.json_cache)
    cache_horizon = _parse_cache_horizon(args.json_cache)

    max_control_size_by_work_dir: Dict[str, Optional[int]] = {
        wd: _load_max_control_size(wd) for wd in work_dir_list
    }
    max_length_by_work_dir: Dict[str, Optional[int]] = {
        wd: _load_max_length(wd) for wd in work_dir_list
    }
    settings: List[Tuple[str, int, int]] = []
    for wd in work_dir_list:
        mcs = max_control_size_by_work_dir[wd]
        ml = max_length_by_work_dir[wd]
        if mcs is None or ml is None:
            continue
        if cache_horizon is not None and ml > cache_horizon:
            # Cache cannot certify beyond its bounded horizon.
            continue
        for inst in args.instances:
            settings.append((inst, mcs, ml))

    minimal_by_setting, minimal_len_by_setting = _build_minimal_controls_by_setting(
        cache_data,
        settings,
    )

    non_minimal_entries: List[Dict[str, object]] = []
    missing_sol: List[Dict[str, str]] = []
    missing_max_control_size: List[str] = []
    missing_max_length: List[str] = []
    skipped_by_cache_horizon: List[str] = []
    total_controls_scanned = 0

    for work_dir in work_dir_list:
        experiment = os.path.basename(os.path.normpath(work_dir))
        max_control_size = max_control_size_by_work_dir.get(work_dir)
        max_length = max_length_by_work_dir.get(work_dir)
        if max_control_size is None:
            print(f"[skip] {experiment}: missing/invalid max_control_size in alg_config.json")
            missing_max_control_size.append(work_dir)
            continue
        if max_length is None:
            print(f"[skip] {experiment}: missing/invalid max_length in alg_config.json")
            missing_max_length.append(work_dir)
            continue
        if cache_horizon is not None and max_length > cache_horizon:
            print(
                f"[skip] {experiment}: max_length={max_length} exceeds cache horizon T={cache_horizon}"
            )
            skipped_by_cache_horizon.append(work_dir)
            continue

        for inst in args.instances:
            sol_path = _find_sol_path(work_dir, inst)
            if sol_path is None:
                missing_sol.append({"work_dir": work_dir, "instance": inst})
                continue

            ctrls = _load_controls_from_sol(sol_path)
            total_controls_scanned += len(ctrls)
            setting = (inst, max_control_size, max_length)
            minimal_set = minimal_by_setting.get(setting, [])
            minimal_len_map = minimal_len_by_setting.get(setting, {})

            # Fast key set for exact-minimal controls.
            minimal_keys = {_ctrl_key(c) for c in minimal_set}
            for ctrl in ctrls:
                key = _ctrl_key(ctrl)
                if key in minimal_keys:
                    continue
                witness_list = _find_subset_witnesses(ctrl, minimal_set)
                if not witness_list:
                    continue
                witness_objs = []
                for witness in witness_list:
                    witness_key = _ctrl_key(witness)
                    witness_objs.append(
                        {
                            "control": dict(sorted(witness.items())),
                            "min_viol_len": minimal_len_map.get(witness_key),
                        }
                    )
                non_minimal_entries.append(
                    {
                        "work_dir": work_dir,
                        "experiment": experiment,
                        "instance": inst,
                        "max_control_size": max_control_size,
                        "max_length": max_length,
                        "control": dict(sorted(ctrl.items())),
                        "witness_minimals": witness_objs,
                    }
                )

    non_minimal_entries.sort(
        key=lambda x: (
            str(x["experiment"]),
            str(x["instance"]),
            len(x["control"]),
            sorted(x["control"].items()),
        )
    )

    minimal_controls_json = {
        f"{inst}|{mcs}|{ml}": [
            dict(sorted(c.items())) for c in minimal_by_setting[(inst, mcs, ml)]
        ]
        for inst, mcs, ml in sorted(
            minimal_by_setting.keys(), key=lambda x: (x[0], x[1], x[2])
        )
    }

    output_obj = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "json_cache": args.json_cache,
        "cache_horizon_T": cache_horizon,
        "work_dirs": work_dir_list,
        "instances": args.instances,
        "summary": {
            "experiments": len(work_dir_list),
            "controls_scanned": total_controls_scanned,
            "non_minimal_found": len(non_minimal_entries),
            "missing_sol_count": len(missing_sol),
            "missing_max_control_size_count": len(missing_max_control_size),
            "missing_max_length_count": len(missing_max_length),
            "skipped_by_cache_horizon_count": len(skipped_by_cache_horizon),
        },
        "minimal_controls_by_setting": minimal_controls_json,
        "non_minimal_controls": non_minimal_entries,
        "missing_sol": missing_sol,
        "missing_max_control_size": missing_max_control_size,
        "missing_max_length": missing_max_length,
        "skipped_by_cache_horizon": skipped_by_cache_horizon,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_obj, f, indent=2)

    print(
        f"Done. scanned={total_controls_scanned}, non_minimal={len(non_minimal_entries)}, "
        f"output={output_path}"
    )

    if args.verify_witness:
        verify_T = args.verify_T if args.verify_T is not None else (cache_horizon or 60)
        if verify_T < 1:
            raise ValueError("--verify-T must be >= 1")
        verify_out = args.verify_output or os.path.join("_results", "verify_witness_report.json")
        write_witness_report(
            verify_data=output_obj,
            verify_output_path=verify_out,
            T=verify_T,
            input_path_label=output_path,
        )


if __name__ == "__main__":
    main()
