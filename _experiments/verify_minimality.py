import argparse
import datetime
import json
import os
from typing import Dict, List, Optional, Tuple

from optboolnet.instances import _INSTANCE_LIST_FULL


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


def _build_minimal_controls_by_setting(
    cache_data: Dict[str, Dict[str, int]],
    settings: List[Tuple[str, int]],
) -> Tuple[
    Dict[Tuple[str, int], List[Dict[str, int]]],
    Dict[Tuple[str, int], Dict[Tuple[Tuple[str, int], ...], int]],
]:
    unique_settings = sorted(set(settings), key=lambda x: (x[0], x[1]))
    out: Dict[Tuple[str, int], List[Dict[str, int]]] = {}
    out_len_by_key: Dict[Tuple[str, int], Dict[Tuple[Tuple[str, int], ...], int]] = {}
    for inst, max_control_size in unique_settings:
        candidates: List[Dict[str, int]] = []
        candidate_len_by_key: Dict[Tuple[Tuple[str, int], ...], int] = {}
        for jk, min_viol_len in cache_data.get(inst, {}).items():
            if min_viol_len == -1 or min_viol_len >= max_control_size:
                ctrl = _parse_ctrl_json_key(jk)
                if ctrl is not None:
                    candidates.append(ctrl)
                    candidate_len_by_key[_ctrl_key(ctrl)] = int(min_viol_len)
        minimal_ctrls = _drop_nonminimal(candidates)
        out[(inst, max_control_size)] = minimal_ctrls
        out_len_by_key[(inst, max_control_size)] = {
            _ctrl_key(ctrl): candidate_len_by_key[_ctrl_key(ctrl)]
            for ctrl in minimal_ctrls
            if _ctrl_key(ctrl) in candidate_len_by_key
        }
    return out, out_len_by_key


def _find_subset_witness(
    ctrl: Dict[str, int],
    minimal_set: List[Dict[str, int]],
) -> Optional[Dict[str, int]]:
    for m in minimal_set:
        if _is_subset(m, ctrl, strict=True):
            return m
    return None


def _detect_work_dirs_from_root(root_dir: str) -> List[str]:
    return [
        os.path.join(root_dir, sub)
        for sub in sorted(os.listdir(root_dir))
        if any(os.path.isdir(os.path.join(root_dir, sub, algo)) for algo in _ALGO_SUBDIRS)
    ]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Detect non-minimal controls in sol.json using bounded-check cache. "
            "For each (instance, max_control_size), minimal controls are computed from cache "
            "entries with min_viol_len == -1 or >= max_control_size."
        )
    )
    group = ap.add_mutually_exclusive_group(required=True)
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
        "--instances",
        nargs="+",
        metavar="INST",
        default=_INSTANCE_LIST_FULL,
        help=f"Instances to scan (default: all). Available: {', '.join(_INSTANCE_LIST_FULL)}",
    )
    ap.add_argument(
        "--json-cache",
        required=True,
        metavar="FILE",
        help="Bounded checker cache JSON (e.g., _experiments/checker_bounded_T60.json).",
    )
    ap.add_argument(
        "--output",
        metavar="FILE",
        default="_experiments/verify_minimality_nonminimal.json",
        help=(
            "Output JSON path for detected non-minimal controls "
            "(default: _experiments/verify_minimality_nonminimal.json)."
        ),
    )
    args = ap.parse_args()

    if args.work_dirs:
        work_dir_list = args.work_dirs
    else:
        work_dir_list = _detect_work_dirs_from_root(args.root_dir)
        print(f"Found {len(work_dir_list)} experiment(s) under {args.root_dir}:")
        for d in work_dir_list:
            print(f"  {d}")

    cache_data = _load_bounded_cache(args.json_cache)

    max_control_size_by_work_dir: Dict[str, Optional[int]] = {
        wd: _load_max_control_size(wd) for wd in work_dir_list
    }
    settings: List[Tuple[str, int]] = []
    for wd in work_dir_list:
        mcs = max_control_size_by_work_dir[wd]
        if mcs is None:
            continue
        for inst in args.instances:
            settings.append((inst, mcs))

    minimal_by_setting, minimal_len_by_setting = _build_minimal_controls_by_setting(
        cache_data,
        settings,
    )

    non_minimal_entries: List[Dict[str, object]] = []
    missing_sol: List[Dict[str, str]] = []
    missing_max_control_size: List[str] = []
    total_controls_scanned = 0

    for work_dir in work_dir_list:
        experiment = os.path.basename(os.path.normpath(work_dir))
        max_control_size = max_control_size_by_work_dir.get(work_dir)
        if max_control_size is None:
            print(f"[skip] {experiment}: missing/invalid max_control_size in alg_config.json")
            missing_max_control_size.append(work_dir)
            continue

        for inst in args.instances:
            sol_path = _find_sol_path(work_dir, inst)
            if sol_path is None:
                missing_sol.append({"work_dir": work_dir, "instance": inst})
                continue

            ctrls = _load_controls_from_sol(sol_path)
            total_controls_scanned += len(ctrls)
            minimal_set = minimal_by_setting.get((inst, max_control_size), [])
            minimal_len_map = minimal_len_by_setting.get((inst, max_control_size), {})

            # Fast key set for exact-minimal controls.
            minimal_keys = {_ctrl_key(c) for c in minimal_set}
            for ctrl in ctrls:
                key = _ctrl_key(ctrl)
                if key in minimal_keys:
                    continue
                witness = _find_subset_witness(ctrl, minimal_set)
                if witness is None:
                    continue
                witness_key = _ctrl_key(witness)
                non_minimal_entries.append(
                    {
                        "work_dir": work_dir,
                        "experiment": experiment,
                        "instance": inst,
                        "max_control_size": max_control_size,
                        "control": dict(sorted(ctrl.items())),
                        "witness_minimal": dict(sorted(witness.items())),
                        "witness_minimal_min_viol_len": minimal_len_map.get(witness_key),
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
        f"{inst}|{mcs}": [dict(sorted(c.items())) for c in minimal_by_setting[(inst, mcs)]]
        for inst, mcs in sorted(minimal_by_setting.keys(), key=lambda x: (x[0], x[1]))
    }

    output_obj = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "json_cache": args.json_cache,
        "work_dirs": work_dir_list,
        "instances": args.instances,
        "summary": {
            "experiments": len(work_dir_list),
            "controls_scanned": total_controls_scanned,
            "non_minimal_found": len(non_minimal_entries),
            "missing_sol_count": len(missing_sol),
            "missing_max_control_size_count": len(missing_max_control_size),
        },
        "minimal_controls_by_setting": minimal_controls_json,
        "non_minimal_controls": non_minimal_entries,
        "missing_sol": missing_sol,
        "missing_max_control_size": missing_max_control_size,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_obj, f, indent=2)

    print(
        f"Done. scanned={total_controls_scanned}, non_minimal={len(non_minimal_entries)}, "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
