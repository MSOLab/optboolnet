import argparse
import datetime
import json
import os
import statistics
import time
from typing import Dict, List, Tuple

from optboolnet.boolnet import Control
from optboolnet.checking import nusmv_check_phenotype
from optboolnet.instances import load_bn_in_repo


DEFAULT_SOL_PATH = (
    "_experiments/260304_bilevel_agg_results/SEP_45_decomp/benders/M1/sol.json"
)
DEFAULT_OUT_CSV = "_experiments/benchmark_ctl_variants_summary.csv"


_VARIANTS: Dict[str, Dict[str, object]] = {
    "ctl_ef_ag": {
        "property_variant": "ctl_ef_ag",
        "constrain_controlled_vars": False,
        "description": "CTLSPEC EF AG phenotype",
    },
    "ctl_ef_ag_lock": {
        "property_variant": "ctl_ef_ag",
        "constrain_controlled_vars": True,
        "description": "CTLSPEC EF AG phenotype + INIT/INVAR control lock",
    },
    "ltl_fg": {
        "property_variant": "ltl_fg",
        "constrain_controlled_vars": False,
        "description": "LTLSPEC F G phenotype",
    },
    "ltl_fg_lock": {
        "property_variant": "ltl_fg",
        "constrain_controlled_vars": True,
        "description": "LTLSPEC F G phenotype + INIT/INVAR control lock",
    },
}


def _ctrl_key(ctrl: Control) -> Tuple[Tuple[str, int], ...]:
    return tuple(sorted(ctrl.items()))


def _load_controls(sol_path: str) -> List[Control]:
    with open(sol_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    controls: List[Control] = []
    for _, sol_list in data.items():
        for sol in sol_list:
            controls.append(Control(sol))
    return controls


def _dedup_controls(controls: List[Control]) -> List[Control]:
    out: List[Control] = []
    seen = set()
    for ctrl in controls:
        key = _ctrl_key(ctrl)
        if key in seen:
            continue
        seen.add(key)
        out.append(ctrl)
    return out


def _run_variant(
    bn,
    controls: List[Control],
    variant_name: str,
    update_mode: str,
    nusmv_opts: Dict[str, bool],
    progress_every: int,
):
    cfg = _VARIANTS[variant_name]
    prop = cfg["property_variant"]
    lock = bool(cfg["constrain_controlled_vars"])

    times: List[float] = []
    oks: List[bool] = []
    n = len(controls)
    started = time.perf_counter()
    last_reported = 0

    print(f"[{variant_name}] {cfg['description']}")
    for i, ctrl in enumerate(controls):
        t0 = time.perf_counter()
        ok = nusmv_check_phenotype(
            bn,
            control=ctrl,
            update_mode=update_mode,
            property_variant=prop,
            constrain_controlled_vars=lock,
            nusmv_opts=nusmv_opts,
        )
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        oks.append(ok)

        done = i + 1
        should_report = (
            done == n
            or (progress_every > 0 and done % progress_every == 0)
            or done == 1
        )
        if should_report and done != last_reported:
            elapsed_total = time.perf_counter() - started
            avg = elapsed_total / done
            eta = avg * (n - done)
            pct = 100.0 * done / n
            print(
                f"  {done}/{n} ({pct:.1f}%) "
                f"elapsed={elapsed_total:.1f}s avg={avg:.3f}s eta={eta:.1f}s"
            )
            last_reported = done

    total = sum(times)
    return {
        "variant": variant_name,
        "property_variant": prop,
        "constrain_controlled_vars": lock,
        "description": cfg["description"],
        "n_controls": n,
        "times": times,
        "oks": oks,
        "total_sec": total,
        "avg_sec": (total / n) if n else 0.0,
        "median_sec": statistics.median(times) if times else 0.0,
        "min_sec": min(times) if times else 0.0,
        "max_sec": max(times) if times else 0.0,
        "true_count": sum(1 for x in oks if x),
        "false_count": sum(1 for x in oks if not x),
    }


def _write_summary_csv(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    header = [
        "timestamp",
        "instance",
        "update_mode",
        "variant",
        "description",
        "property_variant",
        "constrain_controlled_vars",
        "n_controls",
        "total_sec",
        "avg_sec",
        "median_sec",
        "min_sec",
        "max_sec",
        "true_count",
        "false_count",
        "mismatch_vs_baseline",
        "nusmv_opts",
    ]
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(",".join(header) + "\n")
        for r in rows:
            fp.write(
                ",".join(
                    [
                        str(r["timestamp"]),
                        str(r["instance"]),
                        str(r["update_mode"]),
                        str(r["variant"]),
                        str(r["description"]),
                        str(r["property_variant"]),
                        str(r["constrain_controlled_vars"]),
                        str(r["n_controls"]),
                        f"{float(r['total_sec']):.6f}",
                        f"{float(r['avg_sec']):.6f}",
                        f"{float(r['median_sec']):.6f}",
                        f"{float(r['min_sec']):.6f}",
                        f"{float(r['max_sec']):.6f}",
                        str(r["true_count"]),
                        str(r["false_count"]),
                        str(r["mismatch_vs_baseline"]),
                        str(r["nusmv_opts"]),
                    ]
                )
                + "\n"
            )


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark NuSMV phenotype-check variants over controls in sol.json"
    )
    ap.add_argument("--instance", default="M1", help="Instance name (default: M1)")
    ap.add_argument("--sol", default=DEFAULT_SOL_PATH, help="Path to sol.json")
    ap.add_argument(
        "--update_mode",
        default="synchronous",
        choices=["synchronous", "asynchronous", "general"],
        help="NuSMV update mode",
    )
    ap.add_argument(
        "--variants",
        nargs="+",
        default=["ctl_ef_ag", "ctl_ef_ag_lock", "ltl_fg", "ltl_fg_lock"],
        choices=sorted(_VARIANTS.keys()),
        help="Variant names to benchmark",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N controls per variant (default: 10)",
    )
    ap.add_argument(
        "--nusmv-opts",
        nargs="*",
        default=[],
        metavar="OPT",
        help=(
            "Extra NuSMV boolean flags to enable (without leading '-'), "
            "e.g. dynamic reorder flt mono"
        ),
    )
    ap.add_argument(
        "--max-controls",
        type=int,
        default=80,
        help="Limit number of controls after dedup (default: 80)",
    )
    ap.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Do not deduplicate controls",
    )
    ap.add_argument(
        "--csv",
        default=DEFAULT_OUT_CSV,
        help=f"Summary output CSV (default: {DEFAULT_OUT_CSV})",
    )
    args = ap.parse_args()

    controls = _load_controls(args.sol)
    if not args.keep_duplicates:
        controls = _dedup_controls(controls)
    if args.max_controls > 0:
        controls = controls[: args.max_controls]

    if not controls:
        raise RuntimeError("No controls to benchmark after filtering")

    print(f"instance={args.instance} update_mode={args.update_mode}")
    print(f"controls={len(controls)} variants={args.variants}")
    nusmv_opts = {opt: True for opt in args.nusmv_opts}
    if nusmv_opts:
        print(f"nusmv_opts={sorted(nusmv_opts.keys())}")

    bn = load_bn_in_repo(args.instance)

    run_results: Dict[str, Dict[str, object]] = {}
    for variant_name in args.variants:
        run_results[variant_name] = _run_variant(
            bn=bn,
            controls=controls,
            variant_name=variant_name,
            update_mode=args.update_mode,
            nusmv_opts=nusmv_opts,
            progress_every=args.progress_every,
        )

    baseline_name = args.variants[0]
    baseline_oks = run_results[baseline_name]["oks"]
    rows: List[Dict[str, object]] = []
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\nSummary")
    print("variant,total_sec,avg_sec,median_sec,false_count,mismatch_vs_baseline")
    for variant_name in args.variants:
        rr = run_results[variant_name]
        oks = rr["oks"]
        mismatch = sum(1 for a, b in zip(baseline_oks, oks) if a != b)

        print(
            f"{variant_name},"
            f"{rr['total_sec']:.2f},"
            f"{rr['avg_sec']:.3f},"
            f"{rr['median_sec']:.3f},"
            f"{rr['false_count']},"
            f"{mismatch}"
        )

        rows.append(
            {
                "timestamp": timestamp,
                "instance": args.instance,
                "update_mode": args.update_mode,
                "variant": variant_name,
                "description": rr["description"],
                "property_variant": rr["property_variant"],
                "constrain_controlled_vars": rr["constrain_controlled_vars"],
                "n_controls": rr["n_controls"],
                "total_sec": rr["total_sec"],
                "avg_sec": rr["avg_sec"],
                "median_sec": rr["median_sec"],
                "min_sec": rr["min_sec"],
                "max_sec": rr["max_sec"],
                "true_count": rr["true_count"],
                "false_count": rr["false_count"],
                "mismatch_vs_baseline": mismatch,
                "nusmv_opts": ";".join(sorted(nusmv_opts.keys())),
            }
        )

    _write_summary_csv(args.csv, rows)
    print(f"\nWrote summary: {args.csv}")


if __name__ == "__main__":
    main()
