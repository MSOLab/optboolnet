import argparse
import datetime
import json
import os
import statistics
import time
from typing import Dict, List, Tuple

from optboolnet.boolnet import Control
from optboolnet.checking import nusmv_check_phenotype, nusmv_check_phenotype_batch
from optboolnet.instances import load_bn_in_repo


DEFAULT_SOL_PATH = (
    "_experiments/260304_bilevel_agg_results/SEP_45_decomp/benders/M1/sol.json"
)
DEFAULT_OUT_CSV = "_experiments/benchmark_batch_ctl_ltl.csv"


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


def _run_single(
    bn,
    controls: List[Control],
    property_variant: str,
    constrain_controlled_vars: bool,
    nusmv_opts: Dict[str, bool],
    progress_every: int,
):
    times: List[float] = []
    oks: List[bool] = []
    n = len(controls)
    started = time.perf_counter()

    for i, ctrl in enumerate(controls):
        t0 = time.perf_counter()
        ok = nusmv_check_phenotype(
            bn,
            control=ctrl,
            update_mode="synchronous",
            property_variant=property_variant,
            constrain_controlled_vars=constrain_controlled_vars,
            nusmv_opts=nusmv_opts,
        )
        dt = time.perf_counter() - t0
        times.append(dt)
        oks.append(ok)

        done = i + 1
        if done == 1 or done == n or (progress_every > 0 and done % progress_every == 0):
            elapsed = time.perf_counter() - started
            avg = elapsed / done
            eta = avg * (n - done)
            pct = 100.0 * done / n
            print(
                f"    {done}/{n} ({pct:.1f}%) elapsed={elapsed:.1f}s avg={avg:.3f}s eta={eta:.1f}s"
            )

    return times, oks


def _run_batch(
    bn,
    controls: List[Control],
    property_variant: str,
    constrain_controlled_vars: bool,
    nusmv_opts: Dict[str, bool],
):
    t0 = time.perf_counter()
    oks = nusmv_check_phenotype_batch(
        bn,
        controls,
        update_mode="synchronous",
        property_variant=property_variant,
        constrain_controlled_vars=constrain_controlled_vars,
        nusmv_opts=nusmv_opts,
    )
    total = time.perf_counter() - t0
    avg = total / len(controls)
    times = [avg for _ in controls]
    return times, oks, total


def _summarize(label: str, times: List[float], oks: List[bool], total_override: float = None):
    total = sum(times) if total_override is None else total_override
    n = len(times)
    return {
        "label": label,
        "n_controls": n,
        "total_sec": total,
        "avg_sec": total / n if n else 0.0,
        "median_sec": statistics.median(times) if times else 0.0,
        "min_sec": min(times) if times else 0.0,
        "max_sec": max(times) if times else 0.0,
        "true_count": sum(1 for x in oks if x),
        "false_count": sum(1 for x in oks if not x),
        "oks": oks,
    }


def _write_csv(path: str, rows: List[Dict[str, object]]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    header = [
        "timestamp",
        "instance",
        "sol",
        "label",
        "n_controls",
        "total_sec",
        "avg_sec",
        "median_sec",
        "min_sec",
        "max_sec",
        "true_count",
        "false_count",
        "mismatch_vs_single_ctl_lock",
        "speedup_vs_single_ctl_lock",
    ]
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(",".join(header) + "\n")
        for r in rows:
            fp.write(
                ",".join(
                    [
                        str(r["timestamp"]),
                        str(r["instance"]),
                        str(r["sol"]),
                        str(r["label"]),
                        str(r["n_controls"]),
                        f"{float(r['total_sec']):.6f}",
                        f"{float(r['avg_sec']):.6f}",
                        f"{float(r['median_sec']):.6f}",
                        f"{float(r['min_sec']):.6f}",
                        f"{float(r['max_sec']):.6f}",
                        str(r["true_count"]),
                        str(r["false_count"]),
                        str(r["mismatch_vs_single_ctl_lock"]),
                        f"{float(r['speedup_vs_single_ctl_lock']):.6f}",
                    ]
                )
                + "\n"
            )


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Benchmark single-run vs batched model checking and CTL-vs-LTL equivalence "
            "for synchronous controls"
        )
    )
    ap.add_argument("--instance", default="M1")
    ap.add_argument("--sol", default=DEFAULT_SOL_PATH)
    ap.add_argument("--max-controls", type=int, default=0, help="0 means all controls")
    ap.add_argument("--keep-duplicates", action="store_true")
    ap.add_argument("--progress-every", type=int, default=20)
    ap.add_argument("--csv", default=DEFAULT_OUT_CSV)
    args = ap.parse_args()

    controls = _load_controls(args.sol)
    if not args.keep_duplicates:
        controls = _dedup_controls(controls)
    if args.max_controls > 0:
        controls = controls[: args.max_controls]

    if not controls:
        raise RuntimeError("No controls to benchmark")

    print(f"instance={args.instance} controls={len(controls)}")
    bn = load_bn_in_repo(args.instance)

    results = {}

    print("[single_ctl_lock] running...")
    times, oks = _run_single(
        bn,
        controls,
        property_variant="ctl_ef_ag",
        constrain_controlled_vars=True,
        nusmv_opts={},
        progress_every=args.progress_every,
    )
    results["single_ctl_lock"] = _summarize("single_ctl_lock", times, oks)

    print("[batch_ctl_lock] running...")
    times, oks, total = _run_batch(
        bn,
        controls,
        property_variant="ctl_ef_ag",
        constrain_controlled_vars=True,
        nusmv_opts={},
    )
    print(f"    done elapsed={total:.1f}s")
    results["batch_ctl_lock"] = _summarize(
        "batch_ctl_lock", times, oks, total_override=total
    )

    print("[single_ltl_lock] running...")
    times, oks = _run_single(
        bn,
        controls,
        property_variant="ltl_fg",
        constrain_controlled_vars=True,
        nusmv_opts={},
        progress_every=max(5, args.progress_every // 2),
    )
    results["single_ltl_lock"] = _summarize("single_ltl_lock", times, oks)

    print("[batch_ltl_lock] running...")
    times, oks, total = _run_batch(
        bn,
        controls,
        property_variant="ltl_fg",
        constrain_controlled_vars=True,
        nusmv_opts={},
    )
    print(f"    done elapsed={total:.1f}s")
    results["batch_ltl_lock"] = _summarize(
        "batch_ltl_lock", times, oks, total_override=total
    )

    baseline = results["single_ctl_lock"]
    baseline_oks = baseline["oks"]
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\nSummary")
    print("label,total_sec,avg_sec,false_count,mismatch_vs_single_ctl_lock,speedup_vs_single_ctl_lock")
    rows = []
    for label in ["single_ctl_lock", "batch_ctl_lock", "single_ltl_lock", "batch_ltl_lock"]:
        r = results[label]
        mismatch = sum(1 for a, b in zip(baseline_oks, r["oks"]) if a != b)
        speedup = baseline["total_sec"] / r["total_sec"] if r["total_sec"] > 0 else 0.0
        print(
            f"{label},{r['total_sec']:.2f},{r['avg_sec']:.3f},{r['false_count']},{mismatch},{speedup:.3f}"
        )
        rows.append(
            {
                "timestamp": ts,
                "instance": args.instance,
                "sol": args.sol,
                "label": label,
                "n_controls": r["n_controls"],
                "total_sec": r["total_sec"],
                "avg_sec": r["avg_sec"],
                "median_sec": r["median_sec"],
                "min_sec": r["min_sec"],
                "max_sec": r["max_sec"],
                "true_count": r["true_count"],
                "false_count": r["false_count"],
                "mismatch_vs_single_ctl_lock": mismatch,
                "speedup_vs_single_ctl_lock": speedup,
            }
        )

    _write_csv(args.csv, rows)
    print(f"\nWrote summary: {args.csv}")


if __name__ == "__main__":
    main()
