"""Experiment runner for external Boolean network instances.

Loads instances from an external directory (e.g., .tmp/repairnet-tumorigenesis-v1)
using setting.json (inputs/target format) and CNFBooleanNetwork.from_bnet().

Usage:
    python main_external.py \\
        --instance-dir ../.tmp/repairnet-tumorigenesis-v1 \\
        --alg-config path/to/alg_config.json \\
        --output-dir path/to/output \\
        --instances inst1 inst2 ...    # or omit for all
        --workers 16
"""
import argparse
import concurrent.futures
import json
import os
import sys
import traceback
from datetime import datetime
from typing import Optional, Tuple

_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != _THIS_DIR]

from optboolnet import CNFBooleanNetwork
from optboolnet.config import BendersConfig


def load_external_bn(instance_path: str) -> CNFBooleanNetwork:
    """Load a BN from an external instance folder with setting.json."""
    from colomoto import minibn

    with open(os.path.join(instance_path, "setting.json")) as f:
        setting = json.load(f)

    bn = minibn.BooleanNetwork(
        os.path.join(instance_path, "transition_formula.bnet")
    )
    return CNFBooleanNetwork.from_bnet(
        bn,
        inputs=setting.get("inputs", {}),
        target=setting["target"],
        simplify=True,
    )


def _simplified_cnf(ba, formula):
    """Simplify a Boolean formula: simplify() then convert to CNF."""
    from colomoto.minibn import _TRUE, _FALSE
    from optboolnet.boolnet import simplify_cnf, robust_negate_to_cnf
    if isinstance(formula, (_TRUE, type(ba.TRUE))):
        return ba.TRUE
    if isinstance(formula, (_FALSE, type(ba.FALSE))):
        return ba.FALSE
    simplified = formula.simplify()
    if simplified is None:
        return ba.FALSE
    try:
        return simplify_cnf(ba, ba.cnf(simplified))
    except (TypeError, AttributeError):
        return simplify_cnf(ba, ba.cnf(ba.dnf(simplified)))


def _neg_cnf(ba, formula):
    """Compute simplified CNF(¬formula), robust to boolean.py crashes."""
    from optboolnet.boolnet import simplify_cnf, robust_negate_to_cnf
    return simplify_cnf(ba, robust_negate_to_cnf(ba, formula))


def export_hybrid_partition(bn: CNFBooleanNetwork, work_dir: str) -> None:
    """Export CNF/DNF gene partition and per-partition BNs to work_dir.

    Writes:
        partition.json          — gene lists and counts
        bn_cnf_genes.bnet       — f_i (simplified CNF) for CNF genes
        bn_cnf_genes_neg.bnet   — CNF(¬f_i) (simplified) for CNF genes
        bn_dnf_genes.bnet       — f_i (simplified CNF) for DNF genes
        bn_dnf_genes_neg.bnet   — CNF(¬f_i) (simplified) for DNF genes
    """
    if not bn.is_hybrid_enabled:
        bn.compute_hybrid_partition()

    ba = bn.ba

    # Per-gene literal counts for CNF(f_i) and CNF(¬f_i)
    gene_details = {}
    for g in bn.keys():
        pos_lits = bn._total_literals(bn.items_clause(g))
        neg_lits = bn._total_literals(bn.items_neg_clause(g))
        gene_details[g] = {
            "encoding": "CNF" if bn.is_cnf_gene(g) else "DNF",
            "pos_literals": pos_lits,
            "neg_literals": neg_lits,
            "pos_clauses": len(bn.items_clause(g)),
            "neg_clauses": len(bn.items_neg_clause(g)),
        }

    partition = {
        "cnf_genes": bn.cnf_genes,
        "dnf_genes": bn.dnf_genes,
        "num_cnf": len(bn.cnf_genes),
        "num_dnf": len(bn.dnf_genes),
        "gene_details": gene_details,
    }
    with open(os.path.join(work_dir, "partition.json"), "w") as f:
        json.dump(partition, f, indent=2)

    # CNF genes: f_i simplified
    cnf_lines = sorted(
        f"{g}, {_simplified_cnf(ba, bn.get_cnf(g))}"
        for g in bn.cnf_genes
    )
    with open(os.path.join(work_dir, "bn_cnf_genes.bnet"), "w") as f:
        f.write("\n".join(cnf_lines) + "\n")

    # CNF genes: CNF(¬f_i) simplified
    cnf_neg_lines = sorted(
        f"{g}, {_neg_cnf(ba, bn.get_cnf(g))}"
        for g in bn.cnf_genes
    )
    with open(os.path.join(work_dir, "bn_cnf_genes_neg.bnet"), "w") as f:
        f.write("\n".join(cnf_neg_lines) + "\n")

    # DNF genes: f_i simplified
    dnf_lines = sorted(
        f"{g}, {_simplified_cnf(ba, bn.get_cnf(g))}"
        for g in bn.dnf_genes
    )
    with open(os.path.join(work_dir, "bn_dnf_genes.bnet"), "w") as f:
        f.write("\n".join(dnf_lines) + "\n")

    # DNF genes: CNF(¬f_i) simplified — used for z variables in the model
    dnf_neg_lines = sorted(
        f"{g}, {_neg_cnf(ba, bn.get_cnf(g))}"
        for g in bn.dnf_genes
    )
    with open(os.path.join(work_dir, "bn_dnf_genes_neg.bnet"), "w") as f:
        f.write("\n".join(dnf_neg_lines) + "\n")


def run_benders_external(
    instance_path: str, exp_name: str, config: BendersConfig
) -> None:
    from optboolnet.algorithm import BendersAttractorControl

    bn = load_external_bn(instance_path)
    alg = BendersAttractorControl(exp_name, bn, config.logging_config)
    export_hybrid_partition(bn, config.logging_config.fpath)
    alg.allow_empty_attractor = config.allow_empty_attractor
    alg.solve_separation = config.solve_separation
    alg.preprocess_max_forbidden_trap_space = (
        config.preprocess_max_forbidden_trap_space
    )
    alg.separation_heuristic = config.separation_heuristic
    alg.use_high_point_relaxation = config.use_high_point_relaxation
    alg.total_time_limit = config.total_time_limit
    alg.use_aggregated_LLP = getattr(config, "use_aggregated_LLP", False)
    alg.use_hybrid_encoding = getattr(config, "use_hybrid_encoding", True)
    alg.get_control_strategies(
        max_control_size=config.max_control_size,
        max_length=config.max_length,
        master_solver_config=config.master_solver_config,
        LLP_solver_config=config.LLP_solver_config,
        separation_solver_config=config.separation_solver_config,
    )


def load_config(alg_config_path: str):
    with open(alg_config_path, "r") as fp:
        payload = json.load(fp)

    if "total_time_limit" not in payload:
        raise ValueError(
            f"Missing 'total_time_limit' in {alg_config_path}."
        )
    time_limit = payload["total_time_limit"]
    if not isinstance(time_limit, (int, float)) or time_limit <= 0:
        raise ValueError(
            f"Invalid 'total_time_limit' in {alg_config_path}: {time_limit!r}."
        )

    config = BendersConfig.from_dict(payload)
    return config, float(time_limit)


def run_task(task: Tuple[str, str, str, str, Optional[float]]) -> str:
    instance_dir, alg_config_path, output_dir, inst, time_limit_override = task

    config, _ = load_config(alg_config_path)
    if time_limit_override is not None:
        config.total_time_limit = time_limit_override

    work_dir = os.path.join(output_dir, "benders", inst)
    os.makedirs(work_dir, exist_ok=True)
    config.logging_config.fpath = work_dir

    instance_path = os.path.join(instance_dir, inst)
    run_benders_external(instance_path, inst, config)
    return inst


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Experiment runner for external BN instances."
    )
    ap.add_argument(
        "--instance-dir",
        required=True,
        help="Directory containing instance subfolders (each with setting.json + .bnet).",
    )
    ap.add_argument(
        "--alg-config",
        required=True,
        help="Path to alg_config.json (BendersConfig).",
    )
    ap.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for results (benders/<inst>/sol.json).",
    )
    ap.add_argument(
        "--instances",
        nargs="*",
        default=None,
        help="Instance subfolder names. Omit to run all subfolders.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes.",
    )
    ap.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Override total_time_limit (seconds).",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.time_limit is not None and args.time_limit <= 0:
        raise ValueError("--time-limit must be > 0")

    instance_dir = os.path.abspath(args.instance_dir)
    if not os.path.isdir(instance_dir):
        raise NotADirectoryError(f"Instance directory not found: {instance_dir}")

    alg_config_path = os.path.abspath(args.alg_config)
    if not os.path.exists(alg_config_path):
        raise FileNotFoundError(f"Config not found: {alg_config_path}")

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Discover instances
    if args.instances:
        instances = args.instances
    else:
        instances = sorted(
            d
            for d in os.listdir(instance_dir)
            if os.path.isdir(os.path.join(instance_dir, d))
            and os.path.exists(os.path.join(instance_dir, d, "setting.json"))
        )

    _, json_time_limit = load_config(alg_config_path)

    tasks = [
        (instance_dir, alg_config_path, output_dir, inst, args.time_limit)
        for inst in instances
    ]

    run_tag = datetime.now().strftime("%y%m%d_%H%M%S")
    run_log_path = os.path.join(output_dir, f"run_times__{run_tag}.log")
    run_args_path = os.path.join(output_dir, f"run_args__{run_tag}.json")

    run_args_payload = {
        "timestamp": timestamp(),
        "argv": sys.argv[1:],
        "parsed_args": {
            "instance_dir": args.instance_dir,
            "alg_config": args.alg_config,
            "output_dir": args.output_dir,
            "instances": instances,
            "workers": args.workers,
            "time_limit": args.time_limit,
        },
        "json_time_limit": json_time_limit,
    }
    with open(run_args_path, "w", encoding="utf-8") as fp:
        json.dump(run_args_payload, fp, indent=2)

    with open(run_log_path, "a", encoding="utf-8") as run_log_fp:

        def log_line(msg: str) -> None:
            lines = msg.splitlines() or [""]
            for line in lines:
                stamped = f"{timestamp()} {line}"
                print(stamped, flush=True)
                run_log_fp.write(stamped + "\n")
            run_log_fp.flush()

        limit_msg = (
            f"{json_time_limit}s (json)"
            if args.time_limit is None
            else f"{args.time_limit}s (override)"
        )
        log_line("[RUN START]")
        log_line(f"[RUN LOG] {run_log_path}")
        log_line(f"[RUN ARGS] {run_args_path}")
        log_line(
            f"Launching {len(tasks)} runs, workers={args.workers}, "
            f"time_limit={limit_msg}"
        )

        failures = []
        if args.workers == 1:
            for task in tasks:
                log_line(f"[TASK START] {task[3]}")
                try:
                    done = run_task(task)
                    log_line(f"[TASK END][OK] {done}")
                except Exception:
                    log_line(f"[TASK END][FAIL] {task[3]}")
                    failures.append((task[3], traceback.format_exc()))
        else:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers
            ) as executor:
                task_iter = iter(tasks)
                in_flight = {}

                for _ in range(min(args.workers, len(tasks))):
                    task = next(task_iter, None)
                    if task is None:
                        break
                    log_line(f"[TASK START] {task[3]}")
                    in_flight[executor.submit(run_task, task)] = task

                while in_flight:
                    done_futures, _ = concurrent.futures.wait(
                        in_flight,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in done_futures:
                        task = in_flight.pop(future)
                        try:
                            done = future.result()
                            log_line(f"[TASK END][OK] {done}")
                        except Exception:
                            log_line(f"[TASK END][FAIL] {task[3]}")
                            failures.append(
                                (task[3], traceback.format_exc())
                            )

                        next_task = next(task_iter, None)
                        if next_task is not None:
                            log_line(f"[TASK START] {next_task[3]}")
                            in_flight[
                                executor.submit(run_task, next_task)
                            ] = next_task

        log_line("[RUN END]")

        if failures:
            log_line("\nFailures:")
            for inst, err in failures:
                log_line(f"[FAIL] {inst}")
                log_line(err)
            raise SystemExit(1)

        log_line("All runs completed.")


if __name__ == "__main__":
    main()
