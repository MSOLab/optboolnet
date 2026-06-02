# Experiments

Command-line recipes for running the attractor-control benchmark and producing
the publication tables/plots. Commands mirror the configurations in
[.vscode/launch.json](../.vscode/launch.json), except the `External:` /
`Compare external` entries.

The **baseline result folder** used throughout is
[260306_full_no_good/](260306_full_no_good/). Each command below runs the
**full experiment** (all method/T subdirectories, all instances); lines are laid
out so you can comment out entries to run a **partial experiment**.

## Environment

All scripts run from a **single Python ≤ 3.11 environment** with every
dependency installed — all tools below were smoke-tested end-to-end there.
[environment.yml](environment.yml) is a full pinned snapshot of the reference env
(conda `buildtool`, Python 3.11) for exact versions; it is a win-64 export, **not
a portable install spec**. Build an equivalent env from the portable recipe below
(run from the repo root):

```bash
conda create -n optboolnet -c colomoto -c conda-forge python=3.11 clingo mpbn nusmv
conda activate optboolnet
# PBN backend from git (PyPI pyboolnet==3.0.16 also works):
pip install pandas matplotlib "pyboolnet @ git+https://github.com/hklarner/pyboolnet@3394310d07e3e25810a828d8da4f6d7241115113"
pip install -e .   # this repo's optboolnet (editable); also pulls
                   # pyomo, gurobipy, pao, boolean.py, colomoto_jupyter, algorecell_types
```

> If a **stale** conda env literally named `optboolnet` already exists (Python 3.12,
> missing deps, points at a sibling repo), remove it first
> (`conda env remove -n optboolnet`) or pick another name.

Each tool's main dependency:

| Tool | Key dependency |
| --- | --- |
| [main.py](main.py) SEP/BEN (benders) | Gurobi, Pyomo |
| [main.py](main.py) MibS (`MibS_*`) | `pao` + `deps/mibs.exe`¹ |
| [main.py](main.py) PBN (`PBN_PBN_*`) | `pyboolnet` |
| [collect_results.py](collect_results.py) | pandas |
| [verify_control_bounded_parallel.py](verify_control_bounded_parallel.py) | Gurobi |
| [verify_control_parallel.py](verify_control_parallel.py) (infinite) | `mpbn` + NuSMV |
| [pubs_analytics.py](pubs_analytics.py) / [pubs_analytics_infinite.py](pubs_analytics_infinite.py) | pandas |
| [pubs_comptime.py](pubs_comptime.py) / [pubs_cut_strength.py](pubs_cut_strength.py) | pandas |
| [pubs_comptime_plot.py](pubs_comptime_plot.py) | `matplotlib` |
| [longest_attractor.py](longest_attractor.py) / [longest_attractor_check.py](longest_attractor_check.py) | Gurobi |

> Gurobi needs a license (academic licenses are free). NuSMV is provided by the
> `colomoto::nusmv` conda package (binary + python wrapper).
>
> ¹ **MibS / `pao`.** `pao` is needed *only* for the MibS bilevel experiments and
> targets **Python ≤ 3.11**. With recent NumPy (≥ 2.0) `pao` raises `AttributeError`
> at solve time on models that hit its unbounded-bound paths, because `np.PINF` /
> `np.NINF` were removed — fix by replacing every `np.PINF` → `np.inf` and
> `np.NINF` → `-np.inf` in the installed `pao` source (`pao/mpr/convert_repn.py`).
> MibS also needs its bilevel solver binary at the path in each MibS
> `alg_config.json` (`deps\mibs.exe`; see [MibS_config.json](MibS_config.json)).

## Replicate the analytics from shared archives (coauthors)

The full benchmark is expensive to re-run, so coauthors instead receive three
result archives (shared out-of-band — **not** committed to git). Unpack them in
place and the analytics below run **without** re-running the benchmark. Commands
are bash (tested in WSL and Git Bash), from the repo root, after setting up the
[environment](#environment).

| Archive | Unpack into | Provides |
| --- | --- | --- |
| `260306_full_no_good_final_260313.zip` | `_experiments/` | the full baseline folder `260306_full_no_good/`, **including** `_results/` (all `*.csv` + `verify_control_T100.json` / `verify_control_Tinf.json`) — everything the `pubs_*` tables/plots read |
| `260308_minimality_correctness_check_results.zip` | `_experiments/` | curated correctness/minimality reports + checker caches (`checker_bounded_T100.json`, `checker_positive.json`, `checker_negative.json`) |
| `260311_longest_attractor_check.zip` | `_experiments/results/longest_attractor/Tmax_100/` | longest-attractor MILP results (`S1.json … L4.json`, `summary.json`) from step 5 |

Each archive contains a single top-level folder, so extracting into the paths
above reproduces the original layout:

```bash
# 1) Baseline results -> _experiments/260306_full_no_good/ (with _results/)
unzip -q _experiments/260306_full_no_good_final_260313.zip -d _experiments

# 2) Correctness / minimality curation -> _experiments/260308_minimality_correctness_check_results/
unzip -q _experiments/260308_minimality_correctness_check_results.zip -d _experiments

# 3) Longest-attractor MILP results
#    -> _experiments/results/longest_attractor/Tmax_100/260311_longest_attractor_check/
unzip -q _experiments/260311_longest_attractor_check.zip \
    -d _experiments/results/longest_attractor/Tmax_100
# unzip prompts before overwriting; pass -o to overwrite without asking
```

Then reproduce the outputs:

- **Publication tables & plots** (§4 below) run as-is — the baseline `_results/`
  CSVs and `verify_control_T*.json` are already in archive (1), so you can skip the
  benchmark (§1) and `collect_results.py` (§2). To instead rebuild the CSVs from the
  raw per-instance logs, run [collect_results.py](collect_results.py) (§2).
- **Longest-attractor summary** — point [longest_attractor.py](longest_attractor.py)
  (§5) at the bounded-checker cache from archive (2):

  ```bash
  python _experiments/longest_attractor.py \
      "_experiments/260308_minimality_correctness_check_results/checker_bounded_T100.json"
  ```

- **Re-verify controls** (§3) without recomputing — reuse the cached checker results
  from archive (2) by pointing the checkers at them, e.g.
  `--json-cache "_experiments/260308_minimality_correctness_check_results/checker_bounded_T100.json"`
  (bounded), or `--json-positive` / `--json-negative` (infinite). Or copy those JSONs
  to the script defaults (`_experiments/checker_bounded_T100.json`, etc.).
- **Longest-attractor MILP results** from archive (3): inspect `summary.json`; re-solve
  with [longest_attractor_check.py](longest_attractor_check.py) (§5).

## Conventions

- Run everything from the **repository root** with the project's Python
  environment active (`optboolnet`, Gurobi, Pyomo, PyBoolNet, NuSMV available).
  If unsure which environment, resolve it with `/env-manage --resolve`.
- Examples use **bash** (tested in WSL and Git Bash). The `subdirs` / `instances`
  arrays let you comment out a line with `#` to drop that group from a run.
- **Path quirk:** [main.py](main.py) resolves `--root-dir` *relative to the
  `_experiments/` folder*, so it takes `260306_full_no_good` (no prefix). Every
  other script takes the path *relative to the repo root*, i.e.
  `_experiments/260306_full_no_good`.
- The full instance set is `S1 S2 S3 S4 M1 M2 M3 L1 L2 L3 L4`. The subdirectory
  naming is `<METHOD>_<VARIANT>_<Tmax>` for `METHOD ∈ {SEP, BEN, MibS}`,
  `VARIANT ∈ {DEC, AGG}`, `Tmax ∈ {1,3,5,15,30,45,60}`, plus `PBN_PBN_60`.

```bash
# Reused by several commands below — comment a line out to skip that group.
instances=(
    S1 S2 S3 S4   # small
    M1 M2 M3      # medium
    L1 L2 L3 L4   # large
)
```

## 1. Run the benchmark — [main.py](main.py)

Unified entry point for the `benders` (SEP/BEN), `mibs` (MibS), and `pbn` (PBN)
algorithms. Each subdirectory must contain an `alg_config.json` (which fixes the
algorithm and `total_time_limit`); results are written under
`<subdir>/{benders,MibS,pbn}/<instance>/`, and run metadata to
`run_args__*.json` / `run_times__*.log` in the root folder.

```bash
# Full experiment: every method/T subdir. Comment a line for a partial run.
subdirs=(
    SEP_DEC_1  SEP_AGG_1  BEN_DEC_1  BEN_AGG_1  MibS_AGG_1
    SEP_DEC_3  SEP_AGG_3  BEN_DEC_3  BEN_AGG_3  MibS_AGG_3
    SEP_DEC_5  SEP_AGG_5  BEN_DEC_5  BEN_AGG_5  MibS_AGG_5
    SEP_DEC_15 SEP_AGG_15 BEN_DEC_15 BEN_AGG_15 MibS_AGG_15
    SEP_DEC_30 SEP_AGG_30 BEN_DEC_30 BEN_AGG_30 MibS_AGG_30
    SEP_DEC_45 SEP_AGG_45 BEN_DEC_45 BEN_AGG_45 MibS_AGG_45
    SEP_DEC_60 SEP_AGG_60 BEN_DEC_60 BEN_AGG_60 MibS_AGG_60
    PBN_PBN_60
)

python _experiments/main.py \
    --root-dir 260306_full_no_good \
    --subdirs "${subdirs[@]}" \
    --instances "${instances[@]}" \
    --workers 16
    # --time-limit 600   # optional: override total_time_limit (s) from alg_config.json
```

> MibS subdirs (`MibS_*`) require the `pao` package and **Python ≤ 3.11**, plus a
> MibS solver binary — see the **MibS / `pao`** note under [Environment](#environment)
> (including the NumPy `np.PINF`/`np.NINF` → `np.inf`/`-np.inf` patch). `--workers 1`
> runs sequentially; otherwise up to `--workers` instances run in parallel.

## 2. Collect results — [collect_results.py](collect_results.py)

Scans the per-instance run logs and writes `summary.csv`, `summary_per_inst.csv`,
and the `agg_*.csv` tables into `260306_full_no_good/_results/`. Run this after
the benchmark and before any of the `pubs_*` table/plot scripts.

```bash
python _experiments/collect_results.py _experiments/260306_full_no_good
    # --list                 # print detected experiments and exit
    # --variant agg          # restrict comptime/cuts tables to one variant (agg|decomp)
    # --from-summary         # rebuild only aggregate tables from existing summary.csv
    # -o <DIR>               # output dir (default: <folder>/_results)
```

## 3. Verify correctness of the controls

Both checkers read each `sol.json` and verify every attractor against the
phenotype. Results (and a JSON summary consumed by the analytics scripts) are
written under `_results/`.

### Bounded length — [verify_control_bounded_parallel.py](verify_control_bounded_parallel.py)

For every control, computes the min violating / min attractor length in `1..T`
and classifies `INCORRECT` / `NONMINIMAL`. Writes
`_results/verify_control_T<T>.json` (read by `pubs_analytics --tmax-upper`).

```bash
python _experiments/verify_control_bounded_parallel.py \
    --root_dir _experiments/260306_full_no_good \
    --instances "${instances[@]}" \
    --T 100 \
    --workers 16 \
    --output _experiments/260306_full_no_good/_results/correctness_T100.txt \
    --check-subset
    # --work_dirs <DIR> ...  # instead of --root_dir: verify explicit work dirs
```

### Infinite horizon — [verify_control_parallel.py](verify_control_parallel.py)

Model-checks all attractors (CTL by default) via NuSMV. Writes
`_results/verify_control_Tinf.json` (read by `pubs_analytics_infinite`).

```bash
python _experiments/verify_control_parallel.py \
    --root_dir _experiments/260306_full_no_good \
    --workers 16 \
    --output _experiments/260306_full_no_good/_results/correctness_Tinf.txt
    # --instances "${instances[@]}"  # default: all
    # --logic ltl                    # default: ctl (LTL logs counterexample loop length)
```

## 4. Publication tables & plots (`pubs_*`)

All `pubs_*` scripts read the CSVs in `260306_full_no_good/_results/` produced by
[collect_results.py](collect_results.py) (step 2) and emit LaTeX/figures there by
default. `--include-exp` / `--filter-exp` keep / drop experiments by regex.

### Solution-count table (finite) — [pubs_analytics.py](pubs_analytics.py)

```bash
python _experiments/pubs_analytics.py _experiments/260306_full_no_good \
    --tmax 1 3 5 15 \
    --tmax-upper 100
    # --include-exp '^(SEP|BEN)_.*$'   # keep matching experiments
    # --filter-exp  '^BEN_AGG_.*$'     # drop matching experiments
    # -o <FILE>                        # default: _results/agg_solution_count_bold.tex
```

`--tmax-upper 100` reads `_results/verify_control_T100.json` from step 3.

### Solution-count table (infinite) — [pubs_analytics_infinite.py](pubs_analytics_infinite.py)

```bash
python _experiments/pubs_analytics_infinite.py _experiments/260306_full_no_good \
    --include-exp '^(SEP_DEC_[0-9]+|PBN_.*)$' \
    --tmax 1 5 15 60 \
    -o _experiments/260306_full_no_good/_results/agg_solution_count_bold.infinite.tex
```

### Computation-time table — [pubs_comptime.py](pubs_comptime.py)

One `--tmax` per run; `--control-size` selects which control levels appear.

```bash
python _experiments/pubs_comptime.py _experiments/260306_full_no_good \
    --tmax 15 \
    --control-size 1 2 3 4 5 6 7
    # --filter-exp '^BEN_AGG_.*$'   # output: _results/agg_comptime_T<tmax>.tex
```

### Cut-strength table — [pubs_cut_strength.py](pubs_cut_strength.py)

```bash
python _experiments/pubs_cut_strength.py _experiments/260306_full_no_good \
    --tmax 1 \
    --include-exp '^((SEP|BEN)_DEC_[0-9]+)$'
    # -o <FILE>   # default: _results/agg_cut_strength_T<tmax>.tex
```

### Computation-time plot — [pubs_comptime_plot.py](pubs_comptime_plot.py)

Plots cumulative solution counts from `_results/timestamp.csv`.

```bash
python _experiments/pubs_comptime_plot.py _experiments/260306_full_no_good \
    --tmax 60 \
    --time-limit 600 \
    --output-ext pdf \
    --finished-levels 7
    # output: _results/agg_comptime_plot_T<tmax>.<ext>
```

## 5. Longest-attractor analyses

### Solve the longest joint control-attractor MILP — [longest_attractor_check.py](longest_attractor_check.py)

Standalone (no result folder). Writes to
`_experiments/results/longest_attractor/Tmax_<tmax>/<timestamp>/`.

```bash
python _experiments/longest_attractor_check.py \
    --tmax 100 \
    --phenotype-mode violating \
    --max-control-size 7 \
    --instances "${instances[@]}" \
    --time-limit 3600 \
    --state-periodicity
    # --tee   # stream solver logs
```

### Summarize the bounded-checker cache — [longest_attractor.py](longest_attractor.py)

Reads the JSON cache produced by the bounded checker (step 3, default path
`_experiments/checker_bounded_T<T>.json`) and prints the controls with the
longest min violating / min attractor length per instance.

```bash
python _experiments/longest_attractor.py "_experiments/checker_bounded_T100.json"
    # --print-control   # also list the controls attaining each maximum
```

## 6. Cleanup — [_dev/cleanup_experiment_root.py](_dev/cleanup_experiment_root.py)

**Destructive:** removes everything under the root folder *except* files named
`alg_config.json` (i.e. resets a result folder to re-run from scratch). Always
preview with `--dry-run` first.

```bash
python _experiments/_dev/cleanup_experiment_root.py _experiments/260306_full_no_good --dry-run
# Drop --dry-run to actually delete.
```
