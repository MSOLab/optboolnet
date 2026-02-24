# optboolnet: An optimization toolbox for Boolean network analysis

`optboolnet` solves control problems in Boolean networks (BNs) based on [bilevel integer programming](https://doi.org/10.1016/j.ejco.2021.100007).

The main features include:

* Enumeration of *minimal controls* (permanent fix of Boolean functions as constants) that drive every synchronous attractor of the resulting BN to satisfy a given phenotype condition.
  * Limiting the maximum length of attractors to evaluate the phenotype (e.g., fixed point control with length of 1).
  * Efficient removal of controls that drive undesirable attractors or trap spaces via linear constraints.
* Eleven benchmark instances from [Moon et al. (2022)](https://doi.org/10.1016/j.ejor.2021.10.019), collected from published case studies.

## Usage

Jupyter notebooks with worked examples are provided in the [`example/`](example/) directory.
Install `jinja2` to properly render the solution. Follow the instructions below to run the notebooks.

### Installation via Colomoto Docker (recommended)

The easiest way is to use [CoLoMoTo Docker](https://colomoto.github.io/colomoto-docker), which is a reproducible software environment maintained by dozens of research groups in the Boolean network community.

```bash
pip install colomoto-docker
colomoto-docker
```

Then, import `optboolnet` in the CoLoMoTo Jupyter environment. See [`bladder_example.ipynb`](example/bladder_example.ipynb) or [CoLoMoTo Docker documentation](https://colomoto.github.io/colomoto-docker/tutorials/optboolnet/bladder_example.html) for details.

### Manual installation

Install with either [`conda`](https://anaconda.org/channels/msolab/packages/optboolnet/overview) or [`pip`](https://pypi.org/project/optboolnet/) with dependencies in `setup.cfg`:

```bash
conda install -c msolab optboolnet
```

```bash
pip install optboolnet
```

Example notebooks can be run in any Python environment with `optboolnet` and its dependencies installed.

* [`S1_breast_cancer_Sahin_et_al.(2019).ipynb`](example/S1_breast_cancer_Sahin_et_al.(2019).ipynb) — Attractor control of general length on the S1 breast cancer instance.
* [`S4_breast_cancer_Biane_Delaplace(2019).ipynb`](example/S4_breast_cancer_Biane_Delaplace(2019).ipynb) — Fixed point control on the S4 breast cancer instance.

## Commands in `optboolnet.launch`

### Easy-to-use commands for attractor control

* **`control_sync_attr_separation` (SEP)**: Removes undesirable trap spaces if found. Most efficient, but only applicable if the `max_attr_length` is large enough to include all attractors.
* **`control_sync_attr_no_separation` (BEN)**: Safer option that does not remove any trap spaces.
* **`control_fixpoint`**: Specialized solver for fixed point control (length is 1).

### Options

* **`max_attr_length`**: Maximum length of attractors to consider.
* **`target`**: Phenotype condition to satisfy, given as a dictionary representing a partial state (e.g., `{'Apoptosis': 1}`).
* **`exclude`**: Components in BN that should not be included in the control (e.g., `['EGFR']`).
* **`allow_empty_attractor`**: If true, the phenotype condition is considered satisfied if there are no attractors of length at most `max_attr_length`. If false, at least one attractor of the given maximum length must exist and satisfy the phenotype condition.

See the attributes of [`optboolnet.algorithms.BendersAttractorControl`](/src/optboolnet/algorithm.py) for more heuristic options.

## Citation

If you use `optboolnet` in your research, please cite:

* Moon, K., Lee, K., Chopra, S., & Kwon, S. (2022). Bilevel integer programming on a Boolean network for discovering critical genetic alterations in cancer development and therapy. *European Journal of Operational Research, 300*(2), 743–754. [https://doi.org/10.1016/j.ejor.2021.10.019](https://doi.org/10.1016/j.ejor.2021.10.019)
* Moon, K., Lee, K., & Paulevé, L. A bilevel integer programming approach for the synchronous attractor control problem. (working paper).
