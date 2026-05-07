# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**dro-pep**: Data-driven Analysis of First-Order Methods via Distributionally Robust Optimization

Research implementation combining Performance Estimation Problems (PEP) with Wasserstein DRO to analyze first-order optimization methods. The approach unifies worst-case and average-case analyses using data-driven information from sampled problem instances.

Paper: https://arxiv.org/abs/2511.17834

## Common Commands

All experiment commands run from `src/`:

```bash
# Run experiments (sample/pep/dro)
python run_sample_experiment.py Quad local
python run_pep_experiment.py Quad local
python run_dro_experiment.py Quad local

# Run learning experiments (L2O)
python run_learning_experiment.py Quad local

# Experiment names: Quad, Lasso, LogReg, Huber

# Override config params via Hydra
python run_dro_experiment.py Quad local alg=nesterov_fgm dro_obj=cvar eps=0.5
python run_learning_experiment.py Quad local K_max=[5,10,15] sgd_iters=200

# Run tests
pytest src/tests/ -v

# Run single test file
pytest src/tests/test_silver_stepsizes.py -v

# Lint with ruff
ruff check src/
ruff format src/
```

## Architecture

### Experiment Framework

Four experiment types share a common pattern:
- **Sample** (`run_sample_experiment.py`): Generate problem instances and algorithm trajectories
- **PEP** (`run_pep_experiment.py`): Pure worst-case performance estimation (baseline)
- **DRO** (`run_dro_experiment.py`): Data-driven analysis combining PEP constraints with sampled data
- **Learning** (`run_learning_experiment.py`): Learn step sizes via gradient descent through SDP solvers

Each problem class (Quad, Lasso, LogReg, Huber) implements `*_samples()`, `*_pep()`, and `*_dro()` methods.

### Key Directories

- `src/experiment_classes/`: Problem definitions with sampling, PEP, and DRO methods
- `src/learning/`: JAX autodiff infrastructure for differentiating through SDP solvers
  - `jax_clarabel_layer.py`, `jax_scs_layer.py`: JAX-compatible solver layers
  - `pep_construction.py`: PEP constraint matrix construction
  - `trajectories_*.py`: Algorithm trajectory simulation
- `src/learning_experiment_classes/`: L2O experiment runners per problem type
- `src/reformulator/`: DRO problem canonicalization
  - `canonicalizers/`: Backend-specific transformations (CVXPy, Clarabel)
- `src/configs/`: Hydra configs for DRO/PEP experiments
- `src/configs_learning/`: Hydra configs for L2O experiments

### Configuration

Experiments use Hydra. Key DRO/PEP parameters:
- `mu, L, R`: Problem conditioning (strong convexity, smoothness, init radius)
- `K_max`: Number of algorithm iterations
- `alg`: Algorithm (`grad_desc`, `nesterov_fgm`, `ista`, `fista`)
- `dro_obj`: Risk measure (`expectation` or `cvar`)
- `dro_pep_obj`: Performance metric (`obj_val`, `grad_sq_norm`, `opt_dist_sq_norm`)
- `eps`: Wasserstein radius
- `alpha`: CVaR confidence level
- `precond_type`: Preconditioning (`average`, `max`, `min`)

Key L2O parameters (in `configs_learning/`):
- `learning_framework`: `ldro-pep`, `l2o`, or `lpep`
- `sdp_backend`: `scs` or `clarabel`
- `dro_canon_backend`: `manual_jax` or `cvxpylayers`
- `stepsize_type`: `scalar` or `vector`
- `vector_init`: `fixed` or `silver` (silver rule initialization)
- `optimizer_type`: `vanilla_sgd`, `adamw`, `sgd_wd`
- `sgd_iters`, `eta_t`: Training iterations and learning rate

### SLURM Integration

Cluster jobs use `SLURM_ARRAY_TASK_ID` to select algorithm/parameter combinations. Shell scripts in `src/` (e.g., `run_qdro_experiment.sh`, `run_learn_quad.sh`) configure memory (600GB), Intel CPU constraint, and array sizes.

## Dependencies

Requires Python >=3.12. Key packages:
- cvxpy, PEPit: Optimization modeling
- Clarabel, SCS: SDP solvers (Clarabel preferred for learning)
- diffcp: Differentiable cone programming (installed from git master)
- JAX: Autodiff through solvers
- hydra: Configuration management

## Known Issues

- `diffcp_patch.py` applies monkey-patches for Clarabel compatibility (COO→CSC conversion, array→float fixes)
- Some configurations cause memory issues on large problems
- MKL Pardiso detection in JAX Clarabel layer for faster linear solves
