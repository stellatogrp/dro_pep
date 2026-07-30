# Handoff: adding a new experiment to dro_pep (cluster notes)

Everything below was verified live on della and GitHub in July 2026, while
reverse-engineering the DR-L2O paper figures (see `lasso_intro_repro/`).
This file is intentionally untracked.

## Where things are on the cluster (della-stellato)

- **Vinit's checkout**: `/home/vranjan/dro_pep`, branch `vr/learning_refactor`
  at `f8232ee` (May 6 2026). It is **14 commits behind `main`** and read-only
  for other users; his ~10 git stashes hold only config/SLURM tweaks. Do not
  work there — clone your own copy (e.g. `/home/bs37/dro_pep`) and ask Vinit
  to `git pull` his.
- **Learning experiment outputs**:
  `/scratch/gpfs/BSTELLATO/vranjan/learn_dro_pep_out/`
  - `learn_{l2o,dro,lpep}_outputs/<Problem>/<date>/<time>_<taskid>/`
    with `.hydra/config.yaml` (full experiment config) and
    `learn_dro_outputs/K_<k>/progress.csv` — one row per SGD iterate:
    `training_loss, validation_loss, gamma_0..gamma_{k-1}`. The schedule in a
    row is the one that produced that row's `validation_loss`.
  - `<Problem>/runs/*.txt` are the SLURM logs.
- **Problem-instance sets**:
  `/scratch/gpfs/BSTELLATO/vranjan/out_of_sample_out/sample_creation_outputs/<Problem>/<date>/<time>/`
  (`training_set.npz`, `validation_set.npz`, `test_set.npz`, `ood_set.npz`,
  `A_in_dist.npz`, metadata; all `b_batch / x_opt_batch / f_opt_batch`).
- **Non-learning DRO/PEP outputs**: `/scratch/gpfs/BSTELLATO/vranjan/dro_pep_out/`.
- **Canonical sweeps for the paper**: the May 3-4 2026 runs
  (`training_sample_N=1000`, val/test/ood 250). The April 2026 sweeps used
  `N=100` and do NOT correspond to the paper — don't reuse them.

## Repo state

`main` is the up-to-date branch (the `neurips` branch was merged via PR #8;
PR #9 `vr/ista_rework` and the resubmission commits sit on top; `neurips`
only has trailing README commits). **Base any new experiment on `main`.**
Vinit's cluster checkout predates `41f7311` ("fix `_solve_bwd` cache-clear
race that zeroed the first gradient") — make sure any learning run uses code
that includes it.

## Adding a new experiment: registration checklist

Follow the Quad/Lasso pattern:

1. `src/experiment_classes/<name>.py` — implement `<name>_samples()`,
   `<name>_pep()`, `<name>_dro()`.
2. Learning side: `src/learning_experiment_classes/<name>.py` (problem
   module + `<name>_run`, `<name>_sample_creation_run`), a trajectory module
   in `src/learning/trajectories/`, and, for a new algorithm, a PEP
   construction in `src/learning/pep_constructions/`.
3. Hydra configs: `src/configs/<name>.yaml` and
   `src/configs_learning/<name>.yaml` (seeds, sizes, `K_max`, `eta_t`, `eps`,
   `data_source_dir`).
4. Register the driver and output dir in the `func_driver_map` /
   `base_dir_map` of: `run_sample_experiment.py`, `run_pep_experiment.py`,
   `run_dro_experiment.py`, `run_learning_experiment.py`,
   `run_learning_l2o_experiment.py`, `run_learning_lpep_experiment.py`
   (named `optpep` on newer branches), `run_sample_creation.py`.
5. SLURM script in `src/slurm_scripts/` (`SLURM_ARRAY_TASK_ID` selects the
   alg/param combo; 600 GB, Intel constraint — copy an existing one).

## Things to remember

- **Hard-coded scratch paths**: every runner's `cluster` branch points at
  `/scratch/gpfs/BSTELLATO/vranjan/...`. Running as another user, change
  those `base_dir` values (or parameterize them) first.
- **`data_source_dir`** in learning configs pins training/eval to one
  sample-creation output. Regenerate data → update this path, or schedules
  and test sets silently go out of sync.
- **Plotting code is not in the repo**: `experiment_plots_icml/` is
  gitignored and lives only on Vinit's laptop. The paper-figure conventions
  were reverse-engineered and reimplemented in `lasso_intro_repro/`
  (exact-match verified):
  - per K, the plotted schedule is the **min-validation-loss row** across the
    hyperparameter sweep; DR-L2O figures use **fixed eps=10** (not CV'd);
  - the in-distribution panel **excludes the test instances where L2O
    diverges** (loss >= initial loss; for LASSO: instances 111 and 189) for
    all series; the OOD panel uses all instances unfiltered;
  - bands are 10th-90th quantiles (`np.quantile`, linear), lines are means;
  - the LASSO OOD set in the paper is `x ~ N(0, 3^2)` (seed 30000),
    regenerated deterministically by `lasso_intro_repro/regenerate_ood_set.py`
    — it is NOT any archived `ood_set.npz` on the cluster.
- **Determinism**: sets and training are reproducible from the seeds in the
  configs (JAX PRNG is platform-stable); f* values come from CVXPY/CLARABEL.
- **Figures**: pin `matplotlib==3.10.8` to byte-match the paper style
  (usetex tick-label baselines moved in 3.11). Import `_jax_setup` before
  anything that touches JAX; the lasso module enables `jax_enable_x64`.
- Ask Vinit to (a) `git pull` his cluster checkout, (b) commit
  `experiment_plots_icml/` so future figures stay reproducible.
