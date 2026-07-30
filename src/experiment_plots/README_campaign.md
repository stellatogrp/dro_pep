# Real-data logistic regression certification: reproducible pipeline

End-to-end recipe from configs to the LogReg paper figure (a9a instance
distribution, K = 1..30, N = 100 in-sample).

## 1. Configuration (committed)

- `src/configs/logreg.yaml`: dataset/subsampling, `K_max: 30`,
  `training.expectation_N = training.cvar_N = 100`, eps grid
  logspace 1e-5..10^-0.5 (13 points).
- Family constants (L, R) are calibrated at runtime from the fixed
  seeds in the config; nothing is hand-entered. See
  `experiment_classes/logreg_notes.md` for the design decisions.

## 2. Cluster runs

On the cluster clone (`git pull` first):

- samples stage: `sbatch --array=0-2 slurm_scripts/run_logreg_cert_samples.sh`
  (0 = GD at eta=1.9, 1 = FGM, 2 = FGM at eta=1.9, the step-size bonus)
- worst-case PEP: `sbatch --array=0-1 slurm_scripts/run_logreg_cert_pep.sh`
- DRO stage, chunked per K-range and eps grid:

      python slurm_scripts/logreg_campaign.py audit    # completeness report
      python slurm_scripts/logreg_campaign.py submit   # submit whatever is incomplete

  `submit` is idempotent: it only submits units whose chunk output is
  missing rows, and skips units currently in the queue. The manifest
  covers the base eps grid plus the small-radius extension (see its
  docstring for why both exist).

## 3. Collect

Locally, from `src/experiment_plots/`:

    python collect_results.py --pull --merge

`--pull` copies the result CSVs from the cluster into `_mirror/`
(gitignored); `--merge` builds `LogReg/data/`: one `dro.csv` per
(algorithm, measure) concatenated from the chunk outputs and
deduplicated on (K, eps, alpha), plus the newest samples/pep outputs.

## 4. Figure

    cd LogReg && python plot_bounds.py

Produces `logreg_obj_val.pdf` (+ companion CSV with every plotted
number) and `logreg_eps_sweep.pdf`. The Wasserstein radius is
cross-validated per K inside `plot_bounds.py` (95% coverage across the
100 sampling repeats in `sample_summary_dist.csv`); chosen radii are
printed to stdout.
