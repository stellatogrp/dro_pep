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

- **Datasets first**: compute nodes have NO internet. Before submitting
  anything for a new dataset, download it on the login node into the
  cache, e.g.
  `cd /scratch/gpfs/BSTELLATO/bs37/dro_pep_data && curl -sO https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/<name>`
  (a missing file kills every job in seconds with a urllib DNS error).

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

## Quad and Lasso: full campaign

`slurm_scripts/submit_epsx.sh` is the single entry point for the quadratic and
Lasso DRO runs. It submits every eps grid behind the committed figure data:
the base grid from the config plus the refinements that cannot be written as
one hydra grid (quad: 23 radii spanning [1e-5, 10]; Lasso: 19 spanning
[1e-5, 1e-1]). Collect exactly as in step 3 --

    python collect_results.py --pull --merge

-- which now builds `{Quad,Lasso}/data/dro/` alongside `LogReg/data/`. Only the
DRO stage is collected for these two: their samples and pep come from
`run_q{pep,dro}_experiment.sh` under a different scratch root and their
`data/samples`, `data/pep` dirs are committed as-is.

Two things the rerun does not give you. It covers alpha in {0.01, 0.05}
(`CTRIM=3`), the levels the paper figures use; the alpha=0.10 rows in the
committed CSVs are left over from the earlier K=50 campaign, sit on the base
grid only, and pin at the grid bottom for every K. And the merge globs the
`base`/`ext`/`ext2`/`mid` chunk tags by name rather than by wildcard, so the
older `chunk_*_{epsx,epsx2,basefill}_*` dirs still on the cluster are skipped
on purpose: those predate `training.{expectation,cvar}_N = 100`, and since
`dro.csv` carries no `N` column their rows would silently overwrite the N=100
ones under the (K, eps, alpha) dedup.

Grid resolution is not cosmetic. Cross-validation selects the smallest radius
whose certificate covers the empirical threshold, so a grid that bottoms out
at the selection, or is coarse just below it, silently inflates the bound; we
measured up to 12x on the quadratic CVaR at K=40, and a 17.5x gap in the Lasso
grid produced a visible step at K=11. `plot_bounds.py` prints a warning for
both cases, and for a K where no radius covers the threshold at all.
