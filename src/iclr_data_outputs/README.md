# `iclr_data_outputs/` — consolidated results and figures

Everything the ICLR submission needs, in one tree: the new MIT-cluster runs, the
archived della runs that produced the DR-L2O paper figures, and scripts that
regenerate every figure.

```
iclr_data_outputs/
├── plotting/                  ← run figures from here
├── figures/                   ← LogReg figures (new work)
├── archive/                   ← consolidated inputs for the paper figures
│   ├── quad/ lasso/ pdlp/     ← data + its own plotting scripts + paper_plots/
│   └── logreg_synthetic_superseded/
├── learn_dro_outputs/         ← new MIT runs (DR-L2O)
├── learn_l2o_outputs/         ←              (L2O)
├── learn_lpep_outputs/        ←              (OPT-PEP)
├── sample_creation_outputs/   ← problem-instance bundles for the new runs
└── runs/                      ← slurm logs
```

## Regenerating figures

```bash
cd plotting
python make_all_figures.py              # everything
python make_all_figures.py logreg       # just the new LogReg figures
python make_all_figures.py quad lasso   # a subset
```

| figure | written to | source |
|---|---|---|
| `quad_losses.pdf`, `quad_frac_problems_solved.pdf`, `quad_training_set_effect.pdf`, `quad_times.csv` | `archive/quad/paper_plots/` | archived della runs |
| `lasso_losses.pdf`, `lasso_intro.pdf`, `lasso_frac_problems_solved.pdf`, `lasso_times.csv` | `archive/lasso/paper_plots/` | archived della runs |
| `pdlp_losses.pdf`, `pdlp_frac_problems_solved.pdf`, `pdlp_reconstructions.pdf`, `pdlp_more_reconstructions.pdf`, `pdlp_times.csv` | `archive/pdlp/paper_plots/` | archived della runs |
| `logreg_{gd,fgm}_losses.pdf`, `logreg_{gd,fgm}_frac_problems_solved.pdf`, `logreg_{gd,fgm}_times.csv` | `figures/` | **new MIT runs** |

## Regenerating the appendix tables

```bash
cd plotting
python make_paper_tables.py             # all four experiments, then collect
python make_paper_tables.py lasso pdlp  # a subset
```

`make_paper_tables.py` reshapes the long-form CSVs the plotting scripts cache
next to their PDFs (`<exp>_losses.csv`, `<exp>_frac_problems_solved.csv`) into
the two per-experiment tables the paper's appendix typesets with
`pgfplotstable`: `<exp>_losses_table.csv` (mean / 10th quantile / median / 90th
quantile of the test loss, in-distribution and out-of-distribution, every K)
and `<exp>_frac_solved_table.csv` (fraction solved at each eta, both splits,
every K). It recomputes nothing. The losses CSVs carry a `*_median` column
since the tables were added; a losses CSV cached before that must be deleted
and rebuilt with `make_all_figures.py` first. The collect step copies all
eight tables to `figures/paper_tables/` under the names the paper's
`tables/` directory uses.

## What was consolidated, and what was dropped

`archive/{quad,lasso,pdlp}/` holds the inputs from `experiment_plots_icml/`,
together with that problem's own plotting scripts copied verbatim beside its
data. Only one thing was edited in them: the scripts resolve `src/` relative to
their own location, and the consolidated tree is one level deeper, so
`script_dir.parent.parent` became `script_dir.parent.parent.parent`.

The copy is pruned from **25 GB to 1.4 GB** by dropping two files per run that
no plotting script ever opens:

- `learn_dro_outputs/training_set.npz` — a legacy per-run copy (5.8–73.8 MB
  each). `create_test_plots.load_in_sample_data` deliberately reads the
  centralized `problem_instances/training_set.npz` instead.
- `run_learning_*_experiment.log` — 2–15 MB of SGD chatter per run.

Kept in full: `.hydra/config.yaml` and every `learn_dro_outputs/K_*/progress.csv`
for **all** run dirs (not just the ones the current config selects), so
`data_scrape.py` can re-bucket if the selection changes; plus
`problem_instances/`, `plots/` and `paper_plots/`.

Also dropped, because no script references them: `lasso/old_data/` (4.3 GB),
`lasso/old_data_alista/`, `lasso/old_paper_plots/`,
`lasso/problem_instances_{copy,smallerstd,unif}/`, `quad/problem_instances_old/`,
and all `.bak` files.

### Verified

`pdlp_losses.csv`, `pdlp_times.csv`, `quad_times.csv` and `lasso_times.csv` all
come back **identical** to the committed originals. `times.csv` is the strongest
check: it is the only artifact that reads raw `progress.csv` rows, reaching into
the specific `source_dir` named in each best-stepsize CSV, so matching it proves
the pruned copy kept the right runs.

### `logreg_synthetic_superseded/`

The `experiment_plots_icml/logreg/` runs, kept under a name that says what they
are. They are the **old synthetic** logistic-regression experiment (`A_std: 1.0`,
`eps_std: 0.1`, no `data:` key), predating both the `eps_std = 6.0` fix and the
switch to german.numer. The ICLR LogReg figures come from the new runs instead.
Its two JAX scripts also no longer import (`learning.trajectories_logreg_gd_fgm`
moved to `learning.trajectories`), which is why only their CSVs were carried over.

## The LogReg figures

New work — the DR-L2O paper has no LogReg figures. They are built to correspond
to the other three: same three series (L2O / DR-L2O / OPT-PEP), same colours and
markers, same 1×2 in-distribution / out-of-distribution losses panel and 2×3
fraction-solved grid, plus the handcrafted reference (`GD (1/L)` or
`Nesterov FGM`) as a dashed grey line.

Two deliberate differences, both forced by the data:

- **K ∈ {5, 10, 15} only.** The LogReg sweep trains one horizon per run
  (`K_max: [5] | [10] | [15]`), where quad/lasso train K = 1..15 per run.
- **Solved thresholds one decade lower**, η ∈ {1e-4, 1e-3, 1e-2} against the
  paper's {1e-3, 1e-2, 1e-1}. The gap on german.numer spans ~1e-4..7e-2, so the
  paper's thresholds would saturate at 100% nearly everywhere.

Schedule selection is delegated to `logreg_rebuttal/build_logreg_table.py` rather
than reimplemented, because that module owns the per-framework rule: L2O and
DR-L2O are selected on validation loss, OPT-PEP on its **training** loss (the
worst-case PEP bound), since its empirical validation loss rises monotonically as
it trains. All 36 (framework, K, split) means in the generated CSVs match
`logreg_rebuttal/results.csv` exactly.

### Two caveats on the LogReg results

1. **The OOD set is easier in absolute terms.** It is built by scaling the design
   matrix by `ood_std_multiplier = 1.25`, which raises `L` by 1.45× (harder) but
   lowers `R` by 0.78× (easier, since `x0 = 0` starts closer to `x*`). That is why
   the robust methods post lower absolute loss OOD than in-distribution. The
   *ranking* is unaffected, and L2O still collapses.
2. **`OPT-PEP GD` is `GD (1/L)`.** Its learned schedule never leaves the
   initialization, so the dashed grey reference lies exactly on the green curve —
   which is why it is drawn on top rather than underneath. Root cause: at the
   same iterate the cluster's SDP gradient is ~8.2× larger than the same code
   produces locally (diffcp 1.2.0.dev1 from git master on the cluster vs 1.1.6
   locally), and AdamW then overshoots. Locally the same config improves the
   bound 18.5%. Unresolved — the FGM arm is affected less but not immune.

## Provenance

The new runs pin `data_source_dir` to
`sample_creation_outputs/LogReg/2026-09-18/08-30-44`, giving `L = 0.804732`,
`R = 9.158090` identically across all three frameworks, so the comparison is
like-for-like.

`archive/quad/data/` also contains 24 stale run dirs from a pre-cut submission at
`eps ∈ {1e-3, 1e-2}`; the figures pin `eps = 1.0`, so `data_scrape.py` buckets
them out and they do not affect any result.

Figures were produced with matplotlib 3.11.2. `HANDOFF.md` pins **3.10.8** to
byte-match the paper — usetex tick-label baselines moved in 3.11, so labels shift
by a few points. The numbers are unaffected; `_style.py` prints a note.
