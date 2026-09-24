# ICLR line-search comparison, 2026-09-24

Base: origin/iclr 40398f2. Branch: experiments/linesearch-iclr-20260924.
Deadline: tomorrow. Scope: LASSO ISTA and german.numer logistic GD/FGM.
All experiment execution is on della-stellato through the Slurm skill.
No new training. Reuse Vinit's saved schedules and exactly the paper instances.

- [x] Read the latest Zulip thread and clone Vinit's iclr branch.
- [x] Inspect existing baseline, data loaders, and schedule selection.
- [x] User chose remote scratch directory and bstellato account.
- [ ] Bootstrap the isolated remote environment (human runs the skill's command).
- [ ] Reproduce Vinit's coarse line-search losses and learned schedules.
- [ ] Run a predeclared small grid of standard backtracking settings.
- [ ] Compare final objective gaps, quantiles, success rates, oracle/matrix-product cost, and repeated batch timings.
- [ ] Sync CSVs and plots; write a short recommendation and reproduction commands.

## Protocol fixed before new experiments

All settings and outcomes are retained. No selection on test or OOD performance.
Use standard default settings and report the whole sensitivity grid.
LASSO: composite majorization backtracking, shrink factors 0.5 and 0.8;
initial steps 1/L and 1; carry the accepted step with or without factor-2 expansion.
Logistic: Armijo GD (c=1e-4, shrink=0.5 or 0.8), initial 1/L or 1;
FGM with smooth majorization (c=0.5), and Vinit's coarse Armijo FGM.
For FGM, mark expanding-step momentum variants as practical heuristics, not the classical monotone-L theorem.
Reproduce Vinit's growth=2 coarse settings including the noncompounding variant.
Report learned L2O / DR-L2O / OPT-PEP at all saved horizons K<=15.
Primary outputs: means, medians, q10/q90, solved fractions, per-instance costs, and finite-value checks.
Matrix products and function/gradient/prox calls are separate cost measures;
logistic backtracking reuses A*g and is not charged a fresh matrix product per trial.
Wall time is CPU batch timing, excludes data load/schedule selection, and includes
identical trajectory instrumentation. It is not a universal single-instance latency claim.

## Data provenance

Latest Zulip: Vinit messages 626738535, 626738658, 626740193.
LASSO raw instances were absent from the minimal Git push. Recovered from
/Users/bs37/Dropbox/work/research/code/projects/dro_pep/lasso_intro_repro/data/test_sets.
Both f_opt and initial gaps match the committed coarse baseline within 5.2e-12.
Full trajectory identity remains to be checked on the cluster.
The archived LASSO test figure has 248 rows after excluding original rows 111 and 189.
Produce paper-matching and all-250 comparisons explicitly, including those two instances.
Logistic test/OOD data come directly from the committed 2026-09-18/08-30-44 bundle.

## Algorithm references

Beck and Teboulle, FISTA (2009), DOI 10.1137/080716542.
Author's implementation documents both monotone Lipschitz estimates and optional
factor-2 decrease (regret_flag): https://www.tau.ac.il/~becka/solvers/fista.
