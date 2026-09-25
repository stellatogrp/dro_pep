# ICLR line-search comparison, 2026-09-24

Base: origin/iclr 40398f2. Branch: experiments/linesearch-iclr-20260924.
Deadline: tomorrow. Scope: LASSO ISTA and german.numer logistic GD/FGM.
All experiment execution is on della-stellato through the Slurm skill.
No new training. Reuse Vinit's saved schedules and exactly the paper instances.

- [x] Read the latest Zulip thread and clone Vinit's iclr branch.
- [x] Inspect existing baseline, data loaders, and schedule selection.
- [x] User chose remote scratch directory and bstellato account.
- [x] Bootstrap the isolated remote environment (human runs the skill's command).
- [x] Reproduce Vinit's coarse line-search losses and learned schedules.
- [x] Run a predeclared small grid of standard backtracking settings.
- [x] Compare final objective gaps, quantiles, success rates, oracle/matrix-product cost, and repeated batch timings.
- [x] Sync CSVs and plots; write a short recommendation and reproduction commands.

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
Full trajectory identity passed on the cluster against current committed coarse caches. All 270 current learned curve means and their q10/q90 statistics also reproduce.
The archived LASSO test figure has 248 rows after excluding original rows 111 and 189.
Produce paper-matching and all-250 comparisons explicitly, including those two instances.
Logistic test/OOD data come directly from the committed 2026-09-18/08-30-44 bundle.

## Algorithm references

Beck and Teboulle, FISTA (2009), DOI 10.1137/080716542.
Author's implementation documents both monotone Lipschitz estimates and optional
factor-2 decrease (regret_flag): https://www.tau.ac.il/~becka/solvers/fista.

## Completed jobs and decision

- 14389777: initial validation array stopped on an integer dtype in the analytic
  test (the real-data coarse replays had passed). Fixed before production.
- 14389906_0 / _1 / _2: full sweeps completed in 37 / 59 / 57 seconds,
  1 CPU, 2 GB requested per task, no GPU. Code checkpoint 7735ae9.
- 14390157: safeguarded coarse LASSO follow-up, 5 seconds. Added because 5 test
  and 8 OOD default fallback steps failed the original majorization condition.
  Safeguarding does not materially change the ranking.
- All Slurm runs completed; plots and decision page are local under
  results/linesearch-report/index.html. Complete CSV has 1,995 rows.
- The conventional shrinking-only baseline loses to DR-L2O on both problems.
  Step growth reverses many rankings. For LASSO, DR-L2O beats full backtracking
  at equal matrix-product cost at K=15, but not the cheap safeguarded coarse rule.
  The recommendation is an accuracy/cost comparison with these qualifications.

## Reproduce

Use the checked-in Slurm helper configuration and a fresh EXPERIMENT name.
The initial production sweep used origin/iclr 40398f2 plus commit 7735ae9.
The current runner additionally includes the safeguarded coarse rule.

    python3 /path/to/slurm/scripts/slurm_agent.py sync up
    python3 /path/to/slurm/scripts/slurm_agent.py submit slurm/job_array.slurm --time 00:05:00 --cpus 1 --mem 2G --array 0-2%2 --export EXPERIMENT=linesearch-replay --export MODE=full
    # After test-only validation, repeat with --yes.
    python3 /path/to/slurm/scripts/slurm_agent.py sync down
    python src/tools/report_linesearch_audit.py --input results/linesearch-full-v2 --extra results/linesearch-safeguard-v3 --output results/linesearch-report

Reconstructed LASSO input files are untracked under the existing
src/iclr_data_outputs/archive/lasso/problem_instances layout. Their source hashes
are in provenance.json; exact input hashes are also in each run manifest.
The result bundle includes those inputs, current coarse-reference caches, the
code, logs, selected-schedule manifests and all per-instance outputs.
No results or messages were posted externally and no manuscript was edited.

## Boyd backtracking update, 24 September 2026

The requested report now focuses on Boyd and Vandenberghe, Algorithm 9.2.
The book starts each line search at a unit trial step. The earlier carried-step
configuration is not the same rule and will not be relabeled as Boyd backtracking.

Protocol chosen before rerunning: reset trial step to 1 at every iteration,
shrink by 0.5, and use Armijo alpha=0.1 for logistic GD. The user approved
clearly labeled adaptations: LASSO uses the proximal quadratic-majorization
condition; logistic FGM applies Armijo at the extrapolated point with the
existing momentum sequence. No accelerated convergence guarantee is claimed.

- [x] Check the primary book source and agree on the adaptations.
- [ ] Rerun this baseline and the current learned schedules through Slurm only.
- [ ] Recheck current paper curves, step reset, acceptance conditions and costs.
- [ ] Show this baseline only in the plots and webpage, with precise labels.
- [ ] Export the complete webpage to a readable PDF and inspect every page.

New output directory: results/linesearch-boyd-v1.
Replay uses the existing job array with MODE=boyd and EXPERIMENT=linesearch-boyd-v1.
Parameters and all previous experiment outputs remain available for audit.
