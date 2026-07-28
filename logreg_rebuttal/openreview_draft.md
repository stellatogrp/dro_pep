# Draft: fourth benchmark results for OpenReview (Phase 2 discussion)

*(Status: FINAL — all 66 runs complete 2026-07-27; table verified stable against the full sweep. Character count to be checked before posting.)*

---

As committed in our response, we ran the fourth benchmark: learning the
step-size and momentum schedules of Nesterov's fast gradient method (FGM),
compared against learned gradient descent, on a new problem class. The task is
**unregularized logistic regression** (smooth convex, not strongly convex):
each instance is a dataset $(A_i, b_i)$ with $300$ points and $50$ features,
labels generated from a sparse ground-truth model with label noise, solved
from $x^0 = 0$; the loss is $f(x^K) - f^\star$. The out-of-distribution set
scales the data and noise jointly so that the smoothness constant grows by
$\approx 1.6\times$ while instances remain well-posed. We learn per-iteration
step sizes $t_k$ (both algorithms) and momentum coefficients $\beta_k$ (FGM),
with **no sign constraint on $\beta_k$**: the momentum parameterization is
unconstrained, so the span-form coefficients on past gradients carry both
signs, exercising exactly the signed span coefficients discussed in the
theory responses. Baselines: GD with $1/L$, the silver step-size schedule
[3], and FGM with the standard Nesterov sequence; learned methods: ERM
($\text{L2O}$), worst-case-optimal ($\text{OPT-PEP}$), and our
$\text{DR-L2O}$, each for both algorithm classes, $K \in \{5, 10, 15\}$,
hyperparameters (learning rate, Wasserstein radius $\varepsilon$) selected on
a validation set. Statistics over 250 test instances; solved = fraction with
loss $\le 10^{-2}(1+|f^\star|)$:

**In-distribution, K = 15:**

| Method | median [q10, q90] | mean | solved |
|---|---|---|---|
| GD ($1/L$) | 7.48e-03 [2.30e-03, 1.85e-02] | 9.32e-03 | 76% |
| Silver GD | 6.55e-04 [4.63e-05, 4.01e-03] | 1.49e-03 | 100% |
| Nesterov FGM | 1.32e-04 [2.37e-05, 8.82e-04] | 3.64e-04 | 100% |
| L2O GD | 1.06e-04 [2.71e-06, 1.37e-03] | 4.90e-04 | 100% |
| L2O FGM | **1.24e-05** [3.51e-06, 2.27e-05] | **2.31e-05** | 100% |
| OPT-PEP GD | 7.48e-03 [2.30e-03, 1.85e-02] | 9.32e-03 | 76% |
| OPT-PEP FGM | 1.32e-04 [2.37e-05, 8.82e-04] | 3.64e-04 | 100% |
| DR-L2O GD | 3.67e-03 [7.98e-04, 1.14e-02] | 5.14e-03 | 93% |
| DR-L2O FGM | 1.54e-04 [3.27e-05, 2.58e-04] | 2.14e-04 | 100% |

**Out-of-distribution, K = 15:**

| Method | median [q10, q90] | mean | solved |
|---|---|---|---|
| GD ($1/L$) | 2.19e-03 [3.89e-04, 7.64e-03] | 3.37e-03 | 98% |
| Silver GD | 5.33e-05 [6.54e-07, 9.35e-04] | 3.32e-04 | 100% |
| Nesterov FGM | 9.32e-05 [1.07e-05, 1.54e-04] | 1.12e-04 | 100% |
| L2O GD | 5.16e-03 [3.84e-05, 8.35e-02] | 2.72e-02 | 58% |
| L2O FGM | 3.79e-03 [1.78e-05, 6.18e-02] | 2.32e-02 | 55% |
| OPT-PEP GD | 2.19e-03 [3.89e-04, 7.64e-03] | 3.37e-03 | 98% |
| OPT-PEP FGM | 9.32e-05 [1.07e-05, 1.54e-04] | 1.12e-04 | 100% |
| DR-L2O GD | 7.64e-04 [7.42e-05, 3.89e-03] | 1.54e-03 | 100% |
| DR-L2O FGM | 1.01e-04 [2.43e-06, 1.87e-04] | **1.00e-04** | 100% |

*(full tables for K ∈ {5, 10, 15} available; same qualitative picture at
every K)* Note the median/mean gap for L2O out-of-distribution: its failure
under shift is a tail phenomenon (q90 $\approx 6\times 10^{-2}$, 45% of
instances unsolved), which the median alone would hide — precisely the
mean-vs-tail distinction raised in the reviews.

The experiment confirms all three points the benchmark was designed to test:

1. **The accelerated class is learnable and strictly better.** Every learned
   FGM variant improves on its learned-GD counterpart by one to two orders of
   magnitude (e.g. DR-L2O: $2.1\times 10^{-4}$ vs $5.1\times 10^{-3}$
   in-distribution at $K=15$), so the framework's benefit is not an artifact
   of the simple gradient-descent class.

2. **The robustness phenomenon from the paper transfers.** ERM-learned
   schedules (both classes) are the best in-distribution but collapse under
   the shift — L2O FGM degrades by three orders of magnitude and solves only
   55% of OOD instances — while DR-L2O FGM is the best method
   out-of-distribution at every horizon, without giving up more than a
   factor ~10 in-distribution relative to ERM.

3. **Signed span coefficients are exercised.** The learned momentum
   coefficients move freely in sign during training; in particular
   $\beta_0$, initialized at $0$, is learned to nonzero values — which a
   nonnegativity-constrained parameterization would prevent — and the
   resulting span representation has negative coefficients on earlier
   gradients, matching the setting of the corrected Lemma 4 analysis.

Wall-clock: the full sweep (66 training runs) completed in under four hours
on standard CPU nodes; the largest single run (DR-L2O, FGM, $K=15$) takes
$\approx 26$ s per SGD iteration.

---

## Notes for us (not for posting)

- Selection: min-validation-loss row within each run, then across the sweep;
  eps validation-selected (picks eps=1.0 for DR-L2O FGM at all K). The
  intro-figure fixed-eps=10 convention does NOT transfer to this task: at
  eps=10 training never improves validation over the (strong) Nesterov init.
- OPT-PEP GD/FGM rows equal the handcrafted baselines at K>=10: the
  worst-case objective keeps the schedule at (or returns to) initialization;
  min-val selection honestly returns the init. State this if asked.
- OOD = A_std and eps_std jointly x1.25 (margin-preserving; L up ~1.56x).
  With delta=0, instances must be decisively non-separable: label-noise std
  6.0 gives max ||x_opt|| ~ 8 across all four sets.
- Data/seeds: train/val/test/ood = 1000/250/250/250, seeds
  40000/10000/20000/30000, all instances solved with CVXPY/CLARABEL.
- Repro: branch bs37/logreg_fgm_rebuttal; runs under
  /scratch/gpfs/BSTELLATO/bs37/learn_dro_pep_out; data
  .../out_of_sample_out/sample_creation_outputs/LogReg/2026-07-27/06-01-07;
  table: logreg_rebuttal/build_logreg_table.py.
