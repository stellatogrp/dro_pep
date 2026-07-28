# LogReg certification experiment: design notes

Design decisions for the real-data logistic regression certification
experiment (`logreg.py`, `logreg_data.py`), recorded so they survive
context switches. See also the tests in `tests/test_logreg_certification.py`.

## Instance distribution

- Unregularized (`delta = 0`, smooth convex, `mu = 0`) logistic regression:
  acceleration shows its largest improvement on non-strongly-convex problems.
- Instances are uniform `m_sub = 600`-row subsamples of LIBSVM **a9a**
  (123 binary features + appended intercept). The algorithm starts at
  `x0 = 0`, so the initial condition is `||x0 - x*|| = ||x*||`.
- Separability guard: instances whose solve fails or has `||x*|| >
  xopt_norm_max` are rejected and resampled. **Do not tighten the guard
  below ~120**: the `||x*||` distribution is uniformly large, not
  outlier-driven (p10 = 63, median = 86, p90 = 117, max = 221 over the 200
  calibration instances) — a threshold of 50 rejects 100% of instances.
- Family constants are calibrated deterministically from `seed.in_sample`:
  `L = max_i lambda_max(A_i^T A_i)/(4 m) = 1.87`, `R = max_i ||x*_i|| = 191.7`.
  This is the same convention as the Lasso experiment (`R = max ||x*(b)||`).

## Why the worst-case bound is so large (the log-2 fact)

For logistic regression started at `x0 = 0`, the true gap satisfies
`f(0) - f* <= log 2 ~ 0.69` for every instance (`f(0) = log 2`, `f* >= 0`).
The worst-case PEP bound over the full smooth-convex class at these
constants is `L R^2 / (4K+2) ~ 11,458` at `K = 1` — the class contains
quadratic-like functions that attain it. The gap is structural: class-based
worst-case analysis cannot exploit the boundedness of logistic losses,
while the data-driven certificates adapt to it automatically. This is a
feature of the experiment, not an artifact.

## Conventions that must not drift

- FGM iterates follow `construct_fgm_pep_data` / `logreg_fgm_trajectories`:
  gradients at `y_k`, objective at `x_K = y_{K-1} - t g(y_{K-1})`.
  `utils.nesterov_fgm` uses a different labeling of the same recursion and
  measures at the extrapolated point — do not mix the two. The test
  `test_gram_objective_matches_simulation` pins this.
- No preconditioning (`precond: false`).
- The Wasserstein radius is cross-validated **per K** at plot time
  (95% coverage over the `cross_val_repeats = 100` sampling repeats);
  the eps grid is logspace 1e-4..10^-0.5 so the selection is not pinned
  at a grid edge (observed selections: GD 0.0175, FGM 0.0088).

## Cluster profiles (observed usage on della)

- dro: 8 cpu x 14G, 6 h (peak 87 GB, 21-28 min per full-K task)
- samples: 4 cpu x 4G, 2 h (0.6 GB, 41-53 min)
- pep: 2 cpu x 4G, 30 min (0.5 GB, 45 s)

Over-requesting resources caused multi-hour queue waits; these right-sized
profiles backfill within minutes.
