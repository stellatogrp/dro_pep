"""Coarse (two-candidate) line search baselines, pure numpy / scipy.

At every iteration the method tries an aggressive candidate step
``t_try = growth * t_prev`` (``t_prev`` = the previously taken step, initialised
to ``t_default``; with ``compound=False`` the candidate is always
``growth * t_default``) and checks a sufficient-decrease condition at the candidate.
If the check passes the candidate is taken; otherwise the *default* step
``t_default`` -- the value the learned step-size sequences are initialised
from -- is taken unconditionally. So every method spends exactly ``K``
gradient (matvec) oracle calls, and the only overhead is at most one extra
function evaluation per iteration, which for every problem here shares its
matvec with the next gradient. The adaptive rule is K-independent, so one run
of ``K_max`` steps gives the loss at every horizon ``K = 1..K_max``.

Per problem the check is:

* Quad GD / LogReg GD / LogReg FGM: Armijo
  ``f(p - t g) <= f(p) - c t ||g||^2`` at the gradient point ``p``
  (``p = y_k`` for FGM; the momentum sequence is unchanged).
* Lasso ISTA: the Beck-Teboulle composite condition at the prox-gradient
  trial point ``x+ = soft(x - t grad f1(x), t lambd)``:
  ``f1(x+) <= f1(x) + <grad f1(x), x+ - x> + ||x+ - x||^2 / (2 t)``.
* PDLP / Chambolle-Pock: the PDLP adaptive step rule (Applegate et al. 2021)
  with primal weight 1, ``eta = tau = sigma``: after a trial step accept iff
  ``eta_try <= (||dx||^2 + ||dy||^2) / (2 |dx^T K^T dy|)``.

Every function returns a :class:`CoarseLSResult`. ``losses[:, k]`` is the
performance metric after ``k`` steps (column 0 = initial), ``steps`` /
``accepted`` record what happened at each iteration, ``fallback_would_fail``
counts, per iteration, the rejected instances whose *default* step would also
have failed the check (a diagnostic: the fallback is still taken), and
``n_extra_matvec`` counts the matvecs spent on rejected trials that could not
be reused.

All functions are batched over instances (leading axis ``N``; instances are
columns for PDHG so that one sparse operator serves the whole batch). An
optional ``counter`` dict is incremented by one per batched matvec under the
key ``'matvec'`` so tests can pin the oracle cost.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

ARMIJO_C_DEFAULT = 1e-4
GROWTH_DEFAULT = 2.0


@dataclass
class CoarseLSResult:
    losses: np.ndarray            # (N, K_max + 1)
    steps: np.ndarray             # (N, K_max) step actually taken
    accepted: np.ndarray          # (N, K_max) bool, candidate passed the check
    fallback_would_fail: np.ndarray  # (K_max,) int, rejected rows whose default also fails
    n_extra_matvec: int
    t_default: float
    growth: float
    c: float

    @property
    def accept_rate(self) -> np.ndarray:
        """Per-iteration fraction of instances that took the candidate step."""
        return self.accepted.mean(axis=0)


def _bump(counter, n=1):
    if counter is not None:
        counter['matvec'] = counter.get('matvec', 0) + n


def _alloc(N, K_max):
    return (np.empty((N, K_max + 1)), np.empty((N, K_max)),
            np.zeros((N, K_max), dtype=bool), np.zeros(K_max, dtype=np.int64))


# ---------------------------------------------------------------------------
# Quad: f(z) = 0.5 z^T Q z, f* = 0, x* = 0
# ---------------------------------------------------------------------------

def gd_quad(Q, z0, *, t_default, K_max, growth=GROWTH_DEFAULT,
            c=ARMIJO_C_DEFAULT, compound=True, counter=None) -> CoarseLSResult:
    """Coarse-line-search GD on a batch of quadratics.

    Args:
        Q:  (N, d, d) Hessians.
        z0: (N, d) initial points (x* = 0).

    One batched matvec ``Q g`` per iteration gives the exact function value
    ``phi(t) = f - t g^T g + t^2/2 g^T Q g`` at *any* step, so the Armijo check
    and the gradient at the chosen point are free: K + 1 matvecs total.
    """
    Q = np.asarray(Q, dtype=np.float64)
    z = np.array(z0, dtype=np.float64)
    N = z.shape[0]
    losses, steps, accepted, fbf = _alloc(N, K_max)

    g = np.einsum('nij,nj->ni', Q, z)
    _bump(counter)
    f = 0.5 * np.sum(z * g, axis=1)
    losses[:, 0] = f
    t_def_vec = np.full(N, float(t_default))
    t_prev = t_def_vec

    for k in range(K_max):
        Qg = np.einsum('nij,nj->ni', Q, g)
        _bump(counter)
        gg = np.sum(g * g, axis=1)
        gQg = np.sum(g * Qg, axis=1)

        def phi(t):
            return f - t * gg + 0.5 * t * t * gQg

        t_try = growth * (t_prev if compound else t_def_vec)
        ok = phi(t_try) <= f - c * t_try * gg
        ok_def = phi(t_default) <= f - c * t_default * gg
        fbf[k] = int(np.sum(~ok & ~ok_def))
        t = np.where(ok, t_try, t_default)

        z = z - t[:, None] * g
        g = g - t[:, None] * Qg
        f = phi(t)

        losses[:, k + 1] = f
        steps[:, k] = t
        accepted[:, k] = ok
        t_prev = t

    return CoarseLSResult(losses, steps, accepted, fbf, 0,
                          float(t_default), float(growth), float(c))


# ---------------------------------------------------------------------------
# Logistic regression: f(x) = mean_i softplus(a_i^T x) - b_i a_i^T x
# (matches logreg_rebuttal/build_logreg_table.logreg_f / logreg_grad)
# ---------------------------------------------------------------------------

def _logreg_f_from_AX(AX, b):
    return np.mean(np.logaddexp(0.0, AX) - b * AX, axis=1)


def _logreg_grad_from_AX(A, AX, b):
    m = A.shape[1]
    sig = 1.0 / (1.0 + np.exp(-AX))
    return np.einsum('nmi,nm->ni', A, sig - b) / m


def gd_logreg(A, b, f_opt, *, t_default, K_max, growth=GROWTH_DEFAULT,
              c=ARMIJO_C_DEFAULT, compound=True, counter=None) -> CoarseLSResult:
    """Coarse-line-search GD on batched logistic regression, x0 = 0.

    Args:
        A: (N, m, n) design matrices; b: (N, m) labels in {0, 1};
        f_opt: (N,) optimal values. Loss is ``f(x_k) - f_opt``.

    Per iteration: one A^T matvec (gradient) and one A matvec (``A g``), after
    which ``f(x - t g)`` is closed form for any ``t``. Same 2K matvecs as
    fixed-step GD; rejections cost nothing.
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    f_opt = np.asarray(f_opt, dtype=np.float64)
    N, m, n = A.shape
    losses, steps, accepted, fbf = _alloc(N, K_max)

    X = np.zeros((N, n))
    AX = np.zeros((N, m))
    f = _logreg_f_from_AX(AX, b)
    losses[:, 0] = f - f_opt
    t_def_vec = np.full(N, float(t_default))
    t_prev = t_def_vec

    for k in range(K_max):
        G = _logreg_grad_from_AX(A, AX, b)
        AG = np.einsum('nmi,ni->nm', A, G)
        _bump(counter, 2)
        gg = np.sum(G * G, axis=1)

        t_try = growth * (t_prev if compound else t_def_vec)
        f_try = _logreg_f_from_AX(AX - t_try[:, None] * AG, b)
        f_def = _logreg_f_from_AX(AX - t_default * AG, b)
        ok = f_try <= f - c * t_try * gg
        ok_def = f_def <= f - c * t_default * gg
        fbf[k] = int(np.sum(~ok & ~ok_def))
        t = np.where(ok, t_try, t_default)

        X = X - t[:, None] * G
        AX = AX - t[:, None] * AG
        f = np.where(ok, f_try, f_def)

        losses[:, k + 1] = f - f_opt
        steps[:, k] = t
        accepted[:, k] = ok
        t_prev = t

    return CoarseLSResult(losses, steps, accepted, fbf, 0,
                          float(t_default), float(growth), float(c))


def fgm_logreg(A, b, f_opt, beta, *, t_default, K_max, growth=GROWTH_DEFAULT,
               c=ARMIJO_C_DEFAULT, compound=True, counter=None) -> CoarseLSResult:
    """Coarse-line-search Nesterov FGM on batched logistic regression, x0 = 0.

    Update (as ``build_logreg_table.fgm_losses``):
        x_{k+1} = y_k - t_k grad f(y_k)
        y_{k+1} = x_{k+1} + beta_k (x_{k+1} - x_k)
    The Armijo check is at the gradient point ``y_k``; ``beta`` (length >=
    K_max; ``beta[K_max-1]`` unused) is left untouched. Loss is at ``x_k``.
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    f_opt = np.asarray(f_opt, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    N, m, n = A.shape
    losses, steps, accepted, fbf = _alloc(N, K_max)

    Y = np.zeros((N, n))
    AY = np.zeros((N, m))
    X_curr = np.zeros((N, n))
    AX_curr = np.zeros((N, m))
    losses[:, 0] = _logreg_f_from_AX(AX_curr, b) - f_opt
    t_def_vec = np.full(N, float(t_default))
    t_prev = t_def_vec

    for k in range(K_max):
        G = _logreg_grad_from_AX(A, AY, b)
        AG = np.einsum('nmi,ni->nm', A, G)
        _bump(counter, 2)
        f_y = _logreg_f_from_AX(AY, b)
        gg = np.sum(G * G, axis=1)

        t_try = growth * (t_prev if compound else t_def_vec)
        f_try = _logreg_f_from_AX(AY - t_try[:, None] * AG, b)
        f_def = _logreg_f_from_AX(AY - t_default * AG, b)
        ok = f_try <= f_y - c * t_try * gg
        ok_def = f_def <= f_y - c * t_default * gg
        fbf[k] = int(np.sum(~ok & ~ok_def))
        t = np.where(ok, t_try, t_default)

        X_new = Y - t[:, None] * G
        AX_new = AY - t[:, None] * AG
        losses[:, k + 1] = np.where(ok, f_try, f_def) - f_opt

        Y = X_new + beta[k] * (X_new - X_curr)
        AY = AX_new + beta[k] * (AX_new - AX_curr)
        X_curr, AX_curr = X_new, AX_new

        steps[:, k] = t
        accepted[:, k] = ok
        t_prev = t

    return CoarseLSResult(losses, steps, accepted, fbf, 0,
                          float(t_default), float(growth), float(c))


# ---------------------------------------------------------------------------
# Lasso ISTA: F(x) = 0.5 ||A x - b||^2 + lambd ||x||_1, shared A, x0 = 0
# ---------------------------------------------------------------------------

def _soft(v, thr):
    return np.sign(v) * np.maximum(np.abs(v) - thr, 0.0)


def ista_lasso(A, B, f_opt, lambd, *, t_default, K_max, growth=GROWTH_DEFAULT,
               c=ARMIJO_C_DEFAULT, compound=True, counter=None) -> CoarseLSResult:
    """Coarse-line-search ISTA on a batch of Lasso instances sharing ``A``.

    Args:
        A: (m, n); B: (N, m) right-hand sides; f_opt: (N,); lambd: float.
        ``c`` is accepted for interface symmetry but unused: the composite
        (Beck-Teboulle) condition has no Armijo constant.

    Per iteration: one A^T product (gradient) and one A product at the trial
    point, which becomes the next residual when accepted. Rejected rows pay
    one extra A product for the default point (counted in ``n_extra_matvec``).
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    f_opt = np.asarray(f_opt, dtype=np.float64)
    N = B.shape[0]
    n = A.shape[1]
    losses, steps, accepted, fbf = _alloc(N, K_max)

    X = np.zeros((N, n))
    R = -B.copy()                     # A X - B at X = 0
    f1 = 0.5 * np.sum(R * R, axis=1)
    losses[:, 0] = f1 + lambd * np.sum(np.abs(X), axis=1) - f_opt
    t_def_vec = np.full(N, float(t_default))
    t_prev = t_def_vec
    n_extra = 0

    def composite_ok(f1_new, D, G, f1_old, t):
        return f1_new <= f1_old + np.sum(G * D, axis=1) + np.sum(D * D, axis=1) / (2.0 * t)

    for k in range(K_max):
        G = R @ A
        _bump(counter)
        t_try = growth * (t_prev if compound else t_def_vec)
        Xp = _soft(X - t_try[:, None] * G, t_try[:, None] * lambd)
        Rp = Xp @ A.T - B
        _bump(counter)
        f1p = 0.5 * np.sum(Rp * Rp, axis=1)
        ok = composite_ok(f1p, Xp - X, G, f1, t_try)

        rej = ~ok
        if rej.any():
            Xd = _soft(X[rej] - t_default * G[rej], t_default * lambd)
            Rd = Xd @ A.T - B[rej]
            _bump(counter)
            n_extra += int(rej.sum())
            f1d = 0.5 * np.sum(Rd * Rd, axis=1)
            ok_def = composite_ok(f1d, Xd - X[rej], G[rej], f1[rej], t_default)
            fbf[k] = int(np.sum(~ok_def))
            Xp[rej] = Xd
            Rp[rej] = Rd
            f1p[rej] = f1d

        X, R, f1 = Xp, Rp, f1p
        t = np.where(ok, t_try, t_default)
        losses[:, k + 1] = f1 + lambd * np.sum(np.abs(X), axis=1) - f_opt
        steps[:, k] = t
        accepted[:, k] = ok
        t_prev = t

    return CoarseLSResult(losses, steps, accepted, fbf, n_extra,
                          float(t_default), float(growth), float(c))


# ---------------------------------------------------------------------------
# PDLP / Chambolle-Pock on  min c^T x  s.t.  l <= x <= u,  K x (>=,=) q
# ---------------------------------------------------------------------------

def pdhg_lp(K, c, Qmat, lb, ub, m1, X0, Y0, Xstar, Ystar, *, eta_default, K_max,
            growth=GROWTH_DEFAULT, theta=1.0, compound=True,
            counter=None) -> CoarseLSResult:
    """Coarse adaptive-step PDHG on a batch of LPs sharing the operator ``K``.

    Instances are COLUMNS: ``X0, Xstar`` are (n, N); ``Y0, Ystar, Qmat`` are
    (m, N). The first ``m1`` dual rows are inequality multipliers (projected
    to >= 0), the rest are free. Update and metric follow
    ``learning.tv_averages.run_pdhg_capture_gaps`` exactly:

        x+ = clip(x - eta (c - K^T y), l, u)
        y+ = partial_relu(y + eta (q - K (x+ + theta (x+ - x))))
        gap_k = L(x_k, y*) - L(x*, y_k),  L(x, y) = c^T x - y^T K x + q^T y

    with ``tau = sigma = eta``. The step check is the PDLP rule
    ``eta_try <= (||dx||^2 + ||dy||^2) / (2 |dx^T K^T dy|)``; the default
    ``eta_default <= 0.5 / ||K||`` always passes it. Two sparse matvecs per
    iteration, plus one ``K`` matvec on each rejected column.
    """
    K = sp.csr_matrix(K)
    K_T = K.T.tocsr()
    c = np.asarray(c, dtype=np.float64)
    lb = np.asarray(lb, dtype=np.float64)[:, None]
    ub = np.asarray(ub, dtype=np.float64)[:, None]
    X = np.array(X0, dtype=np.float64)
    Y = np.array(Y0, dtype=np.float64)
    Qmat = np.asarray(Qmat, dtype=np.float64)
    Xstar = np.asarray(Xstar, dtype=np.float64)
    Ystar = np.asarray(Ystar, dtype=np.float64)
    N = X.shape[1]
    losses, steps, accepted, fbf = _alloc(N, K_max)

    KX = K @ X
    KXs = K @ Xstar                   # evaluation only, not an algorithm cost
    _bump(counter)
    cXs = c @ Xstar
    qYs = np.sum(Qmat * Ystar, axis=0)

    def gap(X, Y, KX):
        prim = c @ X - np.sum(Ystar * KX, axis=0) + qYs
        dual = cXs - np.sum(Y * KXs, axis=0) + np.sum(Qmat * Y, axis=0)
        return prim - dual

    def partial_relu(V):
        V[:m1] = np.maximum(V[:m1], 0.0)
        return V

    def trial(eta, X, Y, KX, KTY):
        Xp = np.clip(X - eta * (c[:, None] - KTY), lb, ub)
        KXp = K @ Xp
        KXbar = KXp + theta * (KXp - KX)
        Yp = partial_relu(Y + eta * (Qmat - KXbar))
        dX, dY = Xp - X, Yp - Y
        num = np.sum(dX * dX, axis=0) + np.sum(dY * dY, axis=0)
        den = 2.0 * np.abs(np.sum((KXp - KX) * dY, axis=0))
        with np.errstate(divide='ignore', invalid='ignore'):
            eta_hat = np.where(den > 0.0, num / den, np.inf)
        return Xp, KXp, Yp, eta_hat

    losses[:, 0] = gap(X, Y, KX)
    t_def_vec = np.full(N, float(eta_default))
    t_prev = t_def_vec
    n_extra = 0

    for k in range(K_max):
        KTY = K_T @ Y
        _bump(counter)
        eta_try = growth * (t_prev if compound else t_def_vec)
        Xp, KXp, Yp, eta_hat = trial(eta_try[None, :], X, Y, KX, KTY)
        _bump(counter)
        ok = eta_try <= eta_hat

        rej = ~ok
        if rej.any():
            Xd, KXd, Yd, eta_hat_d = trial(eta_default, X[:, rej], Y[:, rej],
                                           KX[:, rej], KTY[:, rej])
            _bump(counter)
            n_extra += int(rej.sum())
            fbf[k] = int(np.sum(eta_default > eta_hat_d))
            Xp[:, rej] = Xd
            KXp[:, rej] = KXd
            Yp[:, rej] = Yd

        X, Y, KX = Xp, Yp, KXp
        t = np.where(ok, eta_try, eta_default)
        losses[:, k + 1] = gap(X, Y, KX)
        steps[:, k] = t
        accepted[:, k] = ok
        t_prev = t

    return CoarseLSResult(losses, steps, accepted, fbf, n_extra,
                          float(eta_default), float(growth), float('nan'))
