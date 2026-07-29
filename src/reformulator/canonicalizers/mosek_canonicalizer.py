"""Direct MOSEK backend for the DRO reformulation.

Reuses ClarabelCanonicalizer's conic assembly (min q'x s.t. b - Ax in K)
and hands the same data to the MOSEK Optimizer API as affine conic
constraints: F x + g in D with F = -A, g = b. No CVXPY in the loop.

Cone mapping (Clarabel -> MOSEK domain):
  NonnegativeConeT(n)   -> rplus(n)
  ZeroConeT(n)          -> rzero(n)
  SecondOrderConeT(n)   -> quadratic cone(n)      (same (t, x) layout)
  PSDTriangleConeT(n)   -> svec psd(n(n+1)/2)

Both svec conventions scale off-diagonals by sqrt(2), but Clarabel stacks
the upper triangle column-wise while MOSEK stacks the lower triangle
column-wise, so PSD blocks get a within-block row permutation.
"""
import logging
import os
import sys

import numpy as np
import scipy.sparse as spa
import clarabel

from .clarabel_canonicalizer import ClarabelCanonicalizer

log = logging.getLogger(__name__)


def svec_clarabel_to_mosek_perm(n):
    # position k in MOSEK order (lower triangle, column-major) reads
    # Clarabel position j(j+1)/2 + i for the entry X_{ij}, i <= j
    perm = np.empty(n * (n + 1) // 2, dtype=np.int64)
    k = 0
    for c in range(n):
        for r in range(c, n):
            perm[k] = r * (r + 1) // 2 + c
            k += 1
    return perm


class MosekCanonicalizer(ClarabelCanonicalizer):

    def set_mosek_opts(self, threads=None, params=None, verbose=False):
        self.mosek_threads = threads
        self.mosek_params = params or {}
        self.mosek_verbose = verbose

    def _row_order_and_domains(self, cones):
        """Global row index array (new -> old) and per-cone domain specs."""
        idx_blocks = []
        domains = []
        offset = 0
        for cone in cones:
            name = type(cone).__name__
            if 'NonnegativeCone' in name:
                n = self._cone_dim(cone)
                idx_blocks.append(offset + np.arange(n))
                domains.append(('rplus', n))
            elif 'ZeroCone' in name:
                n = self._cone_dim(cone)
                idx_blocks.append(offset + np.arange(n))
                domains.append(('rzero', n))
            elif 'SecondOrderCone' in name:
                n = self._cone_dim(cone)
                idx_blocks.append(offset + np.arange(n))
                domains.append(('quad', n))
            elif 'PSDTriangleCone' in name:
                d = self._cone_dim(cone)          # matrix side length
                svec_dim = d * (d + 1) // 2
                idx_blocks.append(offset + svec_clarabel_to_mosek_perm(d))
                domains.append(('svecpsd', svec_dim))
                n = svec_dim
            else:
                raise NotImplementedError(f'cone {name} not supported')
            offset += n
        return np.concatenate(idx_blocks), domains

    @staticmethod
    def _cone_dim(cone):
        # clarabel cone objects repr as e.g. PSDTriangleConeT(26)
        r = repr(cone)
        return int(r[r.index('(') + 1: r.index(')')])

    def solve(self):
        import mosek

        q = self.q
        if self.measure == 'expectation':
            A, b, cones = self.A, self.b, self.cones
        elif self.measure == 'cvar':
            A, b, cones = self.A_full, self.b_full, self.cones_full

        row_order, domains = self._row_order_and_domains(cones)
        assert row_order.shape[0] == A.shape[0]
        F = (-A).tocsr()[row_order, :].tocoo()
        g = np.asarray(b)[row_order]

        x_dim = q.shape[0]
        n_rows = g.shape[0]

        with mosek.Env() as env:
            with env.Task() as task:
                if getattr(self, 'mosek_verbose', False):
                    task.set_Stream(mosek.streamtype.log,
                                    lambda s: sys.stdout.write(s))

                threads = getattr(self, 'mosek_threads', None)
                if threads is None:
                    threads = int(os.environ.get('SLURM_CPUS_PER_TASK', 0))
                if threads:
                    task.putintparam(mosek.iparam.num_threads, int(threads))
                for key, val in getattr(self, 'mosek_params', {}).items():
                    task.putparam(key, str(val))

                task.appendvars(x_dim)
                task.putvarboundsliceconst(0, x_dim, mosek.boundkey.fr,
                                           -np.inf, np.inf)
                nz = np.flatnonzero(q)
                task.putclist(nz, q[nz])
                task.putobjsense(mosek.objsense.minimize)

                task.appendafes(n_rows)
                task.putafefentrylist(F.row, F.col, F.data)
                task.putafegslice(0, n_rows, g)

                afe_offset = 0
                for kind, n in domains:
                    if kind == 'rplus':
                        dom = task.appendrplusdomain(n)
                    elif kind == 'rzero':
                        dom = task.appendrzerodomain(n)
                    elif kind == 'quad':
                        dom = task.appendquadraticconedomain(n)
                    elif kind == 'svecpsd':
                        dom = task.appendsvecpsdconedomain(n)
                    task.appendacc(dom,
                                   list(range(afe_offset, afe_offset + n)),
                                   None)
                    afe_offset += n

                task.optimize()
                solsta = task.getsolsta(mosek.soltype.itr)
                if solsta != mosek.solsta.optimal:
                    log.warning(f'MOSEK solsta: {solsta}')
                xx = np.array(task.getxx(mosek.soltype.itr))
                obj = task.getprimalobj(mosek.soltype.itr)
                solve_time = task.getdouinf(mosek.dinfitem.optimizer_time)

        self.x_sol = xx
        return {
            'obj': obj,
            'solvetime': solve_time,
        }
