"""Evaluate the DR-L2O training objective at fixed PDLP schedules.

For each schedule and each Wasserstein radius eps, reports the DRO SDP value
(the ldro-pep training loss) and the empirical L2O training loss on the same
training minibatches. If the DRO objective ranks a schedule the same way the
empirical loss does, DR-L2O training will drift toward the L2O schedule.

Schedules are given as ``label=path/to/progress.csv[@row]``, where row is an
SGD iteration or ``best`` (validation-best, the default). ``--damp label=a``
adds a schedule blending ``label``'s (tau, sigma, theta) a fraction ``a`` of
the way back toward its untrained row-0 schedule.

Usage (from src/):
    python tools/pdlp_dro_objective_probe.py --data-dir <sample dir> --K 5 \
        --sched l2o=<.../K_5/progress.csv> --damp l2o=0.5 \
        --eps 1 10 100 1000 --n-minibatches 3 --out probe.csv
"""
import argparse
import os
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))
os.chdir(SRC_DIR)

import _jax_setup  # noqa: E402,F401
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from learning.unified_trainer import UnifiedTrainer, _reparam_modes, to_raw_params  # noqa: E402
from learning_experiment_classes.pdlp import PDLPProblemModule  # noqa: E402


def _row(df: pd.DataFrame, K: int, row) -> tuple:
    return tuple(jnp.array(df.loc[row, [f"{p}_{k}" for k in range(K)]].to_numpy(float))
                 for p in ("tau", "sigma", "theta"))


def load_schedules(specs, damps, K) -> dict:
    out, untrained = {}, {}
    for spec in specs:
        label, rest = spec.split("=", 1)
        path, _, row = rest.partition("@")
        df = pd.read_csv(path)
        row = int(df.validation_loss.idxmin()) if row in ("", "best") else int(row)
        out[label] = _row(df, K, row)
        untrained[label] = _row(df, K, 0)
    out["untrained"] = next(iter(untrained.values()))
    for spec in damps:
        label, a = spec.split("=")
        a = float(a)
        out[f"{label}~damp{a:g}"] = tuple((1 - a) * s + a * s0 for s, s0 in zip(out[label], untrained[label]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--sched", nargs="+", required=True)
    ap.add_argument("--damp", nargs="*", default=[])
    ap.add_argument("--eps", nargs="+", type=float, default=[1.0, 10.0, 100.0, 1000.0])
    ap.add_argument("--n-minibatches", type=int, default=3)
    ap.add_argument("--N", type=int, default=5, help="minibatch size (DR-L2O uses 5)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    meta = np.load(Path(args.data_dir) / "out_of_sample_metadata.npz")
    cfg = OmegaConf.load(SRC_DIR / "configs_learning/pdlp.yaml")
    cfg.data_source_dir = str(args.data_dir)
    cfg.learning_framework = "ldro-pep"
    cfg.N = args.N
    cfg.init_type = str(meta["init_type"]) if "init_type" in meta.files else "center"
    if "dual_eq_init" in meta.files and float(meta["dual_eq_init"]) != 0.0:
        cfg.dual_init = float(meta["dual_ineq_init"])
    else:
        cfg.warm_dual = bool(meta["dual_ineq_init"] == 0.5) if "dual_ineq_init" in meta.files else False
    if "warm_delta" in meta.files:
        cfg.warm_delta = float(meta["warm_delta"])

    schedules = load_schedules(args.sched, args.damp, args.K)
    rows = []
    for eps in args.eps:
        cfg.eps = eps
        pm = PDLPProblemModule(cfg)
        tr = UnifiedTrainer(pm, cfg, jax.random.PRNGKey(0))
        tr.prepare_data()
        L, mu, R = pm.compute_L_mu_R()
        tr._modes = _reparam_modes(3, False)
        dro = tr._build_loss_function(args.K, L, mu, R, pm.get_initial_stepsizes("cp", args.K, L, mu))
        emp = tr._build_l2o_loss(args.K)
        for mb in range(args.n_minibatches):
            batch = tr._get_minibatch(mb * 7)
            for label, s in schedules.items():
                raw = to_raw_params(s, tr._modes)
                t0 = time.time()
                d = float(dro(raw, batch))
                rows.append(dict(eps=eps, minibatch=mb, schedule=label, dro=d,
                                 empirical=float(emp(raw, batch)), secs=time.time() - t0))
                print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4g}".format):
        print(df.groupby(["eps", "schedule"])[["dro", "empirical"]].mean().unstack("eps").to_string())
    print(f"wrote {args.out}  (solver failures are logged by [SparseFwd] above)")


if __name__ == "__main__":
    main()
