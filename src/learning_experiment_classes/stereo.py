"""Stereo-disparity LP (learning/stereo_lp.py) wrapped for the unified Chambolle–Pock loop.

Instances are gray Olivetti faces (split by subject 28/4/8 as in PDLP) with a
seeded synthetic disparity. Every gray instance shares the sparsity pattern of

    G = [[-diag(w g), I, 0], [diag(w g), I, 0], [-D, 0, I], [D, 0, I]],

so the pool stores only per-instance values: the data-row weights ``wg``
(= w * horizontal gradient of the left view), the RHS ``h``, the box upper
bound ``u``, and the reference optimum (x*, y*). G is rebuilt from ``wg``
inside the (vmapped) trajectory function.

PDHG starts at x0 = (l + u) / 2 and y0 = dual_init * ones (all rows are
inequality rows; there is no equality block). M_val / R_val are pooled at
sample creation (safety factors times the training maxima) and stored in the
metadata, as in PDLP.

Color OOD evaluation is done by tools/stereo_l2o_check.py (numpy PDHG), since
color instances have three data blocks and a different shape.
"""
import diffcp_patch  # noqa: F401  # PSD inverse-permutation fix for the DRO SDP layer
import logging
import os
from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.experimental import sparse as jsparse
from joblib import Parallel, delayed
from sklearn.datasets import fetch_olivetti_faces

from learning.problem_module import GroundTruth, ParameterNames, ProblemData, ProblemModule, Stepsizes
from learning.stereo_lp import grad_x, make_pair, stereo_lp, stereo_x0, stereo_y0, true_disparity
from learning.trajectories import problem_data_to_cp_lp_trajectories
from learning.tv_inpainting_reduced import reduced_opnorm, solve_reduced
from learning.unified_trainer import UnifiedTrainer
from learning_experiment_classes.pdlp import PDLPProblemModule, pep_data_fn_cp, split_persons_by_subject

log = logging.getLogger(__name__)

SPLITS = ("training", "validation", "test")


def stereo_settings(cfg) -> Dict[str, Any]:
    return dict(lam=float(cfg.stereo_lambda), place=str(cfg.stereo_place),
                dmax=float(cfg.stereo_dmax), dual_init=float(cfg.dual_init),
                slack_init=None if cfg.get("stereo_slack_init", None) is None else float(cfg.stereo_slack_init),
                feasible=bool(cfg.get("stereo_feasible_init", False)),
                dual_warm=bool(cfg.get("stereo_dual_warm", False)))


def disparity_seed(image_index: int) -> int:
    """Seed of the synthetic disparity of face ``image_index`` (matches tools/stereo_l2o_check.py)."""
    return 1000 + int(image_index)


def build_instance(left: np.ndarray, seed: int, s: Dict[str, Any]):
    """-> (mats, d_true, left channels) for an (M, N, C) image in [0, 1]."""
    d = true_disparity(left.shape[0], left.shape[1], np.random.default_rng(seed), s["dmax"])
    L, R = make_pair(left, d)
    return stereo_lp(L, R, s["lam"], s["place"], s["dmax"]), d, L


def pattern_template(M: int, N: int, s: Dict[str, Any]):
    """Gray stereo LP whose d-block has every diagonal entry structurally present (ramp image)."""
    ramp = np.tile(np.linspace(0.1, 0.9, N), (M, 1))
    return stereo_lp([ramp], [ramp], s["lam"], s["place"], s["dmax"])


def _solve_one(face, seed, s, cache_path):
    """(raw_x, raw_y, f_opt, ||G||, R_i) of one gray instance, cached as npz."""
    mats, _, _ = build_instance(face[..., None], seed, s)
    if cache_path is not None and os.path.isfile(cache_path):
        z = np.load(cache_path)
        rx, ry, f, opn = z["raw_x"], z["raw_y"], float(z["f_opt"]), float(z["opnorm"])
    else:
        sol = solve_reduced(mats)
        rx, ry, f, opn = sol["raw_x"], sol["raw_y"], float(sol["objective_value"]), reduced_opnorm(mats)
        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez_compressed(cache_path, raw_x=rx, raw_y=ry, f_opt=f, opnorm=opn)
    x0 = stereo_x0(mats.l, mats.u, face.size, s["slack_init"],
                   feasible_h=mats.h if s["feasible"] else None)
    y0 = (stereo_y0(mats.h, face.size, 1, 1.0 if s["place"] == "inside" else s["lam"]) if s["dual_warm"]
          else s["dual_init"] * np.ones(mats.G.shape[0]))
    R = float(np.linalg.norm(np.r_[x0 - rx, y0 - ry]))
    return rx, ry, f, opn, R


def stereo_sample_creation_run(cfg):
    """Write {training,validation,test,ood}_set.npz and out_of_sample_metadata.npz (cwd)."""
    s = stereo_settings(cfg)
    faces = fetch_olivetti_faces().images.astype(np.float64)
    M, N = faces.shape[1:]
    per = int(cfg.images_per_subject)
    persons = split_persons_by_subject(int(cfg.person_split_seed), int(cfg.n_subjects_train),
                                       int(cfg.n_subjects_val), int(cfg.n_subjects_test),
                                       n_total=int(cfg.n_subjects_total))
    cache_root = cfg.get("stereo_cache_dir", None)
    tag = {"training": "train", "validation": "val", "test": "test"}
    tmpl_w = s["lam"] if s["place"] == "inside" else 1.0
    stats = {}
    for split, pers in zip(SPLITS, persons):
        idx = np.concatenate([np.arange(p * per, (p + 1) * per) for p in pers])
        log.info(f"[{split}] {len(idx)} instances; solving reference LPs")
        sols = Parallel(n_jobs=int(cfg.get("n_jobs", 4)))(
            delayed(_solve_one)(faces[i], disparity_seed(i), s,
                                None if cache_root is None else
                                os.path.join(cache_root, f"stereo_{s['place']}{s['lam']:g}_{tag[split]}", f"{j}.npz"))
            for j, i in enumerate(idx))
        wg, h, u = [], [], []
        for i in idx:
            mats, _, L = build_instance(faces[i][..., None], disparity_seed(i), s)
            wg.append(tmpl_w * grad_x(L[0]).ravel()); h.append(mats.h); u.append(mats.u)
        arrays = dict(
            image_index_batch=idx.astype(np.int32), disp_seed_batch=np.array([disparity_seed(i) for i in idx]),
            wg_batch=np.stack(wg), h_batch=np.stack(h), u_batch=np.stack(u),
            x_opt_batch=np.stack([x[0] for x in sols]), y_opt_batch=np.stack([x[1] for x in sols]),
            f_opt_batch=np.array([x[2] for x in sols]), M_val_batch=np.array([x[3] for x in sols]),
            R_val_batch=np.array([x[4] for x in sols]))
        np.savez_compressed(f"{split}_set.npz", **arrays)
        stats[split] = arrays
        log.info(f"[{split}] ||G|| median {np.median(arrays['M_val_batch']):.3f} max {arrays['M_val_batch'].max():.3f}; "
                 f"R median {np.median(arrays['R_val_batch']):.2f} max {arrays['R_val_batch'].max():.2f}")
    # the trainer's 'ood' split is the gray test split; color OOD is evaluated separately
    np.savez_compressed("ood_set.npz", **stats["test"])
    M_val = float(cfg.m_safety_factor) * float(stats["training"]["M_val_batch"].max())
    R_val = float(cfg.r_safety_factor) * float(stats["training"]["R_val_batch"].max())
    np.savez_compressed(
        "out_of_sample_metadata.npz", M=M, N=N, M_val=M_val, R_val=R_val,
        m_safety_factor=float(cfg.m_safety_factor), r_safety_factor=float(cfg.r_safety_factor),
        stereo_lambda=s["lam"], stereo_place=np.str_(s["place"]), stereo_dmax=s["dmax"],
        dual_init=s["dual_init"], n_train=len(stats["training"]["h_batch"]),
        stereo_slack_init=np.nan if s["slack_init"] is None else s["slack_init"],
        stereo_feasible_init=s["feasible"], stereo_dual_warm=s["dual_warm"])
    log.info(f"[POOL] M_val = {M_val:.4f}, R_val = {R_val:.4f}")
    log.info("=== Stereo sample creation complete ===")


class StereoProblemModule(ProblemModule):
    """Stereo-disparity LP for the unified CP learning loop (see module docstring)."""

    def __init__(self, cfg: Any):
        super().__init__(cfg)
        d = cfg.get("data_source_dir", None)
        if d is None:
            raise ValueError("Stereo requires cfg.data_source_dir (run_sample_creation.py Stereo local)")
        self.data_source_dir = d
        meta = np.load(os.path.join(d, "out_of_sample_metadata.npz"), allow_pickle=False)
        self.s = stereo_settings(cfg)
        meta_s = dict(lam=float(meta["stereo_lambda"]), place=str(meta["stereo_place"]),
                      dmax=float(meta["stereo_dmax"]), dual_init=float(meta["dual_init"]),
                      slack_init=None if "stereo_slack_init" not in meta.files or np.isnan(meta["stereo_slack_init"])
                      else float(meta["stereo_slack_init"]),
                      feasible=bool(meta["stereo_feasible_init"]) if "stereo_feasible_init" in meta.files else False,
                      dual_warm=bool(meta["stereo_dual_warm"]) if "stereo_dual_warm" in meta.files else False)
        if meta_s != self.s:
            raise ValueError(f"cfg stereo settings {self.s} do not match sample creation {meta_s}")
        self.M_img, self.N_img = int(meta["M"]), int(meta["N"])
        self.K_pix = self.M_img * self.N_img
        self.M_val, self.R_val = float(meta["M_val"]), float(meta["R_val"])
        log.info(f"Stereo: {self.M_img}x{self.N_img}, lambda={self.s['lam']} ({self.s['place']}), "
                 f"M_val={self.M_val:.4f} R_val={self.R_val:.4f}")

        tmpl = pattern_template(self.M_img, self.N_img, self.s)
        coo = tmpl.G.tocoo()
        self.G_shape = tmpl.G.shape
        self.m1 = self.G_shape[0]
        self._idx = jnp.stack([jnp.asarray(coo.row), jnp.asarray(coo.col)], 1)
        self._base = jnp.asarray(coo.data)
        is_data = (coo.col < self.K_pix) & (coo.row < 2 * self.K_pix)
        self._is_data = jnp.asarray(is_data)
        self._sign = jnp.asarray(np.where(coo.row < self.K_pix, -1.0, 1.0))
        self._pix = jnp.asarray(np.where(is_data, coo.col, 0))
        self.c = jnp.asarray(tmpl.c)
        self.l = jnp.asarray(tmpl.l)

    # ------------------------------------------------------------------ data
    def _G(self, wg):
        data = jnp.where(self._is_data, self._sign * wg[self._pix], self._base)
        return jsparse.BCOO((data, self._idx), shape=self.G_shape)

    def _load_split(self, split: str, N: int) -> Tuple[ProblemData, GroundTruth]:
        z = np.load(os.path.join(self.data_source_dir, f"{split}_set.npz"))
        total = int(z["h_batch"].shape[0])
        idx = np.arange(min(N, total))
        log.info(f"{split}: loaded {len(idx)} / {total}")
        return ({k: jnp.asarray(z[f"{k}_batch"][idx]) for k in ("wg", "h", "u")},
                {k: jnp.asarray(z[f"{k}_batch"][idx]) for k in ("x_opt", "y_opt")})

    def _suffix(self, pd_gt):
        p, g = pd_gt
        return ({f"{k}_batch": v for k, v in p.items()}, {f"{k}_batch": v for k, v in g.items()})

    def sample_training_batch(self, key, N):
        return self._suffix(self._load_split("training", N))

    def sample_validation_batch(self, key, N):
        return self._suffix(self._load_split("validation", N))

    def sample_test_batch(self, key, N):
        return self._suffix(self._load_split("test", N))

    def generate_out_of_sample_data(self, key):
        return {"validation": self.sample_validation_batch(key, int(self.cfg.get("out_of_sample_val_N", 40))),
                "test": self.sample_test_batch(key, int(self.cfg.get("out_of_sample_test_N", 80))),
                "ood": self._suffix(self._load_split("ood", int(self.cfg.get("out_of_dist_N", 80))))}

    # ------------------------------------------------------------ trajectory
    def get_trajectory_fn(self, alg: str) -> Callable:
        if alg != "cp":
            raise ValueError(f"Stereo supports only alg='cp'; got {alg!r}")
        m, dual_init, slack_init, K_pix = self.m1, self.s["dual_init"], self.s["slack_init"], self.K_pix
        feasible, dual_warm = self.s["feasible"], self.s["dual_warm"]
        a_data = 1.0 if self.s["place"] == "inside" else self.s["lam"]

        def traj_fn(stepsizes, wg, h, u, x_opt, y_opt, K_max, return_Gram_representation=True):
            x0 = 0.5 * (self.l + u)
            if feasible:   # gray: data slacks = |w r| + 0.01 (h[:K] = -w r), TV slacks = 0.01
                x0 = x0.at[K_pix:].set(0.01).at[K_pix:2 * K_pix].set(jnp.abs(h[:K_pix]) + 0.01)
            elif slack_init is not None:
                x0 = x0.at[K_pix:].set(slack_init)
            y0 = dual_init * jnp.ones(m)
            if dual_warm:   # stereo_y0 for one gray channel (see learning/stereo_lp.py)
                top = h[:K_pix]
                w_top = jnp.where(top > 0, 0.99, jnp.where(top < 0, 0.01, 0.5)) * a_data
                y0 = jnp.concatenate([w_top, a_data - w_top, 0.5 * jnp.ones(m - 2 * K_pix)])
            return problem_data_to_cp_lp_trajectories(
                stepsizes, self.c, self._G(wg), h, self.l, u, x_opt, y_opt, x0, y0, K_max, m,
                return_Gram_representation=return_Gram_representation)

        return traj_fn

    def compute_batched_trajectories(self, stepsizes, batched_data, fixed_data, traj_fn, K_max):
        f = jax.vmap(lambda wg, h, u, xo, yo: traj_fn(stepsizes, wg, h, u, xo, yo, K_max,
                                                      return_Gram_representation=True))
        return f(*(batched_data[k] for k in ("wg", "h", "u", "x_opt", "y_opt")))

    def create_metric_fn(self, trajectories, problem_data, ground_truth, pep_obj):
        if pep_obj != "obj_val":
            raise NotImplementedError("Stereo supports only pep_obj='obj_val' (duality gap)")
        G, h, c = self._G(problem_data["wg"]), problem_data["h"], self.c
        xo, yo = ground_truth["x_opt"], ground_truth["y_opt"]
        v_iter, y_iter = trajectories[0], trajectories[1]
        L = lambda vv, yy: c @ vv - yy @ (G @ vv) + h @ yy
        return lambda k: L(v_iter[k], yo) - L(xo, y_iter[k])

    # ------------------------------------------------------------------- PEP
    def get_pep_data_fn(self, alg: str) -> Callable:
        return pep_data_fn_cp

    def compute_L_mu_R(self, samples=None):
        return (self.M_val, 0.0, self.R_val)

    def get_initial_stepsizes(self, alg: str, K: int, L: float, mu: float) -> Stepsizes:
        return PDLPProblemModule.get_initial_stepsizes(self, alg, K, L, mu)

    def build_stepsizes_dataframe(self, *args, **kwargs) -> pd.DataFrame:
        return PDLPProblemModule.build_stepsizes_dataframe(self, *args, **kwargs)

    def get_batched_parameters(self) -> ParameterNames:
        return ("wg", "h", "u", "x_opt", "y_opt")

    def get_fixed_parameters(self) -> ParameterNames:
        return ()

    def get_ground_truth_keys(self) -> ParameterNames:
        return ("x_opt", "y_opt")

    def get_gram_dimensions(self, alg: str, K: int) -> Tuple[int, int]:
        return (4 * K + 11, 2 * (K + 2))

    def get_supported_algorithms(self):
        return ["cp"]

    def validate_config(self) -> None:
        if self.cfg.get("alg", "cp") != "cp":
            raise ValueError("Stereo supports only alg='cp'")


def stereo_run(cfg):
    log.info("Starting Stereo-disparity learning experiment")
    log.info(cfg)
    key = jax.random.PRNGKey(int(cfg.sgd_seed))
    pm = StereoProblemModule(cfg)
    pm.validate_config()
    os.makedirs(cfg.output_dir, exist_ok=True)
    key, train_key = jax.random.split(key)
    trainer = UnifiedTrainer(pm, cfg, train_key)
    trainer.prepare_data(save_dir=cfg.output_dir)
    for K in cfg.K_max:
        K_dir = os.path.join(cfg.output_dir, f"K_{K}")
        os.makedirs(K_dir, exist_ok=True)
        trainer.train(K, os.path.join(K_dir, "progress.csv"), K_dir)
        log.info(f"K={K} complete")
    log.info("=== Stereo experiment complete ===")
