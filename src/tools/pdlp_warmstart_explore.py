"""Evaluate learned PDHG schedules for TV inpainting (warm-start exploration).

For every ``label=path/to/progress.csv`` it evaluates the validation-best
schedule of that run, plus the run's untrained initial schedule (row 0), on

  * ``test``         -- the held-out grayscale test split (test subjects only)
                        of ``--data-dir``, at its in-distribution corruption;
  * ``<source>@<frac>`` -- ``--n-color`` RGB images built with
                        ``build_color_lp(img, frac, MASK_SEED)``, where source is
                        ``color`` (Tiny ImageNet, 64x64) or ``imagenette``
                        (Imagenette val split, center-cropped and resized to
                        ``--color-size``; images are evaluation-only).

The PDHG init (center / warm, delta, dual init) is read from the metadata of
``--data-dir`` so evaluation matches training. Writes one long-form CSV of
final-iterate Lagrangian gaps and prints a summary table.

Usage (from src/):
    python tools/pdlp_warmstart_explore.py --data-dir <sample dir> \
        --runs l2o_k5=<.../K_5/progress.csv> drl2o_k5=<...> \
        --color-fracs 0.1 0.2 --out results.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from learning.tv_averages import run_pdhg_capture_gaps  # noqa: E402
from learning.tv_inpainting_color_averages import (  # noqa: E402
    MASK_SEED,
    build_color_lp,
    load_color_images,
)
from learning.tv_inpainting_test import extract_constraint_matrices  # noqa: E402
from learning_experiment_classes.pdlp import (  # noqa: E402
    blurred_data_block,
    build_pdhg_init,
    gaussian_blur_matrix,
    lp_primal_residual,
    solve_lp_checked,
)

ETAS = (1e-2, 5e-2, 1e-1)


def load_schedules(runs: list[str]) -> dict:
    """``{label: (K, tau, sigma, theta)}`` for each run's selected row and row 0.

    A spec is ``label=path[@row]``; the row defaults to the validation-best one.
    """
    out = {}
    for spec in runs:
        label, rest = spec.split("=", 1)
        path, _, row = rest.partition("@")
        df = pd.read_csv(path)
        K = sum(c.startswith("tau_") for c in df.columns)
        best = int(df.validation_loss.idxmin()) if row in ("", "best") else int(row)
        for tag, row in ((label, best), (f"{label}:untrained", 0)):
            out[tag] = (K, *(df.loc[row, [f"{p}_{k}" for k in range(K)]].to_numpy(float)
                             for p in ("tau", "sigma", "theta")))
        print(f"{label}: K={K}, val {df.validation_loss[best]:.4g} at SGD iter {best}/{len(df) - 1}")
    return out


def final_gaps(mats, raw_x, raw_y, x0, y0, schedules):
    return {
        label: float(run_pdhg_capture_gaps(
            mats.c, mats.G, mats.h, mats.A, mats.b, mats.l, mats.u,
            raw_x, raw_y, x0, y0, tau, sigma, theta)[-1])
        for label, (_, tau, sigma, theta) in schedules.items()
    }


def init_radius(x0, y0, raw_x, raw_y) -> float:
    """The PEP initial-condition radius ||(x0 - x*, y0 - y*)|| of one instance."""
    return float(np.linalg.norm(np.concatenate([x0 - raw_x, y0 - raw_y])))


def load_imagenette(root: Path, n_images: int, size: int, seed: int = 0) -> np.ndarray:
    """``n_images`` uint8 (size, size, 3) images drawn without replacement from the val split."""
    from PIL import Image
    files = sorted(root.glob("val/*/*.JPEG"))
    pick = np.random.default_rng(seed).choice(len(files), size=n_images, replace=False)
    out = []
    for i in pick:
        im = Image.open(files[i]).convert("RGB")
        w, h = im.size
        m = min(w, h)
        left, top = (w - m) // 2, (h - m) // 2
        im = im.crop((left, top, left + m, top + m)).resize((size, size), Image.BICUBIC)
        out.append(np.asarray(im, dtype=np.uint8))
    return np.stack(out)


def eval_gray_test(data_dir: Path, meta, init, schedules) -> list[dict]:
    from sklearn.datasets import fetch_olivetti_faces
    lp_upper = float(meta["lp_upper"])
    M, N = int(meta["M"]), int(meta["N"])
    images = fetch_olivetti_faces().images.astype(np.float64) * lp_upper
    d = np.load(data_dir / "test_set.npz")
    blur_sigma = float(meta["blur_sigma"]) if "blur_sigma" in meta.files else 0.0
    H = gaussian_blur_matrix(M, N, blur_sigma)
    rows = []
    for i in tqdm(range(len(d["image_index_batch"])), desc="gray test"):
        idx, mask = int(d["image_index_batch"][i]), d["mask_batch"][i]
        pix = images[idx].reshape(-1)
        known = np.flatnonzero(mask)
        mats = extract_constraint_matrices(known, pix[known], M, N)
        if blur_sigma > 0:
            A, b = blurred_data_block(known, [pix], [H], mats.c.size)
            mats = mats._replace(A=A, b=b)
        x0, y0 = build_pdhg_init(init[0], mats.c.size, mats.G.shape[0], known.size, lp_upper,
                                 pix_channels=[pix], mask=mask, M=M, N=N,
                                 delta=init[1], dual_ineq_init=init[2], dual_eq_init=init[3])
        f_opt = float(d["f_opt_batch"][i])
        R = init_radius(x0, y0, d["x_opt_batch"][i], d["y_opt_batch"][i])
        for label, g in final_gaps(mats, d["x_opt_batch"][i], d["y_opt_batch"][i], x0, y0, schedules).items():
            rows.append(dict(split="test", idx=i, label=label, gap=g, f_opt=f_opt, R=R))
    return rows


def add_channel_coupling(mats, K: int, lp_upper: float):
    """Append the inter-channel term sum_i sum_{c<c'} |p_{c,i} - p_{c',i}| to a color LP.

    New variables u_{cc'} (K each, pairs RG, RB, GB) with cost 1 and bounds
    [0, lp_upper], and rows u_{cc'} -/+ (p_c - p_c') >= 0 appended to G. All
    coefficients are +-1. Grayscale (one channel) has no such term, so the
    color operator norm grows structurally: the pixel block of G^T G becomes
    2 (D^T D (x) I_3 + I (x) C^T C), whose top eigenvalue is ~8 + 3.
    """
    import scipy.sparse as sp
    nb = mats.c.size // 3                        # per-channel block [p, v, w]
    n_old = mats.c.size
    pairs = [(0, 1), (0, 2), (1, 2)]
    I_K = sp.eye(K, format="csr")
    rows = []
    for j, (a, b) in enumerate(pairs):
        P = sp.lil_matrix((K, n_old))
        P[:, a * nb:a * nb + K] = I_K
        P[:, b * nb:b * nb + K] = -I_K
        U = sp.csr_matrix((np.ones(K), (np.arange(K), j * K + np.arange(K))), shape=(K, 3 * K))
        P = P.tocsr()
        rows += [sp.hstack([-P, U]), sp.hstack([P, U])]
    G = sp.vstack([sp.hstack([mats.G, sp.csr_matrix((mats.G.shape[0], 3 * K))])] + rows, format="csr")
    A = sp.hstack([mats.A, sp.csr_matrix((mats.A.shape[0], 3 * K))], format="csr")
    return mats._replace(
        c=np.concatenate([mats.c, np.ones(3 * K)]), G=G, h=np.zeros(G.shape[0]), A=A,
        l=np.concatenate([mats.l, np.zeros(3 * K)]), u=np.concatenate([mats.u, lp_upper * np.ones(3 * K)]))


def add_gradient_coupling(mats, M: int, N: int, mu: float, lp_upper: float):
    """Append the cross-channel edge-consistency term mu * sum_{c<c'} ||D p_c - D p_c'||_1.

    Colour edges co-occur across channels; this standard prior penalizes channel
    pairs whose finite differences disagree. New slacks g_{cc'} (one per edge and
    pair, cost 1, bounds [0, 2 mu lp_upper]) with rows g -/+ mu (D p_c - D p_c') >= 0.
    The pixel block of G^T G becomes 2 D^T D (x) (I_3 + mu^2 C^T C), whose top
    eigenvalue is ~8 (1 + 3 mu^2), so ||G|| grows with mu (grayscale: no such term).
    """
    import scipy.sparse as sp
    from learning.tv_inpainting_reduced import diff_matrix
    D = diff_matrix(M, N)
    E, K = D.shape
    nb = mats.c.size // 3
    n_old = mats.c.size
    rows = []
    for j, (a, b) in enumerate([(0, 1), (0, 2), (1, 2)]):
        P = sp.lil_matrix((E, n_old))
        P[:, a * nb:a * nb + K] = D
        P[:, b * nb:b * nb + K] = -D
        P = mu * P.tocsr()
        S = sp.csr_matrix((np.ones(E), (np.arange(E), j * E + np.arange(E))), shape=(E, 3 * E))
        rows += [sp.hstack([-P, S]), sp.hstack([P, S])]
    G = sp.vstack([sp.hstack([mats.G, sp.csr_matrix((mats.G.shape[0], 3 * E))])] + rows, format="csr")
    A = sp.hstack([mats.A, sp.csr_matrix((mats.A.shape[0], 3 * E))], format="csr")
    return mats._replace(
        c=np.concatenate([mats.c, np.ones(3 * E)]), G=G, h=np.zeros(G.shape[0]), A=A,
        l=np.concatenate([mats.l, np.zeros(3 * E)]),
        u=np.concatenate([mats.u, 2 * mu * lp_upper * np.ones(3 * E)]))


def _coupling_tag(coupling) -> str:
    if coupling in (False, None, ""):
        return ""
    return "_cpl" if coupling in (True, "pixel") else f"_{coupling}"


def build_color_instance(image: np.ndarray, frac: float, lp_upper: float, channel_blur,
                         coupling=False):
    """Color LP (channel c blurred with std ``channel_blur[c]`` before masking), mask, RGB channels.

    ``coupling`` adds the inter-channel difference term (``add_channel_coupling``).
    """
    M, N, _ = image.shape
    mats = build_color_lp(image, frac, MASK_SEED)
    mask = (np.random.default_rng(MASK_SEED).random((M, N)) >= frac).reshape(-1)  # as in build_color_lp
    rgb = image.astype(np.float64) * (lp_upper / 255.0)
    chans = [rgb[:, :, c].reshape(-1) for c in range(3)]
    if any(s > 0 for s in channel_blur):
        Hs = [gaussian_blur_matrix(M, N, s) for s in channel_blur]
        A, b = blurred_data_block(np.flatnonzero(mask), chans, Hs, mats.c.size // 3)
        mats = mats._replace(A=A, b=b)
    if coupling in (True, "pixel"):
        mats = add_channel_coupling(mats, M * N, lp_upper)
    elif isinstance(coupling, str) and coupling.startswith("grad"):
        mats = add_gradient_coupling(mats, M, N, float(coupling[4:]), lp_upper)
    return mats, mask, chans


def _solve_color_cached(image, frac, lp_upper, channel_blur, cache_path: Path, coupling=False):
    """Reference solution of one color instance, cached as npz.

    The color LP is block-diagonal over channels, so each channel is solved as
    its own grayscale LP (faster and more accurate than the joint solve) and
    the results are stacked in ``build_color_lp``'s layout: x = [x_R; x_G; x_B],
    y = [y_G,R; y_G,G; y_G,B; y_A,R; y_A,G; y_A,B].
    """
    if cache_path.exists():
        d = np.load(cache_path)
        return d["raw_x"], d["raw_y"], float(d["f_opt"])
    if coupling:  # channels are coupled: one joint solve
        mats, _, _ = build_color_instance(image, frac, lp_upper, channel_blur, coupling=coupling)
        sol = solve_lp_checked(mats)
        raw_x, raw_y, f_opt = np.ravel(sol["raw_x"]), np.ravel(sol["raw_y"]), float(sol["objective_value"])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, raw_x=raw_x, raw_y=raw_y, f_opt=f_opt)
        return raw_x, raw_y, f_opt
    mats, mask, chans = build_color_instance(image, frac, lp_upper, channel_blur)
    M, N, _ = image.shape
    known = np.flatnonzero(mask)
    xs, y_ineq, y_eq, f_opt = [], [], [], 0.0
    for pix, sigma in zip(chans, channel_blur):
        m = extract_constraint_matrices(known, pix[known], M, N)
        if sigma > 0:
            A, b = blurred_data_block(known, [pix], [gaussian_blur_matrix(M, N, sigma)], m.c.size)
            m = m._replace(A=A, b=b)
        sol = solve_lp_checked(m)
        ry = np.ravel(sol["raw_y"])
        xs.append(np.ravel(sol["raw_x"]))
        y_ineq.append(ry[:m.G.shape[0]])
        y_eq.append(ry[m.G.shape[0]:])
        f_opt += float(sol["objective_value"])
    raw_x, raw_y = np.concatenate(xs), np.concatenate(y_ineq + y_eq)
    res = lp_primal_residual(mats, raw_x)
    if res > 1e-5:
        raise RuntimeError(f"stacked color solution infeasible for {cache_path.name}: {res:.2e}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, raw_x=raw_x, raw_y=raw_y, f_opt=f_opt)
    return raw_x, raw_y, f_opt


def eval_color(frac: float, images: np.ndarray, source: str, lp_upper: float, init,
               schedules, channel_blur=(0.0, 0.0, 0.0), cache_dir: Path | None = None,
               n_jobs: int = 4, coupling=False) -> list[dict]:
    """Color LPs; channel c is blurred with std ``channel_blur[c]`` before masking.

    Reference solves run in ``n_jobs`` parallel workers and are cached under
    ``cache_dir`` keyed by (source, size, frac, blur, image index).
    """
    from joblib import Parallel, delayed
    M = images[0].shape[0]
    key = f"{source}{M}_f{frac:g}_b{'-'.join(f'{s:g}' for s in channel_blur)}" + _coupling_tag(coupling)
    cache_dir = cache_dir or Path(__file__).resolve().parent.parent / "iclr_data_outputs/explore_warmstart/lp_cache"
    sols = Parallel(n_jobs=n_jobs)(
        delayed(_solve_color_cached)(images[i], frac, lp_upper, channel_blur, cache_dir / key / f"{i}.npz",
                                     coupling)
        for i in tqdm(range(len(images)), desc=f"{source}@{frac} solves"))
    rows = []
    for i, (raw_x, raw_y, f_opt) in enumerate(tqdm(sols, desc=f"{source}@{frac} schedules")):
        mats, mask, chans = build_color_instance(images[i], frac, lp_upper, channel_blur, coupling)
        x0, y0 = build_pdhg_init(init[0], mats.c.size, mats.G.shape[0], mats.A.shape[0], lp_upper,
                                 pix_channels=chans, mask=mask, M=M, N=images[i].shape[1],
                                 delta=init[1], dual_ineq_init=init[2], dual_eq_init=init[3])
        R = init_radius(x0, y0, raw_x, raw_y)
        for label, g in final_gaps(mats, raw_x, raw_y, x0, y0, schedules).items():
            rows.append(dict(split=f"{source}{_coupling_tag(coupling).replace('_', '+')}@{frac}", idx=i, label=label,
                             gap=g, f_opt=f_opt, R=R))
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(rel=df.gap / (1.0 + df.f_opt.clip(lower=0)))
    agg = {"mean": ("gap", "mean"), "median": ("gap", "median"), "q90": ("gap", lambda g: g.quantile(0.9)),
           "R_med": ("R", "median")}
    agg.update({f"solved@{e:g}": ("rel", lambda r, e=e: (r <= e).mean()) for e in ETAS})
    return df.groupby(["split", "label"]).agg(**agg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--runs", nargs="+", required=True, help="label=path/to/progress.csv")
    ap.add_argument("--color-fracs", nargs="*", type=float, default=[0.1, 0.2])
    ap.add_argument("--n-color", type=int, default=40)
    ap.add_argument("--color-source", choices=["color", "imagenette"], default="color",
                    help="'color' = Tiny ImageNet 64x64; 'imagenette' = Imagenette val split")
    ap.add_argument("--imagenette-dir", type=Path,
                    default=SRC_DIR.parent / "data" / "imagenette2-160")
    ap.add_argument("--color-size", type=int, default=160, help="square size for imagenette")
    ap.add_argument("--color-blur", nargs=3, type=float, default=[0.0, 0.0, 0.0],
                    metavar=("SR", "SG", "SB"), help="per-channel blur std (chromatic aberration)")
    ap.add_argument("--n-jobs", type=int, default=4, help="parallel reference LP solves")
    ap.add_argument("--grad-coupling", type=float, default=None,
                    help="cross-channel edge-consistency weight mu (mu * sum ||D p_c - D p_c'||_1)")
    ap.add_argument("--color-coupling", action="store_true",
                    help="add the inter-channel term sum |p_c - p_c'| to the color LP")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    meta = np.load(args.data_dir / "out_of_sample_metadata.npz")
    init = (str(meta["init_type"]) if "init_type" in meta.files else "center",
            float(meta["warm_delta"]) if "warm_delta" in meta.files else 0.01,
            float(meta["dual_ineq_init"]) if "dual_ineq_init" in meta.files else 0.1,
            float(meta["dual_eq_init"]) if "dual_eq_init" in meta.files else 0.0)
    print(f"init={init}, in-dist missing fraction={float(meta['missing_fraction_in_dist'])}")
    coupling = f"grad{args.grad_coupling:g}" if args.grad_coupling is not None else args.color_coupling
    if coupling and init[0] != "center":
        raise ValueError("--color-coupling supports the center init only")
    schedules = load_schedules(args.runs)

    rows = eval_gray_test(args.data_dir, meta, init, schedules)
    if args.color_source == "imagenette":
        images = load_imagenette(args.imagenette_dir, args.n_color, args.color_size)
    else:
        images = load_color_images(args.n_color)
    for frac in args.color_fracs:
        rows += eval_color(frac, images, args.color_source, float(meta["lp_upper"]), init, schedules,
                           args.color_blur, n_jobs=args.n_jobs, coupling=coupling)
    df = pd.DataFrame(rows).assign(init=init[0], dual_ineq_init=init[2],
                                   train_frac=float(meta["missing_fraction_in_dist"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4g}".format):
        print(summarize(df).to_string())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
