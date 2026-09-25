"""Stress-test learned PDHG schedules for TV inpainting on shifted instance families.

Evaluation only (center init). Schedules are chosen as in pdlp_warmstart_explore.py
(validation-best row of each progress.csv, plus the untrained row 0).

Families (``--family``):
  gray_checker  held-out gray test faces + amp * checkerboard (the top singular
                mode of the TV difference operator), clipped to [0, 1]; test masks.
  gray_noise    held-out gray test faces + iid Uniform(-amp, amp) noise, clipped.
  color_scratch Tiny ImageNet RGB images with a scratch mask (random line
                segments) covering ~``--frac`` of the pixels, shared by channels.

Usage (from src/):
    python tools/pdlp_stress_explore.py --data-dir <sample dir> --runs l2o=<csv> dr=<csv> \
        --family gray_checker --amps 0.05 0.1 0.2 --out results.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parent.parent
for _p in (SRC_DIR, SRC_DIR / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from learning.tv_inpainting_color_averages import load_color_images  # noqa: E402
from learning.tv_inpainting_test import extract_constraint_matrices  # noqa: E402
from learning_experiment_classes.pdlp import build_pdhg_init, solve_lp_checked  # noqa: E402
from pdlp_warmstart_explore import final_gaps, init_radius, load_schedules, summarize  # noqa: E402

CACHE = SRC_DIR / "iclr_data_outputs/explore_warmstart/lp_cache"


def scratch_mask(M, N, frac, rng, max_len=24):
    """Known-pixel mask with random 1-2 px wide line scratches covering ~frac of pixels."""
    missing = np.zeros((M, N), bool)
    while missing.mean() < frac:
        r0, c0 = rng.integers(M), rng.integers(N)
        ang, L, w = rng.uniform(0, np.pi), rng.integers(8, max_len), rng.integers(1, 3)
        for t in range(L):
            r = int(round(r0 + t * np.sin(ang)))
            c = int(round(c0 + t * np.cos(ang)))
            missing[max(r, 0):min(r + w, M), max(c, 0):min(c + w, N)] = True
    return ~missing


def gray_instance(pix, mask, M, N):
    known = np.flatnonzero(mask)
    return extract_constraint_matrices(known, pix[known], M, N)


def gray_build(pix, mask, M, N):
    m = gray_instance(pix, mask, M, N)
    return m, [m]


def color_instance(chans, mask, M, N):
    mats = [gray_instance(p, mask, M, N) for p in chans]
    return mats[0]._replace(
        c=np.concatenate([m.c for m in mats]), A=sp.block_diag([m.A for m in mats], format="csr"),
        b=np.concatenate([m.b for m in mats]), G=sp.block_diag([m.G for m in mats], format="csr"),
        h=np.concatenate([m.h for m in mats]), l=np.concatenate([m.l for m in mats]),
        u=np.concatenate([m.u for m in mats])), mats


def solve_cached(key, build):
    """Reference (raw_x, raw_y, f_opt) for one instance; per-channel solves stacked for color."""
    path = CACHE / key
    if path.exists():
        d = np.load(path)
        return d["raw_x"], d["raw_y"], float(d["f_opt"])
    _, parts = build()
    xs, yg, ya, f = [], [], [], 0.0
    for m in parts:
        s = solve_lp_checked(m)
        ry = np.ravel(s["raw_y"])
        xs.append(np.ravel(s["raw_x"])); yg.append(ry[:m.G.shape[0]]); ya.append(ry[m.G.shape[0]:])
        f += float(s["objective_value"])
    raw_x, raw_y = np.concatenate(xs), np.concatenate(yg + ya)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, raw_x=raw_x, raw_y=raw_y, f_opt=f)
    return raw_x, raw_y, f


def make_instances(args, meta):
    """List of (split, idx, builder) where builder() -> (mats, [per-channel mats])."""
    M, N = int(meta["M"]), int(meta["N"])
    out = []
    if args.family.startswith("gray"):
        from sklearn.datasets import fetch_olivetti_faces
        faces = fetch_olivetti_faces().images.astype(np.float64)
        d = np.load(args.data_dir / "test_set.npz")
        checker = np.indices((M, N)).sum(0) % 2 * 2.0 - 1.0
        for amp in args.amps:
            rng = np.random.default_rng(0)
            for i in range(len(d["image_index_batch"])):
                img = faces[int(d["image_index_batch"][i])]
                pert = amp * checker if args.family == "gray_checker" else rng.uniform(-amp, amp, (M, N))
                pix = np.clip(img + pert, 0.0, 1.0).reshape(-1)
                mask = d["mask_batch"][i].reshape(-1)
                out.append((f"{args.family}@{amp:g}", i,
                            lambda pix=pix, mask=mask: gray_build(pix, mask, M, N)))
    else:
        imgs = load_color_images(args.n_color)
        for i, im in enumerate(imgs):
            Mi, Ni, _ = im.shape
            mask = scratch_mask(Mi, Ni, args.frac, np.random.default_rng(1000 + i)).reshape(-1)
            chans = [im[:, :, c].astype(np.float64).reshape(-1) / 255.0 for c in range(3)]
            out.append((f"color_scratch@{args.frac:g}", i,
                        lambda chans=chans, mask=mask, Mi=Mi, Ni=Ni: color_instance(chans, mask, Mi, Ni)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--family", choices=["gray_checker", "gray_noise", "color_scratch"], required=True)
    ap.add_argument("--amps", nargs="*", type=float, default=[0.05, 0.1, 0.2])
    ap.add_argument("--frac", type=float, default=0.1)
    ap.add_argument("--n-color", type=int, default=40)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    meta = np.load(args.data_dir / "out_of_sample_metadata.npz")
    if ("init_type" in meta.files and str(meta["init_type"]) != "center") or float(meta["lp_upper"]) != 1.0:
        raise ValueError("stress tool supports center init with lp_upper = 1 only")
    schedules = load_schedules(args.runs)
    insts = make_instances(args, meta)
    sols = Parallel(n_jobs=args.n_jobs)(
        delayed(solve_cached)(f"stress_{split}/{idx}.npz", b) for split, idx, b in tqdm(insts, desc="solves"))
    rows = []
    for (split, idx, build), (raw_x, raw_y, f_opt) in tqdm(list(zip(insts, sols)), desc="schedules"):
        mats, _ = build()
        x0, y0 = build_pdhg_init("center", mats.c.size, mats.G.shape[0], mats.A.shape[0], 1.0)
        R = init_radius(x0, y0, raw_x, raw_y)
        for label, g in final_gaps(mats, raw_x, raw_y, x0, y0, schedules).items():
            rows.append(dict(split=split, idx=idx, label=label, gap=g, f_opt=f_opt, R=R))
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4g}".format):
        print(summarize(df).to_string())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
