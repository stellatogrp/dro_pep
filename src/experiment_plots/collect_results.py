"""Collect campaign results from the cluster into the plot layout.

Run LOCALLY from src/experiment_plots/:

  python collect_results.py --pull            # ssh+tar the CSVs into _mirror/
  python collect_results.py --merge           # _mirror/ -> LogReg/data/
  python collect_results.py --pull --merge    # both

Pull grabs only CSVs (small). Merge concatenates the per-K-range chunk
dro.csv files into one dro.csv per (algorithm, measure), deduplicating on
(K, eps, alpha) with the last occurrence winning (reruns supersede partial
results from timed-out jobs), and copies the newest samples/pep outputs
per SLURM array task. Everything the figures need is then under
<Exp>/data/, and each plot_bounds.py runs on it directly.

The chunk-dir naming matches the submission side: LogReg comes from
slurm_scripts/logreg_campaign.py (the campaign manifest), Quad and Lasso
from slurm_scripts/submit_epsx.sh. Rename a CTAG there and the globs in
DRO_MAP have to follow.

Quad/Lasso samples and pep are NOT collected here: they are produced by
run_q{pep,dro}_experiment.sh under a different scratch root, and their
data/ dirs are committed. Only the DRO chunks are merged for them.
"""
import argparse
import csv
import glob
import os
import shutil
import subprocess

HOST = os.environ.get('DRO_PEP_CLUSTER_HOST', 'della-stellato')
REMOTE = '/scratch/gpfs/BSTELLATO/bs37/cert_dro_pep_out'
HERE = os.path.dirname(os.path.abspath(__file__))
MIRROR = os.path.join(HERE, '_mirror')

# samples/pep SLURM array task id -> data dir name
SAMPLES_MAP = {
    'LogReg': {'0': 'GD_1_50', '1': 'FGM_1_50', '2': 'FGM19_bonus_1_50'},
}
PEP_MAP = {
    'LogReg': {'0': 'GD_1_50', '1': 'FGM_1_50'},
}
# Eps grids submitted per experiment by slurm_scripts/submit_epsx.sh; the grid
# name is the second field of the CTAG, so <alg>_<grid> identifies a chunk set.
QUAD_GRIDS = ('base', 'ext', 'ext2')
LASSO_GRIDS = ('base', 'ext', 'mid')


def epsx_globs(alg, grids, measure):
    """Chunk globs for one (alg, measure) across submit_epsx.sh's eps grids.

    Deliberately enumerates the grids instead of globbing chunk_<alg>_*: the
    July campaign left chunk_<alg>_{epsx,epsx2,basefill}_* dirs on the cluster
    that were run at training N=1, before configs/{quad,lasso}.yaml moved to
    N=100. Those rows share (K, eps, alpha) keys with the N=100 reruns but
    carry different values (up to ~13% apart at K=1) and dro.csv has no N
    column to tell them apart, so a wildcard here would silently mix runs.
    """
    return [f'chunk_{alg}_{g}_trimA_{measure}_*' for g in grids]


# dro data dir -> chunk glob(s) relative to dro_outputs/<Exp>/
DRO_MAP = {
    'LogReg': {
        'GD_exp_1_50':    ['chunk_grad_desc_eta1.9_expectation_*', 'chunk_gd19epsx_expectation_*'],
        'GD_cvar_1_50':   ['chunk_grad_desc_eta1.9_cvar_*', 'chunk_gd19epsx_*cvar_*'],
        'FGM_exp_1_50':   ['chunk_nesterov_fgm_eta1_expectation_*', 'chunk_fgm1epsx_expectation_*'],
        'FGM_cvar_1_50':  ['chunk_nesterov_fgm_eta1_cvar_*', 'chunk_fgm1epsx_*cvar_*'],
        'FGM19_exp_1_50': ['chunk_nesterov_fgm_eta1.9_expectation_*', 'chunk_fgm19epsx_expectation_*'],
    },
    # The expectation chunks also write an alpha column (0.01), so expectation
    # and cvar must stay in separate globs or the (K, eps, alpha) dedup below
    # would collapse them onto each other.
    'Quad': {
        'grad_desc_exp_1_50':            epsx_globs('gd', QUAD_GRIDS, 'expectation'),
        'grad_desc_cvar_1_50':           epsx_globs('gd', QUAD_GRIDS, 'cvar'),
        'nesterov_grad_desc_exp_1_50':   epsx_globs('fgm', QUAD_GRIDS, 'expectation'),
        'nesterov_grad_desc_cvar_1_50':  epsx_globs('fgm', QUAD_GRIDS, 'cvar'),
    },
    'Lasso': {
        'ISTA_exp_1_50':   epsx_globs('ista', LASSO_GRIDS, 'expectation'),
        'ISTA_cvar_1_50':  epsx_globs('ista', LASSO_GRIDS, 'cvar'),
        'FISTA_exp_1_50':  epsx_globs('fista', LASSO_GRIDS, 'expectation'),
        'FISTA_cvar_1_50': epsx_globs('fista', LASSO_GRIDS, 'cvar'),
    },
}


def pull():
    os.makedirs(MIRROR, exist_ok=True)
    remote_cmd = (
        f"cd {REMOTE} && tar czf - "
        f"$(find sample_outputs pep_outputs -name '*.csv' 2>/dev/null) "
        f"$(ls dro_outputs/*/chunk_*/dro.csv 2>/dev/null)"
    )
    tar_local = os.path.join(MIRROR, 'pull.tgz')
    with open(tar_local, 'wb') as f:
        subprocess.run(['ssh', HOST, remote_cmd], stdout=f, check=True)
    subprocess.run(['tar', 'xzf', tar_local, '-C', MIRROR], check=True)
    os.remove(tar_local)
    print(f'pulled into {MIRROR}')


def newest_task_dirs(stage, exp):
    """{task_suffix: newest dir} among <stage>_outputs/<exp>/<date>/<time>_<task>."""
    out = {}
    for d in sorted(glob.glob(f'{MIRROR}/{stage}_outputs/{exp}/*/*_*')):
        task = d.rsplit('_', 1)[-1]
        out[task] = d   # sorted => later date/time wins
    return out


def merge():
    for exp in DRO_MAP:
        data = os.path.join(HERE, exp, 'data')
        # samples + pep (only for experiments whose stages run in this campaign)
        for stage, mapping in [('sample', SAMPLES_MAP.get(exp, {})),
                               ('pep', PEP_MAP.get(exp, {}))]:
            if not mapping:
                continue
            sub = 'samples' if stage == 'sample' else 'pep'
            found = newest_task_dirs(stage, exp)
            for task, name in mapping.items():
                if task not in found:
                    print(f'  [{exp}] {sub} task {task} ({name}): MISSING')
                    continue
                dst = os.path.join(data, sub, name)
                os.makedirs(dst, exist_ok=True)
                for f in glob.glob(os.path.join(found[task], '*.csv')):
                    shutil.copy(f, dst)
                print(f'  [{exp}] {sub}/{name} <- {os.path.relpath(found[task], MIRROR)}')
        # dro chunks -> merged dro.csv
        for name, patterns in DRO_MAP[exp].items():
            files = []
            for pat in patterns:
                files += glob.glob(f'{MIRROR}/dro_outputs/{exp}/{pat}/dro.csv')
            # Oldest first, so a rerun (written later) supersedes the partial
            # result it replaces. Path order does not imply run order.
            files = sorted(set(files), key=os.path.getmtime)
            if not files:
                print(f'  [{exp}] dro/{name}: no chunks found')
                continue
            header, rows = None, {}
            for f in files:
                with open(f) as fh:
                    r = list(csv.reader(fh))
                if not r:
                    continue
                if header is not None and r[0] != header:
                    raise ValueError(
                        f'{f}: header {r[0]} differs from {header}; '
                        f'chunks were produced by incompatible code versions')
                header = r[0]
                iK = header.index('K')
                iE = header.index('eps')
                iA = header.index('alpha') if 'alpha' in header else None
                for row in r[1:]:
                    key = (float(row[iK]), float(row[iE]),
                           float(row[iA]) if iA is not None else -1.0)
                    rows[key] = row   # later file wins (reruns supersede)
            ordered = [rows[k] for k in sorted(rows)]
            dst = os.path.join(data, 'dro', name)
            os.makedirs(dst, exist_ok=True)
            with open(os.path.join(dst, 'dro.csv'), 'w', newline='') as fh:
                w = csv.writer(fh)
                w.writerow(header)
                w.writerows(ordered)
            Ks = sorted({k[0] for k in rows})
            print(f'  [{exp}] dro/{name}: {len(files)} chunks, {len(ordered)} rows, '
                  f'K {int(Ks[0])}..{int(Ks[-1])}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pull', action='store_true')
    p.add_argument('--merge', action='store_true')
    args = p.parse_args()
    if not (args.pull or args.merge):
        p.error('pass --pull and/or --merge')
    if args.pull:
        pull()
    if args.merge:
        merge()
