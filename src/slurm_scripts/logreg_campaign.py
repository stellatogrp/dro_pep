"""LogReg certification campaign manifest: audit and (re)submission.

Single source of truth for the DRO chunk jobs of the real-data logistic
regression experiment (german.numer standardized, K = 1..30,
N = 100 in-sample). Run ON THE CLUSTER from src/:

  python slurm_scripts/logreg_campaign.py audit             # completeness report
  python slurm_scripts/logreg_campaign.py submit --dry-run  # print sbatch cmds
  python slurm_scripts/logreg_campaign.py submit            # submit incomplete units

A unit is complete when its chunk dir has a dro.csv containing at least
the expected number of rows (eps grid x alphas) for every K in its range.
The audit skips units whose job is currently in the queue.

One wide eps grid per (algorithm, measure): logspace 1e-8..10^-0.5,
16 points (config default). The bottom decade matters: the certificate is
exact in the eps -> 0 limit but has nonzero radius sensitivity, so a grid
that stops too high silently inflates the tail bounds (measured on the
earlier a9a campaign).

NOTE: never pass `-q` to sbatch (it means --qos and silently swallows
the next flag).
"""
import argparse
import csv
import os
import re
import subprocess

BASE = '/scratch/gpfs/BSTELLATO/bs37/cert_dro_pep_out/dro_outputs'
S = 'slurm_scripts'


def _unit(dirpath, a, b, rows, cmd):
    return {'dir': dirpath, 'kmin': a, 'kmax': b, 'rows': rows, 'cmd': cmd}


def build_units():
    units = []

    def base(alg, eta, measure, a, b, rows):
        d = f'{BASE}/LogReg/chunk_{alg}_eta{eta}_{measure}_K{a}_{b}'
        cmd = (f'sbatch --time=23:59:59 --mem-per-cpu=6G '
               f'--export=ALL,CALG={alg},CETA={eta},CMEASURE={measure},'
               f'CKMIN={a},CKMAX={b} {S}/run_logreg_cert_dro_chunk.sh')
        units.append(_unit(d, a, b, rows, cmd))

    # single grid (config: logspace 1e-8..10^-0.5, 16 points): exp = 16
    # rows/K, cvar = 48 rows/K (3 alphas)
    for alg, eta in [('grad_desc', '1.9'), ('nesterov_fgm', '1'),
                     ('nesterov_fgm', '1.9')]:
        for a, b in [(1, 16), (17, 30)]:
            base(alg, eta, 'expectation', a, b, 16)
    for alg, eta in [('grad_desc', '1.9'), ('nesterov_fgm', '1')]:
        for a, b in [(1, 12), (13, 20), (21, 24)]:
            base(alg, eta, 'cvar', a, b, 48)
        for K in range(25, 31):
            base(alg, eta, 'cvar', K, K, 48)
    return units


def is_complete(u):
    f = os.path.join(u['dir'], 'dro.csv')
    if not os.path.isfile(f):
        return False
    counts = {}
    with open(f) as fh:
        for row in csv.DictReader(fh):
            k = int(float(row['K']))
            counts[k] = counts.get(k, 0) + 1
    return all(counts.get(k, 0) >= u['rows']
               for k in range(u['kmin'], u['kmax'] + 1))


def running_unit_dirs():
    """Reconstruct the unit dir of every currently-running job from its
    sbatch SubmitLine, so audit/submit can skip in-flight units."""
    ids = subprocess.run(['squeue', '-u', 'bs37', '-h', '-o', '%i'],
                         capture_output=True, text=True).stdout.split()
    dirs = set()
    for j in ids:
        line = subprocess.run(
            ['sacct', '-j', j, '-X', '-n', '--format=SubmitLine%400'],
            capture_output=True, text=True).stdout

        def gm(pat):
            m = re.search(pat, line)
            return m.group(1) if m else None
        kmin, kmax = gm(r'CKMIN=(\d+)'), gm(r'CKMAX=(\d+)')
        meas = gm(r'CMEASURE=(\w+)')
        sfx = {None: '', '0': '', '1': '_trim', '2': '_trim2',
               '3': '_trimA', '4': '_trimB'}[gm(r'CTRIM=(\d)')]
        if not (kmin and meas):
            continue
        if 'CEXP=' in line:
            tag = gm(r'CTAG=([^,]+)')
            dirs.add(f'{BASE}/LogReg/chunk_{tag}{sfx}_{meas}_K{kmin}_{kmax}')
        elif 'CALG=' in line:
            alg, eta = gm(r'CALG=(\w+)'), gm(r'CETA=([\d.]+)')
            dirs.add(f'{BASE}/LogReg/chunk_{alg}_eta{eta}_{meas}_K{kmin}_{kmax}{sfx}')
    return dirs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['audit', 'submit'])
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    units = build_units()
    running = running_unit_dirs()
    incomplete = [u for u in units
                  if not is_complete(u) and u['dir'] not in running]
    n_run = sum(1 for u in units if u['dir'] in running)
    print(f'{len(units) - len(incomplete) - n_run}/{len(units)} units complete, '
          f'{n_run} running (skipped)')
    for u in incomplete:
        print(f'  INCOMPLETE {os.path.basename(u["dir"])}')
    if args.action == 'submit':
        for u in incomplete:
            if args.dry_run:
                print(u['cmd'])
            else:
                subprocess.run(u['cmd'], shell=True, check=True)
        print(f'{"would submit" if args.dry_run else "submitted"} '
              f'{len(incomplete)} jobs')


if __name__ == '__main__':
    main()
