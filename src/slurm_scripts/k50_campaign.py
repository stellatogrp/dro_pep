"""K=50 certification campaign manifest: audit and (re)submission.

Single source of truth for every DRO chunk job in the K=50 / N=100
campaign (logreg, quad mu=0, lasso). Run ON THE CLUSTER from src/:

  python slurm_scripts/k50_campaign.py audit             # completeness report
  python slurm_scripts/k50_campaign.py submit --dry-run  # print sbatch cmds for incomplete units
  python slurm_scripts/k50_campaign.py submit            # submit incomplete units

A unit is complete when its chunk dir has a dro.csv containing at least
the expected number of rows (eps grid x alphas) for every K in its range.
Currently-running jobs are NOT detected -- audit only when the queue is
drained or check squeue first.

NOTE: never pass `-q` to sbatch (it means --qos and silently swallows
the next flag; this cost us a day of TIMEOUTs and OOMs).

Sizing (measured): cvar solve time ~ (Gram dim)^6, memory ~ (dim)^3 with
MKL Pardiso needing ~2-3x the RSS estimate. Gram dim = K+2 (logreg/quad),
2K+5 (lasso). Anchors: logreg K=24 = 102s/2.5GB; lasso K=25 = 1510s/21GB.
"""
import argparse
import csv
import os
import re
import subprocess

BASE = '/scratch/gpfs/BSTELLATO/bs37/cert_dro_pep_out/dro_outputs'
S = 'slurm_scripts'
EPS4 = ['-3', '-1.468', '-1.174', '-1']   # lasso anchor eps (log10, linspace grid points)


def _unit(exp, dirpath, a, b, rows, cmd):
    return {'exp': exp, 'dir': dirpath, 'kmin': a, 'kmax': b,
            'rows': rows, 'cmd': cmd}


def build_units():
    units = []

    def logreg(alg, eta, measure, a, b, sfx, rows, trim):
        d = f'{BASE}/LogReg/chunk_{alg}_eta{eta}_{measure}_K{a}_{b}{sfx}'
        tflag = f',CTRIM={trim}' if trim else ''
        cmd = (f'sbatch --time=23:59:59 --mem-per-cpu=6G '
               f'--export=ALL,CALG={alg},CETA={eta},CMEASURE={measure},'
               f'CKMIN={a},CKMAX={b}{tflag} {S}/run_logreg_cert_dro_chunk.sh')
        units.append(_unit('logreg', d, a, b, rows, cmd))

    def generic(exp, tag, measure, a, b, sfx, rows, mem, cargs, trim,
                time='23:59:59', env=''):
        top = {'quad': 'Quad', 'lasso': 'Lasso'}[exp]
        d = f'{BASE}/{top}/chunk_{tag}{sfx}_{measure}_K{a}_{b}'
        tflag = f',CTRIM={trim}' if trim else ''
        envflag = f'{env},' if env else ''
        cmd = (f'sbatch --time={time} --mem-per-cpu={mem} '
               f'--export=ALL,{envflag}CEXP={exp},CTAG={tag},CMEASURE={measure},'
               f'CKMIN={a},CKMAX={b}{tflag},CARGS="{cargs}" '
               f'{S}/run_cert_dro_chunk.sh')
        units.append(_unit(exp, d, a, b, rows, cmd))

    # ---- logreg: GD@1.9, FGM@1, FGM@1.9 (bonus, expectation only) ----
    for alg, eta in [('grad_desc', '1.9'), ('nesterov_fgm', '1'),
                     ('nesterov_fgm', '1.9')]:
        for a, b in [(1, 24), (25, 32), (33, 36), (37, 40), (41, 45), (46, 50)]:
            logreg(alg, eta, 'expectation', a, b, '', 13, 0)
    for alg, eta in [('grad_desc', '1.9'), ('nesterov_fgm', '1')]:
        for a, b in [(1, 16), (17, 24)]:
            logreg(alg, eta, 'cvar', a, b, '', 39, 0)
        for K in range(25, 31):
            logreg(alg, eta, 'cvar', K, K, '', 39, 0)
        for K in range(31, 46):
            logreg(alg, eta, 'cvar', K, K, '_trim', 14, 1)
        for K in range(46, 51):
            logreg(alg, eta, 'cvar', K, K, '_trim2', 7, 2)

    # ---- quad, mu=0 (mu=1 is degenerate: rejection_sample_MP(d,1,1)) ----
    for tag, alg in [('gd_mu0', 'grad_desc'), ('ngd_mu0', 'nesterov_grad_desc')]:
        A = f'alg={alg} mu=0 eps.log_min=-1 eps.log_max=1'
        for a, b in [(1, 32), (33, 40), (41, 45), (46, 50)]:
            generic('quad', tag, 'expectation', a, b, '', 13, '6G', A, 0)
        for a, b in [(1, 24), (25, 28), (29, 30)]:
            generic('quad', tag, 'cvar', a, b, '', 39, '6G', A, 0)
        for K in range(31, 46):
            generic('quad', tag, 'cvar', K, K, '_trim', 14, '6G', A, 1)
        for K in range(46, 51):
            generic('quad', tag, 'cvar', K, K, '_trim2', 7, '6G', A, 2)

    # ---- lasso: ista, fista ----
    for alg in ['ista', 'fista']:
        A = f'alg={alg}'
        generic('lasso', alg, 'expectation', 1, 24, '', 13, '12G', A, 0)
        generic('lasso', alg, 'expectation', 25, 32, '', 13, '24G', A, 0)
        MKL = 'DRO_PEP_DIRECT_SOLVER=mkl'   # fail loud, never crawl in qdldl
        for K in range(33, 41):
            generic('lasso', alg, 'expectation', K, K, '', 13, '40G', A, 0,
                    env=MKL)
        for K in range(41, 46):
            generic('lasso', alg, 'expectation', K, K, '', 7, '60G',
                    A + ' eps.space_count=7', 0, env=MKL)
        for K in range(46, 51):
            generic('lasso', alg, 'expectation', K, K, '', 4, '60G',
                    A + ' eps.space_count=4', 0, env=MKL)
        generic('lasso', alg, 'cvar', 1, 16, '', 39, '12G', A, 0)
        for a, b in [(17, 20), (21, 24)]:
            generic('lasso', alg, 'cvar', a, b, '_trim', 14, '16G', A, 1)
        for K in range(25, 29):
            generic('lasso', alg, 'cvar', K, K, '_trim', 14, '24G', A, 1)
        for K in range(29, 33):
            generic('lasso', alg, 'cvar', K, K, '_trim2', 7, '32G', A, 2)
        for K in range(33, 38):
            generic('lasso', f'{alg}_e4', 'cvar', K, K, '', 4, '60G',
                    A + ' eps.space_count=4 alpha_vals=[0.01]', 0, env=MKL)
        for K in range(38, 41):
            generic('lasso', f'{alg}_e4', 'cvar', K, K, '', 4, '90G',
                    A + ' eps.space_count=4 alpha_vals=[0.01]', 0,
                    time='35:59:59', env=MKL)
        # anchors: one eps per job. K=45 is near the MKL LP64 int32 factor
        # ceiling (~2^31 nnz); K=50 is likely beyond it -- the 190G/cpu
        # submission doubles as the falsification test (segfault in
        # minutes = overflow, not memory).
        for K, mem in [(45, '120G'), (50, '190G')]:
            for e in EPS4:
                generic('lasso', f'{alg}_eps{e}', 'cvar', K, K, '', 1, mem,
                        f'{A} eps.log_min={e} eps.log_max={e} '
                        f'eps.space_count=1 alpha_vals=[0.01]', 0,
                        time='47:59:59', env=MKL)
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
        sfx = {None: '', '0': '', '1': '_trim', '2': '_trim2'}[gm(r'CTRIM=(\d)')]
        if not (kmin and meas):
            continue
        if 'CEXP=' in line:
            exp, tag = gm(r'CEXP=(\w+)'), gm(r'CTAG=([^,]+)')
            top = {'quad': 'Quad', 'lasso': 'Lasso', 'logreg': 'LogReg'}[exp]
            dirs.add(f'{BASE}/{top}/chunk_{tag}{sfx}_{meas}_K{kmin}_{kmax}')
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
