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

    def generic(exp, tag, measure, a, b, sfx, rows, mem, cargs, trim):
        top = {'quad': 'Quad', 'lasso': 'Lasso'}[exp]
        d = f'{BASE}/{top}/chunk_{tag}{sfx}_{measure}_K{a}_{b}'
        tflag = f',CTRIM={trim}' if trim else ''
        cmd = (f'sbatch --time=23:59:59 --mem-per-cpu={mem} '
               f'--export=ALL,CEXP={exp},CTAG={tag},CMEASURE={measure},'
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
        for K in range(33, 41):
            generic('lasso', alg, 'expectation', K, K, '', 13, '40G', A, 0)
        for K in range(41, 46):
            generic('lasso', alg, 'expectation', K, K, '', 7, '45G',
                    A + ' eps.space_count=7', 0)
        for K in range(46, 51):
            generic('lasso', alg, 'expectation', K, K, '', 4, '45G',
                    A + ' eps.space_count=4', 0)
        generic('lasso', alg, 'cvar', 1, 16, '', 39, '12G', A, 0)
        for a, b in [(17, 20), (21, 24)]:
            generic('lasso', alg, 'cvar', a, b, '_trim', 14, '16G', A, 1)
        for K in range(25, 29):
            generic('lasso', alg, 'cvar', K, K, '_trim', 14, '24G', A, 1)
        for K in range(29, 33):
            generic('lasso', alg, 'cvar', K, K, '_trim2', 7, '32G', A, 2)
        for K in range(33, 41):
            generic('lasso', f'{alg}_e4', 'cvar', K, K, '', 4, '45G',
                    A + ' eps.space_count=4 alpha_vals=[0.01]', 0)
        for K, mem in [(45, '60G'), (50, '60G')]:
            for e in EPS4:
                # K=50 forces qdldl: MKL Pardiso SEGFAULTS (not a clean
                # panic) at this size, killing the process before any
                # fallback. 48h wall since qdldl is single-threaded.
                extra = 'DRO_PEP_DIRECT_SOLVER=qdldl,' if K == 50 else ''
                generic('lasso', f'{alg}_eps{e}', 'cvar', K, K, '', 1, mem,
                            f'{A} eps.log_min={e} eps.log_max={e} '
                            f'eps.space_count=1 alpha_vals=[0.01]', 0)
                if K == 50:
                    units[-1]['cmd'] = units[-1]['cmd'].replace(
                        '--time=23:59:59', '--time=47:59:59').replace(
                        '--export=ALL,', f'--export=ALL,{extra}')
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['audit', 'submit'])
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    units = build_units()
    incomplete = [u for u in units if not is_complete(u)]
    print(f'{len(units) - len(incomplete)}/{len(units)} units complete')
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
