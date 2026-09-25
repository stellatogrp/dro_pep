#!/usr/bin/env python3
"""Postprocess Slurm outputs into a small decision page, plots, and complete CSVs.

No algorithm simulations occur here. Suitable for local plotting after sync down.
"""
import argparse
import html
import json
import re
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[2]
NAMES = {'lasso': 'LASSO / ISTA', 'logreg_gd': 'Logistic / GD', 'logreg_fgm': 'Logistic / FGM'}
REPRESENTATIVE = ['majorization_s0.5_unit_g1', 'majorization_s0.5_training_g2', 'vinit_coarse', 'vinit_safeguarded']
LABELS = {
    'majorization_s0.5_unit_g1': 'BT: initial step 1, no growth',
    'majorization_s0.5_training_g2': 'BT: initial 1/L, growth 2',
    'vinit_coarse': 'Vinit: coarse growth 2',
    'vinit_safeguarded': 'Coarse growth 2 + safeguard',
}


def md_table(frame):
    lines = ['| ' + ' | '.join(frame.columns) + ' |', '| ' + ' | '.join(['---']*len(frame.columns)) + ' |']
    for row in frame.itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join(str(v) for v in row) + ' |')
    return '\n'.join(lines)


def verify_current_figures(frame):
    checks = []
    for problem in NAMES:
        if problem == 'lasso':
            ref = pd.read_csv(ROOT/'src/iclr_data_outputs/archive/lasso/paper_plots/lasso_losses.csv')
            ref = ref.rename(columns={'row': 'split', 'final_loss_mean': 'mean',
                                      'final_loss_q10': 'q10', 'final_loss_q90': 'q90'})
            ref['arch'] = ref.arch.map({'l2o':'L2O','ldro_pep':'DR-L2O','lpep':'OPT-PEP'})
            got = frame[(frame.problem == problem) & ((frame.split == 'test') & (frame.cohort == 'paper248') | (frame.split == 'ood') & (frame.cohort == 'all'))]
        else:
            ref = pd.read_csv(ROOT/f'src/iclr_data_outputs/figures/{problem}_losses.csv')
            got = frame[frame.problem == problem]
        merged = got[got.label.isin(['L2O','DR-L2O','OPT-PEP'])].merge(ref, left_on=['label','K','split'], right_on=['arch','K','split'], suffixes=('_new','_saved'))
        assert len(merged) == 90, (problem, len(merged))
        for metric in ['mean', 'q10', 'q90']:
            np.testing.assert_allclose(merged[metric+'_new'], merged[metric+'_saved'], rtol=2e-8, atol=1e-8)
            checks.append(dict(problem=problem, metric=metric, rows=len(merged),
                               max_abs_error=float(np.max(np.abs(merged[metric+'_new'] - merged[metric+'_saved']))), status='PASS'))
    return checks


def curves(frame, out, problem, cohort='all', pdf=None):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.3))
    colors = ['#487ca5', '#83a17d', '#777777', '#c59858', '#a47e9b', '#557c72']
    series = [('DR-L2O', 'DR-L2O'), ('OPT-PEP', 'OPT-PEP')] + [(v, LABELS[v]) for v in REPRESENTATIVE]
    for row, column in enumerate(['mean', 'equal_matvec_mean']):
        for col, split in enumerate(['test', 'ood']):
            ax = axes[row, col]
            mask = cohort if problem == 'lasso' and split == 'test' else 'all'
            dat = frame[(frame.problem==problem)&(frame.split==split)&(frame.cohort==mask)]
            for j, (key, label) in enumerate(series):
                d = dat[(dat.label==key)|(dat.method==key)].sort_values('K')
                if d.empty:
                    continue
                x = d.K if row == 0 else 2*d.K
                ax.plot(x, d[column], color=colors[j], label=label, linewidth=1.8,
                        marker='o' if j<2 else None, markersize=2.5,
                        linestyle='-' if j<2 else ['--', ':', '-.', '--'][j-2])
            ax.set_yscale('log')
            ax.grid(alpha=.18)
            ax.set_title(('In-distribution' if split=='test' else 'Out-of-distribution') + (f' ({248 if mask=="paper248" else 250} instances)' if row==0 else ''))
            ax.set_xlabel(r'Iterations $K$' if row==0 else 'Matrix-product budget')
            if col == 0:
                ax.set_ylabel(r'Mean $f(x)-f^\star$')
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=9, frameon=False)
    fig.suptitle(NAMES[problem] + (' (paper test subset)' if cohort=='paper248' else ''), y=.99)
    fig.tight_layout(rect=(0,.14,1,.965))
    stem = problem + ('_paper248' if cohort=='paper248' else '')
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(out/f'{stem}.{ext}', dpi=190, bbox_inches='tight')
    if pdf is not None:
        pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def bootstrap_pairs(args, out):
    rng = np.random.default_rng(20260924)
    rows=[]
    for problem in NAMES:
        for split in ['test','ood']:
            dr = np.load(args.input/problem/'raw'/f'ldro_pep_K15_{split}.npz')['losses'][:,-1]
            idx = rng.integers(0,len(dr),(2000,len(dr)))
            for method in REPRESENTATIVE:
                base = args.extra if method=='vinit_safeguarded' else args.input
                if base is None:
                    continue
                path=base/problem/'raw'/f'{method}_{split}.npz'
                if not path.exists():
                    continue
                z=np.load(path)
                for cost in ['iterations','matvec']:
                    step = np.full(len(dr),15) if cost=='iterations' else np.max(np.where(z['matvec']<=30,np.arange(16),0),axis=1)
                    ls=z['losses'][np.arange(len(dr)),step]
                    samples=ls[idx].mean(axis=1)/dr[idx].mean(axis=1)
                    rows.append(dict(problem=problem,split=split,method=method,cost=cost,
                                     ls_over_dr=float(ls.mean()/dr.mean()),
                                     ci_low=float(np.quantile(samples,.025)),ci_high=float(np.quantile(samples,.975))))
    pd.DataFrame(rows).to_csv(out/'paired_bootstrap.csv',index=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--extra',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    frames=[]
    for problem in NAMES:
        assert (args.input/problem/'manifest.json').exists(), f'Incomplete {problem}'
        frames.append(pd.read_csv(args.input/problem/'summary.csv'))
    if args.extra:
        for p in args.extra.glob('*/summary.csv'):
            assert (p.parent/'manifest.json').exists(), f'Incomplete {p}'
            frames.append(pd.read_csv(p))
    frame=pd.concat(frames,ignore_index=True)
    checks=verify_current_figures(frame)
    (out/'figure_reproduction_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    frame.to_csv(out/'all_results.csv',index=False)
    plt.rcParams.update({'text.usetex':True,'font.family':'serif','font.size':11,
                         'axes.titlesize':11,'axes.labelsize':11,'legend.fontsize':9})
    with PdfPages(out/'comparison_plots.pdf', metadata={'Title': 'DR-L2O line-search comparisons'}) as pdf:
        for problem in NAMES:
            curves(frame,out,problem,pdf=pdf)
        curves(frame,out,'lasso','paper248',pdf=pdf)
    bootstrap_pairs(args,out)
    decision=[]
    for problem in NAMES:
        for split in ['test','ood']:
            d=frame[(frame.problem==problem)&(frame.split==split)&(frame.cohort=='all')&(frame.K==15)]
            ref=d[d.label=='DR-L2O'].iloc[0]
            for method in REPRESENTATIVE:
                q=d[d.method==method]
                if q.empty:
                    continue
                r=q.iloc[0]
                decision.append(dict(problem=problem,split=split,method=method,
                                     dr_gap=ref['mean'],ls_gap=r['mean'],ls_over_dr=r['mean']/ref['mean'],
                                     ls_gap_equal_matvec=r.equal_matvec_mean,
                                     equal_matvec_ratio=r.equal_matvec_mean/ref['mean'],
                                     ls_matvecs=r.matvec_mean,ls_function_values=r.function_mean,
                                     dr_batch_ms=1000*ref.batch_seconds,
                                     ls_batch_ms=(1000*r.batch_seconds if method!='vinit_safeguarded' else np.nan),
                                     time_ratio=(r.batch_seconds/ref.batch_seconds if method!='vinit_safeguarded' else np.nan)))
    dec=pd.DataFrame(decision)
    dec.to_csv(out/'decision.csv',index=False)
    display=dec.copy()
    display['problem']=display.problem.map(NAMES)
    display['method']=display.method.map(LABELS)
    for c in display.columns[3:]:
        display[c]=display[c].map(lambda v:f'{v:.4g}' if pd.notna(v) else 'separate job')
    (out/'comparison.md').write_text('# Comparison at 15 iterations\n\nRatios are line-search gap / DR-L2O gap. Above 1 favors DR-L2O. All 250 instances.\n\n'+md_table(display)+'\n')
    sections=[]
    for problem in NAMES:
        d=display[display.problem==NAMES[problem]].drop(columns='problem')
        svg = (out/f'{problem}.svg').read_text()
        svg = svg[svg.index('<svg '):]
        # Keep fragment references local to each plot in the shared HTML document.
        for ident in re.findall(r'id="([^"]+)"', svg):
            svg = svg.replace(f'id="{ident}"', f'id="{problem}-{ident}"')
            svg = svg.replace(f'href="#{ident}"', f'href="#{problem}-{ident}"')
            svg = svg.replace(f'url(#{ident})', f'url(#{problem}-{ident})')
        svg = svg.replace('<svg ', f'<svg role="img" aria-label="{NAMES[problem]} comparison" ', 1)
        sections.append(f'<section><h2>{NAMES[problem]}</h2>{svg}<p><a href="{problem}.pdf">Vector PDF</a></p><div class="scroll">{d.to_html(index=False,escape=True)}</div></section>')
    table_columns=['problem','split','cohort','K','label','method','mean','median','q90','matvec_mean','function_mean','prox_mean','equal_matvec_mean','batch_seconds','fallback_fail']
    payload=frame[table_columns].to_json(orient='records')
    intro='''<p class="eyebrow">ICLR decision check · 24 September 2026</p>
<h1>Line search versus learned schedules</h1>
<p><b>Current-data replay passed.</b> All 270 learned curve means and their 10th/90th quantiles reproduce Vinit’s current commit <code>40398f2</code>. The full sweeps took 37 s (LASSO), 59 s (GD), and 57 s (FGM) on della-stellato.</p>
<div class="callout"><b>Reading the comparison.</b> DR-L2O beats conventional backtracking that only shrinks steps. Allowing step growth makes line search much stronger. For LASSO, DR-L2O can beat full backtracking at equal matrix-product cost; the cheap coarse rule is a stronger competitor. For logistic regression, trial values reuse matrix products, so function counts and timing matter separately.</div>
<p>Ratios in the tables are <b>line-search gap / DR-L2O gap</b>; above 1 favors DR-L2O. The plots show final objective gaps at equal iterations and equal matrix-product budgets. All settings, including unstable ones, are in the explorer and CSV.</p>
<p><a href="comparison_plots.pdf">All comparison plots (PDF)</a> · <a href="DECISION.md">Short recommendation</a> · <a href="reproduction_bundle.zip">Reproduction bundle</a> · <a href="all_results.csv">Complete results CSV</a> · <a href="decision.csv">Decision table CSV</a> · <a href="paired_bootstrap.csv">Paired bootstrap intervals</a> · <a href="figure_reproduction_checks.json">Reproduction checks</a></p>'''
    methods='''<section><h2>Methods and scope</h2><p>Float64 NumPy/SciPy, one CPU thread, zero initialization, existing paper instances and saved learned schedules. No retraining or hyperparameter selection on test/OOD data. Standard backtracking uses shrink factors 0.5 and 0.8, initial step 1/L or 1, and either carries the previous step or first doubles it. LASSO uses the composite majorization condition. Logistic regression tests Armijo c=10<sup>−4</sup> and majorization c=1/2.</p>
<p>For FGM, Armijo at the extrapolated point with an expanding step is a heuristic and can be unstable. Do not interpret these unstable variants as a representative accelerated line-search baseline. The original FISTA guarantee uses a nondecreasing Lipschitz estimate. Step expansion with unchanged momentum is reported as a practical variant. <a href="https://www.tau.ac.il/~becka/solvers/fista">Author’s description of backtracking and the optional growth rule</a>.</p>
<p>Matrix products count forward and transpose products per instance, including the final objective evaluation. Function values used by search, gradients, and proximal evaluations are separate counters. Logistic regression reuses A times the search direction across trials. For an equal matrix-product budget, we return the last completely accepted iterate that fits. An unfinished trial does not advance the solution.</p>
<p>Timing is the median of five whole-batch runs after warmup. It excludes data loading and learned-schedule selection, and omits intermediate diagnostic objective evaluations for fixed schedules. Counters remain enabled. This is CPU batch latency, not GPU throughput or a universal oracle-cost conversion. Training cost is outside this inference comparison.</p>
<p>LASSO’s primary result uses all 250 test instances. The current paper cache removes original rows 111 and 189 from every method. <a href="lasso_paper248.pdf">Matching 248-instance plot</a>. That subset is also available in the explorer. Recovered raw inputs reproduce both current coarse traces and current learned figures; their hashes are recorded in the run manifests.</p>
<p>All methods use a common cached NumPy implementation; timings are not timings of the original training/evaluation scripts. A faster 15-iteration run need not reach a given accuracy sooner. The safeguarded coarse check ran on another node, so its comparative timing is omitted. Its raw timing remains in its run output.</p><p>The safeguarded coarse LASSO check was added after observing 5 test and 8 OOD fallback steps that failed Vinit’s majorization condition. It backtracks on such fallbacks. The original and repaired outcomes are both retained. Bootstrap intervals resample the 250 paired instances (2,000 draws, fixed seed); they describe test-set sampling uncertainty, not training variability.</p></section>'''
    fields=['problem','split','cohort','K','method','mean','median','q90','matvec_mean','function_mean','equal_matvec_mean','batch_seconds','fallback_fail']
    static_table=frame[fields].to_html(index=False,escape=True,table_id='explorer',float_format=lambda v:f'{v:.4g}',na_rep='')
    explorer='''<section><h2>All configurations</h2><p id="static-note">All result rows are shown below. <a href="all_results.csv">Download CSV</a> for sorting and filtering.</p><div id="filters" hidden><label>Problem <select id="problem"><option value="">All</option><option>lasso</option><option>logreg_gd</option><option>logreg_fgm</option></select></label> <label>Horizon <select id="horizon"><option>15</option><option>10</option><option>5</option><option>1</option><option value="">All</option></select></label> <label>Split <select id="split"><option value="">Both</option><option>test</option><option>ood</option></select></label> <label>Cohort <select id="cohort"><option>all</option><option>paper248</option></select></label></div><div class="scroll">'''+static_table+'</div></section>'
    script='''<script>const rows=PAYLOAD; const fields=FIELDS;function update(){const horizon=document.getElementById('horizon');let d=rows.filter(r=>['problem','split','cohort'].every(k=>!document.getElementById(k).value||r[k]===document.getElementById(k).value)&&(!horizon.value||r.K==horizon.value));document.getElementById('explorer').innerHTML='<thead><tr>'+fields.map(f=>'<th>'+f+'</th>').join('')+'</tr></thead><tbody>'+d.map(r=>'<tr>'+fields.map(f=>'<td>'+(typeof r[f]==='number'?(Number.isInteger(r[f])?r[f]:r[f].toPrecision(4)):(r[f]??''))+'</td>').join('')+'</tr>').join('')+'</tbody>'}document.querySelectorAll('select').forEach(s=>s.addEventListener('change',update));update();document.getElementById('filters').hidden=false;document.getElementById('static-note').hidden=true;</script>'''.replace('PAYLOAD',payload).replace('FIELDS',json.dumps(fields))
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DR-L2O line-search comparison</title><style>body{font:16px/1.5 system-ui,sans-serif;margin:0;background:#f5f6f5;color:#25313a}main{max-width:1150px;margin:auto;padding:35px 25px}h1{font-size:34px;line-height:1.15}h2{font-size:24px}a{color:#315f85}section{background:white;padding:22px;margin:28px 0;border-radius:8px}.callout{background:#e4ecef;padding:18px;border-left:4px solid #487ca5}.eyebrow{color:#60727a}svg{width:100%;height:auto;max-width:1000px}table{border-collapse:collapse;font:12px/1.5 ui-monospace,monospace}th,td{padding:8px;text-align:right;border-bottom:1px solid #ddd;white-space:nowrap}th{background:#eff2f3;position:sticky;top:0}td:first-child{text-align:left}.scroll{overflow:auto;max-height:600px}select{padding:6px;margin:5px}code{background:#eef1f2;padding:2px 5px}</style><main>'''+intro+''.join(sections)+explorer+methods+'</main>'+script+'</html>'
    (out/'index.html').write_text(page)
    print(f'Wrote {out}/index.html; {len(frame)} result rows; all figure checks passed.')


if __name__=='__main__':
    main()
