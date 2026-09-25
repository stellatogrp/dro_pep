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
REPRESENTATIVE = ['boyd_backtracking']
LABELS = {'boyd_backtracking': 'Backtracking'}
BASELINE_LABELS = {'lasso':'Proximal backtracking (unit reset)', 'logreg_gd':'Boyd backtracking', 'logreg_fgm':'Armijo at extrapolated point'}
NOTES = {
    'lasso':'Proximal adaptation: quadratic majorization of the smooth loss; unit trial reset and shrink factor 0.5.',
    'logreg_gd':'Boyd and Vandenberghe, Algorithm 9.2: unit trial reset, Armijo coefficient 0.1, shrink factor 0.5.',
    'logreg_fgm':'Accelerated adaptation: Armijo at the extrapolated point; unit trial reset, coefficient 0.1, shrink factor 0.5.',
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
    colors = ['#487ca5', '#83a17d', '#c59858']
    series = [('DR-L2O', 'DR-L2O'), ('OPT-PEP', 'OPT-PEP')] + [(v, BASELINE_LABELS[problem]) for v in REPRESENTATIVE]
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
                        linestyle='-' if j<2 else '--')
            ax.set_yscale('log')
            ax.grid(alpha=.18)
            ax.set_title(('In-distribution' if split=='test' else 'Out-of-distribution') + (f' ({248 if mask=="paper248" else 250} instances)' if row==0 else ''))
            ax.set_xlabel(r'Iterations $K$' if row==0 else 'Matrix-product budget')
            if col == 0:
                ax.set_ylabel(r'Mean $f(x)-f^\star$')
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=9, frameon=False)
    fig.suptitle(NAMES[problem] + (' (paper test subset)' if cohort=='paper248' else ''), y=.99)
    fig.tight_layout(rect=(0,.09,1,.965))
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
                path=args.input/problem/'raw'/f'{method}_{split}.npz'
                assert path.exists(), f'Missing Boyd run: {path}'
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
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    frames=[]
    for problem in NAMES:
        assert (args.input/problem/'manifest.json').exists(), f'Incomplete {problem}'
        manifest=json.loads((args.input/problem/'manifest.json').read_text())
        assert manifest['line_search']['initial_trial_each_iteration'] == 1.0
        assert manifest['line_search']['shrink'] == 0.5
        assert [v['name'] for v in manifest['settings']] == REPRESENTATIVE
        frames.append(pd.read_csv(args.input/problem/'summary.csv'))
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
                                     ls_batch_ms=1000*r.batch_seconds,
                                     time_ratio=r.batch_seconds/ref.batch_seconds))
    dec=pd.DataFrame(decision)
    dec.to_csv(out/'decision.csv',index=False)
    display=dec.copy()
    display['problem']=display.problem.map(NAMES)
    display['split']=display.split.map({'test':'ID','ood':'OOD'})
    for c in display.columns[3:]:
        display[c]=display[c].map(lambda v:f'{v:.4g}')
    overview=display[['problem','split','ls_over_dr','equal_matvec_ratio','time_ratio']].rename(columns={
        'problem':'Algorithm','split':'Split','ls_over_dr':'Gap ratio, 15 iterations',
        'equal_matvec_ratio':'Gap ratio, 30 products','time_ratio':'Time ratio, 15 iterations'})
    brief_columns={'split':'Split','dr_gap':'DR-L2O gap','ls_gap':'BT gap',
                   'ls_over_dr':'BT / DR, 15 iters','equal_matvec_ratio':'BT / DR, 30 products',
                   'ls_matvecs':'BT products at 15 iters','time_ratio':'BT / DR time'}
    sections=[]
    for problem in NAMES:
        d=display[display.problem==NAMES[problem]][list(brief_columns)].rename(columns=brief_columns)
        svg=(out/f'{problem}.svg').read_text()
        svg=svg[svg.index('<svg '):]
        for ident in re.findall(r'id="([^"]+)"', svg):
            svg=svg.replace(f'id="{ident}"', f'id="{problem}-{ident}"')
            svg=svg.replace(f'href="#{ident}"', f'href="#{problem}-{ident}"')
            svg=svg.replace(f'url(#{ident})', f'url(#{problem}-{ident})')
        svg=svg.replace('<svg ', f'<svg role="img" aria-label="{NAMES[problem]} comparison" ', 1)
        sections.append(f'<section class="plot"><h2>{NAMES[problem]}</h2><p class="caption">{NOTES[problem]}</p>{svg}<div class="scroll">{d.to_html(index=False,escape=True)}</div><p class="caption">BT is the displayed backtracking baseline. Ratios above 1 favor DR-L2O. <a href="{problem}.pdf">Vector plot PDF</a>.</p></section>')
    takeaway=(f'At 15 iterations, DR-L2O has a lower mean gap in {int((dec.ls_over_dr>1).sum())} of 6 comparisons. '
              f'At a budget of 30 matrix products, it has a lower mean gap in {int((dec.equal_matvec_ratio>1).sum())} of 6 comparisons. '
              'For LASSO, backtracking reaches a lower gap at equal iterations; the cost comparison favors DR-L2O. '
              'For logistic regression, DR-L2O reaches a lower gap on both splits. These conclusions concern this specified baseline and initialization.')
    intro='''<p class="eyebrow">ICLR decision check · 24 September 2026</p>
<h1>Boyd backtracking versus learned schedules</h1>
<p>This comparison uses one adaptive baseline: <b>restart each line search at a unit trial step, then halve failed trials</b>. Logistic GD follows Boyd and Vandenberghe, Algorithm 9.2, with Armijo coefficient 0.1. LASSO and logistic FGM use the explicitly labeled adaptations below. Parameters were fixed before rerunning.</p>
<div class="callout">'''+takeaway+'''</div>
<h2>Comparison at 15 iterations</h2><p>Gap ratios are <b>backtracking / DR-L2O</b>; above 1 favors DR-L2O. ID and OOD each use 250 instances. Timing ratios compare CPU batch latency at 15 iterations, with different achieved accuracies.</p>
<div class="scroll">'''+overview.to_html(index=False,escape=True)+'''</div>
<p class="downloads"><a href="report.pdf">Full webpage PDF</a> · <a href="comparison_plots.pdf">All plots PDF</a> · <a href="all_results.csv">Complete results CSV</a> · <a href="paired_bootstrap.csv">Paired bootstrap intervals</a> · <a href="reproduction_bundle.zip">Reproduction bundle</a></p>'''
    methods='''<section class="methods"><h2>Backtracking rule and adaptations</h2>
<p><b>Logistic GD.</b> Use the negative gradient at the current iterate. Start each search at step 1, accept the Armijo decrease condition with coefficient 0.1, and otherwise halve the step. This is <a href="https://www.seas.ucla.edu/~vandenbe/cvxbook/bv_cvxbook.pdf#page=478">Boyd and Vandenberghe, Algorithm 9.2, p. 464</a>.</p>
<p><b>LASSO / ISTA.</b> Use the proximal-gradient trial with soft thresholding. Test quadratic majorization of the smooth least-squares term, as in <a href="https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf#page=30">Parikh and Boyd, Section 4.2</a>. This adaptation resets the trial step to 1 and halves failures; the cited proximal algorithm itself carries the previous step. Smooth Armijo is not applied to the nonsmooth LASSO objective.</p>
<p><b>Logistic FGM.</b> Apply Armijo at the extrapolated point with the existing momentum sequence. This is an adaptation of the line-search rule, not the gradient-descent algorithm in the book. No accelerated convergence guarantee is asserted for this combination.</p>
<p><b>Observed steps.</b> Both logistic baselines accepted step 1 on every instance and iteration. LASSO accepted steps between 0.25 and 1. These are observed outcomes, not settings selected from the test data.</p></section>'''
    provenance='''<section class="provenance"><h2>Validation and cost accounting</h2>
<p>All jobs ran on della-stellato with one CPU and 2 GB each: 23 s (LASSO), 23 s (GD), and 16 s (FGM). Job array: <code>14395605</code>. Learned schedules were rerun in the same jobs. All 270 saved learned-curve means and their 10th/90th quantiles reproduce Vinit’s commit <code>40398f2</code>. An analytic check verifies acceptance of step 0.25 followed by step 1, confirming the reset. Every accepted baseline trial passed its condition.</p>
<p>Float64 NumPy/SciPy, zero initialization, one CPU thread, existing instances and schedules. No retraining or test/OOD tuning. Plots compare DR-L2O, OPT-PEP and the stated baseline. L2O is also included in the table and CSV.</p>
<p>Matrix products count forward and transpose products per instance, including the final objective’s product. Logistic regression reuses the matrix product with the search direction across trials. At an equal product budget, only the last fully accepted iterate that fits is returned. Function values and proximal evaluations are counted separately.</p>
<p>Timing is the median of five batch runs after warmup. It excludes data loading, schedule selection and training; fixed schedules omit intermediate diagnostic objective evaluations. This is inference latency at equal iterations, not time to a common accuracy. Bootstrap intervals resample the 250 paired instances 2,000 times with a fixed seed and measure test-set sampling uncertainty.</p>
<p>Primary LASSO results keep all 250 instances. The paper’s test cache excludes original rows 111 and 189. That exact 248-instance comparison is available in the table and <a href="lasso_paper248.pdf">matching vector plot</a>. Recovered LASSO inputs passed full trajectory checks against current committed caches; logistic inputs came from that commit. Hashes and schedule selections are in the manifests.</p>
<p><a href="figure_reproduction_checks.json">Figure reproduction checks</a> · <a href="decision.csv">Decision CSV</a> · <a href="DECISION.md">Decision memo</a>. Runner checkpoint: <code>fa8a9e9</code>. Data: <code>results/linesearch-boyd-v1</code>. Branch: <code>experiments/linesearch-iclr-20260924</code>. Earlier experimental data remain in the reproduction archive. The displayed comparison is restricted to the rule above.</p></section>'''
    fields=['problem','split','cohort','K','label','mean','matvec_mean','function_mean','batch_ms']
    titles={'problem':'Problem','split':'Split','cohort':'Cohort','K':'K','label':'Method','mean':'Mean gap',
            'matvec_mean':'Products','function_mean':'Search values','batch_ms':'Batch ms'}
    table_frame=frame.copy()
    table_frame['batch_ms']=1000*table_frame.batch_seconds
    table_frame.loc[table_frame.method=='boyd_backtracking','label']='Backtracking'
    payload=table_frame[fields].to_json(orient='records')
    default=table_frame[(table_frame.K==15)&(table_frame.cohort=='all')]
    static_table=default[fields].rename(columns=titles).to_html(index=False,escape=True,table_id='explorer',float_format=lambda v:f'{v:.4g}',na_rep='')
    explorer='''<section class="results"><h2>Results table</h2><p>Default view: 15 iterations on all 250 instances. The <a href="all_results.csv">complete CSV</a> contains all 420 rows, intermediate horizons and the paper’s LASSO subset. Timing covers the whole 250-instance batch.</p><div id="filters" hidden><label>Problem <select id="problem"><option value="">All</option><option>lasso</option><option>logreg_gd</option><option>logreg_fgm</option></select></label> <label>Horizon <select id="horizon"><option>15</option><option>10</option><option>5</option><option>1</option><option value="">All</option></select></label> <label>Split <select id="split"><option value="">Both</option><option>test</option><option>ood</option></select></label> <label>Cohort <select id="cohort"><option>all</option><option>paper248</option></select></label></div><div class="scroll">'''+static_table+'</div></section>'
    script='''<script>const rows=PAYLOAD;const fields=FIELDS;const titles=TITLES;
function update(){const horizon=document.getElementById('horizon');let d=rows.filter(r=>['problem','split','cohort'].every(k=>!document.getElementById(k).value||r[k]===document.getElementById(k).value)&&(!horizon.value||r.K==horizon.value));document.getElementById('explorer').innerHTML='<thead><tr>'+fields.map(f=>'<th>'+titles[f]+'</th>').join('')+'</tr></thead><tbody>'+d.map(r=>'<tr>'+fields.map(f=>'<td>'+(typeof r[f]==='number'?(Number.isInteger(r[f])?r[f]:r[f].toPrecision(4)):(r[f]??''))+'</td>').join('')+'</tr>').join('')+'</tbody>'}
document.querySelectorAll('select').forEach(s=>s.addEventListener('change',update));update();document.getElementById('filters').hidden=false;</script>'''.replace('PAYLOAD',payload).replace('FIELDS',json.dumps(fields)).replace('TITLES',json.dumps(titles))
    style='''<style>
body{font:16px/1.5 system-ui,sans-serif;margin:0;background:#f5f6f5;color:#25313a}main{max-width:1150px;margin:auto;padding:35px 25px}h1{font-size:34px;line-height:1.15}h2{font-size:24px}a{color:#315f85}section{background:white;padding:22px;margin:28px 0;border-radius:8px}.callout{background:#e4ecef;padding:18px;border-left:4px solid #487ca5}.eyebrow{color:#60727a}.caption{font-size:13px}svg{display:block;width:100%;height:auto;max-width:1000px;margin:auto}table{border-collapse:collapse;font:12px/1.5 system-ui,sans-serif;width:100%}th,td{padding:8px;text-align:right;border-bottom:1px solid #ddd;white-space:nowrap}th{background:#eff2f3;position:sticky;top:0}td:first-child,th:first-child{text-align:left}.scroll{overflow:auto;max-height:650px}select{padding:6px;margin:5px}code{background:#eef1f2;padding:2px 5px}
@page{size:A4 landscape;margin:12mm}
@media print{body{font:10pt/1.35 Georgia,serif;background:white;color:#202930;-webkit-print-color-adjust:exact;print-color-adjust:exact}main{max-width:none;padding:0}h1{font-size:24pt;margin:0 0 10pt}h2{font-size:16pt;margin:0 0 8pt}p{margin:7pt 0}section{padding:0;margin:0;border-radius:0}.callout{padding:9pt}.eyebrow{font-size:9pt}.plot,.results,.provenance{break-before:page}.plot{break-inside:avoid}.plot svg{width:190mm;max-width:100%;height:auto}.caption{font-size:8.5pt;line-height:1.2;margin:5pt 0}.scroll{overflow:visible;max-height:none}table{font:8.5pt/1.2 Georgia,serif}th,td{padding:4pt;white-space:normal}th{position:static}thead{display:table-header-group}tr{break-inside:avoid}.results th,.results td{padding:3pt}.methods{margin-top:14pt;font-size:9pt}.methods h2{font-size:13pt}.downloads,#filters{display:none}a{color:#315f85;text-decoration:none}.provenance{font-size:11pt;line-height:1.45}}
</style>'''
    page='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Boyd backtracking comparison</title>'+style+'</head><body><main>'+intro+methods+''.join(sections)+explorer+provenance+'</main>'+script+'</body></html>'
    (out/'index.html').write_text(page)
    memo='# Boyd backtracking comparison\n\n'+takeaway+'\n\n'+md_table(overview)+'\n\n'
    memo+='Every iteration resets the trial step to 1 and halves failed trials. Armijo alpha=0.1 for logistic GD and its extrapolated-point FGM adaptation. LASSO uses proximal quadratic majorization. Parameters were fixed before the rerun.\n\n'
    memo+='Boyd and Vandenberghe, Algorithm 9.2: https://www.seas.ucla.edu/~vandenbe/cvxbook/bv_cvxbook.pdf#page=478\n\n'
    memo+='LASSO acceptance condition: Parikh and Boyd, Section 4.2, https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf#page=30 . We use a unit reset; the cited proximal algorithm carries the previous step. No accelerated convergence guarantee is claimed for FGM.\n\n'
    memo+='Della-stellato jobs: 14395605_0 (23 s), 14395605_1 (23 s), 14395605_2 (16 s). All 270 learned means and q10/q90 match the paper. Runner fa8a9e9; input results/linesearch-boyd-v1.\n\n'
    memo+='Files: index.html; report.pdf (webpage export); comparison_plots.pdf (four vector plots); all_results.csv (420 rows); decision.csv; paired_bootstrap.csv; figure_reproduction_checks.json; reproduction_bundle.zip.\n'
    (out/'DECISION.md').write_text(memo)
    (out/'comparison.md').write_text('# Boyd backtracking at 15 iterations\n\n'+md_table(display)+'\n')
    print(f'Wrote {out}/index.html; {len(frame)} result rows; all figure checks passed.')


if __name__=='__main__':
    main()
