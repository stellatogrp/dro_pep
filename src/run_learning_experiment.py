"""
SGDA Learning Experiment Runner.

Similar structure to run_dro_experiment.py:
- alg and sgda_type are selected via SLURM_ARRAY_TASK_ID for parallel jobs
- K_max loop is inside quad_run() with per-K CSV logging

Usage:
    Local:   python run_learning_experiment.py Quad local
    Cluster: python run_learning_experiment.py Quad cluster
"""
# from site import execsitecustomize
import _jax_setup  # noqa: F401  -- enables JAX persistent compilation cache
import hydra
import logging
import os
import sys

log = logging.getLogger(__name__)

from learning_experiment_classes.lasso import lasso_run
from learning_experiment_classes.logreg import logreg_run
from learning_experiment_classes.pdlp import pdlp_run
from learning_experiment_classes.quad import quad_run
from itertools import product


@hydra.main(version_base='1.2', config_path='configs_learning', config_name='lasso.yaml')
def lasso_driver(cfg):
    lasso_run(cfg)


@hydra.main(version_base='1.2', config_path='configs_learning', config_name='logreg.yaml')
def logreg_driver(cfg):
    logreg_run(cfg)


@hydra.main(version_base='1.2', config_path='configs_learning', config_name='pdlp.yaml')
def pdlp_driver(cfg):
    pdlp_run(cfg)


@hydra.main(version_base='1.2', config_path='configs_learning', config_name='quad.yaml')
def quad_driver(cfg):
    quad_run(cfg)


def cartesian_product(options):
    """
    Create cartesian product of option lists.
    
    Args:
        options: List of lists, where each inner list contains string options
                 e.g. [['alg=vanilla_gd', 'alg=nesterov_gd'], ['sgda_type=vanilla_sgda', 'sgda_type=adamw']]
    
    Returns:
        List of lists representing all combinations
        e.g. [['alg=vanilla_gd', 'sgda_type=vanilla_sgda'], ['alg=vanilla_gd', 'sgda_type=adamw'], ...]
    """
    return [list(combo) for combo in product(*options)]


def conditional_product(common_options, conditional_groups):
    """
    Create cartesian product with conditional dependencies.
    
    Args:
        common_options: List of lists - options included in ALL combinations
            e.g. [['stepsize_type=scalar', 'stepsize_type=vector']]
        
        conditional_groups: List of dicts, where each dict represents a group
            of parameters that must vary together. Each dict maps a "base" option
            to dependent options that only apply when that base is selected.
            e.g. [
                {
                    'mu=0': ['K_max=3', 'K_max=7', 'K_max=15', 'K_max=31'],
                    'mu=1': ['K_max=4', 'K_max=8', 'K_max=16', 'K_max=32'],
                }
            ]
    
    Returns:
        List of lists representing all valid combinations
    """
    # First, expand each conditional group into (base, dependent) pairs
    conditional_pairs = []
    for group in conditional_groups:
        pairs = []
        for base, dependents in group.items():
            for dep in dependents:
                pairs.append([base, dep])  # Each pair becomes one "option" in the product
        conditional_pairs.append(pairs)
    
    # Now cartesian product: common_options × flattened conditional pairs
    all_options = common_options + conditional_pairs
    
    results = []
    for combo in product(*all_options):
        # Flatten: some elements are strings (from common), some are lists (from conditional)
        flat = []
        for item in combo:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)
        results.append(flat)
    
    return results


# Define options for each parameter (each list contains all values for that parameter)
Quad_options = [
    ['learning_framework=ldro-pep'],
    ['alg=vanilla_gd'],
    ['pep_obj=obj_val'],
    # ['training_sample_N=100',
    #  'training_sample_N=200',
    #  'training_sample_N=300',
    #  'training_sample_N=400',
    #  'training_sample_N=500',
    #  'training_sample_N=600',
    #  'training_sample_N=700',
    #  'training_sample_N=800',
    #  'training_sample_N=900',
    #  'training_sample_N=1000'],
    ['training_sample_N=1000'],
    ['dro_obj=expectation'],
    # ['dro_obj=expectation', 'dro_obj=cvar'],
    # ['alpha=0.1'],
    ['stepsize_type=vector'],
    # ['vector_init=fixed', 'vector_init=silver'],
    # ONE eps, not a sweep: the paper figure consumes a single radius per
    # problem -- experiment_plots_icml/quad/data_scrape.py pins eps=1.0.
    # The hyperparameter search that find_best_stepsizes.py actually
    # resolves is over eta_t x weight_decay, which is kept below.
    ['eps=1.0'],
    # ['mu=1'],
    ['N=20'],
    ['sgd_iters=500'],
    ['eta_t=1e-2', 'eta_t=1e-1'],
    ['weight_decay=0', 'weight_decay=1e-5', 'weight_decay=1e-4'],
    # ['weight_decay=0'],
    # Both K blocks, so one submission covers the paper's full K = 1..15 range.
    # Splitting them is what keeps a single task's SDP chain a manageable size;
    # running only the second half (the previous default) silently halved the
    # sweep. 5 eps x 2 eta x 3 wd x 2 blocks = 60 tasks.
    # Three blocks, ~2.1 / 4.7 / 9.5 h per task (iclr_data_outputs/cost_model.py).
    # Split for parallelism, not for the wall limit -- these go to
    # sched_mit_sloan_batch, MaxTime 4-00:00:00.
    ['K_max=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]',
     'K_max=[11, 12, 13, 14, 15]',
     'K_max=[16, 17, 18, 19, 20]'],
]

# Parameter combinations for Slurm array jobs
# Uses conditional_product to tie mu and K_max values together
Learn_Quad_params = conditional_product(
    common_options=Quad_options,
    conditional_groups=[
        # {
        #     'mu=0': ['K_max=[3,7,15]', 'K_max=[31]'],
        #     'mu=1': ['K_max=[4,8,16]', 'K_max=[32]'],
        # },
        # {
        #     'stepsize_type=scalar': ['vector_init=fixed'],
        #     'stepsize_type=vector': ['vector_init=fixed', 'vector_init=silver'],
        # },
    ]
)

# DR-L2O LogReg sweep: 2 algs x 15 K = 30 tasks (array 0-29).
# itertools.product varies the LAST list fastest, so idx = alg_idx*15 + (K-1):
#   vanilla_gd   K=1..15 -> idx  0..14
#   nesterov_fgm K=1..15 -> idx 15..29
#
# ONE K per task, deliberately. The SDP layer leaks ~88 bytes per matrix
# nonzero per SGD step and never frees it, so peak memory is set by the TOTAL
# work a task does, not by its largest problem -- a task holding several K
# values pays the sum. (Measured: a Quad task running K=1..10 finished K=1..6
# and then died at 16G a hundred iterations into K=7, which alone needs ~4G.)
# Splitting one K per task is what keeps the per-task request tractable.
#
# Calibration for the --mem you need at sgd_iters=500. alg=nesterov_fgm K=15
# OOM'd at BOTH memory levels we ran -- 48G at iteration 476 (22976799_5) and
# 16G at iteration 82 (23331838_6) -- which are two exact points on
# mem = base + leak*iters, so the fit is measured rather than extrapolated:
#
#   leak = 86.1 bytes per matrix nonzero per SGD step
#   base = 9.3 GB, essentially independent of K (XLA + the dense A intermediate)
#   need(K, 500 iters) ~= 9.3 GB + 43.0 KB * nnz(K)
#   nnz(K) = 10K^4 + 100K^3 + 390K^2 + 780K + 640
#
# giving  K<=9: 17G   K=10: 20G   K=12: 29G   K=13: 34G   K=14: 42G   K=15: 50G.
# It back-predicts every 16G OOM we saw to within 6-16% on the conservative
# side. Note the 9.3 GB floor: even K=1 needs ~10G, so there is no such thing
# as a cheap task here.
#
# K=15 is the ceiling. The same law puts K=20 at ~120G, which does not fit
# under the per-user memory cap -- raising it needs the leak fixed, not a
# bigger --mem.
# See slurm_scripts/mit/run.sh for the tiered --array/--mem submission.
LogReg_options = [
    ['learning_framework=ldro-pep'],
    ['pep_obj=obj_val'],
    ['dro_obj=expectation'],
    ['N=20'],
    ['sgd_iters=500'],
    ['eta_t=1e-3'],
    ['alg=vanilla_gd', 'alg=nesterov_fgm'],
    # One eps, matching Quad. LogReg has no pinned value in the plotting
    # code yet (logreg/data_scrape.py constrains only alg), so this is a
    # choice -- restore the 5-value list for the eps-vs-robustness curve.
    ['eps=1.0'],
    [f'K_max=[{k}]' for k in range(1, 16)],
]

Learn_LogReg_params = conditional_product(
    common_options=LogReg_options,
    conditional_groups=[],
)

Lasso_options = [
    ['learning_framework=ldro-pep'],
    ['alg=ista'],
    ['N=10'],
    # ['training_sample_N=100',
    #  'training_sample_N=200',
    #  'training_sample_N=300',
    #  'training_sample_N=400',
    #  'training_sample_N=500',
    #  'training_sample_N=600',
    #  'training_sample_N=700',
    #  'training_sample_N=800',
    #  'training_sample_N=900',
    #  'training_sample_N=1000'],
    ['training_sample_N=1000'],
    ['dro_obj=expectation'],
    ['sgd_iters=500'],
    # ONE eps, not a sweep: the paper figure consumes a single radius per
    # problem -- experiment_plots_icml/lasso/data_scrape.py pins eps=10.0.
    # The hyperparameter search that find_best_stepsizes.py actually
    # resolves is over eta_t x weight_decay, which is kept below.
    ['eps=10.0'],
    # ['eta_t=1e-5', 'eta_t=1e-4', 'eta_t=1e-3'],
    ['eta_t=1e-4', 'eta_t=1e-3'],
    ['weight_decay=0', 'weight_decay=1e-5', 'weight_decay=1e-4'],
    # K blocks are sized so that no single task exceeds ~21 h at
    # sgd_iters=1000. Lasso's SDP is 4x Quad's at equal K (nnz leading
    # coefficient 40 vs 10) and its per-iteration cost grows as nnz^0.87,
    # so past K=15 each horizon gets its own task: K=20 alone is ~76 s/iter.
    # Lasso is the expensive one: its SDP is 4x Quad's at equal K and cost
    # grows as nnz^0.87 -- an exponent anchored by a direct K=20 measurement
    # (78.7 s/iter, which the model predicts to 0.4%), not extrapolated.
    # Past K=15 each horizon gets its own task: K=20 alone is ~22 h at
    # sgd_iters=1000, which needs the 4-day sched_mit_sloan_batch partition.
    ['K_max=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]',
     'K_max=[11, 12, 13]',
     'K_max=[14, 15]',
     'K_max=[16]',
     'K_max=[17]',
     'K_max=[18]',
     'K_max=[19]',
     'K_max=[20]'],
]

Learn_Lasso_params = conditional_product(
    common_options=Lasso_options,
    conditional_groups=[
        # {
        #     'm=300': ['n=200'],
        #     'm=200': ['n=300'],
        # },
        # {
        #     'stepsize_type=scalar': ['vector_init=fixed'],
        #     'stepsize_type=vector': ['vector_init=fixed'],
        # },
    ]
)

PDLP_options = [
    ['N=5'],
    # ['dro_obj=expectation', 'dro_obj=cvar'],
    # ['alpha=0.1'],
    # 10 iterations was a smoke-test budget, not a training budget -- the
    # archived runs reach their best training loss as late as iteration ~900,
    # so 10 never left the initialization. PDLP cannot join the other three at
    # 500: one SGD step is ~57 s at K=10 and ~150 s at K=15, so even its
    # cheapest block would need 23 h against the default partition's 12 h cap.
    ['sgd_iters=100'],
    ['eta_t=1e-4', 'eta_t=1e-3'],
    ['eps=1.0', 'eps=10.0'],
    # Capped at 12, NOT 15. Measured at sgd_iters=1: K=12 solves (86.9 s/iter,
    # loss 23.62) but K=14 and K=15 both come back
    # "[SparseFwd] Solver status: Failure". K=14 also takes 546.7 s/iter
    # against a modelled 121 s -- the extra time IS the failure, spent in
    # Clarabel's reduced-tolerance retries. A failed forward solve still
    # returns a number (K=14 reported loss 19.51), so this would have produced
    # plausible-looking garbage rather than an obvious crash.
    # K=13 is untested; it is the only horizon that could still be recovered.
    ['K_max=[8, 9, 10, 11]',
     'K_max=[12]'],
]

Learn_PDLP_params = conditional_product(
    common_options=PDLP_options,
    conditional_groups=[],
)

func_driver_map = {
    'Quad': quad_driver,
    'Lasso': lasso_driver,
    'LogReg': logreg_driver,
    'PDLP': pdlp_driver,
}

base_dir_map = {
    'Quad': 'learn_dro_outputs/Quad',
    'Lasso': 'learn_dro_outputs/Lasso',
    'LogReg': 'learn_dro_outputs/LogReg',
    'PDLP': 'learn_dro_outputs/PDLP'
}


def main():
    print('len of Learn_Quad_params:', len(Learn_Quad_params))
    print('len of Learn_Lasso_params:', len(Learn_Lasso_params))
    print('len of Learn_PDLP_params:', len(Learn_PDLP_params))
    # exit(0)
    if len(sys.argv) < 3:
        print('Usage: python run_learning_experiment.py <experiment> <cluster|local>')
        print('  experiment: Quad')
        print('  target: cluster or local')
        exit(0)

    experiment = sys.argv[1]
    target_machine = sys.argv[2]

    if experiment not in base_dir_map:
        print(f'experiment name "{experiment}" invalid. Valid options: {list(base_dir_map.keys())}')
        exit(0)

    if target_machine == 'cluster':
        base_dir = os.environ.get(
            'DRO_PEP_LEARN_OUT', '/scratch/gpfs/BSTELLATO/vranjan/learn_dro_pep_out'
        )
    elif target_machine == 'local':
        base_dir = os.environ.get('DRO_PEP_LEARN_OUT', '.')
    else:
        print('specify cluster or local')
        exit(0)

    base_dir = f'{base_dir}/{base_dir_map[experiment]}'
    driver = func_driver_map[experiment]

    # The param table is selected by SLURM_ARRAY_TASK_ID alone, not by the
    # target: that is what lets a local run reproduce one cluster task exactly
    # (slurm_scripts/mit/run.sh --local --array N sets it).
    if "SLURM_ARRAY_TASK_ID" not in os.environ:
        # Local run: use defaults from config
        hydra_tags = [
            f'hydra.run.dir={base_dir}/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}',
            'hydra.job.chdir=True'
        ]
    else:
        # Slurm array job: select (alg, sgda_type) based on job index
        job_idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
        log.info(f'SLURM job index: {job_idx}')
        
        hydra_tags = [
            f'hydra.run.dir={base_dir}/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}_{job_idx}',
            'hydra.job.chdir=True'
        ]

        if experiment == 'Quad':
            if job_idx >= len(Learn_Quad_params):
                log.error(f'job_idx {job_idx} >= len(Learn_Quad_params) {len(Learn_Quad_params)}')
                exit(1)
            hydra_tags += Learn_Quad_params[job_idx]
        if experiment == 'Lasso':
            if job_idx >= len(Learn_Lasso_params):
                log.error(f'job_idx {job_idx} >= len(Learn_Lasso_params) {len(Learn_Lasso_params)}')
                exit(1)
            hydra_tags += Learn_Lasso_params[job_idx]
        if experiment == 'LogReg':
            if job_idx >= len(Learn_LogReg_params):
                log.error(f'job_idx {job_idx} >= len(Learn_LogReg_params) {len(Learn_LogReg_params)}')
                exit(1)
            hydra_tags += Learn_LogReg_params[job_idx]
        if experiment == 'PDLP':
            if job_idx >= len(Learn_PDLP_params):
                log.error(f'job_idx {job_idx} >= len(Learn_PDLP_params) {len(Learn_PDLP_params)}')
                exit(1)
            hydra_tags += Learn_PDLP_params[job_idx]

    # Preserve user-supplied hydra overrides (args after experiment & target_machine)
    extra_args = sys.argv[3:]
    sys.argv = [sys.argv[0]] + hydra_tags + extra_args
    driver()


if __name__ == '__main__':
    main()
