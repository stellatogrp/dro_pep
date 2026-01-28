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
    ['dro_obj=expectation'],
    ['eps=0.01', 'eps=0.1', 'eps=1.0', 'eps=5.0', 'eps=10.0'],
    ['mu=1'],
    ['N=20'],
    ['sgd_iters=500'],
    ['K_max=[5]', 'K_max=[10]', 'K_max=[15]'],
]

LogReg_options = [

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

Lasso_options = [
    ['alg=ista'],
    ['N=20'],
    ['dro_obj=expectation', 'dro_obj=cvar'],
    ['alpha=0.1'],
    ['sgd_iters=500'],
    # ['eps=0.01', 'eps=0.1', 'eps=1.0', 'eps=5.0', 'eps=10.0'],
    ['K_max=[5]', 'K_max=[10]', 'K_max=[15]'],
    ['learning_framework=l2o'],
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
    ['N=20'],
    ['dro_obj=expectation', 'dro_obj=cvar'],
    ['alpha=0.1'],
    ['sgd_iters=200'],
    ['eps=0.1', 'eps=1.0', 'eps=10.0'],
    ['K_max=[5]', 'K_max=[10]'],
]

Learn_PDLP_params = conditional_product(
    common_options=PDLP_options,
    conditional_groups=[],
)

LogReg_options = [
    ['alg=vanilla_gd'],
    ['pep_obj=obj_val'],
    ['dro_obj=expectation', 'dro_obj=cvar'],
    ['alpha=0.1'],
    ['eps=0.01', 'eps=0.1', 'eps=1.0', 'eps=5.0', 'eps=10.0'],
    ['N=20'],
    ['sgd_iters=500'],
    ['K_max=[5]', 'K_max=[10]', 'K_max=[15]'],
]

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
        base_dir = '/scratch/gpfs/BSTELLATO/vranjan/learn_dro_pep_out'
    elif target_machine == 'local':
        base_dir = '.'
    else:
        print('specify cluster or local')
        exit(0)

    base_dir = f'{base_dir}/{base_dir_map[experiment]}'
    driver = func_driver_map[experiment]

    if target_machine == 'local' or "SLURM_ARRAY_TASK_ID" not in os.environ:
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
        if experiment == 'PDLP':
            if job_idx >= len(Learn_PDLP_params):
                log.error(f'job_idx {job_idx} >= len(Learn_PDLP_params) {len(Learn_PDLP_params)}')
                exit(1)
            hydra_tags += Learn_PDLP_params[job_idx]

    sys.argv = [sys.argv[0]] + hydra_tags
    driver()


if __name__ == '__main__':
    main()
