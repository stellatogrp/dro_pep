"""
SGDA Learning Experiment Runner.

Similar structure to run_dro_experiment.py:
- alg and sgda_type are selected via SLURM_ARRAY_TASK_ID for parallel jobs
- K_max loop is inside quad_run() with per-K CSV logging

Usage:
    Local:   python run_learning_experiment.py Quad local
    Cluster: python run_learning_experiment.py Quad cluster
"""
import hydra
import logging
import os
import sys

log = logging.getLogger(__name__)

from learning_experiment_classes.quad import quad_run
from itertools import product


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


# Define options for each parameter (each list contains all values for that parameter)
Quad_options = [
    ['alg=vanilla_gd', 'alg=nesterov_gd'],
    ['sgda_type=vanilla_sgda', 'sgda_type=adamw'],
    ['stepsize_type=scalar', 'stepsize_type=vector'],
]

# Parameter combinations for Slurm array jobs (cartesian product of all options)
Learn_Quad_params = cartesian_product(Quad_options)

func_driver_map = {
    'Quad': quad_driver,
}

base_dir_map = {
    'Quad': 'learn_dro_outputs/Quad',
}


def main():
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

    sys.argv = [sys.argv[0]] + hydra_tags
    driver()


if __name__ == '__main__':
    main()
