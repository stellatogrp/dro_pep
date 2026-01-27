"""
Out-of-Sample Test Set Generation Runner.

Generates and saves out-of-sample test problems (Q matrices and z0 vectors)
for consistent evaluation across different algorithms.

Usage:
    Local:   python run_out_of_sample.py Quad local
"""
import hydra
import logging
import os
import sys

log = logging.getLogger(__name__)

from learning_experiment_classes.lasso import lasso_out_of_sample_run as lasso_run
from learning_experiment_classes.quad import quad_out_of_sample_run as quad_run


@hydra.main(version_base='1.2', config_path='configs_learning', config_name='quad.yaml')
def quad_driver(cfg):
    quad_run(cfg)


@hydra.main(version_base='1.2', config_path='configs_learning', config_name='lasso.yaml')
def lasso_driver(cfg):
    lasso_run(cfg)


func_driver_map = {
    'Lasso': lasso_driver,
    'Quad': quad_driver,
}

base_dir_map = {
    'Lasso': 'out_of_sample_outputs/Lasso',
    'Quad': 'out_of_sample_outputs/Quad',
}


def main():
    if len(sys.argv) < 3:
        print('Usage: python run_out_of_sample.py <experiment> <cluster|local>')
        print('  experiment: Quad')
        print('  target: cluster or local')
        exit(0)

    experiment = sys.argv[1]
    target_machine = sys.argv[2]

    if experiment not in base_dir_map:
        print(f'experiment name "{experiment}" invalid. Valid options: {list(base_dir_map.keys())}')
        exit(0)

    if target_machine == 'cluster':
        base_dir = '/scratch/gpfs/BSTELLATO/vranjan/out_of_sample_out'
    elif target_machine == 'local':
        base_dir = '.'
    else:
        print('specify cluster or local')
        exit(0)

    base_dir = f'{base_dir}/{base_dir_map[experiment]}'
    driver = func_driver_map[experiment]

    # Out-of-sample generation: simple run without SLURM complexity
    hydra_tags = [
        f'hydra.run.dir={base_dir}/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}',
        'hydra.job.chdir=True'
    ]

    sys.argv = [sys.argv[0]] + hydra_tags
    driver()


if __name__ == '__main__':
    main()

