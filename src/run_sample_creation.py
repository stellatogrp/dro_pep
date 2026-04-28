"""
Sample Creation Runner.

Generates training, validation, test, and out-of-distribution problem
instances in a unified on-disk format. Lasso writes:

    A_in_dist.npz                  # key: A
    A_out_of_dist.npz              # key: A
    training_set.npz               # keys: b_batch, x_opt_batch, f_opt_batch
    validation_set.npz             # keys: b_batch, x_opt_batch, f_opt_batch
    test_set.npz                   # keys: b_batch, x_opt_batch, f_opt_batch
    ood_set.npz                    # keys: b_batch, x_opt_batch, f_opt_batch
    out_of_sample_metadata.npz

Usage:
    Local:   python run_sample_creation.py Lasso local
"""
import hydra
import logging
import os
import sys

log = logging.getLogger(__name__)

from learning_experiment_classes.lasso import lasso_sample_creation_run as lasso_run
# from learning_experiment_classes.logreg import logreg_out_of_sample_run as logreg_run
from learning_experiment_classes.quad import quad_sample_creation_run as quad_run
from learning_experiment_classes.pdlp import pdlp_sample_creation_run as pdlp_run


@hydra.main(version_base='1.2', config_path='configs_learning', config_name='quad.yaml')
def quad_driver(cfg):
    quad_run(cfg)


@hydra.main(version_base='1.2', config_path='configs_learning', config_name='lasso.yaml')
def lasso_driver(cfg):
    lasso_run(cfg)


@hydra.main(version_base='1.2', config_path='configs_learning', config_name='logreg.yaml')
def logreg_driver(cfg):
    logreg_run(cfg)


@hydra.main(version_base='1.2', config_path='configs_learning', config_name='pdlp.yaml')
def pdlp_driver(cfg):
    pdlp_run(cfg)


func_driver_map = {
    'Lasso': lasso_driver,
    'LogReg': logreg_driver,
    'Quad': quad_driver,
    'PDLP': pdlp_driver,
}

base_dir_map = {
    'Lasso': 'sample_creation_outputs/Lasso',
    'LogReg': 'out_of_sample_outputs/LogReg',
    'Quad': 'sample_creation_outputs/Quad',
    'PDLP': 'sample_creation_outputs/PDLP',
}


def main():
    if len(sys.argv) < 3:
        print('Usage: python run_sample_creation.py <experiment> <cluster|local>')
        print('  experiment: Lasso, LogReg, Quad, or PDLP')
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

    hydra_tags = [
        f'hydra.run.dir={base_dir}/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}',
        'hydra.job.chdir=True'
    ]

    sys.argv = [sys.argv[0]] + hydra_tags
    driver()


if __name__ == '__main__':
    main()
