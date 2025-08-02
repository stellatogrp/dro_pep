import hydra
import logging
import os
import sys

log = logging.getLogger(__name__)

from experiment_classes.huber import huber_samples
from experiment_classes.lasso import lasso_samples
from experiment_classes.logreg import logreg_samples
from experiment_classes.quad import quad_samples
from experiment_classes.simple_lstsq import lstsq_samples


@hydra.main(version_base='1.2', config_path='configs', config_name='quad.yaml')
def quad_driver(cfg):
    quad_samples(cfg)

@hydra.main(version_base='1.2', config_path='configs', config_name='huber.yaml')
def huber_driver(cfg):
    huber_samples(cfg)

@hydra.main(version_base='1.2', config_path='configs', config_name='lasso.yaml')
def lasso_driver(cfg):
    lasso_samples(cfg)

@hydra.main(version_base='1.2', config_path='configs', config_name='logreg.yaml')
def logreg_driver(cfg):
    logreg_samples(cfg)

@hydra.main(version_base='1.2', config_path='configs', config_name='simple_lstsq.yaml')
def simple_lstsq_driver(cfg):
    lstsq_samples(cfg)


Quad_params = [
]

Huber_params = [
]

Lasso_params = [
]

LogReg_params = [
]

func_driver_map = {
    'Huber': huber_driver,
    'Lasso': lasso_driver,
    'LogReg': logreg_driver,
    'Quad': quad_driver,
    'SimpleLstsq': simple_lstsq_driver,
}

base_dir_map = {
    'Huber': 'sample_outputs/Huber',
    'Quad': 'sample_outputs/Quad',
    'SimpleLstsq': 'sample_outputs/SimpleLstsq',
    'Lasso': 'sample_outputs/Lasso',
    'LogReg': 'sample_outputs/LogReg',
}


def main():
    if len(sys.argv) < 3:
        print('not enough command line arguments')
        exit(0)
    if sys.argv[2] == 'cluster':
        # raise NotImplementedError
        base_dir = '/scratch/gpfs/vranjan/dro_pep_out'
    elif sys.argv[2] == 'local':
        base_dir = '.'
    else:
        print('specify cluster or local')
        exit(0)

    experiment = sys.argv[1]
    target_machine = sys.argv[2]

    if sys.argv[1] not in base_dir_map:
        print(f'experiment name "{sys.argv[1]}" invalid')
        exit(0)

    base_dir = f'{base_dir}/{base_dir_map[sys.argv[1]]}'
    driver = func_driver_map[sys.argv[1]]

    if target_machine == 'local' or "SLURM_ARRAY_TASK_ID" not in os.environ:
        hydra_tags = [f'hydra.run.dir={base_dir}/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}', 'hydra.job.chdir=True']
    else:
        job_idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
        log.info(f'job id: {job_idx}')
        hydra_tags = [f'hydra.run.dir={base_dir}/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}_{job_idx}', 'hydra.job.chdir=True']

        if experiment == 'Quad':
            hydra_tags += Quad_params[job_idx]

        if experiment == 'Huber':
            hydra_tags += Huber_params[job_idx]

        if experiment == 'Lasso':
            hydra_tags += Lasso_params[job_idx]

        if experiment == 'LogReg':
            hydra_tags += LogReg_params[job_idx]

    sys.argv = [sys.argv[0]] + hydra_tags

    driver()


if __name__ == '__main__':
    main()
