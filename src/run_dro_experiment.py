import hydra
import logging
import os
import sys

log = logging.getLogger(__name__)

from experiment_classes.huber import huber_dro
from experiment_classes.lasso import lasso_dro
from experiment_classes.logreg import logreg_dro
from experiment_classes.quad import quad_dro
from experiment_classes.simple_quad import simple_quad_dro


@hydra.main(version_base='1.2', config_path='configs', config_name='lasso.yaml')
def lasso_driver(cfg):
    lasso_dro(cfg)


@hydra.main(version_base='1.2', config_path='configs', config_name='logreg.yaml')
def logreg_driver(cfg):
    logreg_dro(cfg)


@hydra.main(version_base='1.2', config_path='configs', config_name='quad.yaml')
def quad_driver(cfg):
    quad_dro(cfg)


@hydra.main(version_base='1.2', config_path='configs', config_name='simple_quad.yaml')
def simple_quad_driver(cfg):
    simple_quad_dro(cfg)


@hydra.main(version_base='1.2', config_path='configs', config_name='huber.yaml')
def huber_driver(cfg):
    huber_dro(cfg)


# Quad_params = [
#     ['alg=grad_desc', 'dro_obj=expectation', 'dro_pep_obj=obj_val'],
#     ['alg=nesterov_grad_desc', 'dro_obj=expectation', 'dro_pep_obj=obj_val'],
#     ['alg=grad_desc', 'dro_obj=cvar', 'dro_pep_obj=obj_val'],
#     ['alg=nesterov_grad_desc', 'dro_obj=cvar', 'dro_pep_obj=obj_val'],
#     ['alg=grad_desc', 'dro_obj=expectation', 'dro_pep_obj=grad_sq_norm'],
#     ['alg=nesterov_grad_desc', 'dro_obj=expectation', 'dro_pep_obj=grad_sq_norm'],
#     ['alg=grad_desc', 'dro_obj=cvar', 'dro_pep_obj=grad_sq_norm'],
#     ['alg=nesterov_grad_desc', 'dro_obj=cvar', 'dro_pep_obj=grad_sq_norm'],
# ]

Quad_params = [
    ['alg=grad_desc', 'dro_obj=expectation', 'mu=0'],
    ['alg=nesterov_grad_desc', 'dro_obj=expectation', 'mu=0'],
    ['alg=grad_desc', 'dro_obj=cvar', 'mu=0'],
    ['alg=nesterov_grad_desc', 'dro_obj=cvar', 'mu=0'],
    ['alg=grad_desc', 'dro_obj=expectation', 'mu=1'],
    ['alg=nesterov_grad_desc', 'dro_obj=expectation', 'mu=1'],
    ['alg=grad_desc', 'dro_obj=cvar', 'mu=1'],
    ['alg=nesterov_grad_desc', 'dro_obj=cvar', 'mu=1'],
]

Huber_params = [
    ['alg=grad_desc', 'dro_obj=expectation'],
    ['alg=nesterov_grad_desc', 'dro_obj=expectation'],
    ['alg=grad_desc', 'dro_obj=cvar'],
    ['alg=nesterov_grad_desc', 'dro_obj=cvar'],
]

Lasso_params = [
    ['alg=ista', 'dro_obj=expectation'],
    ['alg=ista', 'dro_obj=cvar'],
    ['alg=fista', 'dro_obj=expectation'],
    ['alg=fista', 'dro_obj=cvar'],
    ['alg=optista', 'dro_obj=expectation'],
    ['alg=optista', 'dro_obj=cvar'],
]

LogReg_params = [
    ['alg=grad_desc', 'dro_obj=expectation'],
    ['alg=nesterov_grad_desc', 'dro_obj=expectation'],
    ['alg=grad_desc', 'dro_obj=cvar'],
    ['alg=nesterov_grad_desc', 'dro_obj=cvar'],
]

func_driver_map = {
    'Huber': huber_driver,
    'Quad': quad_driver,
    'SimpleQuad': simple_quad_driver,
    'Lasso': lasso_driver,
    'LogReg': logreg_driver,
}

base_dir_map = {
    'Huber': 'dro_outputs/Huber',
    'Quad': 'dro_outputs/Quad',
    'SimpleQuad': 'dro_outputs/SimpleQuad',
    'Lasso': 'dro_outputs/Lasso',
    'LogReg': 'dro_outputs/LogReg',
}


def main():
    if len(sys.argv) < 3:
        print('not enough command line arguments')
        exit(0)
    if sys.argv[2] == 'cluster':
        # raise NotImplementedError
        base_dir = '/scratch/gpfs/BSTELLATO/vranjan/dro_pep_out'
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
