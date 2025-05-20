import hydra
import logging
import os
import sys

log = logging.getLogger(__name__)

# import train_model as train


# @hydra.main(version_base='1.2', config_path='.', config_name='train.yaml')
# def driver(cfg):
#     train.train(cfg)

@hydra.main(version_base='1.2', config_path='configs', config_name='qgd.yaml')
def qgd_driver(cfg):
    pass


QGD_params = [
]

func_driver_map = {
    'QGD': qgd_driver,
}

base_dir_map = {
    'QGD': 'QGD/outputs'
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

        if experiment == 'QGD':
            hydra_tags += QGD_params[job_idx]

    sys.argv = [sys.argv[0]] + hydra_tags

    driver()


if __name__ == '__main__':
    main()
