#!/bin/bash
#SBATCH --job-name=LDRO
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=80G
#SBATCH --time=02-23:59:59
#SBATCH --array=0-3
#SBATCH -o /scratch/gpfs/vranjan/dro_pep_out/Lasso/runs/%A.txt
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=vranjan@princeton.edu
# #SBATCH --gres=gpu:1

# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='true'
# export XLA_PYTHON_CLIENT_MEM_FRACTION='0.30'
# export XLA_PYTHON_CLIENT_ALLOCATOR=platform
# export xla_force_host_platform_device_count=1

module purge
module load anaconda3/2024.10
# module load anaconda3/2023.9 cudnn/cuda-11.x/8.2.0 cudatoolkit/11.3 nvhpc/21.5
conda activate algover

python run_dro_experiment.py Lasso cluster
