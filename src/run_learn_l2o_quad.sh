#!/bin/bash
#SBATCH --job-name=Quad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=24G
# #SBATCH --constraint=intel
#SBATCH --time=00-00:59:59
#SBATCH --array=0-2
#SBATCH -o /scratch/gpfs/BSTELLATO/vranjan/learn_dro_pep_out/Quad/runs/%A.txt
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=vranjan@princeton.edu
# #SBATCH --gres=gpu:1

# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='true'
# export XLA_PYTHON_CLIENT_MEM_FRACTION='0.30'
# export XLA_PYTHON_CLIENT_ALLOCATOR=platform
# export xla_force_host_platform_device_count=1

module purge
module load intel-mkl/2024.2
module load anaconda3/2025.12
# module load anaconda3/2023.9 cudnn/cuda-11.x/8.2.0 cudatoolkit/11.3 nvhpc/21.5
conda activate algover

python run_learning_l2o_experiment.py Quad cluster
