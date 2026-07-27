#!/bin/bash
#SBATCH --job-name=LogRegDRO
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=24G
#SBATCH --time=00-05:59:59
#SBATCH --array=0-29
#SBATCH -o /scratch/gpfs/BSTELLATO/bs37/learn_dro_pep_out/LogReg/runs/%A_%a.txt
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=bs37@princeton.edu

# DR-L2O LogReg sweep: 2 algs (vanilla_gd, nesterov_fgm) x 5 eps x 3 K = 30 tasks.
# Task-index mapping lives in run_learning_experiment.py (Learn_LogReg_params).

module purge
module load intel-mkl/2024.2

REPO=/scratch/gpfs/BSTELLATO/bs37/projects/dr-l2o/dr-l2o_repo
source "$REPO/.venv/bin/activate"
export DRO_PEP_LEARN_OUT=/scratch/gpfs/BSTELLATO/bs37/learn_dro_pep_out

cd "$REPO/src"
python run_learning_experiment.py LogReg cluster
