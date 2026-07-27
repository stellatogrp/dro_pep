#!/bin/bash
#SBATCH --job-name=LogRegLPEP
#SBATCH --account=bstellato
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00-03:59:59
#SBATCH --array=0-17
#SBATCH -o /scratch/gpfs/BSTELLATO/bs37/learn_dro_pep_out/LogReg/runs/%A_%a.txt
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=bs37@princeton.edu

# LPEP (OPT-PEP) LogReg sweep: 2 algs x 3 etas x 3 K = 18 tasks.
# Task-index mapping lives in run_learning_lpep_experiment.py.

# Module system is unavailable in non-login job shells; export the Intel
# runtime paths directly (MKL Pardiso needs libiomp5 from the compiler dir,
# else clarabel dies with a symbol lookup error).
module purge 2>/dev/null || true
module load intel-mkl/2024.2 2>/dev/null || true
export LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/latest/lib:/opt/intel/oneapi/compiler/latest/lib:$LD_LIBRARY_PATH

REPO=/scratch/gpfs/BSTELLATO/bs37/projects/dr-l2o/dr-l2o_repo
source "$REPO/.venv/bin/activate"
export DRO_PEP_LEARN_OUT=/scratch/gpfs/BSTELLATO/bs37/learn_dro_pep_out

cd "$REPO/src"
python run_learning_lpep_experiment.py LogReg cluster
