#!/bin/bash
#SBATCH --job-name=LogRegData
#SBATCH --account=bstellato
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00-01:59:59
#SBATCH -o /scratch/gpfs/BSTELLATO/bs37/learn_dro_pep_out/LogReg/runs/%A_data.txt
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=bs37@princeton.edu

# LogReg sample creation: training/validation/test/ood sets (1000/250/250/250),
# each instance solved with CVXPY/CLARABEL. Output goes under
# $DRO_PEP_SAMPLE_OUT/sample_creation_outputs/LogReg/<date>/<time>/.

module purge
module load intel-mkl/2024.2

REPO=/scratch/gpfs/BSTELLATO/bs37/projects/dr-l2o/dr-l2o_repo
source "$REPO/.venv/bin/activate"
export DRO_PEP_SAMPLE_OUT=/scratch/gpfs/BSTELLATO/bs37/out_of_sample_out

cd "$REPO/src"
python run_sample_creation.py LogReg cluster
