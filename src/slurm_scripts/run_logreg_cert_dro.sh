#!/bin/bash
#SBATCH --job-name=LogRegCertDRO
#SBATCH --account=bstellato
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00-05:59:59
#SBATCH --array=0-3
#SBATCH -o /scratch/gpfs/BSTELLATO/bs37/cert_dro_pep_out/LogReg/runs/%A_%a.txt
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=bs37@princeton.edu

# LogReg certification DRO stage: 4 tasks (gd,fgm) x (expectation,cvar).
# SDP sweep over eps x alpha x K; memory grows with K and training N.

module purge 2>/dev/null || true
module load intel-mkl/2024.2 2>/dev/null || true
export LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/latest/lib:/opt/intel/oneapi/compiler/latest/lib:$LD_LIBRARY_PATH

REPO=/scratch/gpfs/BSTELLATO/bs37/projects/dro_pep_cert
source "$REPO/.venv/bin/activate"
export DRO_PEP_CERT_OUT=/scratch/gpfs/BSTELLATO/bs37/cert_dro_pep_out
export DRO_PEP_DATA=/scratch/gpfs/BSTELLATO/bs37/dro_pep_data

cd "$REPO/src"
python run_dro_experiment.py LogReg cluster
