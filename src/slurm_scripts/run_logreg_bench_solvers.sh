#!/bin/bash
#SBATCH --job-name=LogRegSolverBench
#SBATCH --account=bstellato
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3G
#SBATCH --time=00-05:59:59
#SBATCH -o /scratch/gpfs/BSTELLATO/bs37/cert_dro_pep_out/LogReg/runs/%j_bench.txt
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=bs37@princeton.edu

# Solver benchmark on a single DRO subproblem. Pass BMODE (clarabel |
# cvxpy-clarabel | cvxpy-mosek) and optionally BALG, BK, BMEASURE, BETA
# via --export. One mode per job so sacct MaxRSS is attributable.

module purge 2>/dev/null || true
module load intel-mkl/2024.2 2>/dev/null || true
export LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/latest/lib:/opt/intel/oneapi/compiler/latest/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

REPO=/scratch/gpfs/BSTELLATO/bs37/projects/dro_pep_cert
source "$REPO/.venv/bin/activate"
export DRO_PEP_DATA=/scratch/gpfs/BSTELLATO/bs37/dro_pep_data

BMODE=${BMODE:?set BMODE via --export}
BALG=${BALG:-nesterov_fgm}
BK=${BK:-24}
BMEASURE=${BMEASURE:-cvar}

OUT=/scratch/gpfs/BSTELLATO/bs37/cert_dro_pep_out/bench_outputs
mkdir -p "$OUT"

cd "$REPO/src"
ETA_FLAG=""
if [ -n "$BETA" ]; then ETA_FLAG="--eta $BETA"; fi
python bench_dro_solver.py --mode "$BMODE" --alg "$BALG" --K "$BK" \
    --measure "$BMEASURE" --eps 1e-4 --alpha 0.01 $ETA_FLAG \
    --out "$OUT/${SLURM_JOB_ID}_${BMODE}_${BALG}_K${BK}_${BMEASURE}.json"
