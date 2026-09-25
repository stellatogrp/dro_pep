#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=dr-l2o-linesearch
#SBATCH --output=slurm/logs/%x-%j.out
source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
EXPERIMENT="${EXPERIMENT:-linesearch-smoke}"
OUT_DIR="$PROJECT_DIR/${RESULTS_DIR:-results}/$EXPERIMENT"
mkdir -p "$OUT_DIR"
"$PY" src/tools/run_linesearch_audit.py --problem "${PROBLEM:-logreg_fgm}" --mode "${MODE:-smoke}" --results-dir "$OUT_DIR"
