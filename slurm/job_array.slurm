#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=dr-l2o-linesearch
#SBATCH --output=slurm/logs/%x-%A_%a.out
source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
EXPERIMENT="${EXPERIMENT:-linesearch-full}"
OUT_DIR="$PROJECT_DIR/${RESULTS_DIR:-results}/$EXPERIMENT"
PROBLEMS=(lasso logreg_gd logreg_fgm)
"$PY" src/tools/run_linesearch_audit.py --problem "${PROBLEMS[$SLURM_ARRAY_TASK_ID]}" --mode "${MODE:-full}" --results-dir "$OUT_DIR"
