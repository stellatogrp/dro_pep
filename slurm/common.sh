# Shared environment for every job script. Sourced from $SLURM_SUBMIT_DIR/slurm/common.sh.
# Requires slurm/cluster.env next to it.
set -euo pipefail

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SLURM_DIR/cluster.env"

PROJECT_DIR="${REMOTE_PROJECT_DIR}"   # $USER expands here, on the cluster
cd "$PROJECT_DIR"

# The same entrypoint as on your laptop: the project venv created by env_setup.sh.
PY="$PROJECT_DIR/.venv/bin/python"

# One BLAS/OpenMP thread per process unless the job says otherwise; parallelism comes from
# joblib/array tasks, and oversubscription is the most common cause of slow cluster runs.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

mkdir -p "$PROJECT_DIR/${RESULTS_DIR:-results}" "$PROJECT_DIR/slurm/logs"
echo "[common.sh] host=$(hostname) job=${SLURM_JOB_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-?} python=$PY"
