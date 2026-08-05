#!/bin/bash
#SBATCH --job-name=LogRegDROchunk
#SBATCH --account=bstellato
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00-05:59:59
#SBATCH -o /scratch/gpfs/BSTELLATO/bs37/cert_dro_pep_out/LogReg/runs/%j_chunk.txt
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=bs37@princeton.edu

# Generic per-K-range DRO runner (supersedes run_logreg_cert_dro_catchup.sh).
# Sparse assembly keeps single-K memory at ~2-3GB, so jobs this small
# backfill within minutes. Pass via --export:
#   CALG      grad_desc | nesterov_fgm
#   CETA      step-size multiplier (1 or 1.9)
#   CMEASURE  expectation | cvar
#   CKMIN, CKMAX  K range
#   CTRIM     1 -> high-K trim: eps 1e-5..1e-3 (7 pts), alphas {0.01, 0.05}
# High-K cvar singles (K >= 36) need --time=11:59:59 at submit.

module purge 2>/dev/null || true
module load intel-mkl/2024.2 2>/dev/null || true
export LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/latest/lib:/opt/intel/oneapi/compiler/latest/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

REPO=/scratch/gpfs/BSTELLATO/bs37/projects/dro_pep_cert
source "$REPO/.venv/bin/activate"
export DRO_PEP_DATA=/scratch/gpfs/BSTELLATO/bs37/dro_pep_data

CALG=${CALG:?} ; CETA=${CETA:?} ; CMEASURE=${CMEASURE:?}
CKMIN=${CKMIN:?} ; CKMAX=${CKMAX:?} ; CTRIM=${CTRIM:-0}
export CALG CETA CMEASURE CKMIN CKMAX CTRIM

cd "$REPO/src"
python - <<'PY'
import os, sys
sys.path.insert(0, '.')
from hydra import initialize_config_dir, compose
from experiment_classes.logreg import logreg_dro

alg, eta, measure = os.environ['CALG'], os.environ['CETA'], os.environ['CMEASURE']
kmin, kmax, trim = os.environ['CKMIN'], os.environ['CKMAX'], os.environ['CTRIM']
overrides = [f'alg={alg}', f'eta={eta}', f'dro_obj={measure}',
             f'K_min={kmin}', f'K_max={kmax}']
suffix = ''
if trim == '1':
    overrides += ['eps.log_max=-3', 'eps.space_count=7',
                  'alpha_vals=[0.01,0.05]']
    suffix = '_trim'
elif trim == '2':   # deepest trim for the largest K: figure alpha only
    overrides += ['eps.log_max=-3', 'eps.space_count=7',
                  'alpha_vals=[0.01]']
    suffix = '_trim2'
cfg_dir = os.path.abspath('configs')
suffix += os.environ.get('CSUFFIX', '')
out = (f"/scratch/gpfs/BSTELLATO/bs37/cert_dro_pep_out/dro_outputs/LogReg/"
       f"chunk_{alg}_eta{eta}_{measure}_K{kmin}_{kmax}{suffix}")
os.makedirs(out, exist_ok=True)
with initialize_config_dir(version_base='1.2', config_dir=cfg_dir):
    cfg = compose(config_name='logreg', overrides=overrides)
os.chdir(out)
logreg_dro(cfg)
print('chunk done', alg, eta, measure, kmin, kmax, 'trim' if trim == '1' else '')
PY
