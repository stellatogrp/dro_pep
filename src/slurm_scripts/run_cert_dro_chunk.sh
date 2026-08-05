#!/bin/bash
#SBATCH --job-name=CertDROchunk
#SBATCH --account=bstellato
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00-05:59:59
#SBATCH -o /scratch/gpfs/BSTELLATO/bs37/cert_dro_pep_out/runs/%j_chunk.txt
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=bs37@princeton.edu

# Experiment-agnostic per-K-range DRO runner. Pass via --export:
#   CEXP      experiment config/driver name (currently: logreg)
#   CMEASURE  expectation | cvar
#   CKMIN, CKMAX  K range
#   CARGS     extra hydra overrides, space-separated
#             (e.g. "alg=ista" or "alg=grad_desc eta=1.9 mu=0")
#   CTAG      short label folded into the output dir name
# Memory/time: override at submit for large K (see calibration notes in
# experiment_classes/logreg_notes.md).

module purge 2>/dev/null || true
module load intel-mkl/2024.2 2>/dev/null || true
export LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/latest/lib:/opt/intel/oneapi/compiler/latest/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

REPO=/scratch/gpfs/BSTELLATO/bs37/projects/dro_pep_cert
source "$REPO/.venv/bin/activate"
export DRO_PEP_DATA=/scratch/gpfs/BSTELLATO/bs37/dro_pep_data

CEXP=${CEXP:?} ; CMEASURE=${CMEASURE:?}
CKMIN=${CKMIN:?} ; CKMAX=${CKMAX:?}
CARGS=${CARGS:-} ; CTAG=${CTAG:-run} ; CTRIM=${CTRIM:-0}
export CEXP CMEASURE CKMIN CKMAX CARGS CTAG CTRIM

mkdir -p /scratch/gpfs/BSTELLATO/bs37/cert_dro_pep_out/runs

cd "$REPO/src"
python - <<'PY'
import importlib, os, sys
sys.path.insert(0, '.')
from hydra import initialize_config_dir, compose

exp = os.environ['CEXP']
measure = os.environ['CMEASURE']
kmin, kmax = os.environ['CKMIN'], os.environ['CKMAX']
extra = os.environ.get('CARGS', '').split()
tag = os.environ.get('CTAG', 'run')

dir_name = {'logreg': 'LogReg', 'lasso': 'Lasso', 'quad': 'Quad'}[exp]
driver = getattr(importlib.import_module(f'experiment_classes.{exp}'),
                 f'{exp}_dro')

overrides = [f'dro_obj={measure}', f'K_min={kmin}', f'K_max={kmax}'] + extra
# CTRIM levels avoid commas in --export values (sbatch splits on them):
# 1 -> coarser eps grid + two alphas; 2 -> coarser grid, figure alpha only
trim = os.environ.get('CTRIM', '0')
if trim == '1':
    overrides += ['eps.space_count=7', 'alpha_vals=[0.01,0.05]']
    tag += '_trim'
elif trim == '2':
    overrides += ['eps.space_count=7', 'alpha_vals=[0.01]']
    tag += '_trim2'
elif trim == '3':   # alphas only; eps range comes from CARGS (eps extension)
    overrides += ['alpha_vals=[0.01,0.05]']
    tag += '_trimA'
elif trim == '4':
    overrides += ['alpha_vals=[0.01]']
    tag += '_trimB'
cfg_dir = os.path.abspath('configs')
tag += os.environ.get('CSUFFIX', '')
out = (f"/scratch/gpfs/BSTELLATO/bs37/cert_dro_pep_out/dro_outputs/{dir_name}/"
       f"chunk_{tag}_{measure}_K{kmin}_{kmax}")
os.makedirs(out, exist_ok=True)
with initialize_config_dir(version_base='1.2', config_dir=cfg_dir):
    cfg = compose(config_name=exp, overrides=overrides)
os.chdir(out)
driver(cfg)
print('chunk done', exp, tag, measure, kmin, kmax)
PY
