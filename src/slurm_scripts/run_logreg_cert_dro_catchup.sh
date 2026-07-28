#!/bin/bash
#SBATCH --job-name=LogRegDROcatch
#SBATCH --account=bstellato
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=22G
#SBATCH --time=00-03:59:59
#SBATCH -o /scratch/gpfs/BSTELLATO/bs37/cert_dro_pep_out/LogReg/runs/%j_catch.txt
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=bs37@princeton.edu

# Catch-up for CVaR K-ranges lost to OOM (memory accumulates across the K
# loop; a fresh process at high K_min has full headroom). Pass CALG, CETA,
# CKMIN, CKMAX via --export.

module purge 2>/dev/null || true
module load intel-mkl/2024.2 2>/dev/null || true
export LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/latest/lib:/opt/intel/oneapi/compiler/latest/lib:$LD_LIBRARY_PATH

REPO=/scratch/gpfs/BSTELLATO/bs37/projects/dro_pep_cert
source "$REPO/.venv/bin/activate"
export DRO_PEP_DATA=/scratch/gpfs/BSTELLATO/bs37/dro_pep_data

cd "$REPO/src"
python - <<'PY'
import os, sys
sys.path.insert(0, '.')
from hydra import initialize_config_dir, compose
from experiment_classes.logreg import logreg_dro

alg, eta = os.environ['CALG'], os.environ['CETA']
kmin, kmax = os.environ['CKMIN'], os.environ['CKMAX']
cfg_dir = os.path.abspath('configs')
out = (f"/scratch/gpfs/BSTELLATO/bs37/cert_dro_pep_out/dro_outputs/LogReg/"
       f"catchup_{alg}_eta{eta}_cvar_K{kmin}_{kmax}")
os.makedirs(out, exist_ok=True)
with initialize_config_dir(version_base='1.2', config_dir=cfg_dir):
    cfg = compose(config_name='logreg', overrides=[
        f'alg={alg}', f'eta={eta}', 'dro_obj=cvar',
        f'K_min={kmin}', f'K_max={kmax}'])
os.chdir(out)
logreg_dro(cfg)
print('catchup done', alg, eta, kmin, kmax)
PY
