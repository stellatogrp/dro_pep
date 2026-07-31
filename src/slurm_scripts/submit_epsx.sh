#!/bin/bash
# Small-radius eps-extension DRO runs for Quad (K<=40) and Lasso (K<=25).
# Grids: quad logspace(-3,-1.3,6); lasso logspace(-5,-3.2,5).
# CTRIM=3 keeps alphas {0.01, 0.05} for cvar. NEVER pass -q to sbatch.
set -euo pipefail
cd /scratch/gpfs/BSTELLATO/bs37/projects/dro_pep_cert/src
S=slurm_scripts/run_cert_dro_chunk.sh

QARGS="eps.log_min=-3 eps.log_max=-1.3 eps.space_count=6"
LARGS="eps.space_type=logspace eps.log_min=-5 eps.log_max=-3.2 eps.space_count=5"

submit () { # exp alg tag measure kmin kmax args extra_sbatch...
  local exp=$1 alg=$2 tag=$3 measure=$4 kmin=$5 kmax=$6 args=$7; shift 7
  sbatch "$@" --export=ALL,CEXP=$exp,CMEASURE=$measure,CKMIN=$kmin,CKMAX=$kmax,CTRIM=3,CTAG=$tag,"CARGS=alg=$alg $args" $S
}

for a in grad_desc:gd nesterov_grad_desc:fgm; do
  alg=${a%%:*}; tag=${a##*:}_epsx
  submit quad $alg $tag expectation 1 20  "$QARGS"
  submit quad $alg $tag expectation 21 32 "$QARGS" --time=11:59:59 --mem-per-cpu=6G
  submit quad $alg $tag expectation 33 40 "$QARGS" --time=11:59:59 --mem-per-cpu=8G
  submit quad $alg $tag cvar 1 16  "$QARGS"
  submit quad $alg $tag cvar 17 26 "$QARGS" --time=11:59:59 --mem-per-cpu=6G
  submit quad $alg $tag cvar 27 33 "$QARGS" --time=23:59:59 --cpus-per-task=8 --mem-per-cpu=10G
  submit quad $alg $tag cvar 34 40 "$QARGS" --time=23:59:59 --cpus-per-task=8 --mem-per-cpu=12G
done

for a in ista fista; do
  tag=${a}_epsx
  submit lasso $a $tag expectation 1 12  "$LARGS"
  submit lasso $a $tag expectation 13 20 "$LARGS" --time=11:59:59 --mem-per-cpu=6G
  submit lasso $a $tag expectation 21 25 "$LARGS" --time=11:59:59 --mem-per-cpu=8G
  submit lasso $a $tag cvar 1 10  "$LARGS"
  submit lasso $a $tag cvar 11 16 "$LARGS" --time=11:59:59 --mem-per-cpu=6G
  submit lasso $a $tag cvar 17 21 "$LARGS" --time=23:59:59 --cpus-per-task=8 --mem-per-cpu=10G
  submit lasso $a $tag cvar 22 25 "$LARGS" --time=23:59:59 --cpus-per-task=8 --mem-per-cpu=14G
done
