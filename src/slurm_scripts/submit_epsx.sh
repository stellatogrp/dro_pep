#!/bin/bash
# Full DRO campaign for the Quad and Lasso figures, submitted from scratch.
#
# This is the reproducibility entry point: on a fresh clone, `bash
# submit_epsx.sh` produces exactly the (K, eps, alpha) rows behind the
# committed experiment_plots/{Quad,Lasso}/data/dro CSVs. The configs carry the
# base eps grid; the refinements below cannot be expressed as a single hydra
# grid, which is why they live here rather than in the YAML.
#
#   Quad  (K <= 40): base logspace(-1, 1, 13) from configs/quad.yaml
#                  + logspace(-3, -1.3, 6)   small-radius extension
#                  + logspace(-5, -3.4, 4)   second-level extension
#                  = 23 radii spanning [1e-5, 10]
#   Lasso (K <= 25): base linspace(1e-3, 1e-1, 7) from configs/lasso.yaml
#                  + logspace(-5, -3.2, 5)   small-radius extension
#                  + logspace(-3, -1, 9)     mid-range densification
#                  = 19 radii spanning [1e-5, 1e-1]
#
# The grid must resolve the selected radius from below: cross-validation picks
# the smallest covering bound, so a grid that bottoms out at the selection, or
# is coarse just below it, silently inflates the certificate (measured: quad
# CVaR up to 12x at K=40; a 17.5x Lasso gap put a visible step at K=11).
# plot_bounds.py warns when either happens.
#
# All runs use the in-sample N of the configs (100); CTRIM=3 keeps the alphas
# {0.01, 0.05} for cvar. NEVER pass -q to sbatch (it means --qos and silently
# swallows the next flag).
set -euo pipefail
cd "$(dirname "$0")/.."
S=slurm_scripts/run_cert_dro_chunk.sh

QBASE=""                                                    # configs/quad.yaml
QEXT="eps.log_min=-3 eps.log_max=-1.3 eps.space_count=6"
QEXT2="eps.log_min=-5 eps.log_max=-3.4 eps.space_count=4"
LBASE=""                                                    # configs/lasso.yaml
LEXT="eps.space_type=logspace eps.log_min=-5 eps.log_max=-3.2 eps.space_count=5"
LMID="eps.space_type=logspace eps.log_min=-3 eps.log_max=-1 eps.space_count=9"

submit () { # exp alg tag measure kmin kmax args [extra sbatch flags...]
  local exp=$1 alg=$2 tag=$3 measure=$4 kmin=$5 kmax=$6 args=$7; shift 7
  sbatch "$@" --export=ALL,CEXP=$exp,CMEASURE=$measure,CKMIN=$kmin,CKMAX=$kmax,CTRIM=3,CTAG=$tag,"CARGS=alg=$alg $args" $S
}

# Quadratic: three grids x {expectation, cvar}, chunked by K. CVaR cost grows
# steeply in K (Gram dimension K+2), hence the wider profiles at the tail.
for a in grad_desc:gd nesterov_grad_desc:fgm; do
  alg=${a%%:*}; t=${a##*:}
  for g in base:"$QBASE" ext:"$QEXT" ext2:"$QEXT2"; do
    gn=${g%%:*}; ga=${g#*:}
    submit quad $alg ${t}_${gn} expectation 1 40 "$ga" --time=11:59:59 --mem-per-cpu=6G
    submit quad $alg ${t}_${gn} cvar 1 20  "$ga" --time=23:59:59 --mem-per-cpu=8G
    submit quad $alg ${t}_${gn} cvar 21 30 "$ga" --time=23:59:59 --cpus-per-task=8 --mem-per-cpu=8G
    submit quad $alg ${t}_${gn} cvar 31 35 "$ga" --time=23:59:59 --cpus-per-task=8 --mem-per-cpu=8G
    submit quad $alg ${t}_${gn} cvar 36 40 "$ga" --time=23:59:59 --cpus-per-task=8 --mem-per-cpu=8G
  done
done

# Lasso: the Gram dimension is 2K+5, so the same K costs far more than in Quad;
# the largest K are submitted one at a time.
for alg in ista fista; do
  for g in base:"$LBASE" ext:"$LEXT" mid:"$LMID"; do
    gn=${g%%:*}; ga=${g#*:}
    submit lasso $alg ${alg}_${gn} expectation 1 25 "$ga" --time=11:59:59 --mem-per-cpu=6G
    submit lasso $alg ${alg}_${gn} cvar 1 12  "$ga" --time=23:59:59 --cpus-per-task=8 --mem-per-cpu=8G
    submit lasso $alg ${alg}_${gn} cvar 13 18 "$ga" --time=23:59:59 --cpus-per-task=8 --mem-per-cpu=8G
    submit lasso $alg ${alg}_${gn} cvar 19 23 "$ga" --time=23:59:59 --cpus-per-task=8 --mem-per-cpu=10G
    submit lasso $alg ${alg}_${gn} cvar 24 24 "$ga" --time=23:59:59 --cpus-per-task=8 --mem-per-cpu=10G
    submit lasso $alg ${alg}_${gn} cvar 25 25 "$ga" --time=23:59:59 --cpus-per-task=8 --mem-per-cpu=10G
  done
done
