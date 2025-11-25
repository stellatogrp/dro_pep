# Data-driven Analysis of First-Order Methods via Distributionally Robust Optimization

This repository is by [Jisun Park](https://jisunp515.github.io/), [Vinit Ranjan](https://vinitranjan1.github.io/), and [Bartolomeo Stellato](https://stellato.io/) and contains the Python source code to reproduce experiments in our paper [Data-driven Analysis of First-Order Methods via Distributionally Robust Optimization](https://arxiv.org/abs/2511.17834).

# Abstract
We consider the problem of analyzing the probabilistic performance of first-order methods when solving convex optimization problems drawn from an unknown distribution only accessible through samples. By combining performance estimation (PEP) and Wasserstein distributionally robust optimization (DRO), we formulate the analysis as a tractable semidefinite program. Our approach unifies worst-case and average-case analyses by incorporating data-driven information from the observed convergence of first-order methods on a limited number of problem instances. This yields probabilistic, data-driven performance guarantees in terms of the expectation or conditional value-at-risk of the selected performance metric. Experiments on smooth convex minimization, logistics regression, and Lasso show that our method significantly reduces the conservatism of classical worst-case bounds and narrows the gap between theoretical and empirical performance.

## Installation
To install the package, run
```
$ pip install git+https://github.com/stellatogrp/dro_pep
```

## Packages
The main required packages are 
```
cvxpy
Clarabel
PEPit
hydra
```

### Running experiments
Experiments for the paper should be run from the `src/` folder with the command:
```
python run_<experiment_type>_experiment.py <experiment_name> local
```
where ```<experiment_type>``` is one of the following:
```
sample
dro
pep
```
and ```<experiment_name>``` is one of the following:
```
Quad
LogReg
Lasso
```

### Results
For each experiment, the results are saved in the corresponding `<example>/outputs/` folder and is timestamped by Hydra with the date and time of the experiment. The results include the residual values, times, and other auxiliary information along with the experiment log to track outputs.
