# Data-driven Analysis and Learning of First-Order Methods via Distributionally Robust Optimization

This repository is by [Jisun Park](https://jisunp515.github.io/), [Vinit Ranjan](https://vinitranjan1.github.io/), and [Bartolomeo Stellato](https://stellato.io/) and contains the Python source code to reproduce experiments in our papers:
- [Data-driven Analysis of First-Order Methods via Distributionally Robust Optimization](https://arxiv.org/abs/2511.17834).
- [Distributionally-Robust Learning to Optimize] (https://arxiv.org/abs/2605.06585).

# DRO-PEP
We consider the problem of analyzing the probabilistic performance of first-order methods when solving convex optimization problems drawn from an unknown distribution only accessible through samples. By combining performance estimation (PEP) and Wasserstein distributionally robust optimization (DRO), we formulate the analysis as a tractable semidefinite program. Our approach unifies worst-case and average-case analyses by incorporating data-driven information from the observed convergence of first-order methods on a limited number of problem instances. This yields probabilistic, data-driven performance guarantees in terms of the expectation or conditional value-at-risk of the selected performance metric. Experiments on smooth convex minimization, logistics regression, and Lasso show that our method significantly reduces the conservatism of classical worst-case bounds and narrows the gap between theoretical and empirical performance.

# DR-L2O
We propose a distributionally robust approach to learning hyperparameters for first-order methods in convex optimization. Given a dataset of problem instances, we minimize a Wasserstein distributionally robust version of the performance estimation problem (PEP) over algorithm parameters such as step sizes. Our framework unifies two extremes: as the robustness radius vanishes, we recover classical learning to optimize (L2O); as it grows, we recover worst-case optimal algorithm design via PEP. We solve the resulting problem with stochastic gradient descent, differentiating through the solution of an inner semidefinite program at each step. We prove high-probability bounds showing that the true risk of the learned algorithm is at most the in-sample L2O optimum plus a slack that shrinks with the sample size, and is no worse than the worst-case PEP bound. On unconstrained quadratic minimization, LASSO, and linear programming benchmarks, our learned algorithms achieve strong out-of-sample performance with certifiable robustness, outperforming both worst-case optimal and vanilla L2O baselines.

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
JAX
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
learning
learning_l2oo
learning_optpep
```
and ```<experiment_name>``` is one of the following:
```
Quad
LogReg
Lasso
PDLP
```

### Results
For each experiment, the results are saved in the corresponding `<example>/outputs/` folder and is timestamped by Hydra with the date and time of the experiment. The results include the residual values, times, and other auxiliary information along with the experiment log to track outputs.
