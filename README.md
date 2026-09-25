# Learning Distributionally Robust First-Order Methods for Convex Optimization

This repository is by Vinit Ranjan, Jisun Park, and Bartolomeo Stellato and contains the Python source code to reproduce experiments in our paper `Learning Distributionally Robust First-Order Methods for Convex Optimization`.

# Abstract
We propose a distributionally robust approach for learning hyperparameters (e.g., step sizes and momentum coefficients) for first-order methods in convex optimization. Given a dataset of problem instances, we minimize a Wasserstein distributionally robust version of the performance estimation problem (PEP). Our framework unifies two extremes. As the robustness radius vanishes, we recover the empirical risk minimization problem commonly encountered in learning to optimize (L2O), and as it grows we recover the PEP worst-case algorithm design problem. We solve the resulting problem with stochastic gradient descent, differentiating through the solution of an inner semidefinite program at each step. We prove high-probability bounds showing that the true risk of the learned algorithm is at most the in-sample L2O optimum plus a slack that shrinks with the sample size, and is no worse than the worst-case PEP bound. On logistic regression, LASSO, and linear programming examples, our learned algorithms achieve strong out-of-sample performance with certifiable robustness, outperforming both worst-case optimal and vanilla L2O baselines.

<!-- ## Installation
To install the package, run
```
$ pip install git+https://github.com/stellatogrp/dro_pep
``` -->

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
learning
learning_l2o
learning_lpep
```
and ```<experiment_name>``` is one of the following:
```
LogReg
Lasso
PDLP
```

### Results
For each experiment, the results are saved in the corresponding `<example>/outputs/` folder and is timestamped by Hydra with the date and time of the experiment. The results include the residual values, times, and other auxiliary information along with the experiment log to track outputs.
