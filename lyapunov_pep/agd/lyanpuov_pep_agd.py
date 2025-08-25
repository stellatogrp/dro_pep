import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from copy import copy
from interpolation_conditions import smooth_strongly_convex, smooth_strongly_convex_agd
from sample_generation import sample_generation
from argparse import ArgumentParser