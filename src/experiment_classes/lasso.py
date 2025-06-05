import numpy as np
import pandas as pd
import logging
import time
from tqdm import trange

# from .utils import marchenko_pastur, gradient_descent, nesterov_accelerated_gradient, generate_trajectories
from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexQuadraticFunction, ConvexLipschitzFunction
from reformulator.dro_reformulator import DROReformulator

log = logging.getLogger(__name__)