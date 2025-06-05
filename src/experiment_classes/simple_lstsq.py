import numpy as np
import pandas as pd
import logging
import time
from tqdm import trange

from .utils import generate_P_bounded_mu_L, gradient_descent, generate_trajectories

log = logging.getLogger(__name__)


def lstsq_samples(cfg):
    log.info(cfg)
