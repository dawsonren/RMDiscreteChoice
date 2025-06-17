"""
Generate data from a model and estimate parameters using EM.
"""
import numpy as np
from typing import List, Tuple, Dict
import scipy.optimize as opt

from choice import mnl

def expectation_maximization(beta_init, lambda_init, utility_fcn, S_t, purchase_data):
    # S_t is the assortment offered at each time step
    # purchase_data is the item j from each assortment that was purchased, or None if no purchase was made
    # the None could result from either no-purchase or no-arrival

    def e_step(b, l):
        arrivals = np.zeros(len(S_t))
        for i, S in enumerate(S_t):
            arrivals[i] = l * mnl(0, S, utility_fcn(b)) / (l * mnl(0, S, utility_fcn(b)) + (1 - l))
        return arrivals
    
    def m_step(arrivals):
        # separable parameters
        l = 3