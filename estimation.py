"""
Generate data from a model and estimate parameters using EM.
"""
import numpy as np
from typing import List, Tuple, Dict
import scipy.optimize as opt

from choice import mnl

def expectation_maximization(beta_init, lambda_init, utility_fcn, S_t, purchase_data: np.ndarray):
    # S_t is the assortment offered at each time step
    # purchase_data is the item j from each assortment that was purchased, or -1 if no purchase was made
    # the -1 could result from either no-purchase or no-arrival
    # utility_fcn is a function that takes beta and returns the utility vector for all products
    D = len(S_t)  # number of time steps
    P = purchase_data != -1  # boolean mask for purchases made
    Pbar = purchase_data == -1  # boolean mask for no purchases made (could be no purchase or no arrival)

    def e_step(b, l):
        arrivals = np.zeros(len(S_t))
        for i, S in enumerate(S_t):
            arrivals[i] = l * mnl(-1, S, utility_fcn(b)) / (l * mnl(-1, S, utility_fcn(b)) + (1 - l))

        # arrivals is an array of expected arrivals for each time step
        return arrivals
    
    def m_step(arrivals: np.ndarray, old_beta):
        # separable parameters
        l = (P.sum() + arrivals[Pbar].sum()) / D
        # estimate beta using log-likelihood
        def neg_log_likelihood(beta):
            log_likelihood = 0.0
            for i, S in enumerate(S_t):
                if P[i]:
                    log_likelihood += np.log(mnl(purchase_data[i], S, utility_fcn(beta)))
                elif Pbar[i]:
                    log_likelihood += np.log(arrivals[i] * mnl(-1, S, utility_fcn(beta)))
            return -log_likelihood

        # this is really easy optimization since it's univariate
        beta = opt.minimize(neg_log_likelihood, old_beta, method='nelder-mead').x
        return beta, l
    
    beta = beta_init
    lambda_ = lambda_init
    for _ in range(100):  # max iterations
        arrivals = e_step(beta, lambda_)
        # store old beta and lambda for convergence check
        old_beta, old_lambda = beta, lambda_
        beta, lambda_ = m_step(arrivals, old_beta)
        # Check for convergence
        if np.allclose(beta, old_beta, atol=1e-6) and np.isclose(lambda_, old_lambda, atol=1e-6):
            print("Converged")
            break

    return beta, lambda_