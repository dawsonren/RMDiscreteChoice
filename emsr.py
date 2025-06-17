"""
Calculate Booking Limits for EMSRb
"""
from typing import List

import numpy as np
from scipy.stats import norm

###
### Helpers
###
def check_fares_decreasing(fares: np.ndarray):
    decreasing = all(x >= y for x, y in zip(fares, fares[1:]))
    if not decreasing:
        raise ValueError('Fares must be provided in decreasing order.')

###
### EMSRb Implementation
###
def calc_EMSRb(fares: np.ndarray, demands: np.ndarray, sigmas=None):
    """
    Standard EMSRb algorithm assuming Gaussian distribution of
    demands for the classes.

    Parameters
    ----------
    fares: array of fares (decreasing order)
    demands: array of predicted demands for the fares in `fares`
    sigmas: array of standard deviations of the demand predictions

    Returns
    -------
    array of protection levels for each fare class
    """
    # initialize protection levels y
    y = np.zeros(len(fares) - 1)

    if sigmas is None or np.all(sigmas == 0):
        # Deterministic EMSRb
        y = demands.cumsum()[:-1]
    else:
        for j in range(1, len(fares)):
            S_j = demands[:j].sum()
            # eq. 2.13
            p_j_bar = np.sum(demands[:j]*fares[:j]) / demands[:j].sum()
            p_j_plus_1 = fares[j]
            z_alpha = norm.ppf(1 - p_j_plus_1 / p_j_bar)
            # sigma of joint distribution
            sigma = np.sqrt(np.sum(sigmas[:j]**2))
            # mean of joint distribution
            mu = S_j
            y[j-1] = mu + z_alpha*sigma

        # ensure that protection levels are neither negative (e.g. when
        # demand is low and sigma is high) nor NaN (e.g. when demand is 0)
        y[y < 0] = 0
        y[np.isnan(y)] = 0

        # ensure that protection levels are monotonically increasing.
        y = np.maximum.accumulate(y)

    # protection level for most expensive class should be always 0
    return np.hstack((0, np.round(y)))


###
### Public API
###
def protection_levels(fares, demands, sigmas=None):
    """
    Calculate protection levels.

    Parameters
    ----------
    fares: array of fares (decreasing order)
    demands: array of predicted demands for the fares in `fares`
    sigmas: array of standard deviations of the demand predictions

    Returns
    -------
    array of protection levels for each fare class
    """
    check_fares_decreasing(fares)
    return calc_EMSRb(fares, demands, sigmas)
