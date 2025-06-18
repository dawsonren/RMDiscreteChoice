import numpy as np
import scipy.stats as stats

from data import REVENUES
from emsr import protection_levels


def calculate_emsr_protection_levels(proportion, mean_demand, n):
    # demands and sigmas are constant across products
    demands = proportion * mean_demand / n * np.ones_like(REVENUES)
    sigmas = np.sqrt(proportion * mean_demand / n) * np.ones_like(REVENUES)
    return protection_levels(REVENUES, demands, sigmas)

def precompute_emsr_assortments(C, T, mean_demand, n):
    """
    Precompute, for each time step t and each possible remaining budget b=0..C,
    the EMSR assortment (list of product indices) to offer.
    Returns:
        emsr_assort[t][b] -> list of indices
    """
    # Precompute protection levels for each proportion t/T
    proportions = np.arange(T-1, -1, -1) / T  # shape (T,)
    pl_matrix = np.array([
        calculate_emsr_protection_levels(p, mean_demand, n)
        for p in proportions
    ])  # shape (T, n)

    # For each period t and budget b, store indices with protection_level <= b
    emsr_assort = [None] * T
    for t in range(T):
        pl_t = pl_matrix[t]
        assort_for_b = []
        for b in range(C + 1):
            # products still available if protection level <= budget
            idx = np.nonzero(pl_t <= b)[0]
            assort_for_b.append(idx.tolist())
        emsr_assort[t] = assort_for_b

    return emsr_assort

def assortment_policy_emsr(C, proportion, mean_demand, n):
    """
    Generate an assortment for EMSR.
    """
    protection_levels = calculate_emsr_protection_levels(proportion, mean_demand, n)
    return [i for i in range(n) if protection_levels[i] <= C]

def generate_choice(S, w):
    """
    Choose from S with weights w, but include a no-purchase with weight 1.
    """
    w_S = np.array([w[i] for i in S], dtype=float)
    S = list(S)
    w = np.concatenate((w_S, [1.0]))  # Include no-purchase option
    return np.random.choice(S + [None], p=w / w.sum())

def simulate(T, C, arrival_rate, w, emsr_assort, action_fcn):
    """
    Run one simulation using precomputed EMSR assortments.
    emsr_assort[t][b] gives assortment for period t (time_to_go=T-1-t)
    and budget b.
    """
    esmr_purchases = np.full(T, -2, dtype=int)  # -2=no arrival, -1=no purchase
    mnl_purchases = np.full(T, -2, dtype=int)
    esmr_offers = []
    mnl_offers = []
    rem_budget_emsr = C
    rem_budget_mnl = C

    # Pre-generate arrivals and uniforms for choice inversion
    arrivals = np.random.rand(T) < arrival_rate

    for period in range(T):
        t_remaining = T - 1 - period
        # lookup EMSR assortment
        S_emsr = emsr_assort[t_remaining][rem_budget_emsr]
        # lookup MNL assortment
        S_mnl = action_fcn[rem_budget_mnl][t_remaining]
        esmr_offers.append(S_emsr)
        mnl_offers.append(S_mnl)

        if not arrivals[period]:
            esmr_purchases[period] = -2
            mnl_purchases[period] = -2
            continue

        esmr_choice = generate_choice(S_emsr, w)
        mnl_choice = generate_choice(S_mnl, w)
        if esmr_choice is not None:
            esmr_purchases[period] = esmr_choice
            rem_budget_emsr -= 1
        else:
            esmr_purchases[period] = -1

        if mnl_choice is not None:
            mnl_purchases[period] = mnl_choice
            rem_budget_mnl -= 1
        else:
            mnl_purchases[period] = -1

    return esmr_purchases, mnl_purchases, esmr_offers, mnl_offers

def calculate_statistics(purchases, C):
    """
    Calculate statistics for the purchases.
    """
    load_factor = np.sum(purchases >= 0) / C
    total_revenue = np.sum(REVENUES[purchases[purchases >= 0].astype(int)])
    return load_factor, total_revenue

def confidence_interval(data, confidence=0.99):
    """
    Calculate the confidence interval for the given data.
    """
    n = len(data)
    mean = np.mean(data)
    std_err = np.std(data) / np.sqrt(n)
    h = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean - h, mean + h
