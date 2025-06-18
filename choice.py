"""
Solves the choice-based dynamic programming problem.

We solve a dynamic program with state (x, t) for the optimal
assortment of products given a choice set N where
x is the remaining capacity and t is the time remaining.
"""
from typing import List, Tuple, Dict, Callable
from itertools import combinations
import numpy as np

ChoiceModel = Callable[[int, List[int], List[float]], float]
ValueFcn = Dict[Tuple[int, int], float]
ActionFcn = Dict[Tuple[int, int], List[int]]

def mnl(j: int, S: List[int], u: List[float]) -> float:
    """MNL choice probability."""
    if j != -1 and j not in S:
        return 0.0
    exp_u = np.exp([u[i] for i in S])
    denom = 1 + np.sum(exp_u)
    num = np.exp(u[j]) if j != -1 else 1.0
    return num / denom

def exponomial(j: int, S: List[int], u: List[float]) -> float:
    """
    Exponomial choice probability P_j given:
      j: 0-based index of the chosen alternative (0 <= j < n)
      S: set of 0-based indices offered (subset of {0,...,n-1})
      u: list of utilities of length n, sorted nonincreasingly
    
    Returns
    -------
    float
        Choice probability of alternative j.
    """
    n = len(u)
    if j != -1 and j not in S:
        return 0.0
    
    # build bit‐vector
    x = [1 if i in S else 0 for i in range(n)]
    
    # prefix sums
    X = [0] * n
    running = 0
    for i in range(n):
        running += x[i]
        X[i] = running
    
    # compute G
    G = [0.0] * n
    for i in range(n):
        if X[i] > 0:
            exponent = 0.0
            for k in range(i+1):
                if x[k]:
                    exponent -= (u[k] - u[i])
            G[i] = np.exp(exponent) / X[i]
    
    # correction term
    correction = 0.0
    for k in range(j+1, n):
        if x[k] and X[k-1] > 0:
            correction += G[k] / X[k-1]
    
    return G[j] - correction

def R(S: List[int], r: List[float], u: List[float], P: ChoiceModel) -> float:
    """Expected revenue from assortment S."""
    return sum(r[i] * P(i, S, u) for i in S)

def Q(S: List[int], u: List[float], P: ChoiceModel) -> float:
    """Probability of no-purchase complement: sum of purchase probabilities."""
    return sum(P(i, S, u) for i in S)

def efficient_sets(r: List[float], u: List[float], P: ChoiceModel) -> List[List[int]]:
    """Generate efficient sets."""
    n = len(r)
    efficient = [[]]
    while True:
        best_slope = 0
        best_set = None
        for k in range(1, n+1):
            for subset in combinations(range(n), k):
                subset = list(subset)
                denom_diff = Q(subset, u, P) - Q(efficient[-1], u, P)
                if denom_diff <= 0:
                    continue
                slope = (R(subset, r, u, P) - R(efficient[-1], r, u, P)) / denom_diff
                if slope > best_slope:
                    best_slope = slope
                    best_set = subset
        if best_set is None:
            break
        efficient.append(best_set)
    return efficient

def choice_dp_tabulate(arrival_rate: float, C: int, T: int,
                       r: List[float], u: List[float], P: ChoiceModel,
                       check_monotonicity: bool = False
                      ) -> Tuple[np.ndarray, List[List[List[int]]]]:
    """
    Bottom-up DP with tabulation and pre-computation of R and Q.
    Returns:
        V: (C+1)x(T+1) array of value function
        Aopt: (C+1)x(T+1) list of optimal assortments
    """
    # 1. Precompute efficient sets
    if P == mnl:
        # for MNL, we know the efficient sets are complete
        eff_sets = [list(range(k)) for k in range(1, len(r) + 1)]
    else:
        eff_sets = efficient_sets(r, u, P)
    
    # 2. Precompute R and Q for each efficient set
    R_cache = {tuple(S): R(S, r, u, P) for S in eff_sets}
    Q_cache = {tuple(S): Q(S, u, P) for S in eff_sets}
    
    # 3. Initialize DP tables
    V = np.zeros((C+1, T+1))
    Aopt = [[[] for _ in range(T+1)] for _ in range(C+1)]
    cost_of_capacity = np.zeros((C+1, T+1))
    
    # 4. Tabulation
    for t in range(1, T+1):
        for x in range(1, C+1):
            no_sale = V[x, t-1]
            sale    = V[x-1, t-1]
            delta_cost = no_sale - sale
            cost_of_capacity[x, t] = delta_cost
            
            best_val = -np.inf
            best_set = []
            for S in eff_sets:
                key = tuple(S)
                val = R_cache[key] - Q_cache[key] * delta_cost
                if val > best_val:
                    best_val = val
                    best_set = S
            
            V[x, t] = arrival_rate * best_val + no_sale
            Aopt[x][t] = best_set
    
    # 5. Check monotonicity of cost of capacity
    if check_monotonicity:
        for t in range(2, T):
            for x in range(2, C):
                # we expect cost of capacity to go up when x decreases
                if cost_of_capacity[x, t] > cost_of_capacity[x-1, t]:
                    print(f"Cost of capacity is not monotonic at (x={x}, t={t}), D({x}, {t}) = {cost_of_capacity[x, t]} > {cost_of_capacity[x-1, t]} = D({x-1}, {t})")
                # we expect cost of capacity to go down when t decreases
                if cost_of_capacity[x, t] < cost_of_capacity[x, t-1]:
                    print(f"Cost of capacity is not monotonic at (x={x}, t={t}), D({x}, {t}) = {cost_of_capacity[x, t]} < {cost_of_capacity[x, t-1]} = D({x}, {t-1})")
    return V, Aopt

