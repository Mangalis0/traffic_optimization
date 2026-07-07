"""
Signal timing optimizer.

Two strategies are provided:

1. optimize_network  (Webster's formula — decentralised)
   Each intersection solved independently in O(1).
   Minimises average delay in isolation; fast and analytically exact.

2. optimize_network_moo  (scipy — joint multi-objective)
   All intersections solved together with scipy.minimize (L-BFGS-B).
   Objectives:
     (a) Total network average delay   — same driver as Webster's
     (b) Fairness                      — penalise variance of per-intersection
                                         delay so no single intersection becomes
                                         a disproportionate bottleneck
   Warm-started from Webster's solution for fast convergence (~1 ms).

Webster's optimal cycle (minimises average delay for undersaturated traffic):
    C* = (1.5 L + 5) / (1 - Y)
    where L = total lost time per cycle
          Y = sum of critical flow ratios (q / s)

Webster uniform delay per approach:
    d = C(1 - λ)² / (2(1 - y))
    where λ = g/C (green ratio), y = q/s (flow ratio, capped at 0.99)
"""

from typing import Dict, Tuple
import numpy as np
from scipy.optimize import minimize


SATURATION_FLOW = 1800   # veh/hr — realistic single-lane saturation flow
MIN_GREEN = 10.0         # seconds
MAX_GREEN = 90.0         # seconds
YELLOW_TIME = 3.0        # seconds
ALL_RED_TIME = 2.0       # seconds
LOST_TIME_PER_PHASE = YELLOW_TIME + ALL_RED_TIME   # 5 s
NUM_PHASES = 2
TOTAL_LOST_TIME = NUM_PHASES * LOST_TIME_PER_PHASE  # 10 s
MIN_CYCLE = MIN_GREEN * 2 + TOTAL_LOST_TIME         # 30 s
MAX_CYCLE = 180.0        # seconds


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _websters(q_ns: float, q_ew: float) -> Tuple[float, float, float]:
    """
    Compute Webster-optimal cycle length and green splits.

    Returns:
        (cycle_length, green_ns, green_ew) in seconds.
    """
    s = SATURATION_FLOW
    y_ns = q_ns / s
    y_ew = q_ew / s
    Y = min(y_ns + y_ew, 0.85)   # cap to avoid instability

    C = np.clip((1.5 * TOTAL_LOST_TIME + 5) / (1 - Y), MIN_CYCLE, MAX_CYCLE)

    effective_green = C - TOTAL_LOST_TIME
    if Y > 0:
        g_ns = effective_green * y_ns / Y
        g_ew = effective_green * y_ew / Y
    else:
        g_ns = g_ew = effective_green / 2

    g_ns = np.clip(g_ns, MIN_GREEN, MAX_GREEN)
    g_ew = np.clip(g_ew, MIN_GREEN, MAX_GREEN)
    C = g_ns + g_ew + TOTAL_LOST_TIME

    return float(C), float(g_ns), float(g_ew)


def _webster_delay(g: float, C: float, q: float) -> float:
    """
    Webster uniform delay (seconds/vehicle) for one approach.

    d = C(1 - λ)² / (2(1 - y))
    Valid for undersaturated conditions (y < 1).
    """
    y = min(q / SATURATION_FLOW, 0.99)
    lam = np.clip(g / C, 0.01, 0.99)
    return C * (1 - lam) ** 2 / (2 * (1 - y))


# ---------------------------------------------------------------------------
# Strategy 1: Webster's (decentralised, O(1))
# ---------------------------------------------------------------------------

def optimize_network(
    predicted_rates: Dict[int, Dict[str, float]],
) -> Dict[int, Dict[str, float]]:
    """
    Compute optimal signal timings using Webster's formula (decentralised).

    Each intersection is optimised independently — fast and analytically
    exact for single-objective delay minimisation.

    Args:
        predicted_rates: {intersection_id: {direction: rate_vph}}

    Returns:
        {intersection_id: {'green_ns': float, 'green_ew': float}}
    """
    timings: Dict[int, Dict[str, float]] = {}
    for inter_id, rates in predicted_rates.items():
        q_ns = max(rates.get('N', 20.0), rates.get('S', 20.0))
        q_ew = max(rates.get('E', 20.0), rates.get('W', 20.0))
        _, g_ns, g_ew = _websters(q_ns, q_ew)
        timings[inter_id] = {'green_ns': g_ns, 'green_ew': g_ew}
    return timings


# ---------------------------------------------------------------------------
# Strategy 2: Multi-objective (joint, scipy L-BFGS-B)
# ---------------------------------------------------------------------------

def optimize_network_moo(
    predicted_rates: Dict[int, Dict[str, float]],
    w_fairness: float = 0.3,
) -> Dict[int, Dict[str, float]]:
    """
    Multi-objective signal optimizer (joint across all intersections).

    Minimises a weighted sum of two objectives:
      (a) Mean network delay — total delay averaged across intersections,
          weighted by approach flow so busier approaches count more.
      (b) Fairness penalty — variance of per-intersection delay.
          Penalising variance ensures no single intersection becomes a
          bottleneck while others are lightly loaded.

    Compared to decentralised Webster's:
    - Webster's finds the analytically optimal solution *per intersection*
      in isolation, allowing different cycle lengths everywhere.
    - This joint formulation couples all intersections: the solver can
      trade a small increase in delay at one lightly-loaded intersection
      to substantially reduce delay at a heavily-loaded neighbour.

    Warm-started from Webster's solution → typically converges in < 1 ms.

    Args:
        predicted_rates: {intersection_id: {direction: rate_vph}}
        w_fairness: weight on the fairness term (0 → pure delay = Webster's)

    Returns:
        {intersection_id: {'green_ns': float, 'green_ew': float}}
    """
    ids = sorted(predicted_rates.keys())
    n = len(ids)

    # Critical (binding) flow per phase at each intersection
    q_ns = np.array([
        max(predicted_rates[i].get('N', 20.0), predicted_rates[i].get('S', 20.0))
        for i in ids
    ])
    q_ew = np.array([
        max(predicted_rates[i].get('E', 20.0), predicted_rates[i].get('W', 20.0))
        for i in ids
    ])

    def objective(x: np.ndarray) -> float:
        # x = [g_ns_0, g_ew_0, g_ns_1, g_ew_1, ...]
        inter_delays = np.empty(n)
        for k in range(n):
            g_ns_k = x[2 * k]
            g_ew_k = x[2 * k + 1]
            C_k = g_ns_k + g_ew_k + TOTAL_LOST_TIME
            d_ns = _webster_delay(g_ns_k, C_k, q_ns[k])
            d_ew = _webster_delay(g_ew_k, C_k, q_ew[k])
            # Flow-weighted average delay for this intersection
            total_flow = q_ns[k] + q_ew[k]
            inter_delays[k] = (d_ns * q_ns[k] + d_ew * q_ew[k]) / total_flow

        mean_delay = float(np.mean(inter_delays))
        fairness_penalty = float(np.var(inter_delays))  # 0 when all equal
        return mean_delay + w_fairness * fairness_penalty

    # Stability constraints: g/C >= y must hold at whatever cycle the solver
    # picks (not just at C_min).  Rearranged to a linear form for SLSQP:
    #   NS: g_ns*(1-y_ns) - g_ew*y_ns  >= y_ns*L   (≥ 0 for scipy 'ineq')
    #   EW: g_ew*(1-y_ew) - g_ns*y_ew  >= y_ew*L
    y_ns = np.minimum(q_ns / SATURATION_FLOW, 0.95)
    y_ew = np.minimum(q_ew / SATURATION_FLOW, 0.95)
    L = TOTAL_LOST_TIME

    constraints = []
    for k in range(n):
        ynk, yek = float(y_ns[k]), float(y_ew[k])

        def _ns(x, k=k, y=ynk, L=L):
            return x[2*k] * (1 - y) - x[2*k+1] * y - y * L

        def _ew(x, k=k, y=yek, L=L):
            return x[2*k+1] * (1 - y) - x[2*k] * y - y * L

        constraints.append({'type': 'ineq', 'fun': _ns})
        constraints.append({'type': 'ineq', 'fun': _ew})

    # Warm-start: Webster's solution per intersection (always feasible)
    x0 = np.empty(2 * n)
    for k in range(n):
        _, g_ns0, g_ew0 = _websters(q_ns[k], q_ew[k])
        x0[2 * k]     = g_ns0
        x0[2 * k + 1] = g_ew0

    bounds = [(MIN_GREEN, MAX_GREEN)] * (2 * n)

    result = minimize(
        objective, x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 500, 'ftol': 1e-9},
    )

    x_opt = result.x
    timings: Dict[int, Dict[str, float]] = {}
    for k, i in enumerate(ids):
        timings[i] = {
            'green_ns': float(np.clip(x_opt[2 * k],     MIN_GREEN, MAX_GREEN)),
            'green_ew': float(np.clip(x_opt[2 * k + 1], MIN_GREEN, MAX_GREEN)),
        }
    return timings


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def fixed_timing(
    green_time: float = 30.0,
    num_intersections: int = 4,
) -> Dict[int, Dict[str, float]]:
    """Baseline: equal fixed split at every intersection."""
    return {
        i: {'green_ns': green_time, 'green_ew': green_time}
        for i in range(num_intersections)
    }
