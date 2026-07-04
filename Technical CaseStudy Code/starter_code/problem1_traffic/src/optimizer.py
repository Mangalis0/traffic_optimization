"""
Signal timing optimizer.

Uses Webster's formula to compute analytically optimal cycle length and
green split for each intersection, given predicted approach demand.

Webster's optimal cycle (minimises average delay for undersaturated traffic):
    C* = (1.5 L + 5) / (1 - Y)
    where L = total lost time per cycle
          Y = sum of critical flow ratios (q / s)

Optimal green split:
    g_i = (C* - L) * y_i / Y

Constraints: MIN_GREEN ≤ g_i ≤ MAX_GREEN
"""

from typing import Dict, Tuple
import numpy as np


SATURATION_FLOW = 3600   # veh/hr — simplified single-lane model (1 veh/sec)
MIN_GREEN = 10.0         # seconds
MAX_GREEN = 90.0         # seconds
YELLOW_TIME = 3.0        # seconds
ALL_RED_TIME = 2.0       # seconds
LOST_TIME_PER_PHASE = YELLOW_TIME + ALL_RED_TIME   # 5 s
NUM_PHASES = 2
TOTAL_LOST_TIME = NUM_PHASES * LOST_TIME_PER_PHASE  # 10 s
MIN_CYCLE = MIN_GREEN * 2 + TOTAL_LOST_TIME         # 30 s
MAX_CYCLE = 180.0        # seconds


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


def optimize_network(
    predicted_rates: Dict[int, Dict[str, float]],
) -> Dict[int, Dict[str, float]]:
    """
    Compute optimal signal timings for a 4-intersection network.

    Each intersection is optimised independently (decentralised) using
    Webster's formula applied to its predicted approach volumes.

    Args:
        predicted_rates: {intersection_id: {direction: rate_vph}}

    Returns:
        {intersection_id: {'green_ns': float, 'green_ew': float}}
    """
    timings: Dict[int, Dict[str, float]] = {}
    for inter_id, rates in predicted_rates.items():
        # Webster uses the *critical* flow ratio per phase — the heavier approach
        # (both N and S are served simultaneously, so max is the binding constraint)
        q_ns = max(rates.get('N', 20.0), rates.get('S', 20.0))
        q_ew = max(rates.get('E', 20.0), rates.get('W', 20.0))
        _, g_ns, g_ew = _websters(q_ns, q_ew)
        timings[inter_id] = {'green_ns': g_ns, 'green_ew': g_ew}
    return timings


def fixed_timing(green_time: float = 30.0) -> Dict[int, Dict[str, float]]:
    """Baseline: equal fixed split at every intersection."""
    return {i: {'green_ns': green_time, 'green_ew': green_time} for i in range(4)}
