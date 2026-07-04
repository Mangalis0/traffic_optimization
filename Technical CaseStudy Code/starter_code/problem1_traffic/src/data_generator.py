"""
Synthetic traffic data generator.

Produces realistic arrival rates with:
- Weekday morning rush (7-9am): heavy N-S flow
- Weekday evening rush (5-7pm): heavy E-W flow
- Weekend: lower, shifted to afternoon
- Weather effects: rain reduces demand
"""

import numpy as np
import pandas as pd
from typing import Dict


DIRECTIONS = ['N', 'S', 'E', 'W']
DIRECTION_ENC = {'N': 0, 'S': 1, 'E': 2, 'W': 3}
NUM_INTERSECTIONS = 4


def arrival_rate_for_hour(
    hour: float,
    day_of_week: int,
    direction: str,
    intersection_id: int,
    weather: float = 1.0,
) -> float:
    """
    Compute arrival rate (veh/hr) for given conditions.

    Args:
        hour: hour of day (0-23, may be fractional)
        day_of_week: 0=Monday … 6=Sunday
        direction: 'N' / 'S' / 'E' / 'W'
        intersection_id: 0-3
        weather: 1.0 = clear, 0.5 = heavy rain

    Returns:
        Arrival rate in vehicles per hour (≥ 20).
    """
    is_weekend = day_of_week >= 5

    # --- Base demand by time-of-day ---
    if is_weekend:
        if 9 <= hour < 12:
            base = 250
        elif 12 <= hour < 18:
            base = 300
        elif 18 <= hour < 22:
            base = 200
        elif 6 <= hour < 9:
            base = 150
        else:
            base = 80
    else:
        if 7 <= hour < 9:
            base = 700
        elif 17 <= hour < 19:
            base = 650
        elif 10 <= hour < 16:
            base = 400
        elif 6 <= hour < 7 or 19 <= hour < 22:
            base = 200
        else:
            base = 80

    # --- Directional bias ---
    morning_rush = (not is_weekend) and (7 <= hour < 9)
    evening_rush = (not is_weekend) and (17 <= hour < 19)

    if direction in ('N', 'S'):
        if morning_rush:
            dir_mult = 1.4
        elif evening_rush:
            dir_mult = 0.7
        else:
            dir_mult = 1.0
    else:  # E / W
        if morning_rush:
            dir_mult = 0.7
        elif evening_rush:
            dir_mult = 1.4
        else:
            dir_mult = 1.0

    # --- Per-intersection variation ---
    inter_mult = [1.0, 0.9, 1.1, 0.95][intersection_id]

    # --- Weather: rain reduces demand up to 15% ---
    weather_mult = 0.85 + 0.15 * weather

    return max(20.0, base * dir_mult * inter_mult * weather_mult)


def generate_training_data(days: int = 30, seed: int = 42) -> pd.DataFrame:
    """
    Generate a labelled dataset of historical traffic records.

    Each row represents one (15-minute interval, intersection, direction) observation.
    Returns a DataFrame ready for ML training.
    """
    np.random.seed(seed)
    records = []

    for day in range(days):
        dow = day % 7
        weather = float(np.clip(np.random.normal(0.85, 0.15), 0.5, 1.0))

        for hour in range(24):
            for minute in (0, 15, 30, 45):
                t = hour + minute / 60.0

                for inter_id in range(NUM_INTERSECTIONS):
                    for direction in DIRECTIONS:
                        rate = arrival_rate_for_hour(t, dow, direction, inter_id, weather)
                        rate += float(np.random.normal(0, rate * 0.08))
                        rate = max(10.0, rate)

                        records.append({
                            'day': day,
                            'hour': hour,
                            'minute': minute,
                            'hour_float': t,
                            'day_of_week': dow,
                            'is_weekend': int(dow >= 5),
                            'weather': weather,
                            'intersection_id': inter_id,
                            'direction': direction,
                            'direction_encoded': DIRECTION_ENC[direction],
                            'sin_hour': np.sin(2 * np.pi * t / 24),
                            'cos_hour': np.cos(2 * np.pi * t / 24),
                            'sin_dow': np.sin(2 * np.pi * dow / 7),
                            'cos_dow': np.cos(2 * np.pi * dow / 7),
                            'arrival_rate': rate,
                        })

    return pd.DataFrame(records)
