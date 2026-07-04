# Problem 1 — Smart City Traffic Light Optimization

## Setup

```bash
cd starter_code/problem1_traffic
pip install -r requirements.txt
python main.py
```

## How to Run

| Command | What it does |
|---|---|
| `python main.py` | 2-hour morning-rush simulation (default) |
| `python main.py --hours 1` | Faster 1-hour demo |
| `python main.py --accident` | Accident event at t=1h, watch system adapt |
| `python main.py --start-hour 17` | Evening rush instead |
| `python main.py --weather 0.6` | Rainy day (lower demand) |
| `python main.py --no-plots` | Headless / CI mode, prints metrics only |

Output files (in current directory):
- `results.png` — main comparison chart
- `ml_performance.png` — prediction accuracy + feature importances
- `signal_timings.png` — how green splits vary across 24 h

---

## Approach

### ML Component — GradientBoosting Traffic Predictor
- **Features**: cyclic encodings of hour and day-of-week, weather score, intersection ID, approach direction
- **Target**: arrival rate (veh/hr) per approach
- **Why GBR?** Handles the non-linear AM/PM peaks naturally; trains in < 5 s; no hyperparameter search needed
- **Validation**: 5-fold CV; typical RMSE ~25–40 veh/hr on a base rate of 80–700 veh/hr

### Optimization — Webster's Formula
For each intersection, given predicted N+S and E+W approach volumes:

```
Y = q_NS/s + q_EW/s           # sum of critical flow ratios
C* = (1.5·L + 5) / (1 − Y)    # optimal cycle length (Webster 1958)
g_NS = (C* − L) · y_NS / Y    # proportional green split
```

- L = 10 s total lost time per cycle (yellow + all-red × 2 phases)
- s = 3600 veh/hr (single-lane saturation flow)
- Constraints: g ∈ [10, 90] s, C ∈ [30, 180] s
- Timings are recalculated every 5 minutes using fresh ML predictions

### Simulation — Queue Model
- Discrete time, DT = 1 s
- Poisson arrivals with time-varying rates
- Deterministic discharge: 1 veh/s per approach when green
- Metric: `avg_wait = total_queue_seconds / total_vehicles_served` (Little's Law)
- 4 intersections in a 2×2 grid, two signal phases (NS + EW) per intersection

---

## Results Summary

Typical output for a 2-hour morning-rush (7–9 am) simulation:

| Metric | Baseline (fixed 30-30 s) | Optimized | Improvement |
|---|---|---|---|
| Avg wait time | ~13 s | ~6–7 s | **≈ 50%** |
| Throughput | ~14 000 veh | ~14 500 veh | + 3–5% |

The improvement is driven by two effects:
1. **Shorter optimal cycle** — Webster's formula gives ≈ 30–40 s cycles vs the 70 s fixed cycle, sharply reducing uniform delay.
2. **Asymmetric green split** — extra green time is given to the heavier flow direction (N-S during AM rush, E-W during PM rush).

### Accident / Event Response
With `--accident`, intersection 0's north approach is blocked for 10 minutes at the 1-hour mark. The optimizer immediately reallocates green time away from the blocked direction, keeping average wait time stable.

---

## Key Design Decisions & Trade-offs

| Decision | Rationale |
|---|---|
| Queue model (not particle) | Fast enough for 4 intersections × 3600 s; focuses effort on ML + optimization |
| Webster's formula | Analytically optimal for undersaturated flow; transparent and explainable |
| Decentralized optimization | Each intersection optimized independently — avoids NP-hard joint problem, still yields large gains |
| GBR over neural net | Faster training, interpretable importances, better on tabular data at this scale |
| Cyclic time features | sin/cos encoding avoids discontinuity at midnight/Sunday boundary |

### Known Limitations
- Saturation flow is simplified (1 veh/s, single lane); real intersections have 1–3 lanes at 1800 veh/hr each
- No vehicle-level routing or network-wide green-wave coordination
- Weather API integration stubbed out (synthetic weather used)
- Pedestrian phases not modeled

---

## File Structure

```
problem1_traffic/
├── main.py                    # CLI entry point
├── requirements.txt
├── README.md
└── src/
    ├── data_generator.py      # Synthetic historical traffic data
    ├── ml_model.py            # GradientBoosting predictor
    ├── optimizer.py           # Webster's formula + optimize_network()
    ├── simulator.py           # Queue-based intersection simulator
    └── visualization.py      # matplotlib charts
```
