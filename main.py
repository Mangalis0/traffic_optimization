"""
Smart City Traffic Light Optimization
======================================
Entry point.  Run from this directory:

    python main.py                    # 2-hour morning rush, no accident
    python main.py --hours 1          # faster demo
    python main.py --accident         # trigger accident at 1-hour mark
    python main.py --start-hour 17    # evening rush
    python main.py --no-plots         # headless / CI mode
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List

import numpy as np

from src.data_generator import generate_training_data, arrival_rate_for_hour
from src.ml_model import TrafficPredictor, FEATURE_COLS
from src.optimizer import optimize_network_moo, fixed_timing
from src.simulator import TrafficSimulator, DT
from src.visualization import (
    plot_ml_performance,
    plot_signal_timings,
    plot_simulation_results,
)


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def run_simulation(
    use_optimization: bool,
    predictor: TrafficPredictor,
    sim_hours: float,
    start_hour: float,
    day_of_week: int,
    weather: float,
    trigger_accident: bool,
    verbose: bool,
) -> List[Dict]:
    """
    Run one simulation scenario and return per-minute metric snapshots.

    Args:
        use_optimization: True → ML-adaptive timings via MOO; False → fixed 30-30
        predictor:        trained TrafficPredictor
        sim_hours:        how many simulated hours to run
        start_hour:       hour-of-day at t=0  (0–23)
        day_of_week:      0=Monday … 6=Sunday
        weather:          1.0=clear, 0.5=heavy rain
        trigger_accident: if True, block intersection-0 north approach at t=1 h
        verbose:          print progress every 30 min
    """
    sim = TrafficSimulator(num_intersections=4)
    total_steps = int(sim_hours * 3600 / DT)
    update_every = 300   # re-optimise every 5 min
    log_every = 60       # log metrics every 60 s
    accident_triggered = False
    metrics_log: List[Dict] = []

    current_hour = start_hour
    if use_optimization:
        rates = predictor.predict_rates(current_hour % 24, day_of_week, weather)
        timings = optimize_network_moo(rates)
    else:
        timings = fixed_timing(green_time=30.0)
    sim.set_timings(timings)

    for step in range(total_steps):
        t = step * DT
        current_hour = start_hour + t / 3600.0

        # Ground-truth arrival rates (scale down for approaches that receive routing
        # inflows so total flow stays roughly conserved)
        actual_rates: Dict[int, Dict[str, float]] = {}
        for inter_id in range(4):
            actual_rates[inter_id] = {}
            for direction in ('N', 'S', 'E', 'W'):
                blocked = sim.intersections[inter_id].approaches[direction].blocked
                rate = (
                    0.0
                    if blocked
                    else arrival_rate_for_hour(
                        current_hour % 24, day_of_week, direction, inter_id, weather
                    )
                )
                actual_rates[inter_id][direction] = rate
        actual_rates = sim.scale_external_rates(actual_rates)

        # Re-optimise every 5 simulated minutes
        if use_optimization and (step % update_every == 0):
            predicted = predictor.predict_rates(current_hour % 24, day_of_week, weather)
            for inter_id in range(4):
                for d in ('N', 'S', 'E', 'W'):
                    if sim.intersections[inter_id].approaches[d].blocked:
                        predicted[inter_id][d] = 0.0
            # Augment with estimated routing inflows so the optimizer sizes
            # green phases for total demand, not just external arrivals.
            predicted = sim.augment_rates_with_routing(predicted)
            sim.set_timings(optimize_network_moo(predicted))

        # Accident event at 1-hour mark
        if trigger_accident and not accident_triggered and t >= 3600.0:
            sim.trigger_event('accident', intersection_id=0, direction='N', duration=600.0)
            accident_triggered = True
            if verbose:
                print(
                    f'  [EVENT t={t/60:.0f}min] Accident: intersection 0 north approach'
                    ' blocked for 10 min'
                )
            if use_optimization:
                predicted = predictor.predict_rates(current_hour % 24, day_of_week, weather)
                predicted[0]['N'] = 0.0
                predicted = sim.augment_rates_with_routing(predicted)
                sim.set_timings(optimize_network_moo(predicted))

        sim.step(actual_rates)

        if step % log_every == 0:
            metrics_log.append({
                'time': t,
                'hour': current_hour % 24,
                'avg_wait_time': sim.avg_wait_time,
                'throughput': sim.throughput(),
                'queues': sim.queue_per_intersection(),
            })

        if verbose and step % 1800 == 0:
            print(
                f'  t={t/60:5.1f} min | avg_wait={sim.avg_wait_time:5.1f}s'
                f' | served={sim.throughput():6d}'
            )

    return metrics_log



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Smart Traffic Light Optimization — Problem 1',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--days',        type=int,   default=30,  help='Training data days')
    p.add_argument('--hours',       type=float, default=2.0, help='Simulation duration (hours)')
    p.add_argument('--start-hour',  type=float, default=7.0, help='Simulation start hour (0-23)')
    p.add_argument('--day-of-week', type=int,   default=1,   help='Day of week (0=Mon, 6=Sun)')
    p.add_argument('--weather',     type=float, default=1.0, help='Weather (1=clear, 0.5=rain)')
    p.add_argument('--accident',    action='store_true',     help='Trigger accident at 1-hour mark')
    p.add_argument('--no-plots',    action='store_true',     help='Skip visualization output')
    p.add_argument('--quiet',       action='store_true',     help='Suppress per-step output')
    return p.parse_args()


def main():
    args = parse_args()
    verbose = not args.quiet

    banner = '=' * 62
    print(banner)
    print('  Smart City Traffic Light Optimization')
    print('  ML (GradientBoosting) + Math Opt (Webster + scipy MOO)')
    print(banner)

    # ------------------------------------------------------------------
    # 1. Generate training data
    # ------------------------------------------------------------------
    print(f'\n[1/4] Generating {args.days}-day synthetic traffic dataset ...')
    t0 = time.time()
    df = generate_training_data(days=args.days)
    print(f'  {len(df):,} records generated in {time.time()-t0:.1f}s')

    # ------------------------------------------------------------------
    # 2. Train ML model
    # ------------------------------------------------------------------
    print('\n[2/4] Training GradientBoosting predictor ...')
    t0 = time.time()
    predictor = TrafficPredictor()
    metrics_ml = predictor.train(df)
    print(
        f'  CV RMSE  (train split): {metrics_ml["cv_rmse_mean"]:.1f}'
        f' ± {metrics_ml["cv_rmse_std"]:.1f} veh/hr'
    )
    print(
        f'  Hold-out RMSE (test days): {metrics_ml["test_rmse"]:.1f} veh/hr'
        f'  (trained in {time.time()-t0:.1f}s)'
    )
    top3 = sorted(metrics_ml['feature_importances'].items(), key=lambda x: -x[1])[:3]
    print(f'  Top-3 features: {[(f, f"{v:.3f}") for f, v in top3]}')

    # ------------------------------------------------------------------
    # 3. Run simulations
    # ------------------------------------------------------------------
    print(
        f'\n[3/4] Running simulations ({args.hours}h starting at '
        f'{int(args.start_hour):02d}:00) ...'
    )
    if args.accident:
        print('  Accident event enabled (intersection 0, north approach, at t=1h)')

    sim_kwargs = dict(
        predictor=predictor,
        sim_hours=args.hours,
        start_hour=args.start_hour,
        day_of_week=args.day_of_week,
        weather=args.weather,
        trigger_accident=args.accident,
        verbose=verbose,
    )

    print('\n  --- BASELINE (fixed 30-30 timing) ---')
    base_log = run_simulation(use_optimization=False, **sim_kwargs)

    print('\n  --- OPTIMIZED (ML + MOO Webster) ---')
    opt_log = run_simulation(use_optimization=True, **sim_kwargs)

    # ------------------------------------------------------------------
    # 4. Results
    # ------------------------------------------------------------------
    print('\n[4/4] Results')
    tail = 10
    base_wait = float(np.mean([m['avg_wait_time'] for m in base_log[-tail:]]))
    opt_wait  = float(np.mean([m['avg_wait_time'] for m in opt_log[-tail:]]))
    pct_wait  = (base_wait - opt_wait) / base_wait * 100

    base_tp = base_log[-1]['throughput']
    opt_tp  = opt_log[-1]['throughput']
    pct_tp  = (opt_tp - base_tp) / max(base_tp, 1) * 100

    col = 30
    print()
    print(f"  {'Metric':<{col}} {'Baseline':>10} {'Optimized':>10} {'Change':>10}")
    print(f"  {'-'*(col+33)}")
    print(f"  {'Avg wait time (s)':<{col}} {base_wait:>10.1f} {opt_wait:>10.1f} {pct_wait:>9.1f}%")
    print(f"  {'Total vehicles served':<{col}} {base_tp:>10} {opt_tp:>10} {pct_tp:>9.1f}%")
    print()
    status = 'PASS' if pct_wait >= 20 else 'BELOW TARGET'
    print(f'  [{status}] Wait time reduction = {pct_wait:.1f}%  (target >= 20%)')
    print(f'  [INFO] Optimizer objective: mean delay + 0.3 × delay-variance across intersections'
          f' (fairness term keeps no intersection a disproportionate bottleneck)')

    if not args.no_plots:
        print('\n  Generating plots ...')

        sample = df.sample(min(8000, len(df)), random_state=1)
        preds = predictor.model.predict(sample[FEATURE_COLS].values)
        plot_ml_performance(metrics_ml, preds, sample['arrival_rate'].values)

        timings_history = []
        for h in range(24):
            rates = predictor.predict_rates(h, args.day_of_week, args.weather)
            timings_history.append({'hour': h, 'timings': optimize_network_moo(rates)})
        plot_signal_timings(timings_history)

        sim_label = (
            f'{args.hours}h sim | start={int(args.start_hour):02d}:00 | '
            f'weather={args.weather:.1f}'
        )
        if args.accident:
            sim_label += ' | accident at t=1h'
        plot_simulation_results(
            base_log, opt_log,
            title=f'Traffic Light Optimization — {sim_label}',
        )

    print('\nDone.')


if __name__ == '__main__':
    main()
