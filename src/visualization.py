"""
Visualization helpers.

Four charts produced:
1. ml_performance.png      — prediction scatter + feature importances
2. signal_timings.png      — how green splits vary across the day
3. results.png             — main comparison (wait times, queues, network state)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import Dict, List


# ---------------------------------------------------------------------------
# ML performance
# ---------------------------------------------------------------------------

def plot_ml_performance(
    train_metrics: Dict,
    predictions: np.ndarray,
    actuals: np.ndarray,
    save_path: str = 'ml_performance.png',
):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('ML Traffic Prediction — GradientBoosting', fontsize=13, fontweight='bold')

    # --- Prediction vs actual scatter ---
    ax = axes[0]
    ax.scatter(actuals, predictions, alpha=0.25, s=6, color='steelblue', rasterized=True)
    lo = min(actuals.min(), predictions.min())
    hi = max(actuals.max(), predictions.max())
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='Perfect')
    ax.set_xlabel('Actual arrival rate (veh/hr)')
    ax.set_ylabel('Predicted arrival rate (veh/hr)')
    ax.set_title('Prediction vs Actual (hold-out sample)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    rmse = float(np.sqrt(np.mean((predictions - actuals) ** 2)))
    ax.text(0.04, 0.95, f'RMSE = {rmse:.1f} veh/hr', transform=ax.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # --- Feature importances ---
    ax = axes[1]
    imp = train_metrics.get('feature_importances', {})
    if imp:
        sorted_imp = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)
        labels, vals = zip(*sorted_imp)
        colors = ['steelblue' if v > 0.1 else 'lightsteelblue' for v in vals]
        ax.barh(labels[::-1], vals[::-1], color=colors[::-1])
        ax.set_xlabel('Feature Importance')
        ax.set_title(
            f"Feature Importances\nCV RMSE: {train_metrics['cv_rmse_mean']:.1f}"
            f" ± {train_metrics['cv_rmse_std']:.1f} veh/hr"
        )
        ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f'  Saved {save_path}')
    plt.show()


# ---------------------------------------------------------------------------
# Signal timings over 24 h
# ---------------------------------------------------------------------------

def plot_signal_timings(
    timings_history: List[Dict],
    save_path: str = 'signal_timings.png',
):
    """timings_history = [{'hour': h, 'timings': {0: {'green_ns': g, 'green_ew': g}, ...}}, ...]"""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    fig.suptitle('Adaptive Signal Timings vs Fixed Baseline (by Intersection)', fontsize=13, fontweight='bold')

    hours = [t['hour'] for t in timings_history]

    for i, ax in enumerate(axes.flat):
        gns = [t['timings'][i]['green_ns'] for t in timings_history]
        gew = [t['timings'][i]['green_ew'] for t in timings_history]
        ax.plot(hours, gns, 'b-', linewidth=2, label='N-S green (opt)')
        ax.plot(hours, gew, 'darkorange', linewidth=2, label='E-W green (opt)')
        ax.axhline(30, color='gray', linestyle='--', linewidth=1, label='Fixed 30 s')
        ax.set_title(f'Intersection {i}')
        ax.set_xlabel('Hour of day')
        ax.set_ylabel('Green time (s)')
        ax.set_xlim(0, 23)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        # Shade rush hours
        ax.axvspan(7, 9, alpha=0.08, color='red', label='AM rush')
        ax.axvspan(17, 19, alpha=0.08, color='orange')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f'  Saved {save_path}')
    plt.show()


# ---------------------------------------------------------------------------
# Main comparison results
# ---------------------------------------------------------------------------

def plot_simulation_results(
    baseline_metrics: List[Dict],
    optimized_metrics: List[Dict],
    title: str = 'Traffic Light Optimization Results',
    save_path: str = 'results.png',
):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    times = [m['time'] / 60 for m in baseline_metrics]   # → minutes

    base_wait = [m['avg_wait_time'] for m in baseline_metrics]
    opt_wait  = [m['avg_wait_time'] for m in optimized_metrics]

    # --- 1. Wait time comparison ---
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(times, base_wait, 'r-',  linewidth=2, label='Fixed timing (baseline)')
    ax1.plot(times, opt_wait,  'g-',  linewidth=2, label='ML-optimized (Webster)')
    ax1.fill_between(
        times, base_wait, opt_wait,
        where=[b > o for b, o in zip(base_wait, opt_wait)],
        alpha=0.15, color='green', label='Improvement',
    )
    ax1.set_xlabel('Simulation time (min)')
    ax1.set_ylabel('Avg wait time (s)')
    ax1.set_title('Average Vehicle Wait Time')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- 2. Summary text box ---
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    tail = 10
    final_base = float(np.mean(base_wait[-tail:])) if len(base_wait) >= tail else base_wait[-1]
    final_opt  = float(np.mean(opt_wait[-tail:])) if len(opt_wait) >= tail else opt_wait[-1]
    pct_wait   = (final_base - final_opt) / final_base * 100

    base_tp = baseline_metrics[-1]['throughput']
    opt_tp  = optimized_metrics[-1]['throughput']
    pct_tp  = (opt_tp - base_tp) / max(base_tp, 1) * 100

    status = 'PASS ✓' if pct_wait >= 20 else 'PARTIAL'
    summary = (
        f"PERFORMANCE SUMMARY\n"
        f"{'─' * 28}\n"
        f"Baseline avg wait:  {final_base:6.1f} s\n"
        f"Optimized avg wait: {final_opt:6.1f} s\n"
        f"Wait reduction:     {pct_wait:6.1f}%\n\n"
        f"Baseline throughput: {base_tp:6d} veh\n"
        f"Optimized throughput:{opt_tp:6d} veh\n"
        f"Throughput gain:    {pct_tp:6.1f}%\n\n"
        f"{'─' * 28}\n"
        f"Target ≥20% wait reduction\n"
        f"Result: {status}"
    )
    ax2.text(
        0.05, 0.97, summary,
        transform=ax2.transAxes, fontsize=9.5,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.85),
    )

    # --- 3. Queue lengths ---
    ax3 = fig.add_subplot(gs[1, :2])
    colors = ['royalblue', 'darkorange', 'green', 'crimson']
    for i in range(4):
        bq = [m['queues'][i] for m in baseline_metrics]
        oq = [m['queues'][i] for m in optimized_metrics]
        ax3.plot(times, bq, '--', color=colors[i], alpha=0.45, linewidth=1)
        ax3.plot(times, oq, '-',  color=colors[i], linewidth=1.5, label=f'I{i}')
    ax3.set_xlabel('Simulation time (min)')
    ax3.set_ylabel('Total queue (vehicles)')
    ax3.set_title('Queue Lengths per Intersection (solid=optimized, dashed=baseline)')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # --- 4. Network state (final optimised snapshot) ---
    ax4 = fig.add_subplot(gs[1, 2])
    _draw_network_state(ax4, optimized_metrics[-1])

    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f'  Saved {save_path}')
    plt.show()

    return pct_wait


def _draw_network_state(ax, snapshot: Dict):
    """Draw 2×2 grid coloured by queue intensity."""
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(-0.6, 1.6)
    ax.set_aspect('equal')
    ax.set_title('Network State\n(final optimized step)')
    ax.axis('off')

    positions = [(0, 0), (1, 0), (0, 1), (1, 1)]

    # Roads
    for x in (0, 1):
        ax.plot([x, x], [-0.45, 1.45], color='#aaa', linewidth=10, alpha=0.25, zorder=1)
    for y in (0, 1):
        ax.plot([-0.45, 1.45], [y, y], color='#aaa', linewidth=10, alpha=0.25, zorder=1)

    queues = snapshot.get('queues', [0, 0, 0, 0])
    max_q = max(max(queues), 1)

    for i, (x, y) in enumerate(positions):
        intensity = queues[i] / max_q
        color = plt.cm.RdYlGn(1.0 - intensity * 0.9)
        circle = plt.Circle((x, y), 0.18, color=color, zorder=3, ec='black', lw=0.8)
        ax.add_patch(circle)
        ax.text(x, y + 0.02, f'I{i}', ha='center', va='center',
                fontsize=9, fontweight='bold', zorder=4)
        ax.text(x, y - 0.07, f'{queues[i]}v', ha='center', va='center',
                fontsize=7, zorder=4, color='#222')

    legend_patches = [
        mpatches.Patch(facecolor='green',  label='Low queue'),
        mpatches.Patch(facecolor='yellow', label='Medium queue'),
        mpatches.Patch(facecolor='red',    label='High queue'),
    ]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=7, framealpha=0.8)
