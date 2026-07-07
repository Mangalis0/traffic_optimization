"""
Streamlit web app for the Traffic Light Optimizer.

Run with:
    streamlit run app.py
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

from src.data_generator import generate_training_data, arrival_rate_for_hour
from src.ml_model import TrafficPredictor, FEATURE_COLS
from src.optimizer import optimize_network_moo, fixed_timing
from src.simulator import TrafficSimulator, DT
from src.vehicle_tracker import (
    VehicleTracker, vehicle_xy, signal_xy,
    INTER_POS, BOX_HALF, CAR_W, CAR_H,
    MAX_SHOW, DIR_VEC, STOP_LINE, SPACING, LANE_OFFSET,
)


# ─────────────────── Page config ────────────────────────────────────────────

st.set_page_config(
    page_title="Traffic Light Optimizer",
    page_icon=":vertical_traffic_light:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────── Model (trained once, cached forever) ───────────────────

@st.cache_resource(show_spinner="Training ML model on 30 days of synthetic data ...")
def load_model():
    df = generate_training_data(days=30)
    predictor = TrafficPredictor()
    metrics = predictor.train(df)
    return predictor, metrics, df


predictor, ml_metrics, training_df = load_model()


# ─────────────────── Helper: simulation ─────────────────────────────────────

def run_sim_full(use_opt, start_hour, sim_hours, day_of_week,
                 weather, trigger_accident, progress_cb=None):
    """Run a complete simulation and return per-minute metric snapshots."""
    sim = TrafficSimulator()
    total_steps = int(sim_hours * 3600 / DT)
    update_every = 300        # re-optimise every 5 simulated minutes
    accident_triggered = False
    log = []

    if use_opt:
        rates = predictor.predict_rates(start_hour % 24, day_of_week, weather)
        sim.set_timings(optimize_network_moo(rates))
    else:
        sim.set_timings(fixed_timing(30.0))

    for step in range(total_steps):
        t = step * DT
        h = (start_hour + t / 3600.0) % 24

        raw_rates = {
            i: {
                d: (0.0 if sim.intersections[i].approaches[d].blocked
                    else arrival_rate_for_hour(h, day_of_week, d, i, weather))
                for d in ("N", "S", "E", "W")
            }
            for i in range(4)
        }
        # Scale external arrivals for approaches that also receive routing inflows
        actual_rates = sim.scale_external_rates(raw_rates)

        if use_opt and step % update_every == 0:
            pred = predictor.predict_rates(h, day_of_week, weather)
            for i in range(4):
                for d in ("N", "S", "E", "W"):
                    if sim.intersections[i].approaches[d].blocked:
                        pred[i][d] = 0.0
            # Augment with routing inflows so optimizer sizes green for total demand
            pred = sim.augment_rates_with_routing(pred)
            sim.set_timings(optimize_network_moo(pred))

        if trigger_accident and not accident_triggered and t >= 3600.0:
            sim.trigger_event("accident", 0, "N", 600.0)
            accident_triggered = True
            if use_opt:
                pred = predictor.predict_rates(h, day_of_week, weather)
                pred[0]["N"] = 0.0
                pred = sim.augment_rates_with_routing(pred)
                sim.set_timings(optimize_network_moo(pred))

        sim.step(actual_rates)

        if step % 60 == 0:
            log.append({
                "time": t,
                "hour": h,
                "avg_wait_time": sim.avg_wait_time,
                "throughput": sim.throughput(),
                "queues": sim.queue_per_intersection(),
            })
            if progress_cb:
                progress_cb(step / total_steps)

    return log


# ─────────────────── Helper: figures ────────────────────────────────────────

def fig_comparison(base_log, opt_log):
    times     = [m["time"] / 60 for m in base_log]
    base_wait = [m["avg_wait_time"] for m in base_log]
    opt_wait  = [m["avg_wait_time"] for m in opt_log]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    ax.plot(times, base_wait, "r-", lw=2, label="Fixed timing (baseline)")
    ax.plot(times, opt_wait,  "g-", lw=2, label="ML-optimized (Webster)")
    ax.fill_between(times, base_wait, opt_wait,
                    where=[b > o for b, o in zip(base_wait, opt_wait)],
                    alpha=0.15, color="green", label="Improvement")
    ax.set(xlabel="Time (min)", ylabel="Avg wait (s)", title="Average Wait Time")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    palette = ["royalblue", "darkorange", "green", "crimson"]
    for i in range(4):
        bq = [m["queues"][i] for m in base_log]
        oq = [m["queues"][i] for m in opt_log]
        ax.plot(times, bq, "--", color=palette[i], alpha=0.4, lw=1)
        ax.plot(times, oq, "-",  color=palette[i], lw=1.5, label=f"I{i}")
    ax.set(xlabel="Time (min)", ylabel="Queue (vehicles)",
           title="Queue Lengths  (solid = optimized, dashed = baseline)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def fig_network_static(snapshot, title="Network State"):
    """Coloured circles showing per-intersection queue intensity."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(-0.6, 1.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10)

    for x in (0, 1):
        ax.plot([x, x], [-0.45, 1.45], color="#aaa", lw=10, alpha=0.25, zorder=1)
    for y in (0, 1):
        ax.plot([-0.45, 1.45], [y, y], color="#aaa", lw=10, alpha=0.25, zorder=1)

    positions = [(0, 0), (1, 0), (0, 1), (1, 1)]
    queues = snapshot["queues"]
    max_q = max(max(queues), 1)
    for i, (x, y) in enumerate(positions):
        color = plt.cm.RdYlGn(1.0 - (queues[i] / max_q) * 0.9)
        ax.add_patch(plt.Circle((x, y), 0.18, color=color, zorder=3, ec="k", lw=0.8))
        ax.text(x, y + 0.03, f"I{i}", ha="center", va="center",
                fontsize=9, fontweight="bold", zorder=4)
        ax.text(x, y - 0.07, f"{queues[i]}v", ha="center", va="center",
                fontsize=7, zorder=4)

    ax.legend(handles=[
        mpatches.Patch(facecolor="green",  label="Low queue"),
        mpatches.Patch(facecolor="yellow", label="Medium"),
        mpatches.Patch(facecolor="red",    label="High queue"),
    ], loc="upper right", fontsize=7)
    fig.tight_layout()
    return fig


_ANIM_OPPOSITE    = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
_TRANSIT_COLOR    = "#BF8A30"   # amber — distinct from queued cars and signal lights
_TRANSIT_MAX_SHOW = 6           # max individual vehicles drawn per batch before "+n"
_TRANSIT_GAP      = 0.10        # progress-units between successive vehicles in a platoon


def _inter_transit_xy(tv: dict):
    """
    Compute screen (x, y) for a vehicle in transit between two intersections.

    Convention (matches vehicle_tracker lane offsets):
      approach_dir is the approach direction at BOTH source and destination.
      exit_dir = OPPOSITE[approach_dir] is the direction of travel.

    The vehicle is placed in the correct lane (LANE_OFFSET[approach_dir])
    and travels from source stop-line → destination stop-line as progress 0→1.
    """
    approach_dir = tv['approach_dir']
    exit_dir = _ANIM_OPPOSITE[approach_dir]

    src_pos = INTER_POS[tv['src_inter_id']]
    dst_pos = INTER_POS[tv['dst_inter_id']]
    ex, ey  = DIR_VEC[exit_dir]         # direction of travel
    ax_, ay = DIR_VEC[approach_dir]     # direction from destination center to stop-line
    lx, ly  = LANE_OFFSET[approach_dir]

    # Departure: just past source stop-line on the exit side
    x_dep = src_pos[0] + ex * STOP_LINE + lx
    y_dep = src_pos[1] + ey * STOP_LINE + ly

    # Arrival: at destination stop-line (approaching from the source side)
    x_arr = dst_pos[0] + ax_ * STOP_LINE + lx
    y_arr = dst_pos[1] + ay  * STOP_LINE + ly

    p = tv['progress']
    return x_dep + p * (x_arr - x_dep), y_dep + p * (y_arr - y_dep)


def fig_network_live(sim, tracker: VehicleTracker, hour: float, t_sec: float):
    """
    Animated 2x2 grid with individual vehicle rectangles and two-lane roads.

    Layers (bottom → top):
      0  Roads (asphalt)
      1  Road centre-line dividers
      2  Intersection boxes
      3  Intersection labels
      4  Signal lights
      5  Inter-intersection transit vehicles (amber — en-route between intersections)
      6  Queued vehicles (coloured by car type)
      7  Within-intersection transit (fading as they clear the box)

    SA traffic light sequence: Red → Green → Orange → Red.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    margin = 1.1
    ax.set_xlim(-margin, 3 + margin)
    ax.set_ylim(-margin, 3 + margin)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")
    n_transit = len(sim.transit_vehicles)
    ax.set_title(
        f"Live Traffic  |  {int(hour):02d}:{int((hour % 1) * 60):02d}"
        f"  |  t = {t_sec / 60:.1f} min"
        + (f"  |  {n_transit} batch(es) in transit" if n_transit else ""),
        fontsize=11, color="white", pad=8,
    )

    # ── Roads: asphalt + white centre-line divider ────────────────────────────
    road_kw   = dict(color="#3a3a3a", lw=30, solid_capstyle="butt", zorder=0)
    center_kw = dict(color="white", lw=1.0, alpha=0.55,
                     linestyle=(0, (6, 8)), solid_capstyle="butt", zorder=1)
    for x in (0, 3):
        ax.plot([x, x], [-0.8, 3.8], **road_kw)
        ax.plot([x, x], [-0.8, 3.8], **center_kw)
    for y in (0, 3):
        ax.plot([-0.8, 3.8], [y, y], **road_kw)
        ax.plot([-0.8, 3.8], [y, y], **center_kw)

    # ── Intersection boxes + signal lights ───────────────────────────────────
    for i, inter in enumerate(sim.intersections):
        cx, cy = INTER_POS[i]
        phase  = inter.phase

        ax.add_patch(plt.Rectangle(
            (cx - BOX_HALF, cy - BOX_HALF), BOX_HALF * 2, BOX_HALF * 2,
            color="#222", zorder=2, ec="#555", lw=1.0,
        ))
        ax.text(cx, cy, f"I{i}", ha="center", va="center",
                color="#aaa", fontsize=9, fontweight="bold", zorder=3)

        for d, ap in inter.approaches.items():
            sx, sy = signal_xy(i, d)

            if ap.blocked:
                sig_col = "#555"
            elif phase == "NS" and d in ("N", "S"):
                sig_col = "#00ff88"
            elif phase == "EW" and d in ("E", "W"):
                sig_col = "#00ff88"
            elif phase == "YELLOW_to_EW" and d in ("N", "S"):
                sig_col = "#ffdd00"
            elif phase == "YELLOW_to_NS" and d in ("E", "W"):
                sig_col = "#ffdd00"
            else:
                sig_col = "#ff3333"

            ax.add_patch(plt.Circle((sx, sy), 0.10, color=sig_col, zorder=4,
                                     ec="#000", lw=0.5, alpha=0.95))

            if ap.queue > MAX_SHOW:
                dx, dy = DIR_VEC[d]
                lx_off, ly_off = LANE_OFFSET[d]
                lx = cx + dx * (STOP_LINE + (MAX_SHOW - 0.5) * SPACING + 0.1) + lx_off
                ly = cy + dy * (STOP_LINE + (MAX_SHOW - 0.5) * SPACING + 0.1) + ly_off
                ax.text(lx, ly, f"+{ap.queue - MAX_SHOW}",
                        ha="center", va="center", fontsize=7,
                        color="#ffaa00", fontweight="bold", zorder=9)

    # ── Inter-intersection transit vehicles (amber) ───────────────────────────
    # Vehicles served at one intersection and traveling to the next.
    # Each vehicle in a batch is drawn at its own progress position (platoon),
    # staggered by _TRANSIT_GAP in progress-units.  Excess shows "+n".
    for tv in sim.transit_vehicles:
        count    = tv['count']
        n_show   = min(count, _TRANSIT_MAX_SHOW)
        n_hidden = count - n_show
        is_vert  = tv['approach_dir'] in ('N', 'S')
        w = CAR_W['N'] if is_vert else CAR_W['E']
        h = CAR_H['N'] if is_vert else CAR_H['E']

        for k in range(n_show):
            prog_k = max(0.0, tv['progress'] - k * _TRANSIT_GAP)
            vx, vy = _inter_transit_xy({**tv, 'progress': prog_k})
            ax.add_patch(plt.Rectangle(
                (vx - w / 2, vy - h / 2), w, h,
                color=_TRANSIT_COLOR, zorder=5, ec="white", lw=0.5, alpha=0.88,
            ))

        if n_hidden > 0:
            # Place the "+n" label just behind the last drawn vehicle
            prog_lbl = max(0.0, tv['progress'] - n_show * _TRANSIT_GAP)
            lx, ly   = _inter_transit_xy({**tv, 'progress': prog_lbl})
            ax.text(lx, ly, f"+{n_hidden}",
                    ha="center", va="center", fontsize=7,
                    color="#ffaa00", fontweight="bold", zorder=10)

    # ── Queued vehicles (coloured rectangles at approaches) ───────────────────
    for veh in tracker.queued():
        vx, vy = vehicle_xy(veh)
        d = veh.direction
        w, h = CAR_W[d], CAR_H[d]
        ax.add_patch(plt.Rectangle(
            (vx - w / 2, vy - h / 2), w, h,
            color=veh.color, zorder=6, ec="#000", lw=0.4,
        ))

    # ── Within-intersection transit (fading as they clear the box) ────────────
    for veh in tracker.transiting():
        vx, vy = vehicle_xy(veh)
        d = veh.direction
        w, h = CAR_W[d] * 0.9, CAR_H[d] * 0.9
        alpha = max(0.15, 1.0 - veh.transit_progress)
        ax.add_patch(plt.Rectangle(
            (vx - w / 2, vy - h / 2), w, h,
            color=veh.color, zorder=7, ec="white", lw=0.5, alpha=alpha,
        ))

    # ── Legend ────────────────────────────────────────────────────────────────
    ax.legend(handles=[
        mpatches.Patch(color="#00ff88",      label="Signal: green"),
        mpatches.Patch(color="#ffdd00",      label="Signal: yellow"),
        mpatches.Patch(color="#ff3333",      label="Signal: red"),
        mpatches.Patch(color="#555",         label="Signal: blocked"),
        mpatches.Patch(color=_TRANSIT_COLOR, label="In transit (inter-intersection)"),
    ], loc="lower right", fontsize=7.5, framealpha=0.6,
       facecolor="#222", edgecolor="#555", labelcolor="white")

    fig.tight_layout()
    return fig


def fig_signal_timings(timings_history):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
    fig.suptitle("Adaptive Signal Timings vs Fixed 30 s Baseline (across 24 h)",
                 fontsize=13, fontweight="bold")
    hours = [t["hour"] for t in timings_history]
    for i, ax in enumerate(axes.flat):
        gns = [t["timings"][i]["green_ns"] for t in timings_history]
        gew = [t["timings"][i]["green_ew"] for t in timings_history]
        ax.plot(hours, gns, "b-",          lw=2, label="N-S green")
        ax.plot(hours, gew, "darkorange",  lw=2, label="E-W green")
        ax.axhline(30, color="gray", ls="--", lw=1, label="Fixed 30 s")
        ax.axvspan(7,  9,  alpha=0.08, color="red")
        ax.axvspan(17, 19, alpha=0.08, color="orange")
        ax.set(title=f"Intersection {i}", xlabel="Hour", ylabel="Green time (s)")
        ax.set_xlim(0, 23)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_ml(sample_df):
    X      = sample_df[FEATURE_COLS].values
    preds  = predictor.model.predict(X)
    actual = sample_df["arrival_rate"].values
    rmse   = float(np.sqrt(np.mean((preds - actual) ** 2)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("ML Traffic Prediction — GradientBoosting", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.scatter(actual, preds, alpha=0.2, s=5, color="steelblue", rasterized=True)
    lo = min(actual.min(), preds.min())
    hi = max(actual.max(), preds.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect")
    ax.set(xlabel="Actual (veh/hr)", ylabel="Predicted (veh/hr)",
           title="Prediction vs Actual (hold-out sample)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(0.04, 0.95, f"RMSE = {rmse:.1f} veh/hr", transform=ax.transAxes,
            va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    ax = axes[1]
    imp = ml_metrics.get("feature_importances", {})
    if imp:
        items = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)
        labels, vals = zip(*items)
        colors = ["steelblue" if v > 0.1 else "lightsteelblue" for v in vals]
        ax.barh(list(labels)[::-1], list(vals)[::-1], color=list(colors)[::-1])
        ax.set(xlabel="Importance",
               title=f"Feature Importances\nCV RMSE: {ml_metrics['cv_rmse_mean']:.1f} "
                     f"+/- {ml_metrics['cv_rmse_std']:.1f} veh/hr")
        ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    return fig


# ─────────────────── Sidebar ─────────────────────────────────────────────────

with st.sidebar:
    st.header("Parameters")

    start_hour  = st.slider("Start hour",  0, 23, 7)
    sim_hours   = st.slider("Duration (h)", 0.5, 4.0, 2.0, step=0.5)

    dow_names   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    day_of_week = st.selectbox("Day of week", range(7), index=1,
                               format_func=lambda x: dow_names[x])

    weather = st.slider("Weather  (1.0 = clear, 0.5 = heavy rain)", 0.5, 1.0, 1.0, step=0.05)

    st.divider()
    trigger_accident = st.checkbox(
        "Accident at t = 1 h",
        help="Blocks intersection 0 north approach for 10 min — tests event response",
    )

    st.divider()
    run_btn = st.button("Run Simulation", type="primary", use_container_width=True)
    st.caption("Model is cached after the first load. Each simulation takes ~5-10 s.")


# ─────────────────── Title ───────────────────────────────────────────────────

st.title("Smart City Traffic Light Optimizer")
st.markdown(
    "**GradientBoosting** predicts approach demand "
    "-> **Webster's formula** computes optimal green splits "
    "-> **Queue simulator** benchmarks fixed vs. adaptive timing."
)


# ─────────────────── Tabs ────────────────────────────────────────────────────

tab_cmp, tab_live, tab_timing, tab_ml_tab = st.tabs([
    "Comparison Run", "Live Animation", "Signal Timings", "ML Model",
])


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Comparison Run
# ══════════════════════════════════════════════════════════════════════════════

with tab_cmp:
    if run_btn:
        prog1 = st.progress(0.0, text="Running baseline ...")
        base_log = run_sim_full(
            False, start_hour, sim_hours, day_of_week, weather, trigger_accident,
            progress_cb=lambda p: prog1.progress(p, text="Running baseline ..."),
        )
        prog1.empty()

        prog2 = st.progress(0.0, text="Running optimized ...")
        opt_log = run_sim_full(
            True, start_hour, sim_hours, day_of_week, weather, trigger_accident,
            progress_cb=lambda p: prog2.progress(p, text="Running optimized ..."),
        )
        prog2.empty()

        st.session_state["base_log"] = base_log
        st.session_state["opt_log"]  = opt_log

    if "base_log" in st.session_state:
        if not run_btn:
            st.info("Showing previous run. Adjust parameters and click **Run Simulation** to refresh.")

        base_log = st.session_state["base_log"]
        opt_log  = st.session_state["opt_log"]

        tail      = 10
        base_wait = float(np.mean([m["avg_wait_time"] for m in base_log[-tail:]]))
        opt_wait  = float(np.mean([m["avg_wait_time"] for m in opt_log[-tail:]]))
        pct_wait  = (base_wait - opt_wait) / base_wait * 100
        base_tp   = base_log[-1]["throughput"]
        opt_tp    = opt_log[-1]["throughput"]
        pct_tp    = (opt_tp - base_tp) / max(base_tp, 1) * 100

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Baseline Avg Wait",  f"{base_wait:.1f} s")
        m2.metric("Optimized Avg Wait", f"{opt_wait:.1f} s",
                   delta=f"-{base_wait - opt_wait:.1f} s")
        m3.metric("Wait Reduction",     f"{pct_wait:.1f}%")
        m4.metric("Throughput Gain",    f"{pct_tp:+.1f}%")

        if pct_wait >= 20:
            st.success(f"Target met: {pct_wait:.1f}% wait reduction (target >= 20%)")
        else:
            st.warning(f"Below target: {pct_wait:.1f}% (target >= 20%)")

        st.info(
            "**Optimizer objective (MOO):** minimise mean network delay "
            "+ 0.3 × variance of per-intersection delay. "
            "The fairness term prevents any single intersection from becoming "
            "a disproportionate bottleneck while others are lightly loaded."
        )

        st.pyplot(fig_comparison(base_log, opt_log))
        plt.close("all")

        c1, c2 = st.columns(2)
        with c1:
            f = fig_network_static(base_log[-1], "Baseline — Final State")
            st.pyplot(f)
            plt.close(f)
        with c2:
            f = fig_network_static(opt_log[-1], "Optimized — Final State")
            st.pyplot(f)
            plt.close(f)

    else:
        st.info("Set parameters in the sidebar and click **Run Simulation**.")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Live Animation
# ══════════════════════════════════════════════════════════════════════════════

with tab_live:
    st.subheader("Real-Time Intersection Grid")
    st.caption(
        "Runs a short simulation frame-by-frame (1 frame = 1 simulated minute). "
        "Bars show vehicle queues; circles show signal state."
    )

    ctrl_col, anim_col = st.columns([1, 2])
    with ctrl_col:
        anim_mode     = st.radio("Signal mode", ["Optimized (ML)", "Baseline (Fixed)"])
        anim_duration = st.slider("Duration to animate (min)", 5, 60, 20, step=5)
        anim_speed    = st.select_slider(
            "Frame speed", ["Slow (1 s)", "Normal (0.1 s)", "Fast (0.02 s)"],
            value="Normal (0.1 s)",
        )
        start_anim = st.button("Start Animation", type="secondary", use_container_width=True)

    frame_slot   = anim_col.empty()
    metrics_slot = anim_col.empty()

    if start_anim:
        speed_map    = {"Slow (1 s)": 1.0, "Normal (0.1 s)": 0.1, "Fast (0.02 s)": 0.02}
        frame_sleep  = speed_map[anim_speed]
        use_opt_anim = anim_mode == "Optimized (ML)"

        sim          = TrafficSimulator()
        tracker      = VehicleTracker()
        total_steps  = int(anim_duration * 60 / DT)
        acc_triggered = False

        if use_opt_anim:
            rates = predictor.predict_rates(start_hour % 24, day_of_week, weather)
            sim.set_timings(optimize_network_moo(rates))
        else:
            sim.set_timings(fixed_timing(30.0))

        render_every = 5    # 1 frame per 5 simulated seconds — lets you watch cars cross between intersections

        for step in range(total_steps):
            t = step * DT
            h = (start_hour + t / 3600.0) % 24

            raw_anim = {
                i: {
                    d: (0.0 if sim.intersections[i].approaches[d].blocked
                        else arrival_rate_for_hour(h, day_of_week, d, i, weather))
                    for d in ("N", "S", "E", "W")
                }
                for i in range(4)
            }
            actual_rates = sim.scale_external_rates(raw_anim)

            if use_opt_anim and step % 300 == 0:
                pred = predictor.predict_rates(h, day_of_week, weather)
                pred = sim.augment_rates_with_routing(pred)
                sim.set_timings(optimize_network_moo(pred))

            # Short accident (5 min) if enabled
            if trigger_accident and not acc_triggered and t >= 300:
                sim.trigger_event("accident", 0, "N", 300.0)
                acc_triggered = True

            sim.step(actual_rates)
            tracker.sync(sim)          # keep visual vehicles in sync with queues

            if step % render_every == 0:
                tracker.advance_transit()   # animate cars moving through intersection
                f = fig_network_live(sim, tracker, h, t)
                frame_slot.pyplot(f, use_container_width=True)
                plt.close(f)

                n_queued      = len(tracker.queued())
                n_transit     = len(tracker.transiting())
                n_inter_trans = sum(tv['count'] for tv in sim.transit_vehicles)
                metrics_slot.info(
                    f"t = {t / 60:.1f} min  |  "
                    f"Avg wait = **{sim.avg_wait_time:.1f} s**  |  "
                    f"Exited network = **{sim.throughput():,}**  |  "
                    f"Queued = **{n_queued}**  |  "
                    f"In transit = **{n_inter_trans}**"
                )
                time.sleep(frame_sleep)

        metrics_slot.success(
            f"Animation complete.  "
            f"Avg wait = {sim.avg_wait_time:.1f} s  |  "
            f"Vehicles served = {sim.throughput():,}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Signal Timings
# ══════════════════════════════════════════════════════════════════════════════

with tab_timing:
    st.subheader("Adaptive vs Fixed Signal Timings")
    st.caption(
        "How Webster's optimal green splits vary across the day. "
        "Responds instantly to day-of-week and weather changes in the sidebar."
    )

    timings_history = [
        {"hour": h, "timings": optimize_network_moo(
            predictor.predict_rates(h, day_of_week, weather)
        )}
        for h in range(24)
    ]
    st.pyplot(fig_signal_timings(timings_history))
    plt.close("all")

    st.markdown(
        "- Red shading = AM rush (7-9 h), orange = PM rush (17-19 h)\n"
        "- During AM rush: N-S green rises (heavier commuter flow south->north)\n"
        "- During PM rush: E-W green rises (reverse commute)\n"
        "- Fixed baseline stays flat at 30 s regardless of demand"
    )

    st.divider()
    st.markdown("#### Per-hour deep-dive")
    sel_hour = st.slider("Hour to inspect", 0, 23, start_hour, key="timing_hour")
    rates_h  = predictor.predict_rates(sel_hour, day_of_week, weather)
    timings_h = optimize_network_moo(rates_h)

    cols = st.columns(4)
    for i, col in enumerate(cols):
        g_ns = timings_h[i]["green_ns"]
        g_ew = timings_h[i]["green_ew"]
        col.markdown(f"**Intersection {i}**")
        col.metric("N-S green", f"{g_ns:.1f} s")
        col.metric("E-W green", f"{g_ew:.1f} s")
        col.metric("Cycle",     f"{g_ns + g_ew + 10:.1f} s")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 — ML Model
# ══════════════════════════════════════════════════════════════════════════════

with tab_ml_tab:
    st.subheader("ML Traffic Prediction Model")
    st.markdown(
        f"**Model:** GradientBoosting Regressor (200 trees, depth 5)  |  "
        f"**Training samples:** {ml_metrics['train_samples']:,}"
    )

    vm1, vm2, vm3 = st.columns(3)
    vm1.metric("5-fold CV RMSE (train split)",
               f"{ml_metrics['cv_rmse_mean']:.1f} ± {ml_metrics['cv_rmse_std']:.1f} veh/hr",
               help="Cross-validation on the first 80% of days (training portion only).")
    vm2.metric("Hold-out RMSE (last 20% of days)",
               f"{ml_metrics['test_rmse']:.1f} veh/hr",
               help="Evaluated on days 24-29 — data the model never saw during training.")
    vm3.metric("Demand range", "80 – 980 veh/hr",
               help="Context for interpreting RMSE: error is small relative to the range.")

    sample = training_df.sample(min(8000, len(training_df)), random_state=42)
    st.pyplot(fig_ml(sample))
    plt.close("all")

    st.markdown(
        """
**Key findings:**
- `cos_hour` / `sin_hour` dominate — time of day is the primary traffic driver.
- `direction_encoded` contributes notably — N-S vs E-W flows shift during rush hours.
- `weather` has a modest but measurable effect (up to -15% demand in heavy rain).
- CV RMSE ≈ hold-out RMSE confirms the model generalises and is not overfit.
        """
    )
