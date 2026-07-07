"""Diagnose queue balance and optimizer assignments at 7am rush."""
from src.data_generator import generate_training_data, arrival_rate_for_hour
from src.ml_model import TrafficPredictor
from src.optimizer import optimize_network_moo, fixed_timing, SATURATION_FLOW
from src.simulator import TrafficSimulator, DT, ROUTING_FRACTION

df = generate_training_data(days=30)
p = TrafficPredictor()
p.train(df)

sim = TrafficSimulator()

# Show what optimizer sees after correct scale+augment
pred_raw = p.predict_rates(7, 1, 1.0)
pred_scaled = sim.scale_external_rates(pred_raw)
pred_aug = sim.augment_rates_with_routing(pred_scaled)

print("=== Optimizer input (scale + augment) at 7am ===")
t = optimize_network_moo(pred_aug)
for i in range(4):
    q_ns = max(pred_aug[i]['N'], pred_aug[i]['S'])
    q_ew = max(pred_aug[i]['E'], pred_aug[i]['W'])
    Y = (q_ns + q_ew) / SATURATION_FLOW
    gns = t[i]['green_ns']
    gew = t[i]['green_ew']
    C = gns + gew + 10
    cap_ns = SATURATION_FLOW * gns / C
    cap_ew = SATURATION_FLOW * gew / C
    print(f"  I{i}: q_ns={q_ns:5.0f}  q_ew={q_ew:5.0f}  Y={Y:.3f}  "
          f"g_ns={gns:5.1f} g_ew={gew:5.1f} C={C:5.1f}  "
          f"cap_ns={cap_ns:5.0f} cap_ew={cap_ew:5.0f}")
    surplus_ns = (cap_ns - q_ns)
    surplus_ew = (cap_ew - q_ew)
    flag = " ** OVERSATURATED **" if surplus_ns < 0 or surplus_ew < 0 else ""
    print(f"         surplus_ns={surplus_ns:+.0f}  surplus_ew={surplus_ew:+.0f}{flag}")

print()

# Run 30-min simulation and report per-approach queues
sim2 = TrafficSimulator()
sim2.set_timings(t)

for step in range(int(0.5 * 3600 / DT)):
    h = 7 + step * DT / 3600
    raw = {i: {d: arrival_rate_for_hour(h, 1, d, i) for d in "NSEW"} for i in range(4)}
    actual = sim2.scale_external_rates(raw)

    if step % 300 == 0:
        pred_r = p.predict_rates(h, 1, 1.0)
        pred_r = sim2.scale_external_rates(pred_r)
        pred_r = sim2.augment_rates_with_routing(pred_r)
        sim2.set_timings(optimize_network_moo(pred_r))

    sim2.step(actual)

print("=== Per-approach queues after 30 min (optimized) ===")
total = 0
for i, inter in enumerate(sim2.intersections):
    row = "  I{}: ".format(i)
    for d in "NSEW":
        q = inter.approaches[d].queue
        total += q
        row += f"{d}={q:4d} "
    print(row + f"  total={sum(ap.queue for ap in inter.approaches.values())}")
print(f"  Network total: {total}")
