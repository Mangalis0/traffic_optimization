from src.ml_model import TrafficPredictor
from src.data_generator import generate_training_data
from src.optimizer import optimize_network_moo, SATURATION_FLOW

df = generate_training_data(days=30)
p = TrafficPredictor()
p.train(df)

for h in [0, 7, 8, 12, 17, 22]:
    rates = p.predict_rates(h, 1, 1.0)
    t = optimize_network_moo(rates)
    gns = t[0]['green_ns']
    gew = t[0]['green_ew']
    print(f"Hour {h:2d}: I0 g_ns={gns:5.1f}  g_ew={gew:5.1f}  cycle={gns+gew+10:.1f}")
