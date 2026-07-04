# Starter Code - README

## Quick Start

Each problem has its own directory with starter code to reduce setup friction:

### Problem 1: Traffic Light Optimization 🚦
```bash
cd starter_code/problem1_traffic
python traffic_simulator.py
```

**What's provided:**
- `traffic_simulator.py` - Core simulation framework
- Basic data structures (Vehicle, Intersection)
- Synthetic data generator
- Placeholder for ML and optimization

**What you need to implement:**
- ML model for traffic prediction
- Optimization algorithm for light timing
- Visualization (matplotlib, plotly, or custom)
- Benchmarking against fixed-timing baseline

---

### Problem 2: Fantasy Sports Optimizer ⚽
```bash
cd starter_code/problem2_fantasy
python fantasy_optimizer.py
```

**What's provided:**
- `fantasy_optimizer.py` - Team selection framework
- Data loader (synthetic + API integration stubs)
- Player and constraint data structures
- Baseline strategies to beat

**What you need to implement:**
- ML model for player performance prediction
- Optimization solver (integer programming)
- Interactive dashboard/tool
- Validation against baselines

---

### Problem 3: Warehouse Robot Coordination 🤖
```bash
cd starter_code/problem3_warehouse
python warehouse_simulator.py
```

**What's provided:**
- `warehouse_simulator.py` - Robot fleet simulation
- Grid-based warehouse environment
- Basic robot and order data structures
- Greedy task assignment baseline

**What you need to implement:**
- A* pathfinding algorithm
- Multi-robot collision avoidance
- Optimal task assignment
- Demand forecasting ML model
- Visualization

---

## Installation

All problems use standard Python libraries. Install dependencies:

```bash
pip install -r requirements.txt
```

**Core dependencies:**
- numpy, pandas, scikit-learn (ML)
- scipy, cvxpy, or ortools (optimization)
- matplotlib, plotly, or dash (visualization)

**Optional:**
- requests (for API calls in Problem 2)
- streamlit (for interactive dashboards)
- pygame (for fancy visualizations)

---

## Tips

### Use the Starter Code or Not?
- **Use it if**: You want to save time on boilerplate
- **Don't use it if**: You prefer your own architecture

The starter code is meant to help, not constrain. Feel free to:
- Modify it heavily
- Cherry-pick useful parts
- Ignore it completely and start fresh

### What NOT to Waste Time On
- Perfect OOP design (functional is fine)
- Extensive unit tests (working demo > test coverage)
- Type hints everywhere (helpful but not required)
- Documentation beyond basic README

### What TO Focus On
- **Working demo** that shows your solution
- **Clear thinking** in your approach
- **Good results** on the problem metrics
- **Explainability** of your methods

---

## Common Pitfalls

### Problem 1 (Traffic)
- ❌ Spending too long on realistic physics simulation
- ✅ Simple grid-based movement is fine
- ❌ Over-complex ML models
- ✅ Simple regression/classification works

### Problem 2 (Fantasy)
- ❌ Getting stuck on API rate limits
- ✅ Use synthetic data if APIs are painful
- ❌ Perfect predictions
- ✅ Beat baselines, show improvement

### Problem 3 (Warehouse)
- ❌ Implementing fancy RL from scratch
- ✅ A* + good task assignment gets you far
- ❌ Complex 3D visualization
- ✅ Simple 2D grid is sufficient

---

## Example Workflow

**Phase 1 (3 hours):**
1. Choose your problem
2. Run starter code, understand structure
3. Sketch your approach on paper
4. Set up development environment

**Phase 2 (6 hours):**
1. Implement ML component (3 hours)
2. Implement optimization component (3 hours)
3. Basic integration test

**Phase 3 (6 hours):**
1. Integration and debugging (2 hours)
2. Visualization (2 hours)
3. Benchmarking and metrics (1 hour)
4. Polish demo (1 hour)

**Phase 4 (2 hours):**
1. Final testing
2. Prepare presentation
3. Practice demo

---

## Need Help?

### Debugging
- Use print statements liberally
- Visualize intermediate steps
- Start simple, add complexity gradually

### Resources
- Documentation is your friend (not copying code)
- Stack Overflow for specific errors
- ChatGPT for boilerplate (but understand it!)

---

## Good Luck! 🚀

Remember: We want to see how you think and solve problems, not perfect solutions. Have fun with it!
