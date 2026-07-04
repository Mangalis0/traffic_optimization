# Case Study Problem Options

**Instructions**: Choose ONE of the three problems below to complete over the weekend (Friday → Tuesday). You have 10-15 hours to build a working solution. Present your solution in 15 minutes.

**What We're Testing**:
- **ML Modeling** (33%): Feature engineering, model selection, validation
- **Mathematical Optimization** (33%): Constraint handling, objective optimization, algorithm design
- **Software Engineering** (33%): Clean code, scalable architecture, working demo

**Deliverables**:
1. Working solution (code + demo)
2. Brief README explaining your approach
3. 15-minute presentation showing it works

---

## Problem 1: Smart City Traffic Light Optimization 🚦

### The Challenge
Design an intelligent traffic management system that optimizes traffic flow through a city intersection network during rush hour.

### Your Mission
Build a system that:
1. **Simulates** realistic traffic patterns (vehicles arriving, traveling, waiting)
2. **Predicts** traffic congestion using ML (based on historical patterns, time of day, weather)
3. **Optimizes** traffic light timing to minimize average wait time across the network
4. **Adapts** in real-time to changing conditions

### Provided Data
- **Synthetic traffic data generator** (we provide code to simulate vehicles)
- **Real weather API** (OpenMeteo - free, no auth required)
- **Traffic pattern templates** (morning rush, evening rush, weekend, events)

### Success Criteria
- ✅ Demonstrate 20%+ improvement over fixed-timing lights
- ✅ Handle at least 4-intersection network
- ✅ Real-time visualization of traffic flow
- ✅ System responds to unexpected events (accident, road closure)

### Technical Requirements
- **ML Component**: Predict traffic volume/congestion (regression or classification)
- **Optimization**: Solve multi-objective optimization (minimize wait time, maximize throughput, ensure fairness)
- **Engineering**: Event-driven simulation, efficient state management, visualization

### Bonus Points
- Multi-agent reinforcement learning approach
- Pedestrian crossing integration
- Emergency vehicle priority routing
- Scalable to larger networks (10+ intersections)

### Why This Is Fun
- Visual and interactive (watch cars move!)
- Clear success metric (everyone hates traffic)
- Immediate feedback loop
- Real-world applicable

---

## Problem 2: Automated Fantasy Sports Team Optimizer ⚽🏀

### The Challenge
Build an AI system that constructs optimal fantasy sports lineups by predicting player performance and solving constraint-based team selection.

### Your Mission
Build a system that:
1. **Collects** player statistics and schedules (via free sports APIs)
2. **Predicts** player performance for upcoming matches using ML
3. **Optimizes** team selection under budget and roster constraints
4. **Explains** why specific players were chosen (interpretability)

### Provided Data
- **Sports API**: TheSportsDB (free tier) or ESPN public APIs
- **Historical player stats**: We provide 3 seasons of CSV data (soccer/basketball/your choice)
- **Injury reports**: Simulated data or manual input

### Success Criteria
- ✅ Beat baseline strategy (random, top-salary, last-week-performance) by 15%+
- ✅ Respect all fantasy league constraints (budget, positions, team limits)
- ✅ Provide confidence intervals for predictions
- ✅ Interactive tool to explore trade-offs

### Technical Requirements
- **ML Component**: Player performance prediction (regression with uncertainty quantification)
- **Optimization**: Integer programming or heuristic search (knapsack-style constraints)
- **Engineering**: Data pipeline, API integration, interactive dashboard

### Bonus Points
- Multi-week optimization (considering player fatigue, schedules)
- Opponent strength modeling
- Transfer market strategy (buy/sell recommendations)
- Ensemble models or gradient boosting

### Why This Is Fun
- Gamified problem (actual fantasy sports application)
- Multiple valid approaches
- Easy to understand, hard to master
- Can validate against real-world outcomes

---

## Problem 3: Warehouse Robot Fleet Coordination 🤖📦

### The Challenge
Optimize a fleet of autonomous warehouse robots to fulfill orders efficiently while avoiding collisions and deadlocks.

### Your Mission
Build a system that:
1. **Simulates** a warehouse with shelves, robots, and incoming orders
2. **Predicts** order volume and patterns using ML (seasonality, trends, anomalies)
3. **Optimizes** robot task assignment and pathfinding to minimize order fulfillment time
4. **Coordinates** multi-robot collision avoidance in real-time

### Provided Data
- **Warehouse simulator** (grid-based environment, we provide skeleton code)
- **Synthetic order data** (realistic e-commerce patterns with daily/weekly cycles)
- **Robot specifications** (speed, capacity, battery life)

### Success Criteria
- ✅ Process 100+ orders with 5-10 robots
- ✅ Zero collisions, zero deadlocks
- ✅ 30%+ improvement over naive greedy assignment
- ✅ Real-time visualization of robot movements

### Technical Requirements
- **ML Component**: Demand forecasting (time series), optimal robot positioning prediction
- **Optimization**: Multi-agent pathfinding (A*, conflict-based search), task assignment (Hungarian algorithm, auction-based)
- **Engineering**: Event simulation, state management, collision detection, visualization

### Bonus Points
- Battery management (robots need to recharge)
- Dynamic replanning when new urgent orders arrive
- Multi-floor warehouse
- Reinforcement learning for robot coordination

### Why This Is Fun
- Like a video game (watch robots zip around!)
- Classic CS problem (pathfinding) meets ML/optimization
- Highly visual and satisfying
- Relevant to modern logistics (Amazon, Ocado, Cash in Transit)

---

## Evaluation Rubric (Total: 100 points)

### 1. ML Modeling (25 points)
- Model selection and justification (10 pts)
- Feature engineering creativity (5 pts)
- Validation methodology (5 pts)
- Performance on test set (5 pts)

### 2. Mathematical Optimization (25 points)
- Problem formulation (10 pts)
- Algorithm choice and efficiency (5 pts)
- Constraint handling (5 pts)
- Solution quality (5 pts)

### 3. Software Engineering (25 points)
- Code quality and structure (5 pts)
- Working demo/visualization (10 pts)
- Scalability considerations (5 pts)
- Testing and error handling (5 pts)

### Presentation (25 points)
- Clear explanation (15 pts)
- Answering questions (10 pts)

## Good Luck! 🚀
