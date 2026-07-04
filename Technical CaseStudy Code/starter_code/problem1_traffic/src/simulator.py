"""
Queue-based traffic intersection simulator.

Model summary
-------------
- 4 intersections arranged in a 2×2 grid.
- Each intersection has 4 approaches (N, S, E, W).
- Two signal phases: NS (north + south green) and EW (east + west green).
- Between phases: YELLOW interval (yellow_time + all_red_time seconds).
- Vehicle arrivals: Poisson process with time-varying rates.
- Vehicle discharge: deterministic 1 veh/sec per approach when green
  (simplified single-lane saturation flow).
- Performance: total wait time tracked via queue-seconds (Little's Law).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# --- Data structures -------------------------------------------------------

@dataclass
class Approach:
    direction: str
    queue: int = 0
    cumulative_arrivals: int = 0
    cumulative_served: int = 0
    blocked: bool = False          # True when an accident closes this approach
    _residual: float = field(default=0.0, repr=False)  # fractional discharge carry-over


@dataclass
class Intersection:
    id: int
    position: tuple                # (col, row) in 2-D grid
    approaches: Dict[str, Approach] = field(default_factory=dict)
    phase: str = 'NS'              # 'NS' | 'EW' | 'YELLOW_to_EW' | 'YELLOW_to_NS'
    phase_timer: float = 0.0
    green_ns: float = 30.0        # seconds
    green_ew: float = 30.0        # seconds
    yellow_time: float = 3.0
    all_red_time: float = 2.0

    def __post_init__(self):
        for d in ('N', 'S', 'E', 'W'):
            self.approaches[d] = Approach(direction=d)

    @property
    def cycle_length(self) -> float:
        return (
            self.green_ns + self.green_ew
            + 2 * (self.yellow_time + self.all_red_time)
        )

    def is_green(self, direction: str) -> bool:
        if self.phase == 'NS':
            return direction in ('N', 'S')
        if self.phase == 'EW':
            return direction in ('E', 'W')
        return False   # yellow / all-red


# --- Simulator -------------------------------------------------------------

SATURATION_FLOW_VPH = 3600   # veh/hr = 1 veh/sec per approach (simplified)
DT = 1.0                     # seconds per step


class TrafficSimulator:
    """
    Simulates 4 intersections in a 2×2 grid.

    Usage
    -----
        sim = TrafficSimulator()
        sim.set_timings(timings_dict)
        for each step:
            sim.step(arrival_rates_dict)
        print(sim.avg_wait_time)
    """

    DT = DT

    def __init__(self, num_intersections: int = 4):
        assert num_intersections == 4
        self.intersections: List[Intersection] = self._build_grid()
        self.time: float = 0.0
        self.total_wait_seconds: float = 0.0
        self.total_vehicles_served: int = 0
        self.events: List[dict] = []

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build_grid(self) -> List[Intersection]:
        positions = [(0, 0), (1, 0), (0, 1), (1, 1)]
        return [Intersection(id=i, position=positions[i]) for i in range(4)]

    def set_timings(self, timings: Dict[int, Dict[str, float]]):
        """Apply optimised green times.  timings = {id: {'green_ns': t, 'green_ew': t}}"""
        for inter in self.intersections:
            if inter.id in timings:
                t = timings[inter.id]
                inter.green_ns = float(t.get('green_ns', inter.green_ns))
                inter.green_ew = float(t.get('green_ew', inter.green_ew))

    def reset(self):
        for inter in self.intersections:
            for ap in inter.approaches.values():
                ap.queue = 0
                ap.cumulative_arrivals = 0
                ap.cumulative_served = 0
                ap.blocked = False
                ap._residual = 0.0
            inter.phase = 'NS'
            inter.phase_timer = 0.0
        self.time = 0.0
        self.total_wait_seconds = 0.0
        self.total_vehicles_served = 0
        self.events = []

    # ------------------------------------------------------------------
    # Per-step update
    # ------------------------------------------------------------------

    def step(self, arrival_rates: Dict[int, Dict[str, float]]):
        """
        Advance simulation by DT seconds.

        arrival_rates = {intersection_id: {direction: rate_vph}}
        """
        self._inject_arrivals(arrival_rates)
        self._process_events()

        for inter in self.intersections:
            self._update_signal(inter)
            self._discharge_queues(inter)
            self._accumulate_wait(inter)

        self.time += DT

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _inject_arrivals(self, rates: Dict[int, Dict[str, float]]):
        discharge_per_step = (SATURATION_FLOW_VPH / 3600.0) * DT  # 1.0 veh/step

        for inter in self.intersections:
            inter_rates = rates.get(inter.id, {})
            for direction, approach in inter.approaches.items():
                if approach.blocked:
                    continue
                rate_vph = inter_rates.get(direction, 0.0)
                rate_per_step = rate_vph / 3600.0 * DT
                arrivals = int(np.random.poisson(rate_per_step))
                approach.queue += arrivals
                approach.cumulative_arrivals += arrivals

    def _update_signal(self, inter: Intersection):
        inter.phase_timer += DT
        gap = inter.yellow_time + inter.all_red_time

        if inter.phase == 'NS' and inter.phase_timer >= inter.green_ns:
            inter.phase = 'YELLOW_to_EW'
            inter.phase_timer = 0.0
        elif inter.phase == 'EW' and inter.phase_timer >= inter.green_ew:
            inter.phase = 'YELLOW_to_NS'
            inter.phase_timer = 0.0
        elif inter.phase == 'YELLOW_to_EW' and inter.phase_timer >= gap:
            inter.phase = 'EW'
            inter.phase_timer = 0.0
        elif inter.phase == 'YELLOW_to_NS' and inter.phase_timer >= gap:
            inter.phase = 'NS'
            inter.phase_timer = 0.0

    def _discharge_queues(self, inter: Intersection):
        discharge_per_step = (SATURATION_FLOW_VPH / 3600.0) * DT  # 1.0 veh

        for direction, approach in inter.approaches.items():
            if not inter.is_green(direction) or approach.queue == 0:
                approach._residual = 0.0
                continue

            approach._residual += discharge_per_step
            served = min(int(approach._residual), approach.queue)
            if served > 0:
                approach.queue -= served
                approach.cumulative_served += served
                self.total_vehicles_served += served
                approach._residual -= served

    def _accumulate_wait(self, inter: Intersection):
        for approach in inter.approaches.values():
            self.total_wait_seconds += approach.queue * DT

    # ------------------------------------------------------------------
    # Events (accidents / road closures)
    # ------------------------------------------------------------------

    def trigger_event(
        self,
        event_type: str,
        intersection_id: int,
        direction: str,
        duration: float,
    ):
        """Block an approach for `duration` seconds."""
        inter = self.intersections[intersection_id]
        inter.approaches[direction].blocked = True
        self.events.append({
            'type': event_type,
            'intersection_id': intersection_id,
            'direction': direction,
            'end_time': self.time + duration,
        })

    def _process_events(self):
        active = []
        for ev in self.events:
            if self.time >= ev['end_time']:
                inter = self.intersections[ev['intersection_id']]
                inter.approaches[ev['direction']].blocked = False
            else:
                active.append(ev)
        self.events = active

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def avg_wait_time(self) -> float:
        """Average wait time per served vehicle (seconds), via Little's Law."""
        if self.total_vehicles_served == 0:
            return 0.0
        return self.total_wait_seconds / self.total_vehicles_served

    def queue_per_intersection(self) -> List[int]:
        """Total queued vehicles at each intersection."""
        return [
            sum(ap.queue for ap in inter.approaches.values())
            for inter in self.intersections
        ]

    def throughput(self) -> int:
        return self.total_vehicles_served
