"""
Queue-based traffic intersection simulator.

Model summary
-------------
- N intersections arranged in a ceil(sqrt(N)) × … grid (default: 2×2).
- Each intersection has 4 approaches (N, S, E, W).
- Two signal phases: NS (north + south green) and EW (east + west green).
- Between phases: YELLOW interval (yellow_time + all_red_time seconds).
- Vehicle arrivals: Poisson process with time-varying external rates, PLUS
  routed vehicles arriving from upstream intersections.
- Vehicle routing: ROUTING_FRACTION of served vehicles are forwarded to the
  adjacent downstream intersection after TRAVEL_TIME seconds in transit.
  The remaining fraction exits the network and counts toward throughput.
- Vehicle discharge: deterministic 1 veh/sec per approach when green
  (simplified single-lane saturation flow).
- Performance: total wait time tracked via queue-seconds (Little's Law).

Network topology (2×2 default)
--------------------------------
    I2 (col=0,row=1) --- I3 (col=1,row=1)
          |                       |
    I0 (col=0,row=0) --- I1 (col=1,row=0)

Direction convention: 'N' approach = vehicles that approached from the North
(heading south), etc.  A vehicle discharged from the S approach at I0 is
heading north and routes to I2's S approach after TRAVEL_TIME seconds.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# --- Constants ---------------------------------------------------------------

SATURATION_FLOW_VPH = 1800   # veh/hr — realistic single-lane saturation flow (0.5 veh/s)
DT = 1.0                     # seconds per step

ROUTING_FRACTION = 0.35      # fraction of served vehicles forwarded downstream
TRAVEL_TIME = 20.0           # seconds between adjacent intersections (~200 m @ 36 km/h)

_DIR_DELTA = {'N': (0, 1), 'S': (0, -1), 'E': (1, 0), 'W': (-1, 0)}
_OPPOSITE  = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}


# --- Data structures ---------------------------------------------------------

@dataclass
class Approach:
    direction: str
    queue: int = 0
    cumulative_arrivals: int = 0
    cumulative_served: int = 0
    blocked: bool = False          # True when an accident closes this approach
    _residual: float = field(default=0.0, repr=False)


@dataclass
class Intersection:
    id: int
    position: tuple                # (col, row) in 2-D grid
    approaches: Dict[str, Approach] = field(default_factory=dict)
    phase: str = 'NS'              # 'NS' | 'EW' | 'YELLOW_to_EW' | 'YELLOW_to_NS'
    phase_timer: float = 0.0
    green_ns: float = 30.0
    green_ew: float = 30.0
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
        return False


# --- Simulator ---------------------------------------------------------------

class TrafficSimulator:
    """
    Simulates N intersections arranged in a grid, with inter-intersection
    vehicle routing.

    Usage
    -----
        sim = TrafficSimulator(num_intersections=4)
        sim.set_timings(timings_dict)
        for each step:
            sim.step(arrival_rates_dict)
        print(sim.avg_wait_time)
    """

    DT = DT

    def __init__(self, num_intersections: int = 4):
        self.num_intersections = num_intersections
        self.intersections: List[Intersection] = self._build_grid(num_intersections)
        self.topology: Dict = self._build_topology()
        self.transit_buffer: List[dict] = []   # vehicles en-route between intersections
        self.time: float = 0.0
        self.total_wait_seconds: float = 0.0
        self.total_vehicles_served: int = 0
        self.events: List[dict] = []

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build_grid(self, n: int) -> List[Intersection]:
        """Arrange n intersections in a ceil(sqrt(n))-wide grid."""
        cols = math.ceil(math.sqrt(n))
        positions = [(i % cols, i // cols) for i in range(n)]
        return [Intersection(id=i, position=positions[i]) for i in range(n)]

    def _build_routing_inflow_set(self) -> set:
        """Return set of (intersection_id, approach_dir) that receive routed traffic."""
        return {(dst_id, approach_dir)
                for (_, _), (dst_id, approach_dir) in self.topology.items()}

    def augment_rates_with_routing(
        self,
        rates: Dict[int, Dict[str, float]],
    ) -> Dict[int, Dict[str, float]]:
        """
        Add estimated routing inflows to predicted external arrival rates.

        The optimizer uses ML predictions of external arrivals only.  But
        downstream approaches also receive routed vehicles from upstream.
        This method adds ROUTING_FRACTION × upstream_rate to each approach
        that has a routing inflow, so the optimizer sees the *total* expected
        demand and can size green times accordingly.
        """
        augmented = {i: dict(d) for i, d in rates.items()}
        for (src_id, approach_dir), (dst_id, _) in self.topology.items():
            inflow = rates[src_id][approach_dir] * ROUTING_FRACTION
            augmented[dst_id][approach_dir] = (
                augmented[dst_id].get(approach_dir, 0.0) + inflow
            )
        return augmented

    def scale_external_rates(
        self,
        rates: Dict[int, Dict[str, float]],
    ) -> Dict[int, Dict[str, float]]:
        """
        Scale down external arrival rates for approaches that receive routing.

        Reduces external arrivals by ROUTING_FRACTION for approaches that
        have upstream routing inflow, so that total flow (external + routed)
        stays approximately equal to the original external-only rates.
        This prevents downstream intersections from becoming oversaturated.
        """
        inflow_set = self._build_routing_inflow_set()
        scaled = {}
        for inter_id, dir_rates in rates.items():
            scaled[inter_id] = {
                d: r * (1 - ROUTING_FRACTION)
                if (inter_id, d) in inflow_set else r
                for d, r in dir_rates.items()
            }
        return scaled

    def _build_topology(self) -> Dict:
        """
        Build a routing table: (intersection_id, approach_dir) → (downstream_id, approach_dir).

        Direction convention:
          A vehicle discharged from the S approach is heading North.
          Going north from (col, row) reaches (col, row+1).
          It arrives there via the S approach (approaching from the south).
          So the approach_dir is preserved across the link.
        """
        pos_to_id = {inter.position: inter.id for inter in self.intersections}
        topology: Dict = {}
        for inter in self.intersections:
            col, row = inter.position
            for approach_dir in ('N', 'S', 'E', 'W'):
                exit_dir = _OPPOSITE[approach_dir]   # direction vehicle travels
                dc, dr = _DIR_DELTA[exit_dir]
                downstream_pos = (col + dc, row + dr)
                if downstream_pos in pos_to_id:
                    topology[(inter.id, approach_dir)] = (
                        pos_to_id[downstream_pos],
                        approach_dir,   # same approach direction at destination
                    )
        return topology

    def set_timings(self, timings: Dict[int, Dict[str, float]]):
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
        self.transit_buffer = []
        self.events = []

    # ------------------------------------------------------------------
    # Per-step update
    # ------------------------------------------------------------------

    def step(self, arrival_rates: Dict[int, Dict[str, float]]):
        """
        Advance simulation by DT seconds.

        arrival_rates = {intersection_id: {direction: rate_vph}}
        These represent *external* arrivals only (vehicles entering the network
        from outside).  Routed vehicles from upstream are injected automatically
        from the transit buffer.
        """
        self._flush_buffer()           # inject vehicles that completed transit
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

    def _flush_buffer(self):
        """Deliver vehicles that have completed inter-intersection travel."""
        remaining = []
        for entry in self.transit_buffer:
            if self.time >= entry['arrive_at']:
                dst = self.intersections[entry['dst_inter_id']]
                ap  = dst.approaches[entry['approach_dir']]
                if not ap.blocked:
                    ap.queue += entry['count']
                    ap.cumulative_arrivals += entry['count']
                # (if blocked, vehicles are simply lost — road is closed)
            else:
                remaining.append(entry)
        self.transit_buffer = remaining

    def _inject_arrivals(self, rates: Dict[int, Dict[str, float]]):
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
            if served == 0:
                continue

            approach.queue -= served
            approach.cumulative_served += served
            approach._residual -= served

            # Route a fraction of served vehicles to the downstream intersection.
            # Binomial sampling so routing works correctly even when served=1.
            key = (inter.id, direction)
            n_routed = (
                int(np.random.binomial(served, ROUTING_FRACTION))
                if key in self.topology else 0
            )
            n_exiting = served - n_routed

            self.total_vehicles_served += n_exiting   # count only network exits

            if n_routed > 0:
                dst_id, arrival_dir = self.topology[key]
                self.transit_buffer.append({
                    'src_inter_id': inter.id,
                    'approach_dir': direction,    # preserved: same lane at destination
                    'dst_inter_id': dst_id,
                    'arrive_at': self.time + TRAVEL_TIME,
                    'count': n_routed,
                })

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
        """Average wait time per vehicle exiting the network (seconds)."""
        if self.total_vehicles_served == 0:
            return 0.0
        return self.total_wait_seconds / self.total_vehicles_served

    @property
    def transit_vehicles(self) -> List[dict]:
        """
        Current in-transit vehicles between intersections, with travel progress.

        Each entry:
            src_inter_id  — source intersection
            approach_dir  — approach direction (same at source and destination)
            dst_inter_id  — destination intersection
            progress      — 0.0 (just departed) → 1.0 (about to arrive)
            count         — number of vehicles in this batch
        """
        result = []
        for entry in self.transit_buffer:
            elapsed = TRAVEL_TIME - (entry['arrive_at'] - self.time)
            progress = float(max(0.0, min(1.0, elapsed / TRAVEL_TIME)))
            result.append({
                'src_inter_id': entry['src_inter_id'],
                'approach_dir': entry['approach_dir'],
                'dst_inter_id': entry['dst_inter_id'],
                'progress':     progress,
                'count':        entry['count'],
            })
        return result

    def queue_per_intersection(self) -> List[int]:
        return [
            sum(ap.queue for ap in inter.approaches.values())
            for inter in self.intersections
        ]

    def throughput(self) -> int:
        return self.total_vehicles_served
