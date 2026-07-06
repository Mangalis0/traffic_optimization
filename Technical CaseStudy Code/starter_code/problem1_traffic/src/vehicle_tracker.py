"""
Visual vehicle tracker for the live animation.

Mirrors the simulator's queue state as a list of VisualVehicle objects so the
animation can draw individual cars rather than queue bars.  This module is
purely a presentation layer — it has no effect on simulation math.

Grid layout for the live view (scaled up so vehicles fit between intersections):
    I2 (0,3) --- I3 (3,3)
      |                |
    I0 (0,0) --- I1 (3,0)

Between adjacent intersections the road is 3 units long, giving ~2.1 units of
usable queuing space per approach (after subtracting the intersection box).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List


# ── Grid constants for the live view ──────────────────────────────────────────
INTER_POS: Dict[int, tuple] = {0: (0, 0), 1: (3, 0), 2: (0, 3), 3: (3, 3)}
DIR_VEC:   Dict[str, tuple] = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}

STOP_LINE   = 0.46    # distance (units) from intersection centre to stop line
SPACING     = 0.13    # vehicle spacing in queue
MAX_SHOW    = 9       # maximum vehicles displayed per approach (avoids road overlap)
MAX_TRANSIT = 16      # hard cap on simultaneously-animated transit vehicles
BOX_HALF    = 0.38    # half-width of the intersection box drawn on screen

# Car rectangle dimensions (width x, height y) per approach direction.
# Cars on vertical roads are portrait; cars on horizontal roads are landscape.
CAR_W: Dict[str, float] = {"N": 0.09, "S": 0.09, "E": 0.15, "W": 0.15}
CAR_H: Dict[str, float] = {"N": 0.15, "S": 0.15, "E": 0.09, "W": 0.09}

# Left-hand drive lane offset (South Africa drives on the left).
# Each approach gets an offset perpendicular to its travel direction so that
# approaching traffic sits in the correct (left) lane.
#   N approach = heading south  → left lane is the EAST side  → +x
#   S approach = heading north  → left lane is the WEST side  → -x
#   E approach = heading west   → left lane is the SOUTH side → -y
#   W approach = heading east   → left lane is the NORTH side → +y
LANE_OFFSET: Dict[str, tuple] = {
    "N": ( 0.10, 0.0),
    "S": (-0.10, 0.0),
    "E": (0.0, -0.10),
    "W": (0.0,  0.10),
}

# Signal light positions: just outside the intersection box on each approach
_SIG: Dict[str, tuple] = {
    "N": ( 0.14,  BOX_HALF),
    "S": (-0.14, -BOX_HALF),
    "E": ( BOX_HALF, -0.14),
    "W": (-BOX_HALF,  0.14),
}

# Realistic car colours — deliberately avoiding traffic-light colours
# (no pure red, green or orange).
COLORS = [
    "#C0C0C0",  # silver
    "#2C2C2C",  # near-black
    "#F0EFE8",  # off-white / cream
    "#1C3A5E",  # midnight blue
    "#6B1A1A",  # dark burgundy
    "#3D5A3E",  # forest green (muted, not traffic-light green)
    "#5D4037",  # dark brown
    "#37474F",  # blue-grey
    "#4A148C",  # deep purple
    "#BF8A30",  # gold / sand
    "#546E7A",  # steel blue
    "#4E342E",  # dark umber
]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class VisualVehicle:
    vid:              int
    inter_id:         int
    direction:        str
    queue_pos:        int
    color:            str
    in_transit:       bool  = False
    transit_progress: float = 0.0    # 0 → stop-line, 1 → past intersection


def vehicle_xy(veh: VisualVehicle):
    """
    Return (x, y) screen coordinates for a VisualVehicle.

    Applies the left-hand-drive lane offset so approaching vehicles sit in
    the correct (left) lane of the road.
    """
    cx, cy = INTER_POS[veh.inter_id]
    dx, dy = DIR_VEC[veh.direction]
    lx, ly = LANE_OFFSET[veh.direction]

    if veh.in_transit:
        # Lerp from stop-line to 0.65 units past the intersection centre
        dist = STOP_LINE + veh.transit_progress * (-STOP_LINE - 0.65)
    else:
        dist = STOP_LINE + veh.queue_pos * SPACING

    return cx + dx * dist + lx, cy + dy * dist + ly


def signal_xy(inter_id: int, direction: str):
    """Position for the signal light indicator."""
    cx, cy = INTER_POS[inter_id]
    ox, oy = _SIG[direction]
    return cx + ox, cy + oy


# ── Tracker ───────────────────────────────────────────────────────────────────

class VehicleTracker:
    """
    Maintains per-approach lists of VisualVehicle objects that stay in sync
    with the simulator's queue counts.

    Typical use in animation loop
    ------------------------------
        tracker = VehicleTracker()
        for step in range(total_steps):
            sim.step(actual_rates)
            tracker.sync(sim)
            if step % render_every == 0:
                tracker.advance_transit()
                fig = fig_network_live(sim, tracker, ...)
    """

    def __init__(self):
        self._next_id: int = 0
        self._queues: Dict[tuple, List[VisualVehicle]] = {}
        self._transit: List[VisualVehicle] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def sync(self, sim) -> None:
        """Add / remove visual vehicles to match sim queue counts."""
        for inter in sim.intersections:
            for d, ap in inter.approaches.items():
                key = (inter.id, d)
                queue = self._queues.get(key, [])
                target = min(ap.queue, MAX_SHOW)

                if target > len(queue):
                    for _ in range(target - len(queue)):
                        queue.append(VisualVehicle(
                            vid=self._next_id,
                            inter_id=inter.id,
                            direction=d,
                            queue_pos=len(queue),
                            color=random.choice(COLORS),
                        ))
                        self._next_id += 1

                elif target < len(queue):
                    n_remove = len(queue) - target
                    # Cap total transit vehicles to avoid visual clutter.
                    # Served vehicles beyond the cap are just removed silently.
                    slots = max(0, MAX_TRANSIT - len(self._transit))
                    n_transit = min(n_remove, slots, 1)  # at most 1 per sync call
                    for veh in queue[:n_transit]:
                        veh.in_transit = True
                        veh.transit_progress = 0.0
                        self._transit.append(veh)
                    queue = queue[n_remove:]
                    for i, v in enumerate(queue):
                        v.queue_pos = i

                self._queues[key] = queue

    def advance_transit(self, step: float = 0.55) -> None:
        """Move transit vehicles forward; discard completed ones."""
        still = []
        for v in self._transit:
            v.transit_progress += step
            if v.transit_progress < 1.0:
                still.append(v)
        self._transit = still

    def reset(self) -> None:
        self._queues.clear()
        self._transit.clear()
        self._next_id = 0

    def queued(self) -> List[VisualVehicle]:
        result: List[VisualVehicle] = []
        for vlist in self._queues.values():
            result.extend(vlist)
        return result

    def transiting(self) -> List[VisualVehicle]:
        return list(self._transit)

    def overflow_counts(self, sim) -> Dict[tuple, int]:
        """Return {(inter_id, direction): overflow_count} for approaches with q > MAX_SHOW."""
        out = {}
        for inter in sim.intersections:
            for d, ap in inter.approaches.items():
                if ap.queue > MAX_SHOW:
                    out[(inter.id, d)] = ap.queue - MAX_SHOW
        return out
