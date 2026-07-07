"""
Traffic Light Optimization - Starter Code
Problem 1: Smart City Traffic Management

This is a skeleton to get you started quickly.
Feel free to modify, extend, or completely rewrite!
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
from enum import Enum


class LightState(Enum):
    RED = 0
    GREEN = 1
    YELLOW = 2


@dataclass
class Vehicle:
    """Represents a vehicle in the simulation"""
    id: int
    position: Tuple[float, float]
    destination: Tuple[float, float]
    speed: float
    wait_time: float = 0.0
    
    
@dataclass
class Intersection:
    """Represents a traffic intersection"""
    id: int
    position: Tuple[float, float]
    light_states: Dict[str, LightState]  # direction -> state
    timer: float = 0.0
    

class TrafficSimulator:
    """
    Simulates traffic flow through a network of intersections
    
    TODO: Implement the core simulation logic
    - Vehicle spawning and movement
    - Traffic light timing
    - Collision detection
    - Performance metrics
    """
    
    def __init__(self, num_intersections: int = 4, grid_size: int = 10):
        self.intersections = self._create_intersection_grid(num_intersections)
        self.vehicles = []
        self.time = 0.0
        self.metrics = {
            'total_wait_time': 0.0,
            'avg_wait_time': 0.0,
            'throughput': 0
        }
        
    def _create_intersection_grid(self, n: int) -> List[Intersection]:
        """Create a grid of intersections"""
        # TODO: Implement intersection network topology
        intersections = []
        # Example: 2x2 grid
        for i in range(n):
            intersections.append(Intersection(
                id=i,
                position=(i % 2, i // 2),
                light_states={
                    'north': LightState.GREEN,
                    'south': LightState.RED,
                    'east': LightState.RED,
                    'west': LightState.RED
                }
            ))
        return intersections
    
    def spawn_vehicle(self, spawn_rate: float = 0.1):
        """Spawn new vehicles based on demand patterns"""
        # TODO: Implement vehicle spawning logic
        # Consider: time of day, weather, events
        pass
    
    def step(self, dt: float = 0.1):
        """Advance simulation by dt seconds"""
        # TODO: Implement one simulation step
        # 1. Update traffic lights
        # 2. Move vehicles
        # 3. Check for arrivals/departures
        # 4. Update metrics
        self.time += dt
        
    def update_lights(self, light_timings: Dict[int, Dict[str, float]]):
        """Update traffic light states based on control policy"""
        # TODO: Implement light timing logic
        # This is where your optimization comes in!
        pass
    
    def get_state(self) -> np.ndarray:
        """Get current state for ML prediction"""
        # TODO: Extract features for ML model
        # Example features: queue lengths, waiting times, time of day, etc.
        return np.array([])
    
    def get_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics"""
        # TODO: Compute average wait time, throughput, etc.
        return self.metrics


class TrafficDataGenerator:
    """Generate synthetic traffic data for ML training"""
    
    @staticmethod
    def generate_daily_patterns(days: int = 30) -> np.ndarray:
        """Generate realistic traffic patterns"""
        # TODO: Create synthetic traffic data with patterns
        # - Morning rush (7-9am)
        # - Evening rush (5-7pm)
        # - Weekend patterns
        # - Special events
        hours = np.arange(0, 24 * days)
        
        # Example: simple sinusoidal pattern
        traffic = 50 + 30 * np.sin(2 * np.pi * hours / 24)
        noise = np.random.normal(0, 10, len(hours))
        
        return traffic + noise
    
    @staticmethod
    def add_weather_effects(traffic: np.ndarray, weather_data: np.ndarray) -> np.ndarray:
        """Modify traffic based on weather conditions"""
        # TODO: Add weather impact (rain -> more traffic)
        return traffic


def main():
    """Example usage"""
    print("Traffic Light Optimization - Starter Code")
    print("=" * 50)
    
    # Create simulator
    sim = TrafficSimulator(num_intersections=4)
    
    # Generate training data
    data_gen = TrafficDataGenerator()
    traffic_data = data_gen.generate_daily_patterns(days=30)
    
    print(f"Generated {len(traffic_data)} hours of traffic data")
    print(f"Created {len(sim.intersections)} intersections")
    
    # TODO: Your implementation here!
    # 1. Train ML model to predict traffic
    # 2. Optimize light timings
    # 3. Run simulation
    # 4. Visualize results
    
    print("\nNext steps:")
    print("1. Implement ML model for traffic prediction")
    print("2. Implement optimization algorithm for light timing")
    print("3. Create visualization")
    print("4. Benchmark against baseline")


if __name__ == "__main__":
    main()
