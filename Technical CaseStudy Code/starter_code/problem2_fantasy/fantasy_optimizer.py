"""
Fantasy Sports Team Optimizer - Starter Code
Problem 2: Automated Fantasy Team Selection

This is a skeleton to get you started quickly.
Feel free to modify, extend, or completely rewrite!
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class Player:
    """Represents a player in the fantasy league"""
    id: int
    name: str
    position: str
    team: str
    salary: float
    projected_points: float = 0.0
    actual_points: float = 0.0
    injury_status: str = "healthy"
    

@dataclass
class TeamConstraints:
    """Fantasy team building constraints"""
    total_budget: float = 100.0
    roster_size: int = 11
    position_limits: Dict[str, Tuple[int, int]] = None  # position -> (min, max)
    max_per_team: int = 3
    

class FantasyDataLoader:
    """Load and process player statistics"""
    
    @staticmethod
    def load_historical_data(sport: str = "soccer") -> pd.DataFrame:
        """Load historical player statistics"""
        # TODO: Load real data from API or CSV
        # For now, generate synthetic data
        
        n_players = 200
        positions = ['GK', 'DEF', 'MID', 'FWD']
        teams = [f'Team_{i}' for i in range(20)]
        
        data = {
            'player_id': range(n_players),
            'name': [f'Player_{i}' for i in range(n_players)],
            'position': np.random.choice(positions, n_players),
            'team': np.random.choice(teams, n_players),
            'salary': np.random.uniform(4.0, 12.0, n_players),
            'points_week_1': np.random.poisson(5, n_players),
            'points_week_2': np.random.poisson(5, n_players),
            'points_week_3': np.random.poisson(5, n_players),
            # Add more historical weeks as needed
        }
        
        return pd.DataFrame(data)
    
    @staticmethod
    def fetch_upcoming_fixtures() -> pd.DataFrame:
        """Get upcoming match schedules"""
        # TODO: Fetch from sports API (TheSportsDB, ESPN)
        # For now, return mock data
        return pd.DataFrame({
            'team': [f'Team_{i}' for i in range(20)],
            'opponent': [f'Team_{(i+1)%20}' for i in range(20)],
            'difficulty': np.random.randint(1, 6, 20)  # 1=easy, 5=hard
        })


class PlayerPerformanceModel:
    """ML model to predict player performance"""
    
    def __init__(self):
        self.model = None
        
    def train(self, historical_data: pd.DataFrame):
        """Train model to predict player points"""
        # TODO: Implement ML model
        # Ideas:
        # - Linear regression with player features
        # - Gradient boosting (LightGBM, XGBoost)
        # - Time series model for form
        # - Ensemble approach
        
        print("Training player performance model...")
        # Placeholder
        pass
    
    def predict(self, players: pd.DataFrame, fixtures: pd.DataFrame) -> np.ndarray:
        """Predict points for upcoming week"""
        # TODO: Make predictions with confidence intervals
        # Consider: opponent strength, home/away, recent form, injuries
        
        # Placeholder: random predictions
        return np.random.poisson(5, len(players))
    
    def get_prediction_intervals(self, players: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Return confidence intervals for predictions"""
        # TODO: Implement uncertainty quantification
        predictions = self.predict(players, None)
        lower = predictions - 2
        upper = predictions + 2
        return lower, upper


class TeamOptimizer:
    """Optimize team selection under constraints"""
    
    def __init__(self, constraints: TeamConstraints):
        self.constraints = constraints
        
    def optimize(self, 
                 players: pd.DataFrame, 
                 predicted_points: np.ndarray) -> List[int]:
        """
        Solve the team selection problem
        
        Maximize: sum of predicted points
        Subject to:
        - Total salary <= budget
        - Roster size = N
        - Position requirements (e.g., 1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD)
        - Max M players from same team
        
        This is an integer programming problem (knapsack variant)
        """
        # TODO: Implement optimization
        # Approaches:
        # - Linear programming (PuLP, CVXPY)
        # - Google OR-Tools
        # - Greedy heuristic with local search
        # - Dynamic programming
        
        print("Optimizing team selection...")
        
        # Placeholder: return random team
        selected_indices = np.random.choice(
            len(players), 
            size=self.constraints.roster_size, 
            replace=False
        )
        return selected_indices.tolist()
    
    def explain_selection(self, 
                         players: pd.DataFrame, 
                         selected_indices: List[int]) -> Dict:
        """Explain why players were selected"""
        # TODO: Provide interpretability
        # - Value per dollar
        # - Position scarcity
        # - Expected points contribution
        selected_players = players.iloc[selected_indices]
        
        return {
            'total_salary': selected_players['salary'].sum(),
            'total_predicted_points': selected_players['points_week_1'].sum(),
            'positions': selected_players['position'].value_counts().to_dict()
        }


class BaselineStrategy:
    """Simple baseline strategies to beat"""
    
    @staticmethod
    def random_team(players: pd.DataFrame, constraints: TeamConstraints) -> List[int]:
        """Random valid team"""
        return np.random.choice(len(players), constraints.roster_size, replace=False).tolist()
    
    @staticmethod
    def greedy_highest_salary(players: pd.DataFrame, constraints: TeamConstraints) -> List[int]:
        """Pick most expensive players (often good)"""
        sorted_idx = players['salary'].argsort()[::-1]
        return sorted_idx[:constraints.roster_size].tolist()
    
    @staticmethod
    def greedy_last_week(players: pd.DataFrame, constraints: TeamConstraints) -> List[int]:
        """Pick best performers from last week"""
        sorted_idx = players['points_week_1'].argsort()[::-1]
        return sorted_idx[:constraints.roster_size].tolist()


def main():
    """Example usage"""
    print("Fantasy Sports Team Optimizer - Starter Code")
    print("=" * 50)
    
    # Load data
    loader = FantasyDataLoader()
    historical_data = loader.load_historical_data()
    fixtures = loader.fetch_upcoming_fixtures()
    
    print(f"Loaded {len(historical_data)} players")
    print(f"Loaded {len(fixtures)} fixtures")
    
    # Train model
    model = PlayerPerformanceModel()
    model.train(historical_data)
    predictions = model.predict(historical_data, fixtures)
    
    # Optimize team
    constraints = TeamConstraints(
        total_budget=100.0,
        roster_size=11,
        position_limits={
            'GK': (1, 1),
            'DEF': (3, 5),
            'MID': (2, 5),
            'FWD': (1, 3)
        },
        max_per_team=3
    )
    
    optimizer = TeamOptimizer(constraints)
    selected_team = optimizer.optimize(historical_data, predictions)
    
    # Evaluate
    print("\nOptimized Team:")
    print(historical_data.iloc[selected_team][['name', 'position', 'salary']])
    
    explanation = optimizer.explain_selection(historical_data, selected_team)
    print(f"\nTeam Summary: {explanation}")
    
    # Compare to baseline
    baseline = BaselineStrategy.greedy_highest_salary(historical_data, constraints)
    print(f"\nBaseline team size: {len(baseline)}")
    
    print("\nNext steps:")
    print("1. Implement ML model for point prediction")
    print("2. Implement optimization algorithm")
    print("3. Create interactive dashboard")
    print("4. Validate against real-world results")


if __name__ == "__main__":
    main()
