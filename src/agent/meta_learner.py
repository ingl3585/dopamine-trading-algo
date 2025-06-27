# meta_learner.py

import torch
import torch.nn as nn
import numpy as np
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
from src.neural.uncertainty_estimator import UncertaintyEstimator
import logging

logger = logging.getLogger(__name__)

@dataclass
class MetaParameter:
    value: float
    bounds: Tuple[float, float]
    learning_rate: float = 0.01
    momentum: float = 0.0
    
    def __post_init__(self):
        self.velocity = 0.0
        self.outcomes = deque(maxlen=50)
    
    def update(self, gradient: float):
        self.velocity = self.momentum * self.velocity + self.learning_rate * gradient
        self.value = np.clip(self.value + self.velocity, *self.bounds)
    
    def add_outcome(self, outcome: float):
        self.outcomes.append(outcome)
    
    def get_gradient(self) -> float:
        if len(self.outcomes) < 5:
            return 0.0
        
        recent = list(self.outcomes)[-10:]
        return np.mean(recent)


class AdaptiveWeights(nn.Module):
    def __init__(self, num_components: int, initial_temp: float = 1.0):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_components, dtype=torch.float64))
        self.temperature = nn.Parameter(torch.tensor(initial_temp, dtype=torch.float64))
        
        # Convert to double precision
        self.double()
        
        self.outcomes = deque(maxlen=100)
    
    def forward(self) -> torch.Tensor:
        return torch.softmax(self.logits / torch.clamp(self.temperature, 0.1, 10.0), dim=0)
    
    def update_from_outcome(self, component_contributions: torch.Tensor, outcome: float):
        self.outcomes.append(outcome)
        
        if len(self.outcomes) >= 10:
            gradient = torch.tensor(outcome) * component_contributions
            self.logits.data += 0.01 * gradient


class ExplorationStrategy(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 3, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Convert to double precision
        self.double()
        
        self.recent_outcomes = deque(maxlen=20)
    
    def should_explore(self, state_features: torch.Tensor, recent_performance: float, 
                      time_since_last_trade: float, position_count: int) -> bool:
        context = torch.cat([
            state_features.flatten(),
            torch.tensor([recent_performance, time_since_last_trade, position_count])
        ])
        
        exploration_prob = self.net(context)
        return torch.rand(1, dtype=torch.float64) < exploration_prob
    
    def update_from_outcome(self, was_exploration: bool, outcome: float):
        self.recent_outcomes.append((was_exploration, outcome))


class RewardEngine:
    def __init__(self):
        self.components = {
            'pnl_weight': MetaParameter(1.0, (0.1, 3.0)),
            'drawdown_penalty': MetaParameter(0.5, (0.0, 2.0)),
            'hold_time_factor': MetaParameter(0.1, (0.0, 0.5)),
            'win_rate_bonus': MetaParameter(0.3, (0.0, 1.0)),
            'subsystem_consistency': MetaParameter(0.2, (0.0, 1.0)),
            'account_preservation': MetaParameter(0.4, (0.0, 1.0))  # New component
        }
        
        self.outcome_history = deque(maxlen=200)
    
    def compute_reward(self, trade_data: Dict[str, Any]) -> float:
        pnl = trade_data.get('pnl', 0.0)
        account_balance = trade_data.get('account_balance', 25000)
        hold_time = trade_data.get('hold_time', 1.0)
        was_winner = pnl > 0
        subsystem_agreement = trade_data.get('subsystem_agreement', 0.5)
        
        # Account-normalized PnL component
        pnl_norm = np.tanh(pnl / (account_balance * 0.01))  # Normalize by 1% of account
        
        # Hold time penalty for overly long trades
        hold_penalty = max(0, (hold_time - 3600) / 3600) * 0.1
        
        # Win rate context
        recent_wins = sum(1 for outcome in list(self.outcome_history)[-10:] if outcome > 0)
        win_rate_bonus = (recent_wins / 10.0 - 0.5) * 0.2
        
        # Subsystem consistency bonus
        consistency_bonus = (subsystem_agreement - 0.5) * 0.1
        
        # Account preservation bonus (reward smaller risks on smaller accounts)
        risk_pct = abs(pnl) / account_balance
        preservation_bonus = max(0, 0.02 - risk_pct) * 5.0  # Bonus for risks < 2%
        
        reward = (
            self.components['pnl_weight'].value * pnl_norm +
            self.components['hold_time_factor'].value * (-hold_penalty) +
            self.components['win_rate_bonus'].value * win_rate_bonus +
            self.components['subsystem_consistency'].value * consistency_bonus +
            self.components['account_preservation'].value * preservation_bonus
        )
        
        self.outcome_history.append(reward)
        
        # Update component parameters
        for component in self.components.values():
            component.add_outcome(reward)
        
        return reward
    
    def adapt_components(self):
        for component in self.components.values():
            gradient = component.get_gradient()
            component.update(gradient)

class ArchitectureEvolver:
    def __init__(self, base_sizes: List[int] = [64, 32]):
        self.current_sizes = base_sizes.copy()
        self.performance_window = deque(maxlen=50)
        self.evolution_threshold = -0.1
        self.generations = 0
        
        self.size_bounds = [(16, 128), (8, 64)]
        self.mutation_strength = 0.2
    
    def should_evolve(self) -> bool:
        if len(self.performance_window) < 30:
            return False
        
        recent_perf = np.mean(list(self.performance_window)[-20:])
        return recent_perf < self.evolution_threshold
    
    def evolve_architecture(self) -> List[int]:
        self.generations += 1
        new_sizes = []
        
        for i, (current_size, (min_size, max_size)) in enumerate(zip(self.current_sizes, self.size_bounds)):
            # Mutate size with some randomness but guided by performance
            mutation = np.random.normal(0, self.mutation_strength)
            new_size = int(current_size * (1 + mutation))
            new_size = np.clip(new_size, min_size, max_size)
            new_sizes.append(new_size)
        
        self.current_sizes = new_sizes
        self.performance_window.clear()
        
        return new_sizes
    
    def record_performance(self, performance: float):
        self.performance_window.append(performance)

class MetaLearner:
    def __init__(self, state_dim: int = 20):
        self.state_dim = state_dim
        
        # Account-aware parameters with dynamic bounds
        self.parameters = {
            # Trading frequency (account-aware)
            'trade_frequency_base': MetaParameter(5.0, (1.0, 20.0)),
            
            # Position sizing (completely account-driven)
            'position_size_factor': MetaParameter(0.1, (0.01, 0.5)),  # More conservative upper bound
            'max_position_factor': MetaParameter(0.3, (0.1, 0.7)),   # More conservative for smaller accounts
            
            # Confidence thresholds (learned)
            'confidence_threshold': MetaParameter(0.3, (0.05, 0.9)),
            
            # Risk preferences (account-aware)
            'stop_preference': MetaParameter(0.5, (0.0, 1.0)),
            'target_preference': MetaParameter(0.5, (0.0, 1.0)),
            'stop_distance_factor': MetaParameter(0.015, (0.005, 0.05)),  # Tighter stops for MNQ
            'target_distance_factor': MetaParameter(0.03, (0.01, 0.1)),   # Reasonable targets for MNQ
            
            # Account protection (adaptive based on account size)
            'loss_tolerance_factor': MetaParameter(0.03, (0.01, 0.1)),    # Max 3% daily loss initially
            'consecutive_loss_tolerance': MetaParameter(5.0, (2.0, 15.0)),
            
            # New: Account size awareness
            'small_account_mode': MetaParameter(0.0, (0.0, 1.0)),        # 0=normal, 1=small account mode
            'margin_utilization_limit': MetaParameter(0.7, (0.3, 0.9)),   # Max margin usage
            
            # Enhanced position management
            'max_contracts_limit': MetaParameter(10.0, (5.0, 15.0)),      # Learned max position limit
            'position_concentration_factor': MetaParameter(0.8, (0.5, 1.0)), # How concentrated positions can be
            'exposure_scaling_threshold': MetaParameter(0.6, (0.4, 0.8))   # When to start scaling down
        }
        
        # Adaptive components
        self.subsystem_weights = AdaptiveWeights(4)  # DNA, Micro, Temporal, Immune
        self.exploration_strategy = ExplorationStrategy(state_dim)
        self.reward_engine = RewardEngine()
        self.architecture_evolver = ArchitectureEvolver()
        
        # Learning tracking
        self.total_updates = 0
        self.successful_adaptations = 0
        
        # Account adaptation tracking
        self.last_account_balance = 25000
        self.account_adaptation_count = 0
        
    def adapt_to_account_size(self, account_balance: float):
        """Dynamically adjust parameters based on account size"""
        if abs(account_balance - self.last_account_balance) > self.last_account_balance * 0.1:
            self.account_adaptation_count += 1
            
            # Small account adjustments (< $10k)
            if account_balance < 10000:
                # More conservative for small accounts
                self.parameters['position_size_factor'].bounds = (0.01, 0.3)
                self.parameters['max_position_factor'].bounds = (0.05, 0.5)
                self.parameters['loss_tolerance_factor'].bounds = (0.005, 0.05)
                self.parameters['small_account_mode'].value = 1.0
                
            # Medium accounts ($10k - $50k)
            elif account_balance < 50000:
                self.parameters['position_size_factor'].bounds = (0.02, 0.4)
                self.parameters['max_position_factor'].bounds = (0.1, 0.6)
                self.parameters['loss_tolerance_factor'].bounds = (0.01, 0.08)
                self.parameters['small_account_mode'].value = 0.5
                
            # Larger accounts (> $50k)
            else:
                self.parameters['position_size_factor'].bounds = (0.05, 0.5)
                self.parameters['max_position_factor'].bounds = (0.1, 0.7)
                self.parameters['loss_tolerance_factor'].bounds = (0.02, 0.1)
                self.parameters['small_account_mode'].value = 0.0
            
            self.last_account_balance = account_balance
    
    def get_parameter(self, name: str) -> float:
        return self.parameters[name].value
    
    def get_subsystem_weights(self) -> torch.Tensor:
        return self.subsystem_weights()
    
    def should_explore(self, state_features: torch.Tensor, context: Dict[str, float]) -> bool:
        return self.exploration_strategy.should_explore(
            state_features,
            context.get('recent_performance', 0.0),
            context.get('time_since_last_trade', 0.0),
            context.get('position_count', 0)
        )
    
    def compute_reward(self, trade_data: Dict[str, Any]) -> float:
        return self.reward_engine.compute_reward(trade_data)
    
    def should_evolve_architecture(self) -> bool:
        return self.architecture_evolver.should_evolve()
    
    def evolve_architecture(self) -> List[int]:
        return self.architecture_evolver.evolve_architecture()
    
    def learn_from_outcome(self, trade_data: Dict[str, Any]):
        self.total_updates += 1
        
        # Adapt to account size changes
        account_balance = trade_data.get('account_balance', 25000)
        self.adapt_to_account_size(account_balance)
        
        outcome = trade_data.get('pnl', 0.0)
        normalized_outcome = np.tanh(outcome / (account_balance * 0.01))
        
        # Update all parameters
        old_values = {name: param.value for name, param in self.parameters.items()}
        
        for param in self.parameters.values():
            param.add_outcome(normalized_outcome)
        
        # Update subsystem weights if contribution data available
        if 'subsystem_contributions' in trade_data:
            self.subsystem_weights.update_from_outcome(
                trade_data['subsystem_contributions'],
                normalized_outcome
            )
        
        # Update exploration strategy
        if 'was_exploration' in trade_data:
            self.exploration_strategy.update_from_outcome(
                trade_data['was_exploration'],
                normalized_outcome
            )
        
        # Record architecture performance
        self.architecture_evolver.record_performance(normalized_outcome)
        
        # Adapt reward components
        self.reward_engine.adapt_components()
        
        # Count successful adaptations
        adaptations = sum(1 for name, param in self.parameters.items() 
                         if abs(param.value - old_values[name]) > old_values[name] * 0.02)
        self.successful_adaptations += adaptations
    
    def adapt_parameters(self):
        for param in self.parameters.values():
            gradient = param.get_gradient()
            param.update(gradient)
    
    def get_learning_efficiency(self) -> float:
        if self.total_updates == 0:
            return 0.0
        return self.successful_adaptations / self.total_updates
    
    def save_state(self, filepath: str):
        import os
        # Ensure directory exists with proper error handling
        dir_path = os.path.dirname(filepath)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        
        torch.save({
            'parameters': {name: (param.value, param.bounds) for name, param in self.parameters.items()},
            'subsystem_weights': self.subsystem_weights.state_dict(),
            'exploration_strategy': self.exploration_strategy.state_dict(),
            'architecture_sizes': self.architecture_evolver.current_sizes,
            'total_updates': self.total_updates,
            'successful_adaptations': self.successful_adaptations,
            'last_account_balance': self.last_account_balance,
            'account_adaptation_count': self.account_adaptation_count
        }, filepath)
    
    def load_state(self, filepath: str):
        try:
            checkpoint = torch.load(filepath)
            
            # Restore parameters
            for name, (value, bounds) in checkpoint['parameters'].items():
                if name in self.parameters:
                    self.parameters[name].value = value
                    self.parameters[name].bounds = bounds
            
            # Restore neural components
            self.subsystem_weights.load_state_dict(checkpoint['subsystem_weights'])
            self.exploration_strategy.load_state_dict(checkpoint['exploration_strategy'])
            
            # Restore architecture state
            self.architecture_evolver.current_sizes = checkpoint['architecture_sizes']
            
            self.total_updates = checkpoint.get('total_updates', 0)
            self.successful_adaptations = checkpoint.get('successful_adaptations', 0)
            self.last_account_balance = checkpoint.get('last_account_balance', 25000)
            self.account_adaptation_count = checkpoint.get('account_adaptation_count', 0)
            
        except FileNotFoundError:
            pass