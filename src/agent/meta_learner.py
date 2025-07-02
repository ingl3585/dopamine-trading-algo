# meta_learner.py

import torch
import torch.nn as nn
import numpy as np
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
from src.neural.uncertainty_estimator import UncertaintyEstimator
from src.agent.reward_engine import UnifiedRewardEngine
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


# RewardEngine moved to src.agent.reward_engine

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
            'position_size_factor': MetaParameter(0.1, (0.001, 5.0)),  # Adaptive position sizing
            'max_position_factor': MetaParameter(0.3, (0.1, 0.7)),   # More conservative for smaller accounts
            
            # Confidence thresholds (learned)
            'confidence_threshold': MetaParameter(0.3, (0.05, 0.9)),
            
            # Risk preferences (account-aware)
            'stop_preference': MetaParameter(0.5, (0.0, 1.0)),
            'target_preference': MetaParameter(0.5, (0.0, 1.0)),
            'stop_distance_factor': MetaParameter(0.015, (0.005, 0.05)),  # Tighter stops for MNQ
            'target_distance_factor': MetaParameter(0.03, (0.01, 0.1)),   # Reasonable targets for MNQ
            
            # Account protection (adaptive based on account size)
            'loss_tolerance_factor': MetaParameter(0.03, (0.001, 1.0)),    # Adaptive loss tolerance
            'consecutive_loss_tolerance': MetaParameter(5.0, (2.0, 15.0)),
            
            # New: Account size awareness
            'small_account_mode': MetaParameter(0.0, (0.0, 1.0)),        # 0=normal, 1=small account mode
            'margin_utilization_limit': MetaParameter(0.7, (0.3, 0.9)),   # Max margin usage
            
            # Enhanced position management
            'max_contracts_limit': MetaParameter(10.0, (1.0, 100.0)),      # Adaptive max position limit
            'position_concentration_factor': MetaParameter(0.8, (0.5, 1.0)), # How concentrated positions can be
            'exposure_scaling_threshold': MetaParameter(0.6, (0.4, 0.8))   # When to start scaling down
        }
        
        # Adaptive components  
        self.subsystem_weights = AdaptiveWeights(6)  # DNA, Temporal, Immune, Microstructure, Dopamine, Regime
        self.exploration_strategy = ExplorationStrategy(state_dim)
        self.reward_engine = UnifiedRewardEngine()
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
        return self.reward_engine.compute_trade_reward(trade_data)
    
    def compute_holding_reward(self, decision_confidence: float, market_conditions: Dict[str, Any]) -> float:
        """Compute reward for holding decisions based on context"""
        return self.reward_engine.compute_holding_reward(decision_confidence, market_conditions)
    
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
        self.reward_engine.adapt_parameters()
        
        # Count successful adaptations with more reasonable criteria
        adaptations = 0
        for name, param in self.parameters.items():
            old_val = old_values[name]
            new_val = param.value
            change = abs(new_val - old_val)
            
            # Multiple criteria for "successful adaptation"
            is_successful = (
                change > old_val * 0.005 or  # Lowered from 2% to 0.5%
                change > 0.001 or            # Absolute minimum change
                (hasattr(param, 'outcomes') and len(param.outcomes) > 0)  # Parameter is actively learning
            )
            
            if is_successful:
                adaptations += 1
        
        self.successful_adaptations += max(1, adaptations)  # At least 1 if learning occurred
        
        # Debug logging for learning efficiency tracking
        logger.info(f"META-LEARNER: total_updates={self.total_updates}, "
                   f"successful_adaptations={self.successful_adaptations}, "
                   f"efficiency={self.get_learning_efficiency():.1%}")
    
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
            # Handle PyTorch 2.6+ security requirements
            try:
                checkpoint = torch.load(filepath, weights_only=False)
            except Exception as weights_error:
                # Fallback for older PyTorch versions or if weights_only fails
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