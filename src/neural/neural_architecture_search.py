"""
Neural Architecture Search (NAS) for Trading Networks

This module implements neural architecture search capabilities for the trading system,
allowing networks to evolve their architectures based on performance metrics.

Features:
- Performance-based architecture mutations
- Gradual architecture transitions
- Architecture performance tracking
- Optimal layer size suggestions
- Architecture validation
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import random
import math

logger = logging.getLogger(__name__)


class MutationType(Enum):
    """Types of architecture mutations"""
    EXPAND_LAYER = "expand_layer"
    SHRINK_LAYER = "shrink_layer"
    ADD_LAYER = "add_layer"
    REMOVE_LAYER = "remove_layer"
    CHANGE_ACTIVATION = "change_activation"
    MODIFY_DROPOUT = "modify_dropout"


@dataclass
class ArchitectureConfig:
    """Configuration for neural network architecture"""
    layer_sizes: List[int] = field(default_factory=lambda: [128, 64])
    activation_types: List[str] = field(default_factory=lambda: ["relu", "relu"])
    dropout_rates: List[float] = field(default_factory=lambda: [0.1, 0.1])
    batch_norm: List[bool] = field(default_factory=lambda: [True, True])
    
    # Architecture constraints
    min_layers: int = 2
    max_layers: int = 6
    min_layer_size: int = 16
    max_layer_size: int = 512
    
    # Performance tracking
    performance_score: float = 0.0
    stability_score: float = 0.0
    complexity_penalty: float = 0.0
    
    def __post_init__(self):
        """Validate architecture configuration"""
        if not self.layer_sizes:
            self.layer_sizes = [128, 64]
        
        # Ensure all lists have same length
        num_layers = len(self.layer_sizes)
        if len(self.activation_types) != num_layers:
            self.activation_types = ["relu"] * num_layers
        if len(self.dropout_rates) != num_layers:
            self.dropout_rates = [0.1] * num_layers
        if len(self.batch_norm) != num_layers:
            self.batch_norm = [True] * num_layers
    
    def calculate_complexity(self) -> float:
        """Calculate architecture complexity score"""
        total_params = sum(self.layer_sizes)
        layer_count = len(self.layer_sizes)
        
        # Complexity based on parameter count and layer depth
        complexity = (total_params / 1000.0) + (layer_count * 0.1)
        return complexity
    
    def get_total_score(self) -> float:
        """Get total architecture score (performance - complexity penalty)"""
        complexity = self.calculate_complexity()
        self.complexity_penalty = complexity * 0.1
        return self.performance_score - self.complexity_penalty


@dataclass
class PerformanceMetrics:
    """Performance metrics for architecture evaluation"""
    accuracy: float = 0.0
    loss: float = float('inf')
    training_time: float = 0.0
    memory_usage: float = 0.0
    convergence_speed: float = 0.0
    stability: float = 0.0
    
    def compute_score(self) -> float:
        """Compute composite performance score"""
        # Weighted combination of metrics
        score = (
            self.accuracy * 0.4 +
            max(0, 1.0 - self.loss) * 0.3 +
            max(0, 1.0 - self.training_time / 10.0) * 0.1 +
            max(0, 1.0 - self.memory_usage / 2.0) * 0.1 +
            self.convergence_speed * 0.05 +
            self.stability * 0.05
        )
        return max(0.0, min(1.0, score))


class PerformanceTracker:
    """Tracks performance trends and triggers evolution"""
    
    def __init__(self, window_size: int = 100, patience: int = 50):
        self.window_size = window_size
        self.patience = patience
        self.performance_history = deque(maxlen=window_size)
        self.best_performance = -float('inf')
        self.steps_since_improvement = 0
        
        # Performance trend analysis
        self.trend_window = 20
        self.trend_threshold = 0.02
        
    def record_performance(self, performance: float):
        """Record a performance measurement"""
        self.performance_history.append(performance)
        
        if performance > self.best_performance:
            self.best_performance = performance
            self.steps_since_improvement = 0
        else:
            self.steps_since_improvement += 1
    
    def should_evolve(self) -> bool:
        """Determine if architecture should evolve"""
        if len(self.performance_history) < self.trend_window:
            return False
        
        # Check for performance stagnation
        if self.steps_since_improvement >= self.patience:
            return True
        
        # Check for negative performance trend
        recent_performance = list(self.performance_history)[-self.trend_window:]
        if len(recent_performance) >= self.trend_window:
            trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
            if trend < -self.trend_threshold:
                return True
        
        return False
    
    def get_performance_trend(self) -> float:
        """Get current performance trend (positive = improving)"""
        if len(self.performance_history) < self.trend_window:
            return 0.0
        
        recent_performance = list(self.performance_history)[-self.trend_window:]
        trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
        return trend
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance tracking statistics"""
        if not self.performance_history:
            return {'current': 0.0, 'best': 0.0, 'trend': 0.0}
        
        return {
            'current': self.performance_history[-1],
            'best': self.best_performance,
            'trend': self.get_performance_trend(),
            'steps_since_improvement': self.steps_since_improvement,
            'history_length': len(self.performance_history)
        }


class NeuralArchitectureSearch:
    """Neural Architecture Search for trading networks"""
    
    def __init__(self, 
                 input_size: int = 64,
                 mutation_rate: float = 0.1,
                 population_size: int = 5,
                 elite_ratio: float = 0.3):
        self.input_size = input_size
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.elite_count = max(1, int(population_size * elite_ratio))
        
        # Architecture population
        self.population: List[ArchitectureConfig] = []
        self.performance_tracker = PerformanceTracker()
        
        # Mutation probabilities
        self.mutation_probabilities = {
            MutationType.EXPAND_LAYER: 0.3,
            MutationType.SHRINK_LAYER: 0.3,
            MutationType.ADD_LAYER: 0.15,
            MutationType.REMOVE_LAYER: 0.1,
            MutationType.CHANGE_ACTIVATION: 0.1,
            MutationType.MODIFY_DROPOUT: 0.05
        }
        
        # Evolution statistics
        self.generation = 0
        self.best_architecture = None
        self.evolution_history = []
        
        # Initialize population
        self._initialize_population()
        
        logger.info(f"NAS initialized with population size {population_size}")
    
    def _initialize_population(self):
        """Initialize the architecture population"""
        self.population = []
        
        # Create diverse initial architectures
        base_configs = [
            [128, 64],
            [256, 128, 64],
            [64, 32],
            [128, 128, 64],
            [256, 128, 128, 64]
        ]
        
        for i in range(self.population_size):
            if i < len(base_configs):
                layer_sizes = base_configs[i]
            else:
                # Generate random architecture
                num_layers = random.randint(2, 4)
                layer_sizes = []
                for j in range(num_layers):
                    size = random.choice([32, 64, 128, 256])
                    layer_sizes.append(size)
            
            config = ArchitectureConfig(layer_sizes=layer_sizes)
            self.population.append(config)
    
    def suggest_mutation(self, current_performance: float) -> ArchitectureConfig:
        """Suggest architecture mutation based on current performance"""
        self.performance_tracker.record_performance(current_performance)
        
        if not self.performance_tracker.should_evolve():
            # Return current best architecture
            return self.get_best_architecture()
        
        # Evolve population
        self._evolve_population()
        
        # Return best architecture from new generation
        return self.get_best_architecture()
    
    def _evolve_population(self):
        """Evolve the architecture population"""
        self.generation += 1
        
        # Sort population by performance
        self.population.sort(key=lambda x: x.get_total_score(), reverse=True)
        
        # Keep elite architectures
        new_population = self.population[:self.elite_count].copy()
        
        # Generate new architectures through mutation and crossover
        while len(new_population) < self.population_size:
            if random.random() < 0.7:  # Mutation
                parent = random.choice(self.population[:self.elite_count])
                child = self._mutate_architecture(parent)
            else:  # Crossover
                parent1, parent2 = random.sample(self.population[:self.elite_count], 2)
                child = self._crossover_architectures(parent1, parent2)
            
            new_population.append(child)
        
        self.population = new_population
        
        # Update best architecture
        self.best_architecture = self.population[0]
        
        # Record evolution
        self.evolution_history.append({
            'generation': self.generation,
            'best_score': self.best_architecture.get_total_score(),
            'best_config': self.best_architecture.layer_sizes.copy(),
            'population_diversity': self._calculate_diversity()
        })
        
        logger.info(f"Generation {self.generation}: Best score {self.best_architecture.get_total_score():.4f}")
    
    def _mutate_architecture(self, parent: ArchitectureConfig) -> ArchitectureConfig:
        """Mutate an architecture configuration"""
        # Create copy of parent
        child = ArchitectureConfig(
            layer_sizes=parent.layer_sizes.copy(),
            activation_types=parent.activation_types.copy(),
            dropout_rates=parent.dropout_rates.copy(),
            batch_norm=parent.batch_norm.copy()
        )
        
        # Apply mutations
        for mutation_type, probability in self.mutation_probabilities.items():
            if random.random() < probability:
                child = self._apply_mutation(child, mutation_type)
        
        return child
    
    def _apply_mutation(self, config: ArchitectureConfig, mutation_type: MutationType) -> ArchitectureConfig:
        """Apply specific mutation to architecture"""
        if mutation_type == MutationType.EXPAND_LAYER:
            # Increase size of random layer
            if config.layer_sizes:
                idx = random.randint(0, len(config.layer_sizes) - 1)
                growth_factor = random.uniform(1.2, 1.5)
                new_size = min(config.max_layer_size, int(config.layer_sizes[idx] * growth_factor))
                config.layer_sizes[idx] = new_size
        
        elif mutation_type == MutationType.SHRINK_LAYER:
            # Decrease size of random layer
            if config.layer_sizes:
                idx = random.randint(0, len(config.layer_sizes) - 1)
                shrink_factor = random.uniform(0.7, 0.9)
                new_size = max(config.min_layer_size, int(config.layer_sizes[idx] * shrink_factor))
                config.layer_sizes[idx] = new_size
        
        elif mutation_type == MutationType.ADD_LAYER:
            # Add new layer
            if len(config.layer_sizes) < config.max_layers:
                insert_idx = random.randint(0, len(config.layer_sizes))
                new_size = random.choice([32, 64, 128, 256])
                config.layer_sizes.insert(insert_idx, new_size)
                config.activation_types.insert(insert_idx, "relu")
                config.dropout_rates.insert(insert_idx, 0.1)
                config.batch_norm.insert(insert_idx, True)
        
        elif mutation_type == MutationType.REMOVE_LAYER:
            # Remove layer
            if len(config.layer_sizes) > config.min_layers:
                remove_idx = random.randint(0, len(config.layer_sizes) - 1)
                config.layer_sizes.pop(remove_idx)
                config.activation_types.pop(remove_idx)
                config.dropout_rates.pop(remove_idx)
                config.batch_norm.pop(remove_idx)
        
        elif mutation_type == MutationType.CHANGE_ACTIVATION:
            # Change activation function
            if config.activation_types:
                idx = random.randint(0, len(config.activation_types) - 1)
                activations = ["relu", "gelu", "swish", "leaky_relu"]
                config.activation_types[idx] = random.choice(activations)
        
        elif mutation_type == MutationType.MODIFY_DROPOUT:
            # Modify dropout rate
            if config.dropout_rates:
                idx = random.randint(0, len(config.dropout_rates) - 1)
                new_rate = random.uniform(0.05, 0.3)
                config.dropout_rates[idx] = new_rate
        
        return config
    
    def _crossover_architectures(self, parent1: ArchitectureConfig, parent2: ArchitectureConfig) -> ArchitectureConfig:
        """Create child architecture through crossover"""
        # Single-point crossover for layer sizes
        min_len = min(len(parent1.layer_sizes), len(parent2.layer_sizes))
        if min_len > 1:
            crossover_point = random.randint(1, min_len - 1)
            child_layers = parent1.layer_sizes[:crossover_point] + parent2.layer_sizes[crossover_point:]
        else:
            child_layers = parent1.layer_sizes.copy()
        
        # Create child configuration
        child = ArchitectureConfig(layer_sizes=child_layers)
        
        # Inherit other properties randomly
        for i in range(len(child_layers)):
            if i < len(parent1.activation_types) and i < len(parent2.activation_types):
                child.activation_types[i] = random.choice([parent1.activation_types[i], parent2.activation_types[i]])
            if i < len(parent1.dropout_rates) and i < len(parent2.dropout_rates):
                child.dropout_rates[i] = random.choice([parent1.dropout_rates[i], parent2.dropout_rates[i]])
            if i < len(parent1.batch_norm) and i < len(parent2.batch_norm):
                child.batch_norm[i] = random.choice([parent1.batch_norm[i], parent2.batch_norm[i]])
        
        return child
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity"""
        if len(self.population) < 2:
            return 0.0
        
        total_distance = 0.0
        comparisons = 0
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                distance = self._architecture_distance(self.population[i], self.population[j])
                total_distance += distance
                comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0.0
    
    def _architecture_distance(self, arch1: ArchitectureConfig, arch2: ArchitectureConfig) -> float:
        """Calculate distance between two architectures"""
        # Layer count difference
        layer_diff = abs(len(arch1.layer_sizes) - len(arch2.layer_sizes))
        
        # Layer size differences
        min_layers = min(len(arch1.layer_sizes), len(arch2.layer_sizes))
        size_diff = 0.0
        if min_layers > 0:
            for i in range(min_layers):
                size_diff += abs(arch1.layer_sizes[i] - arch2.layer_sizes[i]) / max(arch1.layer_sizes[i], arch2.layer_sizes[i])
            size_diff /= min_layers
        
        # Activation differences
        activation_diff = 0.0
        if min_layers > 0:
            for i in range(min_layers):
                if arch1.activation_types[i] != arch2.activation_types[i]:
                    activation_diff += 1.0
            activation_diff /= min_layers
        
        # Combine distances
        total_distance = layer_diff * 0.3 + size_diff * 0.5 + activation_diff * 0.2
        return total_distance
    
    def get_best_architecture(self) -> ArchitectureConfig:
        """Get the best architecture from current population"""
        if not self.population:
            return ArchitectureConfig()
        
        best = max(self.population, key=lambda x: x.get_total_score())
        return best
    
    def update_architecture_performance(self, config: ArchitectureConfig, metrics: PerformanceMetrics):
        """Update performance metrics for an architecture"""
        config.performance_score = metrics.compute_score()
        config.stability_score = metrics.stability
        
        # Update in population if exists
        for arch in self.population:
            if arch.layer_sizes == config.layer_sizes:
                arch.performance_score = config.performance_score
                arch.stability_score = config.stability_score
                break
    
    def get_architecture_recommendations(self, current_performance: float) -> Dict[str, Any]:
        """Get architecture recommendations based on current performance"""
        trend = self.performance_tracker.get_performance_trend()
        
        recommendations = {
            'should_evolve': self.performance_tracker.should_evolve(),
            'performance_trend': trend,
            'recommended_actions': []
        }
        
        if trend < -0.01:  # Declining performance
            recommendations['recommended_actions'].append("Consider architecture evolution")
            recommendations['recommended_actions'].append("Reduce model complexity")
        elif trend > 0.01:  # Improving performance
            recommendations['recommended_actions'].append("Consider expanding model capacity")
            recommendations['recommended_actions'].append("Maintain current architecture")
        
        return recommendations
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive NAS statistics"""
        stats = {
            'generation': self.generation,
            'population_size': len(self.population),
            'best_score': self.best_architecture.get_total_score() if self.best_architecture else 0.0,
            'population_diversity': self._calculate_diversity(),
            'performance_tracker': self.performance_tracker.get_stats()
        }
        
        if self.best_architecture:
            stats['best_architecture'] = {
                'layer_sizes': self.best_architecture.layer_sizes,
                'complexity': self.best_architecture.calculate_complexity(),
                'performance_score': self.best_architecture.performance_score
            }
        
        return stats


class GradualTransitionManager:
    """Manages gradual transitions between architectures"""
    
    def __init__(self, transition_steps: int = 1000):
        self.transition_steps = transition_steps
        self.current_step = 0
        self.is_transitioning = False
        self.old_architecture = None
        self.new_architecture = None
        
    def start_transition(self, old_config: ArchitectureConfig, new_config: ArchitectureConfig):
        """Start gradual transition to new architecture"""
        self.old_architecture = old_config
        self.new_architecture = new_config
        self.current_step = 0
        self.is_transitioning = True
        
        logger.info(f"Starting gradual transition from {old_config.layer_sizes} to {new_config.layer_sizes}")
    
    def get_current_architecture(self) -> ArchitectureConfig:
        """Get current architecture during transition"""
        if not self.is_transitioning:
            return self.new_architecture or ArchitectureConfig()
        
        # Calculate transition progress
        progress = self.current_step / self.transition_steps
        progress = min(1.0, progress)
        
        # Interpolate between architectures
        current_config = self._interpolate_architectures(self.old_architecture, self.new_architecture, progress)
        
        self.current_step += 1
        
        if self.current_step >= self.transition_steps:
            self.is_transitioning = False
            logger.info("Architecture transition completed")
        
        return current_config
    
    def _interpolate_architectures(self, old_config: ArchitectureConfig, new_config: ArchitectureConfig, progress: float) -> ArchitectureConfig:
        """Interpolate between two architectures"""
        # Simple interpolation of layer sizes
        if len(old_config.layer_sizes) == len(new_config.layer_sizes):
            interpolated_sizes = []
            for old_size, new_size in zip(old_config.layer_sizes, new_config.layer_sizes):
                interpolated_size = int(old_size + (new_size - old_size) * progress)
                interpolated_sizes.append(interpolated_size)
            
            return ArchitectureConfig(
                layer_sizes=interpolated_sizes,
                activation_types=new_config.activation_types,
                dropout_rates=new_config.dropout_rates,
                batch_norm=new_config.batch_norm
            )
        else:
            # For different layer counts, use threshold-based transition
            if progress < 0.5:
                return old_config
            else:
                return new_config
    
    def get_transition_stats(self) -> Dict[str, Any]:
        """Get transition statistics"""
        return {
            'is_transitioning': self.is_transitioning,
            'current_step': self.current_step,
            'total_steps': self.transition_steps,
            'progress': self.current_step / self.transition_steps if self.is_transitioning else 1.0
        }


def create_nas_system(input_size: int = 64, **kwargs) -> NeuralArchitectureSearch:
    """Factory function to create NAS system"""
    return NeuralArchitectureSearch(input_size=input_size, **kwargs)