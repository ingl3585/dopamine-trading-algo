# multi_objective_optimizer.py

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import time

from .temporal_reward_memory import RewardContext, RewardSignal

logger = logging.getLogger(__name__)

@dataclass
class OptimizationObjective:
    """Represents a single optimization objective"""
    name: str
    weight: float
    target_value: float
    current_value: float
    importance: float
    constraint_type: str  # 'maximize', 'minimize', 'target'
    tolerance: float = 0.1

@dataclass
class OptimizationResult:
    """Result of multi-objective optimization"""
    optimized_reward: Dict[str, float]
    objective_scores: Dict[str, float]
    pareto_efficiency: float
    trade_off_analysis: Dict[str, Any]
    optimization_success: bool
    explanation: str

class MultiObjectiveOptimizer:
    """
    Multi-objective optimizer for reward signals.
    
    Responsibilities:
    - Balance multiple competing objectives
    - Optimize reward components for best overall outcome
    - Handle trade-offs between objectives
    - Adapt objective weights based on performance
    - Generate Pareto-optimal solutions
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize objectives
        self.objectives = self._initialize_objectives(config)
        
        # Optimization parameters
        self.learning_rate = config.get('mo_learning_rate', 0.01)
        self.momentum = config.get('mo_momentum', 0.9)
        self.adaptation_rate = config.get('mo_adaptation_rate', 0.05)
        self.constraint_penalty = config.get('mo_constraint_penalty', 0.1)
        
        # Pareto frontier tracking
        self.pareto_solutions = deque(maxlen=100)
        self.optimization_history = deque(maxlen=500)
        
        # Optimization state
        self.objective_momentum = {}
        self.adaptation_stats = {
            'total_optimizations': 0,
            'pareto_improvements': 0,
            'constraint_violations': 0,
            'avg_optimization_time': 0.0
        }
        
        # Performance tracking
        self.objective_performance = {}
        for obj_name in self.objectives:
            self.objective_performance[obj_name] = deque(maxlen=100)
            self.objective_momentum[obj_name] = 0.0
        
        logger.info("Multi-objective optimizer initialized")
    
    def _initialize_objectives(self, config: Dict[str, Any]) -> Dict[str, OptimizationObjective]:
        """Initialize optimization objectives"""
        objectives = {}
        
        # Profitability objective
        objectives['profitability'] = OptimizationObjective(
            name='profitability',
            weight=config.get('profitability_weight', 0.4),
            target_value=config.get('profitability_target', 0.1),
            current_value=0.0,
            importance=1.0,
            constraint_type='maximize',
            tolerance=0.05
        )
        
        # Risk management objective
        objectives['risk_control'] = OptimizationObjective(
            name='risk_control',
            weight=config.get('risk_weight', 0.25),
            target_value=config.get('risk_target', 0.05),
            current_value=0.0,
            importance=0.9,
            constraint_type='minimize',
            tolerance=0.02
        )
        
        # Learning efficiency objective
        objectives['learning_efficiency'] = OptimizationObjective(
            name='learning_efficiency',
            weight=config.get('learning_weight', 0.2),
            target_value=config.get('learning_target', 0.8),
            current_value=0.0,
            importance=0.8,
            constraint_type='maximize',
            tolerance=0.1
        )
        
        # Exploration balance objective
        objectives['exploration_balance'] = OptimizationObjective(
            name='exploration_balance',
            weight=config.get('exploration_weight', 0.15),
            target_value=config.get('exploration_target', 0.3),
            current_value=0.0,
            importance=0.7,
            constraint_type='target',
            tolerance=0.1
        )
        
        return objectives
    
    def optimize_reward(self, reward_signal: RewardSignal, 
                       context: RewardContext) -> Dict[str, Any]:
        """
        Optimize reward signal using multi-objective optimization
        
        Args:
            reward_signal: Input reward signal
            context: Current reward context
            
        Returns:
            Optimized reward components
        """
        start_time = time.time()
        
        try:
            # Update objective current values
            self._update_objective_values(reward_signal, context)
            
            # Calculate objective scores
            objective_scores = self._calculate_objective_scores()
            
            # Perform multi-objective optimization
            optimization_result = self._perform_optimization(reward_signal, objective_scores)
            
            # Update Pareto frontier
            self._update_pareto_frontier(optimization_result)
            
            # Adapt objective weights
            self._adapt_objective_weights(optimization_result)
            
            # Record optimization
            optimization_time = time.time() - start_time
            self._record_optimization(optimization_result, optimization_time)
            
            return {
                'base_reward': optimization_result.optimized_reward['base_reward'],
                'surprise_bonus': optimization_result.optimized_reward['surprise_bonus'],
                'temporal_adjustment': optimization_result.optimized_reward['temporal_adjustment'],
                'total_reward': optimization_result.optimized_reward['total_reward'],
                'confidence': optimization_result.optimized_reward['confidence'],
                'objective_scores': optimization_result.objective_scores,
                'pareto_efficiency': optimization_result.pareto_efficiency,
                'explanation': optimization_result.explanation
            }
            
        except Exception as e:
            logger.error(f"Error in multi-objective optimization: {e}")
            return {
                'base_reward': reward_signal.base_reward,
                'surprise_bonus': reward_signal.surprise_bonus,
                'temporal_adjustment': reward_signal.temporal_adjustment,
                'total_reward': reward_signal.total_reward,
                'confidence': reward_signal.confidence,
                'objective_scores': {},
                'pareto_efficiency': 0.5,
                'explanation': 'Optimization failed'
            }
    
    def _update_objective_values(self, reward_signal: RewardSignal, context: RewardContext):
        """Update current values for all objectives"""
        try:
            # Profitability
            self.objectives['profitability'].current_value = reward_signal.total_reward
            
            # Risk control (inverse of risk)
            risk_level = self._calculate_risk_level(context)
            self.objectives['risk_control'].current_value = 1.0 - risk_level
            
            # Learning efficiency
            learning_efficiency = self._calculate_learning_efficiency(reward_signal, context)
            self.objectives['learning_efficiency'].current_value = learning_efficiency
            
            # Exploration balance
            exploration_level = self._calculate_exploration_level(reward_signal, context)
            self.objectives['exploration_balance'].current_value = exploration_level
            
        except Exception as e:
            logger.error(f"Error updating objective values: {e}")
    
    def _calculate_risk_level(self, context: RewardContext) -> float:
        """Calculate current risk level"""
        try:
            risk_factors = []
            
            # Position size risk
            if abs(context.position_size) > 5.0:
                risk_factors.append(0.3)
            
            # Volatility risk
            if context.volatility > 0.03:
                risk_factors.append(0.2)
            
            # Confidence risk
            if context.confidence < 0.4:
                risk_factors.append(0.2)
            
            # Account risk
            if context.account_balance > 0:
                account_risk = abs(context.unrealized_pnl) / context.account_balance
                if account_risk > 0.05:
                    risk_factors.append(account_risk)
            
            return sum(risk_factors) / max(1, len(risk_factors))
            
        except Exception as e:
            logger.error(f"Error calculating risk level: {e}")
            return 0.3
    
    def _calculate_learning_efficiency(self, reward_signal: RewardSignal, 
                                     context: RewardContext) -> float:
        """Calculate learning efficiency"""
        try:
            # Base efficiency from reward signal confidence
            base_efficiency = reward_signal.confidence
            
            # Efficiency from surprise learning
            surprise_efficiency = min(1.0, reward_signal.surprise_bonus * 5)
            
            # Efficiency from temporal patterns
            temporal_efficiency = min(1.0, abs(reward_signal.temporal_adjustment) * 2)
            
            # Context efficiency
            context_efficiency = context.confidence
            
            # Combine efficiencies
            overall_efficiency = (
                base_efficiency * 0.4 +
                surprise_efficiency * 0.2 +
                temporal_efficiency * 0.2 +
                context_efficiency * 0.2
            )
            
            return overall_efficiency
            
        except Exception as e:
            logger.error(f"Error calculating learning efficiency: {e}")
            return 0.5
    
    def _calculate_exploration_level(self, reward_signal: RewardSignal, 
                                   context: RewardContext) -> float:
        """Calculate exploration level"""
        try:
            # Exploration indicators
            exploration_factors = []
            
            # Novelty exploration
            if reward_signal.novelty_bonus > 0.01:
                exploration_factors.append(0.3)
            
            # Surprise exploration
            if reward_signal.surprise_bonus > 0.02:
                exploration_factors.append(0.2)
            
            # Confidence exploration (moderate confidence indicates exploration)
            if 0.4 <= context.confidence <= 0.7:
                exploration_factors.append(0.2)
            
            # Position size exploration
            if 1.0 <= abs(context.position_size) <= 3.0:
                exploration_factors.append(0.1)
            
            return sum(exploration_factors)
            
        except Exception as e:
            logger.error(f"Error calculating exploration level: {e}")
            return 0.3
    
    def _calculate_objective_scores(self) -> Dict[str, float]:
        """Calculate scores for all objectives"""
        objective_scores = {}
        
        try:
            for obj_name, objective in self.objectives.items():
                if objective.constraint_type == 'maximize':
                    score = objective.current_value / max(0.1, objective.target_value)
                elif objective.constraint_type == 'minimize':
                    score = objective.target_value / max(0.1, objective.current_value)
                else:  # target
                    error = abs(objective.current_value - objective.target_value)
                    score = max(0.0, 1.0 - error / objective.tolerance)
                
                objective_scores[obj_name] = np.clip(score, 0.0, 2.0)
                
                # Track performance
                self.objective_performance[obj_name].append(score)
            
            return objective_scores
            
        except Exception as e:
            logger.error(f"Error calculating objective scores: {e}")
            return {}
    
    def _perform_optimization(self, reward_signal: RewardSignal, 
                            objective_scores: Dict[str, float]) -> OptimizationResult:
        """Perform multi-objective optimization"""
        try:
            # Calculate weighted objective function
            weighted_score = 0.0
            for obj_name, score in objective_scores.items():
                weight = self.objectives[obj_name].weight
                importance = self.objectives[obj_name].importance
                weighted_score += weight * importance * score
            
            # Apply optimization adjustments
            optimization_factor = self._calculate_optimization_factor(weighted_score)
            
            # Optimize reward components
            optimized_reward = {
                'base_reward': reward_signal.base_reward * optimization_factor,
                'surprise_bonus': reward_signal.surprise_bonus * self._get_surprise_factor(objective_scores),
                'temporal_adjustment': reward_signal.temporal_adjustment * self._get_temporal_factor(objective_scores),
                'confidence': min(1.0, reward_signal.confidence * self._get_confidence_factor(objective_scores))
            }
            
            # Calculate total optimized reward
            optimized_reward['total_reward'] = (
                optimized_reward['base_reward'] +
                optimized_reward['surprise_bonus'] +
                optimized_reward['temporal_adjustment']
            )
            
            # Calculate Pareto efficiency
            pareto_efficiency = self._calculate_pareto_efficiency(objective_scores)
            
            # Generate trade-off analysis
            trade_off_analysis = self._analyze_trade_offs(objective_scores)
            
            # Check optimization success
            optimization_success = weighted_score > 0.6
            
            # Generate explanation
            explanation = self._generate_optimization_explanation(
                optimized_reward, objective_scores, optimization_factor
            )
            
            return OptimizationResult(
                optimized_reward=optimized_reward,
                objective_scores=objective_scores,
                pareto_efficiency=pareto_efficiency,
                trade_off_analysis=trade_off_analysis,
                optimization_success=optimization_success,
                explanation=explanation
            )
            
        except Exception as e:
            logger.error(f"Error performing optimization: {e}")
            return OptimizationResult(
                optimized_reward={
                    'base_reward': reward_signal.base_reward,
                    'surprise_bonus': reward_signal.surprise_bonus,
                    'temporal_adjustment': reward_signal.temporal_adjustment,
                    'total_reward': reward_signal.total_reward,
                    'confidence': reward_signal.confidence
                },
                objective_scores=objective_scores,
                pareto_efficiency=0.5,
                trade_off_analysis={},
                optimization_success=False,
                explanation="Optimization failed"
            )
    
    def _calculate_optimization_factor(self, weighted_score: float) -> float:
        """Calculate overall optimization factor"""
        try:
            # Base factor from weighted score
            base_factor = np.tanh(weighted_score - 0.5) * 0.2 + 1.0
            
            # Constraint penalties
            penalty = 0.0
            for objective in self.objectives.values():
                if objective.constraint_type == 'minimize' and objective.current_value > objective.target_value:
                    penalty += self.constraint_penalty
                elif objective.constraint_type == 'maximize' and objective.current_value < objective.target_value:
                    penalty += self.constraint_penalty * 0.5
            
            return max(0.5, base_factor - penalty)
            
        except Exception as e:
            logger.error(f"Error calculating optimization factor: {e}")
            return 1.0
    
    def _get_surprise_factor(self, objective_scores: Dict[str, float]) -> float:
        """Get surprise component optimization factor"""
        try:
            # Enhance surprise when learning efficiency is low
            learning_score = objective_scores.get('learning_efficiency', 1.0)
            if learning_score < 0.6:
                return 1.2
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"Error calculating surprise factor: {e}")
            return 1.0
    
    def _get_temporal_factor(self, objective_scores: Dict[str, float]) -> float:
        """Get temporal component optimization factor"""
        try:
            # Enhance temporal when profitability is good
            profitability_score = objective_scores.get('profitability', 1.0)
            if profitability_score > 1.2:
                return 1.1
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"Error calculating temporal factor: {e}")
            return 1.0
    
    def _get_confidence_factor(self, objective_scores: Dict[str, float]) -> float:
        """Get confidence optimization factor"""
        try:
            # Boost confidence when risk control is good
            risk_score = objective_scores.get('risk_control', 1.0)
            if risk_score > 1.1:
                return 1.05
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"Error calculating confidence factor: {e}")
            return 1.0
    
    def _calculate_pareto_efficiency(self, objective_scores: Dict[str, float]) -> float:
        """Calculate Pareto efficiency of solution"""
        try:
            if len(self.pareto_solutions) < 2:
                return 0.5
            
            # Compare with existing Pareto solutions
            current_solution = list(objective_scores.values())
            
            dominated_count = 0
            for pareto_solution in self.pareto_solutions:
                if self._dominates(pareto_solution, current_solution):
                    dominated_count += 1
            
            # Pareto efficiency = 1 - fraction dominated
            efficiency = 1.0 - (dominated_count / len(self.pareto_solutions))
            
            return efficiency
            
        except Exception as e:
            logger.error(f"Error calculating Pareto efficiency: {e}")
            return 0.5
    
    def _dominates(self, solution1: List[float], solution2: List[float]) -> bool:
        """Check if solution1 dominates solution2"""
        try:
            if len(solution1) != len(solution2):
                return False
            
            # Solution1 dominates if it's better or equal in all objectives
            # and strictly better in at least one
            better_in_all = all(s1 >= s2 for s1, s2 in zip(solution1, solution2))
            better_in_one = any(s1 > s2 for s1, s2 in zip(solution1, solution2))
            
            return better_in_all and better_in_one
            
        except Exception as e:
            logger.error(f"Error checking dominance: {e}")
            return False
    
    def _analyze_trade_offs(self, objective_scores: Dict[str, float]) -> Dict[str, Any]:
        """Analyze trade-offs between objectives"""
        try:
            trade_offs = {}
            
            # Profitability vs Risk trade-off
            profit_score = objective_scores.get('profitability', 1.0)
            risk_score = objective_scores.get('risk_control', 1.0)
            trade_offs['profit_risk_ratio'] = profit_score / max(0.1, risk_score)
            
            # Learning vs Exploration trade-off
            learning_score = objective_scores.get('learning_efficiency', 1.0)
            exploration_score = objective_scores.get('exploration_balance', 1.0)
            trade_offs['learning_exploration_balance'] = abs(learning_score - exploration_score)
            
            # Overall balance
            scores = list(objective_scores.values())
            trade_offs['objective_balance'] = 1.0 - np.std(scores) / (np.mean(scores) + 1e-6)
            
            return trade_offs
            
        except Exception as e:
            logger.error(f"Error analyzing trade-offs: {e}")
            return {}
    
    def _update_pareto_frontier(self, optimization_result: OptimizationResult):
        """Update Pareto frontier with new solution"""
        try:
            current_solution = list(optimization_result.objective_scores.values())
            
            # Remove dominated solutions
            non_dominated = []
            for existing_solution in self.pareto_solutions:
                if not self._dominates(current_solution, existing_solution):
                    non_dominated.append(existing_solution)
            
            # Add current solution if not dominated
            is_dominated = any(self._dominates(existing, current_solution) 
                              for existing in non_dominated)
            
            if not is_dominated:
                non_dominated.append(current_solution)
                self.adaptation_stats['pareto_improvements'] += 1
            
            # Update Pareto frontier
            self.pareto_solutions.clear()
            self.pareto_solutions.extend(non_dominated)
            
        except Exception as e:
            logger.error(f"Error updating Pareto frontier: {e}")
    
    def _adapt_objective_weights(self, optimization_result: OptimizationResult):
        """Adapt objective weights based on performance"""
        try:
            if self.adaptation_stats['total_optimizations'] < 20:
                return  # Need minimum data for adaptation
            
            # Adapt weights based on objective performance
            for obj_name, objective in self.objectives.items():
                if obj_name in self.objective_performance:
                    recent_performance = list(self.objective_performance[obj_name])[-10:]
                    avg_performance = np.mean(recent_performance)
                    
                    # Increase weight for underperforming objectives
                    if avg_performance < 0.8:
                        weight_adjustment = self.adaptation_rate * (0.8 - avg_performance)
                        objective.weight = min(1.0, objective.weight + weight_adjustment)
                    
                    # Decrease weight for overperforming objectives
                    elif avg_performance > 1.2:
                        weight_adjustment = self.adaptation_rate * (avg_performance - 1.2)
                        objective.weight = max(0.1, objective.weight - weight_adjustment)
            
            # Normalize weights
            total_weight = sum(obj.weight for obj in self.objectives.values())
            for objective in self.objectives.values():
                objective.weight /= total_weight
            
        except Exception as e:
            logger.error(f"Error adapting objective weights: {e}")
    
    def _record_optimization(self, optimization_result: OptimizationResult, 
                           optimization_time: float):
        """Record optimization for analysis"""
        try:
            self.optimization_history.append({
                'timestamp': time.time(),
                'objective_scores': optimization_result.objective_scores,
                'pareto_efficiency': optimization_result.pareto_efficiency,
                'optimization_time': optimization_time,
                'success': optimization_result.optimization_success
            })
            
            # Update statistics
            self.adaptation_stats['total_optimizations'] += 1
            
            # Update average optimization time
            count = self.adaptation_stats['total_optimizations']
            self.adaptation_stats['avg_optimization_time'] = (
                (self.adaptation_stats['avg_optimization_time'] * (count - 1) + 
                 optimization_time) / count
            )
            
            # Count constraint violations
            for obj_name, objective in self.objectives.items():
                if objective.constraint_type == 'minimize':
                    if objective.current_value > objective.target_value + objective.tolerance:
                        self.adaptation_stats['constraint_violations'] += 1
                elif objective.constraint_type == 'maximize':
                    if objective.current_value < objective.target_value - objective.tolerance:
                        self.adaptation_stats['constraint_violations'] += 1
                else:  # target
                    error = abs(objective.current_value - objective.target_value)
                    if error > objective.tolerance:
                        self.adaptation_stats['constraint_violations'] += 1
            
        except Exception as e:
            logger.error(f"Error recording optimization: {e}")
    
    def _generate_optimization_explanation(self, optimized_reward: Dict[str, float],
                                         objective_scores: Dict[str, float],
                                         optimization_factor: float) -> str:
        """Generate explanation of optimization process"""
        try:
            explanations = []
            
            # Optimization factor
            if optimization_factor > 1.05:
                explanations.append("reward enhanced by optimization")
            elif optimization_factor < 0.95:
                explanations.append("reward reduced due to constraints")
            
            # Objective analysis
            for obj_name, score in objective_scores.items():
                if score > 1.2:
                    explanations.append(f"{obj_name} performing well ({score:.2f})")
                elif score < 0.8:
                    explanations.append(f"{obj_name} needs improvement ({score:.2f})")
            
            # Pareto efficiency
            pareto_eff = self._calculate_pareto_efficiency(objective_scores)
            if pareto_eff > 0.8:
                explanations.append("Pareto efficient solution")
            elif pareto_eff < 0.4:
                explanations.append("sub-optimal trade-offs")
            
            if not explanations:
                explanations.append("balanced multi-objective solution")
            
            return "Multi-objective optimization: " + ", ".join(explanations)
            
        except Exception as e:
            logger.error(f"Error generating optimization explanation: {e}")
            return "Multi-objective optimization applied"
    
    def get_optimizer_statistics(self) -> Dict[str, Any]:
        """Get optimizer statistics"""
        try:
            stats = {
                'adaptation_stats': self.adaptation_stats.copy(),
                'objective_weights': {name: obj.weight for name, obj in self.objectives.items()},
                'objective_targets': {name: obj.target_value for name, obj in self.objectives.items()},
                'pareto_frontier_size': len(self.pareto_solutions),
                'optimization_history_size': len(self.optimization_history)
            }
            
            # Add objective performance
            if self.objective_performance:
                stats['objective_performance'] = {}
                for obj_name, performance in self.objective_performance.items():
                    if performance:
                        stats['objective_performance'][obj_name] = {
                            'avg': np.mean(performance),
                            'std': np.std(performance),
                            'recent_trend': np.mean(list(performance)[-5:]) - np.mean(performance)
                        }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting optimizer statistics: {e}")
            return {}
    
    def reset_optimizer(self):
        """Reset optimizer state"""
        try:
            # Reset objectives to defaults
            self.objectives = self._initialize_objectives(self.config)
            
            # Reset tracking
            self.pareto_solutions.clear()
            self.optimization_history.clear()
            
            # Reset performance tracking
            for obj_name in self.objectives:
                self.objective_performance[obj_name].clear()
                self.objective_momentum[obj_name] = 0.0
            
            # Reset statistics
            self.adaptation_stats = {
                'total_optimizations': 0,
                'pareto_improvements': 0,
                'constraint_violations': 0,
                'avg_optimization_time': 0.0
            }
            
            logger.info("Multi-objective optimizer reset")
            
        except Exception as e:
            logger.error(f"Error resetting optimizer: {e}")