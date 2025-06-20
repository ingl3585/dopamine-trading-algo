# real_time_adaptation.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class AdaptationEvent:
    timestamp: float
    event_type: str  # 'market_data', 'trade_outcome', 'drawdown', 'regime_change'
    data: Dict[str, Any]
    urgency: float  # 0.0 to 1.0


class MultiArmedBandit:
    def __init__(self, num_arms: int, exploration_rate: float = 0.1):
        self.num_arms = num_arms
        self.exploration_rate = exploration_rate
        
        # UCB (Upper Confidence Bound) parameters
        self.arm_counts = np.zeros(num_arms)
        self.arm_rewards = np.zeros(num_arms)
        self.total_pulls = 0
        
        # Contextual information
        self.context_history = deque(maxlen=1000)
        self.arm_performance_by_context = defaultdict(list)
        
    def select_arm(self, context: Optional[Dict] = None) -> int:
        self.total_pulls += 1
        
        # Pure exploration phase
        if self.total_pulls <= self.num_arms:
            return self.total_pulls - 1
        
        # Contextual consideration
        if context:
            context_key = self._context_to_key(context)
            if context_key in self.arm_performance_by_context:
                # Use contextual history for better selection
                contextual_scores = self._calculate_contextual_scores(context_key)
                if contextual_scores is not None:
                    return np.argmax(contextual_scores)
        
        # UCB algorithm
        confidence_intervals = np.sqrt(2 * np.log(self.total_pulls) / (self.arm_counts + 1e-8))
        average_rewards = self.arm_rewards / (self.arm_counts + 1e-8)
        
        ucb_scores = average_rewards + confidence_intervals
        
        # Epsilon-greedy exploration
        if np.random.random() < self.exploration_rate:
            return np.random.randint(self.num_arms)
        
        return np.argmax(ucb_scores)
    
    def update_reward(self, arm: int, reward: float, context: Optional[Dict] = None):
        self.arm_counts[arm] += 1
        self.arm_rewards[arm] += reward
        
        if context:
            context_key = self._context_to_key(context)
            self.arm_performance_by_context[context_key].append((arm, reward))
            
            # Keep only recent context history
            if len(self.arm_performance_by_context[context_key]) > 50:
                self.arm_performance_by_context[context_key] = \
                    self.arm_performance_by_context[context_key][-50:]
    
    def _context_to_key(self, context: Dict) -> str:
        # Convert context to string key for storage
        key_parts = []
        for key in sorted(context.keys()):
            if isinstance(context[key], (int, float)):
                # Bucket continuous values
                bucket = int(context[key] * 10) / 10
                key_parts.append(f"{key}:{bucket}")
            else:
                key_parts.append(f"{key}:{context[key]}")
        return "|".join(key_parts)
    
    def _calculate_contextual_scores(self, context_key: str) -> Optional[np.ndarray]:
        history = self.arm_performance_by_context.get(context_key, [])
        if len(history) < 5:
            return None
        
        # Calculate average performance per arm in this context
        arm_scores = np.zeros(self.num_arms)
        arm_context_counts = np.zeros(self.num_arms)
        
        for arm, reward in history:
            arm_scores[arm] += reward
            arm_context_counts[arm] += 1
        
        # Only consider arms with sufficient context data
        valid_arms = arm_context_counts >= 2
        if not np.any(valid_arms):
            return None
        
        # Calculate contextual average
        contextual_averages = np.zeros(self.num_arms)
        for i in range(self.num_arms):
            if arm_context_counts[i] > 0:
                contextual_averages[i] = arm_scores[i] / arm_context_counts[i]
        
        return contextual_averages


class OnlineLearner:
    def __init__(self, model_dim: int = 64, learning_rate: float = 0.001):
        self.model_dim = model_dim
        self.learning_rate = learning_rate
        
        # Simple online model for immediate updates
        self.weights = torch.zeros(model_dim, dtype=torch.float64, requires_grad=True)
        self.bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        
        # Adaptive learning rate
        self.optimizer = optim.Adam([self.weights, self.bias], lr=learning_rate)
        
        # Gradient tracking
        self.gradient_history = deque(maxlen=100)
        self.loss_history = deque(maxlen=100)
        
        # Emergency learning mode
        self.emergency_mode = False
        self.emergency_threshold = -0.5
        self.emergency_learning_rate = 0.01
        
    def quick_update(self, features: torch.Tensor, target: float, urgency: float = 1.0):
        """Immediate model update for real-time adaptation"""
        
        # Adjust learning rate based on urgency
        current_lr = self.learning_rate * (1.0 + urgency * 2.0)
        
        # Check for emergency mode
        if target < self.emergency_threshold:
            self.emergency_mode = True
            current_lr = self.emergency_learning_rate
        
        # Forward pass
        prediction = torch.dot(self.weights, features) + self.bias
        loss = (prediction - target) ** 2
        
        # Backward pass with adjusted learning rate
        self.optimizer.zero_grad()
        loss.backward()
        
        # Scale gradients by urgency
        with torch.no_grad():
            for param in [self.weights, self.bias]:
                if param.grad is not None:
                    param.grad *= (1.0 + urgency)
        
        # Update with temporary learning rate adjustment
        for group in self.optimizer.param_groups:
            group['lr'] = current_lr
        
        self.optimizer.step()
        
        # Reset learning rate
        for group in self.optimizer.param_groups:
            group['lr'] = self.learning_rate
        
        # Track statistics
        self.loss_history.append(float(loss))
        
        if self.weights.grad is not None:
            grad_norm = torch.norm(self.weights.grad).item()
            self.gradient_history.append(grad_norm)
        
        # Exit emergency mode if performance improves
        if self.emergency_mode and len(self.loss_history) >= 5:
            recent_loss = np.mean(list(self.loss_history)[-5:])
            if recent_loss < abs(self.emergency_threshold) * 0.5:
                self.emergency_mode = False
    
    def predict(self, features: torch.Tensor) -> float:
        with torch.no_grad():
            return float(torch.dot(self.weights, features) + self.bias)
    
    def get_adaptation_stats(self) -> Dict:
        return {
            'emergency_mode': self.emergency_mode,
            'recent_loss': np.mean(list(self.loss_history)[-10:]) if self.loss_history else 0.0,
            'gradient_norm': np.mean(list(self.gradient_history)[-10:]) if self.gradient_history else 0.0,
            'learning_stability': 1.0 - (np.std(list(self.loss_history)[-20:]) / 
                                       (np.mean(list(self.loss_history)[-20:]) + 1e-8)) 
                                  if len(self.loss_history) >= 20 else 0.0
        }


class StrategySelector:
    def __init__(self, num_strategies: int = 5):
        self.num_strategies = num_strategies
        self.bandit = MultiArmedBandit(num_strategies)
        
        # Strategy definitions
        self.strategies = {
            0: 'conservative',    # Lower risk, lower return
            1: 'aggressive',      # Higher risk, higher return  
            2: 'momentum',        # Trend following
            3: 'mean_reversion',  # Counter-trend
            4: 'adaptive'         # Dynamic based on conditions
        }
        
        # Performance tracking per strategy
        self.strategy_performance = {i: deque(maxlen=50) for i in range(num_strategies)}
        self.current_strategy = 0
        
    def select_strategy(self, market_context: Dict) -> int:
        # Create bandit context
        bandit_context = {
            'volatility': market_context.get('volatility', 0.5),
            'trend_strength': market_context.get('trend_strength', 0.5),
            'volume_regime': market_context.get('volume_regime', 0.5),
            'time_of_day': market_context.get('time_of_day', 0.5)
        }
        
        selected_strategy = self.bandit.select_arm(bandit_context)
        self.current_strategy = selected_strategy
        
        return selected_strategy
    
    def update_strategy_performance(self, strategy_id: int, outcome: float, 
                                  market_context: Dict):
        # Update bandit
        bandit_context = {
            'volatility': market_context.get('volatility', 0.5),
            'trend_strength': market_context.get('trend_strength', 0.5),
            'volume_regime': market_context.get('volume_regime', 0.5),
            'time_of_day': market_context.get('time_of_day', 0.5)
        }
        
        self.bandit.update_reward(strategy_id, outcome, bandit_context)
        
        # Track strategy performance
        self.strategy_performance[strategy_id].append(outcome)
    
    def get_strategy_stats(self) -> Dict:
        stats = {}
        
        for strategy_id, name in self.strategies.items():
            performance = list(self.strategy_performance[strategy_id])
            if performance:
                stats[name] = {
                    'avg_performance': np.mean(performance),
                    'volatility': np.std(performance),
                    'sharpe_ratio': np.mean(performance) / (np.std(performance) + 1e-8),
                    'total_trades': len(performance),
                    'selection_probability': self.bandit.arm_counts[strategy_id] / 
                                           (self.bandit.total_pulls + 1e-8)
                }
            else:
                stats[name] = {'avg_performance': 0, 'total_trades': 0}
        
        return stats


class UncertaintyQuantifier:
    def __init__(self, ensemble_size: int = 5):
        self.ensemble_size = ensemble_size
        self.prediction_history = deque(maxlen=200)
        self.confidence_calibration = deque(maxlen=100)
        
    def estimate_uncertainty(self, features: torch.Tensor, 
                           ensemble_predictions: List[float]) -> Tuple[float, float]:
        """
        Returns: (prediction_mean, uncertainty_estimate)
        """
        if not ensemble_predictions:
            return 0.0, 1.0  # High uncertainty for no predictions
        
        mean_prediction = np.mean(ensemble_predictions)
        prediction_std = np.std(ensemble_predictions)
        
        # Epistemic uncertainty (model uncertainty)
        epistemic_uncertainty = prediction_std
        
        # Aleatoric uncertainty (data uncertainty) - estimated from historical variance
        aleatoric_uncertainty = self._estimate_aleatoric_uncertainty()
        
        # Total uncertainty
        total_uncertainty = np.sqrt(epistemic_uncertainty**2 + aleatoric_uncertainty**2)
        
        return float(mean_prediction), float(total_uncertainty)
    
    def _estimate_aleatoric_uncertainty(self) -> float:
        if len(self.prediction_history) < 10:
            return 0.5  # Default uncertainty
        
        # Calculate variance in recent prediction errors
        recent_errors = [abs(pred - actual) for pred, actual in self.prediction_history[-20:]]
        return np.std(recent_errors) if recent_errors else 0.5
    
    def update_prediction_history(self, prediction: float, actual: float):
        self.prediction_history.append((prediction, actual))
    
    def calibrate_confidence(self, predicted_confidence: float, actual_outcome: float):
        """Update confidence calibration based on actual outcomes"""
        self.confidence_calibration.append((predicted_confidence, actual_outcome))
    
    def get_calibrated_confidence(self, raw_confidence: float) -> float:
        """Return calibrated confidence based on historical performance"""
        if len(self.confidence_calibration) < 10:
            return raw_confidence
        
        # Find similar confidence levels in history
        similar_confidences = [(conf, outcome) for conf, outcome in self.confidence_calibration
                             if abs(conf - raw_confidence) < 0.1]
        
        if not similar_confidences:
            return raw_confidence
        
        # Calculate actual success rate at this confidence level
        outcomes = [outcome for _, outcome in similar_confidences]
        actual_success_rate = np.mean([1 if o > 0 else 0 for o in outcomes])
        
        # Adjust confidence based on calibration
        calibration_factor = actual_success_rate / (raw_confidence + 1e-8)
        calibrated_confidence = raw_confidence * calibration_factor
        
        return np.clip(calibrated_confidence, 0.0, 1.0)


class RealTimeAdaptationEngine:
    def __init__(self, model_dim: int = 64):
        self.model_dim = model_dim
        
        # Core components
        self.online_learner = OnlineLearner(model_dim)
        self.strategy_selector = StrategySelector()
        self.uncertainty_quantifier = UncertaintyQuantifier()
        
        # Event processing
        self.adaptation_queue = deque(maxlen=1000)
        self.processing_stats = {
            'events_processed': 0,
            'urgent_adaptations': 0,
            'strategy_switches': 0,
            'emergency_activations': 0
        }
        
        # Performance monitoring
        self.adaptation_performance = deque(maxlen=100)
        self.last_adaptation_time = time.time()
        
    def process_market_event(self, event_type: str, data: Dict[str, Any], 
                           urgency: float = 0.5):
        """Process real-time market events for immediate adaptation"""
        
        event = AdaptationEvent(
            timestamp=time.time(),
            event_type=event_type,
            data=data,
            urgency=urgency
        )
        
        self.adaptation_queue.append(event)
        
        # Immediate processing for high urgency events
        if urgency > 0.8:
            self._process_urgent_event(event)
    
    def _process_urgent_event(self, event: AdaptationEvent):
        """Immediate processing for urgent events"""
        self.processing_stats['urgent_adaptations'] += 1
        
        if event.event_type == 'drawdown':
            # Emergency drawdown response
            drawdown_size = event.data.get('drawdown_pct', 0)
            if drawdown_size > 0.05:  # 5% drawdown triggers emergency mode
                self._activate_emergency_mode(drawdown_size)
        
        elif event.event_type == 'regime_change':
            # Rapid strategy adaptation for regime changes
            new_regime = event.data.get('new_regime', {})
            self._adapt_to_regime_change(new_regime)
        
        elif event.event_type == 'trade_outcome':
            # Immediate learning from trade results
            outcome = event.data.get('pnl', 0)
            features = event.data.get('features', torch.zeros(self.model_dim, dtype=torch.float64))
            
            # Quick model update
            self.online_learner.quick_update(features, outcome, event.urgency)
    
    def _activate_emergency_mode(self, drawdown_size: float):
        """Activate emergency learning protocols"""
        self.processing_stats['emergency_activations'] += 1
        
        # Force conservative strategy selection
        conservative_context = {
            'volatility': 1.0,  # High volatility context
            'trend_strength': 0.0,  # No trend
            'volume_regime': 0.5,
            'time_of_day': 0.5
        }
        
        # Override strategy selection temporarily
        emergency_strategy = 0  # Conservative strategy
        self.strategy_selector.current_strategy = emergency_strategy
        
        logger.warning(f"Emergency mode activated - Drawdown: {drawdown_size:.2%}")
    
    def _adapt_to_regime_change(self, new_regime: Dict):
        """Rapid adaptation to regime changes"""
        
        # Create market context from regime
        market_context = {
            'volatility': self._regime_to_numeric(new_regime.get('volatility_regime', 'medium')),
            'trend_strength': self._regime_to_numeric(new_regime.get('trend_regime', 'ranging')),
            'volume_regime': new_regime.get('confidence', 0.5),
            'time_of_day': 0.5  # Default
        }
        
        # Select new strategy based on regime
        old_strategy = self.strategy_selector.current_strategy
        new_strategy = self.strategy_selector.select_strategy(market_context)
        
        if new_strategy != old_strategy:
            self.processing_stats['strategy_switches'] += 1
            logger.info(f"Strategy switch: {old_strategy} -> {new_strategy} due to regime change")
    
    def get_adaptation_decision(self, current_features: torch.Tensor, 
                              market_context: Dict) -> Dict[str, Any]:
        """Get adaptation-informed trading decision"""
        
        # Strategy selection
        selected_strategy = self.strategy_selector.select_strategy(market_context)
        
        # Online prediction
        online_prediction = self.online_learner.predict(current_features)
        
        # Uncertainty estimation
        uncertainty = self._estimate_decision_uncertainty(current_features, online_prediction)
        
        # Calibrated confidence
        raw_confidence = 1.0 / (1.0 + uncertainty)
        calibrated_confidence = self.uncertainty_quantifier.get_calibrated_confidence(raw_confidence)
        
        # Adaptation stats
        adaptation_stats = self.online_learner.get_adaptation_stats()
        
        return {
            'selected_strategy': selected_strategy,
            'strategy_name': self.strategy_selector.strategies[selected_strategy],
            'online_prediction': online_prediction,
            'uncertainty': uncertainty,
            'calibrated_confidence': calibrated_confidence,
            'emergency_mode': adaptation_stats['emergency_mode'],
            'adaptation_quality': adaptation_stats['learning_stability'],
            'processing_stats': self.processing_stats.copy()
        }
    
    def _estimate_decision_uncertainty(self, features: torch.Tensor, 
                                     prediction: float) -> float:
        """Estimate uncertainty for current decision"""
        
        # Simple uncertainty based on feature magnitude and prediction confidence
        feature_uncertainty = torch.std(features).item()
        prediction_uncertainty = abs(prediction) if abs(prediction) < 0.5 else 0.5
        
        return (feature_uncertainty + prediction_uncertainty) / 2.0
    
    def _regime_to_numeric(self, regime: str) -> float:
        """Convert regime strings to numeric values"""
        mappings = {
            'low': 0.2, 'medium': 0.5, 'high': 0.8,
            'ranging': 0.2, 'transitional': 0.5, 'trending': 0.8,
            'conservative': 0.2, 'aggressive': 0.8, 'adaptive': 0.5
        }
        return mappings.get(regime, 0.5)
    
    def update_from_outcome(self, outcome: float, context: Dict):
        """Update adaptation engine from trading outcome"""
        
        # Update strategy performance
        current_strategy = self.strategy_selector.current_strategy
        self.strategy_selector.update_strategy_performance(
            current_strategy, outcome, context
        )
        
        # Update uncertainty calibration
        predicted_confidence = context.get('predicted_confidence', 0.5)
        self.uncertainty_quantifier.calibrate_confidence(predicted_confidence, outcome)
        
        # Track adaptation performance
        self.adaptation_performance.append(outcome)
        
        # Process any queued events
        self._process_adaptation_queue()
    
    def _process_adaptation_queue(self):
        """Process queued adaptation events"""
        processed_count = 0
        
        while self.adaptation_queue and processed_count < 10:  # Limit processing per call
            event = self.adaptation_queue.popleft()
            
            # Process based on event type
            if event.event_type == 'market_data':
                self._process_market_data_event(event)
            elif event.event_type == 'trade_outcome':
                self._process_trade_outcome_event(event)
            
            processed_count += 1
            self.processing_stats['events_processed'] += 1
    
    def _process_market_data_event(self, event: AdaptationEvent):
        """Process market data events for learning"""
        # Extract features if available
        features = event.data.get('features')
        target = event.data.get('target')
        
        if features is not None and target is not None:
            self.online_learner.quick_update(features, target, event.urgency * 0.5)
    
    def _process_trade_outcome_event(self, event: AdaptationEvent):
        """Process trade outcome events"""
        outcome = event.data.get('pnl', 0)
        prediction = event.data.get('prediction', 0)
        
        # Update prediction history for uncertainty calibration
        self.uncertainty_quantifier.update_prediction_history(prediction, outcome)
    
    def get_comprehensive_stats(self) -> Dict:
        """Get comprehensive adaptation statistics"""
        return {
            'online_learning': self.online_learner.get_adaptation_stats(),
            'strategy_selection': self.strategy_selector.get_strategy_stats(),
            'processing_stats': self.processing_stats.copy(),
            'recent_performance': {
                'mean': np.mean(list(self.adaptation_performance)) if self.adaptation_performance else 0.0,
                'std': np.std(list(self.adaptation_performance)) if self.adaptation_performance else 0.0,
                'count': len(self.adaptation_performance)
            },
            'queue_status': {
                'queued_events': len(self.adaptation_queue),
                'avg_urgency': np.mean([e.urgency for e in self.adaptation_queue]) if self.adaptation_queue else 0.0
            }
        }