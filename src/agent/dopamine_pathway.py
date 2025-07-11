# dopamine_pathway.py

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
from dataclasses import dataclass, asdict
import time
import math

from .temporal_reward_memory import RewardContext, RewardSignal

logger = logging.getLogger(__name__)

@dataclass
class DopamineNeuron:
    """Represents a dopamine neuron with firing characteristics"""
    baseline_firing_rate: float = 0.1
    current_firing_rate: float = 0.1
    burst_threshold: float = 0.3
    pause_threshold: float = -0.2
    adaptation_rate: float = 0.95
    sensitivity: float = 1.0
    last_update: float = 0.0

@dataclass
class DopamineSignal:
    """Dopamine signal with neuromorphic characteristics"""
    firing_rate: float
    burst_intensity: float
    pause_duration: float
    prediction_error: float
    learning_signal: float
    motivation_level: float
    confidence: float
    timestamp: float

class DopaminePathway:
    """
    Neuromorphic dopamine pathway for sophisticated reward processing.
    
    Responsibilities:
    - Model dopamine neuron firing patterns
    - Process prediction errors and surprise signals
    - Generate motivation and learning signals
    - Implement reward prediction error (RPE) coding
    - Manage dopamine neuron adaptation and plasticity
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Dopamine pathway parameters
        self.num_neurons = config.get('dopamine_neurons', 100)
        self.baseline_firing = config.get('baseline_firing_rate', 0.1)
        self.burst_multiplier = config.get('burst_multiplier', 5.0)
        self.pause_multiplier = config.get('pause_multiplier', 0.01)
        self.adaptation_rate = config.get('adaptation_rate', 0.95)
        
        # Prediction error parameters
        self.rpe_learning_rate = config.get('rpe_learning_rate', 0.1)
        self.rpe_decay = config.get('rpe_decay', 0.9)
        self.surprise_sensitivity = config.get('surprise_sensitivity', 2.0)
        
        # Initialize dopamine neurons
        self.dopamine_neurons = []
        for i in range(self.num_neurons):
            neuron = DopamineNeuron(
                baseline_firing_rate=self.baseline_firing + np.random.normal(0, 0.02),
                current_firing_rate=self.baseline_firing,
                burst_threshold=0.3 + np.random.normal(0, 0.05),
                pause_threshold=-0.2 + np.random.normal(0, 0.03),
                adaptation_rate=self.adaptation_rate + np.random.normal(0, 0.01),
                sensitivity=1.0 + np.random.normal(0, 0.1)
            )
            self.dopamine_neurons.append(neuron)
        
        # Reward prediction and learning
        self.reward_predictor = RewardPredictor(config)
        self.prediction_history = deque(maxlen=1000)
        self.dopamine_history = deque(maxlen=1000)
        
        # Pathway state
        self.pathway_state = {
            'average_firing_rate': self.baseline_firing,
            'burst_activity': 0.0,
            'pause_activity': 0.0,
            'learning_rate': self.rpe_learning_rate,
            'motivation_level': 0.5,
            'adaptation_level': 0.0
        }
        
        logger.info(f"Dopamine pathway initialized with {self.num_neurons} neurons")
    
    def process_reward_signal(self, base_reward: float, surprise_bonus: float,
                            temporal_adjustment: float, context: RewardContext) -> RewardSignal:
        """
        Process reward through dopamine pathway
        
        Args:
            base_reward: Base reward from outcome
            surprise_bonus: Surprise-based reward bonus
            temporal_adjustment: Temporal memory adjustment
            context: Reward context
            
        Returns:
            Processed reward signal
        """
        try:
            # Calculate reward prediction error
            predicted_reward = self.reward_predictor.predict_reward(context)
            prediction_error = base_reward - predicted_reward
            
            # Process through dopamine neurons
            dopamine_signal = self._generate_dopamine_signal(
                prediction_error, surprise_bonus, context
            )
            
            # Calculate final reward components
            processed_reward = self._calculate_processed_reward(
                base_reward, surprise_bonus, temporal_adjustment, dopamine_signal
            )
            
            # Update reward predictor
            self.reward_predictor.update_prediction(context, base_reward)
            
            # Store history
            self._record_dopamine_activity(dopamine_signal, context)
            
            # Create reward signal
            reward_signal = RewardSignal(
                base_reward=processed_reward['base_reward'],
                temporal_adjustment=processed_reward['temporal_adjustment'],
                surprise_bonus=processed_reward['surprise_bonus'],
                novelty_bonus=processed_reward['novelty_bonus'],
                efficiency_bonus=processed_reward['efficiency_bonus'],
                risk_adjustment=processed_reward['risk_adjustment'],
                total_reward=processed_reward['total_reward'],
                confidence=dopamine_signal.confidence,
                explanation=self._generate_reward_explanation(processed_reward, dopamine_signal)
            )
            
            logger.debug(f"Dopamine processing: RPE={prediction_error:.3f}, "
                        f"firing_rate={dopamine_signal.firing_rate:.3f}, "
                        f"total_reward={reward_signal.total_reward:.3f}")
            
            return reward_signal
            
        except Exception as e:
            logger.error(f"Error processing reward signal: {e}")
            return RewardSignal(
                base_reward=base_reward,
                temporal_adjustment=temporal_adjustment,
                surprise_bonus=surprise_bonus,
                novelty_bonus=0.0,
                efficiency_bonus=0.0,
                risk_adjustment=0.0,
                total_reward=base_reward + surprise_bonus + temporal_adjustment,
                confidence=0.5,
                explanation="Error in dopamine processing"
            )
    
    def _generate_dopamine_signal(self, prediction_error: float, 
                                surprise_bonus: float, context: RewardContext) -> DopamineSignal:
        """Generate dopamine signal from prediction error"""
        try:
            current_time = time.time()
            
            # Calculate aggregate neuron response
            firing_rates = []
            burst_neurons = 0
            pause_neurons = 0
            
            for neuron in self.dopamine_neurons:
                # Calculate neuron response to prediction error
                neuron_input = (
                    prediction_error * neuron.sensitivity +
                    surprise_bonus * self.surprise_sensitivity
                )
                
                # Apply neuron dynamics
                if neuron_input > neuron.burst_threshold:
                    # Burst firing
                    neuron.current_firing_rate = neuron.baseline_firing_rate * self.burst_multiplier
                    burst_neurons += 1
                elif neuron_input < neuron.pause_threshold:
                    # Pause firing
                    neuron.current_firing_rate = neuron.baseline_firing_rate * self.pause_multiplier
                    pause_neurons += 1
                else:
                    # Tonic firing with modulation
                    modulation = np.tanh(neuron_input)
                    neuron.current_firing_rate = neuron.baseline_firing_rate * (1 + modulation)
                
                firing_rates.append(neuron.current_firing_rate)
                
                # Apply adaptation
                neuron.current_firing_rate *= neuron.adaptation_rate
                neuron.last_update = current_time
            
            # Calculate aggregate signals
            avg_firing_rate = np.mean(firing_rates)
            burst_intensity = burst_neurons / self.num_neurons
            pause_duration = pause_neurons / self.num_neurons
            
            # Calculate learning and motivation signals
            learning_signal = self._calculate_learning_signal(
                prediction_error, avg_firing_rate, burst_intensity
            )
            motivation_level = self._calculate_motivation_level(
                avg_firing_rate, burst_intensity, context
            )
            
            # Calculate confidence
            confidence = self._calculate_signal_confidence(
                prediction_error, burst_intensity, pause_duration
            )
            
            # Create dopamine signal
            dopamine_signal = DopamineSignal(
                firing_rate=avg_firing_rate,
                burst_intensity=burst_intensity,
                pause_duration=pause_duration,
                prediction_error=prediction_error,
                learning_signal=learning_signal,
                motivation_level=motivation_level,
                confidence=confidence,
                timestamp=current_time
            )
            
            # Update pathway state
            self._update_pathway_state(dopamine_signal)
            
            return dopamine_signal
            
        except Exception as e:
            logger.error(f"Error generating dopamine signal: {e}")
            return DopamineSignal(
                firing_rate=self.baseline_firing,
                burst_intensity=0.0,
                pause_duration=0.0,
                prediction_error=0.0,
                learning_signal=0.0,
                motivation_level=0.5,
                confidence=0.5,
                timestamp=time.time()
            )
    
    def _calculate_learning_signal(self, prediction_error: float, 
                                  firing_rate: float, burst_intensity: float) -> float:
        """Calculate learning signal from dopamine activity"""
        try:
            # Learning is enhanced by prediction errors and bursts
            base_learning = abs(prediction_error) * self.rpe_learning_rate
            burst_enhancement = burst_intensity * 0.5
            
            # Firing rate modulation
            firing_modulation = np.tanh((firing_rate - self.baseline_firing) * 5)
            
            learning_signal = base_learning + burst_enhancement + firing_modulation * 0.2
            
            return np.clip(learning_signal, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating learning signal: {e}")
            return 0.0
    
    def _calculate_motivation_level(self, firing_rate: float, 
                                   burst_intensity: float, context: RewardContext) -> float:
        """Calculate motivation level from dopamine activity"""
        try:
            # Base motivation from firing rate
            base_motivation = (firing_rate - self.baseline_firing) / self.baseline_firing
            
            # Burst enhancement
            burst_motivation = burst_intensity * 0.3
            
            # Context modulation
            context_motivation = 0.0
            if context.confidence > 0.7:
                context_motivation = 0.1
            elif context.confidence < 0.3:
                context_motivation = -0.1
            
            # Account balance effect
            if context.account_balance > 0:
                balance_effect = np.tanh(context.unrealized_pnl / 1000.0) * 0.1
            else:
                balance_effect = -0.1
            
            motivation = base_motivation + burst_motivation + context_motivation + balance_effect
            
            return np.clip(motivation, -1.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating motivation level: {e}")
            return 0.0
    
    def _calculate_signal_confidence(self, prediction_error: float, 
                                   burst_intensity: float, pause_duration: float) -> float:
        """Calculate confidence in dopamine signal"""
        try:
            # Confidence based on signal clarity
            signal_strength = abs(prediction_error) + burst_intensity + pause_duration
            
            # Consistency with recent signals
            consistency = self._calculate_signal_consistency()
            
            # Combine factors
            confidence = (signal_strength * 0.6 + consistency * 0.4)
            
            return np.clip(confidence, 0.1, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating signal confidence: {e}")
            return 0.5
    
    def _calculate_signal_consistency(self) -> float:
        """Calculate consistency of recent dopamine signals"""
        try:
            if len(self.dopamine_history) < 5:
                return 0.5
            
            recent_signals = list(self.dopamine_history)[-5:]
            
            # Calculate variance in firing rates
            firing_rates = [s.firing_rate for s in recent_signals]
            firing_variance = np.var(firing_rates)
            
            # Calculate variance in prediction errors
            prediction_errors = [s.prediction_error for s in recent_signals]
            error_variance = np.var(prediction_errors)
            
            # Lower variance = higher consistency
            consistency = 1.0 / (1.0 + firing_variance + error_variance)
            
            return consistency
            
        except Exception as e:
            logger.error(f"Error calculating signal consistency: {e}")
            return 0.5
    
    def _calculate_processed_reward(self, base_reward: float, surprise_bonus: float,
                                   temporal_adjustment: float, dopamine_signal: DopamineSignal) -> Dict[str, float]:
        """Calculate processed reward components"""
        try:
            # Base reward modulated by dopamine
            processed_base = base_reward * (1 + dopamine_signal.learning_signal * 0.2)
            
            # Surprise bonus enhanced by burst activity
            processed_surprise = surprise_bonus * (1 + dopamine_signal.burst_intensity * 0.5)
            
            # Temporal adjustment modulated by motivation
            processed_temporal = temporal_adjustment * (1 + dopamine_signal.motivation_level * 0.3)
            
            # Novelty bonus from dopamine activity
            novelty_bonus = dopamine_signal.burst_intensity * 0.1
            
            # Efficiency bonus from learning signal
            efficiency_bonus = dopamine_signal.learning_signal * 0.05
            
            # Risk adjustment from pause activity
            risk_adjustment = -dopamine_signal.pause_duration * 0.1
            
            # Total reward
            total_reward = (
                processed_base + processed_surprise + processed_temporal +
                novelty_bonus + efficiency_bonus + risk_adjustment
            )
            
            return {
                'base_reward': processed_base,
                'surprise_bonus': processed_surprise,
                'temporal_adjustment': processed_temporal,
                'novelty_bonus': novelty_bonus,
                'efficiency_bonus': efficiency_bonus,
                'risk_adjustment': risk_adjustment,
                'total_reward': total_reward
            }
            
        except Exception as e:
            logger.error(f"Error calculating processed reward: {e}")
            return {
                'base_reward': base_reward,
                'surprise_bonus': surprise_bonus,
                'temporal_adjustment': temporal_adjustment,
                'novelty_bonus': 0.0,
                'efficiency_bonus': 0.0,
                'risk_adjustment': 0.0,
                'total_reward': base_reward + surprise_bonus + temporal_adjustment
            }
    
    def _update_pathway_state(self, dopamine_signal: DopamineSignal):
        """Update overall pathway state"""
        try:
            # Update moving averages
            alpha = 0.1  # Learning rate for state updates
            
            self.pathway_state['average_firing_rate'] = (
                (1 - alpha) * self.pathway_state['average_firing_rate'] +
                alpha * dopamine_signal.firing_rate
            )
            
            self.pathway_state['burst_activity'] = (
                (1 - alpha) * self.pathway_state['burst_activity'] +
                alpha * dopamine_signal.burst_intensity
            )
            
            self.pathway_state['pause_activity'] = (
                (1 - alpha) * self.pathway_state['pause_activity'] +
                alpha * dopamine_signal.pause_duration
            )
            
            self.pathway_state['learning_rate'] = (
                (1 - alpha) * self.pathway_state['learning_rate'] +
                alpha * dopamine_signal.learning_signal
            )
            
            self.pathway_state['motivation_level'] = (
                (1 - alpha) * self.pathway_state['motivation_level'] +
                alpha * dopamine_signal.motivation_level
            )
            
            # Calculate adaptation level
            self.pathway_state['adaptation_level'] = self._calculate_adaptation_level()
            
        except Exception as e:
            logger.error(f"Error updating pathway state: {e}")
    
    def _calculate_adaptation_level(self) -> float:
        """Calculate adaptation level of dopamine pathway"""
        try:
            if len(self.dopamine_history) < 10:
                return 0.0
            
            # Calculate how much the pathway has adapted from baseline
            recent_signals = list(self.dopamine_history)[-10:]
            
            avg_firing = np.mean([s.firing_rate for s in recent_signals])
            adaptation = (avg_firing - self.baseline_firing) / self.baseline_firing
            
            return np.tanh(adaptation)
            
        except Exception as e:
            logger.error(f"Error calculating adaptation level: {e}")
            return 0.0
    
    def _record_dopamine_activity(self, dopamine_signal: DopamineSignal, 
                                 context: RewardContext):
        """Record dopamine activity for analysis"""
        try:
            self.dopamine_history.append(dopamine_signal)
            
            # Log significant events
            if dopamine_signal.burst_intensity > 0.5:
                logger.info(f"High dopamine burst detected: intensity={dopamine_signal.burst_intensity:.3f}")
            
            if dopamine_signal.pause_duration > 0.5:
                logger.info(f"Significant dopamine pause: duration={dopamine_signal.pause_duration:.3f}")
            
        except Exception as e:
            logger.error(f"Error recording dopamine activity: {e}")
    
    def _generate_reward_explanation(self, processed_reward: Dict[str, float],
                                   dopamine_signal: DopamineSignal) -> str:
        """Generate explanation for reward processing"""
        try:
            explanations = []
            
            if processed_reward['base_reward'] > 0.1:
                explanations.append(f"positive outcome ({processed_reward['base_reward']:.3f})")
            elif processed_reward['base_reward'] < -0.1:
                explanations.append(f"negative outcome ({processed_reward['base_reward']:.3f})")
            
            if processed_reward['surprise_bonus'] > 0.05:
                explanations.append(f"surprise bonus ({processed_reward['surprise_bonus']:.3f})")
            
            if processed_reward['temporal_adjustment'] > 0.05:
                explanations.append(f"positive temporal pattern ({processed_reward['temporal_adjustment']:.3f})")
            elif processed_reward['temporal_adjustment'] < -0.05:
                explanations.append(f"negative temporal pattern ({processed_reward['temporal_adjustment']:.3f})")
            
            if dopamine_signal.burst_intensity > 0.3:
                explanations.append(f"dopamine burst ({dopamine_signal.burst_intensity:.3f})")
            
            if dopamine_signal.pause_duration > 0.3:
                explanations.append(f"dopamine pause ({dopamine_signal.pause_duration:.3f})")
            
            if not explanations:
                explanations.append("neutral reward processing")
            
            return "Reward processed: " + ", ".join(explanations)
            
        except Exception as e:
            logger.error(f"Error generating reward explanation: {e}")
            return "Reward processed through dopamine pathway"
    
    def get_pathway_statistics(self) -> Dict[str, Any]:
        """Get dopamine pathway statistics"""
        try:
            stats = self.pathway_state.copy()
            
            # Add neuron statistics
            firing_rates = [neuron.current_firing_rate for neuron in self.dopamine_neurons]
            stats['neuron_firing_stats'] = {
                'mean': np.mean(firing_rates),
                'std': np.std(firing_rates),
                'min': np.min(firing_rates),
                'max': np.max(firing_rates)
            }
            
            # Add signal statistics
            if self.dopamine_history:
                recent_signals = list(self.dopamine_history)[-100:]
                stats['signal_stats'] = {
                    'avg_prediction_error': np.mean([s.prediction_error for s in recent_signals]),
                    'avg_burst_intensity': np.mean([s.burst_intensity for s in recent_signals]),
                    'avg_pause_duration': np.mean([s.pause_duration for s in recent_signals]),
                    'avg_learning_signal': np.mean([s.learning_signal for s in recent_signals])
                }
            
            # Add predictor statistics
            stats['predictor_stats'] = self.reward_predictor.get_statistics()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting pathway statistics: {e}")
            return {}
    
    def reset_pathway(self):
        """Reset dopamine pathway to initial state"""
        try:
            # Reset neurons
            for neuron in self.dopamine_neurons:
                neuron.current_firing_rate = neuron.baseline_firing_rate
                neuron.last_update = time.time()
            
            # Reset predictor
            self.reward_predictor.reset()
            
            # Clear history
            self.dopamine_history.clear()
            
            # Reset state
            self.pathway_state = {
                'average_firing_rate': self.baseline_firing,
                'burst_activity': 0.0,
                'pause_activity': 0.0,
                'learning_rate': self.rpe_learning_rate,
                'motivation_level': 0.5,
                'adaptation_level': 0.0
            }
            
            logger.info("Dopamine pathway reset")
            
        except Exception as e:
            logger.error(f"Error resetting pathway: {e}")


class RewardPredictor:
    """Simple reward predictor for dopamine pathway"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.weights = {}
        self.bias = 0.0
        self.learning_rate = config.get('predictor_learning_rate', 0.01)
        self.prediction_count = 0
        self.prediction_error_history = deque(maxlen=100)
    
    def predict_reward(self, context: RewardContext) -> float:
        """Predict reward based on context"""
        try:
            if not self.weights:
                return 0.0  # No prediction initially
            
            # Extract features
            features = self._extract_features(context)
            
            # Calculate prediction
            prediction = self.bias
            for feature_name, feature_value in features.items():
                if feature_name in self.weights:
                    prediction += self.weights[feature_name] * feature_value
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting reward: {e}")
            return 0.0
    
    def update_prediction(self, context: RewardContext, actual_reward: float):
        """Update predictor with new data"""
        try:
            predicted_reward = self.predict_reward(context)
            prediction_error = actual_reward - predicted_reward
            
            # Store error
            self.prediction_error_history.append(prediction_error)
            
            # Extract features
            features = self._extract_features(context)
            
            # Update weights
            for feature_name, feature_value in features.items():
                if feature_name not in self.weights:
                    self.weights[feature_name] = 0.0
                
                self.weights[feature_name] += self.learning_rate * prediction_error * feature_value
            
            # Update bias
            self.bias += self.learning_rate * prediction_error
            
            self.prediction_count += 1
            
        except Exception as e:
            logger.error(f"Error updating prediction: {e}")
    
    def _extract_features(self, context: RewardContext) -> Dict[str, float]:
        """Extract features from context"""
        try:
            features = {
                'position_size': context.position_size / 10.0,
                'volatility': context.volatility,
                'volume': min(context.volume / 10000.0, 1.0),
                'confidence': context.confidence,
                'unrealized_pnl': context.unrealized_pnl / 1000.0,
                'account_balance': context.account_balance / 100000.0
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return {}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get predictor statistics"""
        try:
            stats = {
                'prediction_count': self.prediction_count,
                'weight_count': len(self.weights),
                'bias': self.bias
            }
            
            if self.prediction_error_history:
                errors = list(self.prediction_error_history)
                stats['prediction_error_stats'] = {
                    'mean': np.mean(errors),
                    'std': np.std(errors),
                    'mae': np.mean(np.abs(errors))
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting predictor statistics: {e}")
            return {}
    
    def reset(self):
        """Reset predictor"""
        self.weights.clear()
        self.bias = 0.0
        self.prediction_count = 0
        self.prediction_error_history.clear()