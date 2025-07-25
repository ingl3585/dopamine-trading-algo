# surprise_detector.py

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import time

from .temporal_reward_memory import RewardContext

logger = logging.getLogger(__name__)

@dataclass
class SurpriseEvent:
    """Represents a detected surprise event"""
    timestamp: float
    surprise_magnitude: float
    surprise_type: str
    context: RewardContext
    expected_outcome: float
    actual_outcome: float
    novelty_score: float
    explanation: str

@dataclass
class ExpectationModel:
    """Model for predicting expected outcomes"""
    feature_weights: Dict[str, float]
    bias: float
    confidence: float
    update_count: int
    last_update: float

class SurpriseDetector:
    """
    Detects surprise events and novelty in trading outcomes.
    
    Responsibilities:
    - Detect unexpected market events and outcomes
    - Calculate surprise magnitude and novelty scores
    - Maintain prediction models for expectation
    - Generate surprise-based reward bonuses
    - Track surprise patterns over time
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Surprise detection parameters
        self.surprise_threshold = config.get('surprise_threshold', 0.3)
        self.novelty_threshold = config.get('novelty_threshold', 0.4)
        self.expectation_decay = config.get('expectation_decay', 0.95)
        self.learning_rate = config.get('surprise_learning_rate', 0.1)
        
        # Expectation models for different contexts
        self.expectation_models = {
            'price_movement': ExpectationModel({}, 0.0, 0.5, 0, time.time()),
            'volatility_change': ExpectationModel({}, 0.0, 0.5, 0, time.time()),
            'volume_change': ExpectationModel({}, 0.0, 0.5, 0, time.time()),
            'pnl_outcome': ExpectationModel({}, 0.0, 0.5, 0, time.time())
        }
        
        # Surprise history and patterns
        self.surprise_history = deque(maxlen=1000)
        self.novelty_patterns = {}
        self.surprise_statistics = {
            'total_surprises': 0,
            'high_magnitude_surprises': 0,
            'novel_events': 0,
            'avg_surprise_magnitude': 0.0
        }
        
        # Feature tracking for novelty detection
        self.feature_history = deque(maxlen=5000)
        self.feature_clusters = {}
        
        logger.info("Surprise detector initialized")
    
    def detect_surprise(self, context: RewardContext, 
                       actual_outcome: float,
                       outcome_type: str = 'pnl') -> Tuple[float, float]:
        """
        Detect surprise and novelty in trading outcome
        
        Args:
            context: Current reward context
            actual_outcome: Actual outcome that occurred
            outcome_type: Type of outcome being evaluated
            
        Returns:
            Tuple of (surprise_bonus, novelty_bonus)
        """
        try:
            # Calculate expected outcome
            expected_outcome = self._calculate_expected_outcome(context, outcome_type)
            
            # Calculate surprise magnitude
            surprise_magnitude = self._calculate_surprise_magnitude(
                expected_outcome, actual_outcome, context
            )
            
            # Calculate novelty score
            novelty_score = self._calculate_novelty_score(context)
            
            # Generate surprise event if significant
            if surprise_magnitude > self.surprise_threshold:
                surprise_event = SurpriseEvent(
                    timestamp=context.timestamp,
                    surprise_magnitude=surprise_magnitude,
                    surprise_type=outcome_type,
                    context=context,
                    expected_outcome=expected_outcome,
                    actual_outcome=actual_outcome,
                    novelty_score=novelty_score,
                    explanation=self._generate_surprise_explanation(
                        expected_outcome, actual_outcome, surprise_magnitude
                    )
                )
                
                self._record_surprise_event(surprise_event)
            
            # Update expectation models
            self._update_expectation_models(context, actual_outcome, outcome_type)
            
            # Calculate reward bonuses
            surprise_bonus = self._calculate_surprise_bonus(surprise_magnitude)
            novelty_bonus = self._calculate_novelty_bonus(novelty_score)
            
            logger.debug(f"Surprise detection: magnitude={surprise_magnitude:.3f}, "
                        f"novelty={novelty_score:.3f}, expected={expected_outcome:.3f}, "
                        f"actual={actual_outcome:.3f}")
            
            return surprise_bonus, novelty_bonus
            
        except Exception as e:
            logger.error(f"Error detecting surprise: {e}")
            return 0.0, 0.0
    
    def _calculate_expected_outcome(self, context: RewardContext, 
                                   outcome_type: str) -> float:
        """Calculate expected outcome based on context"""
        try:
            model = self.expectation_models.get(outcome_type)
            if not model or not model.feature_weights:
                # Use simple heuristics for initialization
                if outcome_type == 'pnl':
                    return context.unrealized_pnl + context.realized_pnl
                elif outcome_type == 'price_movement':
                    return 0.0  # Expect no movement
                elif outcome_type == 'volatility_change':
                    return context.volatility
                else:
                    return 0.0
            
            # Calculate prediction using feature weights
            prediction = model.bias
            
            # Add weighted features
            features = self._extract_prediction_features(context)
            for feature_name, feature_value in features.items():
                if feature_name in model.feature_weights:
                    prediction += model.feature_weights[feature_name] * feature_value
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error calculating expected outcome: {e}")
            return 0.0
    
    def _extract_prediction_features(self, context: RewardContext) -> Dict[str, float]:
        """Extract features for prediction"""
        try:
            features = {
                'price': context.price / 10000.0,  # Normalize
                'volatility': context.volatility,
                'volume': min(context.volume / 10000.0, 1.0),  # Normalize and cap
                'position_size': context.position_size / 10.0,  # Normalize
                'account_balance': context.account_balance / 100000.0,  # Normalize
                'unrealized_pnl': context.unrealized_pnl / 1000.0,  # Normalize
                'confidence': context.confidence,
                'price_momentum': self._calculate_price_momentum(context),
                'volatility_regime': self._get_volatility_regime(context.volatility),
                'volume_regime': self._get_volume_regime(context.volume),
                'time_of_day': self._get_time_of_day_feature(context.timestamp),
                'position_duration': self._estimate_position_duration(context)
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting prediction features: {e}")
            return {}
    
    def _calculate_surprise_magnitude(self, expected: float, actual: float, 
                                    context: RewardContext) -> float:
        """Calculate surprise magnitude"""
        try:
            # Base surprise from prediction error
            prediction_error = abs(actual - expected)
            
            # Normalize by context volatility or typical scale
            if context.volatility > 0:
                normalized_error = prediction_error / (context.volatility * 1000)
            else:
                normalized_error = prediction_error / 100.0  # Default scale
            
            # Apply non-linear scaling
            surprise_magnitude = np.tanh(normalized_error)
            
            # Boost surprise for extreme events
            if prediction_error > 1000:  # Large absolute error
                surprise_magnitude *= 1.5
            
            return min(1.0, surprise_magnitude)
            
        except Exception as e:
            logger.error(f"Error calculating surprise magnitude: {e}")
            return 0.0
    
    def _calculate_novelty_score(self, context: RewardContext) -> float:
        """Calculate novelty score for the current context"""
        try:
            # Extract context features
            features = self._extract_context_features(context)
            
            # Store features for novelty detection
            self.feature_history.append(features)
            
            if len(self.feature_history) < 10:
                return 0.5  # Default novelty for early experiences
            
            # Calculate novelty based on distance from past experiences
            novelty_score = self._calculate_feature_novelty(features)
            
            # Update novelty patterns
            self._update_novelty_patterns(features, novelty_score)
            
            return novelty_score
            
        except Exception as e:
            logger.error(f"Error calculating novelty score: {e}")
            return 0.0
    
    def _extract_context_features(self, context: RewardContext) -> np.ndarray:
        """Extract features for novelty detection"""
        try:
            features = [
                context.price / 10000.0,
                context.volatility,
                min(context.volume / 10000.0, 1.0),
                context.position_size / 10.0,
                context.confidence,
                hash(context.market_regime) % 100 / 100.0,  # Categorical to numeric
                hash(context.trade_action) % 100 / 100.0,
                self._get_time_of_day_feature(context.timestamp),
                self._get_volatility_regime(context.volatility),
                self._get_volume_regime(context.volume)
            ]
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error extracting context features: {e}")
            return np.array([0.0] * 10)
    
    def _calculate_feature_novelty(self, features: np.ndarray) -> float:
        """Calculate novelty based on feature distance"""
        try:
            if len(self.feature_history) < 2:
                return 0.5
            
            # Calculate distances to recent experiences
            recent_features = list(self.feature_history)[-100:]  # Last 100 experiences
            distances = []
            
            for past_features in recent_features:
                if len(past_features) == len(features):
                    distance = np.linalg.norm(features - past_features)
                    distances.append(distance)
            
            if not distances:
                return 0.5
            
            # Novelty is based on minimum distance (most similar experience)
            min_distance = min(distances)
            avg_distance = np.mean(distances)
            
            # Normalize novelty score
            novelty_score = min_distance / (avg_distance + 1e-6)
            novelty_score = np.tanh(novelty_score)
            
            return novelty_score
            
        except Exception as e:
            logger.error(f"Error calculating feature novelty: {e}")
            return 0.0
    
    def _calculate_price_momentum(self, context: RewardContext) -> float:
        """Calculate price momentum feature"""
        try:
            # Simple momentum based on recent price history
            if len(self.feature_history) < 5:
                return 0.0
            
            recent_prices = []
            for past_context in list(self.feature_history)[-5:]:
                if len(past_context) > 0:
                    recent_prices.append(past_context[0] * 10000.0)  # Denormalize
            
            if len(recent_prices) < 2:
                return 0.0
            
            # Calculate momentum as price change rate
            momentum = (context.price - recent_prices[0]) / (len(recent_prices) * context.price)
            return np.tanh(momentum * 100)  # Normalize
            
        except Exception as e:
            logger.error(f"Error calculating price momentum: {e}")
            return 0.0
    
    def _get_volatility_regime(self, volatility: float) -> float:
        """Get volatility regime indicator"""
        if volatility < 0.01:
            return 0.0  # Low volatility
        elif volatility < 0.03:
            return 0.5  # Medium volatility
        else:
            return 1.0  # High volatility
    
    def _get_volume_regime(self, volume: float) -> float:
        """Get volume regime indicator"""
        if volume < 1000:
            return 0.0  # Low volume
        elif volume < 5000:
            return 0.5  # Medium volume
        else:
            return 1.0  # High volume
    
    def _get_time_of_day_feature(self, timestamp: float) -> float:
        """Get time of day feature"""
        try:
            dt = datetime.fromtimestamp(timestamp)
            hour = dt.hour
            
            # Convert to cyclical feature
            return np.sin(2 * np.pi * hour / 24)
            
        except Exception as e:
            logger.error(f"Error getting time of day feature: {e}")
            return 0.0
    
    def _estimate_position_duration(self, context: RewardContext) -> float:
        """Estimate position duration feature"""
        try:
            # This would need position entry time information
            # For now, return a default value
            return 0.5
            
        except Exception as e:
            logger.error(f"Error estimating position duration: {e}")
            return 0.0
    
    def _update_expectation_models(self, context: RewardContext, 
                                  actual_outcome: float, outcome_type: str):
        """Update expectation models with new data"""
        try:
            model = self.expectation_models.get(outcome_type)
            if not model:
                return
            
            # Extract features
            features = self._extract_prediction_features(context)
            
            # Calculate prediction error
            predicted = self._calculate_expected_outcome(context, outcome_type)
            error = actual_outcome - predicted
            
            # Update weights using gradient descent
            for feature_name, feature_value in features.items():
                if feature_name not in model.feature_weights:
                    model.feature_weights[feature_name] = 0.0
                
                # Weight update
                model.feature_weights[feature_name] += self.learning_rate * error * feature_value
                
                # Apply decay to prevent overfitting
                model.feature_weights[feature_name] *= self.expectation_decay
            
            # Update bias
            model.bias += self.learning_rate * error
            model.bias *= self.expectation_decay
            
            # Update model metadata
            model.update_count += 1
            model.last_update = time.time()
            
            # Update confidence based on recent accuracy
            model.confidence = self._calculate_model_confidence(model, outcome_type)
            
        except Exception as e:
            logger.error(f"Error updating expectation models: {e}")
    
    def _calculate_model_confidence(self, model: ExpectationModel, 
                                   outcome_type: str) -> float:
        """Calculate confidence in expectation model"""
        try:
            # Base confidence on update count and recent accuracy
            base_confidence = min(0.9, model.update_count / 100.0)
            
            # Could calculate recent prediction accuracy here
            # For now, use simple heuristic
            confidence = base_confidence * 0.7 + 0.3  # Minimum 30% confidence
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating model confidence: {e}")
            return 0.5
    
    def _calculate_surprise_bonus(self, surprise_magnitude: float) -> float:
        """Calculate surprise bonus for reward"""
        try:
            if surprise_magnitude < self.surprise_threshold:
                return 0.0
            
            # Non-linear bonus scaling
            bonus = np.power(surprise_magnitude, 1.5)
            
            # Cap bonus
            bonus = min(0.5, bonus)
            
            return bonus
            
        except Exception as e:
            logger.error(f"Error calculating surprise bonus: {e}")
            return 0.0
    
    def _calculate_novelty_bonus(self, novelty_score: float) -> float:
        """Calculate novelty bonus for reward"""
        try:
            if novelty_score < self.novelty_threshold:
                return 0.0
            
            # Bonus proportional to novelty
            bonus = (novelty_score - self.novelty_threshold) * 0.3
            
            return min(0.3, bonus)
            
        except Exception as e:
            logger.error(f"Error calculating novelty bonus: {e}")
            return 0.0
    
    def _record_surprise_event(self, surprise_event: SurpriseEvent):
        """Record surprise event for analysis"""
        try:
            self.surprise_history.append(surprise_event)
            
            # Update statistics
            self.surprise_statistics['total_surprises'] += 1
            
            if surprise_event.surprise_magnitude > 0.7:
                self.surprise_statistics['high_magnitude_surprises'] += 1
            
            if surprise_event.novelty_score > self.novelty_threshold:
                self.surprise_statistics['novel_events'] += 1
            
            # Update average surprise magnitude
            total_magnitude = sum(s.surprise_magnitude for s in self.surprise_history)
            self.surprise_statistics['avg_surprise_magnitude'] = total_magnitude / len(self.surprise_history)
            
            logger.info(f"Surprise event recorded: {surprise_event.explanation}")
            
        except Exception as e:
            logger.error(f"Error recording surprise event: {e}")
    
    def _generate_surprise_explanation(self, expected: float, actual: float, 
                                     magnitude: float) -> str:
        """Generate explanation for surprise event"""
        try:
            difference = actual - expected
            
            if abs(difference) < 0.1:
                return f"Minor surprise: outcome close to expectation"
            elif difference > 0:
                return f"Positive surprise: outcome {difference:.2f} above expectation ({magnitude:.2f})"
            else:
                return f"Negative surprise: outcome {abs(difference):.2f} below expectation ({magnitude:.2f})"
                
        except Exception as e:
            logger.error(f"Error generating surprise explanation: {e}")
            return "Surprise event detected"
    
    def _update_novelty_patterns(self, features: np.ndarray, novelty_score: float):
        """Update novelty patterns for future reference"""
        try:
            # Simple pattern storage - could be more sophisticated
            pattern_key = str(int(novelty_score * 10))  # Discretize novelty score
            
            if pattern_key not in self.novelty_patterns:
                self.novelty_patterns[pattern_key] = []
            
            self.novelty_patterns[pattern_key].append(features)
            
            # Keep only recent patterns
            if len(self.novelty_patterns[pattern_key]) > 100:
                self.novelty_patterns[pattern_key] = self.novelty_patterns[pattern_key][-100:]
                
        except Exception as e:
            logger.error(f"Error updating novelty patterns: {e}")
    
    def get_surprise_statistics(self) -> Dict[str, Any]:
        """Get surprise detection statistics"""
        try:
            stats = self.surprise_statistics.copy()
            
            # Add additional metrics
            stats['recent_surprises'] = len([
                s for s in self.surprise_history
                if time.time() - s.timestamp < 3600  # Last hour
            ])
            
            stats['model_confidences'] = {
                outcome_type: model.confidence
                for outcome_type, model in self.expectation_models.items()
            }
            
            stats['novelty_patterns'] = len(self.novelty_patterns)
            stats['feature_history_size'] = len(self.feature_history)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting surprise statistics: {e}")
            return {}
    
    def reset_models(self):
        """Reset expectation models"""
        try:
            for model in self.expectation_models.values():
                model.feature_weights.clear()
                model.bias = 0.0
                model.confidence = 0.5
                model.update_count = 0
                model.last_update = time.time()
            
            logger.info("Expectation models reset")
            
        except Exception as e:
            logger.error(f"Error resetting models: {e}")