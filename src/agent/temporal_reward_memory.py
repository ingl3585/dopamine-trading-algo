# temporal_reward_memory.py

import logging
import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class RewardContext:
    """Context information for reward calculation"""
    timestamp: float
    market_data: Dict[str, Any]
    trade_action: str
    position_size: float
    price: float
    account_balance: float
    market_regime: str
    volatility: float
    volume: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    confidence: float = 0.0
    features: Optional[Dict[str, Any]] = None

@dataclass
class RewardSignal:
    """Reward signal with multiple components"""
    base_reward: float
    temporal_adjustment: float
    surprise_bonus: float
    novelty_bonus: float
    efficiency_bonus: float
    risk_adjustment: float
    total_reward: float
    confidence: float
    explanation: str

@dataclass
class RewardMemory:
    """Memory of past reward experiences"""
    context: RewardContext
    reward_signal: RewardSignal
    outcome_quality: float
    learning_value: float
    decay_factor: float = 1.0

class TemporalRewardMemory:
    """
    Manages temporal aspects of reward memory with sophisticated decay and recall.
    
    Responsibilities:
    - Store reward experiences with temporal context
    - Apply sophisticated decay functions
    - Retrieve relevant past experiences
    - Calculate temporal reward adjustments
    - Maintain memory consolidation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Memory parameters
        self.max_memory_size = config.get('max_reward_memory', 10000)
        self.short_term_window = config.get('short_term_minutes', 30)
        self.medium_term_window = config.get('medium_term_hours', 6)
        self.long_term_window = config.get('long_term_days', 30)
        
        # Decay parameters
        self.base_decay_rate = config.get('base_decay_rate', 0.001)
        self.importance_threshold = config.get('importance_threshold', 0.7)
        self.consolidation_interval = config.get('consolidation_minutes', 60)
        
        # Memory stores
        self.short_term_memory = deque(maxlen=1000)
        self.medium_term_memory = deque(maxlen=5000)
        self.long_term_memory = deque(maxlen=10000)
        
        # Consolidated patterns
        self.reward_patterns = {}
        self.temporal_correlations = {}
        
        # Memory statistics
        self.total_memories_stored = 0
        self.last_consolidation = time.time()
        
        logger.info("Temporal reward memory initialized")
    
    def store_reward_experience(self, context: RewardContext, 
                               reward_signal: RewardSignal,
                               outcome_quality: float) -> bool:
        """
        Store a reward experience in temporal memory
        
        Args:
            context: Context of the reward experience
            reward_signal: The reward signal generated
            outcome_quality: Quality of the actual outcome (0-1)
            
        Returns:
            bool: True if stored successfully
        """
        try:
            # Calculate learning value
            learning_value = self._calculate_learning_value(
                reward_signal, outcome_quality
            )
            
            # Create memory entry
            memory = RewardMemory(
                context=context,
                reward_signal=reward_signal,
                outcome_quality=outcome_quality,
                learning_value=learning_value
            )
            
            # Store in appropriate memory based on importance
            if learning_value > self.importance_threshold:
                self.long_term_memory.append(memory)
                logger.debug(f"Stored high-value memory (learning_value: {learning_value:.3f})")
            elif learning_value > 0.4:
                self.medium_term_memory.append(memory)
            else:
                self.short_term_memory.append(memory)
            
            self.total_memories_stored += 1
            
            # Periodic consolidation
            if time.time() - self.last_consolidation > self.consolidation_interval * 60:
                self._consolidate_memories()
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing reward experience: {e}")
            return False
    
    def get_temporal_adjustment(self, context: RewardContext) -> float:
        """
        Calculate temporal reward adjustment based on past experiences
        
        Args:
            context: Current reward context
            
        Returns:
            float: Temporal adjustment factor (-1 to 1)
        """
        try:
            # Get relevant memories
            similar_memories = self._retrieve_similar_memories(context)
            
            if not similar_memories:
                return 0.0
            
            # Calculate temporal patterns
            recent_performance = self._analyze_recent_performance(similar_memories)
            temporal_trend = self._calculate_temporal_trend(similar_memories)
            context_similarity = self._calculate_context_similarity(context, similar_memories)
            
            # Combine factors for temporal adjustment
            adjustment = (
                recent_performance * 0.4 +
                temporal_trend * 0.3 +
                context_similarity * 0.3
            )
            
            # Normalize to [-1, 1] range
            adjustment = np.tanh(adjustment)
            
            logger.debug(f"Temporal adjustment: {adjustment:.3f} "
                        f"(recent: {recent_performance:.3f}, "
                        f"trend: {temporal_trend:.3f}, "
                        f"similarity: {context_similarity:.3f})")
            
            return adjustment
            
        except Exception as e:
            logger.error(f"Error calculating temporal adjustment: {e}")
            return 0.0
    
    def _calculate_learning_value(self, reward_signal: RewardSignal, 
                                 outcome_quality: float) -> float:
        """Calculate the learning value of a reward experience"""
        try:
            # Base learning value from reward magnitude
            reward_magnitude = abs(reward_signal.total_reward)
            
            # Surprise factor enhances learning
            surprise_factor = reward_signal.surprise_bonus
            
            # Prediction error enhances learning
            prediction_error = abs(reward_signal.total_reward - outcome_quality)
            
            # Novelty enhances learning
            novelty_factor = reward_signal.novelty_bonus
            
            # Combine factors
            learning_value = (
                reward_magnitude * 0.3 +
                surprise_factor * 0.25 +
                prediction_error * 0.25 +
                novelty_factor * 0.2
            )
            
            # Normalize to [0, 1]
            learning_value = np.tanh(learning_value)
            
            return learning_value
            
        except Exception as e:
            logger.error(f"Error calculating learning value: {e}")
            return 0.5
    
    def _retrieve_similar_memories(self, context: RewardContext, 
                                  max_memories: int = 50) -> List[RewardMemory]:
        """Retrieve memories similar to current context"""
        try:
            all_memories = []
            
            # Collect from all memory stores
            all_memories.extend(self.short_term_memory)
            all_memories.extend(self.medium_term_memory)
            all_memories.extend(self.long_term_memory)
            
            if not all_memories:
                return []
            
            # Calculate similarity scores
            memory_scores = []
            for memory in all_memories:
                similarity = self._calculate_memory_similarity(context, memory.context)
                
                # Apply temporal decay
                time_diff = context.timestamp - memory.context.timestamp
                decay_factor = np.exp(-self.base_decay_rate * time_diff)
                
                # Combine similarity and decay
                score = similarity * decay_factor * memory.learning_value
                memory_scores.append((memory, score))
            
            # Sort by score and return top memories
            memory_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [memory for memory, score in memory_scores[:max_memories]]
            
        except Exception as e:
            logger.error(f"Error retrieving similar memories: {e}")
            return []
    
    def _calculate_memory_similarity(self, context1: RewardContext, 
                                    context2: RewardContext) -> float:
        """Calculate similarity between two reward contexts"""
        try:
            similarities = []
            
            # Market regime similarity
            if context1.market_regime == context2.market_regime:
                similarities.append(1.0)
            else:
                similarities.append(0.0)
            
            # Price similarity (normalized)
            price_diff = abs(context1.price - context2.price) / max(context1.price, context2.price)
            price_sim = np.exp(-price_diff * 5)
            similarities.append(price_sim)
            
            # Volatility similarity
            vol_diff = abs(context1.volatility - context2.volatility)
            vol_sim = np.exp(-vol_diff * 10)
            similarities.append(vol_sim)
            
            # Position size similarity
            size_diff = abs(context1.position_size - context2.position_size)
            size_sim = np.exp(-size_diff * 2)
            similarities.append(size_sim)
            
            # Action similarity
            if context1.trade_action == context2.trade_action:
                similarities.append(1.0)
            else:
                similarities.append(0.0)
            
            # Volume similarity
            if context1.volume > 0 and context2.volume > 0:
                vol_ratio = min(context1.volume, context2.volume) / max(context1.volume, context2.volume)
                similarities.append(vol_ratio)
            else:
                similarities.append(0.5)
            
            # Average similarity
            return np.mean(similarities)
            
        except Exception as e:
            logger.error(f"Error calculating memory similarity: {e}")
            return 0.0
    
    def _analyze_recent_performance(self, memories: List[RewardMemory], 
                                   hours: int = 6) -> float:
        """Analyze recent performance from memories"""
        try:
            if not memories:
                return 0.0
            
            current_time = time.time()
            cutoff_time = current_time - (hours * 3600)
            
            # Filter recent memories
            recent_memories = [
                m for m in memories 
                if m.context.timestamp >= cutoff_time
            ]
            
            if not recent_memories:
                return 0.0
            
            # Calculate performance metrics
            rewards = [m.reward_signal.total_reward for m in recent_memories]
            outcomes = [m.outcome_quality for m in recent_memories]
            
            # Performance indicators
            avg_reward = np.mean(rewards)
            avg_outcome = np.mean(outcomes)
            reward_trend = np.corrcoef(range(len(rewards)), rewards)[0, 1] if len(rewards) > 1 else 0.0
            
            # Combine into performance score
            performance = (avg_reward + avg_outcome + reward_trend) / 3.0
            
            return performance
            
        except Exception as e:
            logger.error(f"Error analyzing recent performance: {e}")
            return 0.0
    
    def _calculate_temporal_trend(self, memories: List[RewardMemory]) -> float:
        """Calculate temporal trend from memories"""
        try:
            if len(memories) < 3:
                return 0.0
            
            # Sort by timestamp
            memories.sort(key=lambda m: m.context.timestamp)
            
            # Extract time series data
            timestamps = [m.context.timestamp for m in memories]
            rewards = [m.reward_signal.total_reward for m in memories]
            outcomes = [m.outcome_quality for m in memories]
            
            # Calculate trends
            reward_trend = np.polyfit(timestamps, rewards, 1)[0] if len(rewards) > 1 else 0.0
            outcome_trend = np.polyfit(timestamps, outcomes, 1)[0] if len(outcomes) > 1 else 0.0
            
            # Combine trends
            trend = (reward_trend + outcome_trend) / 2.0
            
            # Normalize
            return np.tanh(trend * 1000)  # Scale for numerical stability
            
        except Exception as e:
            logger.error(f"Error calculating temporal trend: {e}")
            return 0.0
    
    def _calculate_context_similarity(self, current_context: RewardContext,
                                     memories: List[RewardMemory]) -> float:
        """Calculate average context similarity"""
        try:
            if not memories:
                return 0.0
            
            similarities = [
                self._calculate_memory_similarity(current_context, m.context)
                for m in memories
            ]
            
            return np.mean(similarities)
            
        except Exception as e:
            logger.error(f"Error calculating context similarity: {e}")
            return 0.0
    
    def _consolidate_memories(self):
        """Consolidate memories and extract patterns"""
        try:
            logger.info("Consolidating reward memories...")
            
            # Decay old memories
            self._apply_memory_decay()
            
            # Extract patterns
            self._extract_reward_patterns()
            
            # Update temporal correlations
            self._update_temporal_correlations()
            
            self.last_consolidation = time.time()
            
            logger.info(f"Memory consolidation complete. "
                       f"Short-term: {len(self.short_term_memory)}, "
                       f"Medium-term: {len(self.medium_term_memory)}, "
                       f"Long-term: {len(self.long_term_memory)}")
            
        except Exception as e:
            logger.error(f"Error during memory consolidation: {e}")
    
    def _apply_memory_decay(self):
        """Apply decay to memories"""
        try:
            current_time = time.time()
            
            # Decay short-term memories
            for memory in list(self.short_term_memory):
                time_diff = current_time - memory.context.timestamp
                memory.decay_factor = np.exp(-self.base_decay_rate * time_diff)
                
                # Remove very old short-term memories
                if time_diff > self.short_term_window * 60:
                    self.short_term_memory.remove(memory)
            
            # Similar for medium-term and long-term
            for memory in list(self.medium_term_memory):
                time_diff = current_time - memory.context.timestamp
                memory.decay_factor = np.exp(-self.base_decay_rate * time_diff * 0.5)
                
                if time_diff > self.medium_term_window * 3600:
                    self.medium_term_memory.remove(memory)
            
            for memory in list(self.long_term_memory):
                time_diff = current_time - memory.context.timestamp
                memory.decay_factor = np.exp(-self.base_decay_rate * time_diff * 0.1)
                
                if time_diff > self.long_term_window * 24 * 3600:
                    self.long_term_memory.remove(memory)
            
        except Exception as e:
            logger.error(f"Error applying memory decay: {e}")
    
    def _extract_reward_patterns(self):
        """Extract patterns from consolidated memories"""
        try:
            all_memories = []
            all_memories.extend(self.short_term_memory)
            all_memories.extend(self.medium_term_memory)
            all_memories.extend(self.long_term_memory)
            
            if not all_memories:
                return
            
            # Group by market regime
            regime_patterns = {}
            for memory in all_memories:
                regime = memory.context.market_regime
                if regime not in regime_patterns:
                    regime_patterns[regime] = []
                regime_patterns[regime].append(memory)
            
            # Extract patterns for each regime
            for regime, memories in regime_patterns.items():
                if len(memories) < 5:
                    continue
                
                # Calculate average reward components
                avg_base_reward = np.mean([m.reward_signal.base_reward for m in memories])
                avg_surprise = np.mean([m.reward_signal.surprise_bonus for m in memories])
                avg_novelty = np.mean([m.reward_signal.novelty_bonus for m in memories])
                
                self.reward_patterns[regime] = {
                    'avg_base_reward': avg_base_reward,
                    'avg_surprise': avg_surprise,
                    'avg_novelty': avg_novelty,
                    'sample_count': len(memories)
                }
            
        except Exception as e:
            logger.error(f"Error extracting reward patterns: {e}")
    
    def _update_temporal_correlations(self):
        """Update temporal correlations between contexts and outcomes"""
        try:
            all_memories = []
            all_memories.extend(self.medium_term_memory)
            all_memories.extend(self.long_term_memory)
            
            if len(all_memories) < 10:
                return
            
            # Calculate correlations between context features and outcomes
            volatilities = [m.context.volatility for m in all_memories]
            outcomes = [m.outcome_quality for m in all_memories]
            
            if len(volatilities) > 1:
                vol_corr = np.corrcoef(volatilities, outcomes)[0, 1]
                self.temporal_correlations['volatility_outcome'] = vol_corr
            
            # Similar for other features
            prices = [m.context.price for m in all_memories]
            if len(prices) > 1:
                price_corr = np.corrcoef(prices, outcomes)[0, 1]
                self.temporal_correlations['price_outcome'] = price_corr
            
        except Exception as e:
            logger.error(f"Error updating temporal correlations: {e}")
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get statistics about memory usage"""
        return {
            'total_memories_stored': self.total_memories_stored,
            'short_term_count': len(self.short_term_memory),
            'medium_term_count': len(self.medium_term_memory),
            'long_term_count': len(self.long_term_memory),
            'reward_patterns': len(self.reward_patterns),
            'temporal_correlations': len(self.temporal_correlations),
            'last_consolidation': self.last_consolidation,
            'memory_utilization': (
                len(self.short_term_memory) + 
                len(self.medium_term_memory) + 
                len(self.long_term_memory)
            ) / self.max_memory_size
        }
    
    def clear_memory(self, memory_type: str = 'all'):
        """Clear memories of specified type"""
        if memory_type in ['all', 'short_term']:
            self.short_term_memory.clear()
        if memory_type in ['all', 'medium_term']:
            self.medium_term_memory.clear()
        if memory_type in ['all', 'long_term']:
            self.long_term_memory.clear()
        
        if memory_type == 'all':
            self.reward_patterns.clear()
            self.temporal_correlations.clear()
        
        logger.info(f"Cleared {memory_type} memories")