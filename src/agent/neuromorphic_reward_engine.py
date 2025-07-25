# neuromorphic_reward_engine.py

import logging
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time

from .temporal_reward_memory import TemporalRewardMemory, RewardContext, RewardSignal
from .surprise_detector import SurpriseDetector
from .dopamine_pathway import DopaminePathway
from .multi_objective_optimizer import MultiObjectiveOptimizer

logger = logging.getLogger(__name__)

@dataclass
class RewardMetrics:
    """Comprehensive reward metrics"""
    total_reward: float
    base_reward: float
    surprise_bonus: float
    novelty_bonus: float
    temporal_adjustment: float
    efficiency_bonus: float
    risk_adjustment: float
    learning_signal: float
    motivation_level: float
    confidence: float
    processing_time: float

class NeuromorphicRewardEngine:
    """
    Sophisticated neuromorphic reward engine that integrates all reward components.
    
    Responsibilities:
    - Coordinate all reward processing components
    - Compute multi-factor rewards with temporal dependencies
    - Generate comprehensive reward signals
    - Track reward system performance
    - Adapt reward processing based on outcomes
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.temporal_memory = TemporalRewardMemory(config)
        self.surprise_detector = SurpriseDetector(config)
        self.dopamine_pathway = DopaminePathway(config)
        self.multi_objective_optimizer = MultiObjectiveOptimizer(config)
        
        # Reward processing parameters
        self.reward_scaling = config.get('reward_scaling', 1.0)
        self.risk_aversion = config.get('risk_aversion', 0.1)
        self.exploration_bonus = config.get('exploration_bonus', 0.05)
        self.efficiency_weight = config.get('efficiency_weight', 0.1)
        
        # Performance tracking
        self.reward_history = []
        self.performance_metrics = {
            'total_rewards_processed': 0,
            'avg_processing_time': 0.0,
            'avg_reward_magnitude': 0.0,
            'surprise_detection_rate': 0.0,
            'temporal_accuracy': 0.0,
            'system_efficiency': 0.0
        }
        
        # Adaptation parameters
        self.adaptation_enabled = config.get('adaptation_enabled', True)
        self.adaptation_rate = config.get('adaptation_rate', 0.01)
        self.performance_window = config.get('performance_window', 100)
        
        logger.info("Neuromorphic reward engine initialized")
    
    def compute_reward(self, context: RewardContext) -> RewardSignal:
        """
        Compute comprehensive reward signal
        
        Args:
            context: Current reward context
            
        Returns:
            Complete reward signal with all components
        """
        start_time = time.time()
        
        try:
            # Calculate base reward
            base_reward = self._calculate_base_reward(context)
            
            # Detect surprise and novelty
            surprise_bonus, novelty_bonus = self.surprise_detector.detect_surprise(
                context, base_reward, 'pnl'
            )
            
            # Get temporal adjustment
            temporal_adjustment = self.temporal_memory.get_temporal_adjustment(context)
            
            # Process through dopamine pathway
            reward_signal = self.dopamine_pathway.process_reward_signal(
                base_reward, surprise_bonus, temporal_adjustment, context
            )
            
            # Apply multi-objective optimization
            optimized_reward = self.multi_objective_optimizer.optimize_reward(
                reward_signal, context
            )
            
            # Calculate additional bonuses
            efficiency_bonus = self._calculate_efficiency_bonus(context)
            risk_adjustment = self._calculate_risk_adjustment(context)
            
            # Update reward signal with optimized components
            final_reward_signal = RewardSignal(
                base_reward=optimized_reward['base_reward'],
                temporal_adjustment=optimized_reward['temporal_adjustment'],
                surprise_bonus=optimized_reward['surprise_bonus'],
                novelty_bonus=novelty_bonus,
                efficiency_bonus=efficiency_bonus,
                risk_adjustment=risk_adjustment,
                total_reward=optimized_reward['total_reward'] + efficiency_bonus + risk_adjustment,
                confidence=optimized_reward['confidence'],
                explanation=self._generate_comprehensive_explanation(
                    optimized_reward, efficiency_bonus, risk_adjustment, context
                )
            )
            
            # Record processing metrics
            processing_time = time.time() - start_time
            self._record_reward_processing(final_reward_signal, context, processing_time)
            
            # Store in temporal memory
            self.temporal_memory.store_reward_experience(
                context, final_reward_signal, self._estimate_outcome_quality(context)
            )
            
            # Adapt system if enabled
            if self.adaptation_enabled:
                self._adapt_reward_system(final_reward_signal, context)
            
            return final_reward_signal
            
        except Exception as e:
            logger.error(f"Error computing reward: {e}")
            # Return basic reward signal on error
            return RewardSignal(
                base_reward=self._calculate_base_reward(context),
                temporal_adjustment=0.0,
                surprise_bonus=0.0,
                novelty_bonus=0.0,
                efficiency_bonus=0.0,
                risk_adjustment=0.0,
                total_reward=self._calculate_base_reward(context),
                confidence=0.5,
                explanation="Error in reward processing"
            )
    
    def _calculate_base_reward(self, context: RewardContext) -> float:
        """Calculate base reward from context"""
        try:
            # Primary reward from P&L
            pnl_reward = context.realized_pnl + context.unrealized_pnl
            
            # Normalize by account size
            if context.account_balance > 0:
                normalized_reward = pnl_reward / (context.account_balance * 0.01)  # 1% of account
            else:
                normalized_reward = pnl_reward / 1000.0  # Default normalization
            
            # Apply scaling
            scaled_reward = normalized_reward * self.reward_scaling
            
            # Apply sigmoid to prevent extreme values
            final_reward = np.tanh(scaled_reward)
            
            return final_reward
            
        except Exception as e:
            logger.error(f"Error calculating base reward: {e}")
            return 0.0
    
    def _calculate_efficiency_bonus(self, context: RewardContext) -> float:
        """Calculate efficiency bonus"""
        try:
            # Reward for quick decisions with high confidence
            confidence_bonus = (context.confidence - 0.5) * 0.1
            
            # Reward for appropriate position sizing
            size_efficiency = 0.0
            if 0.1 <= abs(context.position_size) <= 5.0:  # Reasonable size range
                size_efficiency = 0.05
            
            # Reward for trading in good market conditions
            market_efficiency = 0.0
            if context.volatility > 0.01 and context.volume > 1000:  # Good liquidity
                market_efficiency = 0.02
            
            total_efficiency = confidence_bonus + size_efficiency + market_efficiency
            
            return total_efficiency * self.efficiency_weight
            
        except Exception as e:
            logger.error(f"Error calculating efficiency bonus: {e}")
            return 0.0
    
    def _calculate_risk_adjustment(self, context: RewardContext) -> float:
        """Calculate risk adjustment"""
        try:
            # Penalize excessive risk
            risk_penalty = 0.0
            
            # Position size risk
            if abs(context.position_size) > 10.0:  # Large position
                risk_penalty += 0.1
            
            # Volatility risk
            if context.volatility > 0.05:  # High volatility
                risk_penalty += 0.05
            
            # Account risk
            if context.account_balance > 0:
                risk_ratio = abs(context.unrealized_pnl) / context.account_balance
                if risk_ratio > 0.1:  # More than 10% account at risk
                    risk_penalty += risk_ratio * 0.2
            
            # Low confidence risk
            if context.confidence < 0.3:
                risk_penalty += (0.3 - context.confidence) * 0.1
            
            return -risk_penalty * self.risk_aversion
            
        except Exception as e:
            logger.error(f"Error calculating risk adjustment: {e}")
            return 0.0
    
    def _estimate_outcome_quality(self, context: RewardContext) -> float:
        """Estimate quality of outcome for learning"""
        try:
            # Base quality from P&L
            pnl = context.realized_pnl + context.unrealized_pnl
            
            # Normalize
            if context.account_balance > 0:
                normalized_pnl = pnl / (context.account_balance * 0.01)
            else:
                normalized_pnl = pnl / 1000.0
            
            # Apply sigmoid
            quality = (np.tanh(normalized_pnl) + 1.0) / 2.0  # Scale to [0, 1]
            
            # Adjust for confidence
            quality *= context.confidence
            
            return quality
            
        except Exception as e:
            logger.error(f"Error estimating outcome quality: {e}")
            return 0.5
    
    def _record_reward_processing(self, reward_signal: RewardSignal, 
                                 context: RewardContext, processing_time: float):
        """Record reward processing metrics"""
        try:
            # Create reward metrics
            metrics = RewardMetrics(
                total_reward=reward_signal.total_reward,
                base_reward=reward_signal.base_reward,
                surprise_bonus=reward_signal.surprise_bonus,
                novelty_bonus=reward_signal.novelty_bonus,
                temporal_adjustment=reward_signal.temporal_adjustment,
                efficiency_bonus=reward_signal.efficiency_bonus,
                risk_adjustment=reward_signal.risk_adjustment,
                learning_signal=0.0,  # Would come from dopamine pathway
                motivation_level=0.0,  # Would come from dopamine pathway
                confidence=reward_signal.confidence,
                processing_time=processing_time
            )
            
            self.reward_history.append(metrics)
            
            # Update performance metrics
            self._update_performance_metrics(metrics)
            
        except Exception as e:
            logger.error(f"Error recording reward processing: {e}")
    
    def _update_performance_metrics(self, metrics: RewardMetrics):
        """Update system performance metrics"""
        try:
            self.performance_metrics['total_rewards_processed'] += 1
            
            # Update averages
            count = self.performance_metrics['total_rewards_processed']
            
            # Moving average of processing time
            self.performance_metrics['avg_processing_time'] = (
                (self.performance_metrics['avg_processing_time'] * (count - 1) + 
                 metrics.processing_time) / count
            )
            
            # Moving average of reward magnitude
            self.performance_metrics['avg_reward_magnitude'] = (
                (self.performance_metrics['avg_reward_magnitude'] * (count - 1) + 
                 abs(metrics.total_reward)) / count
            )
            
            # Calculate surprise detection rate
            if len(self.reward_history) >= self.performance_window:
                recent_rewards = self.reward_history[-self.performance_window:]
                surprise_count = sum(1 for r in recent_rewards if r.surprise_bonus > 0.01)
                self.performance_metrics['surprise_detection_rate'] = surprise_count / len(recent_rewards)
            
            # Calculate system efficiency
            self.performance_metrics['system_efficiency'] = self._calculate_system_efficiency()
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def _calculate_system_efficiency(self) -> float:
        """Calculate overall system efficiency"""
        try:
            if len(self.reward_history) < 10:
                return 0.5
            
            recent_rewards = self.reward_history[-100:]
            
            # Efficiency metrics
            avg_confidence = np.mean([r.confidence for r in recent_rewards])
            avg_processing_time = np.mean([r.processing_time for r in recent_rewards])
            reward_variance = np.var([r.total_reward for r in recent_rewards])
            
            # Normalize and combine
            confidence_score = avg_confidence
            speed_score = max(0.0, 1.0 - avg_processing_time)  # Faster is better
            stability_score = max(0.0, 1.0 - reward_variance)  # Lower variance is better
            
            efficiency = (confidence_score + speed_score + stability_score) / 3.0
            
            return efficiency
            
        except Exception as e:
            logger.error(f"Error calculating system efficiency: {e}")
            return 0.5
    
    def _adapt_reward_system(self, reward_signal: RewardSignal, context: RewardContext):
        """Adapt reward system based on performance"""
        try:
            if self.performance_metrics['total_rewards_processed'] < 50:
                return  # Need minimum data for adaptation
            
            # Adapt based on system efficiency
            efficiency = self.performance_metrics['system_efficiency']
            
            if efficiency < 0.6:  # Low efficiency
                # Increase exploration
                self.exploration_bonus = min(0.1, self.exploration_bonus + 0.001)
                
                # Adjust surprise sensitivity
                self.surprise_detector.surprise_threshold *= 0.99
                
            elif efficiency > 0.8:  # High efficiency
                # Reduce exploration
                self.exploration_bonus = max(0.01, self.exploration_bonus - 0.001)
                
                # Increase surprise threshold
                self.surprise_detector.surprise_threshold *= 1.01
            
            # Adapt risk aversion based on recent performance
            if len(self.reward_history) >= 20:
                recent_rewards = [r.total_reward for r in self.reward_history[-20:]]
                if np.mean(recent_rewards) < -0.1:  # Poor recent performance
                    self.risk_aversion = min(0.3, self.risk_aversion + 0.01)
                elif np.mean(recent_rewards) > 0.1:  # Good recent performance
                    self.risk_aversion = max(0.05, self.risk_aversion - 0.01)
            
        except Exception as e:
            logger.error(f"Error adapting reward system: {e}")
    
    def _generate_comprehensive_explanation(self, optimized_reward: Dict[str, Any],
                                          efficiency_bonus: float, risk_adjustment: float,
                                          context: RewardContext) -> str:
        """Generate comprehensive explanation of reward processing"""
        try:
            components = []
            
            # Base reward
            base = optimized_reward.get('base_reward', 0.0)
            if abs(base) > 0.01:
                components.append(f"base reward: {base:.3f}")
            
            # Surprise bonus
            surprise = optimized_reward.get('surprise_bonus', 0.0)
            if surprise > 0.01:
                components.append(f"surprise bonus: {surprise:.3f}")
            
            # Temporal adjustment
            temporal = optimized_reward.get('temporal_adjustment', 0.0)
            if abs(temporal) > 0.01:
                components.append(f"temporal adjustment: {temporal:.3f}")
            
            # Efficiency bonus
            if abs(efficiency_bonus) > 0.01:
                components.append(f"efficiency bonus: {efficiency_bonus:.3f}")
            
            # Risk adjustment
            if abs(risk_adjustment) > 0.01:
                components.append(f"risk adjustment: {risk_adjustment:.3f}")
            
            # Context information
            if context.confidence < 0.3:
                components.append("low confidence")
            elif context.confidence > 0.7:
                components.append("high confidence")
            
            if context.volatility > 0.05:
                components.append("high volatility")
            
            if abs(context.position_size) > 5.0:
                components.append("large position")
            
            if not components:
                return "Neutral reward processing"
            
            return "Neuromorphic reward: " + ", ".join(components)
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return "Reward processed through neuromorphic engine"
    
    def get_reward_statistics(self) -> Dict[str, Any]:
        """Get comprehensive reward system statistics"""
        try:
            stats = {
                'performance_metrics': self.performance_metrics.copy(),
                'system_parameters': {
                    'reward_scaling': self.reward_scaling,
                    'risk_aversion': self.risk_aversion,
                    'exploration_bonus': self.exploration_bonus,
                    'efficiency_weight': self.efficiency_weight
                },
                'component_statistics': {
                    'temporal_memory': self.temporal_memory.get_memory_statistics(),
                    'surprise_detector': self.surprise_detector.get_surprise_statistics(),
                    'dopamine_pathway': self.dopamine_pathway.get_pathway_statistics(),
                    'multi_objective_optimizer': self.multi_objective_optimizer.get_optimizer_statistics()
                }
            }
            
            # Add recent reward analysis
            if len(self.reward_history) >= 10:
                recent_rewards = self.reward_history[-100:]
                stats['recent_analysis'] = {
                    'avg_total_reward': np.mean([r.total_reward for r in recent_rewards]),
                    'reward_volatility': np.std([r.total_reward for r in recent_rewards]),
                    'avg_confidence': np.mean([r.confidence for r in recent_rewards]),
                    'surprise_rate': np.mean([1 if r.surprise_bonus > 0.01 else 0 for r in recent_rewards]),
                    'efficiency_rate': np.mean([1 if r.efficiency_bonus > 0.01 else 0 for r in recent_rewards])
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting reward statistics: {e}")
            return {}
    
    def reset_reward_system(self):
        """Reset the entire reward system"""
        try:
            # Reset all components
            self.temporal_memory.clear_memory()
            self.surprise_detector.reset_models()
            self.dopamine_pathway.reset_pathway()
            self.multi_objective_optimizer.reset_optimizer()
            
            # Reset performance metrics
            self.performance_metrics = {
                'total_rewards_processed': 0,
                'avg_processing_time': 0.0,
                'avg_reward_magnitude': 0.0,
                'surprise_detection_rate': 0.0,
                'temporal_accuracy': 0.0,
                'system_efficiency': 0.0
            }
            
            # Clear history
            self.reward_history.clear()
            
            # Reset parameters to defaults
            self.reward_scaling = self.config.get('reward_scaling', 1.0)
            self.risk_aversion = self.config.get('risk_aversion', 0.1)
            self.exploration_bonus = self.config.get('exploration_bonus', 0.05)
            self.efficiency_weight = self.config.get('efficiency_weight', 0.1)
            
            logger.info("Neuromorphic reward system reset")
            
        except Exception as e:
            logger.error(f"Error resetting reward system: {e}")
    
    def get_reward_insights(self) -> Dict[str, Any]:
        """Get insights about reward system behavior"""
        try:
            if len(self.reward_history) < 20:
                return {'message': 'Insufficient data for insights'}
            
            recent_rewards = self.reward_history[-100:]
            
            insights = {
                'dominant_reward_component': self._identify_dominant_component(recent_rewards),
                'reward_trends': self._analyze_reward_trends(recent_rewards),
                'efficiency_patterns': self._analyze_efficiency_patterns(recent_rewards),
                'risk_patterns': self._analyze_risk_patterns(recent_rewards),
                'surprise_patterns': self._analyze_surprise_patterns(recent_rewards),
                'recommendations': self._generate_recommendations(recent_rewards)
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting reward insights: {e}")
            return {}
    
    def _identify_dominant_component(self, rewards: List[RewardMetrics]) -> str:
        """Identify which reward component is most influential"""
        try:
            component_impacts = {
                'base_reward': np.mean([abs(r.base_reward) for r in rewards]),
                'surprise_bonus': np.mean([abs(r.surprise_bonus) for r in rewards]),
                'temporal_adjustment': np.mean([abs(r.temporal_adjustment) for r in rewards]),
                'efficiency_bonus': np.mean([abs(r.efficiency_bonus) for r in rewards]),
                'risk_adjustment': np.mean([abs(r.risk_adjustment) for r in rewards])
            }
            
            return max(component_impacts, key=component_impacts.get)
            
        except Exception as e:
            logger.error(f"Error identifying dominant component: {e}")
            return 'base_reward'
    
    def _analyze_reward_trends(self, rewards: List[RewardMetrics]) -> Dict[str, Any]:
        """Analyze reward trends"""
        try:
            total_rewards = [r.total_reward for r in rewards]
            
            if len(total_rewards) < 2:
                return {'trend': 'insufficient_data'}
            
            # Calculate trend
            x = np.arange(len(total_rewards))
            slope, _ = np.polyfit(x, total_rewards, 1)
            
            trend_direction = 'increasing' if slope > 0.001 else 'decreasing' if slope < -0.001 else 'stable'
            
            return {
                'trend': trend_direction,
                'slope': slope,
                'recent_avg': np.mean(total_rewards[-10:]),
                'overall_avg': np.mean(total_rewards),
                'volatility': np.std(total_rewards)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing reward trends: {e}")
            return {'trend': 'error'}
    
    def _analyze_efficiency_patterns(self, rewards: List[RewardMetrics]) -> Dict[str, Any]:
        """Analyze efficiency patterns"""
        try:
            efficiency_bonuses = [r.efficiency_bonus for r in rewards]
            
            return {
                'avg_efficiency_bonus': np.mean(efficiency_bonuses),
                'efficiency_rate': np.mean([1 if e > 0.01 else 0 for e in efficiency_bonuses]),
                'max_efficiency': np.max(efficiency_bonuses),
                'efficiency_trend': 'improving' if np.mean(efficiency_bonuses[-10:]) > np.mean(efficiency_bonuses) else 'declining'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing efficiency patterns: {e}")
            return {}
    
    def _analyze_risk_patterns(self, rewards: List[RewardMetrics]) -> Dict[str, Any]:
        """Analyze risk patterns"""
        try:
            risk_adjustments = [r.risk_adjustment for r in rewards]
            
            return {
                'avg_risk_adjustment': np.mean(risk_adjustments),
                'risk_penalty_rate': np.mean([1 if r < -0.01 else 0 for r in risk_adjustments]),
                'max_risk_penalty': np.min(risk_adjustments),
                'risk_trend': 'improving' if np.mean(risk_adjustments[-10:]) > np.mean(risk_adjustments) else 'worsening'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing risk patterns: {e}")
            return {}
    
    def _analyze_surprise_patterns(self, rewards: List[RewardMetrics]) -> Dict[str, Any]:
        """Analyze surprise patterns"""
        try:
            surprise_bonuses = [r.surprise_bonus for r in rewards]
            
            return {
                'avg_surprise_bonus': np.mean(surprise_bonuses),
                'surprise_rate': np.mean([1 if s > 0.01 else 0 for s in surprise_bonuses]),
                'max_surprise': np.max(surprise_bonuses),
                'surprise_consistency': 1.0 - np.std(surprise_bonuses)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing surprise patterns: {e}")
            return {}
    
    def _generate_recommendations(self, rewards: List[RewardMetrics]) -> List[str]:
        """Generate recommendations based on reward analysis"""
        try:
            recommendations = []
            
            # Efficiency recommendations
            efficiency_rate = np.mean([1 if r.efficiency_bonus > 0.01 else 0 for r in rewards])
            if efficiency_rate < 0.3:
                recommendations.append("Consider improving decision confidence and position sizing")
            
            # Risk recommendations
            risk_penalty_rate = np.mean([1 if r.risk_adjustment < -0.01 else 0 for r in rewards])
            if risk_penalty_rate > 0.5:
                recommendations.append("Reduce risk exposure - frequent risk penalties detected")
            
            # Surprise recommendations
            surprise_rate = np.mean([1 if r.surprise_bonus > 0.01 else 0 for r in rewards])
            if surprise_rate > 0.7:
                recommendations.append("High surprise rate - consider improving prediction models")
            elif surprise_rate < 0.1:
                recommendations.append("Low surprise rate - system may be too conservative")
            
            # Temporal recommendations
            temporal_impact = np.mean([abs(r.temporal_adjustment) for r in rewards])
            if temporal_impact > 0.1:
                recommendations.append("Strong temporal patterns detected - leverage memory more effectively")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []