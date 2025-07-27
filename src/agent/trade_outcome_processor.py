"""
Trade Outcome Processor - Centralized learning from trade outcomes

This module handles all trade outcome processing and learning:
1. Trade outcome analysis and validation
2. Meta-learner updates and parameter adaptation
3. Dopamine system integration for psychological modeling
4. Performance tracking and statistics
5. Multi-component learning coordination
6. Error handling and recovery mechanisms

Extracted from TradingAgent.learn_from_trade() to improve maintainability and testability.
"""

import torch
import numpy as np
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from src.agent.meta_learner import MetaLearner
from src.agent.real_time_adaptation import RealTimeAdaptationEngine
from src.agent.experience_manager import ExperienceManager
from src.agent.neural_network_manager import NeuralNetworkManager

logger = logging.getLogger(__name__)


@dataclass
class TradeOutcome:
    """Standardized trade outcome data structure"""
    pnl: float
    entry_price: float
    exit_price: float
    entry_time: float
    exit_time: float
    size: float
    action: str
    exit_reason: str
    account_balance: float
    confidence: float
    primary_tool: str
    exploration: bool
    intelligence_data: Optional[Dict[str, Any]] = None
    decision_data: Optional[Dict[str, Any]] = None
    state_features: Optional[List[float]] = None
    adaptation_strategy: str = 'conservative'
    uncertainty_estimate: float = 0.5
    
    def __post_init__(self):
        """Validate trade outcome data"""
        if self.pnl is None:
            raise ValueError("PnL cannot be None")
        if self.account_balance <= 0:
            raise ValueError("Account balance must be positive")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")


class LearningCoordinator:
    """Coordinates learning across multiple components to prevent conflicts"""
    
    def __init__(self):
        self.learning_lock = None
        self.learning_stats = {
            'total_learning_cycles': 0,
            'successful_cycles': 0,
            'failed_cycles': 0,
            'last_learning_time': 0.0,
            'average_cycle_time': 0.0
        }
    
    def ensure_thread_safety(self):
        """Ensure thread-safe learning coordination"""
        if self.learning_lock is None:
            import threading
            self.learning_lock = threading.Lock()
    
    def coordinate_learning(self, learning_func):
        """Execute learning function with thread safety"""
        self.ensure_thread_safety()
        
        start_time = time.time()
        
        with self.learning_lock:
            try:
                self.learning_stats['total_learning_cycles'] += 1
                result = learning_func()
                self.learning_stats['successful_cycles'] += 1
                
                # Update timing statistics
                cycle_time = time.time() - start_time
                total_cycles = self.learning_stats['total_learning_cycles']
                self.learning_stats['average_cycle_time'] = (
                    (self.learning_stats['average_cycle_time'] * (total_cycles - 1) + cycle_time) / total_cycles
                )
                self.learning_stats['last_learning_time'] = time.time()
                
                return result
                
            except Exception as e:
                self.learning_stats['failed_cycles'] += 1
                logger.error(f"Learning coordination failed: {e}")
                raise


class RewardCalculator:
    """Calculates comprehensive rewards for trade outcomes"""
    
    def __init__(self):
        self.reward_components = {
            'pnl_weight': 1.0,
            'confidence_bonus': 0.1,
            'exploration_bonus': 0.05,
            'consistency_bonus': 0.1,
            'strategy_bonus': 0.05
        }
    
    def calculate_comprehensive_reward(self, trade_outcome: TradeOutcome) -> float:
        """
        Calculate comprehensive reward incorporating multiple factors
        
        Args:
            trade_outcome: Standardized trade outcome
            
        Returns:
            Calculated reward value
        """
        try:
            # Base P&L reward (normalized by account size)
            account_normalized_pnl = trade_outcome.pnl / (trade_outcome.account_balance * 0.01)
            base_reward = np.tanh(account_normalized_pnl)
            
            # Confidence bonus (reward high-confidence correct decisions)
            confidence_bonus = 0.0
            if (trade_outcome.pnl > 0 and trade_outcome.confidence > 0.7) or \
               (trade_outcome.pnl <= 0 and trade_outcome.confidence < 0.3):
                confidence_bonus = trade_outcome.confidence * self.reward_components['confidence_bonus']
            
            # Exploration bonus (reward successful exploration)
            exploration_bonus = 0.0
            if trade_outcome.exploration and trade_outcome.pnl > 0:
                exploration_bonus = self.reward_components['exploration_bonus']
            
            # Strategy consistency bonus
            strategy_bonus = self._calculate_strategy_bonus(trade_outcome)
            
            # Hold time penalty for overly long trades
            hold_time_hours = (trade_outcome.exit_time - trade_outcome.entry_time) / 3600
            hold_penalty = max(0, (hold_time_hours - 24) / 24) * 0.1  # Penalty for > 24h holds
            
            # Total reward
            total_reward = (
                base_reward * self.reward_components['pnl_weight'] +
                confidence_bonus +
                exploration_bonus +
                strategy_bonus -
                hold_penalty
            )
            
            logger.debug(f"Reward breakdown: base={base_reward:.3f}, "
                        f"confidence={confidence_bonus:.3f}, exploration={exploration_bonus:.3f}, "
                        f"strategy={strategy_bonus:.3f}, hold_penalty={hold_penalty:.3f}, "
                        f"total={total_reward:.3f}")
            
            return total_reward
            
        except Exception as e:
            logger.error(f"Error calculating reward: {e}")
            return 0.0  # Safe default
    
    def _calculate_strategy_bonus(self, trade_outcome: TradeOutcome) -> float:
        """Calculate bonus based on strategy effectiveness"""
        try:
            # Simple strategy bonus based on profitability
            if trade_outcome.pnl > 0:
                strategy_multiplier = {
                    'conservative': 1.0,
                    'aggressive': 1.2,
                    'momentum': 1.1,
                    'mean_reversion': 1.1,
                    'adaptive': 1.3
                }.get(trade_outcome.adaptation_strategy, 1.0)
                
                return self.reward_components['strategy_bonus'] * strategy_multiplier
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating strategy bonus: {e}")
            return 0.0


class TradeOutcomeProcessor:
    """
    Processes trade outcomes and coordinates learning across all system components.
    
    This class centralizes the complex learning logic that was previously embedded
    in TradingAgent.learn_from_trade(), providing better error handling, modularity,
    and coordination between different learning components.
    """
    
    def __init__(self,
                 config: Dict[str, Any],
                 reward_engine: Optional[Any] = None,
                 meta_learner: Optional[MetaLearner] = None,
                 adaptation_engine: Optional[RealTimeAdaptationEngine] = None,
                 experience_manager: Optional[ExperienceManager] = None,
                 network_manager: Optional[NeuralNetworkManager] = None,
                 intelligence_engine: Optional[Any] = None):
        """
        Initialize the trade outcome processor
        
        Args:
            config: Configuration dictionary
            reward_engine: Reward calculation engine (optional)
            meta_learner: Meta-learning component (optional)
            adaptation_engine: Real-time adaptation engine (optional)
            experience_manager: Experience storage and sampling (optional)
            network_manager: Neural network management (optional)
            intelligence_engine: Intelligence subsystems (optional)
        """
        # Store injected components
        self.config = config
        self.reward_engine = reward_engine
        self.meta_learner = meta_learner
        self.adaptation_engine = adaptation_engine
        self.experience_manager = experience_manager
        self.network_manager = network_manager
        self.intelligence_engine = intelligence_engine
        
        # Initialize components
        self.learning_coordinator = LearningCoordinator()
        self.reward_calculator = RewardCalculator()
        
        # Processing statistics
        self.processing_stats = {
            'total_trades_processed': 0,
            'successful_processing': 0,
            'failed_processing': 0,
            'dopamine_integrations': 0,
            'meta_learner_updates': 0,
            'network_training_cycles': 0
        }
        
        logger.info("TradeOutcomeProcessor initialized")
    
    def process_trade_outcome(self, trade) -> bool:
        """
        Main entry point for processing trade outcomes
        
        Args:
            trade: Trade object with outcome data
            
        Returns:
            True if processing was successful
        """
        try:
            logger.info(f"=== PROCESSING TRADE OUTCOME START ===")
            logger.info(f"Trade PnL: {trade.pnl}, Entry: {trade.entry_price}, Exit: {trade.exit_price}")
            
            # Convert trade to standardized outcome
            trade_outcome = self._convert_to_trade_outcome(trade)
            
            # Coordinate learning sequence
            success = self.learning_coordinator.coordinate_learning(
                lambda: self._execute_learning_sequence(trade_outcome, trade)
            )
            
            if success:
                self.processing_stats['successful_processing'] += 1
                logger.info(f"=== TRADE OUTCOME PROCESSING COMPLETE ===")
            else:
                self.processing_stats['failed_processing'] += 1
                logger.error(f"=== TRADE OUTCOME PROCESSING FAILED ===")
            
            self.processing_stats['total_trades_processed'] += 1
            return success
            
        except Exception as e:
            logger.error(f"CRITICAL: Trade outcome processing failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            self.processing_stats['failed_processing'] += 1
            self.processing_stats['total_trades_processed'] += 1
            return False
    
    def _convert_to_trade_outcome(self, trade) -> TradeOutcome:
        """Convert trade object to standardized TradeOutcome"""
        try:
            return TradeOutcome(
                pnl=float(getattr(trade, 'pnl', 0.0)),
                entry_price=float(getattr(trade, 'entry_price', 0.0)),
                exit_price=float(getattr(trade, 'exit_price', 0.0)),
                entry_time=float(getattr(trade, 'entry_time', time.time())),
                exit_time=float(getattr(trade, 'exit_time', time.time())),
                size=float(getattr(trade, 'size', 1.0)),
                action=str(getattr(trade, 'action', 'unknown')),
                exit_reason=str(getattr(trade, 'exit_reason', 'completed')),
                account_balance=float(getattr(trade, 'exit_account_balance', 25000.0)),
                confidence=float(getattr(trade, 'confidence', 0.5)),
                primary_tool=str(getattr(trade, 'primary_tool', 'unknown')),
                exploration=bool(getattr(trade, 'exploration', False)),
                intelligence_data=getattr(trade, 'intelligence_data', None),
                decision_data=getattr(trade, 'decision_data', None),
                state_features=getattr(trade, 'state_features', None),
                adaptation_strategy=str(getattr(trade, 'adaptation_strategy', 'conservative')),
                uncertainty_estimate=float(getattr(trade, 'uncertainty_estimate', 0.5))
            )
            
        except Exception as e:
            logger.error(f"Error converting trade to outcome: {e}")
            # Return minimal valid outcome
            return TradeOutcome(
                pnl=0.0, entry_price=0.0, exit_price=0.0,
                entry_time=time.time(), exit_time=time.time(),
                size=1.0, action='unknown', exit_reason='error',
                account_balance=25000.0, confidence=0.5,
                primary_tool='unknown', exploration=False
            )
    
    def _execute_learning_sequence(self, trade_outcome: TradeOutcome, original_trade) -> bool:
        """Execute the complete learning sequence"""
        try:
            success_flags = []
            
            # Phase 1: Dopamine integration (psychological modeling)
            success_flags.append(self._integrate_dopamine_learning(trade_outcome, original_trade))
            
            # Phase 2: Calculate comprehensive reward
            reward = self.reward_calculator.calculate_comprehensive_reward(trade_outcome)
            logger.info(f"Calculated comprehensive reward: {reward:.6f}")
            
            # Phase 3: Meta-learner update
            success_flags.append(self._update_meta_learner(trade_outcome, reward))
            
            # Phase 4: Adaptation engine update
            success_flags.append(self._update_adaptation_engine(trade_outcome, reward))
            
            # Phase 5: Intelligence subsystems learning
            success_flags.append(self._update_intelligence_subsystems(trade_outcome, original_trade))
            
            # Phase 6: Network performance tracking
            success_flags.append(self._update_network_performance(reward))
            
            # Phase 7: Experience storage for future training
            success_flags.append(self._store_learning_experience(trade_outcome, reward, original_trade))
            
            # Phase 8: Trigger network training if sufficient data
            success_flags.append(self._conditional_network_training())
            
            # Return True if majority of phases succeeded
            success_rate = sum(success_flags) / len(success_flags)
            logger.info(f"Learning sequence success rate: {success_rate:.1%}")
            
            return success_rate >= 0.6  # 60% success threshold
            
        except Exception as e:
            logger.error(f"Error in learning sequence: {e}")
            return False
    
    def _integrate_dopamine_learning(self, trade_outcome: TradeOutcome, original_trade) -> bool:
        """Integrate dopamine system for psychological modeling"""
        try:
            # DOPAMINE PHASE: REALIZATION
            realization_market_data = {
                'unrealized_pnl': 0.0,  # Position closed
                'daily_pnl': trade_outcome.pnl,
                'realized_pnl': trade_outcome.pnl,
                'open_positions': 0.0,
                'current_price': trade_outcome.exit_price,
                'trade_duration': trade_outcome.exit_time - trade_outcome.entry_time
            }
            
            realization_context = {
                'expected_outcome': getattr(original_trade, 'expected_outcome', 0.0),
                'confidence': trade_outcome.confidence,
                'action': trade_outcome.action
            }
            
            dopamine_realization = self.intelligence_engine.dopamine_subsystem.process_trading_event(
                'realization', realization_market_data, realization_context
            )
            
            # DOPAMINE PHASE: REFLECTION
            reflection_context = {
                'learned_something': True,
                'trade_outcome': 'profit' if trade_outcome.pnl > 0 else 'loss',
                'strategy_effectiveness': trade_outcome.adaptation_strategy,
                'confidence_outcome': trade_outcome.confidence
            }
            
            dopamine_reflection = self.intelligence_engine.dopamine_subsystem.process_trading_event(
                'reflection', realization_market_data, reflection_context
            )
            
            # Log dopamine integration with safe attribute access
            logger.info(f"DOPAMINE LEARNING: Realization signal={self._safe_get_signal(dopamine_realization):.3f}, "
                       f"Reflection signal={self._safe_get_signal(dopamine_reflection):.3f}, "
                       f"State={self._safe_get_state(dopamine_reflection)}, "
                       f"Tolerance={self._safe_get_tolerance(dopamine_reflection):.3f}, "
                       f"Addiction={self._safe_get_addiction_risk(dopamine_reflection):.3f}")
            
            self.processing_stats['dopamine_integrations'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error in dopamine integration: {e}")
            return False
    
    def _update_meta_learner(self, trade_outcome: TradeOutcome, reward: float) -> bool:
        """Update meta-learner with trade outcome"""
        try:
            logger.info("Starting meta-learner update...")
            
            # Prepare trade data for meta-learner
            trade_data = {
                'pnl': trade_outcome.pnl,
                'account_balance': trade_outcome.account_balance,
                'hold_time': trade_outcome.exit_time - trade_outcome.entry_time,
                'was_exploration': trade_outcome.exploration,
                'subsystem_contributions': self._extract_subsystem_contributions(trade_outcome),
                'subsystem_agreement': self._calculate_subsystem_agreement(trade_outcome),
                'confidence': trade_outcome.confidence,
                'primary_tool': trade_outcome.primary_tool,
                'stop_used': False,  # Would need to extract from trade data
                'target_used': False,  # Would need to extract from trade data
                'adaptation_strategy': trade_outcome.adaptation_strategy,
                'uncertainty_estimate': trade_outcome.uncertainty_estimate,
                'regime_confidence': 0.5  # Would need to extract from intelligence data
            }
            
            # Track update counts
            old_total_updates = getattr(self.meta_learner, 'total_updates', 0)
            
            # Perform meta-learning update
            self.meta_learner.learn_from_outcome(trade_data)
            
            new_total_updates = getattr(self.meta_learner, 'total_updates', 0)
            
            if new_total_updates > old_total_updates:
                logger.info(f"Meta-learner updated successfully: {old_total_updates} -> {new_total_updates}")
                self.processing_stats['meta_learner_updates'] += 1
                return True
            else:
                logger.warning("Meta-learner update failed - no increment in total_updates")
                return False
                
        except Exception as e:
            logger.error(f"Meta-learner update failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _update_adaptation_engine(self, trade_outcome: TradeOutcome, reward: float) -> bool:
        """Update real-time adaptation engine"""
        try:
            logger.info("Starting adaptation engine update...")
            
            adaptation_context = {
                'predicted_confidence': trade_outcome.confidence,
                'actual_outcome': 'profit' if trade_outcome.pnl > 0 else 'loss',
                'strategy_used': trade_outcome.adaptation_strategy,
                'uncertainty_estimate': trade_outcome.uncertainty_estimate
            }
            
            # Extract regime context if available
            if trade_outcome.intelligence_data:
                regime_context = trade_outcome.intelligence_data.get('regime_context', {})
                adaptation_context.update(regime_context)
            
            self.adaptation_engine.update_from_outcome(reward, adaptation_context)
            logger.info("Adaptation engine updated successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Adaptation engine update failed: {e}")
            return False
    
    def _update_intelligence_subsystems(self, trade_outcome: TradeOutcome, original_trade) -> bool:
        """Update intelligence subsystems with trade outcome"""
        try:
            logger.info("Starting intelligence subsystems update...")
            
            # Extract numeric PnL value from trade object safely
            trade_pnl = self._safe_extract_trade_pnl(original_trade)
            
            # Create learning context from trade outcome data
            learning_context = {
                'dna_sequence': getattr(original_trade, 'dna_sequence', '') or trade_outcome.intelligence_data.get('dna_sequence', '') if trade_outcome.intelligence_data else '',
                'cycles_info': getattr(original_trade, 'cycles_info', []) or trade_outcome.intelligence_data.get('cycles_info', []) if trade_outcome.intelligence_data else [],
                'market_state': getattr(original_trade, 'market_state', {}) or trade_outcome.intelligence_data.get('market_state', {}) if trade_outcome.intelligence_data else {
                    'volatility': 0.02,
                    'price_momentum': 0.0,
                    'volume_momentum': 0.0,
                    'regime': 'ranging'
                },
                'microstructure_signal': getattr(original_trade, 'microstructure_signal', 0.0),
                'is_bootstrap': False
            }
            
            # Let intelligence engine learn from the numeric outcome with proper context
            self.intelligence_engine.learn_from_outcome(trade_pnl, learning_context)
            logger.info("Intelligence subsystems updated successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Intelligence subsystems update failed: {e}")
            return False
    
    def _safe_extract_trade_pnl(self, trade) -> float:
        """Safely extract numeric PnL value from trade object"""
        try:
            # Try different possible attributes for PnL
            if hasattr(trade, 'pnl'):
                return float(trade.pnl)
            elif hasattr(trade, 'profit_loss'):
                return float(trade.profit_loss)
            elif hasattr(trade, 'realized_pnl'):
                return float(trade.realized_pnl)
            elif hasattr(trade, 'net_pnl'):
                return float(trade.net_pnl)
            else:
                logger.warning(f"Could not extract PnL from trade object: {type(trade)}")
                return 0.0
        except (ValueError, TypeError, AttributeError) as e:
            logger.error(f"Error extracting trade PnL: {e}")
            return 0.0
    
    def _update_network_performance(self, reward: float) -> bool:
        """Update network performance tracking"""
        try:
            self.network_manager.record_performance(reward)
            logger.info(f"Network performance recorded: {reward:.6f}")
            return True
            
        except Exception as e:
            logger.error(f"Network performance recording failed: {e}")
            return False
    
    def _store_learning_experience(self, trade_outcome: TradeOutcome, reward: float, original_trade) -> bool:
        """Store experience for future network training"""
        try:
            if not trade_outcome.state_features:
                logger.warning("Trade missing state_features - skipping experience storage")
                return False
            
            experience = {
                'state_features': trade_outcome.state_features,
                'action': ['hold', 'buy', 'sell'].index(trade_outcome.action) if trade_outcome.action in ['hold', 'buy', 'sell'] else 0,
                'reward': reward,
                'done': True,
                'trade_data': {
                    'pnl': trade_outcome.pnl,
                    'confidence': trade_outcome.confidence,
                    'uncertainty': trade_outcome.uncertainty_estimate,
                    'strategy': trade_outcome.adaptation_strategy,
                    'primary_tool': trade_outcome.primary_tool,
                    'exploration': trade_outcome.exploration
                },
                'uncertainty': trade_outcome.uncertainty_estimate,
                'regime_confidence': 0.5  # Would extract from intelligence data
            }
            
            # Determine if this should be priority experience
            force_priority = abs(reward) > 0.5 or trade_outcome.uncertainty_estimate > 0.7
            
            success = self.experience_manager.store_experience(experience, force_priority)
            
            if success:
                if force_priority:
                    logger.info("Experience added to priority buffer")
                else:
                    logger.info("Experience added to regular buffer")
            
            return success
            
        except Exception as e:
            logger.error(f"Experience storage failed: {e}")
            return False
    
    def _conditional_network_training(self) -> bool:
        """Trigger network training if sufficient experience is available"""
        try:
            buffer_stats = self.experience_manager.get_buffer_stats()
            experience_size = buffer_stats['buffer_sizes']['experience']
            priority_size = buffer_stats['buffer_sizes']['priority']
            
            # Train if we have sufficient experience
            if experience_size >= 64 or priority_size >= 32:
                logger.info("Triggering network training...")
                
                # Sample training batch
                batch = self.experience_manager.sample_training_batch(
                    batch_size=32,
                    priority_ratio=0.25,
                    include_previous_tasks=True,
                    previous_task_ratio=0.25
                )
                
                if len(batch) >= 16:
                    # Train networks
                    loss_breakdown = self.network_manager.train_networks(batch)
                    
                    if 'error' not in loss_breakdown:
                        logger.info(f"Network training completed! Total loss: {loss_breakdown.get('total_loss', 'unknown')}")
                        self.processing_stats['network_training_cycles'] += 1
                        return True
                    else:
                        logger.error(f"Network training failed: {loss_breakdown['error']}")
                        return False
                else:
                    logger.warning(f"Insufficient batch size for training: {len(batch)}")
                    return False
            else:
                logger.debug(f"Insufficient experience for training: exp={experience_size}, pri={priority_size}")
                return True  # Not an error, just not enough data yet
                
        except Exception as e:
            logger.error(f"Network training failed: {e}")
            return False
    
    def _extract_subsystem_contributions(self, trade_outcome: TradeOutcome) -> torch.Tensor:
        """Extract subsystem contributions from trade outcome"""
        try:
            if trade_outcome.intelligence_data and 'subsystem_signals' in trade_outcome.intelligence_data:
                signals = trade_outcome.intelligence_data['subsystem_signals']
                return torch.tensor(signals[:6] + [0] * (6 - len(signals)), dtype=torch.float32)
            else:
                return torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.float32)
                
        except Exception as e:
            logger.error(f"Error extracting subsystem contributions: {e}")
            return torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.float32)
    
    def _calculate_subsystem_agreement(self, trade_outcome: TradeOutcome) -> float:
        """Calculate agreement between subsystems"""
        try:
            if trade_outcome.intelligence_data and 'subsystem_signals' in trade_outcome.intelligence_data:
                signals = trade_outcome.intelligence_data['subsystem_signals']
                
                if not signals or all(s == 0 for s in signals):
                    return 0.5
                
                # Calculate directional agreement
                positive_signals = sum(1 for s in signals if s > 0.1)
                negative_signals = sum(1 for s in signals if s < -0.1)
                total_signals = len([s for s in signals if abs(s) > 0.05])
                
                if total_signals == 0:
                    return 0.5
                
                directional_agreement = max(positive_signals, negative_signals) / len(signals)
                return directional_agreement
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error calculating subsystem agreement: {e}")
            return 0.5
    
    def _safe_get_signal(self, dopamine_response) -> float:
        """Safely extract dopamine signal from response object or dictionary"""
        try:
            if hasattr(dopamine_response, 'signal'):
                return float(dopamine_response.signal)
            elif isinstance(dopamine_response, dict):
                return float(dopamine_response.get('signal', 0.0))
            else:
                logger.warning(f"Unexpected dopamine response type: {type(dopamine_response)}")
                return 0.0
        except (ValueError, TypeError) as e:
            logger.error(f"Error extracting dopamine signal: {e}")
            return 0.0
    
    def _safe_get_state(self, dopamine_response) -> str:
        """Safely extract dopamine state from response object or dictionary"""
        try:
            if hasattr(dopamine_response, 'state'):
                state = dopamine_response.state
                return state.value if hasattr(state, 'value') else str(state)
            elif isinstance(dopamine_response, dict):
                state = dopamine_response.get('state', 'balanced')
                return state.value if hasattr(state, 'value') else str(state)
            else:
                logger.warning(f"Unexpected dopamine response type: {type(dopamine_response)}")
                return 'balanced'
        except (AttributeError, TypeError) as e:
            logger.error(f"Error extracting dopamine state: {e}")
            return 'balanced'
    
    def _safe_get_tolerance(self, dopamine_response) -> float:
        """Safely extract tolerance level from response object or dictionary"""
        try:
            if hasattr(dopamine_response, 'tolerance_level'):
                return float(dopamine_response.tolerance_level)
            elif isinstance(dopamine_response, dict):
                return float(dopamine_response.get('tolerance_level', 0.5))
            else:
                logger.warning(f"Unexpected dopamine response type: {type(dopamine_response)}")
                return 0.5
        except (ValueError, TypeError) as e:
            logger.error(f"Error extracting tolerance level: {e}")
            return 0.5
    
    def _safe_get_addiction_risk(self, dopamine_response) -> float:
        """Safely extract addiction risk from response object or dictionary"""
        try:
            if hasattr(dopamine_response, 'addiction_risk'):
                return float(dopamine_response.addiction_risk)
            elif isinstance(dopamine_response, dict):
                return float(dopamine_response.get('addiction_risk', 0.0))
            else:
                logger.warning(f"Unexpected dopamine response type: {type(dopamine_response)}")
                return 0.0
        except (ValueError, TypeError) as e:
            logger.error(f"Error extracting addiction risk: {e}")
            return 0.0

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        try:
            coordinator_stats = self.learning_coordinator.learning_stats.copy()
            
            return {
                'processing_stats': self.processing_stats.copy(),
                'learning_coordination': coordinator_stats,
                'success_rate': (
                    self.processing_stats['successful_processing'] / 
                    max(1, self.processing_stats['total_trades_processed'])
                ),
                'component_update_rates': {
                    'meta_learner_rate': (
                        self.processing_stats['meta_learner_updates'] / 
                        max(1, self.processing_stats['total_trades_processed'])
                    ),
                    'dopamine_integration_rate': (
                        self.processing_stats['dopamine_integrations'] / 
                        max(1, self.processing_stats['total_trades_processed'])
                    ),
                    'network_training_rate': (
                        self.processing_stats['network_training_cycles'] / 
                        max(1, self.processing_stats['total_trades_processed'])
                    )
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting processing stats: {e}")
            return {'error': str(e)}