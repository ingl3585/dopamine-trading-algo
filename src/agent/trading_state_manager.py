"""
Trading State Manager - Centralized state and confidence management

This module handles all trading state management:
1. Confidence calculation and recovery mechanisms
2. Enhanced state creation and aggregation
3. Meta-context building for decision making
4. Trading constraints evaluation
5. Performance tracking and statistics
6. State persistence and recovery

Extracted from TradingAgent to centralize state management and improve maintainability.
"""

import torch
import numpy as np
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import deque

from src.shared.types import Features
from src.core.market_data_processor import MarketData
from src.agent.confidence import ConfidenceManager
from src.agent.meta_learner import MetaLearner
from src.neural.adaptive_network import StateEncoder

logger = logging.getLogger(__name__)


class StateBuilder:
    """Builds enhanced state representations for decision making"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.state_encoder = StateEncoder()
        
    def create_enhanced_state(self, 
                            market_data: MarketData, 
                            features: Features, 
                            meta_context: Dict[str, Any]) -> torch.Tensor:
        """
        Create enhanced state representation with comprehensive features
        
        Args:
            market_data: Current market state
            features: Processed market features
            meta_context: Meta-context for enhanced features
            
        Returns:
            Enhanced state tensor with exactly 100 features
        """
        try:
            # Base state from encoder
            base_state = self.state_encoder.create_full_state(market_data, features, meta_context)
            
            # Enhanced features with explicit float conversion
            enhanced_features = torch.tensor([
                # Microstructure features
                float(features.microstructure_signal),
                float(features.regime_adjusted_signal),
                float(features.smart_money_flow),
                float(features.liquidity_depth),
                float(features.regime_confidence),
                
                # Regime and adaptation features
                float(features.adaptation_quality),
                float(meta_context.get('regime_transitions', 0.0)),
                float(meta_context.get('adaptation_events', 0.0)),
                
                # Account context
                float(min(1.0, market_data.account_balance / 50000)),
                float(market_data.margin_utilization),
                float(market_data.buying_power_ratio),
                
                # Position limit awareness
                float(getattr(market_data, 'total_position_size', 0) / 10.0),
                float(abs(getattr(market_data, 'total_position_size', 0)) / 10.0),  # Position ratio placeholder
                float(min(1.0, meta_context.get('recent_position_rejections', 0) / 10.0)),
                
                # Time-based features
                float(np.sin(2 * np.pi * features.time_of_day)),
                float(np.cos(2 * np.pi * features.time_of_day)),
                
                # Volatility regime features
                float(min(1.0, features.volatility / 0.05)),
                float(np.tanh(features.price_momentum * 10)),
                
                # Pattern strength indicators
                float(features.pattern_score),
                float(features.confidence),
                
                # Cross-timeframe features (if available)
                float(getattr(features, 'tf_5m_momentum', 0.0)),
                float(getattr(features, 'tf_15m_momentum', 0.0))
            ], dtype=torch.float32, device=self.device)
            
            # Combine base state with enhanced features
            full_state = torch.cat([base_state, enhanced_features])
            
            # Pad or truncate to exactly 100 features
            if len(full_state) < 100:
                padding = torch.zeros(100 - len(full_state), dtype=torch.float32, device=self.device)
                full_state = torch.cat([full_state, padding])
            else:
                full_state = full_state[:100]
            
            return full_state
            
        except Exception as e:
            logger.error(f"Error creating enhanced state: {e}")
            # Return safe default state
            return torch.zeros(100, dtype=torch.float32, device=self.device)


class MetaContextBuilder:
    """Builds meta-context for enhanced decision making"""
    
    def build_enhanced_meta_context(self,
                                  market_data: MarketData,
                                  features: Features,
                                  portfolio_summary: Dict[str, Any],
                                  trading_stats: Dict[str, Any]) -> Dict[str, float]:
        """
        Build enhanced meta-context with comprehensive information
        
        Args:
            market_data: Current market data
            features: Processed features
            portfolio_summary: Portfolio state summary
            trading_stats: Trading statistics
            
        Returns:
            Enhanced meta-context dictionary
        """
        try:
            # Base context from portfolio and trading stats
            base_context = {
                'recent_performance': np.tanh(portfolio_summary.get('daily_pnl', 0) / (market_data.account_balance * 0.01)),
                'consecutive_losses': portfolio_summary.get('consecutive_losses', 0),
                'position_count': portfolio_summary.get('pending_orders', 0),
                'trades_today': portfolio_summary.get('total_trades', 0),
                'time_since_last_trade': 0.0,  # Will be updated by caller
                'learning_efficiency': trading_stats.get('learning_efficiency', 0.5),
                'architecture_generation': trading_stats.get('architecture_generation', 0)
            }
            
            # Enhanced context with regime and adaptation awareness
            enhanced_context = {
                'regime_confidence': features.regime_confidence,
                'microstructure_strength': abs(features.microstructure_signal),
                'adaptation_quality': features.adaptation_quality,
                'regime_transitions': trading_stats.get('regime_transitions', 0) / max(1, trading_stats.get('total_decisions', 1)),
                'adaptation_events': trading_stats.get('adaptation_events', 0) / max(1, trading_stats.get('total_decisions', 1)),
                'volatility_regime': min(1.0, features.volatility / 0.05),
                'liquidity_regime': features.liquidity_depth,
                'smart_money_activity': abs(features.smart_money_flow),
                'recent_position_rejections': trading_stats.get('recent_position_rejections', 0)
            }
            
            return {**base_context, **enhanced_context}
            
        except Exception as e:
            logger.error(f"Error building meta-context: {e}")
            return {
                'recent_performance': 0.0,
                'consecutive_losses': 0,
                'position_count': 0,
                'trades_today': 0,
                'time_since_last_trade': 0.0,
                'learning_efficiency': 0.5,
                'regime_confidence': 0.5,
                'volatility_regime': 0.5
            }


class ConstraintsEvaluator:
    """Evaluates trading constraints and limitations"""
    
    def __init__(self, meta_learner: MetaLearner):
        self.meta_learner = meta_learner
    
    def should_consider_trading(self,
                              market_data: MarketData,
                              meta_context: Dict[str, Any],
                              features: Features,
                              last_trade_time: float) -> bool:
        """
        Evaluate whether trading should be considered given current constraints
        
        Args:
            market_data: Current market data
            meta_context: Enhanced meta-context
            features: Processed features
            last_trade_time: Timestamp of last trade
            
        Returns:
            True if trading should be considered
        """
        try:
            # Base constraints from meta-learner
            loss_tolerance = self.meta_learner.get_parameter('loss_tolerance_factor')
            max_loss = market_data.account_balance * loss_tolerance
            
            if market_data.daily_pnl <= -max_loss:
                logger.info(f"Trading blocked: Daily loss limit reached ({market_data.daily_pnl:.2f} <= {-max_loss:.2f})")
                return False
            
            # Consecutive loss limit
            consecutive_limit = self.meta_learner.get_parameter('consecutive_loss_tolerance')
            if meta_context['consecutive_losses'] >= consecutive_limit:
                logger.info(f"Trading blocked: Consecutive loss limit ({meta_context['consecutive_losses']} >= {consecutive_limit})")
                return False
            
            # Frequency constraints
            frequency_limit = self.meta_learner.get_parameter('trade_frequency_base')
            time_since_last = market_data.timestamp - last_trade_time
            
            if time_since_last < (1 / frequency_limit):
                logger.debug(f"Trading blocked: Frequency limit ({time_since_last:.1f}s < {1/frequency_limit:.1f}s)")
                return False
            
            # Intelligence signal strength check
            intelligence_threshold = self.meta_learner.get_parameter('intelligence_threshold', 0.3)
            intelligence_signal_strength = abs(features.overall_signal)
            
            if intelligence_signal_strength < intelligence_threshold:
                logger.info(f"Trading blocked: Intelligence signal {intelligence_signal_strength:.3f} "
                           f"below threshold {intelligence_threshold:.3f}")
                return False
            
            # Adaptation engine emergency mode
            if meta_context.get('adaptation_events', 0) > 0.1:
                logger.info("Trading blocked: Too many adaptation events")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error evaluating trading constraints: {e}")
            return False


class TradingStateManager:
    """
    Centralized manager for all trading state and confidence operations.
    
    This class consolidates state management functionality that was previously
    scattered throughout TradingAgent, providing a clean interface for state
    creation, confidence management, and trading constraints evaluation.
    """
    
    def __init__(self,
                 config: Dict[str, Any],
                 confidence_manager: Optional[ConfidenceManager] = None,
                 meta_learner: Optional[MetaLearner] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize the trading state manager
        
        Args:
            config: Configuration dictionary
            confidence_manager: Confidence calculation and recovery (created if None)
            meta_learner: Meta-learning component (optional)
            device: PyTorch device for tensor operations (auto-detected if None)
        """
        # Handle device
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        
        # Create or use provided components
        self.confidence_manager = confidence_manager or ConfidenceManager()
        self.meta_learner = meta_learner
        
        # Store config
        self.config = config
        
        # Initialize components
        self.state_builder = StateBuilder(device)
        self.context_builder = MetaContextBuilder()
        self.constraints_evaluator = ConstraintsEvaluator(meta_learner)
        
        # State tracking
        self.last_trade_time = 0.0
        self.recent_position_rejections = 0
        self.total_decisions = 0
        self.regime_transitions = 0
        self.adaptation_events = 0
        
        # Performance tracking
        self.state_management_stats = {
            'states_created': 0,
            'confidence_calculations': 0,
            'constraint_evaluations': 0,
            'trading_blocks': 0,
            'trading_allows': 0
        }
        
        logger.info("TradingStateManager initialized")
    
    def create_decision_state(self,
                            market_data: MarketData,
                            features: Features,
                            portfolio_summary: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Create comprehensive state for decision making
        
        Args:
            market_data: Current market data
            features: Processed market features
            portfolio_summary: Portfolio state summary
            
        Returns:
            Tuple of (enhanced_state_tensor, meta_context_dict)
        """
        try:
            # Build trading statistics for context
            trading_stats = {
                'total_decisions': self.total_decisions,
                'regime_transitions': self.regime_transitions,
                'adaptation_events': self.adaptation_events,
                'recent_position_rejections': self.recent_position_rejections,
                'learning_efficiency': self.meta_learner.get_learning_efficiency(),
                'architecture_generation': getattr(self.meta_learner.architecture_evolver, 'generations', 0)
            }
            
            # Build enhanced meta-context
            meta_context = self.context_builder.build_enhanced_meta_context(
                market_data, features, portfolio_summary, trading_stats
            )
            
            # Update time since last trade
            if self.last_trade_time > 0:
                meta_context['time_since_last_trade'] = np.log(1 + (time.time() - self.last_trade_time) / 3600)
            
            # Create enhanced state
            enhanced_state = self.state_builder.create_enhanced_state(
                market_data, features, meta_context
            )
            
            self.state_management_stats['states_created'] += 1
            
            return enhanced_state, meta_context
            
        except Exception as e:
            logger.error(f"Error creating decision state: {e}")
            # Return safe defaults
            safe_state = torch.zeros(100, dtype=torch.float32, device=self.device)
            safe_context = {'regime_confidence': 0.5, 'volatility_regime': 0.5}
            return safe_state, safe_context
    
    def process_confidence(self,
                         raw_confidence: float,
                         market_context: Dict[str, Any]) -> float:
        """
        Process confidence with recovery mechanisms
        
        Args:
            raw_confidence: Raw confidence from neural network
            market_context: Market context for confidence processing
            
        Returns:
            Processed confidence value
        """
        try:
            logger.info(f"CONFIDENCE TRACE: raw_network_output={raw_confidence:.6f}")
            
            # Validate network output
            if raw_confidence < 0.1:
                logger.error(f"CRITICAL: Neural network outputting very low confidence: {raw_confidence:.6f}")
            
            # Use centralized confidence manager
            processed_confidence = self.confidence_manager.process_neural_output(
                raw_confidence, market_context
            )
            
            logger.info(f"CONFIDENCE TRACE: after_processing={processed_confidence:.6f}")
            
            if processed_confidence != raw_confidence:
                logger.info(f"Confidence processed: {raw_confidence:.3f} -> {processed_confidence:.3f}")
            
            if processed_confidence < 0.15:
                logger.error(f"CRITICAL: Final confidence very low after processing: {processed_confidence:.6f}")
            
            self.state_management_stats['confidence_calculations'] += 1
            
            return processed_confidence
            
        except Exception as e:
            logger.error(f"Error processing confidence: {e}")
            return max(0.15, raw_confidence)  # Safe fallback
    
    def evaluate_trading_constraints(self,
                                   market_data: MarketData,
                                   meta_context: Dict[str, Any],
                                   features: Features) -> bool:
        """
        Evaluate whether trading should be allowed given current constraints
        
        Args:
            market_data: Current market data
            meta_context: Enhanced meta-context
            features: Processed features
            
        Returns:
            True if trading should be allowed
        """
        try:
            should_trade = self.constraints_evaluator.should_consider_trading(
                market_data, meta_context, features, self.last_trade_time
            )
            
            self.state_management_stats['constraint_evaluations'] += 1
            
            if should_trade:
                self.state_management_stats['trading_allows'] += 1
            else:
                self.state_management_stats['trading_blocks'] += 1
            
            return should_trade
            
        except Exception as e:
            logger.error(f"Error evaluating trading constraints: {e}")
            return False  # Safe default - block trading on error
    
    def update_decision_tracking(self):
        """Update decision tracking counters"""
        self.total_decisions += 1
    
    def update_trade_time(self, timestamp: float):
        """Update last trade timestamp"""
        self.last_trade_time = timestamp
    
    def handle_position_rejection(self, rejection_data: Dict[str, Any]):
        """Handle position limit rejection"""
        try:
            current_time = time.time()
            
            # Use confidence manager for rejection handling
            self.confidence_manager.handle_position_rejection(rejection_data)
            
            # Update local tracking
            self.recent_position_rejections += 1
            
            logger.info(f"Position rejection handled. Recent rejections: {self.recent_position_rejections}")
            
        except Exception as e:
            logger.error(f"Error handling position rejection: {e}")
    
    def get_current_confidence(self) -> float:
        """Get current confidence level"""
        try:
            return self.confidence_manager.get_current_confidence()
        except Exception as e:
            logger.error(f"Error getting current confidence: {e}")
            return 0.5
    
    def get_confidence_health(self) -> Dict[str, Any]:
        """Get confidence health status"""
        try:
            return self.confidence_manager.get_confidence_health()
        except Exception as e:
            logger.error(f"Error getting confidence health: {e}")
            return {'status': 'error', 'confidence': 0.5}
    
    def get_confidence_debug_info(self) -> Dict[str, Any]:
        """Get detailed confidence debug information"""
        try:
            return self.confidence_manager.get_debug_info()
        except Exception as e:
            logger.error(f"Error getting confidence debug info: {e}")
            return {'error': str(e)}
    
    def increment_regime_transitions(self):
        """Increment regime transition counter"""
        self.regime_transitions += 1
    
    def increment_adaptation_events(self):
        """Increment adaptation events counter"""
        self.adaptation_events += 1
    
    def get_state_statistics(self) -> Dict[str, Any]:
        """Get comprehensive state management statistics"""
        try:
            confidence_health = self.get_confidence_health()
            
            return {
                'decision_tracking': {
                    'total_decisions': self.total_decisions,
                    'regime_transitions': self.regime_transitions,
                    'adaptation_events': self.adaptation_events,
                    'recent_position_rejections': self.recent_position_rejections,
                    'last_trade_time': self.last_trade_time
                },
                'state_management': self.state_management_stats.copy(),
                'confidence_health': confidence_health,
                'performance_ratios': {
                    'trading_allow_rate': (
                        self.state_management_stats['trading_allows'] / 
                        max(1, self.state_management_stats['constraint_evaluations'])
                    ),
                    'confidence_processing_rate': (
                        self.state_management_stats['confidence_calculations'] /
                        max(1, self.state_management_stats['states_created'])
                    )
                },
                'current_state': {
                    'current_confidence': self.get_current_confidence(),
                    'time_since_last_trade': time.time() - self.last_trade_time if self.last_trade_time > 0 else 0,
                    'recent_rejection_rate': min(1.0, self.recent_position_rejections / 10.0)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting state statistics: {e}")
            return {'error': str(e)}
    
    def reset_session_state(self):
        """Reset state for new trading session"""
        try:
            # Reset tracking counters
            self.total_decisions = 0
            self.regime_transitions = 0
            self.adaptation_events = 0
            self.recent_position_rejections = 0
            self.last_trade_time = 0.0
            
            # Reset statistics
            self.state_management_stats = {
                'states_created': 0,
                'confidence_calculations': 0,
                'constraint_evaluations': 0,
                'trading_blocks': 0,
                'trading_allows': 0
            }
            
            logger.info("Trading state manager session reset")
            
        except Exception as e:
            logger.error(f"Error resetting session state: {e}")
    
    def save_state(self, filepath: str) -> bool:
        """
        Save state manager state to file
        
        Args:
            filepath: Path to save state
            
        Returns:
            True if save was successful
        """
        try:
            import json
            import os
            
            # Ensure directory exists
            dir_path = os.path.dirname(filepath)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            
            state_data = {
                'decision_tracking': {
                    'total_decisions': self.total_decisions,
                    'regime_transitions': self.regime_transitions,
                    'adaptation_events': self.adaptation_events,
                    'recent_position_rejections': self.recent_position_rejections,
                    'last_trade_time': self.last_trade_time
                },
                'state_management_stats': self.state_management_stats.copy(),
                'saved_at': time.time()
            }
            
            with open(filepath, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            logger.info(f"State manager state saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving state manager state: {e}")
            return False
    
    def load_state(self, filepath: str) -> bool:
        """
        Load state manager state from file
        
        Args:
            filepath: Path to load state from
            
        Returns:
            True if load was successful
        """
        try:
            import json
            
            with open(filepath, 'r') as f:
                state_data = json.load(f)
            
            # Load decision tracking
            tracking = state_data.get('decision_tracking', {})
            self.total_decisions = tracking.get('total_decisions', 0)
            self.regime_transitions = tracking.get('regime_transitions', 0)
            self.adaptation_events = tracking.get('adaptation_events', 0)
            self.recent_position_rejections = tracking.get('recent_position_rejections', 0)
            self.last_trade_time = tracking.get('last_trade_time', 0.0)
            
            # Load statistics
            self.state_management_stats = state_data.get('state_management_stats', {
                'states_created': 0,
                'confidence_calculations': 0,
                'constraint_evaluations': 0,
                'trading_blocks': 0,
                'trading_allows': 0
            })
            
            logger.info(f"State manager state loaded from {filepath}")
            return True
            
        except FileNotFoundError:
            logger.info("No existing state manager state found, starting fresh")
            return False
        except Exception as e:
            logger.error(f"Error loading state manager state: {e}")
            return False