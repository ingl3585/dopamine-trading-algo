"""
Trading Decision Engine - Core decision-making logic extracted from TradingAgent

This module handles the core trading decision pipeline:
1. Feature processing and validation
2. Confidence calculation and recovery
3. Trading constraints evaluation  
4. Action selection (exploration vs exploitation)
5. Position sizing and risk parameters
6. Decision construction and validation

Extracted from TradingAgent.decide() to improve maintainability and testability.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from src.shared.types import Features
from src.core.market_data_processor import MarketData
from src.agent.confidence import ConfidenceManager
from src.agent.meta_learner import MetaLearner

logger = logging.getLogger(__name__)

@dataclass
class DecisionContext:
    """Context information for decision making"""
    features: Features
    market_data: MarketData
    meta_context: Dict[str, Any]
    dopamine_anticipation: Any
    learned_state: torch.Tensor
    subsystem_signals: torch.Tensor
    subsystem_weights: torch.Tensor
    
@dataclass  
class Decision:
    """Trading decision with comprehensive context"""
    action: str
    confidence: float
    size: float
    stop_price: float = 0.0
    target_price: float = 0.0
    primary_tool: str = 'unknown'
    exploration: bool = False
    intelligence_data: Dict = None
    state_features: list = None
    adaptation_strategy: str = 'conservative'
    uncertainty_estimate: float = 0.5
    few_shot_prediction: float = 0.0
    regime_awareness: Dict = None
    
    def __post_init__(self):
        """Validate decision parameters"""
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        if self.size < 0.0:
            raise ValueError(f"Size must be non-negative, got {self.size}")
        if self.action not in ['hold', 'buy', 'sell']:
            raise ValueError(f"Action must be 'hold', 'buy', or 'sell', got {self.action}")


class TradingDecisionEngine:
    """
    Core trading decision engine responsible for processing market data and features
    into actionable trading decisions.
    
    This class encapsulates the decision-making pipeline that was previously embedded
    in the monolithic TradingAgent class. It focuses solely on the decision logic
    without concerns about network training, experience storage, or other responsibilities.
    """
    
    def __init__(self,
                 config: Dict[str, Any],
                 confidence_manager: Optional[ConfidenceManager] = None,
                 meta_learner: Optional[MetaLearner] = None,
                 neural_manager: Optional[Any] = None,
                 intelligence_engine: Optional[Any] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize the decision engine
        
        Args:
            config: Configuration dictionary
            confidence_manager: Manages confidence calculation and recovery (optional)
            meta_learner: Provides adaptive parameters and exploration strategy (optional)
            neural_manager: Neural network manager (optional, passed but not used)
            device: PyTorch device for tensor operations (auto-detected if None)
        """
        # Store config
        self.config = config
        
        # Handle device
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        
        # Initialize or use provided components
        self.confidence_manager = confidence_manager or ConfidenceManager()
        self.meta_learner = meta_learner
        
        # Store neural manager reference (not used currently but passed by integrator)
        self.neural_manager = neural_manager
        
        # Store intelligence engine for real-time subsystem consultation
        self.intelligence_engine = intelligence_engine
        
        # Decision tracking
        self.total_decisions = 0
        
        # Initialize timestamps intelligently to prevent cold-start emergency override
        self.system_start_time = time.time()
        self.last_trade_time = self._get_intelligent_default_timestamp()
        
        # Emergency trading mechanisms
        self.last_emergency_override_time = 0.0
        self.consecutive_hold_decisions = 0
        self.last_successful_trade_time = self._get_intelligent_default_timestamp()
        
        # Cold-start protection
        self.startup_grace_period = config.get('emergency_override', {}).get('startup_grace_period_seconds', 7200)  # 2 hours
        self.is_cold_start = True  # Flag to track if we're in cold-start mode
        
        # Trading floor parameters - configurable emergency settings
        emergency_config = config.get('emergency_override', {})
        self.emergency_trading_floor = {
            'max_consecutive_holds': emergency_config.get('max_consecutive_holds', 50),
            'max_time_without_trade': emergency_config.get('max_time_without_trade_seconds', 3600),
            'confidence_floor': emergency_config.get('confidence_floor', 0.2),
            'intelligence_floor': emergency_config.get('intelligence_floor', 0.15),
            'emergency_cooldown': emergency_config.get('emergency_cooldown_seconds', 600),
            'startup_grace_period': self.startup_grace_period
        }
        
        # Log initialization with cold-start information
        logger.info(f"TradingDecisionEngine initialized with intelligent timestamps")
        logger.info(f"System start time: {time.ctime(self.system_start_time)}")
        logger.info(f"Default timestamp: {time.ctime(self.last_trade_time)} (startup grace period: {self.startup_grace_period}s)")
        logger.info(f"Cold-start mode: {self.is_cold_start}")
        logger.info(f"Emergency override config: {self.emergency_trading_floor}")
    
    def _get_intelligent_default_timestamp(self) -> float:
        """
        Get intelligent default timestamp for cold-start scenarios.
        
        Instead of using 0.0 which causes immediate emergency override,
        we use current_time - startup_grace_period to provide reasonable
        default that prevents false emergency activation.
        
        Returns:
            float: Intelligent default timestamp (current_time - grace_period)
        """
        current_time = time.time()
        grace_period = 7200  # 2 hours default
        default_timestamp = current_time - grace_period
        
        logger.debug(f"Intelligent default timestamp: {time.ctime(default_timestamp)} "
                    f"(current - {grace_period}s)")
        
        return default_timestamp
    
    def _is_cold_start(self, current_time: float) -> bool:
        """
        Detect if system is in cold-start mode.
        
        Cold-start is defined as the period immediately after system startup
        where emergency trading mechanisms should be disabled to prevent
        false activation due to uninitialized timestamps.
        
        Args:
            current_time: Current system timestamp
            
        Returns:
            bool: True if system is in cold-start grace period
        """
        time_since_startup = current_time - self.system_start_time
        in_grace_period = time_since_startup < self.startup_grace_period
        
        if in_grace_period and self.is_cold_start:
            logger.debug(f"Cold-start mode: {time_since_startup:.0f}s since startup "
                        f"(grace period: {self.startup_grace_period}s)")
            return True
        elif not in_grace_period and self.is_cold_start:
            # Exit cold-start mode
            self.is_cold_start = False
            logger.info(f"Exiting cold-start mode after {time_since_startup:.0f}s")
            return False
        
        return False
    
    def decide(self, 
               features: Features, 
               market_data: MarketData,
               network_outputs: Dict[str, torch.Tensor],
               dopamine_anticipation: Any,
               learned_state: torch.Tensor,
               adaptation_decision: Dict[str, Any]) -> Decision:
        """
        Core decision-making pipeline
        
        Args:
            features: Processed market features
            market_data: Current market state
            network_outputs: Neural network predictions
            dopamine_anticipation: Dopamine system state
            learned_state: Processed feature state
            adaptation_decision: Real-time adaptation context
            
        Returns:
            Trading decision with full context
        """
        self.total_decisions += 1
        
        try:
            # Build decision context with live subsystem consultation
            context = self._build_decision_context_with_tools(
                features, market_data, dopamine_anticipation, 
                learned_state, adaptation_decision
            )
            
            # Process confidence with recovery mechanisms
            confidence = self._process_confidence(context, network_outputs)
            
            # Check for emergency trading override BEFORE normal constraints
            emergency_override = self._check_emergency_trading_override(
                context, confidence, market_data.timestamp
            )
            
            if emergency_override:
                # Force an emergency trade to break conservative feedback loop
                logger.warning("EMERGENCY TRADING OVERRIDE ACTIVATED")
                logger.info(f"Emergency override context: cold_start={self.is_cold_start}, "
                           f"consecutive_holds={self.consecutive_hold_decisions}, "
                           f"time_since_startup={market_data.timestamp - self.system_start_time:.0f}s")
                return self._create_emergency_decision(context, confidence, network_outputs)
            
            # Check trading constraints
            if not self._should_consider_trading(context, confidence):
                self.consecutive_hold_decisions += 1
                return self._create_hold_decision(confidence, context)
            
            # Determine action (exploration vs exploitation)
            action_idx, exploration = self._select_action(
                context, network_outputs, adaptation_decision
            )
            
            # Apply confidence and intelligence thresholds with emergency floor
            if self._should_block_trade_with_emergency_floor(confidence, features, exploration):
                action_idx = 0  # Force hold
                self.consecutive_hold_decisions += 1
            else:
                # Reset consecutive holds counter on potential trade
                self.consecutive_hold_decisions = 0
            
            # Calculate position sizing and risk parameters
            position_size, risk_params = self._calculate_position_and_risk(
                context, network_outputs, dopamine_anticipation
            )
            
            # Convert to decision
            return self._create_decision(
                action_idx, confidence, position_size, risk_params,
                context, adaptation_decision, exploration
            )
            
        except Exception as e:
            logger.error(f"Error in decision pipeline: {e}")
            # Return safe hold decision on error
            return Decision('hold', 0.3, 0, regime_awareness=adaptation_decision)
    
    def _get_dopamine_value(self, dopamine_anticipation: Any, dict_key: str, attr_key: str, default_value: Any) -> Any:
        """Helper method to safely extract values from dopamine_anticipation object or dict"""
        if not dopamine_anticipation:
            return default_value
        
        # Try dictionary access first (current implementation)
        if isinstance(dopamine_anticipation, dict):
            return dopamine_anticipation.get(dict_key, default_value)
        
        # Try object attribute access
        if hasattr(dopamine_anticipation, attr_key):
            attr_value = getattr(dopamine_anticipation, attr_key)
            # Handle nested attribute access for state.value
            if attr_key == 'state' and hasattr(attr_value, 'value'):
                return attr_value.value
            return attr_value
        
        return default_value

    def _build_decision_context(self,
                               features: Features,
                               market_data: MarketData, 
                               dopamine_anticipation: Any,
                               learned_state: torch.Tensor,
                               adaptation_decision: Dict[str, Any]) -> DecisionContext:
        """Build comprehensive decision context"""
        
        # Enhanced subsystem contributions with dopamine influence
        base_dopamine_signal = float(features.dopamine_signal)
        # Handle both object attributes and dictionary keys
        if dopamine_anticipation:
            if hasattr(dopamine_anticipation, 'position_size_modifier'):
                position_modifier = dopamine_anticipation.position_size_modifier
            elif isinstance(dopamine_anticipation, dict):
                position_modifier = dopamine_anticipation.get('position_size_modifier', 1.0)
            else:
                position_modifier = 1.0
        else:
            position_modifier = 1.0
        dopamine_amplified_signal = base_dopamine_signal * position_modifier
        
        subsystem_signals = torch.tensor([
            float(features.dna_signal),
            float(features.temporal_signal),
            float(features.immune_signal),
            float(features.microstructure_signal),
            dopamine_amplified_signal,
            float(features.regime_adjusted_signal)
        ], dtype=torch.float32, device=self.device)
        
        subsystem_weights = self.meta_learner.get_subsystem_weights()
        
        # Expand weights to match signals if needed
        if len(subsystem_weights) < len(subsystem_signals):
            additional_weights = torch.ones(
                len(subsystem_signals) - len(subsystem_weights), 
                dtype=torch.float32, device=self.device
            ) * 0.1
            subsystem_weights = torch.cat([subsystem_weights, additional_weights])
        
        # Build meta context
        meta_context = {
            'regime_confidence': features.regime_confidence,
            'microstructure_strength': abs(features.microstructure_signal),
            'adaptation_quality': features.adaptation_quality,
            'volatility_regime': min(1.0, features.volatility / 0.05),
            'liquidity_regime': features.liquidity_depth,
            'smart_money_activity': abs(features.smart_money_flow),
            'dopamine_state': self._get_dopamine_value(dopamine_anticipation, 'dopamine_state', 'state', 'baseline'),
            'dopamine_urgency': self._get_dopamine_value(dopamine_anticipation, 'urgency_factor', 'urgency_factor', 0.5)
        }
        
        return DecisionContext(
            features=features,
            market_data=market_data,
            meta_context=meta_context,
            dopamine_anticipation=dopamine_anticipation,
            learned_state=learned_state,
            subsystem_signals=subsystem_signals,
            subsystem_weights=subsystem_weights
        )
    
    def _build_decision_context_with_tools(self,
                                         features: Features,
                                         market_data: MarketData, 
                                         dopamine_anticipation: Any,
                                         learned_state: torch.Tensor,
                                         adaptation_decision: Dict[str, Any]) -> DecisionContext:
        """Build decision context by actively consulting AI subsystems as tools"""
        
        # If we have an intelligence engine, use subsystems as real-time tools
        if self.intelligence_engine:
            # Consult the 5 AI subsystems in real-time for this specific decision
            tool_consultations = self._consult_ai_subsystem_tools(market_data, features)
            
            # Build subsystem signals from live tool consultation
            subsystem_signals = torch.tensor([
                tool_consultations['dna_recommendation'],
                tool_consultations['temporal_recommendation'], 
                tool_consultations['immune_recommendation'],
                tool_consultations['microstructure_recommendation'],
                tool_consultations['dopamine_recommendation'],
                float(features.regime_adjusted_signal)  # Keep existing regime signal
            ], dtype=torch.float32, device=self.device)
            
            # Weight subsystems based on their confidence in current market conditions
            subsystem_weights = torch.tensor([
                tool_consultations['dna_confidence'],
                tool_consultations['temporal_confidence'],
                tool_consultations['immune_confidence'], 
                tool_consultations['microstructure_confidence'],
                tool_consultations['dopamine_confidence'],
                0.5  # Default regime weight
            ], dtype=torch.float32, device=self.device)
            
        else:
            # Fallback to original method if no intelligence engine available
            return self._build_decision_context(features, market_data, dopamine_anticipation, learned_state, adaptation_decision)
        
        # Build meta context for learning and adaptation
        meta_context = self._build_meta_context(features, adaptation_decision)
        
        return DecisionContext(
            features=features,
            market_data=market_data,
            meta_context=meta_context,
            dopamine_anticipation=dopamine_anticipation,
            learned_state=learned_state,
            subsystem_signals=subsystem_signals,
            subsystem_weights=subsystem_weights
        )
    
    def _consult_ai_subsystem_tools(self, market_data: MarketData, features: Features) -> Dict[str, float]:
        """
        Actively consult each AI subsystem as an intelligent tool for decision-making
        
        This is the core method that makes subsystems true tools rather than signal generators
        """
        try:
            # Prepare market context for subsystem consultation
            market_context = {
                'prices': getattr(market_data, 'prices_1m', [market_data.close]),
                'volumes': getattr(market_data, 'volumes_1m', [market_data.volume]),
                'timestamps': getattr(market_data, 'timestamps', [market_data.timestamp]),
                'current_price': market_data.close,
                'current_volume': market_data.volume,
                'volatility': features.volatility,
                'price_momentum': features.price_momentum,
                'volume_momentum': features.volume_momentum
            }
            
            # Get comprehensive intelligence analysis
            intelligence_signals = self.intelligence_engine.analyze_market(
                historical_context={
                    'prices': market_context['prices'],
                    'volumes': market_context['volumes'], 
                    'timestamps': market_context['timestamps']
                },
                market_features={
                    'volatility': market_context['volatility'],
                    'price_momentum': market_context['price_momentum'],
                    'volume_momentum': market_context['volume_momentum']
                }
            )
            
            # Extract tool recommendations and confidence levels
            consultations = {
                # DNA Subsystem Tool: Pattern recognition and sequence analysis
                'dna_recommendation': intelligence_signals['dna'].value,
                'dna_confidence': intelligence_signals['dna'].confidence,
                
                # Temporal Subsystem Tool: Cycle detection and timing analysis  
                'temporal_recommendation': intelligence_signals['temporal'].value,
                'temporal_confidence': intelligence_signals['temporal'].confidence,
                
                # Immune Subsystem Tool: Risk assessment and threat detection
                'immune_recommendation': intelligence_signals['immune'].value, 
                'immune_confidence': intelligence_signals['immune'].confidence,
                
                # Microstructure Subsystem Tool: Market regime and liquidity analysis
                'microstructure_recommendation': intelligence_signals['microstructure'].value,
                'microstructure_confidence': intelligence_signals['microstructure'].confidence,
                
                # Dopamine Subsystem Tool: Reward optimization and trading psychology
                'dopamine_recommendation': getattr(intelligence_signals.get('dopamine', None), 'value', features.dopamine_signal),
                'dopamine_confidence': getattr(intelligence_signals.get('dopamine', None), 'confidence', 0.5)
            }
            
            # Log tool consultation for transparency
            logger.debug(f"AI Subsystem Tool Consultation: "
                        f"DNA: {consultations['dna_recommendation']:.3f}({consultations['dna_confidence']:.2f}), "
                        f"Temporal: {consultations['temporal_recommendation']:.3f}({consultations['temporal_confidence']:.2f}), "
                        f"Immune: {consultations['immune_recommendation']:.3f}({consultations['immune_confidence']:.2f}), "
                        f"Micro: {consultations['microstructure_recommendation']:.3f}({consultations['microstructure_confidence']:.2f}), "
                        f"Dopamine: {consultations['dopamine_recommendation']:.3f}({consultations['dopamine_confidence']:.2f})")
            
            return consultations
            
        except Exception as e:
            logger.error(f"Error consulting AI subsystem tools: {e}")
            # Fallback to feature-based signals
            return {
                'dna_recommendation': float(features.dna_signal),
                'dna_confidence': 0.5,
                'temporal_recommendation': float(features.temporal_signal),
                'temporal_confidence': 0.5,
                'immune_recommendation': float(features.immune_signal),
                'immune_confidence': 0.5,
                'microstructure_recommendation': float(features.microstructure_signal),
                'microstructure_confidence': 0.5,
                'dopamine_recommendation': float(features.dopamine_signal),
                'dopamine_confidence': 0.5
            }
    
    def _process_confidence(self, 
                           context: DecisionContext,
                           network_outputs: Dict[str, torch.Tensor]) -> float:
        """Process confidence with recovery mechanisms"""
        
        # Extract raw confidence from network
        raw_confidence = float(network_outputs['confidence'].detach().cpu().numpy()[0])
        
        logger.info(f"CONFIDENCE TRACE: raw_from_network={raw_confidence:.6f}")
        
        # Validate network output
        if raw_confidence < 0.1:
            logger.error(f"CRITICAL: Neural network outputting raw_confidence={raw_confidence:.6f}")
            logger.error(f"Network confidence tensor: {network_outputs['confidence']}")
        
        # Use centralized confidence manager
        market_context = {
            'volatility': context.features.volatility,
            'price_momentum': context.features.price_momentum,
            'regime_confidence': context.features.regime_confidence
        }
        
        confidence = self.confidence_manager.process_neural_output(raw_confidence, market_context)
        
        logger.info(f"CONFIDENCE TRACE: after_recovery={confidence:.6f}")
        
        if confidence != raw_confidence:
            logger.info(f"Confidence processed: {raw_confidence:.3f} -> {confidence:.3f}")
        
        if confidence < 0.15:
            logger.error(f"CRITICAL: Final confidence={confidence:.6f} after processing")
        
        return confidence
    
    def _should_consider_trading(self, 
                                context: DecisionContext,
                                confidence: float) -> bool:
        """Evaluate whether trading should be considered given current constraints"""

        return True 
    
    def _check_emergency_trading_override(self, 
                                        context: DecisionContext, 
                                        confidence: float,
                                        current_time: float) -> bool:
        """
        Check if emergency trading override should be activated to break conservative feedback loops.
        
        This method implements emergency mechanisms to force trading when the system
        becomes trapped in overly conservative states, preventing permanent paralysis.
        
        Enhanced with cold-start protection to prevent false emergency activation
        immediately after system startup when timestamps are uninitialized.
        """
        # CRITICAL: Cold-start protection - disable emergency override during startup grace period
        if self._is_cold_start(current_time):
            logger.debug("Emergency override blocked: System in cold-start grace period")
            return False
        
        # Check cooldown period
        time_since_last_emergency = current_time - self.last_emergency_override_time
        if time_since_last_emergency < self.emergency_trading_floor['emergency_cooldown']:
            logger.debug(f"Emergency override blocked: Cooldown period "
                        f"({time_since_last_emergency:.0f}s < {self.emergency_trading_floor['emergency_cooldown']}s)")
            return False
        
        # Emergency condition 1: Too many consecutive hold decisions
        too_many_holds = (
            self.consecutive_hold_decisions >= self.emergency_trading_floor['max_consecutive_holds']
        )
        
        # Emergency condition 2: Too much time without any trading
        # Enhanced with intelligent timestamp validation
        last_trade_timestamp = max(self.last_trade_time, self.last_successful_trade_time)
        time_without_trade = current_time - last_trade_timestamp
        too_long_without_trade = (
            time_without_trade >= self.emergency_trading_floor['max_time_without_trade']
        )
        
        # Enhanced logging for debugging timestamp issues
        logger.debug(f"Emergency timestamp check: current={time.ctime(current_time)}, "
                    f"last_trade={time.ctime(self.last_trade_time)}, "
                    f"last_successful={time.ctime(self.last_successful_trade_time)}, "
                    f"time_without_trade={time_without_trade:.0f}s")
        
        # Emergency condition 3: Meta-learner indicates emergency override needed
        meta_learner_emergency = False
        if self.meta_learner:
            meta_learner_emergency = self.meta_learner._check_for_emergency_trading_override()
        
        # Emergency condition 4: Confidence manager indicates emergency reset needed
        confidence_manager_emergency = False
        if self.confidence_manager:
            confidence_manager_emergency = self.confidence_manager.check_for_emergency_confidence_reset()
        
        # Emergency condition 5: Intelligence signal exists but thresholds are blocking
        intelligence_signal_strength = abs(context.features.overall_signal)
        has_reasonable_signal = intelligence_signal_strength > 0.05  # Very low threshold
        blocked_by_thresholds = (
            confidence < self.meta_learner.get_parameter('confidence_threshold') or
            intelligence_signal_strength < self.meta_learner.get_parameter('intelligence_threshold')
        )
        signal_blockage_emergency = has_reasonable_signal and blocked_by_thresholds
        
        # Determine if emergency override is needed
        emergency_needed = (
            too_many_holds or 
            too_long_without_trade or 
            meta_learner_emergency or
            confidence_manager_emergency or
            signal_blockage_emergency
        )
        
        if emergency_needed:
            logger.warning(f"EMERGENCY TRADING OVERRIDE CONDITIONS: "
                         f"consecutive_holds={self.consecutive_hold_decisions}, "
                         f"time_without_trade={time_without_trade:.0f}s, "
                         f"meta_emergency={meta_learner_emergency}, "
                         f"confidence_emergency={confidence_manager_emergency}, "
                         f"signal_blockage={signal_blockage_emergency}")
            
            # Update emergency timestamp
            self.last_emergency_override_time = current_time
        
        return emergency_needed
    
    def _should_block_trade_with_emergency_floor(self, 
                                               confidence: float,
                                               features: Features,
                                               exploration: bool) -> bool:
        """
        Enhanced version of _should_block_trade that includes emergency trading floor.
        
        This method applies emergency floors to prevent complete trading paralysis
        while maintaining reasonable risk controls.
        """
        # Apply emergency floors - much lower thresholds to prevent paralysis
        emergency_confidence_floor = self.emergency_trading_floor['confidence_floor']
        emergency_intelligence_floor = self.emergency_trading_floor['intelligence_floor']
        
        # If we're at emergency levels, use emergency floors instead of normal thresholds
        if (confidence < emergency_confidence_floor * 1.5 or 
            abs(features.overall_signal) < emergency_intelligence_floor * 1.5):
            
            # Use emergency floors
            confidence_threshold = emergency_confidence_floor
            intelligence_threshold = emergency_intelligence_floor
            
            logger.info(f"EMERGENCY FLOORS ACTIVE: Using emergency thresholds "
                       f"conf={confidence_threshold:.3f}, intel={intelligence_threshold:.3f}")
        else:
            # Use normal thresholds
            base_threshold = self.meta_learner.get_parameter('confidence_threshold')
            regime_adjustment = (1.0 - features.regime_confidence) * 0.2
            volatility_adjustment = features.volatility * 0.3
            confidence_threshold = base_threshold + regime_adjustment + volatility_adjustment
            
            intelligence_threshold = self.meta_learner.get_parameter('intelligence_threshold', 0.3)
        
        intelligence_signal_strength = abs(features.overall_signal)
        
        # Block trades if either confidence OR intelligence signal is too weak
        # (but not during exploration or when emergency floors are protecting)
        if confidence < confidence_threshold and not exploration:
            logger.info(f"Trade blocked: Low confidence {confidence:.3f} < {confidence_threshold:.3f}")
            return True
        
        if intelligence_signal_strength < intelligence_threshold and not exploration:
            logger.info(f"Trade blocked: Weak intelligence signal {intelligence_signal_strength:.3f} "
                       f"< {intelligence_threshold:.3f}")
            return True
        
        return False
    
    def _create_emergency_decision(self,
                                 context: DecisionContext,
                                 confidence: float,
                                 network_outputs: Dict[str, torch.Tensor]) -> Decision:
        """
        Create an emergency trading decision to break conservative feedback loops.
        
        This method forces a small, safe trade to restart the learning process
        when the system becomes trapped in conservative states.
        """
        # Use weighted intelligence signal to determine direction
        weighted_signal = torch.sum(
            context.subsystem_signals * context.subsystem_weights[:len(context.subsystem_signals)]
        ).item()
        
        # Determine action based on signal direction
        if weighted_signal > 0.001:
            action = 'buy'
            action_idx = 1
        elif weighted_signal < -0.001:
            action = 'sell'
            action_idx = 2
        else:
            # If no clear signal, default to buy for emergency (slightly optimistic bias)
            action = 'buy'
            action_idx = 1
        
        # Use minimum safe position size for emergency trade
        emergency_position_size = 0.1  # Very small position
        
        # Calculate conservative stop and target levels
        base_price = context.market_data.price
        stop_distance = 0.01  # 1% stop
        target_distance = 0.02  # 2% target
        
        if action == 'buy':
            stop_price = base_price * (1 - stop_distance)
            target_price = base_price * (1 + target_distance)
        else:
            stop_price = base_price * (1 + stop_distance)
            target_price = base_price * (1 - target_distance)
        
        # Boost confidence for emergency decision
        emergency_confidence = max(confidence, self.emergency_trading_floor['confidence_floor'])
        
        # Create emergency decision
        decision = Decision(
            action=action,
            confidence=emergency_confidence,
            size=emergency_position_size,
            stop_price=stop_price,
            target_price=target_price,
            primary_tool='emergency_override',
            exploration=True,  # Mark as exploration to bypass normal blocks
            intelligence_data={
                'emergency_override': True,
                'weighted_signal': weighted_signal,
                'consecutive_holds': self.consecutive_hold_decisions,
                'subsystem_signals': context.subsystem_signals.detach().cpu().numpy().tolist(),
                'subsystem_weights': context.subsystem_weights.detach().cpu().numpy().tolist(),
                'regime_context': {
                    'regime_confidence': context.features.regime_confidence,
                    'volatility': context.features.volatility,
                    'trend_strength': abs(context.features.price_momentum)
                }
            },
            adaptation_strategy='emergency',
            uncertainty_estimate=0.3,  # Lower uncertainty for emergency decision
            regime_awareness={
                'emergency_override': True,
                'regime_confidence': context.features.regime_confidence,
                'volatility': context.features.volatility
            }
        )
        
        # Reset consecutive holds and update last trade time
        self.consecutive_hold_decisions = 0
        self.last_trade_time = context.market_data.timestamp
        
        logger.warning(f"EMERGENCY DECISION CREATED: {action} {emergency_position_size} contracts, "
                      f"confidence={emergency_confidence:.3f}, signal={weighted_signal:.4f}")
        
        return decision
    
    def _select_action(self,
                      context: DecisionContext,
                      network_outputs: Dict[str, torch.Tensor],
                      adaptation_decision: Dict[str, Any]) -> Tuple[int, bool]:
        """Select action using exploration vs exploitation strategy"""
        
        # Get action probabilities with temperature scaling
        temperature = 1.0 + context.features.volatility * 2.0
        action_logits = network_outputs['action_logits'] / temperature
        action_probs = F.softmax(action_logits, dim=-1).detach().cpu().numpy()[0]
        
        # Enhanced exploration decision
        should_explore = self.meta_learner.should_explore(
            context.learned_state.squeeze(), 
            context.meta_context
        ) or adaptation_decision.get('emergency_mode', False)
        
        # Strategy selection
        selected_strategy = adaptation_decision.get('strategy_name', 'conservative')
        
        if should_explore:
            # Strategic exploration based on current strategy
            weighted_signal = torch.sum(
                context.subsystem_signals * context.subsystem_weights[:len(context.subsystem_signals)]
            )
            action_idx = self._strategic_exploration(
                weighted_signal, selected_strategy, context.features, adaptation_decision
            )
            exploration = True
        else:
            # Exploitation with uncertainty consideration
            uncertainty = adaptation_decision.get('uncertainty', 0.5)
            if uncertainty > 0.7:  # High uncertainty, be more conservative
                action_probs[0] *= 1.5  # Boost hold probability
                action_probs = action_probs / np.sum(action_probs)  # Renormalize
            
            action_idx = np.argmax(action_probs)
            exploration = False
        
        return action_idx, exploration
    
    def _should_block_trade(self, 
                           confidence: float,
                           features: Features,
                           exploration: bool) -> bool:
        """Apply confidence and intelligence thresholds to block weak trades"""
        
        # Enhanced confidence threshold with regime awareness
        base_threshold = self.meta_learner.get_parameter('confidence_threshold')
        regime_adjustment = (1.0 - features.regime_confidence) * 0.2
        volatility_adjustment = features.volatility * 0.3
        confidence_threshold = base_threshold + regime_adjustment + volatility_adjustment
        
        # Intelligence signal strength threshold
        intelligence_threshold = self.meta_learner.get_parameter('intelligence_threshold', 0.3)
        intelligence_signal_strength = abs(features.overall_signal)
        
        # Block trades if either confidence OR intelligence signal is too weak
        if confidence < confidence_threshold and not exploration:
            logger.info(f"Trade blocked: Low confidence {confidence:.3f} < {confidence_threshold:.3f}")
            return True
        
        if intelligence_signal_strength < intelligence_threshold and not exploration:
            logger.info(f"Trade blocked: Weak intelligence signal {intelligence_signal_strength:.3f} "
                       f"< {intelligence_threshold:.3f}")
            return True
        
        return False
    
    def _calculate_position_and_risk(self,
                                   context: DecisionContext,
                                   network_outputs: Dict[str, torch.Tensor],
                                   dopamine_anticipation: Any) -> Tuple[float, Dict[str, float]]:
        """Calculate position size and risk parameters"""
        
        # Base position size from network
        raw_position_size = float(network_outputs['position_size'].detach().cpu().numpy()[0])
        base_position_size = max(0.1, abs(raw_position_size))
        
        # Apply dopamine psychological modifiers with null safety
        if dopamine_anticipation:
            if hasattr(dopamine_anticipation, 'position_size_modifier'):
                dopamine_position_modifier = dopamine_anticipation.position_size_modifier
                dopamine_risk_modifier = dopamine_anticipation.risk_tolerance_modifier
            elif isinstance(dopamine_anticipation, dict):
                dopamine_position_modifier = dopamine_anticipation.get('position_size_modifier', 1.0)
                dopamine_risk_modifier = dopamine_anticipation.get('risk_tolerance_modifier', 1.0)
            else:
                dopamine_position_modifier = 1.0
                dopamine_risk_modifier = 1.0
        else:
            dopamine_position_modifier = 1.0
            dopamine_risk_modifier = 1.0
        
        # Adjust position size based on dopamine state
        position_size = base_position_size * dopamine_position_modifier
        
        # Uncertainty-adjusted position sizing
        uncertainty_factor = 1.0 - context.meta_context.get('uncertainty', 0.5)
        position_size *= max(0.1, uncertainty_factor)
        
        # Apply dopamine urgency factor with null safety
        dopamine_urgency = self._get_dopamine_value(dopamine_anticipation, 'urgency_factor', 'urgency_factor', 0.5)
        urgency_adjustment = 0.8 + (dopamine_urgency * 0.4)  # 0.8 to 1.2 range
        position_size *= urgency_adjustment
        
        # Risk parameters
        risk_params_tensor = network_outputs['risk_params'].detach().cpu().numpy()[0]
        
        risk_params = {
            'use_stop': risk_params_tensor[0] > 0.5,
            'stop_distance': risk_params_tensor[1] * dopamine_risk_modifier,
            'use_target': risk_params_tensor[2] > 0.5,
            'target_distance': risk_params_tensor[3] * dopamine_risk_modifier
        }
        
        # Adjust risk parameters based on regime
        if context.features.regime_confidence < 0.5:  # Uncertain regime
            risk_params['stop_distance'] *= 0.8  # Tighter stops
        
        return position_size, risk_params
    
    def _strategic_exploration(self,
                             weighted_signal: torch.Tensor,
                             strategy: str,
                             features: Features,
                             adaptation_decision: Dict[str, Any]) -> int:
        """Enhanced exploration based on selected strategy"""
        
        if strategy == 'conservative':
            if abs(weighted_signal) > 0.003:
                return 1 if weighted_signal > 0 else 2
            return 0
        
        elif strategy == 'aggressive':
            if abs(weighted_signal) > 0.001:
                return 1 if weighted_signal > 0 else 2
            return 0
        
        elif strategy == 'momentum':
            if features.price_momentum > 0.0005:
                return 1
            elif features.price_momentum < -0.0005:
                return 2
            return 0
        
        elif strategy == 'mean_reversion':
            if features.price_position > 0.55:
                return 2  # Sell at highs
            elif features.price_position < 0.45:
                return 1  # Buy at lows
            return 0
        
        else:  # adaptive
            if adaptation_decision.get('emergency_mode', False):
                return 0  # Hold in emergency
            elif features.regime_confidence > 0.7:
                return 1 if weighted_signal > 0.002 else (2 if weighted_signal < -0.002 else 0)
            else:
                return 1 if weighted_signal > 0.003 else (2 if weighted_signal < -0.003 else 0)
    
    def _create_hold_decision(self, 
                             confidence: float,
                             context: DecisionContext) -> Decision:
        """Create a hold decision with proper context"""
        
        logger.info(f"CONFIDENCE TRACE: creating_hold_decision={confidence:.6f}")
        
        # Create intelligence_data for hold decisions to prevent AttributeError in risk manager
        intelligence_data = {
            'subsystem_signals': context.subsystem_signals.detach().cpu().numpy().tolist() if hasattr(context.subsystem_signals, 'detach') else [],
            'subsystem_weights': context.subsystem_weights.detach().cpu().numpy().tolist() if hasattr(context.subsystem_weights, 'detach') else [],
            'weighted_signal': 0.0,  # Hold decision has neutral signal
            'volatility': context.features.volatility,
            'price_momentum': context.features.price_momentum,
            'volume_momentum': getattr(context.features, 'volume_momentum', 0.0),
            'regime_confidence': context.features.regime_confidence,
            'consensus_strength': 0.5,  # Neutral consensus for hold
            'regime_context': {
                'regime_confidence': context.features.regime_confidence,
                'volatility': context.features.volatility,
                'trend_strength': abs(context.features.price_momentum)
            }
        }
        
        decision = Decision(
            'hold', confidence, 0,
            adaptation_strategy='conservative',
            uncertainty_estimate=0.5,
            few_shot_prediction=0.0,
            intelligence_data=intelligence_data,  # Add intelligence_data to prevent AttributeError
            regime_awareness={
                'regime_confidence': context.features.regime_confidence,
                'volatility': context.features.volatility,
                'trend_strength': abs(context.features.price_momentum)
            }
        )
        
        logger.info(f"CONFIDENCE TRACE: hold_decision_created={decision.confidence:.6f}")
        return decision
    
    def _create_decision(self,
                        action_idx: int,
                        confidence: float,
                        position_size: float,
                        risk_params: Dict[str, float],
                        context: DecisionContext,
                        adaptation_decision: Dict[str, Any],
                        exploration: bool) -> Decision:
        """Create comprehensive trading decision"""
        
        actions = ['hold', 'buy', 'sell']
        action = actions[action_idx]
        
        if action == 'hold':
            return self._create_hold_decision(confidence, context)
        
        # Calculate stop and target prices
        stop_price, target_price = self._calculate_enhanced_levels(
            action, context.market_data, risk_params, context.features
        )
        
        # Determine primary tool
        primary_tool = self._get_enhanced_primary_tool(
            context.subsystem_signals, context.subsystem_weights, context.features
        )
        
        # Store intelligence data
        intelligence_data = {
            'subsystem_signals': context.subsystem_signals.detach().cpu().numpy().tolist(),
            'subsystem_weights': context.subsystem_weights.detach().cpu().numpy().tolist(),
            'weighted_signal': float(torch.sum(
                context.subsystem_signals * context.subsystem_weights[:len(context.subsystem_signals)]
            )),
            'adaptation_decision': adaptation_decision,
            'regime_context': {
                'regime_confidence': context.features.regime_confidence,
                'volatility': context.features.volatility,
                'trend_strength': abs(context.features.price_momentum)
            }
        }
        
        # Update last trade time
        self.last_trade_time = context.market_data.timestamp
        
        # Extract and validate state_features before creating Decision
        extracted_state_features = self._safe_extract_state_features(context.learned_state)
        logger.debug(f"STATE_FEATURES_TRACE: Extracted {len(extracted_state_features) if extracted_state_features else 'None'} features from learned_state")
        
        return Decision(
            action=action,
            confidence=confidence,
            size=position_size,
            stop_price=stop_price,
            target_price=target_price,
            primary_tool=primary_tool,
            exploration=exploration,
            intelligence_data=intelligence_data,
            state_features=extracted_state_features,
            adaptation_strategy=adaptation_decision.get('strategy_name', 'conservative'),
            uncertainty_estimate=adaptation_decision.get('uncertainty', 0.5),
            few_shot_prediction=0.0,  # Will be set by caller if available
            regime_awareness=intelligence_data['regime_context']
        )
    
    def _calculate_enhanced_levels(self,
                                 action: str,
                                 market_data: MarketData,
                                 risk_params: Dict[str, float],
                                 features: Features) -> Tuple[float, float]:
        """Calculate intelligent stop and target levels"""
        
        stop_price = 0.0
        target_price = 0.0
        
        if risk_params['use_stop']:
            stop_price = self._calculate_intelligent_stop(
                action, market_data, features, risk_params['stop_distance']
            )
        
        if risk_params['use_target']:
            target_price = self._calculate_intelligent_target(
                action, market_data, features, risk_params['target_distance']
            )
        
        return stop_price, target_price
    
    def _calculate_intelligent_stop(self,
                                  action: str,
                                  market_data: MarketData,
                                  features: Features,
                                  stop_distance: float) -> float:
        """Calculate intelligent stop placement"""
        
        base_stop_distance = self.meta_learner.get_parameter('stop_distance_factor')
        
        # Volatility-based adjustment
        vol_adjustment = 1.0 + (features.volatility * 10)
        
        # Regime-based adjustment
        regime_adjustment = 1.0
        if features.regime_confidence < 0.4:
            regime_adjustment = 0.7  # Tighter stops in uncertain regimes
        elif features.volatility > 0.04:
            regime_adjustment = 1.5  # Wider stops in high volatility
        
        # Time-based adjustment
        time_adjustment = 1.0
        if 0.35 < features.time_of_day < 0.65:  # Market open hours
            time_adjustment = 1.2
        
        # Combined adjustment
        final_distance = base_stop_distance * (1 + stop_distance) * vol_adjustment * regime_adjustment * time_adjustment
        final_distance = min(0.05, max(0.005, final_distance))
        
        if action == 'buy':
            return market_data.price * (1 - final_distance)
        else:
            return market_data.price * (1 + final_distance)
    
    def _calculate_intelligent_target(self,
                                    action: str,
                                    market_data: MarketData,
                                    features: Features,
                                    target_distance: float) -> float:
        """Calculate intelligent target placement"""
        
        base_target_distance = self.meta_learner.get_parameter('target_distance_factor')
        
        # Trend strength adjustment
        trend_adjustment = 1.0 + abs(features.price_momentum) * 5
        
        # Confidence adjustment
        confidence_adjustment = 0.5 + features.confidence
        
        # Pattern strength adjustment
        pattern_adjustment = 1.0 + features.pattern_score * 0.5
        
        # Combined adjustment
        final_distance = base_target_distance * (1 + target_distance) * trend_adjustment * confidence_adjustment * pattern_adjustment
        final_distance = min(0.15, max(0.01, final_distance))
        
        if action == 'buy':
            return market_data.price * (1 + final_distance)
        else:
            return market_data.price * (1 - final_distance)
    
    def _get_enhanced_primary_tool(self,
                                 signals: torch.Tensor,
                                 weights: torch.Tensor,
                                 features: Features) -> str:
        """Enhanced primary tool identification"""
        
        weighted_signals = torch.abs(signals * weights[:len(signals)])
        tool_names = ['dna', 'temporal', 'immune', 'microstructure', 'dopamine', 'regime']
        
        if torch.sum(weighted_signals) == 0:
            return 'basic'
        
        primary_idx = torch.argmax(weighted_signals)
        primary_tool = tool_names[min(primary_idx, len(tool_names) - 1)]
        
        # Add context about conditions
        if features.regime_confidence < 0.5:
            primary_tool += '_uncertain'
        elif features.volatility > 0.05:
            primary_tool += '_highvol'
        elif abs(features.smart_money_flow) > 0.3:
            primary_tool += '_smartmoney'
        
        return primary_tool
    
    def _build_meta_context(self, 
                           features: Features, 
                           adaptation_decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build comprehensive meta-context for decision making and learning.
        
        This method extracts and processes regime, microstructure, and adaptation
        information to create a rich context that supports intelligent decision
        making and meta-learning. Following clean architecture principles,
        it separates context building from decision logic.
        
        Args:
            features: Processed market features containing regime and technical data
            adaptation_decision: Real-time adaptation context from meta-learner
            
        Returns:
            Dict[str, Any]: Comprehensive meta-context with normalized values
                regime_confidence: Market regime certainty (0.0 to 1.0)
                microstructure_strength: Order book pattern strength
                adaptation_quality: Quality of recent adaptations
                volatility_regime: Normalized volatility classification
                liquidity_regime: Market liquidity assessment
                smart_money_activity: Institutional flow detection
                market_phase: Current market cycle phase
                uncertainty_estimate: Overall uncertainty level
                risk_regime: Risk environment classification
                
        Raises:
            AttributeError: If required features are missing
            ValueError: If adaptation_decision contains invalid data
        """
        try:
            # Extract regime and microstructure information
            # Following the Interface Segregation Principle - only extract what we need
            regime_info = self._extract_regime_info(features)
            microstructure_info = self._extract_microstructure_info(features)
            adaptation_metrics = self._process_adaptation_metrics(adaptation_decision)
            
            # Build comprehensive meta-context
            meta_context = {
                # Regime Analysis
                'regime_confidence': regime_info['confidence'],
                'regime_classification': regime_info['classification'],
                'regime_stability': regime_info['stability'],
                
                # Microstructure Analysis  
                'microstructure_strength': microstructure_info['strength'],
                'liquidity_regime': microstructure_info['liquidity'],
                'smart_money_activity': microstructure_info['smart_money'],
                'order_flow_imbalance': microstructure_info['flow_imbalance'],
                
                # Adaptation Metrics
                'adaptation_quality': adaptation_metrics['quality'],
                'uncertainty_estimate': adaptation_metrics['uncertainty'],
                'learning_confidence': adaptation_metrics['learning_confidence'],
                'strategy_effectiveness': adaptation_metrics['strategy_effectiveness'],
                
                # Volatility and Risk Classification
                'volatility_regime': self._classify_volatility_regime(features.volatility),
                'risk_regime': self._classify_risk_regime(features, adaptation_decision),
                'market_phase': self._determine_market_phase(features),
                
                # Additional Context for Learning
                'feature_quality': self._assess_feature_quality(features),
                'signal_reliability': self._assess_signal_reliability(features),
                'market_efficiency': self._estimate_market_efficiency(features)
            }
            
            # Validate meta-context completeness and bounds
            meta_context = self._validate_and_normalize_context(meta_context)
            
            logger.debug(f"Meta-context built: regime_conf={meta_context['regime_confidence']:.3f}, "
                        f"micro_strength={meta_context['microstructure_strength']:.3f}, "
                        f"adapt_quality={meta_context['adaptation_quality']:.3f}")
            
            return meta_context
            
        except Exception as e:
            logger.error(f"Error building meta-context: {e}")
            # Return safe fallback context
            return self._get_fallback_meta_context()
    
    def _extract_regime_info(self, features: Features) -> Dict[str, float]:
        """Extract and process regime-related information"""
        try:
            # Regime confidence with bounds checking
            regime_confidence = getattr(features, 'regime_confidence', 0.5)
            regime_confidence = max(0.0, min(1.0, regime_confidence))
            
            # Estimate regime stability from momentum consistency
            price_momentum = getattr(features, 'price_momentum', 0.0)
            volume_momentum = getattr(features, 'volume_momentum', 0.0)
            
            # Regime stability based on momentum alignment
            momentum_alignment = 1.0 - abs(price_momentum - volume_momentum) / 2.0
            regime_stability = max(0.0, min(1.0, momentum_alignment))
            
            # Classify regime type based on market characteristics
            volatility = getattr(features, 'volatility', 0.02)
            if volatility < 0.01:
                regime_classification = 'low_vol'
            elif volatility < 0.03:
                regime_classification = 'normal_vol'
            elif volatility < 0.05:
                regime_classification = 'high_vol'
            else:
                regime_classification = 'extreme_vol'
            
            return {
                'confidence': regime_confidence,
                'stability': regime_stability,
                'classification': regime_classification
            }
            
        except Exception as e:
            logger.warning(f"Error extracting regime info: {e}")
            return {
                'confidence': 0.5,
                'stability': 0.5, 
                'classification': 'unknown'
            }
    
    def _extract_microstructure_info(self, features: Features) -> Dict[str, float]:
        """Extract and process microstructure-related information"""
        try:
            # Microstructure signal strength
            microstructure_signal = getattr(features, 'microstructure_signal', 0.0)
            microstructure_strength = abs(microstructure_signal)
            
            # Liquidity assessment from depth and spread indicators
            liquidity_depth = getattr(features, 'liquidity_depth', 0.5)
            bid_ask_spread = getattr(features, 'bid_ask_spread', 0.001)
            
            # Liquidity regime: higher depth and lower spread = better liquidity
            liquidity_score = liquidity_depth * (1.0 / (1.0 + bid_ask_spread * 1000))
            liquidity_regime = max(0.0, min(1.0, liquidity_score))
            
            # Smart money activity detection
            smart_money_flow = getattr(features, 'smart_money_flow', 0.0)
            smart_money_activity = abs(smart_money_flow)
            
            # Order flow imbalance estimation
            volume_momentum = getattr(features, 'volume_momentum', 0.0)
            price_momentum = getattr(features, 'price_momentum', 0.0)
            
            # Flow imbalance when volume and price momentum diverge
            flow_imbalance = abs(volume_momentum - price_momentum) / 2.0
            
            return {
                'strength': microstructure_strength,
                'liquidity': liquidity_regime,
                'smart_money': smart_money_activity,
                'flow_imbalance': flow_imbalance
            }
            
        except Exception as e:
            logger.warning(f"Error extracting microstructure info: {e}")
            return {
                'strength': 0.0,
                'liquidity': 0.5,
                'smart_money': 0.0,
                'flow_imbalance': 0.0
            }
    
    def _process_adaptation_metrics(self, adaptation_decision: Dict[str, Any]) -> Dict[str, float]:
        """Process adaptation decision metrics into normalized values"""
        try:
            # Extract adaptation quality metrics
            adaptation_quality = adaptation_decision.get('adaptation_quality', 0.5)
            adaptation_quality = max(0.0, min(1.0, adaptation_quality))
            
            # Extract uncertainty estimate
            uncertainty_estimate = adaptation_decision.get('uncertainty', 0.5)
            uncertainty_estimate = max(0.0, min(1.0, uncertainty_estimate))
            
            # Learning confidence from recent performance
            learning_confidence = adaptation_decision.get('learning_confidence', 0.5)
            learning_confidence = max(0.0, min(1.0, learning_confidence))
            
            # Strategy effectiveness assessment
            strategy_name = adaptation_decision.get('strategy_name', 'conservative')
            recent_performance = adaptation_decision.get('recent_performance', 0.0)
            
            # Convert performance to effectiveness score [0, 1]
            if recent_performance >= 0:
                strategy_effectiveness = min(1.0, recent_performance * 2.0 + 0.5)
            else:
                strategy_effectiveness = max(0.0, 0.5 + recent_performance * 2.0)
            
            return {
                'quality': adaptation_quality,
                'uncertainty': uncertainty_estimate,
                'learning_confidence': learning_confidence,
                'strategy_effectiveness': strategy_effectiveness
            }
            
        except Exception as e:
            logger.warning(f"Error processing adaptation metrics: {e}")
            return {
                'quality': 0.5,
                'uncertainty': 0.5,
                'learning_confidence': 0.5,
                'strategy_effectiveness': 0.5
            }
    
    def _classify_volatility_regime(self, volatility: float) -> str:
        """Classify volatility into regime categories"""
        if volatility < 0.005:
            return 'ultra_low'
        elif volatility < 0.015:
            return 'low'
        elif volatility < 0.025:
            return 'normal'
        elif volatility < 0.04:
            return 'elevated'
        elif volatility < 0.06:
            return 'high'
        else:
            return 'extreme'
    
    def _classify_risk_regime(self, features: Features, adaptation_decision: Dict[str, Any]) -> str:
        """Classify overall risk regime based on multiple factors"""
        try:
            volatility = getattr(features, 'volatility', 0.02)
            regime_confidence = getattr(features, 'regime_confidence', 0.5)
            uncertainty = adaptation_decision.get('uncertainty', 0.5)
            
            # Risk score combines volatility, regime uncertainty, and adaptation uncertainty
            risk_score = (volatility * 20 + (1.0 - regime_confidence) + uncertainty) / 3.0
            
            if risk_score < 0.3:
                return 'low_risk'
            elif risk_score < 0.5:
                return 'moderate_risk'
            elif risk_score < 0.7:
                return 'elevated_risk'
            else:
                return 'high_risk'
                
        except Exception:
            return 'unknown_risk'
    
    def _determine_market_phase(self, features: Features) -> str:
        """Determine current market cycle phase"""
        try:
            price_momentum = getattr(features, 'price_momentum', 0.0)
            volatility = getattr(features, 'volatility', 0.02)
            volume_momentum = getattr(features, 'volume_momentum', 0.0)
            
            # Phase determination based on momentum and volatility patterns
            if price_momentum > 0.002 and volume_momentum > 0.0:
                return 'uptrend'
            elif price_momentum < -0.002 and volume_momentum > 0.0:
                return 'downtrend'
            elif volatility < 0.01 and abs(price_momentum) < 0.001:
                return 'consolidation'
            elif volatility > 0.04:
                return 'volatile_range'
            else:
                return 'sideways'
                
        except Exception:
            return 'unknown'
    
    def _assess_feature_quality(self, features: Features) -> float:
        """Assess overall quality of input features"""
        try:
            # Check for completeness and validity of key features
            key_features = ['volatility', 'price_momentum', 'volume_momentum', 
                          'regime_confidence', 'liquidity_depth']
            
            quality_score = 0.0
            for feature_name in key_features:
                feature_value = getattr(features, feature_name, None)
                if feature_value is not None and not (np.isnan(feature_value) or np.isinf(feature_value)):
                    quality_score += 0.2  # Each feature contributes 20%
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception:
            return 0.5
    
    def _assess_signal_reliability(self, features: Features) -> float:
        """Assess reliability of trading signals"""
        try:
            # Signal reliability based on consistency and strength
            overall_signal = getattr(features, 'overall_signal', 0.0)
            confidence = getattr(features, 'confidence', 0.5)
            regime_confidence = getattr(features, 'regime_confidence', 0.5)
            
            # Combine signal strength with confidence measures
            signal_strength = abs(overall_signal)
            reliability_score = (signal_strength + confidence + regime_confidence) / 3.0
            
            return max(0.0, min(1.0, reliability_score))
            
        except Exception:
            return 0.5
    
    def _estimate_market_efficiency(self, features: Features) -> float:
        """Estimate current market efficiency level"""
        try:
            # Market efficiency indicators: low spread, high liquidity, regime stability
            bid_ask_spread = getattr(features, 'bid_ask_spread', 0.001)
            liquidity_depth = getattr(features, 'liquidity_depth', 0.5)
            regime_confidence = getattr(features, 'regime_confidence', 0.5)
            
            # Efficiency score: high liquidity, low spread, stable regime
            spread_efficiency = 1.0 / (1.0 + bid_ask_spread * 1000)
            efficiency_score = (spread_efficiency + liquidity_depth + regime_confidence) / 3.0
            
            return max(0.0, min(1.0, efficiency_score))
            
        except Exception:
            return 0.5
    
    def _validate_and_normalize_context(self, meta_context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize meta-context values"""
        try:
            normalized_context = {}
            
            for key, value in meta_context.items():
                if isinstance(value, (int, float)):
                    # Ensure numeric values are within [0, 1] range where applicable
                    if key.endswith(('_confidence', '_quality', '_regime', '_activity', 
                                   '_strength', '_reliability', '_efficiency')):
                        normalized_context[key] = max(0.0, min(1.0, float(value)))
                    else:
                        normalized_context[key] = float(value)
                else:
                    # Keep non-numeric values as-is
                    normalized_context[key] = value
            
            return normalized_context
            
        except Exception as e:
            logger.warning(f"Error validating meta-context: {e}")
            return meta_context
    
    def _get_fallback_meta_context(self) -> Dict[str, Any]:
        """
        Provide intelligent fallback meta-context with realistic regime classifications.
        
        Instead of defaulting to 'unknown', we provide realistic market regime defaults
        based on typical market conditions. This ensures the learning system continues
        to function even when regime detection temporarily fails.
        """
        return {
            'regime_confidence': 0.4,  # Lower confidence for fallback
            'regime_classification': 'ranging',  # Most common market state
            'regime_stability': 0.5,
            'microstructure_strength': 0.0,
            'liquidity_regime': 0.5,
            'smart_money_activity': 0.0,
            'order_flow_imbalance': 0.0,
            'adaptation_quality': 0.4,  # Lower quality for fallback
            'uncertainty_estimate': 0.6,  # Higher uncertainty for fallback
            'learning_confidence': 0.4,
            'strategy_effectiveness': 0.5,
            'volatility_regime': 'normal',  # Most common volatility state
            'risk_regime': 'moderate_risk',
            'market_phase': 'ranging',  # Most common phase
            'feature_quality': 0.4,  # Lower quality indicates fallback
            'signal_reliability': 0.4,
            'market_efficiency': 0.5,
            # Additional regime context for better adaptation
            'trend_regime': 'sideways',
            'momentum_regime': 'neutral',
            'volume_regime': 'normal'
        }
    
    def _safe_extract_state_features(self, learned_state: torch.Tensor) -> list:
        """
        Safely extract state features from learned state tensor with defensive programming.
        
        Args:
            learned_state: PyTorch tensor containing learned state representation
            
        Returns:
            List of floats representing state features, or empty list if extraction fails
        """
        try:
            if learned_state is None:
                logger.warning("learned_state is None - creating default state features")
                return [0.0] * 64  # Default feature vector size
            
            if not isinstance(learned_state, torch.Tensor):
                logger.warning(f"learned_state is not a tensor (type: {type(learned_state)}) - creating default")
                return [0.0] * 64
            
            # Handle empty tensors
            if learned_state.numel() == 0:
                logger.warning("learned_state tensor is empty - creating default state features")
                return [0.0] * 64
            
            # Safe tensor conversion with proper handling
            try:
                # Ensure tensor is on CPU and detached
                cpu_tensor = learned_state.detach().cpu()
                
                # Handle different tensor shapes
                if cpu_tensor.dim() == 0:  # Scalar
                    state_features = [float(cpu_tensor.item())]
                elif cpu_tensor.dim() == 1:  # 1D tensor
                    state_features = cpu_tensor.numpy().tolist()
                else:  # Multi-dimensional - flatten
                    state_features = cpu_tensor.squeeze().flatten().numpy().tolist()
                
                # Validate extracted features
                if not state_features:
                    logger.warning("Extracted state_features is empty - using default")
                    return [0.0] * 64
                
                # Check for NaN or infinite values
                if any(not np.isfinite(x) for x in state_features):
                    logger.warning("state_features contains NaN/inf values - filtering")
                    state_features = [x if np.isfinite(x) else 0.0 for x in state_features]
                
                logger.debug(f"Successfully extracted {len(state_features)} state features")
                return state_features
                
            except Exception as tensor_error:
                logger.error(f"Error converting tensor to state features: {tensor_error}")
                return [0.0] * 64
                
        except Exception as e:
            logger.error(f"Critical error in state feature extraction: {e}")
            return [0.0] * 64
    
    def get_stats(self) -> Dict[str, Any]:
        """Get decision engine statistics"""
        return {
            'total_decisions': self.total_decisions,
            'last_trade_time': self.last_trade_time,
            'consecutive_hold_decisions': self.consecutive_hold_decisions,
            'last_emergency_override_time': self.last_emergency_override_time,
            'last_successful_trade_time': self.last_successful_trade_time,
            'emergency_trading_floor': self.emergency_trading_floor
        }
    
    def record_successful_trade(self, timestamp: float):
        """
        Record successful trade for emergency tracking.
        
        This method should be called by the trading system when a trade
        completes successfully to update emergency override timing.
        """
        self.last_successful_trade_time = timestamp
        logger.debug(f"Successful trade recorded at timestamp {timestamp}")
    
    def sync_timestamps(self, last_trade_time: float, last_successful_trade_time: Optional[float] = None):
        """
        Sync decision engine timestamps with external state (e.g., portfolio).
        
        Enhanced version that handles cold-start scenarios and validates timestamps
        before synchronization to prevent emergency override issues.
        
        This method fixes the issue where decision engine starts with intelligent
        default timestamps while portfolio/system may have restored different
        timestamps from saved state.
        
        Args:
            last_trade_time: Last trade timestamp from portfolio or state manager
            last_successful_trade_time: Last successful trade timestamp (optional)
        """
        current_time = time.time()
        
        # Validate input timestamps
        if last_trade_time < 0 or (last_successful_trade_time is not None and last_successful_trade_time < 0):
            logger.warning(f"Invalid negative timestamps provided for sync: "
                          f"last_trade_time={last_trade_time}, last_successful_trade_time={last_successful_trade_time}")
            return
        
        # Check if timestamps are reasonable (not too far in future)
        max_future_time = current_time + 3600  # Allow 1 hour in future
        if last_trade_time > max_future_time:
            logger.warning(f"last_trade_time {last_trade_time} ({time.ctime(last_trade_time)}) "
                          f"is too far in future, ignoring sync")
            return
        
        # Sync last_trade_time if valid and more recent than current
        if last_trade_time > 0:
            # Only sync if the external timestamp is more recent than our intelligent default
            # or if we're still using the exact intelligent default
            if (last_trade_time > self.last_trade_time or 
                abs(self.last_trade_time - self._get_intelligent_default_timestamp()) < 60):
                
                old_time = self.last_trade_time
                self.last_trade_time = last_trade_time
                logger.info(f"Decision engine last_trade_time synced: "
                           f"{time.ctime(old_time)} -> {time.ctime(last_trade_time)}")
                
                # Exit cold-start mode if we receive valid external timestamps
                if self.is_cold_start:
                    logger.info("Exiting cold-start mode due to timestamp sync from external state")
                    self.is_cold_start = False
            else:
                logger.debug(f"External last_trade_time {time.ctime(last_trade_time)} is older than "
                           f"current {time.ctime(self.last_trade_time)}, keeping current")
        
        # Sync last_successful_trade_time if provided and valid
        if last_successful_trade_time is not None and last_successful_trade_time > 0:
            if (last_successful_trade_time > self.last_successful_trade_time or
                abs(self.last_successful_trade_time - self._get_intelligent_default_timestamp()) < 60):
                
                old_time = self.last_successful_trade_time
                self.last_successful_trade_time = last_successful_trade_time
                logger.info(f"Decision engine last_successful_trade_time synced: "
                           f"{time.ctime(old_time)} -> {time.ctime(last_successful_trade_time)}")
        
        # Fallback logic: If no successful trade time provided, use last_trade_time
        elif (last_trade_time > 0 and 
              abs(self.last_successful_trade_time - self._get_intelligent_default_timestamp()) < 60):
            self.last_successful_trade_time = last_trade_time
            logger.info(f"Decision engine last_successful_trade_time set to last_trade_time fallback: "
                       f"{time.ctime(last_trade_time)}")
        
        # Log final state for debugging
        logger.debug(f"Timestamp sync complete - last_trade: {time.ctime(self.last_trade_time)}, "
                    f"last_successful: {time.ctime(self.last_successful_trade_time)}, "
                    f"cold_start: {self.is_cold_start}")