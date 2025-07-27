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
        self.last_trade_time = 0.0
        
        # Recent rejection tracking (for constraints)
        self.recent_position_rejections = 0
        
        logger.info("TradingDecisionEngine initialized")
    
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
            
            # Check trading constraints
            if not self._should_consider_trading(context, confidence):
                return self._create_hold_decision(confidence, context)
            
            # Determine action (exploration vs exploitation)
            action_idx, exploration = self._select_action(
                context, network_outputs, adaptation_decision
            )
            
            # Apply confidence and intelligence thresholds
            if self._should_block_trade(confidence, features, exploration):
                action_idx = 0  # Force hold
            
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
    
    def _build_decision_context(self,
                               features: Features,
                               market_data: MarketData, 
                               dopamine_anticipation: Any,
                               learned_state: torch.Tensor,
                               adaptation_decision: Dict[str, Any]) -> DecisionContext:
        """Build comprehensive decision context"""
        
        # Enhanced subsystem contributions with dopamine influence
        base_dopamine_signal = float(features.dopamine_signal)
        dopamine_amplified_signal = base_dopamine_signal * dopamine_anticipation.position_size_modifier
        
        subsystem_signals = torch.tensor([
            float(features.dna_signal),
            float(features.temporal_signal),
            float(features.immune_signal),
            float(features.microstructure_signal),
            dopamine_amplified_signal,
            float(features.regime_adjusted_signal)
        ], dtype=torch.float64, device=self.device)
        
        subsystem_weights = self.meta_learner.get_subsystem_weights()
        
        # Expand weights to match signals if needed
        if len(subsystem_weights) < len(subsystem_signals):
            additional_weights = torch.ones(
                len(subsystem_signals) - len(subsystem_weights), 
                dtype=torch.float64, device=self.device
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
            'dopamine_state': dopamine_anticipation.state.value,
            'dopamine_urgency': dopamine_anticipation.urgency_factor
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
            ], dtype=torch.float64, device=self.device)
            
            # Weight subsystems based on their confidence in current market conditions
            subsystem_weights = torch.tensor([
                tool_consultations['dna_confidence'],
                tool_consultations['temporal_confidence'],
                tool_consultations['immune_confidence'], 
                tool_consultations['microstructure_confidence'],
                tool_consultations['dopamine_confidence'],
                0.5  # Default regime weight
            ], dtype=torch.float64, device=self.device)
            
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
        
        # Basic constraints from meta-learner
        loss_tolerance = self.meta_learner.get_parameter('loss_tolerance_factor')
        max_loss = context.market_data.account_balance * loss_tolerance
        
        if context.market_data.daily_pnl <= -max_loss:
            logger.info(f"Trading blocked: Daily loss limit reached")
            return False
        
        # Frequency constraints
        frequency_limit = self.meta_learner.get_parameter('trade_frequency_base')
        time_since_last = context.market_data.timestamp - self.last_trade_time
        
        if time_since_last < (1 / frequency_limit):
            logger.info(f"Trading blocked: Frequency limit")
            return False
        
        # Intelligence signal strength check
        intelligence_threshold = self.meta_learner.get_parameter('intelligence_threshold', 0.3)
        intelligence_signal_strength = abs(context.features.overall_signal)
        
        if intelligence_signal_strength < intelligence_threshold:
            logger.info(f"Trading blocked: Intelligence signal {intelligence_signal_strength:.3f} "
                       f"below threshold {intelligence_threshold:.3f}")
            return False
        
        return True
    
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
        
        # Apply dopamine psychological modifiers
        dopamine_position_modifier = dopamine_anticipation.position_size_modifier
        dopamine_risk_modifier = dopamine_anticipation.risk_tolerance_modifier
        
        # Adjust position size based on dopamine state
        position_size = base_position_size * dopamine_position_modifier
        
        # Uncertainty-adjusted position sizing
        uncertainty_factor = 1.0 - context.meta_context.get('uncertainty', 0.5)
        position_size *= max(0.1, uncertainty_factor)
        
        # Apply dopamine urgency factor
        urgency_adjustment = 0.8 + (dopamine_anticipation.urgency_factor * 0.4)  # 0.8 to 1.2 range
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
        
        decision = Decision(
            'hold', confidence, 0,
            adaptation_strategy='conservative',
            uncertainty_estimate=0.5,
            few_shot_prediction=0.0,
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
        
        return Decision(
            action=action,
            confidence=confidence,
            size=position_size,
            stop_price=stop_price,
            target_price=target_price,
            primary_tool=primary_tool,
            exploration=exploration,
            intelligence_data=intelligence_data,
            state_features=context.learned_state.squeeze().detach().cpu().numpy().tolist(),
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
    
    def get_stats(self) -> Dict[str, Any]:
        """Get decision engine statistics"""
        return {
            'total_decisions': self.total_decisions,
            'last_trade_time': self.last_trade_time,
            'recent_position_rejections': self.recent_position_rejections
        }