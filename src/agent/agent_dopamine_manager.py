"""
Agent Dopamine Manager - Integrates dopamine processing within the trading agent

This module moves dopamine functionality from being a separate intelligence subsystem
to being an integral part of the trading agent's decision-making process. It processes
intelligence signals from the 4 subsystems and applies dopamine-based psychology
to enhance trading decisions.

Key design principles:
- Single Responsibility: Handles only dopamine-related processing within the agent
- Dependency Inversion: Depends on intelligence interfaces, not concrete implementations
- Clean Integration: Seamlessly integrates with existing agent architecture
- Psychological Realism: Maintains all existing dopamine functionality
"""

import numpy as np
import time
import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any

from src.shared.intelligence_types import (
    IntelligenceSignal, IntelligenceUpdate, IntelligenceContext,
    IntelligenceConsumer, IntelligenceSignalType
)

# Import dopamine types from the existing subsystem for compatibility
from src.intelligence.subsystems.enhanced_dopamine_subsystem import (
    DopaminePhase, DopamineState, DopamineResponse, DopamineSnapshot
)

logger = logging.getLogger(__name__)


@dataclass
class TradingDecisionContext:
    """Context for a trading decision including all relevant factors"""
    action: str                           # 'buy', 'sell', 'hold'
    confidence: float                     # Decision confidence [0.0, 1.0]
    position_size: float                  # Intended position size
    expected_outcome: float               # Expected P&L
    market_conditions: Dict[str, Any]     # Current market state
    intelligence_signals: List[IntelligenceSignal]  # Contributing signals
    risk_factors: Dict[str, float]        # Risk assessment factors
    primary_tool: str = 'unknown'        # Tool that generated the decision


@dataclass
class DopamineIntegratedDecision:
    """Trading decision enhanced with dopamine psychology"""
    base_decision: TradingDecisionContext
    dopamine_response: DopamineResponse
    psychological_adjustments: Dict[str, float]  # Adjustments made due to psychology
    final_action: str                     # Final action after dopamine integration
    final_confidence: float               # Final confidence after adjustments
    final_position_size: float            # Final position size after adjustments
    integration_metadata: Dict[str, Any]  # Additional integration information


class AgentDopamineManager(IntelligenceConsumer):
    """
    Dopamine processing manager integrated within the trading agent
    
    This class processes intelligence signals and applies dopamine-based trading
    psychology to enhance decision making. It maintains all the sophisticated
    dopamine modeling from the original subsystem while integrating cleanly
    with the agent architecture.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the agent dopamine manager
        
        Args:
            config: Configuration parameters for dopamine processing
        """
        self.config = config
        
        # Validate required configuration parameters
        required_params = ['dna_weight', 'temporal_weight', 'immune_weight', 'microstructure_weight']
        for param in required_params:
            if param not in config:
                raise ValueError(f"Missing required dopamine config parameter: {param}")
        
        # Validate weight values sum to reasonable total
        weight_sum = sum(config.get(param, 0.0) for param in required_params)
        if not (0.8 <= weight_sum <= 1.2):  # Allow some flexibility
            logger.warning(f"Intelligence weights sum to {weight_sum:.3f}, expected ~1.0")
        
        # Validate other critical parameters
        if config.get('confidence_threshold', 0.3) < 0.0 or config.get('confidence_threshold', 0.3) > 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        
        if config.get('max_position_size', 1.0) <= 0.0:
            raise ValueError("max_position_size must be positive")
        
        # Import and initialize the core dopamine system
        from src.intelligence.subsystems.enhanced_dopamine_subsystem import (
            ConsolidatedDopamineSubsystem
        )
        self.dopamine_core = ConsolidatedDopamineSubsystem(config)
        
        # Intelligence signal processing
        self.signal_history = deque(maxlen=100)
        self.signal_weights = {
            IntelligenceSignalType.PATTERN_RECOGNITION: config.get('dna_weight', 0.25),
            IntelligenceSignalType.TEMPORAL_ANALYSIS: config.get('temporal_weight', 0.25),
            IntelligenceSignalType.ANOMALY_DETECTION: config.get('immune_weight', 0.25),
            IntelligenceSignalType.MICROSTRUCTURE: config.get('microstructure_weight', 0.25),
            IntelligenceSignalType.REGIME_ANALYSIS: config.get('regime_weight', 0.1)
        }
        
        # Decision integration parameters
        self.position_size_bounds = (
            config.get('min_position_size', 0.01),
            config.get('max_position_size', 1.0)
        )
        self.confidence_threshold = config.get('confidence_threshold', 0.3)
        self.intelligence_integration_factor = config.get('intelligence_integration_factor', 0.7)
        
        # Performance tracking
        self.integration_stats = {
            'decisions_processed': 0,
            'psychology_adjustments_made': 0,
            'signals_processed': 0,
            'phase_transitions': 0,
            'last_integration_time': 0.0
        }
        
        # Current state
        self.current_intelligence_update: Optional[IntelligenceUpdate] = None
        self.last_decision_context: Optional[TradingDecisionContext] = None
        
        logger.info("AgentDopamineManager initialized with intelligence integration")
    
    def process_intelligence_update(self, update: IntelligenceUpdate) -> Dict[str, Any]:
        """
        Process intelligence update and prepare for decision integration
        
        Args:
            update: Intelligence update from the 4 subsystems
            
        Returns:
            Processing results and context for decision making
        """
        try:
            self.current_intelligence_update = update
            self.signal_history.append(update)
            self.integration_stats['signals_processed'] += len(update.signals)
            
            # Calculate weighted intelligence signal for dopamine processing
            intelligence_strength = self._calculate_intelligence_strength(update.signals)
            intelligence_confidence = update.average_confidence
            
            # Analyze signal consensus and market context
            consensus_analysis = self._analyze_signal_consensus(update)
            market_psychology = self._assess_market_psychology(update.context)
            
            # Prepare context for dopamine decision integration
            processing_results = {
                'intelligence_strength': intelligence_strength,
                'intelligence_confidence': intelligence_confidence,
                'signal_consensus': update.signal_consensus,
                'consensus_analysis': consensus_analysis,
                'market_psychology': market_psychology,
                'signal_count': update.signal_count,
                'primary_signal_type': update.primary_signal.signal_type.value if update.primary_signal else None,
                'processing_timestamp': time.time()
            }
            
            logger.debug(f"Intelligence processed: strength={intelligence_strength:.3f}, "
                        f"confidence={intelligence_confidence:.3f}, "
                        f"consensus={update.signal_consensus:.3f}")
            
            return processing_results
            
        except Exception as e:
            logger.error(f"Error processing intelligence update: {e}")
            return {'error': str(e), 'processing_timestamp': time.time()}
    
    def handle_signal_consensus(self, signals: List[IntelligenceSignal]) -> float:
        """
        Calculate consensus between multiple intelligence signals
        
        Args:
            signals: List of intelligence signals
            
        Returns:
            Consensus strength [-1.0, 1.0]
        """
        if not signals:
            return 0.0
        
        # Group signals by type and calculate weighted consensus
        type_consensus = {}
        for signal in signals:
            signal_type = signal.signal_type
            weighted_strength = signal.weighted_strength
            
            if signal_type not in type_consensus:
                type_consensus[signal_type] = []
            type_consensus[signal_type].append(weighted_strength)
        
        # Calculate overall consensus using subsystem weights
        total_weight = 0.0
        weighted_consensus = 0.0
        
        for signal_type, strengths in type_consensus.items():
            avg_strength = np.mean(strengths)
            weight = self.signal_weights.get(signal_type, 0.2)
            
            weighted_consensus += avg_strength * weight
            total_weight += weight
        
        if total_weight > 0:
            return weighted_consensus / total_weight
        else:
            return 0.0
    
    def integrate_intelligence_context(self, context: IntelligenceContext) -> Dict[str, Any]:
        """
        Integrate intelligence context into dopamine decision making
        
        Args:
            context: Intelligence context from subsystems
            
        Returns:
            Integrated context for dopamine processing
        """
        try:
            # Convert intelligence context to dopamine-compatible format
            market_regime_factor = self._map_regime_to_psychology(context.market_regime)
            volatility_factor = self._map_volatility_to_psychology(context.volatility_level)
            volume_factor = self._map_volume_to_psychology(context.volume_profile)
            
            integrated_context = {
                'market_regime_psychology': market_regime_factor,
                'volatility_psychology': volatility_factor,
                'volume_psychology': volume_factor,
                'time_of_day_factor': self._map_time_to_psychology(context.time_of_day),
                'overall_market_stress': self._calculate_market_stress(context),
                'intelligence_reliability': self._assess_intelligence_reliability(context)
            }
            
            return integrated_context
            
        except Exception as e:
            logger.error(f"Error integrating intelligence context: {e}")
            return {}
    
    def process_trading_decision(self, decision_context: TradingDecisionContext) -> DopamineIntegratedDecision:
        """
        Process a trading decision through dopamine psychology integration
        
        This is the main method that takes a base trading decision and enhances it
        with dopamine-based psychological factors derived from intelligence signals.
        
        Args:
            decision_context: Base trading decision context
            
        Returns:
            Enhanced decision with dopamine psychology integration
        """
        try:
            self.integration_stats['decisions_processed'] += 1
            self.last_decision_context = decision_context
            
            # Phase 1: Process anticipation (pre-decision)
            anticipation_response = self._process_anticipation_phase(decision_context)
            
            # Phase 2: Create dopamine market data from decision context
            dopamine_market_data = self._create_dopamine_market_data(decision_context)
            
            # Phase 3: Get intelligence-enhanced dopamine context
            intelligence_dopamine_context = self._create_intelligence_dopamine_context(decision_context)
            
            # Phase 4: Process through dopamine core with intelligence integration
            dopamine_response = self.dopamine_core.process_trading_event(
                'monitoring',  # Use monitoring phase for decision processing
                dopamine_market_data,
                intelligence_dopamine_context
            )
            
            # Phase 5: Apply psychological adjustments to the decision
            psychological_adjustments = self._calculate_psychological_adjustments(
                decision_context, dopamine_response, anticipation_response
            )
            
            # Phase 6: Create final integrated decision
            integrated_decision = self._create_integrated_decision(
                decision_context, dopamine_response, psychological_adjustments
            )
            
            # Phase 7: Update statistics and tracking
            self._update_integration_tracking(integrated_decision)
            
            logger.info(f"Decision integrated: {decision_context.action} -> {integrated_decision.final_action}, "
                       f"confidence: {decision_context.confidence:.3f} -> {integrated_decision.final_confidence:.3f}, "
                       f"dopamine: {dopamine_response.signal:.3f} ({dopamine_response.state.value})")
            
            return integrated_decision
            
        except Exception as e:
            logger.error(f"Error processing trading decision through dopamine: {e}")
            # Return safe fallback decision
            return self._create_safe_fallback_decision(decision_context)
    
    def process_trade_outcome(self, trade_outcome: Dict[str, Any]) -> DopamineResponse:
        """
        Process trade outcome through dopamine learning system
        
        Args:
            trade_outcome: Trade outcome data including P&L and context
            
        Returns:
            Dopamine response from outcome processing
        """
        try:
            # Extract trade outcome data
            pnl = trade_outcome.get('pnl', 0.0)
            duration = trade_outcome.get('duration', 0.0)
            confidence = trade_outcome.get('confidence', 0.5)
            action = trade_outcome.get('action', 'unknown')
            
            # Create realization market data
            realization_data = {
                'unrealized_pnl': 0.0,  # Trade is closed
                'daily_pnl': pnl,
                'open_positions': 0.0,
                'current_price': trade_outcome.get('exit_price', 0.0),
                'trade_duration': duration
            }
            
            # Create realization context
            realization_context = {
                'pnl': pnl,
                'expected_outcome': trade_outcome.get('expected_outcome', 0.0),
                'confidence': confidence,
                'action': action,
                'intelligence_accuracy': self._assess_intelligence_accuracy(trade_outcome)
            }
            
            # Process realization phase
            realization_response = self.dopamine_core.process_trading_event(
                'realization', realization_data, realization_context
            )
            
            # Process reflection phase
            reflection_response = self.dopamine_core.process_trading_event(
                'reflection', realization_data, realization_context
            )
            
            # Learn from the outcome
            self.dopamine_core.learn_from_outcome(pnl, trade_outcome)
            
            logger.info(f"Trade outcome processed: PnL={pnl:.2f}, "
                       f"realization_signal={realization_response.signal:.3f}, "
                       f"reflection_signal={reflection_response.signal:.3f}")
            
            return reflection_response
            
        except Exception as e:
            logger.error(f"Error processing trade outcome: {e}")
            # Return neutral response
            return DopamineResponse(
                signal=0.0, phase=DopaminePhase.REFLECTION, state=DopamineState.BALANCED,
                anticipation_factor=0.0, satisfaction_factor=0.0, tolerance_level=0.5,
                addiction_risk=0.0, withdrawal_intensity=0.0, position_size_modifier=1.0,
                risk_tolerance_modifier=1.0, urgency_factor=0.5
            )
    
    def get_current_psychological_state(self) -> Dict[str, Any]:
        """
        Get current psychological state and dopamine factors
        
        Returns:
            Comprehensive psychological state information
        """
        try:
            current_response = self.dopamine_core.get_current_response()
            dopamine_context = self.dopamine_core.get_comprehensive_context()
            
            # Add intelligence integration information
            intelligence_context = {}
            if self.current_intelligence_update:
                intelligence_context = {
                    'signal_count': self.current_intelligence_update.signal_count,
                    'signal_consensus': self.current_intelligence_update.signal_consensus,
                    'average_confidence': self.current_intelligence_update.average_confidence,
                    'strongest_signal': self.current_intelligence_update.strongest_signal_strength,
                    'primary_signal_type': (self.current_intelligence_update.primary_signal.signal_type.value 
                                          if self.current_intelligence_update.primary_signal else None)
                }
            
            return {
                'dopamine_state': current_response.state,
                'dopamine_signal': current_response.signal,
                'psychological_phase': current_response.phase,
                'tolerance_level': current_response.tolerance_level,
                'addiction_risk': current_response.addiction_risk,
                'withdrawal_intensity': current_response.withdrawal_intensity,
                'position_size_modifier': current_response.position_size_modifier,
                'risk_tolerance_modifier': current_response.risk_tolerance_modifier,
                'urgency_factor': current_response.urgency_factor,
                'intelligence_integration': intelligence_context,
                'dopamine_context': dopamine_context,
                'integration_stats': self.integration_stats.copy()
            }
            
        except Exception as e:
            logger.error(f"Error getting psychological state: {e}")
            return {'error': str(e)}
    
    # Helper methods for intelligence signal processing
    
    def _calculate_intelligence_strength(self, signals: List[IntelligenceSignal]) -> float:
        """Calculate overall intelligence strength from signals"""
        if not signals:
            return 0.0
        
        weighted_strength = 0.0
        total_weight = 0.0
        
        for signal in signals:
            weight = self.signal_weights.get(signal.signal_type, 0.2)
            weighted_strength += signal.weighted_strength * weight
            total_weight += weight
        
        return weighted_strength / total_weight if total_weight > 0 else 0.0
    
    def _analyze_signal_consensus(self, update: IntelligenceUpdate) -> Dict[str, Any]:
        """Analyze consensus between intelligence signals"""
        signals = update.signals
        if not signals:
            return {'consensus_strength': 0.0, 'agreement_level': 'none'}
        
        # Count directional agreement
        bullish_count = sum(1 for s in signals if s.direction == 'bullish')
        bearish_count = sum(1 for s in signals if s.direction == 'bearish')
        neutral_count = sum(1 for s in signals if s.direction == 'neutral')
        
        total_signals = len(signals)
        max_directional = max(bullish_count, bearish_count, neutral_count)
        agreement_ratio = max_directional / total_signals
        
        # Determine agreement level
        if agreement_ratio >= 0.8:
            agreement_level = 'strong'
        elif agreement_ratio >= 0.6:
            agreement_level = 'moderate'
        elif agreement_ratio >= 0.4:
            agreement_level = 'weak'
        else:
            agreement_level = 'conflicted'
        
        return {
            'consensus_strength': update.signal_consensus,
            'agreement_level': agreement_level,
            'agreement_ratio': agreement_ratio,
            'directional_breakdown': {
                'bullish': bullish_count,
                'bearish': bearish_count,
                'neutral': neutral_count
            }
        }
    
    def _assess_market_psychology(self, context: IntelligenceContext) -> Dict[str, float]:
        """Assess market psychology factors from intelligence context"""
        return {
            'regime_stress': self._map_regime_to_psychology(context.market_regime),
            'volatility_anxiety': min(context.volatility_level * 1.5, 1.0),
            'volume_confidence': 1.0 if context.volume_profile == 'high' else 0.5,
            'market_fear_index': self._calculate_market_stress(context)
        }
    
    # Helper methods for dopamine integration
    
    def _process_anticipation_phase(self, decision_context: TradingDecisionContext) -> DopamineResponse:
        """Process anticipation phase before decision execution"""
        anticipation_data = {
            'unrealized_pnl': 0.0,  # No position yet
            'daily_pnl': 0.0,
            'open_positions': 0.0,
            'current_price': 0.0,  # Not relevant for anticipation
            'trade_duration': 0.0
        }
        
        anticipation_context = {
            'confidence': decision_context.confidence,
            'expected_outcome': decision_context.expected_outcome,
            'intelligence_support': self._calculate_intelligence_support(decision_context)
        }
        
        return self.dopamine_core.process_trading_event(
            'anticipation', anticipation_data, anticipation_context
        )
    
    def _create_dopamine_market_data(self, decision_context: TradingDecisionContext) -> Dict[str, Any]:
        """Create dopamine-compatible market data from decision context"""
        market_conditions = decision_context.market_conditions
        
        return {
            'unrealized_pnl': market_conditions.get('unrealized_pnl', 0.0),
            'daily_pnl': market_conditions.get('daily_pnl', 0.0),
            'open_positions': market_conditions.get('open_positions', 0.0),
            'current_price': market_conditions.get('current_price', 0.0),
            'trade_duration': market_conditions.get('trade_duration', 0.0)
        }
    
    def _create_intelligence_dopamine_context(self, decision_context: TradingDecisionContext) -> Dict[str, Any]:
        """Create dopamine context enhanced with intelligence information"""
        base_context = {
            'confidence': decision_context.confidence,
            'expected_outcome': decision_context.expected_outcome,
            'action': decision_context.action
        }
        
        # Add intelligence-specific context
        if self.current_intelligence_update:
            intelligence_context = {
                'intelligence_strength': self._calculate_intelligence_strength(
                    self.current_intelligence_update.signals
                ),
                'intelligence_consensus': self.current_intelligence_update.signal_consensus,
                'signal_quality': self.current_intelligence_update.average_confidence,
                'subsystem_agreement': self._calculate_subsystem_agreement()
            }
            base_context.update(intelligence_context)
        
        return base_context
    
    def _calculate_psychological_adjustments(self, decision_context: TradingDecisionContext,
                                           dopamine_response: DopamineResponse,
                                           anticipation_response: DopamineResponse) -> Dict[str, float]:
        """Calculate psychological adjustments to the base decision"""
        adjustments = {}
        
        # Position size adjustment based on dopamine state
        size_adjustment = dopamine_response.position_size_modifier
        if dopamine_response.state == DopamineState.EUPHORIC:
            size_adjustment *= 0.8  # Reduce size when euphoric (overconfidence protection)
        elif dopamine_response.state == DopamineState.WITHDRAWN:
            size_adjustment *= 0.6  # Further reduce when withdrawn
        elif dopamine_response.state == DopamineState.ADDICTED:
            size_adjustment *= 0.7  # Reduce when addicted (risk protection)
        
        adjustments['position_size_factor'] = size_adjustment
        
        # Confidence adjustment based on psychological state
        confidence_adjustment = 1.0
        if dopamine_response.tolerance_level > 0.7:
            confidence_adjustment *= 0.9  # Slightly reduce confidence when tolerant
        if dopamine_response.withdrawal_intensity > 0.5:
            confidence_adjustment *= 0.85  # Reduce confidence during withdrawal
        
        adjustments['confidence_factor'] = confidence_adjustment
        
        # Risk tolerance adjustment
        risk_adjustment = dopamine_response.risk_tolerance_modifier
        adjustments['risk_factor'] = risk_adjustment
        
        # Urgency factor (affects timing and patience)
        adjustments['urgency_factor'] = dopamine_response.urgency_factor
        
        # Intelligence integration factor
        if self.current_intelligence_update:
            intelligence_factor = min(self.current_intelligence_update.average_confidence * 1.2, 1.0)
            adjustments['intelligence_factor'] = intelligence_factor
        else:
            adjustments['intelligence_factor'] = 1.0
        
        return adjustments
    
    def _create_integrated_decision(self, decision_context: TradingDecisionContext,
                                  dopamine_response: DopamineResponse,
                                  psychological_adjustments: Dict[str, float]) -> DopamineIntegratedDecision:
        """Create the final integrated decision with all adjustments applied"""
        
        # Apply adjustments to confidence
        final_confidence = (decision_context.confidence * 
                          psychological_adjustments.get('confidence_factor', 1.0) *
                          psychological_adjustments.get('intelligence_factor', 1.0))
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        # Apply adjustments to position size
        base_size = decision_context.position_size
        size_factor = psychological_adjustments.get('position_size_factor', 1.0)
        final_position_size = base_size * size_factor
        
        # Ensure position size is within bounds
        final_position_size = max(self.position_size_bounds[0], 
                                min(self.position_size_bounds[1], final_position_size))
        
        # Determine final action based on confidence threshold and psychological state
        final_action = decision_context.action
        if final_confidence < self.confidence_threshold:
            final_action = 'hold'  # Override with hold if confidence too low
        elif dopamine_response.state == DopamineState.WITHDRAWN and decision_context.action != 'hold':
            # Consider reducing action aggressiveness when withdrawn
            if np.random.random() < 0.3:  # 30% chance to hold when withdrawn
                final_action = 'hold'
        
        # Create integration metadata
        integration_metadata = {
            'dopamine_signal': dopamine_response.signal,
            'psychological_state': dopamine_response.state.value,
            'adjustments_applied': psychological_adjustments,
            'confidence_delta': final_confidence - decision_context.confidence,
            'size_delta': final_position_size - decision_context.position_size,
            'action_changed': final_action != decision_context.action,
            'integration_timestamp': time.time()
        }
        
        if self.integration_stats['decisions_processed'] % 10 == 0:  # Log every 10th decision
            logger.info(f"Psychology adjustments: confidence={final_confidence:.3f} "
                       f"(change{final_confidence - decision_context.confidence:+.3f}), "
                       f"size={final_position_size:.3f} "
                       f"(change{final_position_size - decision_context.position_size:+.3f})")
        
        return DopamineIntegratedDecision(
            base_decision=decision_context,
            dopamine_response=dopamine_response,
            psychological_adjustments=psychological_adjustments,
            final_action=final_action,
            final_confidence=final_confidence,
            final_position_size=final_position_size,
            integration_metadata=integration_metadata
        )
    
    def _create_safe_fallback_decision(self, decision_context: TradingDecisionContext) -> DopamineIntegratedDecision:
        """Create a safe fallback decision when integration fails"""
        neutral_response = DopamineResponse(
            signal=0.0, phase=DopaminePhase.MONITORING, state=DopamineState.BALANCED,
            anticipation_factor=0.0, satisfaction_factor=0.0, tolerance_level=0.5,
            addiction_risk=0.0, withdrawal_intensity=0.0, position_size_modifier=1.0,
            risk_tolerance_modifier=1.0, urgency_factor=0.5
        )
        
        return DopamineIntegratedDecision(
            base_decision=decision_context,
            dopamine_response=neutral_response,
            psychological_adjustments={'error': True},
            final_action='hold',  # Safe action
            final_confidence=0.3,  # Conservative confidence
            final_position_size=0.0,  # No position
            integration_metadata={'error': True, 'fallback_used': True}
        )
    
    # Utility helper methods
    
    def _update_integration_tracking(self, integrated_decision: DopamineIntegratedDecision):
        """Update integration tracking statistics"""
        if integrated_decision.integration_metadata.get('action_changed', False):
            self.integration_stats['psychology_adjustments_made'] += 1
        
        if integrated_decision.dopamine_response.phase != DopaminePhase.MONITORING:
            self.integration_stats['phase_transitions'] += 1
        
        self.integration_stats['last_integration_time'] = time.time()
    
    def _calculate_intelligence_support(self, decision_context: TradingDecisionContext) -> float:
        """Calculate how well intelligence signals support the decision"""
        if not self.current_intelligence_update:
            return 0.5  # Neutral support
        
        signals = self.current_intelligence_update.signals
        decision_direction = decision_context.action
        
        if decision_direction == 'hold':
            return 0.5  # Neutral for hold decisions
        
        # Count supporting signals
        supporting_signals = 0
        total_signals = len(signals)
        
        for signal in signals:
            if ((decision_direction in ['buy', 'long'] and signal.direction == 'bullish') or
                (decision_direction in ['sell', 'short'] and signal.direction == 'bearish')):
                supporting_signals += 1
        
        return supporting_signals / total_signals if total_signals > 0 else 0.5
    
    def _calculate_subsystem_agreement(self) -> float:
        """Calculate agreement level between subsystems"""
        if not self.current_intelligence_update:
            return 0.5
        
        return abs(self.current_intelligence_update.signal_consensus)
    
    def _assess_intelligence_accuracy(self, trade_outcome: Dict[str, Any]) -> float:
        """Assess accuracy of intelligence signals for this trade outcome"""
        if not self.last_decision_context:
            return 0.5
        
        # Compare predicted vs actual outcome
        expected = self.last_decision_context.expected_outcome
        actual = trade_outcome.get('pnl', 0.0)
        
        if expected == 0.0 and actual == 0.0:
            return 1.0  # Perfect prediction
        elif expected == 0.0 or actual == 0.0:
            return 0.3  # Poor prediction
        else:
            # Calculate accuracy based on direction and magnitude
            direction_correct = (expected > 0) == (actual > 0)
            magnitude_error = abs(expected - actual) / max(abs(expected), abs(actual))
            
            if direction_correct:
                return max(0.0, 1.0 - magnitude_error)
            else:
                return 0.2  # Wrong direction
    
    # Market psychology mapping methods
    
    def _map_regime_to_psychology(self, regime: str) -> float:
        """Map market regime to psychological stress factor"""
        regime_map = {
            'trending_up': 0.2,
            'trending_down': 0.8,
            'volatile': 0.7,
            'stable': 0.3,
            'uncertain': 0.6,
            'breakout': 0.5,
            'reversal': 0.6
        }
        return regime_map.get(regime.lower(), 0.5)
    
    def _map_volatility_to_psychology(self, volatility: float) -> float:
        """Map volatility level to psychological impact"""
        # Higher volatility increases psychological stress
        return min(volatility * 1.2, 1.0)
    
    def _map_volume_to_psychology(self, volume_profile: str) -> float:
        """Map volume profile to psychological confidence"""
        volume_map = {
            'high': 0.8,    # High confidence
            'normal': 0.5,  # Neutral
            'low': 0.3      # Lower confidence
        }
        return volume_map.get(volume_profile.lower(), 0.5)
    
    def _map_time_to_psychology(self, time_of_day: str) -> float:
        """Map time of day to psychological factors"""
        time_map = {
            'market_open': 0.7,    # Higher energy/risk
            'mid_morning': 0.6,
            'midday': 0.5,
            'afternoon': 0.4,
            'market_close': 0.6,   # Last chance energy
            'after_hours': 0.3     # Lower activity
        }
        return time_map.get(time_of_day.lower(), 0.5)
    
    def _calculate_market_stress(self, context: IntelligenceContext) -> float:
        """Calculate overall market stress level"""
        stress_factors = [
            context.volatility_level,
            1.0 if context.volume_profile == 'low' else 0.0,
            0.8 if context.market_regime in ['volatile', 'uncertain'] else 0.2
        ]
        return np.mean(stress_factors)
    
    def _assess_intelligence_reliability(self, context: IntelligenceContext) -> float:
        """Assess reliability of intelligence signals given market context"""
        reliability = 1.0
        
        # Reduce reliability in high volatility
        if context.is_high_volatility():
            reliability *= 0.8
        
        # Reduce reliability in low volume
        if context.is_low_volume():
            reliability *= 0.9
        
        # Adjust for market regime
        if context.market_regime in ['uncertain', 'volatile']:
            reliability *= 0.85
        
        return max(0.3, reliability)  # Minimum 30% reliability
    
    # Public interface methods
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration performance statistics"""
        return {
            **self.integration_stats.copy(),
            'dopamine_stats': self.dopamine_core.get_stats(),
            'signal_weights': self.signal_weights.copy(),
            'position_size_bounds': self.position_size_bounds,
            'confidence_threshold': self.confidence_threshold
        }
    
    def reset_session(self, preserve_learning: bool = True):
        """Reset dopamine session state"""
        self.dopamine_core.reset_session(preserve_learning)
        self.signal_history.clear()
        self.current_intelligence_update = None
        self.last_decision_context = None
        
        # Reset stats but preserve configuration
        self.integration_stats = {
            'decisions_processed': 0,
            'psychology_adjustments_made': 0,
            'signals_processed': 0,
            'phase_transitions': 0,
            'last_integration_time': 0.0
        }
        
        logger.info(f"AgentDopamineManager session reset (preserve_learning={preserve_learning})")
    
    def save_state(self, filepath: str):
        """Save dopamine manager state"""
        try:
            import json
            import os
            
            # Prepare state data
            state_data = {
                'integration_stats': self.integration_stats,
                'signal_weights': self.signal_weights,
                'position_size_bounds': self.position_size_bounds,
                'confidence_threshold': self.confidence_threshold,
                'intelligence_integration_factor': self.intelligence_integration_factor,
                'dopamine_stats': self.dopamine_core.get_stats(),
                'saved_at': time.time()
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save state
            with open(filepath, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
            
            logger.info(f"AgentDopamineManager state saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving dopamine manager state: {e}")
    
    def load_state(self, filepath: str) -> bool:
        """Load dopamine manager state"""
        try:
            import json
            
            with open(filepath, 'r') as f:
                state_data = json.load(f)
            
            # Restore configuration
            self.integration_stats = state_data.get('integration_stats', self.integration_stats)
            self.signal_weights = state_data.get('signal_weights', self.signal_weights)
            self.position_size_bounds = tuple(state_data.get('position_size_bounds', self.position_size_bounds))
            self.confidence_threshold = state_data.get('confidence_threshold', self.confidence_threshold)
            self.intelligence_integration_factor = state_data.get('intelligence_integration_factor', 
                                                                self.intelligence_integration_factor)
            
            logger.info(f"AgentDopamineManager state loaded from {filepath}")
            return True
            
        except FileNotFoundError:
            logger.info("No dopamine manager state file found, using defaults")
            return False
        except Exception as e:
            logger.error(f"Error loading dopamine manager state: {e}")
            return False


# Export main classes
__all__ = [
    'AgentDopamineManager',
    'TradingDecisionContext', 
    'DopamineIntegratedDecision'
]