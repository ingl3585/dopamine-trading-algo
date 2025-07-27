"""
Consolidated Dopamine Subsystem - The Central Nervous System of Trading
Combines simple and enhanced dopamine pathways for comprehensive trading psychology
with backward compatibility for legacy interfaces.

This module provides:
- Complete neurological dopamine pathway simulation
- Multiple trading phases (anticipation, execution, monitoring, realization, reflection)
- Advanced psychological modeling (tolerance, addiction, withdrawal)
- Backward compatibility with simple dopamine interfaces
- Clean architecture with SOLID principles
"""

import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple, Protocol
from collections import deque
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class DopaminePhase(Enum):
    """Different phases of dopamine release in trading cycle"""
    ANTICIPATION = "anticipation"  # Before trade entry
    EXECUTION = "execution"        # During trade execution
    MONITORING = "monitoring"      # While position is open
    REALIZATION = "realization"    # At trade exit
    REFLECTION = "reflection"      # Post-trade analysis

class DopamineState(Enum):
    """Current psychological state based on dopamine levels"""
    EUPHORIC = "euphoric"          # High dopamine, risk of overconfidence
    CONFIDENT = "confident"        # Healthy positive state
    BALANCED = "balanced"          # Neutral, optimal state
    CAUTIOUS = "cautious"          # Low dopamine, conservative
    WITHDRAWN = "withdrawn"        # Very low, risk averse
    ADDICTED = "addicted"          # Tolerance built up, chasing highs

@dataclass
class DopamineSnapshot:
    timestamp: float
    phase: DopaminePhase
    unrealized_pnl: float
    realized_pnl: float
    position_size: float
    current_price: float
    trade_duration: float = 0.0
    expected_outcome: float = 0.0
    confidence_level: float = 0.5

@dataclass
class DopamineResponse:
    """Complete dopamine response with contextual information"""
    signal: float                    # Primary dopamine signal [-2.0 to +2.0]
    phase: DopaminePhase
    state: DopamineState
    anticipation_factor: float       # Pre-trade excitement
    satisfaction_factor: float       # Current reward satisfaction
    tolerance_level: float           # Current tolerance to rewards
    addiction_risk: float            # Risk of overtrading
    withdrawal_intensity: float      # Severity of current withdrawal
    position_size_modifier: float    # Suggested position size adjustment
    risk_tolerance_modifier: float   # Suggested risk tolerance adjustment
    urgency_factor: float           # Trading urgency/patience level

# Define protocols for clean architecture
class DopamineSignalProvider(Protocol):
    """Protocol for providing dopamine signals"""
    def get_signal(self) -> float: ...
    def get_signal_with_context(self) -> Dict: ...

class DopamineEventProcessor(Protocol):
    """Protocol for processing trading events"""
    def process_trading_event(self, event_type: str, market_data: Dict, context: Dict = None) -> DopamineResponse: ...

class DopamineLearningInterface(Protocol):
    """Protocol for learning from trading outcomes"""
    def learn_from_outcome(self, outcome: float, context: Optional[Dict] = None): ...

# Backward compatibility interface
class LegacyDopamineInterface(ABC):
    """Abstract base class for backward compatibility with simple dopamine interface"""
    
    @abstractmethod
    def process_pnl_update(self, market_data: Dict) -> float:
        """Process P&L update for simple interface compatibility"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict:
        """Get comprehensive statistics for backward compatibility"""
        pass
    
    @abstractmethod
    def reset_session(self):
        """Reset session for backward compatibility"""
        pass

class ConsolidatedDopamineSubsystem(LegacyDopamineInterface):
    """
    Enhanced Dopamine-inspired trading psychology system
    
    Core Concepts:
    - Complete dopamine pathway simulation (anticipation → execution → satisfaction)
    - Tolerance and addiction modeling
    - Withdrawal and craving simulation
    - Dynamic risk and position sizing based on dopamine state
    - Integration with confidence and personality systems
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Enhanced tracking systems
        self.pnl_history = deque(maxlen=200)  # Longer history for better patterns
        self.response_history = deque(maxlen=100)
        self.phase_transitions = deque(maxlen=50)
        
        # Current state
        self.current_phase = DopaminePhase.MONITORING
        self.current_state = DopamineState.BALANCED
        self.current_response = DopamineResponse(
            signal=0.0, phase=DopaminePhase.MONITORING, state=DopamineState.BALANCED,
            anticipation_factor=0.0, satisfaction_factor=0.0, tolerance_level=0.5,
            addiction_risk=0.0, withdrawal_intensity=0.0, position_size_modifier=1.0,
            risk_tolerance_modifier=1.0, urgency_factor=0.5
        )
        
        # ENHANCED: Increased sensitivity for stronger signals
        self.base_sensitivity = config.get('dopamine_sensitivity', 0.1)  # INCREASED from 0.01 to 0.1 for stronger signals
        self.current_sensitivity = self.base_sensitivity
        self.momentum_factor = config.get('momentum_factor', 0.05)  # INCREASED for better momentum detection
        self.max_signal = config.get('max_dopamine_signal', 10.0)  # Higher ceiling for discovery
        
        # Simplified psychological parameters - let AI discover psychology
        self.tolerance_buildup_rate = config.get('tolerance_buildup_rate', 0.001)
        self.tolerance_decay_rate = config.get('tolerance_decay_rate', 0.001) 
        self.addiction_threshold = config.get('addiction_threshold', 10.0)  # Higher threshold
        self.withdrawal_severity = config.get('withdrawal_severity', 0.01)  # Lower severity
        self.anticipation_multiplier = config.get('anticipation_multiplier', 1.01)  # Minimal initial
        
        # Tracking variables
        self.consecutive_positive = 0
        self.consecutive_negative = 0
        self.peak_pnl = 0.0
        self.trough_pnl = 0.0
        self.session_peak_dopamine = 0.0
        self.session_trough_dopamine = 0.0
        
        # Tolerance and addiction tracking
        self.tolerance_level = 0.5  # 0 = highly sensitive, 1 = maximum tolerance
        self.addiction_score = 0.0  # 0 = healthy, 1 = severe addiction
        self.withdrawal_intensity = 0.0  # 0 = no withdrawal, 1 = severe withdrawal
        self.time_since_last_high = 0.0
        self.recent_high_count = 0
        
        # Expectation and prediction
        self.expected_pnl = 0.0
        self.expectation_confidence = 0.5
        self.prediction_accuracy = deque(maxlen=20)
        
        # Position and risk modifiers
        self.position_size_modifier = 1.0
        self.risk_tolerance_modifier = 1.0
        self.urgency_factor = 0.5  # 0 = very patient, 1 = urgent/impulsive

    def process_trading_event(self, event_type: str, market_data: Dict, 
                            context: Dict = None) -> DopamineResponse:
        """
        Process any trading event and generate comprehensive dopamine response
        
        Args:
            event_type: 'anticipation', 'execution', 'monitoring', 'realization', 'reflection'
            market_data: Market and P&L data
            context: Additional context (confidence, decision data, etc.)
            
        Returns:
            DopamineResponse: Complete dopamine response with all factors
        """
        try:
            current_time = time.time()
            phase = DopaminePhase(event_type)
            
            # Extract data
            unrealized_pnl = market_data.get('unrealized_pnl', 0.0)
            realized_pnl = market_data.get('daily_pnl', 0.0)
            position_size = market_data.get('open_positions', 0.0)
            current_price = market_data.get('current_price', 0.0)
            trade_duration = market_data.get('trade_duration', 0.0)
            
            # Extract context
            context = context or {}
            expected_outcome = context.get('expected_outcome', 0.0)
            confidence_level = context.get('confidence', 0.5)
            
            # Create snapshot
            snapshot = DopamineSnapshot(
                timestamp=current_time,
                phase=phase,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl,
                position_size=position_size,
                current_price=current_price,
                trade_duration=trade_duration,
                expected_outcome=expected_outcome,
                confidence_level=confidence_level
            )
            
            # Calculate comprehensive dopamine response
            response = self._calculate_enhanced_dopamine_response(snapshot)
            
            # Update all tracking systems
            self._update_enhanced_tracking(snapshot, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in dopamine event processing: {e}")
            return self.current_response
    
    # ========================================
    # BACKWARD COMPATIBILITY METHODS
    # ========================================
    
    def process_pnl_update(self, market_data: Dict) -> float:
        """
        Process P&L update for backward compatibility with simple interface.
        
        Args:
            market_data: Dictionary containing unrealized_pnl, position_size, etc.
            
        Returns:
            float: Dopamine signal [-10.0 to +10.0] (enhanced range)
        """
        response = self.process_trading_event('monitoring', market_data)
        return response.signal
    
    def get_simple_signal(self, market_data: Dict) -> float:
        """
        Backwards compatibility - returns simple dopamine signal
        """
        response = self.process_trading_event('monitoring', market_data)
        return response.signal
    
    def get_signal_with_context(self) -> Dict:
        """Get dopamine signal with context for backward compatibility"""
        response = self.current_response
        
        return {
            'signal': response.signal,
            'previous_pnl': self.pnl_history[-1].unrealized_pnl if self.pnl_history else 0.0,
            'consecutive_positive': self.consecutive_positive,
            'consecutive_negative': self.consecutive_negative,
            'peak_pnl': self.peak_pnl,
            'trough_pnl': self.trough_pnl,
            'momentum_state': self._get_momentum_state(),
            # Enhanced context
            'psychological_state': response.state.value,
            'phase': response.phase.value,
            'tolerance_level': self.tolerance_level,
            'addiction_score': self.addiction_score,
            'withdrawal_intensity': self.withdrawal_intensity,
            'position_size_modifier': response.position_size_modifier,
            'risk_tolerance_modifier': response.risk_tolerance_modifier,
            'urgency_factor': response.urgency_factor
        }
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics for backward compatibility"""
        signal_history = [r.signal for r in list(self.response_history)]
        pnl_history_values = [s.unrealized_pnl for s in list(self.pnl_history)]
        
        # Calculate statistics
        if signal_history:
            avg_signal = np.mean(signal_history)
            signal_volatility = np.std(signal_history)
            max_signal = np.max(signal_history)
            min_signal = np.min(signal_history)
        else:
            avg_signal = signal_volatility = max_signal = min_signal = 0.0
        
        return {
            # Original stats for backward compatibility
            'current_signal': self.current_response.signal,
            'avg_signal': avg_signal,
            'signal_volatility': signal_volatility,
            'max_signal': max_signal,
            'min_signal': min_signal,
            'signal_history_length': len(signal_history),
            'pnl_history_length': len(pnl_history_values),
            'consecutive_positive': self.consecutive_positive,
            'consecutive_negative': self.consecutive_negative,
            'peak_pnl': self.peak_pnl,
            'trough_pnl': self.trough_pnl,
            'momentum_state': self._get_momentum_state(),
            'expected_pnl': self.expected_pnl,
            'expectation_confidence': self.expectation_confidence,
            'max_consecutive_positive': max(self.consecutive_positive, 0),
            'max_consecutive_negative': max(self.consecutive_negative, 0),
            
            # Enhanced stats
            'psychological_state': self.current_response.state.value,
            'current_phase': self.current_response.phase.value,
            'tolerance_level': self.tolerance_level,
            'addiction_score': self.addiction_score,
            'withdrawal_intensity': self.withdrawal_intensity,
            'position_size_modifier': self.current_response.position_size_modifier,
            'risk_tolerance_modifier': self.current_response.risk_tolerance_modifier,
            'urgency_factor': self.current_response.urgency_factor,
            'session_peak_dopamine': self.session_peak_dopamine,
            'session_trough_dopamine': self.session_trough_dopamine,
            'psychological_health_score': self._calculate_psychological_health(),
            'addiction_risk_level': self._assess_addiction_risk(),
            'withdrawal_risk_level': self._assess_withdrawal_risk()
        }
    
    def _calculate_enhanced_dopamine_response(self, snapshot: DopamineSnapshot) -> DopamineResponse:
        """Calculate comprehensive dopamine response based on all factors"""
        
        # Phase-specific calculations
        if snapshot.phase == DopaminePhase.ANTICIPATION:
            return self._calculate_anticipation_response(snapshot)
        elif snapshot.phase == DopaminePhase.EXECUTION:
            return self._calculate_execution_response(snapshot)
        elif snapshot.phase == DopaminePhase.MONITORING:
            return self._calculate_monitoring_response(snapshot)
        elif snapshot.phase == DopaminePhase.REALIZATION:
            return self._calculate_realization_response(snapshot)
        elif snapshot.phase == DopaminePhase.REFLECTION:
            return self._calculate_reflection_response(snapshot)
        else:
            return self._calculate_monitoring_response(snapshot)  # Default
    
    def _calculate_monitoring_response(self, snapshot: DopamineSnapshot) -> DopamineResponse:
        """Calculate dopamine response for active position monitoring"""
        
        # Get previous P&L for comparison
        previous_pnl = self.pnl_history[-1].unrealized_pnl if self.pnl_history else 0.0
        pnl_change = snapshot.unrealized_pnl - previous_pnl
        
        # Base reward with tolerance adjustment
        adjusted_sensitivity = self.current_sensitivity * (1.0 - self.tolerance_level * 0.5)
        base_reward = np.tanh(pnl_change * adjusted_sensitivity)
        
        # DEBUG: Log key calculations for troubleshooting
        if abs(pnl_change) > 0.01 or len(self.pnl_history) < 5:  # Log significant changes or initial calls
            logger.debug(f"DOPAMINE: PnL change={pnl_change:.2f}, sensitivity={adjusted_sensitivity:.4f}, base_reward={base_reward:.4f}")
        
        # Enhanced momentum calculation
        momentum_multiplier = self._calculate_momentum_multiplier(pnl_change)
        
        # Expectation error with confidence weighting
        expectation_error = self._calculate_expectation_error(snapshot)
        
        # Peak/trough analysis
        peak_trough_factor = self._calculate_peak_trough_factor(snapshot.unrealized_pnl)
        
        # Tolerance and addiction factors
        tolerance_damping = 1.0 - (self.tolerance_level * 0.3)
        addiction_amplification = 1.0 + (self.addiction_score * 0.2)
        
        # Withdrawal effects
        withdrawal_penalty = -self.withdrawal_intensity * 0.5
        
        # Combine all factors
        raw_signal = (
            (base_reward * momentum_multiplier * tolerance_damping * addiction_amplification) +
            expectation_error + peak_trough_factor + withdrawal_penalty
        )
        
        # Bound the signal
        dopamine_signal = np.clip(raw_signal, -self.max_signal, self.max_signal)
        
        # DEBUG: Log final signal calculation
        if abs(dopamine_signal) > 0.01 or len(self.pnl_history) < 5:
            logger.debug(f"DOPAMINE FINAL: raw_signal={raw_signal:.4f}, final_signal={dopamine_signal:.4f}, tolerance={self.tolerance_level:.3f}")
        
        # Determine psychological state
        psychological_state = self._determine_psychological_state(dopamine_signal)
        
        # Calculate modifiers
        position_modifier = self._calculate_position_size_modifier()
        risk_modifier = self._calculate_risk_tolerance_modifier()
        urgency = self._calculate_urgency_factor()
        
        return DopamineResponse(
            signal=dopamine_signal,
            phase=snapshot.phase,
            state=psychological_state,
            anticipation_factor=0.0,  # Not applicable for monitoring
            satisfaction_factor=np.tanh(snapshot.unrealized_pnl * 0.1),
            tolerance_level=self.tolerance_level,
            addiction_risk=self.addiction_score,
            withdrawal_intensity=self.withdrawal_intensity,
            position_size_modifier=position_modifier,
            risk_tolerance_modifier=risk_modifier,
            urgency_factor=urgency
        )
    
    def _calculate_anticipation_response(self, snapshot: DopamineSnapshot) -> DopamineResponse:
        """Calculate dopamine response for pre-trade anticipation"""
        
        # Anticipation builds based on confidence and expected outcome
        anticipation_level = snapshot.confidence_level * self.anticipation_multiplier
        
        # Expected outcome drives anticipation
        expected_reward = np.tanh(snapshot.expected_outcome * 0.1)
        
        # Tolerance reduces anticipation satisfaction
        tolerance_damping = 1.0 - (self.tolerance_level * 0.4)
        
        # Addiction increases craving for trades
        addiction_craving = self.addiction_score * 0.5
        
        # Withdrawal makes anticipation more intense (desperation)
        withdrawal_boost = self.withdrawal_intensity * 0.3
        
        # Calculate anticipation signal
        anticipation_signal = (
            (expected_reward * anticipation_level * tolerance_damping) +
            addiction_craving + withdrawal_boost
        )
        
        dopamine_signal = np.clip(anticipation_signal, -self.max_signal, self.max_signal)
        
        return DopamineResponse(
            signal=dopamine_signal,
            phase=snapshot.phase,
            state=self._determine_psychological_state(dopamine_signal),
            anticipation_factor=anticipation_level,
            satisfaction_factor=0.0,
            tolerance_level=self.tolerance_level,
            addiction_risk=self.addiction_score,
            withdrawal_intensity=self.withdrawal_intensity,
            position_size_modifier=self._calculate_position_size_modifier(),
            risk_tolerance_modifier=self._calculate_risk_tolerance_modifier(),
            urgency_factor=self._calculate_urgency_factor()
        )
    
    def _calculate_execution_response(self, snapshot: DopamineSnapshot) -> DopamineResponse:
        """Calculate dopamine response during trade execution"""
        
        # Execution provides immediate reward/punishment
        if snapshot.position_size != 0:
            # Trade was executed - immediate satisfaction
            execution_satisfaction = 0.3 * (1.0 - self.tolerance_level * 0.5)
            
            # Addiction amplifies execution high
            if self.addiction_score > 0.5:
                execution_satisfaction *= (1.0 + self.addiction_score)
        else:
            # Trade was rejected - disappointment
            execution_satisfaction = -0.2 - (self.withdrawal_intensity * 0.3)
        
        dopamine_signal = np.clip(execution_satisfaction, -self.max_signal, self.max_signal)
        
        return DopamineResponse(
            signal=dopamine_signal,
            phase=snapshot.phase,
            state=self._determine_psychological_state(dopamine_signal),
            anticipation_factor=0.0,
            satisfaction_factor=execution_satisfaction,
            tolerance_level=self.tolerance_level,
            addiction_risk=self.addiction_score,
            withdrawal_intensity=self.withdrawal_intensity,
            position_size_modifier=self._calculate_position_size_modifier(),
            risk_tolerance_modifier=self._calculate_risk_tolerance_modifier(),
            urgency_factor=self._calculate_urgency_factor()
        )
    
    def _calculate_realization_response(self, snapshot: DopamineSnapshot) -> DopamineResponse:
        """Calculate dopamine response at trade exit (profit/loss realization)"""
        
        # Realized P&L drives the response
        realized_reward = np.tanh(snapshot.realized_pnl * 0.05)
        
        # Compare to expectation
        expectation_error = snapshot.realized_pnl - snapshot.expected_outcome
        surprise_factor = np.tanh(expectation_error * 0.1) * 0.5
        
        # Tolerance reduces satisfaction from wins
        tolerance_damping = 1.0 - (self.tolerance_level * 0.6)
        
        # Build or reduce tolerance based on outcome
        if snapshot.realized_pnl > 0:
            # Win - builds tolerance, may trigger addiction
            tolerance_buildup = 0.1
        else:
            # Loss - reduces tolerance, may trigger withdrawal
            tolerance_buildup = -0.05
        
        realization_signal = (
            (realized_reward * tolerance_damping) + surprise_factor
        )
        
        dopamine_signal = np.clip(realization_signal, -self.max_signal, self.max_signal)
        
        return DopamineResponse(
            signal=dopamine_signal,
            phase=snapshot.phase,
            state=self._determine_psychological_state(dopamine_signal),
            anticipation_factor=0.0,
            satisfaction_factor=realized_reward,
            tolerance_level=self.tolerance_level,
            addiction_risk=self.addiction_score,
            withdrawal_intensity=self.withdrawal_intensity,
            position_size_modifier=self._calculate_position_size_modifier(),
            risk_tolerance_modifier=self._calculate_risk_tolerance_modifier(),
            urgency_factor=self._calculate_urgency_factor()
        )
    
    def _calculate_reflection_response(self, snapshot: DopamineSnapshot) -> DopamineResponse:
        """Calculate dopamine response during post-trade reflection"""
        
        # FIXED: Reflection should match the trade outcome, not always be positive
        # Base reflection on actual trade result
        base_reflection = np.tanh(snapshot.realized_pnl * 0.03)  # Muted version of realization
        
        # Learning component (small positive regardless of outcome)
        learning_satisfaction = 0.05 if abs(snapshot.realized_pnl) > 0 else 0.0
        
        # Recent performance trend (now properly weighted)
        recent_performance = self._get_recent_performance_trend()
        trend_influence = np.tanh(recent_performance * 0.02) * 0.3
        
        # FIXED: Reflection signal properly combines outcome + trend + learning
        reflection_signal = base_reflection + trend_influence + learning_satisfaction
        
        dopamine_signal = np.clip(reflection_signal, -self.max_signal, self.max_signal)
        
        return DopamineResponse(
            signal=dopamine_signal,
            phase=snapshot.phase,
            state=self._determine_psychological_state(dopamine_signal),
            anticipation_factor=0.0,
            satisfaction_factor=base_reflection,  # FIXED: Use actual trade satisfaction
            tolerance_level=self.tolerance_level,
            addiction_risk=self.addiction_score,
            withdrawal_intensity=self.withdrawal_intensity,
            position_size_modifier=self._calculate_position_size_modifier(),
            risk_tolerance_modifier=self._calculate_risk_tolerance_modifier(),
            urgency_factor=self._calculate_urgency_factor()
        )
    
    def _calculate_momentum_multiplier(self, pnl_change: float) -> float:
        """Enhanced momentum calculation with addiction and tolerance factors"""
        
        if pnl_change > 0:
            self.consecutive_positive += 1
            self.consecutive_negative = 0
            momentum = 1.0 + (self.consecutive_positive * self.momentum_factor)
            
            # Addiction amplifies positive momentum
            if self.addiction_score > 0.3:
                momentum *= (1.0 + self.addiction_score * 0.5)
                
        elif pnl_change < 0:
            self.consecutive_negative += 1
            self.consecutive_positive = 0
            momentum = 1.0 + (self.consecutive_negative * self.momentum_factor)
            
            # Withdrawal amplifies negative momentum
            if self.withdrawal_intensity > 0.3:
                momentum *= (1.0 + self.withdrawal_intensity * 0.3)
        else:
            self.consecutive_positive = 0
            self.consecutive_negative = 0
            momentum = 1.0
        
        return momentum
    
    def _calculate_expectation_error(self, snapshot: DopamineSnapshot) -> float:
        """Enhanced expectation error calculation"""
        
        if self.expectation_confidence > 0.3:
            actual_vs_expected = snapshot.unrealized_pnl - self.expected_pnl
            
            # Weight by confidence and recency
            confidence_weight = min(self.expectation_confidence, 1.0)
            base_error = np.tanh(actual_vs_expected * 0.05) * 0.3
            
            return base_error * confidence_weight
        
        return 0.0
    
    def _calculate_peak_trough_factor(self, current_pnl: float) -> float:
        """Enhanced peak/trough analysis with tolerance consideration"""
        
        factor = 0.0
        
        # Update peaks and troughs
        if current_pnl > self.peak_pnl:
            self.peak_pnl = current_pnl
            # New highs less exciting with high tolerance
            factor = 0.15 * (1.0 - self.tolerance_level * 0.5)
            
        elif current_pnl < self.trough_pnl:
            self.trough_pnl = current_pnl
            # New lows more painful with withdrawal
            factor = -0.15 * (1.0 + self.withdrawal_intensity * 0.5)
        
        # Enhanced drawdown analysis
        if self.peak_pnl > 0:
            drawdown_pct = (self.peak_pnl - current_pnl) / self.peak_pnl
            if drawdown_pct > 0.05:  # 5% drawdown threshold
                # Drawdown pain amplified by low tolerance
                drawdown_pain = -0.3 * drawdown_pct * (1.0 + (1.0 - self.tolerance_level))
                factor += drawdown_pain
        
        return factor
    
    def _determine_psychological_state(self, dopamine_signal: float) -> DopamineState:
        """Determine current psychological state based on dopamine and other factors"""
        
        # FIXED: More sensitive state detection for realistic behavior
        # Consider signal strength, tolerance, addiction, and withdrawal
        if self.addiction_score > 0.4:  # Reduced from 0.7 - earlier detection
            return DopamineState.ADDICTED
        elif self.withdrawal_intensity > 0.4:  # Reduced from 0.6 - earlier detection
            return DopamineState.WITHDRAWN
        elif dopamine_signal > 0.8:  # Reduced from 1.2 - more realistic
            return DopamineState.EUPHORIC
        elif dopamine_signal > 0.2:  # Reduced from 0.3
            return DopamineState.CONFIDENT
        elif dopamine_signal < -0.4:  # Increased from -0.8 - more sensitive
            return DopamineState.CAUTIOUS
        else:
            return DopamineState.BALANCED
    
    def _calculate_position_size_modifier(self) -> float:
        """Calculate position size modifier based on dopamine state"""
        
        modifier = 1.0
        
        # Addiction tends to increase position sizes (risk)
        if self.addiction_score > 0.3:
            modifier *= (1.0 + self.addiction_score * 0.5)
        
        # High tolerance reduces position sizes (need bigger wins)
        if self.tolerance_level > 0.6:
            modifier *= (1.0 + (self.tolerance_level - 0.6) * 0.3)
        
        # Withdrawal reduces position sizes (fear)
        if self.withdrawal_intensity > 0.3:
            modifier *= (1.0 - self.withdrawal_intensity * 0.4)
        
        # Euphoric state increases risk-taking
        if self.current_state == DopamineState.EUPHORIC:
            modifier *= 1.3
        elif self.current_state == DopamineState.WITHDRAWN:
            modifier *= 0.5
        
        return np.clip(modifier, 0.1, 3.0)
    
    def _calculate_risk_tolerance_modifier(self) -> float:
        """Calculate risk tolerance modifier based on dopamine state"""
        
        modifier = 1.0
        
        # Recent wins increase risk tolerance
        if self.consecutive_positive > 2:
            modifier *= (1.0 + self.consecutive_positive * 0.1)
        
        # Recent losses decrease risk tolerance  
        if self.consecutive_negative > 2:
            modifier *= (1.0 - self.consecutive_negative * 0.1)
        
        # Addiction increases risk tolerance
        if self.addiction_score > 0.4:
            modifier *= (1.0 + self.addiction_score * 0.3)
        
        # Withdrawal decreases risk tolerance
        if self.withdrawal_intensity > 0.4:
            modifier *= (1.0 - self.withdrawal_intensity * 0.5)
        
        return np.clip(modifier, 0.3, 2.0)
    
    def _calculate_urgency_factor(self) -> float:
        """Calculate trading urgency/patience level"""
        
        urgency = 0.5  # Baseline
        
        # Addiction increases urgency (need for action)
        if self.addiction_score > 0.3:
            urgency += self.addiction_score * 0.4
        
        # Withdrawal increases urgency (desperation)
        if self.withdrawal_intensity > 0.3:
            urgency += self.withdrawal_intensity * 0.3
        
        # Recent losses increase urgency (revenge trading)
        if self.consecutive_negative > 1:
            urgency += min(self.consecutive_negative * 0.1, 0.3)
        
        # High tolerance decreases urgency (need bigger opportunities)
        if self.tolerance_level > 0.5:
            urgency -= (self.tolerance_level - 0.5) * 0.2
        
        return np.clip(urgency, 0.0, 1.0)
    
    def _get_recent_performance_trend(self) -> float:
        """Get recent performance trend for reflection calculations"""
        
        if len(self.pnl_history) < 3:
            return 0.0
        
        # FIXED: Use both realized and unrealized P&L for trend calculation
        recent_snapshots = list(self.pnl_history)[-5:]
        recent_pnls = []
        
        for snapshot in recent_snapshots:
            # Combine realized and unrealized P&L for comprehensive performance view
            total_pnl = snapshot.realized_pnl + (snapshot.unrealized_pnl * 0.5)  # Weight unrealized less
            recent_pnls.append(total_pnl)
        
        if len(recent_pnls) < 2:
            return 0.0
            
        # Calculate trend slope rather than just average
        trend_slope = (recent_pnls[-1] - recent_pnls[0]) / len(recent_pnls)
        return trend_slope
    
    def _update_enhanced_tracking(self, snapshot: DopamineSnapshot, response: DopamineResponse):
        """Update all enhanced tracking systems"""
        
        # Add to histories
        self.pnl_history.append(snapshot)
        self.response_history.append(response)
        
        # Track phase transitions
        if len(self.response_history) > 1:
            prev_phase = self.response_history[-2].phase
            if prev_phase != snapshot.phase:
                self.phase_transitions.append({
                    'timestamp': snapshot.timestamp,
                    'from_phase': prev_phase,
                    'to_phase': snapshot.phase
                })
        
        # Update current state
        self.current_phase = snapshot.phase
        self.current_state = response.state
        self.current_response = response
        
        # Update expectations with enhanced logic
        self._update_expectations(snapshot)
        
        # Update tolerance, addiction, and withdrawal
        self._update_tolerance_addiction_withdrawal(response)
        
        # Update session peaks and troughs for dopamine
        if response.signal > self.session_peak_dopamine:
            self.session_peak_dopamine = response.signal
        if response.signal < self.session_trough_dopamine:
            self.session_trough_dopamine = response.signal
        
        # Update timing for withdrawal calculations
        if response.signal > 0.5:
            self.time_since_last_high = 0.0
            self.recent_high_count += 1
        else:
            self.time_since_last_high += 1
        
        # Update modifiers for next calculations
        self.position_size_modifier = response.position_size_modifier
        self.risk_tolerance_modifier = response.risk_tolerance_modifier
        self.urgency_factor = response.urgency_factor
    
    def _update_expectations(self, snapshot: DopamineSnapshot):
        """Update expectation tracking with enhanced logic"""
        
        if len(self.pnl_history) >= 3:
            recent_pnls = [s.unrealized_pnl for s in list(self.pnl_history)[-5:]]
            
            # Weighted average favoring recent data
            weights = np.linspace(0.5, 1.0, len(recent_pnls))
            weights /= weights.sum()
            
            self.expected_pnl = np.average(recent_pnls, weights=weights)
            
            # Enhanced confidence calculation
            pnl_variance = np.var(recent_pnls)
            trend_consistency = self._calculate_trend_consistency(recent_pnls)
            
            base_confidence = 1.0 / (1.0 + pnl_variance)
            trend_bonus = trend_consistency * 0.3
            
            self.expectation_confidence = np.clip(base_confidence + trend_bonus, 0.1, 1.0)
    
    def _calculate_trend_consistency(self, values: List[float]) -> float:
        """Calculate how consistent the trend is in recent values"""
        
        if len(values) < 3:
            return 0.0
        
        # Calculate differences
        diffs = np.diff(values)
        
        # Count consistent direction changes
        positive_changes = sum(1 for d in diffs if d > 0)
        negative_changes = sum(1 for d in diffs if d < 0)
        
        # Consistency is high when most changes are in same direction
        total_changes = len(diffs)
        if total_changes == 0:
            return 0.0
        
        max_consistent = max(positive_changes, negative_changes)
        return max_consistent / total_changes
    
    def _update_tolerance_addiction_withdrawal(self, response: DopamineResponse):
        """Update tolerance, addiction, and withdrawal levels"""
        
        # Tolerance buildup and decay
        if response.signal > 0.8:  # High dopamine builds tolerance
            self.tolerance_level += self.tolerance_buildup_rate
        else:  # Low dopamine reduces tolerance
            self.tolerance_level -= self.tolerance_decay_rate
        
        self.tolerance_level = np.clip(self.tolerance_level, 0.0, 1.0)
        
        # FIXED: Addiction scoring - more realistic triggers
        # Track winning streaks regardless of tolerance initially
        if response.signal > 0.5:  # Any decent positive signal builds addiction risk
            self.addiction_score += 0.02
        elif self.consecutive_positive >= 2 and self.tolerance_level > 0.3:  # Lower thresholds
            self.addiction_score += 0.03
        elif response.signal < -0.2:  # Negative signals reduce addiction
            self.addiction_score -= 0.025
        
        self.addiction_score = np.clip(self.addiction_score, 0.0, 1.0)
        
        # Withdrawal intensity with recovery mechanism
        if self.time_since_last_high > 5 and self.tolerance_level > 0.4:
            self.withdrawal_intensity += 0.1
        elif response.signal > 0.3:
            self.withdrawal_intensity -= 0.05
        
        # RECOVERY MECHANISM: Accelerated withdrawal recovery when stuck in withdrawn state
        if self.withdrawal_intensity > 0.7 and self.time_since_last_high > 10:
            # Force recovery after extended withdrawal period
            recovery_rate = 0.15 + (self.time_since_last_high * 0.01)  # Accelerating recovery
            self.withdrawal_intensity -= recovery_rate
            logger.info(f"DOPAMINE RECOVERY: Accelerated withdrawal recovery, intensity reduced by {recovery_rate:.3f}")
        
        self.withdrawal_intensity = np.clip(self.withdrawal_intensity, 0.0, 1.0)
        
        # Adjust sensitivity based on tolerance
        tolerance_factor = 1.0 - (self.tolerance_level * 0.5)
        self.current_sensitivity = self.base_sensitivity * tolerance_factor

    def get_signal(self) -> float:
        """Get current dopamine signal (backwards compatibility)"""
        return self.current_response.signal
    
    def get_current_response(self) -> DopamineResponse:
        """Get complete current dopamine response"""
        return self.current_response
    
    def get_psychological_state(self) -> DopamineState:
        """Get current psychological state"""
        return self.current_state
    
    def get_position_size_modifier(self) -> float:
        """Get current position size modifier"""
        return self.position_size_modifier
    
    def get_risk_tolerance_modifier(self) -> float:
        """Get current risk tolerance modifier"""
        return self.risk_tolerance_modifier
    
    def get_urgency_factor(self) -> float:
        """Get current trading urgency factor"""
        return self.urgency_factor

    def get_comprehensive_context(self) -> Dict:
        """Get comprehensive dopamine context with all psychological factors"""
        
        response = self.current_response
        
        # Calculate signal statistics
        recent_signals = [r.signal for r in list(self.response_history)[-10:]]
        avg_recent_signal = np.mean(recent_signals) if recent_signals else 0.0
        signal_volatility = np.std(recent_signals) if len(recent_signals) > 1 else 0.0
        
        # Phase distribution
        recent_phases = [r.phase.value for r in list(self.response_history)[-20:]]
        phase_distribution = {phase.value: recent_phases.count(phase.value) for phase in DopaminePhase}
        
        return {
            # Current state
            'dopamine_signal': response.signal,
            'psychological_state': response.state.value,
            'current_phase': response.phase.value,
            'signal_strength': abs(response.signal),
            'signal_direction': "positive" if response.signal > 0 else "negative" if response.signal < 0 else "neutral",
            
            # Advanced psychological factors
            'tolerance_level': self.tolerance_level,
            'addiction_score': self.addiction_score,
            'withdrawal_intensity': self.withdrawal_intensity,
            'anticipation_factor': response.anticipation_factor,
            'satisfaction_factor': response.satisfaction_factor,
            
            # Behavioral modifiers
            'position_size_modifier': response.position_size_modifier,
            'risk_tolerance_modifier': response.risk_tolerance_modifier,
            'urgency_factor': response.urgency_factor,
            
            # Historical patterns
            'consecutive_positive': self.consecutive_positive,
            'consecutive_negative': self.consecutive_negative,
            'peak_pnl': self.peak_pnl,
            'trough_pnl': self.trough_pnl,
            'session_peak_dopamine': self.session_peak_dopamine,
            'session_trough_dopamine': self.session_trough_dopamine,
            
            # Expectations and learning
            'expected_pnl': self.expected_pnl,
            'expectation_confidence': self.expectation_confidence,
            'time_since_last_high': self.time_since_last_high,
            'recent_high_count': self.recent_high_count,
            
            # Signal statistics
            'avg_recent_signal': avg_recent_signal,
            'signal_volatility': signal_volatility,
            'phase_distribution': phase_distribution,
            
            # Momentum and trends
            'momentum_state': self._get_momentum_state(),
            'performance_trend': self._get_recent_performance_trend()
        }
    
    def _get_momentum_state(self) -> str:
        """Get enhanced descriptive momentum state"""
        
        if self.addiction_score > 0.7:
            return "addictive_momentum"
        elif self.withdrawal_intensity > 0.6:
            return "withdrawal_momentum"
        elif self.consecutive_positive >= 5:
            return "euphoric_momentum"
        elif self.consecutive_positive >= 3:
            return "strong_positive_momentum"
        elif self.consecutive_positive >= 1:
            return "positive_momentum"
        elif self.consecutive_negative >= 5:
            return "despair_momentum"
        elif self.consecutive_negative >= 3:
            return "strong_negative_momentum"
        elif self.consecutive_negative >= 1:
            return "negative_momentum"
        else:
            return "neutral_momentum"

    def reset_session(self, preserve_learning: bool = True):
        """
        Reset dopamine system for new trading session.
        
        Args:
            preserve_learning: If True, preserves tolerance, addiction, and learning state.
                             If False, performs complete reset (backward compatibility).
        """
        
        # Always reset these
        self.pnl_history.clear()
        self.response_history.clear() 
        self.phase_transitions.clear()
        self.consecutive_positive = 0
        self.consecutive_negative = 0
        self.peak_pnl = 0.0
        self.trough_pnl = 0.0
        self.session_peak_dopamine = 0.0
        self.session_trough_dopamine = 0.0
        self.time_since_last_high = 0.0
        self.recent_high_count = 0
        
        # Reset current state
        self.current_phase = DopaminePhase.MONITORING
        self.current_state = DopamineState.BALANCED
        self.current_response = DopamineResponse(
            signal=0.0, phase=DopaminePhase.MONITORING, state=DopamineState.BALANCED,
            anticipation_factor=0.0, satisfaction_factor=0.0, tolerance_level=self.tolerance_level,
            addiction_risk=self.addiction_score, withdrawal_intensity=self.withdrawal_intensity,
            position_size_modifier=1.0, risk_tolerance_modifier=1.0, urgency_factor=0.5
        )
        
        # Conditionally reset learning-related state
        if not preserve_learning:
            self.tolerance_level = 0.5
            self.addiction_score = 0.0
            self.withdrawal_intensity = 0.0
            self.expected_pnl = 0.0
            self.expectation_confidence = 0.5
            self.current_sensitivity = self.base_sensitivity
        else:
            # Gradual decay of psychological factors for new session
            self.tolerance_level *= 0.8
            self.addiction_score *= 0.7
            self.withdrawal_intensity *= 0.6
    
    def force_withdrawal_state(self):
        """Force system into withdrawal state (for testing or therapeutic reset)"""
        self.withdrawal_intensity = 0.8
        self.tolerance_level = max(self.tolerance_level, 0.6)
        self.addiction_score *= 0.5
        self.urgency_factor = 0.8
    
    def therapeutic_reset(self):
        """Therapeutic reset to break addiction cycles"""
        logger.info("DOPAMINE: Performing therapeutic reset to break addiction patterns")
        
        self.tolerance_level = 0.3
        self.addiction_score = 0.0
        self.withdrawal_intensity = 0.2  # Mild withdrawal for learning
        self.current_sensitivity = self.base_sensitivity
        self.urgency_factor = 0.3  # Force patience
        
        # Clear recent high patterns
        self.recent_high_count = 0
        self.session_peak_dopamine = 0.0

    def get_enhanced_performance_metrics(self) -> Dict:
        """Get comprehensive dopamine system performance metrics"""
        
        if not self.pnl_history or not self.response_history:
            return {'status': 'insufficient_data'}
        
        # Extract values
        pnl_values = [s.unrealized_pnl for s in self.pnl_history]
        signal_values = [r.signal for r in self.response_history]
        
        # P&L metrics
        total_pnl_change = pnl_values[-1] - pnl_values[0] if len(pnl_values) > 1 else 0.0
        max_pnl = max(pnl_values)
        min_pnl = min(pnl_values)
        pnl_range = max_pnl - min_pnl
        
        # Enhanced signal analysis
        positive_signals = sum(1 for s in signal_values if s > 0.1)
        negative_signals = sum(1 for s in signal_values if s < -0.1)
        neutral_signals = len(signal_values) - positive_signals - negative_signals
        euphoric_signals = sum(1 for s in signal_values if s > 1.0)
        despair_signals = sum(1 for s in signal_values if s < -1.0)
        
        # Psychological health metrics
        avg_tolerance = np.mean([getattr(r, 'tolerance_level', self.tolerance_level) for r in self.response_history[-20:]])
        max_addiction = max([getattr(r, 'addiction_risk', 0) for r in self.response_history], default=0)
        avg_withdrawal = np.mean([getattr(r, 'withdrawal_intensity', 0) for r in self.response_history[-10:]])
        
        # Behavioral analysis
        avg_position_modifier = np.mean([getattr(r, 'position_size_modifier', 1.0) for r in self.response_history[-10:]])
        avg_risk_modifier = np.mean([getattr(r, 'risk_tolerance_modifier', 1.0) for r in self.response_history[-10:]])
        avg_urgency = np.mean([getattr(r, 'urgency_factor', 0.5) for r in self.response_history[-10:]])
        
        # Phase distribution
        phases = [r.phase.value for r in self.response_history]
        phase_counts = {phase.value: phases.count(phase.value) for phase in DopaminePhase}
        
        # State distribution
        states = [r.state.value for r in self.response_history]
        state_counts = {state.value: states.count(state.value) for state in DopamineState}
        
        return {
            # Basic metrics
            'total_updates': len(self.pnl_history),
            'total_pnl_change': total_pnl_change,
            'max_pnl': max_pnl,
            'min_pnl': min_pnl,
            'pnl_range': pnl_range,
            
            # Signal distribution
            'positive_signals': positive_signals,
            'negative_signals': negative_signals,
            'neutral_signals': neutral_signals,
            'euphoric_signals': euphoric_signals,
            'despair_signals': despair_signals,
            'avg_signal_strength': np.mean([abs(s) for s in signal_values]),
            'signal_volatility': np.std(signal_values),
            
            # Psychological health
            'current_tolerance_level': self.tolerance_level,
            'current_addiction_score': self.addiction_score,
            'current_withdrawal_intensity': self.withdrawal_intensity,
            'avg_tolerance_level': avg_tolerance,
            'max_addiction_score': max_addiction,
            'avg_withdrawal_intensity': avg_withdrawal,
            
            # Behavioral patterns
            'avg_position_size_modifier': avg_position_modifier,
            'avg_risk_tolerance_modifier': avg_risk_modifier,
            'avg_urgency_factor': avg_urgency,
            'max_consecutive_positive': self.consecutive_positive,
            'max_consecutive_negative': self.consecutive_negative,
            
            # Phase and state analysis
            'phase_distribution': phase_counts,
            'state_distribution': state_counts,
            'total_phase_transitions': len(self.phase_transitions),
            
            # Session metrics
            'session_peak_dopamine': self.session_peak_dopamine,
            'session_trough_dopamine': self.session_trough_dopamine,
            'time_since_last_high': self.time_since_last_high,
            'recent_high_count': self.recent_high_count,
            
            # Health assessment
            'psychological_health_score': self._calculate_psychological_health(),
            'addiction_risk_level': self._assess_addiction_risk(),
            'withdrawal_risk_level': self._assess_withdrawal_risk()
        }
    
    def _calculate_psychological_health(self) -> float:
        """Calculate overall psychological health score (0-1)"""
        
        health_score = 1.0
        
        # Penalize high tolerance
        health_score -= self.tolerance_level * 0.3
        
        # Penalize addiction
        health_score -= self.addiction_score * 0.4
        
        # Penalize withdrawal
        health_score -= self.withdrawal_intensity * 0.3
        
        # Bonus for balanced state
        if self.current_state == DopamineState.BALANCED:
            health_score += 0.1
        
        return np.clip(health_score, 0.0, 1.0)
    
    def _assess_addiction_risk(self) -> str:
        """Assess addiction risk level"""
        
        if self.addiction_score > 0.8:
            return "severe"
        elif self.addiction_score > 0.6:
            return "high"
        elif self.addiction_score > 0.4:
            return "moderate"
        elif self.addiction_score > 0.2:
            return "mild"
        else:
            return "low"
    
    def _assess_withdrawal_risk(self) -> str:
        """Assess withdrawal risk level"""
        
        if self.withdrawal_intensity > 0.7:
            return "severe"
        elif self.withdrawal_intensity > 0.5:
            return "high"
        elif self.withdrawal_intensity > 0.3:
            return "moderate"
        elif self.withdrawal_intensity > 0.1:
            return "mild"
        else:
            return "none"

    def learn_from_outcome(self, outcome: float, context: Optional[Dict] = None):
        """
        Learn from trading outcome by updating dopamine response patterns
        
        Args:
            outcome: Trade outcome (positive for profit, negative for loss)
            context: Optional context containing trade and market information
        """
        try:
            # Normalize outcome to expected range [-1.0, 1.0]
            normalized_outcome = np.tanh(outcome * 2.0)
            
            # Create synthetic P&L change if we have context
            if context and 'trade_data' in context:
                trade_data = context['trade_data']
                pnl_change = getattr(trade_data, 'pnl', outcome)
            else:
                # Synthetic P&L change based on outcome
                pnl_change = outcome * 100  # Convert to dollar-like units
            
            # Create learning snapshot with realistic data
            learning_snapshot = DopamineSnapshot(
                timestamp=time.time(),
                phase=DopaminePhase.REALIZATION,  # Learning happens at trade realization
                unrealized_pnl=0.0,  # Trade is closed
                realized_pnl=pnl_change,
                position_size=0.0,  # Position closed
                current_price=0.0,  # Not relevant for learning
                trade_duration=0.0,
                expected_outcome=self.expected_pnl,
                confidence_level=self.expectation_confidence
            )
            
            # Generate dopamine response for this outcome
            learning_response = self._calculate_realization_response(learning_snapshot)
            
            # Update internal learning parameters based on outcome
            self._update_learning_from_outcome(normalized_outcome, learning_response)
            
            # Store learning experience
            self.response_history.append(learning_response)
            self.pnl_history.append(learning_snapshot)
            
            # Update expectations based on actual outcome
            if hasattr(self, '_learning_outcomes'):
                self._learning_outcomes.append(normalized_outcome)
            else:
                self._learning_outcomes = [normalized_outcome]
            
            # Keep only recent learning outcomes
            if len(self._learning_outcomes) > 50:
                self._learning_outcomes = self._learning_outcomes[-50:]
            
            # Update expectation confidence based on prediction accuracy
            if self.expected_pnl != 0:
                prediction_error = abs(pnl_change - self.expected_pnl) / max(abs(self.expected_pnl), 1.0)
                confidence_adjustment = -0.1 * prediction_error  # Reduce confidence for bad predictions
                self.expectation_confidence = np.clip(
                    self.expectation_confidence + confidence_adjustment, 0.1, 1.0
                )
            
        except Exception as e:
            logger.error(f"Error in dopamine learning from outcome: {e}")
    
    def _update_learning_from_outcome(self, outcome: float, response: DopamineResponse):
        """Update internal parameters based on learning outcome with adaptive adjustments"""
        
        # Adaptive learning rate based on confidence and recent performance
        recent_accuracy = self._calculate_recent_prediction_accuracy()
        learning_rate = 0.05 + (0.15 * self.expectation_confidence * recent_accuracy)
        
        # Enhanced prediction accuracy with graded assessment
        prediction_error = self._calculate_prediction_error(outcome, response.signal)
        prediction_accuracy = 1.0 / (1.0 + prediction_error)  # 0-1 scale
        
        # Adaptive sensitivity adjustment based on prediction quality and market volatility
        volatility_factor = min(2.0, 1.0 + abs(outcome) * 2.0)  # Higher volatility = smaller adjustments
        base_adjustment = 0.01 + (0.02 * prediction_accuracy)  # 0.01-0.03 range
        sensitivity_adjustment = base_adjustment / volatility_factor
        
        if prediction_accuracy > 0.7:  # Good prediction
            self.current_sensitivity *= (1.0 + sensitivity_adjustment)
        elif prediction_accuracy < 0.3:  # Poor prediction
            self.current_sensitivity *= (1.0 - sensitivity_adjustment * 0.5)
        
        # Bound sensitivity to reasonable range
        self.current_sensitivity = np.clip(self.current_sensitivity, 0.05, 0.3)
        
        # Adaptive tolerance adjustment based on outcome magnitude and frequency
        outcome_magnitude = abs(outcome)
        magnitude_threshold = 0.3 + (0.4 * self.tolerance_level)  # Adaptive threshold
        
        if outcome > magnitude_threshold:  # Adaptive positive threshold
            tolerance_increase = self.tolerance_buildup_rate * (1.0 + outcome_magnitude)
            self.tolerance_level += tolerance_increase
        elif outcome < -magnitude_threshold:  # Adaptive negative threshold
            tolerance_decrease = self.tolerance_decay_rate * (1.0 + outcome_magnitude)
            self.tolerance_level -= tolerance_decrease
            # Adaptive withdrawal based on loss severity
            withdrawal_increase = 0.02 + (0.08 * outcome_magnitude)
            self.withdrawal_intensity += withdrawal_increase
        
        # Adaptive addiction patterns based on streak strength and outcome quality
        addiction_threshold = 0.2 + (0.3 * self.tolerance_level)  # Higher tolerance = higher threshold
        streak_factor = min(3.0, self.consecutive_positive / 2.0) if self.consecutive_positive > 0 else 1.0
        
        if outcome > addiction_threshold and self.consecutive_positive >= 2:
            # Addiction risk scales with streak strength and outcome magnitude
            addiction_increase = (0.01 + 0.04 * outcome_magnitude) * streak_factor
            self.addiction_score += addiction_increase
        elif outcome < -addiction_threshold:
            # Recovery scales with loss magnitude and current addiction level
            recovery_factor = 0.01 + (0.03 * outcome_magnitude * self.addiction_score)
            self.addiction_score -= recovery_factor
        
        # Bound all psychological parameters
        self.tolerance_level = np.clip(self.tolerance_level, 0.0, 1.0)
        self.addiction_score = np.clip(self.addiction_score, 0.0, 1.0)
        self.withdrawal_intensity = np.clip(self.withdrawal_intensity, 0.0, 1.0)
        
        # Update base sensitivity to match current sensitivity
        self.base_sensitivity = self.current_sensitivity
        
        # Store prediction accuracy for future adaptive adjustments
        if not hasattr(self, '_prediction_accuracy_history'):
            self._prediction_accuracy_history = deque(maxlen=20)
        self._prediction_accuracy_history.append(prediction_accuracy)
    
    def _calculate_prediction_error(self, actual_outcome: float, predicted_signal: float) -> float:
        """Calculate prediction error with consideration for magnitude and direction"""
        
        # Normalize both values to comparable scale
        normalized_actual = np.tanh(actual_outcome * 2.0)
        normalized_predicted = np.tanh(predicted_signal)
        
        # Direction error (most important)
        direction_error = 0.0
        if (normalized_actual > 0.1 and normalized_predicted < -0.1) or \
           (normalized_actual < -0.1 and normalized_predicted > 0.1):
            direction_error = 1.0  # Complete direction mismatch
        elif abs(normalized_actual) < 0.1 and abs(normalized_predicted) > 0.3:
            direction_error = 0.5  # Predicted movement when should be neutral
        
        # Magnitude error
        magnitude_error = abs(normalized_actual - normalized_predicted)
        
        # Combined error (direction weighted more heavily)
        total_error = direction_error * 0.7 + magnitude_error * 0.3
        
        return total_error
    
    def _calculate_recent_prediction_accuracy(self) -> float:
        """Calculate recent prediction accuracy for adaptive learning"""
        
        if not hasattr(self, '_prediction_accuracy_history') or not self._prediction_accuracy_history:
            return 0.5  # Neutral starting point
        
        recent_accuracies = list(self._prediction_accuracy_history)[-10:]  # Last 10 predictions
        return np.mean(recent_accuracies)


# ========================================
# BACKWARD COMPATIBILITY ALIASES
# ========================================

# Main class aliases for backward compatibility
DopamineSubsystem = ConsolidatedDopamineSubsystem
EnhancedDopamineSubsystem = ConsolidatedDopamineSubsystem

# Export the main interface for external use
__all__ = [
    'DopamineSubsystem',
    'EnhancedDopamineSubsystem', 
    'ConsolidatedDopamineSubsystem',
    'DopamineResponse',
    'DopaminePhase',
    'DopamineState',
    'DopamineSnapshot',
    'DopamineSignalProvider',
    'DopamineEventProcessor',
    'DopamineLearningInterface'
]