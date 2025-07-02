"""
Confidence Management System

Centralized confidence tracking, adjustment, and recovery for the trading agent.
Replaces scattered confidence logic with a clean, debuggable system.
"""

import time
import logging
from typing import Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ConfidenceEvent(Enum):
    """Types of events that can affect confidence"""
    NEURAL_OUTPUT = "neural_output"
    POSITION_REJECTION = "position_rejection"
    SUCCESSFUL_TRADE = "successful_trade"
    FAILED_TRADE = "failed_trade"
    MARKET_CHANGE = "market_change"
    SYSTEM_STARTUP = "system_startup"
    MANUAL_ADJUSTMENT = "manual_adjustment"

@dataclass
class ConfidenceAdjustment:
    """Record of a confidence adjustment"""
    timestamp: float
    event_type: ConfidenceEvent
    raw_confidence: float
    adjusted_confidence: float
    adjustment_reason: str
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class ConfidenceState:
    """Current state of the confidence system"""
    
    def __init__(self):
        self.current_confidence: float = 0.6  # Default starting confidence
        self.raw_neural_confidence: float = 0.6
        self.base_confidence: float = 0.6  # Long-term baseline
        self.recovery_factor: float = 1.0
        
        # Event tracking
        self.recent_rejections: int = 0
        self.successful_trades: int = 0
        self.failed_trades: int = 0
        self.last_rejection_time: float = 0.0
        self.last_success_time: float = 0.0
        
        # History for analysis
        self.adjustment_history: deque = deque(maxlen=1000)
        self.violation_timestamps: deque = deque(maxlen=100)

class ConfidenceManager:
    """
    Centralized confidence management system
    
    Handles all confidence calculations, adjustments, and recovery mechanisms
    in a single, debuggable location.
    """
    
    def __init__(self, 
                 initial_confidence: float = 0.6,
                 min_confidence: float = 0.01,
                 max_confidence: float = 1.00,
                 debug_mode: bool = True):
        """
        Initialize confidence manager
        
        Args:
            initial_confidence: Starting confidence level
            min_confidence: Absolute minimum confidence (safety floor)
            max_confidence: Maximum allowed confidence
            debug_mode: Enable detailed logging
        """
        self.config = {
            'min_confidence': 0.01,
            'max_confidence': 1.00,
            'debug_mode': debug_mode,
            
            # Recovery parameters
            'base_recovery_rate': 0.02,  # Per minute recovery rate
            'max_recovery_time': 15.0,   # Minutes for full recovery
            'success_boost': 0.08,       # Boost from successful trades
            'rejection_penalty': 0.05,   # Per rejection penalty
            
            # Protection parameters
            'violation_protection_threshold': 3,  # Rejections before protection kicks in
            'protection_floor_base': 0.25,       # Base protection floor
            'protection_duration': 600,          # Protection duration in seconds
        }
        
        self.state = ConfidenceState()
        self.state.current_confidence = initial_confidence
        self.state.base_confidence = initial_confidence
        self.state.raw_neural_confidence = initial_confidence
        
        logger.info(f"ConfidenceManager initialized: min={min_confidence}, max={max_confidence}")
    
    def process_neural_output(self, raw_confidence: float, market_context: Dict = None) -> float:
        """
        Process raw neural network confidence output
        
        Args:
            raw_confidence: Raw confidence from neural network
            market_context: Optional market context for adjustments
            
        Returns:
            Adjusted confidence value
        """
        self.state.raw_neural_confidence = raw_confidence
        
        if self.config['debug_mode']:
            logger.info(f"CONFIDENCE: Neural output={raw_confidence:.4f}")
        
        # Start with raw confidence
        adjusted = raw_confidence
        adjustment_reasons = []
        
        # Apply recovery mechanisms
        adjusted, recovery_reasons = self._apply_recovery_adjustments(adjusted)
        adjustment_reasons.extend(recovery_reasons)
        
        # Apply protection mechanisms
        adjusted, protection_reasons = self._apply_protection_mechanisms(adjusted)
        adjustment_reasons.extend(protection_reasons)
        
        # Apply bounds
        adjusted = self._apply_bounds(adjusted)
        
        # Record the adjustment
        adjustment = ConfidenceAdjustment(
            timestamp=time.time(),
            event_type=ConfidenceEvent.NEURAL_OUTPUT,
            raw_confidence=raw_confidence,
            adjusted_confidence=adjusted,
            adjustment_reason="; ".join(adjustment_reasons) if adjustment_reasons else "No adjustments",
            metadata=market_context or {}
        )
        self.state.adjustment_history.append(adjustment)
        
        self.state.current_confidence = adjusted
        
        if self.config['debug_mode'] and abs(adjusted - raw_confidence) > 0.01:
            logger.info(f"CONFIDENCE: Adjusted {raw_confidence:.4f} -> {adjusted:.4f} ({adjustment.adjustment_reason})")
        
        return adjusted
    
    def handle_position_rejection(self, rejection_context: Dict = None) -> None:
        """
        Handle position rejection event
        
        Args:
            rejection_context: Context about the rejection
        """
        current_time = time.time()
        
        # Update rejection tracking
        self.state.recent_rejections += 1
        self.state.last_rejection_time = current_time
        self.state.violation_timestamps.append(current_time)
        
        # Clean old violations (older than protection duration)
        cutoff_time = current_time - self.config['protection_duration']
        while (self.state.violation_timestamps and 
               self.state.violation_timestamps[0] < cutoff_time):
            self.state.violation_timestamps.popleft()
        
        # Apply rejection penalty to recovery factor
        penalty = self.config['rejection_penalty']
        self.state.recovery_factor = max(0.5, self.state.recovery_factor - penalty)
        
        # Record the event
        adjustment = ConfidenceAdjustment(
            timestamp=current_time,
            event_type=ConfidenceEvent.POSITION_REJECTION,
            raw_confidence=self.state.current_confidence,
            adjusted_confidence=self.state.current_confidence,  # No immediate adjustment
            adjustment_reason=f"Rejection penalty applied to recovery factor: -{penalty:.3f}",
            metadata=rejection_context or {}
        )
        self.state.adjustment_history.append(adjustment)
        
        if self.config['debug_mode']:
            violation_count = len(self.state.violation_timestamps)
            logger.warning(f"CONFIDENCE: Position rejection #{self.state.recent_rejections}, "
                          f"recent violations: {violation_count}, recovery_factor: {self.state.recovery_factor:.3f}")
    
    def handle_trade_outcome(self, pnl: float, trade_context: Dict = None, dopamine_response = None) -> None:
        """
        Handle completed trade outcome with dopamine integration
        
        Args:
            pnl: Profit/loss from the trade
            trade_context: Context about the trade
            dopamine_response: DopamineResponse object from the dopamine subsystem
        """
        current_time = time.time()
        
        # Base confidence adjustment
        base_adjustment = 0.0
        
        if pnl > 0:
            # Successful trade
            self.state.successful_trades += 1
            self.state.last_success_time = current_time
            base_adjustment = 0.02
            
            # Boost recovery factor
            boost = 0.02
            self.state.recovery_factor = min(1.2, self.state.recovery_factor + boost)
            
            event_type = ConfidenceEvent.SUCCESSFUL_TRADE
            reason = f"Successful trade: +{boost:.3f} to recovery factor"
            
        else:
            # Failed trade
            self.state.failed_trades += 1
            base_adjustment = -0.015
            
            # Small penalty to recovery factor
            penalty = 0.01
            self.state.recovery_factor = max(0.7, self.state.recovery_factor - penalty)
            
            event_type = ConfidenceEvent.FAILED_TRADE
            reason = f"Failed trade: -{penalty:.3f} to recovery factor"
        
        # DOPAMINE INTEGRATION: Modify confidence based on dopamine state
        if dopamine_response:
            original_adjustment = base_adjustment
            
            # Addicted traders get overconfident after wins
            if hasattr(dopamine_response, 'state') and dopamine_response.state.value == 'addicted' and pnl > 0:
                base_adjustment *= 1.5  # Amplify confidence boost
                reason += f" [DOPAMINE: Addiction amplification x1.5]"
            
            # Withdrawn traders get less confidence boost from wins
            elif hasattr(dopamine_response, 'state') and dopamine_response.state.value == 'withdrawn':
                if pnl > 0:
                    base_adjustment *= 0.5  # Reduced confidence boost
                    reason += f" [DOPAMINE: Withdrawal dampening x0.5]"
                else:
                    base_adjustment *= 1.3  # Amplified confidence loss
                    reason += f" [DOPAMINE: Withdrawal amplification x1.3]"
            
            # Euphoric state should be dampened
            elif hasattr(dopamine_response, 'state') and dopamine_response.state.value == 'euphoric':
                base_adjustment *= 0.7  # Moderate the euphoria
                reason += f" [DOPAMINE: Euphoria moderation x0.7]"
            
            # High tolerance means less impact from individual trades
            if hasattr(dopamine_response, 'tolerance_level'):
                tolerance_factor = 1.0 - (dopamine_response.tolerance_level * 0.3)
                base_adjustment *= tolerance_factor
                reason += f" [DOPAMINE: Tolerance factor x{tolerance_factor:.2f}]"
            
            if self.config['debug_mode'] and base_adjustment != original_adjustment:
                logger.debug(f"CONFIDENCE: Dopamine modified adjustment from {original_adjustment:.4f} to {base_adjustment:.4f}")
        
        # Apply confidence adjustment if there is one
        if base_adjustment != 0.0:
            old_confidence = self.state.current_confidence
            new_confidence = self._apply_bounds(old_confidence + base_adjustment)
            
            adjustment = ConfidenceAdjustment(
                timestamp=current_time,
                event_type=event_type,
                raw_confidence=old_confidence,
                adjusted_confidence=new_confidence,
                adjustment_reason=reason,
                metadata=trade_context or {}
            )
            
            self.state.current_confidence = new_confidence
            self.state.adjustment_history.append(adjustment)
        
        # Record the event (only if we haven't already recorded an adjustment)
        if base_adjustment == 0.0:
            adjustment = ConfidenceAdjustment(
                timestamp=current_time,
                event_type=event_type,
                raw_confidence=self.state.current_confidence,
                adjusted_confidence=self.state.current_confidence,
                adjustment_reason=reason,
                metadata={'pnl': pnl, **(trade_context or {})}
            )
            self.state.adjustment_history.append(adjustment)
        
        if self.config['debug_mode']:
            logger.info(f"CONFIDENCE: Trade outcome PnL={pnl:.2f}, recovery_factor={self.state.recovery_factor:.3f}")
    
    def _apply_recovery_adjustments(self, confidence: float) -> Tuple[float, List[str]]:
        """Apply time-based and success-based recovery adjustments"""
        adjusted = confidence
        reasons = []
        current_time = time.time()
        
        # Time-based recovery from rejections
        if self.state.recent_rejections > 0 and self.state.last_rejection_time > 0:
            time_since_rejection = current_time - self.state.last_rejection_time
            recovery_minutes = time_since_rejection / 60.0
            
            # Gradual recovery based on configured rate
            max_recovery = self.config['base_recovery_rate'] * recovery_minutes
            recovery_boost = min(0.3, max_recovery) * self.state.recovery_factor
            
            if recovery_boost > 0.005:  # Only apply meaningful boosts
                adjusted = min(self.config['max_confidence'], adjusted + recovery_boost)
                reasons.append(f"Time recovery: +{recovery_boost:.3f} ({recovery_minutes:.1f}min)")
        
        # Success-based recovery boost
        if self.state.last_success_time > 0:
            time_since_success = current_time - self.state.last_success_time
            if time_since_success < 300:  # Within 5 minutes
                success_factor = 1.0 - (time_since_success / 300.0)
                success_boost = self.config['success_boost'] * success_factor
                
                if success_boost > 0.01:
                    adjusted = min(self.config['max_confidence'], adjusted + success_boost)
                    reasons.append(f"Success boost: +{success_boost:.3f}")
        
        return adjusted, reasons
    
    def _apply_protection_mechanisms(self, confidence: float) -> Tuple[float, List[str]]:
        """Apply protection mechanisms during high violation periods"""
        reasons = []
        current_time = time.time()
        
        # Count recent violations
        recent_violations = len(self.state.violation_timestamps)
        
        if recent_violations >= self.config['violation_protection_threshold']:
            # Calculate adaptive protection floor
            violation_factor = recent_violations - self.config['violation_protection_threshold'] + 1
            protection_floor = self.config['protection_floor_base'] - (0.02 * (violation_factor - 1))
            protection_floor = max(self.config['min_confidence'], protection_floor)
            
            if confidence < protection_floor:
                confidence = protection_floor
                reasons.append(f"Violation protection: floor={protection_floor:.3f} (violations={recent_violations})")
        
        return confidence, reasons
    
    def _apply_bounds(self, confidence: float) -> float:
        """Apply hard bounds to confidence"""
        return max(self.config['min_confidence'], 
                  min(self.config['max_confidence'], confidence))
    
    def get_current_confidence(self) -> float:
        """Get current confidence value"""
        return self.state.current_confidence
    
    def get_confidence_health(self) -> Dict:
        """Get comprehensive confidence health status"""
        current_time = time.time()
        recent_violations = len(self.state.violation_timestamps)
        
        # Calculate health score
        health_factors = []
        
        # Base confidence health
        base_health = min(1.0, self.state.current_confidence / 0.6)
        health_factors.append(('base_confidence', base_health, 0.4))
        
        # Recovery factor health
        recovery_health = min(1.0, self.state.recovery_factor / 1.0)
        health_factors.append(('recovery_factor', recovery_health, 0.3))
        
        # Violation health (inverse of recent violations)
        violation_health = max(0.0, 1.0 - (recent_violations / 10.0))
        health_factors.append(('violations', violation_health, 0.3))
        
        # Calculate weighted health score
        health_score = sum(score * weight for _, score, weight in health_factors)
        
        # Determine status
        if health_score >= 0.8:
            status = "EXCELLENT"
        elif health_score >= 0.6:
            status = "GOOD"
        elif health_score >= 0.4:
            status = "FAIR"
        elif health_score >= 0.2:
            status = "POOR"
        else:
            status = "CRITICAL"
        
        return {
            'status': status,
            'health_score': health_score,
            'current_confidence': self.state.current_confidence,
            'raw_neural_confidence': self.state.raw_neural_confidence,
            'recovery_factor': self.state.recovery_factor,
            'recent_violations': recent_violations,
            'successful_trades': self.state.successful_trades,
            'failed_trades': self.state.failed_trades,
            'time_since_last_rejection': (current_time - self.state.last_rejection_time) / 60.0 if self.state.last_rejection_time > 0 else 0,
            'health_factors': dict((name, score) for name, score, _ in health_factors)
        }
    
    def get_debug_info(self) -> Dict:
        """Get detailed debug information"""
        return {
            'state': {
                'current_confidence': self.state.current_confidence,
                'raw_neural_confidence': self.state.raw_neural_confidence,
                'base_confidence': self.state.base_confidence,
                'recovery_factor': self.state.recovery_factor,
                'recent_rejections': self.state.recent_rejections,
                'successful_trades': self.state.successful_trades,
                'failed_trades': self.state.failed_trades,
            },
            'config': self.config,
            'recent_adjustments': list(self.state.adjustment_history)[-10:],  # Last 10 adjustments
            'violation_history': list(self.state.violation_timestamps)
        }
    
    
    def adjust_base_confidence(self, new_base: float, reason: str = "Manual adjustment") -> None:
        """Manually adjust base confidence level"""
        old_base = self.state.base_confidence
        self.state.base_confidence = self._apply_bounds(new_base)
        
        adjustment = ConfidenceAdjustment(
            timestamp=time.time(),
            event_type=ConfidenceEvent.MANUAL_ADJUSTMENT,
            raw_confidence=old_base,
            adjusted_confidence=self.state.base_confidence,
            adjustment_reason=reason,
            metadata={'old_base': old_base, 'new_base': new_base}
        )
        self.state.adjustment_history.append(adjustment)
        
        logger.info(f"CONFIDENCE: Base confidence adjusted: {old_base:.3f} -> {self.state.base_confidence:.3f} ({reason})")