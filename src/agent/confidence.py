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
    """Enhanced state of the confidence system with performance tracking"""
    
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
        
        # NEW: Performance tracking
        self.trade_history: deque = deque(maxlen=200)  # Recent trade outcomes
        self.strategy_performance: Dict = {}  # Per-strategy performance
        self.current_strategy: str = 'conservative'
        
        # NEW: Market awareness
        self.last_unrealized_pnl: float = 0.0
        self.last_market_volatility: float = 0.02
        self.subsystem_consensus_history: deque = deque(maxlen=50)

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
            
            # Enhanced Recovery parameters for faster scaling
            'base_recovery_rate': 0.015,  # Increased for faster recovery
            'max_recovery_time': 30.0,    # Faster recovery cycles
            'success_boost': 0.06,        # Increased for better confidence building
            'rejection_penalty': 0.03,    # Reduced to be less punitive
            'failed_trade_penalty': 0.03, # Reduced to allow faster recovery
            
            # Market Reality parameters (NEW)
            'drawdown_sensitivity': 0.3,   # Impact of unrealized losses
            'volatility_sensitivity': 3.0, # Market volatility impact factor
            'consensus_weight': 0.5,       # Subsystem agreement influence
            
            # Protection parameters  
            'violation_protection_threshold': 3,
            'protection_floor_base': 0.25,
            'protection_duration': 600,
            
            # Performance tracking (NEW)
            'strategy_memory_length': 50,  # Trades to remember per strategy
            'performance_window': 20,      # Recent trades for performance calc
        }
        
        self.state = ConfidenceState()
        self.state.current_confidence = initial_confidence
        self.state.base_confidence = initial_confidence
        self.state.raw_neural_confidence = initial_confidence
        
        logger.info(f"ConfidenceManager initialized: min={min_confidence}, max={max_confidence}")
    
    def process_neural_output(self, raw_confidence: float, market_context: Dict = None, market_data = None) -> float:
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
        
        # Apply market reality check (NEW)
        if market_data:
            adjusted, reality_reasons = self._apply_market_reality_check(adjusted, market_data)
            adjustment_reasons.extend(reality_reasons)
        
        # Apply intelligent recovery mechanisms (ENHANCED)
        adjusted, recovery_reasons = self._apply_intelligent_recovery(adjusted)
        adjustment_reasons.extend(recovery_reasons)
        
        # Apply strategy performance (NEW)
        adjusted, strategy_reasons = self._apply_strategy_performance(adjusted)
        adjustment_reasons.extend(strategy_reasons)
        
        # Apply subsystem consensus (NEW)
        if market_context and 'subsystem_signals' in market_context:
            adjusted, consensus_reasons = self._apply_subsystem_consensus(adjusted, market_context['subsystem_signals'])
            adjustment_reasons.extend(consensus_reasons)
        
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
            base_adjustment = self.config['success_boost']  # Now 0.04 instead of 0.02
            
            # Boost recovery factor
            boost = 0.02
            self.state.recovery_factor = min(1.2, self.state.recovery_factor + boost)
            
            event_type = ConfidenceEvent.SUCCESSFUL_TRADE
            reason = f"Successful trade: +{boost:.3f} to recovery factor"
            
        else:
            # Failed trade
            self.state.failed_trades += 1
            base_adjustment = -self.config['failed_trade_penalty']  # Now -0.05 instead of -0.015
            
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
        
        # NEW: Track trade history for performance-based recovery
        self.state.trade_history.append({
            'pnl': pnl,
            'timestamp': current_time,
            'strategy': self.state.current_strategy,
            'confidence': self.state.current_confidence
        })
        
        # NEW: Update strategy performance tracking
        self._update_strategy_performance(pnl, trade_context)
        
        if self.config['debug_mode']:
            logger.info(f"CONFIDENCE: Trade outcome PnL={pnl:.2f}, recovery_factor={self.state.recovery_factor:.3f}")
    
    def _apply_market_reality_check(self, confidence: float, market_data) -> Tuple[float, List[str]]:
        """Apply real-time market awareness to confidence"""
        adjusted = confidence
        reasons = []
        
        try:
            # LEARNING PHASE PROTECTION: Prevent over-dampening during recovery
            learning_floor = 0.35 if self.state.failed_trades > self.state.successful_trades else 0.25
            
            # 1. Unrealized P&L impact (gradual, not panic-inducing)
            if hasattr(market_data, 'unrealized_pnl') and hasattr(market_data, 'position_size'):
                if market_data.unrealized_pnl < 0 and market_data.position_size != 0:
                    account_balance = getattr(market_data, 'account_balance', 25000)
                    drawdown_pct = abs(market_data.unrealized_pnl) / account_balance
                    
                    if drawdown_pct > 0.01:  # 1%+ drawdown
                        drawdown_factor = 1.0 - (drawdown_pct * self.config['drawdown_sensitivity'])
                        old_adjusted = adjusted
                        adjusted *= max(learning_floor, drawdown_factor)  # Use learning floor instead of 0.85
                        
                        if adjusted != old_adjusted:
                            reduction = old_adjusted - adjusted
                            reasons.append(f"Drawdown reality check: -{reduction:.3f} ({drawdown_pct:.1%} unrealized loss, floor={learning_floor:.2f})")
                
                # Update state tracking
                self.state.last_unrealized_pnl = market_data.unrealized_pnl
            
            # 2. Market volatility adjustment with learning floor
            volatility = getattr(market_data, 'volatility', 0.02)
            if volatility > 0.04:  # High volatility
                vol_factor = 1.0 - ((volatility - 0.04) * self.config['volatility_sensitivity'])
                old_adjusted = adjusted
                adjusted *= max(learning_floor, vol_factor)  # Use same learning floor
                
                if adjusted != old_adjusted:
                    reduction = old_adjusted - adjusted
                    reasons.append(f"High volatility adjustment: -{reduction:.3f} (vol={volatility:.3f}, floor={learning_floor:.2f})")
            
            # Update state tracking
            self.state.last_market_volatility = volatility
            
        except Exception as e:
            if self.config['debug_mode']:
                logger.warning(f"Error in market reality check: {e}")
        
        return adjusted, reasons
    
    def _apply_intelligent_recovery(self, confidence: float) -> Tuple[float, List[str]]:
        """Performance-based recovery instead of pure time-based"""
        adjusted = confidence
        reasons = []
        current_time = time.time()
        
        # Get recent performance metrics
        recent_trades = list(self.state.trade_history)[-self.config['performance_window']:]
        
        # Always define base_rate first
        base_rate = self.config['base_recovery_rate']
        
        if len(recent_trades) >= 5:  # Enough data for intelligent recovery
            win_rate = sum(1 for trade in recent_trades if trade.get('pnl', 0) > 0) / len(recent_trades)
            avg_profit_factor = self._calculate_profit_factor(recent_trades)
            
            if win_rate > 0.7 and avg_profit_factor > 1.5:
                recovery_multiplier = 3.0  # Much faster recovery when crushing it
                performance_desc = "excellent"
            elif win_rate > 0.6:
                recovery_multiplier = 2.0  # Fast recovery for good performance
                performance_desc = "very good"
            elif win_rate > 0.5:
                recovery_multiplier = 1.5  # Enhanced normal recovery
                performance_desc = "good"
            elif win_rate < 0.3:
                recovery_multiplier = 0.6  # Less punitive for poor performance
                performance_desc = "poor"
            else:
                recovery_multiplier = 1.0  # Neutral recovery
                performance_desc = "average"
        else:
            # Fall back to original time-based recovery with reduced rate
            recovery_multiplier = 1.0
            performance_desc = "insufficient data"
        
        # Time-based recovery from rejections
        if self.state.recent_rejections > 0 and self.state.last_rejection_time > 0:
            time_since_rejection = current_time - self.state.last_rejection_time
            recovery_minutes = time_since_rejection / 60.0
            
            # Apply intelligent recovery rate with higher limits
            max_recovery = base_rate * recovery_minutes * recovery_multiplier
            recovery_boost = min(0.4, max_recovery) * self.state.recovery_factor  # Increased max for faster scaling
            
            if recovery_boost > 0.001:  # Lower threshold for more frequent boosts
                adjusted = min(self.config['max_confidence'], adjusted + recovery_boost)
                reasons.append(f"Smart recovery: +{recovery_boost:.3f} ({performance_desc} performance, {recovery_minutes:.1f}min)")
        
        return adjusted, reasons
    
    def _apply_strategy_performance(self, confidence: float) -> Tuple[float, List[str]]:
        """Apply strategy-specific confidence adjustments"""
        reasons = []
        
        if self.state.current_strategy not in self.state.strategy_performance:
            return confidence, reasons
        
        strategy_stats = self.state.strategy_performance[self.state.current_strategy]
        
        if strategy_stats.get('trades', 0) > 5:  # Reduced requirement for faster scaling
            strategy_win_rate = strategy_stats.get('win_rate', 0.5)
            # Enhanced scaling: 0.6 to 1.6 range for better rewards
            strategy_confidence_multiplier = 0.6 + (strategy_win_rate * 1.0)
            
            old_confidence = confidence
            confidence *= strategy_confidence_multiplier
            
            if abs(confidence - old_confidence) > 0.01:
                change = confidence - old_confidence
                reasons.append(f"Strategy performance: {change:+.3f} ({self.state.current_strategy} win rate: {strategy_win_rate:.1%})")
        
        return confidence, reasons
    
    def _apply_subsystem_consensus(self, confidence: float, subsystem_signals: Dict) -> Tuple[float, List[str]]:
        """Apply subsystem consensus weighting to confidence"""
        reasons = []
        
        try:
            # Extract signals
            signals = [
                subsystem_signals.get('dna_signal', 0),
                subsystem_signals.get('temporal_signal', 0), 
                subsystem_signals.get('immune_signal', 0),
                subsystem_signals.get('microstructure_signal', 0),
                subsystem_signals.get('dopamine_signal', 0)
            ]
            
            # Calculate consensus strength
            import numpy as np
            signal_std = np.std(signals)
            consensus_factor = 1.0 - (signal_std * self.config['consensus_weight'])
            consensus_factor = max(0.7, min(1.3, consensus_factor))  # Bounded impact
            
            # Immune system warning override
            immune_signal = subsystem_signals.get('immune_signal', 0)
            if immune_signal < -0.5:
                consensus_factor *= 0.8  # Additional reduction for strong immune warnings
                reasons.append(f"Immune warning: {immune_signal:.2f}")
            
            old_confidence = confidence
            confidence *= consensus_factor
            
            if abs(confidence - old_confidence) > 0.005:
                change = confidence - old_confidence
                reasons.append(f"Subsystem consensus: {change:+.3f} (std={signal_std:.2f})")
            
            # Track consensus history
            self.state.subsystem_consensus_history.append({
                'signals': signals,
                'std': signal_std,
                'factor': consensus_factor,
                'timestamp': time.time()
            })
            
        except Exception as e:
            if self.config['debug_mode']:
                logger.warning(f"Error in subsystem consensus: {e}")
        
        return confidence, reasons
    
    def _calculate_profit_factor(self, trades: list) -> float:
        """Calculate profit factor from trade history"""
        if not trades:
            return 1.0
        
        gross_profit = sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) > 0)
        gross_loss = abs(sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) < 0))
        
        if gross_loss == 0:
            return 2.0 if gross_profit > 0 else 1.0
        
        return gross_profit / gross_loss
    
    def _update_strategy_performance(self, pnl: float, trade_context: Dict = None):
        """Update strategy-specific performance tracking"""
        strategy = self.state.current_strategy
        
        if strategy not in self.state.strategy_performance:
            self.state.strategy_performance[strategy] = {
                'trades': 0,
                'wins': 0,
                'total_pnl': 0.0,
                'win_rate': 0.5,
                'recent_trades': deque(maxlen=self.config['strategy_memory_length'])
            }
        
        stats = self.state.strategy_performance[strategy]
        stats['trades'] += 1
        stats['total_pnl'] += pnl
        
        if pnl > 0:
            stats['wins'] += 1
        
        stats['recent_trades'].append({
            'pnl': pnl,
            'timestamp': time.time()
        })
        
        # Update win rate
        if stats['trades'] > 0:
            stats['win_rate'] = stats['wins'] / stats['trades']
        
        if self.config['debug_mode']:
            logger.debug(f"Strategy {strategy} performance: {stats['win_rate']:.1%} win rate over {stats['trades']} trades")
    
    def update_current_strategy(self, new_strategy: str):
        """Update the current trading strategy"""
        if new_strategy != self.state.current_strategy:
            old_strategy = self.state.current_strategy
            self.state.current_strategy = new_strategy
            
            if self.config['debug_mode']:
                logger.info(f"CONFIDENCE: Strategy changed from {old_strategy} to {new_strategy}")

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