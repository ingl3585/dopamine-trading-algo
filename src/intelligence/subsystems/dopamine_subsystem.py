"""
Dopamine Subsystem - Real-time P&L-based reward system
Mimics neurological dopamine responses for immediate trading feedback
"""

import numpy as np
import time
from typing import Dict, List, Optional
from collections import deque
from dataclasses import dataclass

@dataclass
class PnLSnapshot:
    timestamp: float
    unrealized_pnl: float
    realized_pnl: float
    position_size: float
    current_price: float

class DopamineSubsystem:
    """
    Dopamine-inspired reward system that monitors unrealized P&L changes
    
    Core Concept:
    - Green P&L = Positive dopamine signal (reward)
    - Red P&L = Negative dopamine signal (punishment)  
    - Momentum amplifies signals (consecutive gains/losses)
    - Expectation errors trigger adaptation
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # P&L tracking history
        self.pnl_history = deque(maxlen=100)
        self.previous_unrealized_pnl = 0.0
        self.baseline_unrealized_pnl = 0.0
        
        # Dopamine calculation parameters
        self.sensitivity = config.get('dopamine_sensitivity', 0.1)
        self.momentum_factor = config.get('momentum_factor', 0.2)
        self.max_signal = config.get('max_dopamine_signal', 2.0)
        
        # Momentum tracking
        self.consecutive_positive = 0
        self.consecutive_negative = 0
        self.peak_pnl = 0.0
        self.trough_pnl = 0.0
        
        # Expectation tracking
        self.expected_pnl = 0.0
        self.expectation_confidence = 0.5
        
        # Signal output
        self.current_dopamine_signal = 0.0
        self.signal_history = deque(maxlen=50)
        
    def process_pnl_update(self, market_data: Dict) -> float:
        """
        Process real-time P&L update and generate dopamine signal
        
        Args:
            market_data: Dictionary containing unrealized_pnl, position_size, etc.
            
        Returns:
            float: Dopamine signal [-2.0 to +2.0]
        """
        try:
            current_time = time.time()
            unrealized_pnl = market_data.get('unrealized_pnl', 0.0)
            realized_pnl = market_data.get('daily_pnl', 0.0)
            position_size = market_data.get('open_positions', 0.0)
            current_price = market_data.get('current_price', 0.0)
            
            # Create P&L snapshot
            snapshot = PnLSnapshot(
                timestamp=current_time,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl,
                position_size=position_size,
                current_price=current_price
            )
            
            # Calculate dopamine signal
            dopamine_signal = self._calculate_dopamine_signal(snapshot)
            
            # Update tracking
            self._update_tracking(snapshot, dopamine_signal)
            
            return dopamine_signal
            
        except Exception as e:
            print(f"Error in dopamine P&L processing: {e}")
            return 0.0
    
    def _calculate_dopamine_signal(self, snapshot: PnLSnapshot) -> float:
        """Calculate dopamine signal based on P&L changes"""
        
        # Calculate P&L change from previous update
        pnl_change = snapshot.unrealized_pnl - self.previous_unrealized_pnl
        
        # Base reward calculation using tanh for bounded output
        base_reward = np.tanh(pnl_change * self.sensitivity)
        
        # Momentum multiplier based on consecutive direction
        momentum_multiplier = 1.0
        if pnl_change > 0:
            momentum_multiplier = 1.0 + (self.consecutive_positive * self.momentum_factor)
            self.consecutive_positive += 1
            self.consecutive_negative = 0
        elif pnl_change < 0:
            momentum_multiplier = 1.0 + (self.consecutive_negative * self.momentum_factor)
            self.consecutive_negative += 1
            self.consecutive_positive = 0
        else:
            # No change - reset momentum
            self.consecutive_positive = 0
            self.consecutive_negative = 0
        
        # Expectation error component
        expectation_error = 0.0
        if self.expectation_confidence > 0.3:
            actual_vs_expected = snapshot.unrealized_pnl - self.expected_pnl
            expectation_error = np.tanh(actual_vs_expected * 0.05) * 0.3
        
        # Peak/trough analysis for additional context
        peak_trough_factor = self._calculate_peak_trough_factor(snapshot.unrealized_pnl)
        
        # Combine all factors
        raw_signal = (base_reward * momentum_multiplier) + expectation_error + peak_trough_factor
        
        # Bound the signal to [-max_signal, +max_signal]
        dopamine_signal = np.clip(raw_signal, -self.max_signal, self.max_signal)
        
        return dopamine_signal
    
    def _calculate_peak_trough_factor(self, current_pnl: float) -> float:
        """Calculate additional signal based on peak/trough analysis"""
        
        # Update peaks and troughs
        if current_pnl > self.peak_pnl:
            self.peak_pnl = current_pnl
            return 0.1  # Small positive boost for new highs
        elif current_pnl < self.trough_pnl:
            self.trough_pnl = current_pnl
            return -0.1  # Small negative signal for new lows
        
        # Drawdown from peak
        if self.peak_pnl > 0:
            drawdown_pct = (self.peak_pnl - current_pnl) / self.peak_pnl
            if drawdown_pct > 0.1:  # 10% drawdown threshold
                return -0.2 * drawdown_pct  # Negative signal proportional to drawdown
        
        return 0.0
    
    def _update_tracking(self, snapshot: PnLSnapshot, dopamine_signal: float):
        """Update internal tracking state"""
        
        # Add to history
        self.pnl_history.append(snapshot)
        self.signal_history.append(dopamine_signal)
        
        # Update previous P&L for next calculation
        self.previous_unrealized_pnl = snapshot.unrealized_pnl
        
        # Update current signal
        self.current_dopamine_signal = dopamine_signal
        
        # Update expectation (simple moving average of recent P&L)
        if len(self.pnl_history) >= 5:
            recent_pnls = [s.unrealized_pnl for s in list(self.pnl_history)[-5:]]
            self.expected_pnl = np.mean(recent_pnls)
            
            # Update expectation confidence based on P&L variance
            pnl_variance = np.var(recent_pnls)
            self.expectation_confidence = max(0.1, 1.0 / (1.0 + pnl_variance))
    
    def get_signal(self) -> float:
        """Get current dopamine signal"""
        return self.current_dopamine_signal
    
    def get_signal_with_context(self) -> Dict:
        """Get dopamine signal with full context"""
        
        # Calculate signal strength and direction
        signal_strength = abs(self.current_dopamine_signal)
        signal_direction = "positive" if self.current_dopamine_signal > 0 else "negative" if self.current_dopamine_signal < 0 else "neutral"
        
        # Recent signal statistics
        recent_signals = list(self.signal_history)[-10:] if len(self.signal_history) >= 10 else list(self.signal_history)
        avg_recent_signal = np.mean(recent_signals) if recent_signals else 0.0
        signal_volatility = np.std(recent_signals) if len(recent_signals) > 1 else 0.0
        
        return {
            'dopamine_signal': self.current_dopamine_signal,
            'signal_strength': signal_strength,
            'signal_direction': signal_direction,
            'consecutive_positive': self.consecutive_positive,
            'consecutive_negative': self.consecutive_negative,
            'peak_pnl': self.peak_pnl,
            'trough_pnl': self.trough_pnl,
            'expected_pnl': self.expected_pnl,
            'expectation_confidence': self.expectation_confidence,
            'avg_recent_signal': avg_recent_signal,
            'signal_volatility': signal_volatility,
            'momentum_state': self._get_momentum_state()
        }
    
    def _get_momentum_state(self) -> str:
        """Get descriptive momentum state"""
        if self.consecutive_positive >= 3:
            return "strong_positive_momentum"
        elif self.consecutive_positive >= 1:
            return "positive_momentum"
        elif self.consecutive_negative >= 3:
            return "strong_negative_momentum"
        elif self.consecutive_negative >= 1:
            return "negative_momentum"
        else:
            return "neutral_momentum"
    
    def reset_session(self):
        """Reset dopamine system for new trading session"""
        self.pnl_history.clear()
        self.signal_history.clear()
        self.previous_unrealized_pnl = 0.0
        self.baseline_unrealized_pnl = 0.0
        self.consecutive_positive = 0
        self.consecutive_negative = 0
        self.peak_pnl = 0.0
        self.trough_pnl = 0.0
        self.expected_pnl = 0.0
        self.expectation_confidence = 0.5
        self.current_dopamine_signal = 0.0
    
    def get_performance_metrics(self) -> Dict:
        """Get dopamine system performance metrics"""
        if not self.pnl_history:
            return {}
        
        # Extract P&L values
        pnl_values = [s.unrealized_pnl for s in self.pnl_history]
        signal_values = list(self.signal_history)
        
        # Calculate metrics
        total_pnl_change = pnl_values[-1] - pnl_values[0] if len(pnl_values) > 1 else 0.0
        max_pnl = max(pnl_values)
        min_pnl = min(pnl_values)
        pnl_range = max_pnl - min_pnl
        
        # Signal analysis
        positive_signals = sum(1 for s in signal_values if s > 0.1)
        negative_signals = sum(1 for s in signal_values if s < -0.1)
        neutral_signals = len(signal_values) - positive_signals - negative_signals
        
        return {
            'total_pnl_updates': len(self.pnl_history),
            'total_pnl_change': total_pnl_change,
            'max_pnl': max_pnl,
            'min_pnl': min_pnl,
            'pnl_range': pnl_range,
            'positive_signals': positive_signals,
            'negative_signals': negative_signals,
            'neutral_signals': neutral_signals,
            'avg_signal_strength': np.mean([abs(s) for s in signal_values]) if signal_values else 0.0,
            'max_consecutive_positive': max(self.consecutive_positive, 0),
            'max_consecutive_negative': max(self.consecutive_negative, 0)
        }