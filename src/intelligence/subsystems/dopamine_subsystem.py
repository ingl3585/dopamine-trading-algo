"""
Dopamine Subsystem - Real-time P&L-based reward system
Mimics neurological dopamine responses for immediate trading feedback
"""

import numpy as np
import time
from typing import Dict, List, Optional
from collections import deque
from dataclasses import dataclass
from src.agent.reward_engine import DopamineRewardComponent

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
        
        # Use the DopamineRewardComponent for the core functionality
        self.dopamine_component = DopamineRewardComponent(config)
        
    def process_pnl_update(self, market_data: Dict) -> float:
        """
        Process real-time P&L update and generate dopamine signal
        
        Args:
            market_data: Dictionary containing unrealized_pnl, position_size, etc.
            
        Returns:
            float: Dopamine signal [-2.0 to +2.0]
        """
        return self.dopamine_component.process_pnl_update(market_data)
    
    def get_signal(self) -> float:
        """Get current dopamine signal"""
        return self.dopamine_component.get_signal()
    
    def get_signal_with_context(self) -> Dict:
        """Get dopamine signal with full context"""
        return {
            'signal': self.dopamine_component.get_signal(),
            'previous_pnl': self.dopamine_component.previous_unrealized_pnl,
            'consecutive_positive': self.dopamine_component.consecutive_positive,
            'consecutive_negative': self.dopamine_component.consecutive_negative,
            'peak_pnl': self.dopamine_component.peak_pnl,
            'trough_pnl': self.dopamine_component.trough_pnl,
            'momentum_state': self._get_momentum_state()
        }
    
    def reset_session(self):
        """Reset dopamine tracking for new session"""
        self.dopamine_component.reset_session()
    
    def _get_momentum_state(self) -> str:
        """Get current momentum state description"""
        if self.dopamine_component.consecutive_positive > 3:
            return "strong_positive"
        elif self.dopamine_component.consecutive_positive > 1:
            return "positive"
        elif self.dopamine_component.consecutive_negative > 3:
            return "strong_negative"
        elif self.dopamine_component.consecutive_negative > 1:
            return "negative"
        else:
            return "neutral"

    def get_stats(self) -> Dict:
        """Get comprehensive statistics"""
        signal_history = list(self.dopamine_component.signal_history)
        pnl_history = list(self.dopamine_component.pnl_history)
        
        # Calculate statistics
        if signal_history:
            avg_signal = np.mean(signal_history)
            signal_volatility = np.std(signal_history)
            max_signal = np.max(signal_history)
            min_signal = np.min(signal_history)
        else:
            avg_signal = signal_volatility = max_signal = min_signal = 0.0
        
        return {
            'current_signal': self.dopamine_component.get_signal(),
            'avg_signal': avg_signal,
            'signal_volatility': signal_volatility,
            'max_signal': max_signal,
            'min_signal': min_signal,
            'signal_history_length': len(signal_history),
            'pnl_history_length': len(pnl_history),
            'consecutive_positive': self.dopamine_component.consecutive_positive,
            'consecutive_negative': self.dopamine_component.consecutive_negative,
            'peak_pnl': self.dopamine_component.peak_pnl,
            'trough_pnl': self.dopamine_component.trough_pnl,
            'momentum_state': self._get_momentum_state(),
            'expected_pnl': self.dopamine_component.expected_pnl,
            'expectation_confidence': self.dopamine_component.expectation_confidence,
            'max_consecutive_positive': max(self.dopamine_component.consecutive_positive, 0),
            'max_consecutive_negative': max(self.dopamine_component.consecutive_negative, 0)
        }