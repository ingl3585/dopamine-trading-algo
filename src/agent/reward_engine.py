"""
Centralized Reward Engine - Consolidates all reward calculation logic
"""

import torch
import torch.nn as nn
import numpy as np
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class MetaParameter:
    """Adaptive parameter that learns from outcomes"""
    value: float
    bounds: Tuple[float, float]
    learning_rate: float = 0.01
    momentum: float = 0.0
    
    def __post_init__(self):
        self.velocity = 0.0
        self.outcomes = deque(maxlen=50)
    
    def update(self, gradient: float):
        self.velocity = self.momentum * self.velocity + self.learning_rate * gradient
        self.value = np.clip(self.value + self.velocity, *self.bounds)
    
    def add_outcome(self, outcome: float):
        self.outcomes.append(outcome)
    
    def get_gradient(self) -> float:
        if len(self.outcomes) < 5:
            return 0.0
        
        recent = list(self.outcomes)[-10:]
        return np.mean(recent)

@dataclass
class PnLSnapshot:
    """Snapshot of P&L state for dopamine calculations"""
    timestamp: float
    unrealized_pnl: float
    realized_pnl: float
    position_size: float
    current_price: float

class DopamineRewardComponent:
    """
    Dopamine-inspired real-time P&L reward system
    Provides immediate feedback based on unrealized P&L changes
    """
    
    def __init__(self, config: Dict = None):
        config = config or {}
        
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
        """Process real-time P&L update and generate dopamine signal"""
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
            logger.error(f"Error in dopamine P&L processing: {e}")
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

class CoreRewardEngine:
    """
    Core reward calculation engine for trading decisions
    Combines multiple reward components for comprehensive feedback
    """
    
    def __init__(self):
        self.components = {
            'pnl_weight': MetaParameter(3.0, (1.0, 5.0)),  # Increase PnL weight significantly
            'drawdown_penalty': MetaParameter(0.5, (0.0, 2.0)),
            'hold_time_factor': MetaParameter(0.02, (0.0, 0.1)),  # Reduce hold penalty drastically
            'win_rate_bonus': MetaParameter(0.1, (0.0, 0.5)),  # Reduce win rate bonus weight
            'subsystem_consistency': MetaParameter(0.1, (0.0, 0.5)),  # Reduce consistency weight
            'account_preservation': MetaParameter(0.2, (0.0, 0.5))  # Reduce preservation weight
        }
        
        self.outcome_history = deque(maxlen=200)
    
    def compute_reward(self, trade_data: Dict[str, Any]) -> float:
        """Compute comprehensive reward for a trading decision"""
        pnl = trade_data.get('pnl', 0.0)
        account_balance = trade_data.get('account_balance', 25000)
        hold_time = trade_data.get('hold_time', 1.0)
        was_winner = pnl > 0
        subsystem_agreement = trade_data.get('subsystem_agreement', 0.5)
        
        # Check if this was a holding decision vs trading decision
        decision_action = trade_data.get('action', 'trade')  # 'hold' or 'trade'
        decision_confidence = trade_data.get('decision_confidence', 0.5)
        
        # Moderate penalty for position limit violations - prevent confidence system destruction
        position_limit_violation = trade_data.get('position_limit_violation', False)
        if position_limit_violation:
            # Much more moderate penalty to prevent overwhelming the confidence system
            # The escalating penalty from RejectionRewardEngine will handle progression
            penalty = -0.5  # Reduced from -2.0 to work with new confidence system
            logger.warning(f"POSITION LIMIT VIOLATION PENALTY: {penalty}")
            return penalty
        
        # Account-normalized PnL component - heavily reward profitable trades
        pnl_norm = np.tanh(pnl / (account_balance * 0.01))  # Normalize by 1% of account
        
        # Convert hold_time from nanoseconds to seconds if needed
        if hold_time > 1000000000:  # If in nanoseconds (> 1 billion)
            hold_time_seconds = hold_time / 1000000000.0  # Convert to seconds
        else:
            hold_time_seconds = hold_time  # Already in seconds
        
        # Hold time penalty for overly long trades (only for trades > 1 hour)
        hold_penalty = max(0, (hold_time_seconds - 3600) / 3600) * 0.1
        
        # Win rate context - only reward good performance, don't penalize learning
        recent_wins = sum(1 for outcome in list(self.outcome_history)[-10:] if outcome > 0)
        win_rate_bonus = max(0, (recent_wins / 10.0 - 0.5) * 0.2)  # Only positive bonus, no penalty
        
        # Direct profit bonus - strongly reward profitable trades
        profit_bonus = 0.0
        if pnl > 0:
            profit_bonus = min(pnl / (account_balance * 0.01), 5.0) * 2.0  # Cap at 500% of 1% account risk, then double it
        
        # Subsystem consistency bonus
        consistency_bonus = (subsystem_agreement - 0.5) * 0.1
        
        # Account preservation bonus (reward smaller risks on smaller accounts)
        risk_pct = abs(pnl) / account_balance
        preservation_bonus = max(0, 0.02 - risk_pct) * 5.0  # Bonus for risks < 2%
        
        # Context-dependent holding rewards
        holding_bonus = 0.0
        if decision_action == 'hold':
            # Reward holding when confidence is low (uncertain conditions)
            if decision_confidence < 0.3:
                holding_bonus = 0.05 * (0.3 - decision_confidence)  # Up to +0.015 for very uncertain
            # Small penalty for holding during high-confidence signals (missed opportunities)
            elif decision_confidence > 0.7:
                holding_bonus = -0.02 * (decision_confidence - 0.7)  # Up to -0.006 for high confidence
        elif decision_action == 'trade' and decision_confidence < 0.2:
            # Small penalty for trading when very uncertain
            holding_bonus = -0.01
        
        # Debug logging for reward calculation
        pnl_component = self.components['pnl_weight'].value * pnl_norm
        hold_component = self.components['hold_time_factor'].value * (-hold_penalty)
        win_component = self.components['win_rate_bonus'].value * win_rate_bonus
        consistency_component = self.components['subsystem_consistency'].value * consistency_bonus
        preservation_component = self.components['account_preservation'].value * preservation_bonus
        # Note: holding_bonus is not weighted, it's a direct behavioral incentive
        
        reward = (
            pnl_component +
            hold_component +
            win_component +
            consistency_component +
            preservation_component +
            holding_bonus +
            profit_bonus  # Add direct profit bonus
        )
        
        # Log reward breakdown for debugging
        logger.info(f"REWARD BREAKDOWN - PnL=${pnl:.2f}, Action={decision_action}, Conf={decision_confidence:.2f} -> Total={reward:.6f}")
        logger.info(f"  PnL: {pnl_norm:.3f} * {self.components['pnl_weight'].value:.3f} = {pnl_component:.6f}")
        logger.info(f"  Hold: {-hold_penalty:.3f} * {self.components['hold_time_factor'].value:.3f} = {hold_component:.6f}")
        logger.info(f"  Win: {win_rate_bonus:.3f} * {self.components['win_rate_bonus'].value:.3f} = {win_component:.6f}")
        logger.info(f"  Consistency: {consistency_bonus:.3f} * {self.components['subsystem_consistency'].value:.3f} = {consistency_component:.6f}")
        logger.info(f"  Preservation: {preservation_bonus:.3f} * {self.components['account_preservation'].value:.3f} = {preservation_component:.6f}")
        logger.info(f"  Holding: {holding_bonus:.6f} (direct incentive)")
        logger.info(f"  Profit: {profit_bonus:.6f} (profit bonus)")
        
        self.outcome_history.append(reward)
        
        # Update component parameters
        for component in self.components.values():
            component.add_outcome(reward)
        
        return reward
    
    def compute_holding_reward(self, decision_confidence: float, market_conditions: Dict[str, Any]) -> float:
        """Compute reward for holding decisions based on context"""
        holding_data = {
            'pnl': 0.0,  # No immediate PnL from holding
            'action': 'hold',
            'decision_confidence': decision_confidence,
            'account_balance': market_conditions.get('account_balance', 25000),
            'hold_time': 0.0,  # No hold time for instant decision
            'subsystem_agreement': market_conditions.get('subsystem_agreement', 0.5)
        }
        return self.compute_reward(holding_data)
    
    def adapt_components(self):
        """Adapt reward component parameters based on outcomes"""
        for component in self.components.values():
            gradient = component.get_gradient()
            component.update(gradient)

class RejectionRewardEngine:
    """
    Specialized reward engine for handling trade rejections
    Provides negative feedback to prevent repeated rejection patterns
    """
    
    def __init__(self):
        self.rejection_history = deque(maxlen=100)
        self.recent_position_rejections = 0
        self.position_rejection_timestamps = deque(maxlen=50)
        
    def compute_rejection_reward(self, rejection_type: str, rejection_data: Dict, base_reward: float = -1.0) -> float:
        """Compute reward penalty for trade rejections"""
        
        if rejection_type == 'position_limit':
            # Moderate escalating penalty - cap maximum to prevent confidence destruction
            penalty_multiplier = 1.0 + (self.recent_position_rejections * 0.2)  # Reduced from 0.5
            penalty_multiplier = min(penalty_multiplier, 2.0)  # Cap at 2x maximum
            rejection_reward = base_reward * penalty_multiplier
            
            # Track rejection for learning
            current_time = time.time()
            self.position_rejection_timestamps.append(current_time)
            self.recent_position_rejections += 1
            
            # Clean up old timestamps (older than 10 minutes)
            cutoff_time = current_time - 600
            while (self.position_rejection_timestamps and 
                   self.position_rejection_timestamps[0] < cutoff_time):
                self.position_rejection_timestamps.popleft()
            
            # Update recent count based on cleaned timestamps
            self.recent_position_rejections = len(self.position_rejection_timestamps)
            
            logger.warning(f"Position limit rejection penalty: {rejection_reward:.2f} (multiplier: {penalty_multiplier:.2f})")
            return rejection_reward
            
        elif rejection_type == 'insufficient_margin':
            return base_reward * 0.8  # Moderate penalty
            
        elif rejection_type == 'invalid_order':
            return base_reward * 0.3  # Light penalty
            
        else:
            return base_reward  # Default penalty

class UnifiedRewardEngine:
    """
    Unified reward engine that combines all reward calculation components
    Provides a single interface for all reward computations
    """
    
    def __init__(self, config: Dict = None):
        config = config or {}
        
        # Initialize component engines
        self.core_engine = CoreRewardEngine()
        self.dopamine_engine = DopamineRewardComponent(config.get('dopamine', {}))
        self.rejection_engine = RejectionRewardEngine()
        
        # Engine weights
        self.weights = {
            'core_weight': config.get('core_weight', 1.0),
            'dopamine_weight': config.get('dopamine_weight', 0.3),
            'rejection_weight': config.get('rejection_weight', 1.0)
        }
        
        # Overall tracking
        self.total_rewards_computed = 0
        self.average_reward = 0.0
        self.reward_variance = 0.0
        self.reward_history = deque(maxlen=1000)
    
    def compute_trade_reward(self, trade_data: Dict[str, Any]) -> float:
        """Compute comprehensive reward for a completed trade"""
        # Core reward calculation
        core_reward = self.core_engine.compute_reward(trade_data)
        
        # Dopamine component (if real-time P&L data available)
        dopamine_reward = 0.0
        if 'market_data' in trade_data:
            dopamine_reward = self.dopamine_engine.process_pnl_update(trade_data['market_data'])
        
        # Combine rewards
        total_reward = (
            self.weights['core_weight'] * core_reward +
            self.weights['dopamine_weight'] * dopamine_reward
        )
        
        # Update tracking
        self._update_tracking(total_reward)
        
        return total_reward
    
    def compute_holding_reward(self, decision_confidence: float, market_conditions: Dict[str, Any]) -> float:
        """Compute reward for holding decisions"""
        return self.core_engine.compute_holding_reward(decision_confidence, market_conditions)
    
    def compute_rejection_reward(self, rejection_type: str, rejection_data: Dict) -> float:
        """Compute reward for trade rejections"""
        rejection_reward = self.rejection_engine.compute_rejection_reward(rejection_type, rejection_data)
        
        # Weight and track rejection reward
        total_reward = self.weights['rejection_weight'] * rejection_reward
        self._update_tracking(total_reward)
        
        return total_reward
    
    def process_realtime_pnl(self, market_data: Dict) -> float:
        """Process real-time P&L for immediate dopamine feedback"""
        return self.dopamine_engine.process_pnl_update(market_data)
    
    def adapt_parameters(self):
        """Adapt all reward engine parameters"""
        self.core_engine.adapt_components()
    
    def reset_session(self):
        """Reset engines for new trading session"""
        self.dopamine_engine.reset_session()
        
    def _update_tracking(self, reward: float):
        """Update reward tracking statistics"""
        self.total_rewards_computed += 1
        self.reward_history.append(reward)
        
        # Update running statistics
        if len(self.reward_history) > 0:
            rewards = list(self.reward_history)
            self.average_reward = np.mean(rewards)
            self.reward_variance = np.var(rewards)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive reward engine statistics"""
        recent_rewards = list(self.reward_history)[-20:] if len(self.reward_history) >= 20 else list(self.reward_history)
        
        stats = {
            'total_rewards_computed': self.total_rewards_computed,
            'average_reward': self.average_reward,
            'reward_variance': self.reward_variance,
            'recent_average_reward': np.mean(recent_rewards) if recent_rewards else 0.0,
            'recent_reward_std': np.std(recent_rewards) if len(recent_rewards) > 1 else 0.0,
            'dopamine_signal': self.dopamine_engine.get_signal(),
            'position_rejections': self.rejection_engine.recent_position_rejections,
        }
        
        # Core engine component values
        stats['core_components'] = {
            name: component.value for name, component in self.core_engine.components.items()
        }
        
        return stats
    
    def save_state(self, filepath: str):
        """Save reward engine state"""
        import os
        import pickle
        
        # Ensure directory exists
        dir_path = os.path.dirname(filepath)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        
        state = {
            'core_components': {name: (comp.value, comp.bounds) for name, comp in self.core_engine.components.items()},
            'dopamine_state': {
                'previous_unrealized_pnl': self.dopamine_engine.previous_unrealized_pnl,
                'consecutive_positive': self.dopamine_engine.consecutive_positive,
                'consecutive_negative': self.dopamine_engine.consecutive_negative,
                'peak_pnl': self.dopamine_engine.peak_pnl,
                'trough_pnl': self.dopamine_engine.trough_pnl,
            },
            'rejection_state': {
                'recent_position_rejections': self.rejection_engine.recent_position_rejections,
            },
            'weights': self.weights,
            'total_rewards_computed': self.total_rewards_computed,
            'average_reward': self.average_reward,
            'reward_variance': self.reward_variance,
            'reward_history': list(self.reward_history)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self, filepath: str):
        """Load reward engine state"""
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            # Restore core components
            for name, (value, bounds) in state['core_components'].items():
                if name in self.core_engine.components:
                    self.core_engine.components[name].value = value
                    self.core_engine.components[name].bounds = bounds
            
            # Restore dopamine state
            dopamine_state = state['dopamine_state']
            self.dopamine_engine.previous_unrealized_pnl = dopamine_state['previous_unrealized_pnl']
            self.dopamine_engine.consecutive_positive = dopamine_state['consecutive_positive']
            self.dopamine_engine.consecutive_negative = dopamine_state['consecutive_negative']
            self.dopamine_engine.peak_pnl = dopamine_state['peak_pnl']
            self.dopamine_engine.trough_pnl = dopamine_state['trough_pnl']
            
            # Restore rejection state
            rejection_state = state['rejection_state']
            self.rejection_engine.recent_position_rejections = rejection_state['recent_position_rejections']
            
            # Restore other state
            self.weights = state.get('weights', self.weights)
            self.total_rewards_computed = state.get('total_rewards_computed', 0)
            self.average_reward = state.get('average_reward', 0.0)
            self.reward_variance = state.get('reward_variance', 0.0)
            
            reward_history = state.get('reward_history', [])
            self.reward_history = deque(reward_history, maxlen=1000)
            
            logger.info("Reward engine state loaded successfully")
            
        except FileNotFoundError:
            logger.info("No existing reward engine state found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading reward engine state: {e}")