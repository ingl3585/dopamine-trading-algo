# risk_learning_engine.py

import numpy as np
import logging
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time

logger = logging.getLogger(__name__)

@dataclass
class RiskEvent:
    timestamp: float
    event_type: str  # 'position_taken', 'stop_hit', 'target_hit', 'manual_exit', 'drawdown'
    position_size: int
    account_balance: float
    risk_percentage: float
    outcome: float  # P&L
    market_conditions: Dict
    decision_factors: Dict

class RiskLearningEngine:
    """Advanced risk learning system that learns optimal risk parameters from outcomes"""
    
    def __init__(self):
        # Risk event history
        self.risk_events = deque(maxlen=1000)
        
        # Learning metrics
        self.position_size_outcomes = defaultdict(list)  # size -> outcomes
        self.stop_usage_outcomes = {'used': [], 'not_used': []}
        self.target_usage_outcomes = {'used': [], 'not_used': []}
        self.regime_risk_outcomes = defaultdict(list)  # regime -> outcomes
        
        # Adaptive risk parameters (learned)
        self.learned_optimal_size = 2.0  # Contracts
        self.learned_stop_effectiveness = 0.5
        self.learned_target_effectiveness = 0.5
        self.learned_max_risk_per_trade = 0.02  # 2% of account
        
        # Risk pattern recognition
        self.risk_patterns = {}
        self.pattern_performance = defaultdict(list)
        
        # Account size adaptation
        self.account_size_risk_mapping = {}
        
    def record_risk_event(self, event_type: str, position_size: int, account_balance: float,
                         outcome: float, market_conditions: Dict, decision_factors: Dict):
        """Record a risk-related event for learning"""
        
        risk_percentage = abs(outcome) / account_balance if account_balance > 0 else 0
        
        event = RiskEvent(
            timestamp=time.time(),
            event_type=event_type,
            position_size=position_size,
            account_balance=account_balance,
            risk_percentage=risk_percentage,
            outcome=outcome,
            market_conditions=market_conditions,
            decision_factors=decision_factors
        )
        
        self.risk_events.append(event)
        self._update_learning_metrics(event)
        self._update_risk_patterns(event)
        
    def _update_learning_metrics(self, event: RiskEvent):
        """Update learning metrics from risk event"""
        
        # Position size learning
        size_bucket = min(10, max(1, event.position_size))
        self.position_size_outcomes[size_bucket].append(event.outcome)
        
        # Keep only recent outcomes for each size
        if len(self.position_size_outcomes[size_bucket]) > 50:
            self.position_size_outcomes[size_bucket] = self.position_size_outcomes[size_bucket][-50:]
        
        # Stop/target effectiveness learning
        if event.event_type == 'stop_hit':
            self.stop_usage_outcomes['used'].append(event.outcome)
        elif event.event_type == 'target_hit':
            self.target_usage_outcomes['used'].append(event.outcome)
        elif event.event_type == 'manual_exit':
            # Assume no stop/target was used
            self.stop_usage_outcomes['not_used'].append(event.outcome)
            self.target_usage_outcomes['not_used'].append(event.outcome)
        
        # Regime-based risk learning
        regime = event.market_conditions.get('regime', 'normal')
        self.regime_risk_outcomes[regime].append({
            'outcome': event.outcome,
            'risk_pct': event.risk_percentage,
            'size': event.position_size
        })
        
        # Update learned parameters
        self._update_learned_parameters()
    
    def _update_learned_parameters(self):
        """Update learned risk parameters based on accumulated data"""
        
        # Learn optimal position size
        if len(self.position_size_outcomes) >= 3:
            size_performance = {}
            for size, outcomes in self.position_size_outcomes.items():
                if len(outcomes) >= 10:
                    # Risk-adjusted return (Sharpe-like ratio)
                    avg_return = np.mean(outcomes)
                    volatility = np.std(outcomes)
                    risk_adjusted = avg_return / (volatility + 0.01)
                    size_performance[size] = risk_adjusted
            
            if size_performance:
                best_size = max(size_performance.keys(), key=lambda x: size_performance[x])
                # Smooth update
                self.learned_optimal_size = 0.8 * self.learned_optimal_size + 0.2 * best_size
        
        # Learn stop effectiveness
        if (len(self.stop_usage_outcomes['used']) >= 10 and 
            len(self.stop_usage_outcomes['not_used']) >= 10):
            
            avg_with_stop = np.mean(self.stop_usage_outcomes['used'])
            avg_without_stop = np.mean(self.stop_usage_outcomes['not_used'])
            
            # Effectiveness score (-1 to 1)
            effectiveness = np.tanh((avg_with_stop - avg_without_stop) * 5)
            self.learned_stop_effectiveness = 0.9 * self.learned_stop_effectiveness + 0.1 * effectiveness
        
        # Learn target effectiveness
        if (len(self.target_usage_outcomes['used']) >= 10 and 
            len(self.target_usage_outcomes['not_used']) >= 10):
            
            avg_with_target = np.mean(self.target_usage_outcomes['used'])
            avg_without_target = np.mean(self.target_usage_outcomes['not_used'])
            
            effectiveness = np.tanh((avg_with_target - avg_without_target) * 5)
            self.learned_target_effectiveness = 0.9 * self.learned_target_effectiveness + 0.1 * effectiveness
    
    def _update_risk_patterns(self, event: RiskEvent):
        """Identify and learn from risk patterns"""
        
        # Create pattern signature
        pattern_key = self._create_risk_pattern_key(event.market_conditions, event.decision_factors)
        
        if pattern_key not in self.risk_patterns:
            self.risk_patterns[pattern_key] = {
                'count': 0,
                'total_outcome': 0.0,
                'avg_risk': 0.0,
                'best_size': 1
            }
        
        pattern = self.risk_patterns[pattern_key]
        pattern['count'] += 1
        pattern['total_outcome'] += event.outcome
        pattern['avg_risk'] = (pattern['avg_risk'] * (pattern['count'] - 1) + event.risk_percentage) / pattern['count']
        
        # Track best performing size for this pattern
        if event.outcome > 0:
            pattern['best_size'] = event.position_size
        
        # Store pattern performance
        self.pattern_performance[pattern_key].append(event.outcome)
        if len(self.pattern_performance[pattern_key]) > 20:
            self.pattern_performance[pattern_key] = self.pattern_performance[pattern_key][-20:]
    
    def _create_risk_pattern_key(self, market_conditions: Dict, decision_factors: Dict) -> str:
        """Create a pattern key for risk learning"""
        
        # Market condition buckets
        volatility = market_conditions.get('volatility', 0.02)
        vol_bucket = 'low' if volatility < 0.02 else 'med' if volatility < 0.04 else 'high'
        
        regime = market_conditions.get('regime', 'normal')
        
        # Decision factor buckets
        confidence = decision_factors.get('confidence', 0.5)
        conf_bucket = 'low' if confidence < 0.4 else 'med' if confidence < 0.7 else 'high'
        
        consensus = decision_factors.get('consensus_strength', 0.5)
        cons_bucket = 'low' if consensus < 0.4 else 'med' if consensus < 0.7 else 'high'
        
        return f"{vol_bucket}_{regime}_{conf_bucket}_{cons_bucket}"
    
    def get_optimal_position_size(self, market_conditions: Dict, decision_factors: Dict,
                                account_balance: float) -> int:
        """Get learned optimal position size for current conditions"""
        
        # Check for specific pattern
        pattern_key = self._create_risk_pattern_key(market_conditions, decision_factors)
        
        if pattern_key in self.risk_patterns and self.risk_patterns[pattern_key]['count'] >= 5:
            pattern = self.risk_patterns[pattern_key]
            avg_outcome = pattern['total_outcome'] / pattern['count']
            
            if avg_outcome > 0:
                # Use pattern-specific best size
                base_size = pattern['best_size']
            else:
                # Reduce size for poor-performing patterns
                base_size = max(1, int(self.learned_optimal_size * 0.5))
        else:
            # Use general learned optimal size
            base_size = int(self.learned_optimal_size)
        
        # Account size adjustment
        if account_balance < 10000:
            base_size = max(1, int(base_size * 0.5))  # Smaller sizes for small accounts
        elif account_balance > 50000:
            base_size = min(10, int(base_size * 1.2))  # Slightly larger for big accounts
        
        # Regime adjustment
        regime = market_conditions.get('regime', 'normal')
        if regime == 'crisis':
            base_size = max(1, int(base_size * 0.3))
        elif regime == 'volatile':
            base_size = max(1, int(base_size * 0.7))
        
        return min(10, max(1, base_size))  # Enforce 1-10 contract range
    
    def should_use_stop(self, market_conditions: Dict, decision_factors: Dict) -> bool:
        """Learned decision on whether to use stop loss"""
        
        # Base decision on learned effectiveness
        if self.learned_stop_effectiveness < -0.2:
            return False  # Stops are clearly harmful
        
        # Pattern-based decision
        pattern_key = self._create_risk_pattern_key(market_conditions, decision_factors)
        
        if pattern_key in self.pattern_performance:
            pattern_outcomes = self.pattern_performance[pattern_key]
            if len(pattern_outcomes) >= 10:
                avg_outcome = np.mean(pattern_outcomes)
                if avg_outcome < 0:
                    return False  # Poor pattern performance, skip stops
        
        # Regime-based decision
        regime = market_conditions.get('regime', 'normal')
        if regime == 'crisis':
            return True  # Use stops in crisis
        
        # Volatility-based decision
        volatility = market_conditions.get('volatility', 0.02)
        if volatility > 0.05:
            return True  # Use stops in high volatility
        
        # Default to learned effectiveness
        return self.learned_stop_effectiveness > 0.1
    
    def should_use_target(self, market_conditions: Dict, decision_factors: Dict) -> bool:
        """Learned decision on whether to use profit target"""
        
        # Base decision on learned effectiveness
        if self.learned_target_effectiveness < -0.2:
            return False  # Targets are clearly harmful
        
        # Confidence-based decision
        confidence = decision_factors.get('confidence', 0.5)
        if confidence < 0.3:
            return False  # Low confidence, let winners run
        
        # Trend strength decision
        trend_strength = market_conditions.get('trend_strength', 0.5)
        if trend_strength > 0.7:
            return False  # Strong trend, let winners run
        
        # Default to learned effectiveness
        return self.learned_target_effectiveness > 0.1
    
    def get_risk_summary(self) -> Dict:
        """Get comprehensive risk learning summary"""
        
        return {
            'learned_optimal_size': self.learned_optimal_size,
            'learned_stop_effectiveness': self.learned_stop_effectiveness,
            'learned_target_effectiveness': self.learned_target_effectiveness,
            'learned_max_risk_per_trade': self.learned_max_risk_per_trade,
            'total_risk_events': len(self.risk_events),
            'risk_patterns_discovered': len(self.risk_patterns),
            'position_size_data_points': sum(len(outcomes) for outcomes in self.position_size_outcomes.values()),
            'stop_usage_data': {
                'used': len(self.stop_usage_outcomes['used']),
                'not_used': len(self.stop_usage_outcomes['not_used'])
            },
            'target_usage_data': {
                'used': len(self.target_usage_outcomes['used']),
                'not_used': len(self.target_usage_outcomes['not_used'])
            }
        }
    
    def adapt_to_account_size(self, account_balance: float):
        """Adapt risk parameters to account size"""
        
        if account_balance not in self.account_size_risk_mapping:
            self.account_size_risk_mapping[account_balance] = {
                'optimal_size': self.learned_optimal_size,
                'max_risk': self.learned_max_risk_per_trade
            }
        
        # Account size-based adjustments
        if account_balance < 5000:
            self.learned_max_risk_per_trade = min(0.01, self.learned_max_risk_per_trade)  # 1% max for small accounts
        elif account_balance < 25000:
            self.learned_max_risk_per_trade = min(0.02, self.learned_max_risk_per_trade)  # 2% max for medium accounts
        else:
            self.learned_max_risk_per_trade = min(0.03, self.learned_max_risk_per_trade)  # 3% max for large accounts