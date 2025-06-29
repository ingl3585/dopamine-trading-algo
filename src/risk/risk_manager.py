# risk_manager.py

from dataclasses import dataclass
from typing import Optional, Dict
import logging
import math
import time
from collections import deque

from src.agent.trading_agent import Decision
from src.market_analysis.data_processor import MarketData
from src.risk.advanced_risk_manager import AdvancedRiskManager
from src.risk.risk_engine import RiskLearningEngine

logger = logging.getLogger(__name__)

@dataclass
class Order:
    action: str
    size: int
    price: float
    stop_price: float = 0.0
    target_price: float = 0.0
    timestamp: float = 0.0
    confidence: float = 0.0
    features: Optional[object] = None
    market_data: Optional[object] = None
    intelligence_data: Optional[Dict] = None
    decision_data: Optional[Dict] = None


class RiskManager:
    def __init__(self, portfolio, meta_learner, agent=None):
        self.portfolio = portfolio
        self.meta_learner = meta_learner
        self.agent = agent  # Reference to trading agent for immediate feedback
        self.advanced_risk = AdvancedRiskManager(meta_learner)
        self.risk_learning = RiskLearningEngine()
        
        # Track violation history for escalating penalties
        self.violation_history = deque(maxlen=50)
        self.recent_violations = 0  # Count of violations in recent time window
        
    def validate_order(self, decision: Decision, market_data: MarketData) -> Optional[Order]:
        if decision.action == 'hold':
            return None
        
        # Update advanced risk metrics
        self.advanced_risk.update_risk_metrics(market_data)
        
        # Check learned loss tolerance using actual account balance
        loss_tolerance = self.meta_learner.get_parameter('loss_tolerance_factor')
        max_loss = market_data.account_balance * loss_tolerance
        if market_data.daily_pnl <= -max_loss:
            logger.info(f"Daily loss limit reached: {market_data.daily_pnl:.2f} <= -{max_loss:.2f}")
            return None
        
        # Calculate position size using enhanced account data
        size = self._calculate_adaptive_position_size(decision, market_data)
        if size == 0:
            # Check if this was due to position limits for learning
            current_position = getattr(market_data, 'total_position_size', 0)
            max_contracts = int(self.meta_learner.get_parameter('max_contracts_limit'))
            
            if abs(current_position) >= max_contracts:
                logger.warning(f"Order REJECTED: Position limit reached ({current_position}/{max_contracts} contracts)")
                # Provide negative learning feedback for position limit violations
                self._learn_from_position_limit_rejection(decision, market_data, current_position, max_contracts)
            else:
                logger.info("Position size calculated as 0 (other risk factors)")
            return None
        
        # Apply advanced risk management - Kelly optimization and drawdown prevention
        intelligence_data = getattr(decision, 'intelligence_data', {})
        kelly_optimized_size = self.advanced_risk.optimize_kelly_position_size(
            size, market_data, intelligence_data
        )
        
        # Real-time drawdown prevention
        approved, final_size, reason = self.advanced_risk.check_drawdown_prevention(
            kelly_optimized_size, market_data
        )
        
        if not approved:
            logger.info(f"Order rejected by advanced risk management: {reason}")
            return None
        
        if final_size != kelly_optimized_size:
            logger.info(f"Position size adjusted by advanced risk: {kelly_optimized_size} -> {final_size} ({reason})")
        
        # Apply learned stop/target preferences
        stop_price, target_price = self._calculate_adaptive_levels(decision, market_data)
        
        return Order(
            action=decision.action,
            size=final_size,
            price=market_data.price,
            stop_price=stop_price,
            target_price=target_price,
            timestamp=market_data.timestamp,
            confidence=decision.confidence
        )
    
    def _learn_from_position_limit_rejection(self, decision, market_data, current_position, max_contracts):
        """Provide learning feedback when orders are rejected due to position limits"""
        try:
            # Record this violation for escalating penalty tracking
            current_time = time.time()
            self.violation_history.append(current_time)
            
            # Count recent violations (last 10 minutes)
            self.recent_violations = sum(1 for t in self.violation_history if current_time - t < 600)
            
            # Create negative reward for position limit violations
            violation_data = {
                'decision_confidence': decision.confidence,
                'current_position': current_position,
                'max_contracts': max_contracts,
                'violation_severity': abs(current_position) / max_contracts,
                'primary_tool': getattr(decision, 'primary_tool', 'unknown'),
                'exploration_mode': getattr(decision, 'exploration', False),
                'recent_violations': self.recent_violations
            }
            
            # Use the new reward engine for consistent rejection rewards
            negative_reward = -0.05 * violation_data['violation_severity']
            # The reward engine will handle escalation internally
            
            # Update meta-learner with negative feedback
            if hasattr(self.meta_learner, 'update_position_limit_awareness'):
                self.meta_learner.update_position_limit_awareness(violation_data, negative_reward)
            
            # CRITICAL: Send immediate feedback to the trading agent
            if self.agent and hasattr(self.agent, 'learn_from_rejection'):
                self.agent.learn_from_rejection('position_limit', violation_data, negative_reward)
            
            # Also send feedback to the main agent for learning (legacy)
            if hasattr(self.meta_learner, 'learn_from_rejection'):
                self.meta_learner.learn_from_rejection('position_limit', violation_data, negative_reward)
            
            # Update risk learning engine
            if hasattr(self.risk_learning, 'learn_from_violation'):
                self.risk_learning.learn_from_violation('position_limit', violation_data, negative_reward)
            
            logger.info(f"Learning from position limit violation: reward={negative_reward:.3f} "
                       f"recent_violations={self.recent_violations}, "
                       f"tool={violation_data['primary_tool']}, exploration={violation_data['exploration_mode']}")
            
        except Exception as e:
            logger.error(f"Error learning from position limit rejection: {e}")
    
    def _calculate_adaptive_position_size(self, decision: Decision, market_data: MarketData) -> int:
        """Enhanced position sizing with intelligent learning-based awareness"""
        
        # Get current position exposure from NinjaTrader (real position)
        current_position_size = getattr(market_data, 'total_position_size', 0)
        # Fallback to portfolio tracking if NinjaTrader data unavailable
        if current_position_size == 0:
            current_position_size = self.portfolio.get_total_position_size()
        
        # Debug logging for position tracking
        logger.info(f"Position check: NinjaTrader={getattr(market_data, 'total_position_size', 'MISSING')}, "
                   f"Portfolio={self.portfolio.get_total_position_size()}, Using={current_position_size}")
        
        # Learned position sizing parameters
        max_contracts = int(self.meta_learner.get_parameter('max_contracts_limit'))  # Learned, starts at 10
        
        # Market conditions for risk learning
        market_conditions = {
            'volatility': getattr(decision, 'intelligence_data', {}).get('volatility', 0.02),
            'regime': getattr(decision, 'intelligence_data', {}).get('regime', 'normal'),
            'trend_strength': abs(getattr(decision, 'intelligence_data', {}).get('price_momentum', 0))
        }
        
        # Decision factors for risk learning
        decision_factors = {
            'confidence': decision.confidence,
            'consensus_strength': getattr(decision, 'intelligence_data', {}).get('consensus_strength', 0.5),
            'primary_tool': getattr(decision, 'primary_tool', 'unknown')
        }
        
        # Get learned optimal size from risk learning engine
        learned_optimal_size = self.risk_learning.get_optimal_position_size(
            market_conditions, decision_factors, market_data.account_balance
        )
        
        # Traditional sizing approaches for comparison
        position_factor = self.meta_learner.get_parameter('position_size_factor')
        account_balance = market_data.account_balance
        
        sizes = []
        
        # 1. Learned optimal size (primary)
        sizes.append(learned_optimal_size)
        
        # 2. Account-based sizing
        if account_balance > 0:
            account_size = max(1, int(account_balance * position_factor / 2000))
            sizes.append(account_size)
        
        # 3. Confidence-based sizing
        confidence_multiplier = decision.confidence ** 2
        confidence_size = max(1, int(3 * confidence_multiplier))
        sizes.append(confidence_size)
        
        # 4. Agent's suggested size
        agent_size = max(1, int(decision.size))
        sizes.append(agent_size)
        
        # Use learned size as primary, but cap with traditional methods
        base_size = min(learned_optimal_size, min(sizes[1:]))  # Learned size capped by traditional
        
        # Apply position concentration limits
        remaining_capacity = max_contracts - abs(current_position_size)
        
        if remaining_capacity <= 0:
            logger.warning(f"POSITION LIMIT HIT: {current_position_size}/{max_contracts} contracts - BLOCKING TRADE")
            # Trigger immediate learning feedback
            self._learn_from_position_limit_rejection(decision, market_data, current_position_size, max_contracts)
            return 0
        
        # Intelligent exposure-based scaling
        exposure_ratio = abs(current_position_size) / max_contracts
        exposure_threshold = self.meta_learner.get_parameter('exposure_scaling_threshold')
        
        if exposure_ratio > exposure_threshold:
            # Scale down based on exposure
            scaling_factor = 1.0 - ((exposure_ratio - exposure_threshold) / (1.0 - exposure_threshold)) * 0.7
            final_size = max(1, int(base_size * scaling_factor))
            final_size = min(final_size, remaining_capacity)
            logger.info(f"Exposure scaling applied: {exposure_ratio:.1%} > {exposure_threshold:.1%}, "
                       f"scaling factor: {scaling_factor:.2f}")
        else:
            final_size = min(base_size, remaining_capacity)
        
        # Absolute maximum enforcement
        final_size = min(final_size, max_contracts)
        
        # Record this sizing decision for learning
        self.risk_learning.record_risk_event(
            'position_sizing',
            final_size,
            account_balance,
            0.0,  # Outcome unknown at this point
            market_conditions,
            decision_factors
        )
        
        # Log sizing decision
        if final_size != agent_size:
            logger.info(f"Intelligent position sizing: Agent={agent_size}, Learned={learned_optimal_size}, "
                       f"Final={final_size} (Exposure: {current_position_size}/{max_contracts}, "
                       f"Confidence: {decision.confidence:.2f})")
        
        return final_size
    
    def _calculate_adaptive_levels(self, decision: Decision, market_data: MarketData) -> tuple:
        """Intelligent stop/target calculation using risk learning"""
        
        # Market conditions for learning
        market_conditions = {
            'volatility': getattr(decision, 'intelligence_data', {}).get('volatility', 0.02),
            'regime': getattr(decision, 'intelligence_data', {}).get('regime', 'normal'),
            'trend_strength': abs(getattr(decision, 'intelligence_data', {}).get('price_momentum', 0))
        }
        
        # Decision factors for learning
        decision_factors = {
            'confidence': decision.confidence,
            'consensus_strength': getattr(decision, 'intelligence_data', {}).get('consensus_strength', 0.5),
            'primary_tool': getattr(decision, 'primary_tool', 'unknown')
        }
        
        # Learned decisions on whether to use stops/targets
        should_use_stop = self.risk_learning.should_use_stop(market_conditions, decision_factors)
        should_use_target = self.risk_learning.should_use_target(market_conditions, decision_factors)
        
        stop_price = 0.0
        target_price = 0.0
        
        # Apply intelligent stop logic
        if should_use_stop and decision.stop_price > 0:
            # Use agent's suggested stop with validation
            if decision.action == 'buy' and decision.stop_price < market_data.price:
                stop_price = decision.stop_price
            elif decision.action == 'sell' and decision.stop_price > market_data.price:
                stop_price = decision.stop_price
        elif should_use_stop:
            # Calculate intelligent stop based on learned parameters
            stop_distance_factor = self.meta_learner.get_parameter('stop_distance_factor')
            
            # Volatility adjustment
            vol_adjustment = 1.0 + (market_conditions['volatility'] * 10)
            adjusted_distance = stop_distance_factor * vol_adjustment
            
            if decision.action == 'buy':
                stop_price = market_data.price * (1 - adjusted_distance)
            else:
                stop_price = market_data.price * (1 + adjusted_distance)
        
        # Apply intelligent target logic
        if should_use_target and decision.target_price > 0:
            # Use agent's suggested target with validation
            if decision.action == 'buy' and decision.target_price > market_data.price:
                target_price = decision.target_price
            elif decision.action == 'sell' and decision.target_price < market_data.price:
                target_price = decision.target_price
        elif should_use_target:
            # Calculate intelligent target based on learned parameters
            target_distance_factor = self.meta_learner.get_parameter('target_distance_factor')
            
            # Trend strength adjustment
            trend_adjustment = 1.0 + market_conditions['trend_strength'] * 2
            adjusted_distance = target_distance_factor * trend_adjustment
            
            if decision.action == 'buy':
                target_price = market_data.price * (1 + adjusted_distance)
            else:
                target_price = market_data.price * (1 - adjusted_distance)
        
        # Log the decision for learning
        logger.info(f"Risk learning decisions: Stop={should_use_stop} (${stop_price:.2f}), "
                   f"Target={should_use_target} (${target_price:.2f})")
        
        return stop_price, target_price
    
    def process_trade_outcome(self, trade_outcome: Dict):
        """Process trade outcome for comprehensive risk learning"""
        
        # Update advanced risk metrics
        self.advanced_risk.update_risk_metrics(None, trade_outcome)
        
        # Extract trade information for risk learning
        pnl = trade_outcome.get('pnl', 0.0)
        exit_reason = trade_outcome.get('exit_reason', 'unknown')
        position_size = trade_outcome.get('size', 1)
        account_balance = trade_outcome.get('account_balance', 25000)
        
        # Determine event type for learning
        if exit_reason in ['stop_hit', 'stop_loss']:
            event_type = 'stop_hit'
        elif exit_reason in ['target_hit', 'profit_target']:
            event_type = 'target_hit'
        else:
            event_type = 'manual_exit'
        
        # Market conditions (reconstruct from available data)
        market_conditions = {
            'volatility': trade_outcome.get('volatility', 0.02),
            'regime': trade_outcome.get('regime', 'normal'),
            'trend_strength': trade_outcome.get('trend_strength', 0.5)
        }
        
        # Decision factors (reconstruct from available data)
        decision_factors = {
            'confidence': trade_outcome.get('confidence', 0.5),
            'consensus_strength': trade_outcome.get('consensus_strength', 0.5),
            'primary_tool': trade_outcome.get('primary_tool', 'unknown')
        }
        
        # Record the outcome for risk learning
        self.risk_learning.record_risk_event(
            event_type,
            position_size,
            account_balance,
            pnl,
            market_conditions,
            decision_factors
        )
        
        # Adapt risk learning to account size changes
        self.risk_learning.adapt_to_account_size(account_balance)
        
        logger.info(f"Risk learning updated: {event_type}, Size={position_size}, "
                   f"P&L=${pnl:.2f}, Account=${account_balance:.0f}")
    
    def get_risk_summary(self) -> Dict:
        """Get comprehensive risk summary including advanced metrics"""
        basic_summary = {
            'consecutive_losses': self.portfolio.get_consecutive_losses(),
            'position_count': self.portfolio.get_position_count(),
            'win_rate': self.portfolio.get_win_rate()
        }
        
        advanced_summary = self.advanced_risk.get_risk_summary()
        
        return {**basic_summary, **advanced_summary}
    
    def run_monte_carlo_analysis(self, decision: Decision, market_data: MarketData) -> Dict:
        """Run Monte Carlo analysis for position sizing"""
        intelligence_data = getattr(decision, 'intelligence_data', {})
        scenarios = self.advanced_risk.run_monte_carlo_simulation(
            decision.size, market_data, intelligence_data
        )
        
        return {
            'scenarios': [
                {
                    'scenario_id': s.scenario_id,
                    'probability': s.probability,
                    'expected_pnl': s.expected_pnl,
                    'var_95': s.var_95,
                    'var_99': s.var_99,
                    'stress_factor': s.stress_factor
                }
                for s in scenarios
            ],
            'recommended_action': 'proceed' if scenarios[0].expected_pnl > 0 else 'reduce_size'
        }