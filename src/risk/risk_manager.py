# risk_manager.py

from dataclasses import dataclass
from typing import Optional, Dict
import logging
import math
import time
from collections import deque

from src.agent.trading_decision_engine import Decision
from src.core.market_data_processor import MarketData
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
        
        # Track violations for monitoring only (no penalties)
        self.violation_history = deque(maxlen=50)
        
    def validate_order(self, decision: Decision, market_data: MarketData) -> Optional[Order]:
        if decision.action == 'hold':
            return None
        
        # Update advanced risk metrics
        self.advanced_risk.update_risk_metrics(market_data)
        
        # Daily P&L is tracked but no longer enforced as blocking constraint
        # This preserves monitoring capabilities while removing trade blocking
        
        # Calculate position size using enhanced account data
        size = self._calculate_adaptive_position_size(decision, market_data)
        if size == 0:
            # Check if this was due to position limits for learning
            current_position = getattr(market_data, 'total_position_size', 0)
            max_contracts = int(self.meta_learner.get_parameter('max_contracts_limit'))
            
            if abs(current_position) >= max_contracts:
                logger.warning(f"Order REJECTED: Position limit reached ({current_position}/{max_contracts} contracts)")
                # Position limit enforcement without learning penalties
                self.violation_history.append(time.time())
            else:
                logger.info("Position size calculated as 0 (other risk factors)")
            return None
        
        # Apply advanced risk management - Kelly optimization and drawdown prevention
        # Defensive handling of intelligence_data with sensible fallbacks
        intelligence_data = getattr(decision, 'intelligence_data', None)
        if intelligence_data is None:
            # Create fallback intelligence_data to prevent AttributeError
            intelligence_data = {
                'volatility': 0.02,
                'price_momentum': 0.0,
                'volume_momentum': 0.0,
                'regime_confidence': 0.5,
                'consensus_strength': 0.5,
                'regime': 'normal'
            }
            logger.warning("Missing intelligence_data in decision, using fallback values")
        
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
        
        # Extract and validate state_features before creating Order
        decision_state_features = getattr(decision, 'state_features', None)
        logger.debug(f"STATE_FEATURES_TRACE: Decision has {len(decision_state_features) if decision_state_features else 'None'} state_features")
        
        return Order(
            action=decision.action,
            size=final_size,
            price=market_data.price,
            stop_price=stop_price,
            target_price=target_price,
            timestamp=market_data.timestamp,
            confidence=decision.confidence,
            features=decision_state_features,
            market_data=market_data,
            intelligence_data=getattr(decision, 'intelligence_data', None),
            decision_data={
                'state_features': decision_state_features,
                'adaptation_strategy': getattr(decision, 'adaptation_strategy', 'conservative'),
                'uncertainty_estimate': getattr(decision, 'uncertainty_estimate', 0.5),
                'primary_tool': getattr(decision, 'primary_tool', 'unknown'),
                'exploration': getattr(decision, 'exploration', False),
                'regime_awareness': getattr(decision, 'regime_awareness', None)
            }
        )
    
    def _alert_insufficient_margin(self, net_liquidation: float, margin_per_contract: float, max_affordable: int):
        """Alert system to insufficient margin for trading"""
        try:
            logger.critical(f"SYSTEM ALERT: Insufficient margin - Net Liquidation: ${net_liquidation:,.2f}, Required: ${margin_per_contract}")
            
            # Track margin insufficiency events
            if not hasattr(self, 'margin_alerts'):
                self.margin_alerts = []
            
            self.margin_alerts.append({
                'net_liquidation': net_liquidation,
                'margin_per_contract': margin_per_contract,
                'max_affordable': max_affordable,
                'time': time.time()
            })
            
            # Keep only last 100 events
            self.margin_alerts = self.margin_alerts[-100:]
            
        except Exception as e:
            logger.error(f"Failed to log margin insufficiency alert: {e}")
    
    def _alert_margin_adjustment(self, original_size: int, adjusted_size: int, net_liquidation: float):
        """Alert system to margin-based position size adjustment"""
        try:
            logger.warning(f"MARGIN ADJUSTMENT: Position reduced from {original_size} to {adjusted_size} contracts")
            logger.warning(f"Net Liquidation: ${net_liquidation:,.2f}, MNQ Margin: $100/contract")
            
            # Track margin adjustments for pattern analysis
            if not hasattr(self, 'margin_adjustments'):
                self.margin_adjustments = []
            
            self.margin_adjustments.append({
                'original_size': original_size,
                'adjusted_size': adjusted_size,
                'net_liquidation': net_liquidation,
                'reduction_percentage': (original_size - adjusted_size) / original_size,
                'time': time.time()
            })
            
            # Keep only last 100 events
            self.margin_adjustments = self.margin_adjustments[-100:]
            
        except Exception as e:
            logger.error(f"Failed to log margin adjustment alert: {e}")

    
    def _calculate_adaptive_position_size(self, decision: Decision, market_data: MarketData) -> int:
        """Enhanced position sizing with intelligent learning-based awareness"""
        
        # Initialize learned position sizing parameters first (CRITICAL: must be before any usage)
        try:
            max_contracts = int(self.meta_learner.get_parameter('max_contracts_limit'))  # Learned, starts at 10
        except (AttributeError, ValueError, TypeError) as e:
            logger.error(f"Failed to retrieve max_contracts_limit from meta_learner: {e}")
            max_contracts = 10  # Safe fallback value
            logger.warning(f"Using fallback max_contracts={max_contracts}")
        
        # Get current position exposure with validation
        ninja_position = getattr(market_data, 'total_position_size', None)
        portfolio_position = self.portfolio.get_total_position_size()
        
        # Validate position consistency
        if ninja_position is not None:
            position_diff = abs(ninja_position - portfolio_position)
            if position_diff > 0:
                logger.warning(f"POSITION MISMATCH: NinjaTrader={ninja_position}, Portfolio={portfolio_position}, Diff={position_diff}")
                # Use NinjaTrader as authoritative source
                current_position_size = ninja_position
                # Update portfolio to match NinjaTrader
                self.portfolio.sync_position_with_ninjatrader(ninja_position)
            else:
                current_position_size = ninja_position
        else:
            logger.warning("No position data from NinjaTrader - using portfolio fallback")
            current_position_size = portfolio_position
        
        # Debug logging for position tracking
        logger.debug(f"Position validated: NinjaTrader={ninja_position}, Portfolio={portfolio_position}, Using={current_position_size}")
        
        # Log position limit analysis for transparency (now max_contracts is properly initialized)
        logger.info(f"POSITION LIMIT ANALYSIS: Current={current_position_size}, Max={max_contracts}, Action={decision.action.upper()}")
        
        # Market conditions for risk learning with defensive null safety
        intelligence_data = getattr(decision, 'intelligence_data', None) or {}
        if not intelligence_data:
            logger.debug("Using fallback intelligence_data for position sizing calculations")
        
        market_conditions = {
            'volatility': intelligence_data.get('volatility', 0.02),
            'regime': intelligence_data.get('regime', 'normal'),
            'trend_strength': abs(intelligence_data.get('price_momentum', 0.0))
        }
        
        # Decision factors for risk learning with defensive null safety
        decision_factors = {
            'confidence': decision.confidence,
            'consensus_strength': intelligence_data.get('consensus_strength', 0.5),
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
        
        # 2. Enhanced account-based sizing for growth scaling
        if account_balance > 0:
            # Progressive scaling: more aggressive as account grows
            if account_balance > 10000:
                scaling_factor = 1500  # More aggressive for larger accounts
            elif account_balance > 5000:
                scaling_factor = 1750  # Moderate scaling  
            else:
                scaling_factor = 2000  # Conservative for smaller accounts
                
            account_size = max(1, int(account_balance * position_factor / scaling_factor))
            sizes.append(account_size)
        
        # 3. Enhanced confidence-based sizing for larger positions
        if decision.confidence > 0.7:
            # Exponential scaling for high confidence
            confidence_multiplier = (decision.confidence ** 1.5) * 2.0
        elif decision.confidence > 0.4:
            # Moderate scaling for good confidence  
            confidence_multiplier = (decision.confidence ** 1.2) * 1.5
        else:
            # Conservative scaling for lower confidence
            confidence_multiplier = decision.confidence ** 2
        
        confidence_size = max(1, int(5 * confidence_multiplier))  # Increased base multiplier
        sizes.append(confidence_size)
        
        # 4. Agent's suggested size
        agent_size = max(1, int(decision.size))
        sizes.append(agent_size)
        
        # Use learned size as primary, but cap with traditional methods
        base_size = min(learned_optimal_size, min(sizes[1:]))  # Learned size capped by traditional
        
        # Apply direction-aware position concentration limits
        remaining_capacity = max_contracts - abs(current_position_size)
        
        # Check for emergency override flag
        intelligence_data = getattr(decision, 'intelligence_data', None) or {}
        is_emergency_override = intelligence_data.get('emergency_override', False)
        
        # Check if trade direction reduces or increases current position
        is_position_reducing = self._is_position_reducing_trade(current_position_size, decision.action)
        
        if remaining_capacity <= 0:
            if is_emergency_override:
                # Always allow emergency override trades to bypass position limits
                logger.warning(f"EMERGENCY OVERRIDE: {current_position_size}/{max_contracts} contracts - ALLOWING emergency {decision.action.upper()} trade")
                # Don't return 0, continue with normal sizing logic
            elif is_position_reducing:
                # Always allow position-reducing trades regardless of limits
                logger.info(f"POSITION LIMIT: {current_position_size}/{max_contracts} contracts - ALLOWING position-reducing {decision.action.upper()} trade")
                # Don't return 0, continue with normal sizing logic
            else:
                # Block position-increasing trades when at limit
                logger.warning(f"POSITION LIMIT HIT: {current_position_size}/{max_contracts} contracts - BLOCKING position-increasing {decision.action.upper()} trade")
                # Record violation for monitoring (no learning penalties)
                self.violation_history.append(time.time())
                return 0
        
        # Intelligent exposure-based scaling with direction awareness
        exposure_ratio = abs(current_position_size) / max_contracts
        exposure_threshold = self.meta_learner.get_parameter('exposure_scaling_threshold')
        
        if exposure_ratio > exposure_threshold:
            # Scale down based on exposure
            scaling_factor = 1.0 - ((exposure_ratio - exposure_threshold) / (1.0 - exposure_threshold)) * 0.7
            final_size = max(1, int(base_size * scaling_factor))
            
            # For emergency overrides, position-reducing trades, or special handling
            if is_emergency_override:
                # Emergency overrides get priority - use calculated size but with safety limits
                emergency_max_size = min(final_size, 2)  # Cap emergency trades at 2 contracts for safety
                final_size = emergency_max_size
                logger.warning(f"Emergency override: allowing {emergency_max_size} contracts (calculated: {final_size})")
            elif is_position_reducing:
                # Allow full position reduction, but don't exceed current position size
                max_reduction_size = abs(current_position_size)
                final_size = min(final_size, max_reduction_size)
                logger.info(f"Position-reducing trade: allowing up to {max_reduction_size} contracts (current position: {current_position_size})")
            else:
                final_size = min(final_size, remaining_capacity)
                
            logger.info(f"Exposure scaling applied: {exposure_ratio:.1%} > {exposure_threshold:.1%}, "
                       f"scaling factor: {scaling_factor:.2f}")
        else:
            # For emergency overrides, position-reducing trades, or normal trades
            if is_emergency_override:
                # Emergency overrides get priority - use base size but with safety limits
                emergency_max_size = min(base_size, 2)  # Cap emergency trades at 2 contracts for safety
                final_size = emergency_max_size
                logger.warning(f"Emergency override: allowing {emergency_max_size} contracts (base: {base_size})")
            elif is_position_reducing:
                # Allow full position reduction, but don't exceed current position size
                max_reduction_size = abs(current_position_size)
                final_size = min(base_size, max_reduction_size)
                logger.info(f"Position-reducing trade: allowing up to {max_reduction_size} contracts (current position: {current_position_size})")
            else:
                final_size = min(base_size, remaining_capacity)
        
        # CRITICAL: Margin requirement validation for MNQ intraday trading
        # Each MNQ contract requires $100 margin intraday
        margin_per_contract = 100.0
        net_liquidation = market_data.net_liquidation
        
        # Calculate maximum contracts based on net liquidation
        max_affordable_contracts = int(net_liquidation / margin_per_contract)
        
        if max_affordable_contracts <= 0:
            logger.error(f"CRITICAL: Insufficient margin - Net Liquidation: ${net_liquidation:,.2f}, Required: ${margin_per_contract}")
            logger.error("ALERT: No contracts can be traded due to insufficient margin")
            self._alert_insufficient_margin(net_liquidation, margin_per_contract, 0)
            return 0
        
        # Apply margin-based position limit
        if final_size > max_affordable_contracts:
            logger.warning(f"Position size {final_size} exceeds margin capacity {max_affordable_contracts}. Reducing to margin limit.")
            logger.warning(f"Net Liquidation: ${net_liquidation:,.2f}, Required: ${final_size * margin_per_contract:,.2f}")
            final_size = max_affordable_contracts
            self._alert_margin_adjustment(final_size, max_affordable_contracts, net_liquidation)
        
        # Final validation: ensure we have sufficient margin for the calculated size
        required_margin = final_size * margin_per_contract
        if required_margin > net_liquidation:
            logger.error(f"CRITICAL: Final size validation failed - Required: ${required_margin:,.2f}, Available: ${net_liquidation:,.2f}")
            final_size = max(1, int(net_liquidation / margin_per_contract))
            logger.warning(f"Emergency position size reduction to {final_size} contracts")
        
        # Absolute maximum enforcement (both contract limit and margin limit)
        final_size = min(final_size, max_contracts, max_affordable_contracts)
        
        # Record this sizing decision for learning
        self.risk_learning.record_risk_event(
            'position_sizing',
            final_size,
            account_balance,
            0.0,  # Outcome unknown at this point
            market_conditions,
            decision_factors
        )
        
        # Log comprehensive sizing decision summary
        if final_size != agent_size:
            logger.info(f"Intelligent position sizing: Agent={agent_size}, Learned={learned_optimal_size}, "
                       f"Final={final_size} (Exposure: {current_position_size}/{max_contracts}, "
                       f"Confidence: {decision.confidence:.2f})")
        
        # Log position limit decision summary
        status_flags = []
        if is_emergency_override:
            status_flags.append("EMERGENCY")
        if is_position_reducing:
            status_flags.append("REDUCING")
        if remaining_capacity <= 0:
            status_flags.append("AT_LIMIT")
        
        status_str = f"[{','.join(status_flags)}]" if status_flags else "[NORMAL]"
        logger.info(f"POSITION LIMIT DECISION {status_str}: "
                   f"Position={current_position_size}/{max_contracts}, "
                   f"Action={decision.action.upper()}, "
                   f"Capacity={remaining_capacity}, "
                   f"Final_Size={final_size}")
        
        return final_size
    
    def _is_position_reducing_trade(self, current_position_size: int, trade_action: str) -> bool:
        """
        Determine if a trade reduces the current position size.
        
        Args:
            current_position_size: Current position size (positive = long, negative = short, 0 = flat)
            trade_action: Trade action ('buy' or 'sell')
            
        Returns:
            True if the trade reduces position size, False if it increases it
        """
        is_reducing = False
        position_type = "flat"
        
        if current_position_size == 0:
            # No current position, any trade increases position
            is_reducing = False
            position_type = "flat"
        elif current_position_size > 0:
            # Currently long: sell reduces position, buy increases it
            is_reducing = trade_action.lower() == 'sell'
            position_type = "long"
        else:
            # Currently short: buy reduces position, sell increases it
            is_reducing = trade_action.lower() == 'buy'
            position_type = "short"
        
        logger.debug(f"TRADE DIRECTION ANALYSIS: Position={current_position_size} ({position_type}), "
                    f"Action={trade_action.upper()}, Reduces={is_reducing}")
        
        return is_reducing
    
    def _calculate_adaptive_levels(self, decision: Decision, market_data: MarketData) -> tuple:
        """Intelligent stop/target calculation using risk learning"""
        
        # Market conditions for learning with defensive null safety
        intelligence_data = getattr(decision, 'intelligence_data', None) or {}
        market_conditions = {
            'volatility': intelligence_data.get('volatility', 0.02),
            'regime': intelligence_data.get('regime', 'normal'),
            'trend_strength': abs(intelligence_data.get('price_momentum', 0.0))
        }
        
        # Decision factors for learning with defensive null safety
        decision_factors = {
            'confidence': decision.confidence,
            'consensus_strength': intelligence_data.get('consensus_strength', 0.5),
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
    
    def assess_risk(self, decision, account_info) -> float:
        """
        Backward compatibility method for simple risk assessment
        
        Args:
            decision: Trading decision (Decision or TradeDecision object)
            account_info: Account information (AccountInfo or similar)
            
        Returns:
            Risk level [0.0, 1.0]
        """
        try:
            # Extract values with backward compatibility
            if hasattr(decision, 'size'):
                position_size = abs(decision.size)
                confidence = getattr(decision, 'confidence', 0.5)
                action = getattr(decision, 'action', 'hold')
            else:
                # Handle different decision object formats
                position_size = abs(getattr(decision, 'position_size', 1))
                confidence = getattr(decision, 'confidence', 0.5)
                action = getattr(decision, 'action', 'hold')
            
            if hasattr(account_info, 'balance'):
                account_balance = account_info.balance
            else:
                # Handle different account object formats
                account_balance = getattr(account_info, 'account_balance', 25000)
            
            # Calculate risk factors
            risk_factors = []
            
            # 1. Position size risk (larger positions = higher risk)
            max_position = 10  # Max reasonable position size
            position_risk = min(position_size / max_position, 1.0)
            risk_factors.append(position_risk * 0.4)  # 40% weight
            
            # 2. Confidence risk (lower confidence = higher risk)
            confidence_risk = 1.0 - confidence
            risk_factors.append(confidence_risk * 0.3)  # 30% weight
            
            # 3. Account balance risk (smaller account = higher relative risk)
            balance_threshold = 25000  # Baseline account size
            balance_risk = max(0.0, 1.0 - (account_balance / balance_threshold))
            balance_risk = min(balance_risk, 1.0)
            risk_factors.append(balance_risk * 0.2)  # 20% weight
            
            # 4. Action risk (hold = no risk, trade = some risk)
            action_risk = 0.0 if action == 'hold' else 0.1
            risk_factors.append(action_risk * 0.1)  # 10% weight
            
            # Combine risk factors
            total_risk = sum(risk_factors)
            
            return min(total_risk, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"Error in risk assessment: {e}")
            return 0.5  # Default moderate risk
    
    def run_monte_carlo_analysis(self, decision: Decision, market_data: MarketData) -> Dict:
        """Run Monte Carlo analysis for position sizing"""
        # Defensive handling of intelligence_data for Monte Carlo analysis
        intelligence_data = getattr(decision, 'intelligence_data', None)
        if intelligence_data is None:
            intelligence_data = {
                'volatility': 0.02,
                'price_momentum': 0.0,
                'regime_confidence': 0.5
            }
            logger.warning("Missing intelligence_data for Monte Carlo analysis, using fallback values")
        
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