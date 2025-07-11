# portfolio_optimizer.py

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class OptimizationType(Enum):
    """Types of optimization suggestions"""
    PROFIT_TAKING = "profit_taking"
    STOP_LOSS = "stop_loss"
    TIME_STOP = "time_stop"
    RISK_REDUCTION = "risk_reduction"
    DIVERSIFICATION = "diversification"
    POSITION_SIZING = "position_sizing"
    REBALANCING = "rebalancing"
    OVERTRADING = "overtrading"

class UrgencyLevel(Enum):
    """Urgency levels for optimization suggestions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class OptimizationSuggestion:
    """Container for optimization suggestions"""
    suggestion_type: OptimizationType
    symbol: str
    reason: str
    suggested_action: str
    urgency: UrgencyLevel
    impact_score: float = 0.0
    confidence: float = 0.0
    metadata: Dict[str, Any] = None

@dataclass
class AllocationTarget:
    """Target allocation for portfolio optimization"""
    symbol: str
    target_weight: float
    current_weight: float
    adjustment_needed: float
    rationale: str

class PortfolioOptimizer:
    """
    Optimizes portfolio allocation and suggests improvements.
    
    Responsibilities:
    - Analyze positions for optimization opportunities
    - Calculate optimal allocation based on market conditions
    - Generate optimization suggestions with urgency levels
    - Perform risk-based portfolio adjustments
    - Monitor for overtrading and concentration risk
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Optimization thresholds
        self.profit_taking_threshold = 0.7  # Take profit when down 30% from peak
        self.stop_loss_threshold = 0.5  # Stop loss when loss increases 50%
        self.max_hold_time = config.get('max_hold_time_hours', 24)
        self.max_concentration = 0.7  # Maximum 70% in single position
        self.max_daily_trades = config.get('max_daily_trades', 10)
        self.risk_threshold = 0.8  # Risk level above which to reduce positions
        
        # Optimization history
        self.optimization_history = []
        
        logger.info("Portfolio optimizer initialized")
    
    def optimize_portfolio(self, positions: Dict[str, Any], 
                          market_conditions: Dict[str, Any], 
                          risk_metrics: Dict[str, Any],
                          trade_history: List[Any] = None) -> Dict[str, Any]:
        """
        Perform comprehensive portfolio optimization
        
        Args:
            positions: Current portfolio positions
            market_conditions: Current market conditions
            risk_metrics: Risk assessment metrics
            trade_history: Recent trade history
            
        Returns:
            Optimization results and suggestions
        """
        try:
            optimization_suggestions = []
            
            # Analyze individual positions
            for symbol, position in positions.items():
                if isinstance(position, dict):
                    suggestions = self._analyze_position_optimization(
                        symbol, position, market_conditions, risk_metrics
                    )
                    optimization_suggestions.extend(suggestions)
            
            # Portfolio-level optimization
            portfolio_suggestions = self._analyze_portfolio_optimization(
                positions, market_conditions, risk_metrics, trade_history
            )
            optimization_suggestions.extend(portfolio_suggestions)
            
            # Calculate optimal allocation
            optimal_allocation = self._calculate_optimal_allocation(
                positions, market_conditions, risk_metrics
            )
            
            # Current allocation
            current_allocation = self._calculate_current_allocation(positions)
            
            # Rebalancing analysis
            rebalancing_needed = self._assess_rebalancing_need(
                current_allocation, optimal_allocation
            )
            
            # Sort suggestions by urgency and impact
            optimization_suggestions.sort(
                key=lambda x: (x.urgency.value, -x.impact_score)
            )
            
            return {
                'optimization_suggestions': [
                    {
                        'type': s.suggestion_type.value,
                        'symbol': s.symbol,
                        'reason': s.reason,
                        'suggested_action': s.suggested_action,
                        'urgency': s.urgency.value,
                        'impact_score': s.impact_score,
                        'confidence': s.confidence,
                        'metadata': s.metadata or {}
                    } for s in optimization_suggestions
                ],
                'optimal_allocation': optimal_allocation,
                'current_allocation': current_allocation,
                'rebalancing_needed': rebalancing_needed,
                'total_suggestions': len(optimization_suggestions),
                'high_urgency_count': sum(1 for s in optimization_suggestions if s.urgency == UrgencyLevel.HIGH),
                'risk_level': risk_metrics.get('total_risk', 0.5)
            }
            
        except Exception as e:
            logger.error(f"Error optimizing portfolio: {e}")
            return {
                'optimization_suggestions': [],
                'optimal_allocation': {},
                'current_allocation': {},
                'rebalancing_needed': False,
                'total_suggestions': 0,
                'high_urgency_count': 0,
                'risk_level': 0.5
            }
    
    def _analyze_position_optimization(self, symbol: str, position: Dict[str, Any],
                                      market_conditions: Dict[str, Any], 
                                      risk_metrics: Dict[str, Any]) -> List[OptimizationSuggestion]:
        """Analyze individual position for optimization opportunities"""
        suggestions = []
        
        try:
            unrealized_pnl = position.get('unrealized_pnl', 0.0)
            max_unrealized = position.get('max_unrealized_pnl', 0.0)
            min_unrealized = position.get('min_unrealized_pnl', 0.0)
            entry_time = position.get('entry_time')
            
            # Profit taking analysis
            if unrealized_pnl > 0 and max_unrealized > 0:
                pnl_ratio = unrealized_pnl / max_unrealized
                if pnl_ratio < self.profit_taking_threshold:
                    decline_pct = (1 - pnl_ratio) * 100
                    suggestions.append(OptimizationSuggestion(
                        suggestion_type=OptimizationType.PROFIT_TAKING,
                        symbol=symbol,
                        reason=f"Position down {decline_pct:.1f}% from peak (${max_unrealized:.2f} -> ${unrealized_pnl:.2f})",
                        suggested_action="partial_close",
                        urgency=UrgencyLevel.MEDIUM if decline_pct > 40 else UrgencyLevel.LOW,
                        impact_score=decline_pct,
                        confidence=0.7,
                        metadata={'peak_pnl': max_unrealized, 'current_pnl': unrealized_pnl}
                    ))
            
            # Stop loss analysis
            if unrealized_pnl < 0 and min_unrealized < 0:
                loss_increase = abs(unrealized_pnl) / abs(min_unrealized)
                if loss_increase > self.stop_loss_threshold:
                    suggestions.append(OptimizationSuggestion(
                        suggestion_type=OptimizationType.STOP_LOSS,
                        symbol=symbol,
                        reason=f"Position loss increasing: ${unrealized_pnl:.2f} (worst: ${min_unrealized:.2f})",
                        suggested_action="close_position",
                        urgency=UrgencyLevel.HIGH,
                        impact_score=abs(unrealized_pnl),
                        confidence=0.8,
                        metadata={'current_loss': unrealized_pnl, 'worst_loss': min_unrealized}
                    ))
            
            # Time-based stop analysis
            if entry_time:
                if isinstance(entry_time, str):
                    entry_time = datetime.fromisoformat(entry_time)
                elif not isinstance(entry_time, datetime):
                    entry_time = datetime.now()
                
                hold_time = (datetime.now() - entry_time).total_seconds() / 3600
                if hold_time > self.max_hold_time:
                    suggestions.append(OptimizationSuggestion(
                        suggestion_type=OptimizationType.TIME_STOP,
                        symbol=symbol,
                        reason=f"Position held for {hold_time:.1f} hours (max: {self.max_hold_time})",
                        suggested_action="close_position",
                        urgency=UrgencyLevel.MEDIUM,
                        impact_score=hold_time - self.max_hold_time,
                        confidence=0.6,
                        metadata={'hold_time': hold_time, 'max_hold_time': self.max_hold_time}
                    ))
            
            # Position sizing analysis
            position_size = abs(position.get('current_size', 0.0))
            if position_size > 0:
                # Check if position is too large relative to risk
                position_risk = self._calculate_position_risk(position, market_conditions)
                if position_risk > 0.1:  # 10% risk threshold
                    suggestions.append(OptimizationSuggestion(
                        suggestion_type=OptimizationType.POSITION_SIZING,
                        symbol=symbol,
                        reason=f"Position risk too high: {position_risk:.1%}",
                        suggested_action="reduce_position_size",
                        urgency=UrgencyLevel.MEDIUM,
                        impact_score=position_risk * 100,
                        confidence=0.7,
                        metadata={'position_risk': position_risk}
                    ))
            
        except Exception as e:
            logger.error(f"Error analyzing position optimization for {symbol}: {e}")
        
        return suggestions
    
    def _analyze_portfolio_optimization(self, positions: Dict[str, Any],
                                       market_conditions: Dict[str, Any],
                                       risk_metrics: Dict[str, Any],
                                       trade_history: List[Any] = None) -> List[OptimizationSuggestion]:
        """Analyze portfolio-level optimization opportunities"""
        suggestions = []
        
        try:
            # Overall risk analysis
            total_risk = risk_metrics.get('total_risk', 0.5)
            if total_risk > self.risk_threshold:
                suggestions.append(OptimizationSuggestion(
                    suggestion_type=OptimizationType.RISK_REDUCTION,
                    symbol="PORTFOLIO",
                    reason=f"Portfolio risk too high: {total_risk:.1%}",
                    suggested_action="reduce_position_sizes",
                    urgency=UrgencyLevel.HIGH,
                    impact_score=total_risk * 100,
                    confidence=0.9,
                    metadata={'total_risk': total_risk}
                ))
            
            # Concentration risk analysis
            if positions:
                concentration_ratio = self._calculate_concentration_ratio(positions)
                if concentration_ratio > self.max_concentration:
                    suggestions.append(OptimizationSuggestion(
                        suggestion_type=OptimizationType.DIVERSIFICATION,
                        symbol="PORTFOLIO",
                        reason=f"High concentration ratio: {concentration_ratio:.1%}",
                        suggested_action="diversify_positions",
                        urgency=UrgencyLevel.MEDIUM,
                        impact_score=concentration_ratio * 100,
                        confidence=0.8,
                        metadata={'concentration_ratio': concentration_ratio}
                    ))
            
            # Overtrading analysis
            if trade_history:
                recent_trades = self._count_recent_trades(trade_history, hours=24)
                if recent_trades > self.max_daily_trades:
                    suggestions.append(OptimizationSuggestion(
                        suggestion_type=OptimizationType.OVERTRADING,
                        symbol="PORTFOLIO",
                        reason=f"Too many trades today: {recent_trades} (max: {self.max_daily_trades})",
                        suggested_action="reduce_trading_frequency",
                        urgency=UrgencyLevel.MEDIUM,
                        impact_score=recent_trades - self.max_daily_trades,
                        confidence=0.7,
                        metadata={'recent_trades': recent_trades, 'max_daily_trades': self.max_daily_trades}
                    ))
            
            # Market condition adjustments
            volatility = market_conditions.get('volatility', 0.02)
            if volatility > 0.05:  # High volatility
                suggestions.append(OptimizationSuggestion(
                    suggestion_type=OptimizationType.RISK_REDUCTION,
                    symbol="PORTFOLIO",
                    reason=f"High market volatility: {volatility:.1%}",
                    suggested_action="reduce_exposure",
                    urgency=UrgencyLevel.MEDIUM,
                    impact_score=volatility * 100,
                    confidence=0.6,
                    metadata={'volatility': volatility}
                ))
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio optimization: {e}")
        
        return suggestions
    
    def _calculate_optimal_allocation(self, positions: Dict[str, Any],
                                     market_conditions: Dict[str, Any],
                                     risk_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate optimal portfolio allocation"""
        try:
            # For single-instrument system (MNQ), allocation is simplified
            volatility = market_conditions.get('volatility', 0.02)
            risk_level = risk_metrics.get('total_risk', 0.5)
            
            # Base allocation based on volatility
            if volatility > 0.05:  # High volatility
                base_allocation = 0.5
            elif volatility < 0.01:  # Low volatility
                base_allocation = 1.0
            else:
                base_allocation = 0.75
            
            # Adjust for risk level
            risk_adjustment = 1.0 - risk_level
            optimal_allocation = base_allocation * risk_adjustment
            
            # Ensure allocation is within bounds
            optimal_allocation = max(0.1, min(1.0, optimal_allocation))
            
            return {
                'MNQ': optimal_allocation,
                'CASH': 1.0 - optimal_allocation
            }
            
        except Exception as e:
            logger.error(f"Error calculating optimal allocation: {e}")
            return {'MNQ': 0.5, 'CASH': 0.5}
    
    def _calculate_current_allocation(self, positions: Dict[str, Any]) -> Dict[str, float]:
        """Calculate current portfolio allocation"""
        try:
            if not positions:
                return {'CASH': 1.0}
            
            # Calculate total position value (simplified)
            total_position_value = sum(
                abs(pos.get('current_size', 0.0)) for pos in positions.values()
                if isinstance(pos, dict)
            )
            
            # Normalize to allocation weights
            mnq_allocation = min(1.0, total_position_value / 10.0)  # Normalize by typical size
            cash_allocation = max(0.0, 1.0 - mnq_allocation)
            
            return {
                'MNQ': mnq_allocation,
                'CASH': cash_allocation
            }
            
        except Exception as e:
            logger.error(f"Error calculating current allocation: {e}")
            return {'CASH': 1.0}
    
    def _assess_rebalancing_need(self, current_allocation: Dict[str, float],
                                optimal_allocation: Dict[str, float]) -> bool:
        """Assess if portfolio rebalancing is needed"""
        try:
            rebalancing_threshold = 0.1  # 10% threshold
            
            for symbol in optimal_allocation:
                current_weight = current_allocation.get(symbol, 0.0)
                optimal_weight = optimal_allocation.get(symbol, 0.0)
                
                deviation = abs(current_weight - optimal_weight)
                if deviation > rebalancing_threshold:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error assessing rebalancing need: {e}")
            return False
    
    def _calculate_position_risk(self, position: Dict[str, Any], 
                                market_conditions: Dict[str, Any]) -> float:
        """Calculate risk for individual position"""
        try:
            position_size = abs(position.get('current_size', 0.0))
            unrealized_pnl = position.get('unrealized_pnl', 0.0)
            volatility = market_conditions.get('volatility', 0.02)
            
            # Simplified risk calculation
            # Risk = position_size * volatility * price_impact
            risk = position_size * volatility * 0.1  # Simplified
            
            # Adjust for current P&L
            if unrealized_pnl < 0:
                risk *= 1.5  # Increase risk for losing positions
            
            return risk
            
        except Exception as e:
            logger.error(f"Error calculating position risk: {e}")
            return 0.0
    
    def _calculate_concentration_ratio(self, positions: Dict[str, Any]) -> float:
        """Calculate portfolio concentration ratio"""
        try:
            position_sizes = [
                abs(pos.get('current_size', 0.0)) for pos in positions.values()
                if isinstance(pos, dict)
            ]
            
            if not position_sizes:
                return 0.0
            
            total_size = sum(position_sizes)
            max_position = max(position_sizes)
            
            return max_position / total_size if total_size > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating concentration ratio: {e}")
            return 0.0
    
    def _count_recent_trades(self, trade_history: List[Any], hours: int = 24) -> int:
        """Count recent trades within specified hours"""
        try:
            if not trade_history:
                return 0
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_count = 0
            
            for trade in trade_history:
                trade_time = getattr(trade, 'exit_time', None)
                if trade_time:
                    if isinstance(trade_time, (int, float)):
                        trade_time = datetime.fromtimestamp(trade_time)
                    elif isinstance(trade_time, str):
                        trade_time = datetime.fromisoformat(trade_time)
                    
                    if trade_time > cutoff_time:
                        recent_count += 1
            
            return recent_count
            
        except Exception as e:
            logger.error(f"Error counting recent trades: {e}")
            return 0
    
    def get_optimization_summary(self, optimization_result: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary of optimization results"""
        try:
            suggestions = optimization_result.get('optimization_suggestions', [])
            
            # Count by type
            type_counts = {}
            for suggestion in suggestions:
                suggestion_type = suggestion.get('type', 'unknown')
                type_counts[suggestion_type] = type_counts.get(suggestion_type, 0) + 1
            
            # Count by urgency
            urgency_counts = {}
            for suggestion in suggestions:
                urgency = suggestion.get('urgency', 'low')
                urgency_counts[urgency] = urgency_counts.get(urgency, 0) + 1
            
            return {
                'total_suggestions': len(suggestions),
                'suggestions_by_type': type_counts,
                'suggestions_by_urgency': urgency_counts,
                'rebalancing_needed': optimization_result.get('rebalancing_needed', False),
                'risk_level': optimization_result.get('risk_level', 0.5),
                'optimization_score': self._calculate_optimization_score(suggestions)
            }
            
        except Exception as e:
            logger.error(f"Error getting optimization summary: {e}")
            return {}
    
    def _calculate_optimization_score(self, suggestions: List[Dict[str, Any]]) -> float:
        """Calculate optimization score (0-100, higher is better)"""
        try:
            if not suggestions:
                return 100.0  # No suggestions needed = perfect
            
            # Score based on urgency and number of suggestions
            high_urgency = sum(1 for s in suggestions if s.get('urgency') == 'high')
            critical_urgency = sum(1 for s in suggestions if s.get('urgency') == 'critical')
            
            # Penalty for urgent suggestions
            score = 100.0
            score -= critical_urgency * 20  # -20 for each critical
            score -= high_urgency * 10      # -10 for each high
            score -= len(suggestions) * 2   # -2 for each suggestion
            
            return max(0.0, score)
            
        except Exception as e:
            logger.error(f"Error calculating optimization score: {e}")
            return 50.0