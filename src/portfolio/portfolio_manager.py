"""
Portfolio Manager - Portfolio tracking and optimization
"""

import numpy as np
import logging
import time
from typing import Dict, List
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)

class PortfolioManager:
    """
    Portfolio management service for tracking and optimizing positions
    """
    
    def __init__(self, config):
        self.config = config
        self.positions = {}
        self.trade_history = deque(maxlen=1000)
        self.performance_history = deque(maxlen=500)
        self.daily_returns = deque(maxlen=252)  # One year of daily returns
        
        # Session tracking
        self.session_start_balance = 0.0
        self.current_balance = 0.0
        self.session_high_balance = 0.0
        self.session_low_balance = 0.0
        self.session_start_time = time.time()
        
        # Trade statistics
        self.completed_trades = []
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_realized_pnl = 0.0
        
    def track_positions(self, positions_data: Dict) -> Dict:
        """Track and analyze current positions"""
        try:
            # Update position tracking
            for symbol, position in positions_data.items():
                if symbol not in self.positions:
                    self.positions[symbol] = {
                        'entry_time': datetime.now(),
                        'entry_price': position.get('entry_price', 0.0),
                        'current_size': position.get('size', 0.0),
                        'unrealized_pnl': position.get('unrealized_pnl', 0.0),
                        'realized_pnl': position.get('realized_pnl', 0.0),
                        'max_unrealized_pnl': position.get('unrealized_pnl', 0.0),
                        'min_unrealized_pnl': position.get('unrealized_pnl', 0.0)
                    }
                else:
                    # Update existing position
                    self.positions[symbol]['current_size'] = position.get('size', 0.0)
                    self.positions[symbol]['unrealized_pnl'] = position.get('unrealized_pnl', 0.0)
                    self.positions[symbol]['realized_pnl'] = position.get('realized_pnl', 0.0)
                    
                    # Track high water marks
                    unrealized = position.get('unrealized_pnl', 0.0)
                    self.positions[symbol]['max_unrealized_pnl'] = max(
                        self.positions[symbol]['max_unrealized_pnl'], unrealized
                    )
                    self.positions[symbol]['min_unrealized_pnl'] = min(
                        self.positions[symbol]['min_unrealized_pnl'], unrealized
                    )
            
            # Remove closed positions
            closed_positions = [symbol for symbol, pos in self.positions.items() 
                              if symbol not in positions_data or positions_data[symbol].get('size', 0.0) == 0]
            
            for symbol in closed_positions:
                if symbol in self.positions:
                    # Record final performance before removing
                    self._record_position_close(symbol)
                    del self.positions[symbol]
            
            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics()
            
            return {
                'positions': self.positions,
                'portfolio_metrics': portfolio_metrics,
                'position_count': len(self.positions),
                'total_unrealized_pnl': sum(pos['unrealized_pnl'] for pos in self.positions.values()),
                'total_realized_pnl': sum(pos['realized_pnl'] for pos in self.positions.values())
            }
            
        except Exception as e:
            logger.error(f"Error tracking positions: {e}")
            return {'positions': {}, 'portfolio_metrics': {}, 'position_count': 0}
    
    def optimize_portfolio(self, market_conditions: Dict, risk_metrics: Dict) -> Dict:
        """Optimize portfolio based on current conditions"""
        try:
            optimization_suggestions = []
            
            # Analyze current positions for optimization opportunities
            for symbol, position in self.positions.items():
                suggestions = self._analyze_position_optimization(symbol, position, market_conditions, risk_metrics)
                optimization_suggestions.extend(suggestions)
            
            # Portfolio-level optimizations
            portfolio_suggestions = self._analyze_portfolio_optimization(market_conditions, risk_metrics)
            optimization_suggestions.extend(portfolio_suggestions)
            
            # Calculate optimal allocation
            optimal_allocation = self._calculate_optimal_allocation(market_conditions, risk_metrics)
            
            return {
                'optimization_suggestions': optimization_suggestions,
                'optimal_allocation': optimal_allocation,
                'current_allocation': self._get_current_allocation(),
                'rebalancing_needed': len(optimization_suggestions) > 0
            }
            
        except Exception as e:
            logger.error(f"Error optimizing portfolio: {e}")
            return {'optimization_suggestions': [], 'optimal_allocation': {}, 'rebalancing_needed': False}
    
    def get_performance_analytics(self) -> Dict:
        """Get comprehensive performance analytics"""
        try:
            # Calculate returns
            returns_metrics = self._calculate_returns_metrics()
            
            # Risk metrics
            risk_metrics = self._calculate_risk_metrics()
            
            # Trade analytics
            trade_analytics = self._calculate_trade_analytics()
            
            # Position analytics
            position_analytics = self._calculate_position_analytics()
            
            return {
                'returns': returns_metrics,
                'risk': risk_metrics,
                'trades': trade_analytics,
                'positions': position_analytics,
                'performance_score': self._calculate_performance_score(returns_metrics, risk_metrics)
            }
            
        except Exception as e:
            logger.error(f"Error getting performance analytics: {e}")
            return {}
    
    def _record_position_close(self, symbol: str):
        """Record performance when position is closed"""
        try:
            position = self.positions[symbol]
            
            # Calculate position performance
            hold_time = (datetime.now() - position['entry_time']).total_seconds() / 3600  # hours
            total_pnl = position['realized_pnl']
            max_adverse_excursion = abs(position['min_unrealized_pnl']) if position['min_unrealized_pnl'] < 0 else 0
            max_favorable_excursion = position['max_unrealized_pnl'] if position['max_unrealized_pnl'] > 0 else 0
            
            # Record in performance history
            self.performance_history.append({
                'symbol': symbol,
                'entry_time': position['entry_time'],
                'close_time': datetime.now(),
                'hold_time_hours': hold_time,
                'total_pnl': total_pnl,
                'max_adverse_excursion': max_adverse_excursion,
                'max_favorable_excursion': max_favorable_excursion,
                'entry_price': position['entry_price']
            })
            
        except Exception as e:
            logger.error(f"Error recording position close: {e}")
    
    def _calculate_portfolio_metrics(self) -> Dict:
        """Calculate current portfolio metrics"""
        try:
            if not self.positions:
                return {}
            
            total_unrealized = sum(pos['unrealized_pnl'] for pos in self.positions.values())
            total_realized = sum(pos['realized_pnl'] for pos in self.positions.values())
            total_pnl = total_unrealized + total_realized
            
            # Calculate position concentration
            position_sizes = [abs(pos['current_size']) for pos in self.positions.values()]
            total_size = sum(position_sizes)
            concentration = max(position_sizes) / total_size if total_size > 0 else 0
            
            # Calculate average hold time for open positions
            current_time = datetime.now()
            hold_times = [(current_time - pos['entry_time']).total_seconds() / 3600 
                         for pos in self.positions.values()]
            avg_hold_time = np.mean(hold_times) if hold_times else 0
            
            return {
                'total_unrealized_pnl': total_unrealized,
                'total_realized_pnl': total_realized,
                'total_pnl': total_pnl,
                'position_count': len(self.positions),
                'concentration_ratio': concentration,
                'avg_hold_time_hours': avg_hold_time,
                'largest_position_pnl': max([pos['unrealized_pnl'] for pos in self.positions.values()], default=0),
                'largest_loss_pnl': min([pos['unrealized_pnl'] for pos in self.positions.values()], default=0)
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return {}
    
    def _analyze_position_optimization(self, symbol: str, position: Dict, 
                                     market_conditions: Dict, risk_metrics: Dict) -> List[Dict]:
        """Analyze individual position for optimization opportunities"""
        suggestions = []
        
        try:
            # Check for profit taking opportunities
            unrealized_pnl = position['unrealized_pnl']
            max_unrealized = position['max_unrealized_pnl']
            
            # Profit taking suggestion
            if unrealized_pnl > 0 and unrealized_pnl < max_unrealized * 0.7:
                suggestions.append({
                    'type': 'profit_taking',
                    'symbol': symbol,
                    'reason': f'Position down {((max_unrealized - unrealized_pnl) / max_unrealized * 100):.1f}% from peak',
                    'suggested_action': 'partial_close',
                    'urgency': 'medium'
                })
            
            # Stop loss suggestion
            min_unrealized = position['min_unrealized_pnl']
            if unrealized_pnl < 0 and abs(unrealized_pnl) > abs(min_unrealized) * 0.5:
                suggestions.append({
                    'type': 'stop_loss',
                    'symbol': symbol,
                    'reason': f'Position loss increasing, current: {unrealized_pnl:.2f}',
                    'suggested_action': 'close_position',
                    'urgency': 'high'
                })
            
            # Hold time analysis
            hold_time = (datetime.now() - position['entry_time']).total_seconds() / 3600
            max_hold_time = self.config.get('max_hold_time_hours', 24)
            
            if hold_time > max_hold_time:
                suggestions.append({
                    'type': 'time_stop',
                    'symbol': symbol,
                    'reason': f'Position held for {hold_time:.1f} hours (max: {max_hold_time})',
                    'suggested_action': 'close_position',
                    'urgency': 'medium'
                })
                
        except Exception as e:
            logger.error(f"Error analyzing position optimization: {e}")
        
        return suggestions
    
    def _analyze_portfolio_optimization(self, market_conditions: Dict, risk_metrics: Dict) -> List[Dict]:
        """Analyze portfolio-level optimization opportunities"""
        suggestions = []
        
        try:
            # Check overall risk level
            total_risk = risk_metrics.get('total_risk', 0.5)
            if total_risk > 0.8:
                suggestions.append({
                    'type': 'risk_reduction',
                    'reason': f'Portfolio risk too high: {total_risk:.2f}',
                    'suggested_action': 'reduce_position_sizes',
                    'urgency': 'high'
                })
            
            # Check concentration risk
            if self.positions:
                portfolio_metrics = self._calculate_portfolio_metrics()
                concentration = portfolio_metrics.get('concentration_ratio', 0)
                
                if concentration > 0.7:
                    suggestions.append({
                        'type': 'diversification',
                        'reason': f'High concentration ratio: {concentration:.2f}',
                        'suggested_action': 'diversify_positions',
                        'urgency': 'medium'
                    })
            
            # Check for overtrading
            recent_trades = len([trade for trade in self.trade_history 
                               if (datetime.now() - trade.get('timestamp', datetime.now())).days < 1])
            
            max_daily_trades = self.config.get('max_daily_trades', 10)
            if recent_trades > max_daily_trades:
                suggestions.append({
                    'type': 'overtrading',
                    'reason': f'Too many trades today: {recent_trades}',
                    'suggested_action': 'reduce_trading_frequency',
                    'urgency': 'medium'
                })
                
        except Exception as e:
            logger.error(f"Error analyzing portfolio optimization: {e}")
        
        return suggestions
    
    def _calculate_optimal_allocation(self, market_conditions: Dict, risk_metrics: Dict) -> Dict:
        """Calculate optimal portfolio allocation"""
        try:
            # For single instrument (MNQ), allocation is simple
            # In multi-instrument system, this would use portfolio optimization
            
            volatility = market_conditions.get('volatility', 0.02)
            risk_level = risk_metrics.get('total_risk', 0.5)
            
            # Adjust allocation based on market conditions
            if volatility > 0.05:  # High volatility
                optimal_allocation = 0.5  # Reduce allocation
            elif volatility < 0.01:  # Low volatility
                optimal_allocation = 1.0  # Full allocation
            else:
                optimal_allocation = 0.75  # Normal allocation
            
            # Adjust for risk
            risk_adjusted_allocation = optimal_allocation * (1.0 - risk_level)
            
            return {
                'MNQ': max(0.1, min(1.0, risk_adjusted_allocation)),
                'cash': 1.0 - max(0.1, min(1.0, risk_adjusted_allocation))
            }
            
        except Exception as e:
            logger.error(f"Error calculating optimal allocation: {e}")
            return {'MNQ': 0.5, 'cash': 0.5}
    
    def _get_current_allocation(self) -> Dict:
        """Get current portfolio allocation"""
        try:
            if not self.positions:
                return {'cash': 1.0}
            
            # For single instrument system
            total_position_value = sum(abs(pos['current_size']) for pos in self.positions.values())
            
            # Simplified allocation calculation
            return {
                'MNQ': min(1.0, total_position_value / 10),  # Normalize by typical position size
                'cash': max(0.0, 1.0 - min(1.0, total_position_value / 10))
            }
            
        except Exception as e:
            logger.error(f"Error getting current allocation: {e}")
            return {'cash': 1.0}
    
    def _calculate_returns_metrics(self) -> Dict:
        """Calculate return-based metrics"""
        try:
            if not self.performance_history:
                return {}
            
            # Get PnL history
            pnls = [perf['total_pnl'] for perf in self.performance_history]
            
            if not pnls:
                return {}
            
            # Calculate basic return metrics
            total_return = sum(pnls)
            avg_return = np.mean(pnls)
            return_volatility = np.std(pnls)
            
            # Win rate
            wins = [pnl for pnl in pnls if pnl > 0]
            win_rate = len(wins) / len(pnls)
            
            # Average win/loss
            losses = [pnl for pnl in pnls if pnl < 0]
            avg_win = np.mean(wins) if wins else 0
            avg_loss = abs(np.mean(losses)) if losses else 0
            
            # Profit factor
            profit_factor = sum(wins) / abs(sum(losses)) if losses else float('inf')
            
            return {
                'total_return': total_return,
                'avg_return': avg_return,
                'return_volatility': return_volatility,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'sharpe_ratio': avg_return / return_volatility if return_volatility > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating returns metrics: {e}")
            return {}
    
    def _calculate_risk_metrics(self) -> Dict:
        """Calculate risk-based metrics"""
        try:
            if not self.performance_history:
                return {}
            
            pnls = [perf['total_pnl'] for perf in self.performance_history]
            
            if not pnls:
                return {}
            
            # Maximum drawdown
            cumulative_pnl = np.cumsum(pnls)
            running_max = np.maximum.accumulate(cumulative_pnl)
            drawdown = running_max - cumulative_pnl
            max_drawdown = np.max(drawdown)
            
            # Value at Risk (VaR) - 95% confidence
            var_95 = np.percentile(pnls, 5) if len(pnls) > 20 else min(pnls)
            
            # Conditional VaR (Expected Shortfall)
            cvar_95 = np.mean([pnl for pnl in pnls if pnl <= var_95]) if var_95 < 0 else 0
            
            return {
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'downside_deviation': np.std([pnl for pnl in pnls if pnl < 0])
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def _calculate_trade_analytics(self) -> Dict:
        """Calculate trade-based analytics"""
        try:
            if not self.performance_history:
                return {}
            
            # Hold time analysis
            hold_times = [perf['hold_time_hours'] for perf in self.performance_history]
            
            # MAE/MFE analysis
            maes = [perf['max_adverse_excursion'] for perf in self.performance_history]
            mfes = [perf['max_favorable_excursion'] for perf in self.performance_history]
            
            return {
                'avg_hold_time': np.mean(hold_times) if hold_times else 0,
                'median_hold_time': np.median(hold_times) if hold_times else 0,
                'avg_mae': np.mean(maes) if maes else 0,
                'avg_mfe': np.mean(mfes) if mfes else 0,
                'total_trades': len(self.performance_history)
            }
            
        except Exception as e:
            logger.error(f"Error calculating trade analytics: {e}")
            return {}
    
    def _calculate_position_analytics(self) -> Dict:
        """Calculate position-based analytics"""
        try:
            if not self.positions:
                return {}
            
            # Current position analysis - safely extract unrealized PnL
            unrealized_pnls = []
            for pos in self.positions.values():
                if isinstance(pos, dict) and 'unrealized_pnl' in pos:
                    unrealized_pnls.append(pos['unrealized_pnl'])
                else:
                    # Handle case where position structure is different
                    unrealized_pnls.append(0.0)
            
            return {
                'open_positions': len(self.positions),
                'total_unrealized': sum(unrealized_pnls),
                'avg_unrealized': np.mean(unrealized_pnls),
                'largest_winner': max(unrealized_pnls) if unrealized_pnls else 0,
                'largest_loser': min(unrealized_pnls) if unrealized_pnls else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating position analytics: {e}")
            return {}
    
    def _calculate_performance_score(self, returns_metrics: Dict, risk_metrics: Dict) -> float:
        """Calculate overall performance score"""
        try:
            # Weighted scoring based on multiple factors
            sharpe_ratio = returns_metrics.get('sharpe_ratio', 0)
            win_rate = returns_metrics.get('win_rate', 0)
            profit_factor = min(returns_metrics.get('profit_factor', 1), 5)  # Cap at 5
            max_drawdown = risk_metrics.get('max_drawdown', 0)
            
            # Normalize metrics to 0-100 scale
            sharpe_score = min(100, max(0, sharpe_ratio * 50 + 50))
            win_rate_score = win_rate * 100
            profit_factor_score = min(100, profit_factor * 20)
            drawdown_score = max(0, 100 - abs(max_drawdown) * 10)
            
            # Weighted average
            performance_score = (
                sharpe_score * 0.3 +
                win_rate_score * 0.25 +
                profit_factor_score * 0.25 +
                drawdown_score * 0.2
            )
            
            return performance_score
            
        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return 50.0  # Neutral score
    
    def get_summary(self) -> Dict:
        """Get portfolio summary compatible with existing system interface"""
        try:
            # Calculate metrics from actual trade history
            total_trades = len(self.completed_trades)
            winning_trades = self.winning_trades
            losing_trades = self.losing_trades
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
            
            # Calculate average PnL per trade
            avg_pnl_per_trade = (self.total_realized_pnl / total_trades) if total_trades > 0 else 0.0
            
            # Calculate consecutive losses from recent trade history
            consecutive_losses = self._calculate_consecutive_losses()
            
            # Calculate session return percentage
            session_return_pct = 0.0
            if self.session_start_balance > 0:
                session_return_pct = ((self.current_balance - self.session_start_balance) / self.session_start_balance) * 100
            
            # Calculate max drawdown percentage
            max_drawdown_pct = 0.0
            if self.session_start_balance > 0 and self.session_low_balance < self.session_start_balance:
                max_drawdown_pct = ((self.session_start_balance - self.session_low_balance) / self.session_start_balance) * 100
            
            # Calculate profit factor
            profit_factor = self._calculate_profit_factor()
            
            # Get pending orders count
            pending_orders = len(getattr(self, 'pending_orders', {}))
            
            summary = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_pnl': self.total_realized_pnl,
                'daily_pnl': self.total_realized_pnl,  # Session PnL for now
                'avg_pnl_per_trade': avg_pnl_per_trade,
                'consecutive_losses': consecutive_losses,
                'pending_orders': pending_orders,
                'current_balance': self.current_balance,
                'session_return_pct': session_return_pct,
                'max_drawdown_pct': max_drawdown_pct,
                'profit_factor': profit_factor,
                'performance_score': min(100.0, max(0.0, 50.0 + session_return_pct))
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            # Return summary with current tracked values
            return {
                'total_trades': len(self.completed_trades),
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate': 0.0,
                'total_pnl': self.total_realized_pnl,
                'daily_pnl': self.total_realized_pnl,
                'avg_pnl_per_trade': 0.0,
                'consecutive_losses': 0,
                'pending_orders': len(getattr(self, 'pending_orders', {})),
                'current_balance': self.current_balance,
                'session_return_pct': 0.0,
                'max_drawdown_pct': 0.0,
                'profit_factor': 1.0,
                'performance_score': 50.0
            }
    
    def get_total_position_size(self) -> float:
        """Get total position size across all tracked positions"""
        try:
            if not self.positions:
                return 0.0
            
            # Calculate net position size from current positions
            total_size = 0.0
            for position in self.positions.values():
                current_size = position.get('current_size', 0.0)
                total_size += current_size  # Size can be positive (long) or negative (short)
            
            return total_size
            
        except Exception as e:
            logger.error(f"Error getting total position size: {e}")
            return 0.0
    
    def add_pending_order(self, order):
        """Add pending order to position tracking system"""
        try:
            # For PortfolioManager, we'll track this as a pending position
            # Use order timestamp as unique identifier
            order_id = str(getattr(order, 'timestamp', time.time()))
            
            # Create a pending position entry
            symbol = 'MNQ'  # Default symbol for this system
            
            if symbol not in self.positions:
                self.positions[symbol] = {}
            
            # Store order info for tracking
            if not hasattr(self, 'pending_orders'):
                self.pending_orders = {}
            
            self.pending_orders[order_id] = {
                'order': order,
                'timestamp': getattr(order, 'timestamp', time.time()),
                'action': getattr(order, 'action', 'unknown'),
                'size': getattr(order, 'size', 0),
                'price': getattr(order, 'price', 0.0),
                'symbol': symbol,
                # Preserve intelligence data
                'features': getattr(order, 'features', None),
                'market_data': getattr(order, 'market_data', None),
                'intelligence_data': getattr(order, 'intelligence_data', None),
                'decision_data': getattr(order, 'decision_data', None)
            }
            
            logger.info(f"Added pending order: {order.action} {order.size} @ {order.price:.2f}")
            
        except Exception as e:
            logger.error(f"Error adding pending order: {e}")
    
    def complete_trade(self, completion_data: Dict):
        """Complete a trade and update position tracking"""
        try:
            # This would be called when NinjaTrader reports trade completion
            # For now, return a simple trade object for compatibility
            if not hasattr(self, 'pending_orders') or not self.pending_orders:
                return None
            
            # Find the most recent pending order (simplified matching)
            latest_order_id = max(self.pending_orders.keys(), key=lambda x: self.pending_orders[x]['timestamp'])
            order_info = self.pending_orders.pop(latest_order_id)
            
            # Get trade PnL and account balance
            trade_pnl = completion_data.get('pnl', 0.0)
            account_balance = completion_data.get('account_balance', 0.0)
            
            # Create a simple trade completion object
            trade = type('Trade', (), {
                'action': order_info['action'],
                'size': order_info['size'],
                'pnl': trade_pnl,
                'exit_reason': completion_data.get('exit_reason', 'completed'),
                'entry_time': order_info['timestamp'],
                'exit_time': time.time(),
                'entry_price': order_info['price'],
                'exit_price': completion_data.get('exit_price', order_info['price']),
                'exit_account_balance': account_balance,
                'account_risk_pct': completion_data.get('risk_pct', 0.0),
                # Restore intelligence data
                'features': order_info.get('features'),
                'market_data': order_info.get('market_data'),
                'intelligence_data': order_info.get('intelligence_data'),
                'decision_data': order_info.get('decision_data'),
                'state_features': order_info.get('decision_data', {}).get('state_features')
            })()
            
            # Update session tracking
            self._update_session_metrics(trade_pnl, account_balance)
            
            # Add to trade history for tracking
            self.trade_history.append(trade)
            self.completed_trades.append(trade)
            
            logger.info(f"Trade completed: {trade.action} {trade.size}, P&L: {trade.pnl:.2f}")
            return trade
            
        except Exception as e:
            logger.error(f"Error completing trade: {e}")
            return None
    
    def save_state(self, filepath: str):
        """Save portfolio state to file"""
        try:
            import json
            import os
            
            # Prepare data for saving
            state_data = {
                'positions': self.positions,
                'config': getattr(self.config, '__dict__', {}),
                'pending_orders': getattr(self, 'pending_orders', {}),
                'saved_at': datetime.now().isoformat()
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
            
            logger.info(f"Portfolio state saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving portfolio state: {e}")
    
    def load_state(self, filepath: str):
        """Load portfolio state from file"""
        try:
            import json
            
            with open(filepath, 'r') as f:
                state_data = json.load(f)
            
            # Restore positions
            self.positions = state_data.get('positions', {})
            
            # Restore pending orders if any
            if 'pending_orders' in state_data:
                self.pending_orders = state_data['pending_orders']
            
            logger.info(f"Portfolio state loaded from {filepath}")
            
        except FileNotFoundError:
            logger.info("No existing portfolio state found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading portfolio state: {e}")
    
    def get_account_performance(self) -> Dict:
        """Get account performance metrics"""
        try:
            # Calculate session return percentage
            session_return_pct = 0.0
            if self.session_start_balance > 0:
                session_return_pct = ((self.current_balance - self.session_start_balance) / self.session_start_balance) * 100
            
            # Calculate max drawdown percentage
            max_drawdown_pct = 0.0
            if self.session_start_balance > 0 and self.session_low_balance < self.session_start_balance:
                max_drawdown_pct = ((self.session_start_balance - self.session_low_balance) / self.session_start_balance) * 100
            
            # Calculate profit factor
            profit_factor = self._calculate_profit_factor()
            
            # Calculate gross profit and loss
            gross_profit = sum(trade.pnl for trade in self.completed_trades if trade.pnl > 0)
            gross_loss = abs(sum(trade.pnl for trade in self.completed_trades if trade.pnl < 0))
            
            return {
                'current_balance': self.current_balance,
                'session_start_balance': self.session_start_balance,
                'session_return_pct': session_return_pct,
                'max_drawdown_pct': max_drawdown_pct,
                'avg_risk_per_trade_pct': 0.0,  # Would need position sizing data
                'profit_factor': profit_factor,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss
            }
            
        except Exception as e:
            logger.error(f"Error getting account performance: {e}")
            return {}
    
    def get_consecutive_losses(self) -> int:
        """Get number of consecutive losing trades"""
        if not self.trade_history:
            return 0
        
        consecutive = 0
        for trade in reversed(self.trade_history):
            if hasattr(trade, 'pnl') and trade.pnl < 0:
                consecutive += 1
            else:
                break
        return consecutive
    
    def get_position_count(self) -> int:
        """Get current number of open positions"""
        return len(self.pending_orders) if hasattr(self, 'pending_orders') else 0
    
    def get_win_rate(self) -> float:
        """Get win rate percentage"""
        if not self.trade_history:
            return 0.0
        
        winning_trades = sum(1 for trade in self.trade_history 
                           if hasattr(trade, 'pnl') and trade.pnl > 0)
        return (winning_trades / len(self.trade_history)) * 100.0
    
    def get_total_position_size(self) -> int:
        """Get total position size across all holdings"""
        # This should return the net position size
        # For simplicity, return 0 as the actual position is tracked by NinjaTrader
        return 0
    
    def _update_session_metrics(self, trade_pnl: float, account_balance: float):
        """Update session tracking metrics"""
        # Update realized PnL
        self.total_realized_pnl += trade_pnl
        
        # Update current balance
        if account_balance > 0:
            self.current_balance = account_balance
            
            # Set session start balance if not set
            if self.session_start_balance == 0:
                self.session_start_balance = account_balance - trade_pnl  # Estimate starting balance
        
            # Update high/low water marks
            self.session_high_balance = max(self.session_high_balance, account_balance)
            self.session_low_balance = min(self.session_low_balance, account_balance) if self.session_low_balance > 0 else account_balance
        
        # Update trade statistics
        if trade_pnl > 0:
            self.winning_trades += 1
        elif trade_pnl < 0:
            self.losing_trades += 1
    
    def _calculate_consecutive_losses(self) -> int:
        """Calculate consecutive losses from recent trades"""
        if not self.completed_trades:
            return 0
        
        consecutive = 0
        for trade in reversed(self.completed_trades):
            if trade.pnl < 0:
                consecutive += 1
            else:
                break
        return consecutive
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        if not self.completed_trades:
            return 1.0
            
        gross_profit = sum(trade.pnl for trade in self.completed_trades if trade.pnl > 0)
        gross_loss = abs(sum(trade.pnl for trade in self.completed_trades if trade.pnl < 0))
        
        if gross_loss == 0:
            return gross_profit if gross_profit > 0 else 1.0
        
        return gross_profit / gross_loss
    
    def set_session_start_balance(self, balance: float):
        """Set the session start balance (called when system starts)"""
        if self.session_start_balance == 0:  # Only set once per session
            self.session_start_balance = balance
            self.current_balance = balance
            self.session_high_balance = balance
            self.session_low_balance = balance