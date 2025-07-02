# portfolio.py

import json
import time
import logging

from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from src.risk.risk_manager import Order

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    entry_time: float
    exit_time: float
    action: str
    entry_price: float
    exit_price: float
    size: int
    pnl: float
    exit_reason: str
    features: Optional[object] = None
    market_data: Optional[object] = None
    intelligence_data: Optional[Dict] = None
    decision_data: Optional[Dict] = None
    stop_used: bool = False
    target_used: bool = False
    state_features: Optional[List] = None
    # Enhanced account tracking
    entry_account_balance: float = 0.0
    exit_account_balance: float = 0.0
    margin_used: float = 0.0
    account_risk_pct: float = 0.0


class Portfolio:
    def __init__(self):
        self.pending_orders: Dict[str, Order] = {}
        self.completed_trades: List[Trade] = []
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.winning_trades = 0
        self.losing_trades = 0
        self.consecutive_losses = 0
        self.last_trade_time = 0.0
        
        # Enhanced account tracking
        self.session_start_balance = 0.0
        self.peak_balance = 0.0
        self.max_drawdown = 0.0
        self.current_margin_usage = 0.0
        self.account_balance_history = deque(maxlen=100)
        
    def add_pending_order(self, order: Order):
        # Use timestamp as key
        key = str(order.timestamp)
        self.pending_orders[key] = order
        
        # Track account state at order entry
        if hasattr(order, 'market_data') and order.market_data:
            self._update_account_metrics(order.market_data)
        
    def complete_trade(self, completion_data: Dict) -> Optional[Trade]:
        # Store completion data for use in stats update
        self._completion_data = completion_data
        
        # Use NinjaTrader's authoritative P&L data
        pnl = completion_data.get('pnl', completion_data.get('final_pnl', 0.0))  # Try 'pnl' first, fallback to 'final_pnl'
        exit_price = completion_data.get('exit_price', 0.0)
        exit_reason = completion_data.get('exit_reason', 'unknown')
        
        # Validate and sync daily P&L with NinjaTrader (authoritative source)
        ninja_daily_pnl = completion_data.get('daily_pnl')
        if ninja_daily_pnl is not None:
            if abs(self.daily_pnl - ninja_daily_pnl) > 0.01:  # Significant difference
                logger.warning(f"DAILY P&L MISMATCH: Portfolio={self.daily_pnl:.2f}, NinjaTrader={ninja_daily_pnl:.2f}")
            self.daily_pnl = ninja_daily_pnl  # Always use NinjaTrader's authoritative value
        
        # Find matching pending order
        matching_order = self._find_matching_order(completion_data)
        if not matching_order:
            # If no matching order found, create a minimal trade record from completion_data
            logger.warning("No matching order found for trade completion - creating minimal trade record")
            trade_dict = {
                'entry_time': completion_data.get('entry_time', time.time()),
                'exit_time': completion_data.get('exit_time', time.time()),
                'action': 'unknown',
                'entry_price': completion_data.get('entry_price', 0.0),
                'exit_price': exit_price,
                'size': completion_data.get('size', 1),
                'pnl': pnl,
                'exit_reason': exit_reason,
                'account_risk_pct': 0.0
            }
            self._update_stats(trade_dict)
            self.completed_trades.append(trade_dict)
            return None
        
        # Enhanced trade creation with account data
        entry_balance = 0.0
        margin_used = 0.0
        
        if hasattr(matching_order, 'market_data') and matching_order.market_data:
            entry_balance = matching_order.market_data.account_balance
            margin_used = matching_order.market_data.margin_used
            
        trade = Trade(
            entry_time=matching_order.timestamp,
            exit_time=time.time(),
            action=matching_order.action,
            entry_price=matching_order.price,
            exit_price=exit_price,
            size=matching_order.size,
            pnl=pnl,
            exit_reason=exit_reason,
            features=getattr(matching_order, 'features', None),
            market_data=getattr(matching_order, 'market_data', None),
            intelligence_data=getattr(matching_order, 'intelligence_data', None),
            decision_data=getattr(matching_order, 'decision_data', None),
            stop_used=exit_reason in ['stop_hit', 'stop_loss'],
            target_used=exit_reason in ['target_hit', 'profit_target'],
            state_features=matching_order.decision_data.get('state_features') 
                         if hasattr(matching_order, 'decision_data') and matching_order.decision_data else None,
            # Enhanced account data
            entry_account_balance=entry_balance,
            exit_account_balance=entry_balance + pnl,  # Estimated
            margin_used=margin_used,
            account_risk_pct=abs(pnl) / max(entry_balance, 1000) if entry_balance > 0 else 0.0
        )
        
        self._update_stats(trade)
        self.completed_trades.append(trade)
        self.last_trade_time = time.time()
        
        return trade
    
    def _find_matching_order(self, completion_data: Dict) -> Optional[Order]:
        # Simple matching - find the most recent order
        # In a real system, you'd match by order ID
        if not self.pending_orders:
            return None
            
        # Get most recent order
        latest_key = max(self.pending_orders.keys(), key=float)
        order = self.pending_orders.pop(latest_key)
        
        return order
    
    def _update_stats(self, trade):
        # Handle both Trade objects and dictionaries
        if isinstance(trade, dict):
            pnl = trade.get('pnl', 0.0)
            exit_time = trade.get('exit_time', time.time())
            exit_account_balance = trade.get('exit_account_balance', 0.0)
        else:
            pnl = getattr(trade, 'pnl', 0.0)
            exit_time = getattr(trade, 'exit_time', time.time())
            exit_account_balance = getattr(trade, 'exit_account_balance', 0.0)
        
        self.total_pnl += pnl
        self.daily_pnl += pnl
        
        # Update account balance tracking - use NinjaTrader data when available
        exit_balance = self._completion_data.get('account_balance', exit_account_balance) if hasattr(self, '_completion_data') else exit_account_balance
        if exit_balance > 0:
            self.account_balance_history.append({
                'timestamp': exit_time,
                'balance': exit_balance,
                'pnl': pnl
            })
            
            # Update peak and drawdown
            if exit_balance > self.peak_balance:
                self.peak_balance = exit_balance
            
            current_drawdown = (self.peak_balance - exit_balance) / self.peak_balance if self.peak_balance > 0 else 0.0
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
        
        if pnl > 0:
            self.winning_trades += 1
            self.consecutive_losses = 0
        else:
            self.losing_trades += 1
            self.consecutive_losses += 1
    
    def _update_account_metrics(self, market_data):
        """Update account metrics from current market data"""
        if hasattr(market_data, 'account_balance'):
            self.account_balance_history.append({
                'timestamp': time.time(),
                'balance': market_data.account_balance,
                'pnl': 0.0
            })
            
            if self.session_start_balance == 0.0:
                self.session_start_balance = market_data.account_balance
                self.peak_balance = market_data.account_balance
        
        if hasattr(market_data, 'margin_used'):
            self.current_margin_usage = market_data.margin_used
    
    def get_win_rate(self) -> float:
        total = self.winning_trades + self.losing_trades
        return self.winning_trades / total if total > 0 else 0.5
    
    def get_position_count(self) -> int:
        return len(self.pending_orders)
    
    def get_total_position_size(self) -> int:
        """Get total position size across all pending orders"""
        total_long = 0
        total_short = 0
        
        for order in self.pending_orders.values():
            if order.action == 'buy':
                total_long += order.size
            elif order.action == 'sell':
                total_short += order.size
        
        # Return net position (positive for long, negative for short)
        return total_long - total_short
    
    def get_position_exposure(self) -> Dict:
        """Get detailed position exposure information"""
        total_long = 0
        total_short = 0
        
        for order in self.pending_orders.values():
            if order.action == 'buy':
                total_long += order.size
            elif order.action == 'sell':
                total_short += order.size
        
        return {
            'total_long': total_long,
            'total_short': total_short,
            'net_position': total_long - total_short,
            'gross_exposure': total_long + total_short,
            'position_count': len(self.pending_orders)
        }
    
    def get_consecutive_losses(self) -> int:
        return self.consecutive_losses
    
    def get_recent_trade_count(self, minutes: int) -> int:
        cutoff_time = time.time() - (minutes * 60)
        count = 0
        for trade in self.completed_trades:
            # Handle both Trade objects and dictionaries
            if isinstance(trade, dict):
                entry_time = trade.get('entry_time', 0.0)
            else:
                entry_time = getattr(trade, 'entry_time', 0.0)
            
            if entry_time > cutoff_time:
                count += 1
        return count
    
    def get_account_performance(self) -> Dict:
        """Get account-specific performance metrics"""
        if not self.account_balance_history:
            return {}
        
        recent_balance = self.account_balance_history[-1]['balance'] if self.account_balance_history else 0
        
        # Calculate various account metrics
        session_return = 0.0
        if self.session_start_balance > 0:
            session_return = (recent_balance - self.session_start_balance) / self.session_start_balance
        
        # Risk-adjusted metrics
        avg_risk_per_trade = 0.0
        if self.completed_trades:
            total_risk = 0.0
            for trade in self.completed_trades:
                # Handle both Trade objects and dictionary objects for backward compatibility
                if isinstance(trade, dict):
                    risk_pct = trade.get('account_risk_pct', 0.0)
                else:
                    risk_pct = getattr(trade, 'account_risk_pct', 0.0)
                total_risk += risk_pct
            avg_risk_per_trade = total_risk / len(self.completed_trades)
        
        # Profit factor - handle both Trade objects and dictionaries
        gross_profit = 0.0
        gross_loss = 0.0
        
        for trade in self.completed_trades:
            # Handle both Trade objects and dictionary objects for backward compatibility
            if isinstance(trade, dict):
                pnl = trade.get('pnl', 0.0)
            else:
                pnl = getattr(trade, 'pnl', 0.0)
            
            if pnl > 0:
                gross_profit += pnl
            elif pnl < 0:
                gross_loss += abs(pnl)
        profit_factor = gross_profit / max(gross_loss, 1.0)
        
        return {
            'current_balance': recent_balance,
            'session_start_balance': self.session_start_balance,
            'session_return_pct': session_return * 100,
            'peak_balance': self.peak_balance,
            'max_drawdown_pct': self.max_drawdown * 100,
            'current_margin_usage': self.current_margin_usage,
            'avg_risk_per_trade_pct': avg_risk_per_trade * 100,
            'profit_factor': profit_factor,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }
    
    def get_summary(self) -> Dict:
        total_trades = len(self.completed_trades)
        account_perf = self.get_account_performance()
        
        summary = {
            'total_trades': total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.get_win_rate(),
            'total_pnl': self.total_pnl,
            'daily_pnl': self.daily_pnl,  # Updated from NinjaTrader's authoritative daily_pnl
            'avg_pnl_per_trade': self.total_pnl / max(1, total_trades),
            'consecutive_losses': self.consecutive_losses,
            'pending_orders': len(self.pending_orders)
        }
        
        # Add account performance metrics
        summary.update(account_perf)
        
        return summary
    
    def sync_position_with_ninjatrader(self, ninja_position: int):
        """Sync portfolio position tracking with NinjaTrader's authoritative position"""
        logger.info(f"Syncing portfolio position to match NinjaTrader: {ninja_position}")
        # This would require more complex position tracking
        # For now, we log the sync event for monitoring
        pass
    
    def get_total_position_size(self) -> int:
        """Get current total position size from portfolio tracking"""
        # Calculate from pending orders (approximation)
        total_size = 0
        for order in self.pending_orders.values():
            if order.action == 'buy':
                total_size += order.size
            elif order.action == 'sell':
                total_size -= order.size
        return abs(total_size)
    
    def save_state(self, filepath: str):
        # Convert trades to serializable format
        trades_data = []
        for trade in self.completed_trades:
            trade_dict = asdict(trade)
            # Remove non-serializable fields
            trade_dict.pop('features', None)
            trade_dict.pop('market_data', None)
            trades_data.append(trade_dict)
        
        data = {
            'completed_trades': trades_data,
            'total_pnl': self.total_pnl,
            'daily_pnl': self.daily_pnl,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'consecutive_losses': self.consecutive_losses,
            # Enhanced account data
            'session_start_balance': self.session_start_balance,
            'peak_balance': self.peak_balance,
            'max_drawdown': self.max_drawdown,
            'current_margin_usage': self.current_margin_usage,
            'account_balance_history': list(self.account_balance_history),
            'saved_at': datetime.now().isoformat()
        }
        
        import os, numpy as np
        
        def _np_encoder(obj):
            """Convert NumPy scalars/arrays to vanilla Python types."""
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"{type(obj)} is not JSON serializable")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=_np_encoder)
    
    def load_state(self, filepath: str):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Restore basic stats
            self.total_pnl = data.get('total_pnl', 0.0)
            self.daily_pnl = data.get('daily_pnl', 0.0)
            self.winning_trades = data.get('winning_trades', 0)
            self.losing_trades = data.get('losing_trades', 0)
            self.consecutive_losses = data.get('consecutive_losses', 0)
            
            # Restore enhanced account data
            self.session_start_balance = data.get('session_start_balance', 0.0)
            self.peak_balance = data.get('peak_balance', 0.0)
            self.max_drawdown = data.get('max_drawdown', 0.0)
            self.current_margin_usage = data.get('current_margin_usage', 0.0)
            
            # Restore account history
            history_data = data.get('account_balance_history', [])
            self.account_balance_history = deque(history_data, maxlen=100)
            
            # Restore recent trades (for statistics)
            trades_data = data.get('completed_trades', [])
            self.completed_trades = []
            
            for trade_dict in trades_data[-100:]:  # Keep only recent 100 trades
                trade = Trade(**trade_dict)
                self.completed_trades.append(trade)
                
        except FileNotFoundError:
            pass  # Start fresh