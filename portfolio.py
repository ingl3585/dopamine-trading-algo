# portfolio.py

import json
import time

from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from risk_manager import Order

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
        
    def add_pending_order(self, order: Order):
        # Use timestamp as key
        key = str(order.timestamp)
        self.pending_orders[key] = order
        
    def complete_trade(self, completion_data: Dict) -> Optional[Trade]:
        pnl = completion_data.get('final_pnl', 0.0)
        exit_price = completion_data.get('exit_price', 0.0)
        exit_reason = completion_data.get('exit_reason', 'unknown')
        
        # Find matching pending order
        matching_order = self._find_matching_order(completion_data)
        if not matching_order:
            return None
            
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
                         if hasattr(matching_order, 'decision_data') and matching_order.decision_data else None
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
    
    def _update_stats(self, trade: Trade):
        self.total_pnl += trade.pnl
        self.daily_pnl += trade.pnl
        
        if trade.pnl > 0:
            self.winning_trades += 1
            self.consecutive_losses = 0
        else:
            self.losing_trades += 1
            self.consecutive_losses += 1
    
    def get_win_rate(self) -> float:
        total = self.winning_trades + self.losing_trades
        return self.winning_trades / total if total > 0 else 0.5
    
    def get_position_count(self) -> int:
        return len(self.pending_orders)
    
    def get_consecutive_losses(self) -> int:
        return self.consecutive_losses
    
    def get_recent_trade_count(self, minutes: int) -> int:
        cutoff_time = time.time() - (minutes * 60)
        return sum(1 for trade in self.completed_trades 
                  if trade.entry_time > cutoff_time)
    
    def get_summary(self) -> Dict:
        total_trades = len(self.completed_trades)
        
        return {
            'total_trades': total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.get_win_rate(),
            'total_pnl': self.total_pnl,
            'daily_pnl': self.daily_pnl,
            'avg_pnl_per_trade': self.total_pnl / max(1, total_trades),
            'consecutive_losses': self.consecutive_losses,
            'pending_orders': len(self.pending_orders)
        }
    
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
            'saved_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
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
            
            # Restore recent trades (for statistics)
            trades_data = data.get('completed_trades', [])
            self.completed_trades = []
            
            for trade_dict in trades_data[-100:]:  # Keep only recent 100 trades
                trade = Trade(**trade_dict)
                self.completed_trades.append(trade)
                
        except FileNotFoundError:
            pass  # Start fresh