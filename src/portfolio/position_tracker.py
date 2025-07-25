# position_tracker.py

import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import deque
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Represents a single trading position"""
    symbol: str
    entry_time: datetime
    entry_price: float
    current_size: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    max_unrealized_pnl: float = 0.0
    min_unrealized_pnl: float = 0.0

@dataclass
class PendingOrder:
    """Represents a pending order"""
    order_id: str
    symbol: str
    action: str
    size: float
    price: float
    timestamp: float
    features: Optional[Dict] = None
    market_data: Optional[Dict] = None
    intelligence_data: Optional[Dict] = None
    decision_data: Optional[Dict] = None

class PositionTracker:
    """
    Tracks and manages trading positions and pending orders.
    
    Responsibilities:
    - Track open positions and their performance
    - Manage pending orders
    - Record position history
    - Calculate position-level metrics
    - Handle position lifecycle events
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.pending_orders: Dict[str, PendingOrder] = {}
        self.position_history = deque(maxlen=1000)
        
        # Position tracking metrics
        self.total_positions_opened = 0
        self.total_positions_closed = 0
        
        logger.info("Position tracker initialized")
    
    def track_positions(self, positions_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Track and update current positions
        
        Args:
            positions_data: Dictionary of position data from broker
            
        Returns:
            Dictionary containing position tracking results
        """
        try:
            # Update existing positions
            for symbol, position_data in positions_data.items():
                if symbol not in self.positions:
                    self._create_new_position(symbol, position_data)
                else:
                    self._update_existing_position(symbol, position_data)
            
            # Remove closed positions
            closed_positions = [
                symbol for symbol in self.positions.keys()
                if symbol not in positions_data or positions_data[symbol].get('size', 0.0) == 0
            ]
            
            for symbol in closed_positions:
                self._close_position(symbol)
            
            # Calculate position metrics
            position_metrics = self._calculate_position_metrics()
            
            return {
                'positions': {k: asdict(v) for k, v in self.positions.items()},
                'position_metrics': position_metrics,
                'position_count': len(self.positions),
                'total_unrealized_pnl': sum(pos.unrealized_pnl for pos in self.positions.values()),
                'total_realized_pnl': sum(pos.realized_pnl for pos in self.positions.values()),
                'pending_orders_count': len(self.pending_orders)
            }
            
        except Exception as e:
            logger.error(f"Error tracking positions: {e}")
            return {
                'positions': {},
                'position_metrics': {},
                'position_count': 0,
                'total_unrealized_pnl': 0.0,
                'total_realized_pnl': 0.0,
                'pending_orders_count': 0
            }
    
    def _create_new_position(self, symbol: str, position_data: Dict[str, Any]):
        """Create a new position entry"""
        position = Position(
            symbol=symbol,
            entry_time=datetime.now(),
            entry_price=position_data.get('entry_price', 0.0),
            current_size=position_data.get('size', 0.0),
            unrealized_pnl=position_data.get('unrealized_pnl', 0.0),
            realized_pnl=position_data.get('realized_pnl', 0.0),
            max_unrealized_pnl=position_data.get('unrealized_pnl', 0.0),
            min_unrealized_pnl=position_data.get('unrealized_pnl', 0.0)
        )
        
        self.positions[symbol] = position
        self.total_positions_opened += 1
        
        logger.info(f"Created new position: {symbol} {position.current_size} @ {position.entry_price:.2f}")
    
    def _update_existing_position(self, symbol: str, position_data: Dict[str, Any]):
        """Update existing position with new data"""
        position = self.positions[symbol]
        
        # Update position data
        position.current_size = position_data.get('size', 0.0)
        position.unrealized_pnl = position_data.get('unrealized_pnl', 0.0)
        position.realized_pnl = position_data.get('realized_pnl', 0.0)
        
        # Update high/low water marks
        position.max_unrealized_pnl = max(position.max_unrealized_pnl, position.unrealized_pnl)
        position.min_unrealized_pnl = min(position.min_unrealized_pnl, position.unrealized_pnl)
    
    def _close_position(self, symbol: str):
        """Close a position and record its performance"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # Record position performance
        self._record_position_performance(position)
        
        # Remove from active positions
        del self.positions[symbol]
        self.total_positions_closed += 1
        
        logger.info(f"Closed position: {symbol} - Final P&L: {position.realized_pnl:.2f}")
    
    def _record_position_performance(self, position: Position):
        """Record position performance in history"""
        hold_time = (datetime.now() - position.entry_time).total_seconds() / 3600  # hours
        
        performance_record = {
            'symbol': position.symbol,
            'entry_time': position.entry_time,
            'close_time': datetime.now(),
            'hold_time_hours': hold_time,
            'entry_price': position.entry_price,
            'size': position.current_size,
            'total_pnl': position.realized_pnl,
            'max_unrealized_pnl': position.max_unrealized_pnl,
            'min_unrealized_pnl': position.min_unrealized_pnl,
            'max_adverse_excursion': abs(position.min_unrealized_pnl) if position.min_unrealized_pnl < 0 else 0,
            'max_favorable_excursion': position.max_unrealized_pnl if position.max_unrealized_pnl > 0 else 0
        }
        
        self.position_history.append(performance_record)
    
    def _calculate_position_metrics(self) -> Dict[str, Any]:
        """Calculate current position metrics"""
        try:
            if not self.positions:
                return {}
            
            positions_list = list(self.positions.values())
            
            # Calculate basic metrics
            total_unrealized = sum(pos.unrealized_pnl for pos in positions_list)
            total_realized = sum(pos.realized_pnl for pos in positions_list)
            total_pnl = total_unrealized + total_realized
            
            # Calculate position concentration
            position_sizes = [abs(pos.current_size) for pos in positions_list]
            total_size = sum(position_sizes)
            max_position_size = max(position_sizes) if position_sizes else 0
            concentration = max_position_size / total_size if total_size > 0 else 0
            
            # Calculate average hold time
            current_time = datetime.now()
            hold_times = [
                (current_time - pos.entry_time).total_seconds() / 3600 
                for pos in positions_list
            ]
            avg_hold_time = sum(hold_times) / len(hold_times) if hold_times else 0
            
            # Performance metrics
            winners = [pos for pos in positions_list if pos.unrealized_pnl > 0]
            losers = [pos for pos in positions_list if pos.unrealized_pnl < 0]
            
            return {
                'total_unrealized_pnl': total_unrealized,
                'total_realized_pnl': total_realized,
                'total_pnl': total_pnl,
                'position_count': len(positions_list),
                'concentration_ratio': concentration,
                'avg_hold_time_hours': avg_hold_time,
                'winners_count': len(winners),
                'losers_count': len(losers),
                'largest_winner': max([pos.unrealized_pnl for pos in positions_list], default=0),
                'largest_loser': min([pos.unrealized_pnl for pos in positions_list], default=0),
                'total_positions_opened': self.total_positions_opened,
                'total_positions_closed': self.total_positions_closed
            }
            
        except Exception as e:
            logger.error(f"Error calculating position metrics: {e}")
            return {}
    
    def add_pending_order(self, order: Any):
        """
        Add a pending order to tracking
        
        Args:
            order: Order object to track
        """
        try:
            order_id = str(getattr(order, 'timestamp', time.time()))
            
            pending_order = PendingOrder(
                order_id=order_id,
                symbol=getattr(order, 'symbol', 'MNQ'),
                action=getattr(order, 'action', 'unknown'),
                size=getattr(order, 'size', 0),
                price=getattr(order, 'price', 0.0),
                timestamp=getattr(order, 'timestamp', time.time()),
                features=getattr(order, 'features', None),
                market_data=getattr(order, 'market_data', None),
                intelligence_data=getattr(order, 'intelligence_data', None),
                decision_data=getattr(order, 'decision_data', None)
            )
            
            self.pending_orders[order_id] = pending_order
            
            logger.info(f"Added pending order: {pending_order.action} {pending_order.size} @ {pending_order.price:.2f}")
            
        except Exception as e:
            logger.error(f"Error adding pending order: {e}")
    
    def complete_trade(self, completion_data: Dict[str, Any]) -> Optional[Any]:
        """
        Complete a trade and create trade object
        
        Args:
            completion_data: Trade completion data
            
        Returns:
            Trade object or None if no pending orders
        """
        try:
            if not self.pending_orders:
                return None
            
            # Find the most recent pending order
            latest_order_id = max(
                self.pending_orders.keys(),
                key=lambda x: self.pending_orders[x].timestamp
            )
            
            order_info = self.pending_orders.pop(latest_order_id)
            
            # Create trade object
            trade = type('Trade', (), {
                'action': order_info.action,
                'size': order_info.size,
                'pnl': completion_data.get('pnl', 0.0),
                'exit_reason': completion_data.get('exit_reason', 'completed'),
                'entry_time': order_info.timestamp,
                'exit_time': time.time(),
                'entry_price': order_info.price,
                'exit_price': completion_data.get('exit_price', order_info.price),
                'exit_account_balance': completion_data.get('account_balance', 0.0),
                'account_risk_pct': completion_data.get('risk_pct', 0.0),
                'features': order_info.features,
                'market_data': order_info.market_data,
                'intelligence_data': order_info.intelligence_data,
                'decision_data': order_info.decision_data,
                'state_features': order_info.decision_data.get('state_features') if order_info.decision_data else None
            })()
            
            logger.info(f"Trade completed: {trade.action} {trade.size}, P&L: {trade.pnl:.2f}")
            return trade
            
        except Exception as e:
            logger.error(f"Error completing trade: {e}")
            return None
    
    def get_position_analytics(self) -> Dict[str, Any]:
        """Get current position analytics"""
        try:
            if not self.positions:
                return {
                    'open_positions': 0,
                    'total_unrealized': 0.0,
                    'avg_unrealized': 0.0,
                    'largest_winner': 0.0,
                    'largest_loser': 0.0
                }
            
            unrealized_pnls = [pos.unrealized_pnl for pos in self.positions.values()]
            
            return {
                'open_positions': len(self.positions),
                'total_unrealized': sum(unrealized_pnls),
                'avg_unrealized': sum(unrealized_pnls) / len(unrealized_pnls),
                'largest_winner': max(unrealized_pnls),
                'largest_loser': min(unrealized_pnls)
            }
            
        except Exception as e:
            logger.error(f"Error calculating position analytics: {e}")
            return {}
    
    def get_total_position_size(self) -> float:
        """Get total position size across all positions"""
        try:
            return sum(pos.current_size for pos in self.positions.values())
        except Exception as e:
            logger.error(f"Error getting total position size: {e}")
            return 0.0
    
    def get_position_count(self) -> int:
        """Get number of pending orders"""
        return len(self.pending_orders)
    
    def get_position_history(self) -> List[Dict[str, Any]]:
        """Get position history"""
        return list(self.position_history)
    
    def get_current_positions(self) -> Dict[str, Position]:
        """Get current positions"""
        return self.positions.copy()
    
    def get_pending_orders(self) -> Dict[str, PendingOrder]:
        """Get pending orders"""
        return self.pending_orders.copy()
    
    def reset_metrics(self):
        """Reset position tracking metrics"""
        self.total_positions_opened = 0
        self.total_positions_closed = 0
        self.position_history.clear()
        logger.info("Position tracker metrics reset")