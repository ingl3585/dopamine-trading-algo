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
    
    def __add__(self, other):
        """Handle arithmetic operations - return pnl value for Trade + number operations"""
        if isinstance(other, (int, float)):
            return self.pnl + other
        elif isinstance(other, Trade):
            return self.pnl + other.pnl
        else:
            raise TypeError(f"unsupported operand type(s) for +: 'Trade' and '{type(other).__name__}'")
    
    def __radd__(self, other):
        """Handle reverse addition (number + Trade)"""
        if isinstance(other, (int, float)):
            return other + self.pnl
        else:
            raise TypeError(f"unsupported operand type(s) for +: '{type(other).__name__}' and 'Trade'")
    
    def __float__(self):
        """Allow Trade to be converted to float (returns pnl)"""
        return float(self.pnl)
    
    def __int__(self):
        """Allow Trade to be converted to int (returns pnl as int)"""
        return int(self.pnl)


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
        """
        Add pending order with defensive validation to ensure proper Order objects.
        
        This method implements clean validation by:
        - Ensuring the order is a proper Order object, not a dict
        - Converting dict orders to Order objects if necessary
        - Validating required fields are present
        """
        try:
            # Defensive validation: ensure we have a proper Order object
            if isinstance(order, dict):
                logger.warning("Received dict instead of Order object, converting to Order")
                order = self._deserialize_order(order)
            elif not hasattr(order, 'action') or not hasattr(order, 'size'):
                logger.error(f"Invalid order object: missing required attributes")
                return
            
            # Validate timestamp for key generation
            timestamp = getattr(order, 'timestamp', time.time())
            if not timestamp or timestamp <= 0:
                timestamp = time.time()
                order.timestamp = timestamp
            
            # Use timestamp as key
            key = str(timestamp)
            self.pending_orders[key] = order
            
            # Track account state at order entry
            if hasattr(order, 'market_data') and order.market_data:
                self._update_account_metrics(order.market_data)
                
            logger.debug(f"Added pending order: {order.action} {order.size} @ {order.price}")
            
        except Exception as e:
            logger.error(f"Error adding pending order: {e}")
            # Don't add invalid orders to prevent corruption
        
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
        
        # Enhanced trade creation with account data and defensive validation
        entry_balance = 0.0
        margin_used = 0.0
        
        # Defensive validation for matching_order attributes
        if isinstance(matching_order, dict):
            order_timestamp = matching_order.get('timestamp', time.time())
            order_action = matching_order.get('action', 'unknown')
            order_price = matching_order.get('price', 0.0)
            order_size = matching_order.get('size', 0)
            # Dict orders won't have market_data
            order_market_data = None
        else:
            order_timestamp = getattr(matching_order, 'timestamp', time.time())
            order_action = getattr(matching_order, 'action', 'unknown')
            order_price = getattr(matching_order, 'price', 0.0)
            order_size = getattr(matching_order, 'size', 0)
            order_market_data = getattr(matching_order, 'market_data', None)
        
        if order_market_data and hasattr(order_market_data, 'account_balance'):
            entry_balance = order_market_data.account_balance
            margin_used = getattr(order_market_data, 'margin_used', 0.0)
            
        trade = Trade(
            entry_time=order_timestamp,
            exit_time=time.time(),
            action=order_action,
            entry_price=order_price,
            exit_price=exit_price,
            size=order_size,
            pnl=pnl,
            exit_reason=exit_reason,
            features=getattr(matching_order, 'features', None) if not isinstance(matching_order, dict) else None,
            market_data=order_market_data,
            intelligence_data=getattr(matching_order, 'intelligence_data', None) if not isinstance(matching_order, dict) else None,
            decision_data=getattr(matching_order, 'decision_data', None) if not isinstance(matching_order, dict) else None,
            stop_used=exit_reason in ['stop_hit', 'stop_loss'],
            target_used=exit_reason in ['target_hit', 'profit_target'],
            state_features=self._extract_state_features(matching_order),
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
    
    def _extract_state_features(self, order) -> Optional[List[float]]:
        """Safely extract state_features from order with multiple fallback strategies"""
        try:
            # Primary path: decision_data.state_features
            if hasattr(order, 'decision_data') and order.decision_data:
                state_features = order.decision_data.get('state_features')
                if state_features is not None:
                    logger.debug(f"STATE_FEATURES_TRACE: Successfully extracted {len(state_features) if isinstance(state_features, list) else 'not_list'} features from decision_data")
                    return state_features
            
            # Fallback 1: features attribute
            if hasattr(order, 'features') and order.features is not None:
                # If features is already a list, return it
                if isinstance(order.features, list):
                    return order.features
                # If features is an object with state_features, extract it
                elif hasattr(order.features, 'state_features'):
                    return getattr(order.features, 'state_features', None)
            
            # Fallback 2: direct state_features attribute
            if hasattr(order, 'state_features'):
                return getattr(order, 'state_features', None)
            
            # No state_features found - log debugging info
            logger.warning(f"STATE_FEATURES_TRACE: No state_features found in order. Order type: {type(order)}, has decision_data: {hasattr(order, 'decision_data')}, has features: {hasattr(order, 'features')}, has state_features: {hasattr(order, 'state_features')}")
            if hasattr(order, 'decision_data') and order.decision_data:
                logger.debug(f"Decision_data keys: {list(order.decision_data.keys())}")
            return None
            
        except Exception as e:
            logger.warning(f"Error extracting state_features from order: {e}")
            return None
    
    def _safe_json_serializer(self, obj):
        """Safe JSON serializer for testing serializability during save_state"""
        import numpy as np
        
        # Handle NumPy types
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Handle torch tensors
        if hasattr(obj, 'detach') and hasattr(obj, 'cpu') and hasattr(obj, 'numpy'):
            try:
                return obj.detach().cpu().numpy().tolist()
            except:
                return str(obj)
        
        # Handle other complex objects
        return str(obj)
    
    def get_win_rate(self) -> float:
        total = self.winning_trades + self.losing_trades
        return self.winning_trades / total if total > 0 else 0.5
    
    def get_position_count(self) -> int:
        return len(self.pending_orders)
    
    def get_total_position_size(self) -> int:
        """Get total position size across all pending orders with defensive validation"""
        total_long = 0
        total_short = 0
        
        for order in self.pending_orders.values():
            # Defensive validation to handle both Order objects and dicts
            if isinstance(order, dict):
                action = order.get('action', 'unknown')
                size = order.get('size', 0)
            else:
                action = getattr(order, 'action', 'unknown')
                size = getattr(order, 'size', 0)
            
            try:
                size = int(size)
                if action == 'buy':
                    total_long += size
                elif action == 'sell':
                    total_short += size
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid order size in pending_orders: {size}, error: {e}")
                continue
        
        # Return net position (positive for long, negative for short)
        return total_long - total_short
    
    def get_position_exposure(self) -> Dict:
        """Get detailed position exposure information with defensive validation"""
        total_long = 0
        total_short = 0
        
        for order in self.pending_orders.values():
            # Defensive validation to handle both Order objects and dicts
            if isinstance(order, dict):
                action = order.get('action', 'unknown')
                size = order.get('size', 0)
            else:
                action = getattr(order, 'action', 'unknown')
                size = getattr(order, 'size', 0)
            
            try:
                size = int(size)
                if action == 'buy':
                    total_long += size
                elif action == 'sell':
                    total_short += size
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid order size in position exposure calculation: {size}, error: {e}")
                continue
        
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
    
    def get_current_position_size(self) -> int:
        """Get current total position size from portfolio tracking with defensive validation"""
        # Calculate from pending orders (approximation)
        total_size = 0
        for order in self.pending_orders.values():
            # Defensive validation to handle both Order objects and dicts
            if isinstance(order, dict):
                action = order.get('action', 'unknown')
                size = order.get('size', 0)
            else:
                action = getattr(order, 'action', 'unknown')
                size = getattr(order, 'size', 0)
            
            try:
                size = int(size)
                if action == 'buy':
                    total_size += size
                elif action == 'sell':
                    total_size -= size
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid order size in current position calculation: {size}, error: {e}")
                continue
        
        return abs(total_size)
    
    def _serialize_order(self, order: Order) -> Dict:
        """Convert Order object to serializable dictionary with proper type handling"""
        if isinstance(order, dict):
            # Already a dictionary, validate it has required fields
            required_fields = ['action', 'size', 'price', 'timestamp']
            for field in required_fields:
                if field not in order:
                    logger.warning(f"Order dict missing required field: {field}")
                    order[field] = 0.0 if field in ['size', 'price', 'timestamp'] else 'unknown'
            return order
        
        try:
            # Convert Order dataclass to dict, handling complex objects
            order_dict = {
                'action': getattr(order, 'action', 'unknown'),
                'size': getattr(order, 'size', 0),
                'price': getattr(order, 'price', 0.0),
                'stop_price': getattr(order, 'stop_price', 0.0),
                'target_price': getattr(order, 'target_price', 0.0),
                'timestamp': getattr(order, 'timestamp', time.time()),
                'confidence': getattr(order, 'confidence', 0.0),
                # Mark as serialized for reconstruction
                '_is_serialized_order': True
            }
            
            # Handle complex objects that shouldn't be serialized
            complex_fields = ['features', 'market_data', 'intelligence_data', 'decision_data']
            for field in complex_fields:
                if hasattr(order, field):
                    value = getattr(order, field)
                    if value is not None:
                        # Store type information for reconstruction
                        order_dict[f'{field}_type'] = type(value).__name__
                        # Don't serialize the actual complex object
                        order_dict[field] = None
            
            return order_dict
            
        except Exception as e:
            logger.error(f"Error serializing order: {e}")
            # Return minimal valid order dict
            return {
                'action': 'unknown',
                'size': 0,
                'price': 0.0,
                'timestamp': time.time(),
                '_is_serialized_order': True,
                '_serialization_error': str(e)
            }
    
    def _deserialize_order(self, order_dict: Dict) -> Order:
        """Convert serialized dictionary back to Order object with proper validation"""
        try:
            # Validate required fields
            required_fields = ['action', 'size', 'price', 'timestamp']
            for field in required_fields:
                if field not in order_dict:
                    logger.warning(f"Missing required field '{field}' in order dict, using default")
                    if field == 'action':
                        order_dict[field] = 'unknown'
                    elif field in ['size', 'price', 'timestamp']:
                        order_dict[field] = 0.0
            
            # Create Order object with validated data
            order = Order(
                action=order_dict.get('action', 'unknown'),
                size=int(order_dict.get('size', 0)),
                price=float(order_dict.get('price', 0.0)),
                stop_price=float(order_dict.get('stop_price', 0.0)),
                target_price=float(order_dict.get('target_price', 0.0)),
                timestamp=float(order_dict.get('timestamp', time.time())),
                confidence=float(order_dict.get('confidence', 0.0)),
                # Complex objects remain None after deserialization
                features=None,
                market_data=None,
                intelligence_data=None,
                decision_data=None
            )
            
            return order
            
        except Exception as e:
            logger.error(f"Error deserializing order: {e}")
            # Return minimal valid Order object
            return Order(
                action='unknown',
                size=0,
                price=0.0,
                timestamp=time.time()
            )
    
    def get_persistent_state(self) -> Dict:
        """
        Get serializable portfolio state excluding transient data like pending_orders.
        
        This method implements proper state management by:
        - Excluding transient pending_orders that should not persist across restarts
        - Converting complex objects to serializable format
        - Providing only essential state data for reconstruction
        
        Returns:
            Dict: Serializable state data for persistence
        """
        try:
            # Serialize completed trades safely
            trades_data = []
            for trade in self.completed_trades:
                if isinstance(trade, dict):
                    trade_dict = trade.copy()
                else:
                    trade_dict = asdict(trade)
                
                # Remove non-serializable fields
                non_serializable_fields = [
                    'features', 'market_data', 'intelligence_data', 'decision_data'
                ]
                for field in non_serializable_fields:
                    trade_dict.pop(field, None)
                
                # Validate serializability of remaining fields
                for key, value in list(trade_dict.items()):
                    if value is None:
                        continue
                    try:
                        json.dumps(value, default=self._safe_json_serializer)
                    except (TypeError, ValueError):
                        logger.warning(f"Removing non-serializable field '{key}' from trade data")
                        trade_dict.pop(key, None)
                
                trades_data.append(trade_dict)
            
            # Build core persistent state
            persistent_state = {
                # Core trading statistics
                'total_pnl': self.total_pnl,
                'daily_pnl': self.daily_pnl,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'consecutive_losses': self.consecutive_losses,
                'last_trade_time': self.last_trade_time,
                
                # Account tracking data
                'session_start_balance': self.session_start_balance,
                'peak_balance': self.peak_balance,
                'max_drawdown': self.max_drawdown,
                'current_margin_usage': self.current_margin_usage,
                'account_balance_history': list(self.account_balance_history),
                
                # Trade history (cleaned)
                'completed_trades': trades_data,
                
                # Metadata
                'state_version': '1.0',
                'saved_at': datetime.now().isoformat()
            }
            
            # Note: pending_orders are intentionally excluded as they are transient
            # and should not persist across system restarts
            
            return persistent_state
            
        except Exception as e:
            logger.error(f"Error creating persistent state: {e}")
            # Return minimal valid state
            return {
                'total_pnl': 0.0,
                'daily_pnl': 0.0,
                'winning_trades': 0,
                'losing_trades': 0,
                'consecutive_losses': 0,
                'state_version': '1.0',
                'saved_at': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def load_persistent_state(self, state: Dict) -> None:
        """
        Load portfolio state from serialized data with proper object reconstruction.
        
        This method implements clean state restoration by:
        - Validating state data integrity
        - Properly reconstructing complex objects
        - Maintaining backward compatibility
        - Ensuring pending_orders remains empty (transient data)
        
        Args:
            state: Serialized state dictionary from get_persistent_state()
        """
        try:
            if not isinstance(state, dict):
                logger.error(f"Invalid state data type: {type(state)}")
                return
            
            # Validate state version for compatibility
            state_version = state.get('state_version', '1.0')
            if state_version != '1.0':
                logger.warning(f"Loading state version {state_version}, current version is 1.0")
            
            # Restore core statistics with validation
            self.total_pnl = float(state.get('total_pnl', 0.0))
            self.daily_pnl = float(state.get('daily_pnl', 0.0))
            self.winning_trades = int(state.get('winning_trades', 0))
            self.losing_trades = int(state.get('losing_trades', 0))
            self.consecutive_losses = int(state.get('consecutive_losses', 0))
            self.last_trade_time = float(state.get('last_trade_time', 0.0))
            
            # Restore account tracking data
            self.session_start_balance = float(state.get('session_start_balance', 0.0))
            self.peak_balance = float(state.get('peak_balance', 0.0))
            self.max_drawdown = float(state.get('max_drawdown', 0.0))
            self.current_margin_usage = float(state.get('current_margin_usage', 0.0))
            
            # Restore account balance history
            history_data = state.get('account_balance_history', [])
            if isinstance(history_data, list):
                self.account_balance_history = deque(history_data, maxlen=100)
            else:
                logger.warning("Invalid account_balance_history format, initializing empty")
                self.account_balance_history = deque(maxlen=100)
            
            # Restore completed trades
            trades_data = state.get('completed_trades', [])
            self.completed_trades = []
            
            for trade_dict in trades_data[-100:]:  # Keep only recent 100 trades
                try:
                    if isinstance(trade_dict, dict):
                        # Reconstruct Trade object from dictionary
                        trade = Trade(**trade_dict)
                        self.completed_trades.append(trade)
                    else:
                        logger.warning(f"Invalid trade data format: {type(trade_dict)}")
                except Exception as e:
                    logger.warning(f"Error reconstructing trade: {e}")
                    continue
            
            # Ensure pending_orders is empty (transient data should not persist)
            self.pending_orders = {}
            
            logger.info(f"Successfully loaded portfolio state: {len(self.completed_trades)} trades, "
                       f"total_pnl={self.total_pnl:.2f}, daily_pnl={self.daily_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error loading persistent state: {e}")
            # Initialize with safe defaults
            self.pending_orders = {}
            self.completed_trades = []
            self.daily_pnl = 0.0
            self.total_pnl = 0.0
            self.winning_trades = 0
            self.losing_trades = 0
            self.consecutive_losses = 0
    
    def save_state(self, filepath: str):
        # Convert trades to serializable format with comprehensive field cleaning
        trades_data = []
        for trade in self.completed_trades:
            if isinstance(trade, dict):
                # Handle dictionary trades
                trade_dict = trade.copy()
            else:
                # Handle Trade dataclass objects
                trade_dict = asdict(trade)
            
            # Remove non-serializable fields comprehensively
            non_serializable_fields = [
                'features', 'market_data', 'intelligence_data', 'decision_data'
            ]
            for field in non_serializable_fields:
                trade_dict.pop(field, None)
            
            # Clean nested data structures that might contain non-serializable objects
            for key, value in list(trade_dict.items()):
                if value is None:
                    continue
                try:
                    # Test JSON serializability
                    json.dumps(value, default=self._safe_json_serializer)
                except (TypeError, ValueError):
                    logger.warning(f"Removing non-serializable field '{key}' from trade data")
                    trade_dict.pop(key, None)
            
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
        
        # Use the enhanced safe JSON serializer
        def _comprehensive_json_encoder(obj):
            """Enhanced JSON encoder that handles various non-serializable types safely."""
            # Handle NumPy types
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            
            # Handle torch tensors
            if hasattr(obj, 'detach') and hasattr(obj, 'cpu') and hasattr(obj, 'numpy'):
                try:
                    return obj.detach().cpu().numpy().tolist()
                except:
                    return str(obj)
            
            # Handle complex objects with string representation
            if hasattr(obj, '__dict__'):
                return str(obj)
            
            # Handle other iterables
            if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
                try:
                    return list(obj)
                except:
                    return str(obj)
            
            # Fallback to string representation
            return str(obj)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=_comprehensive_json_encoder)
    
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