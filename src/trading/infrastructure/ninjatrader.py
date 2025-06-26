"""
NinjaTrader Infrastructure - TCP bridge implementation
"""

import json
import logging
from typing import Dict

from ..domain.repositories import TradingRepository, ExecutionResult
from ..domain.models import Trade, Position, Account
from src.communication.tcp_bridge import TCPBridge

logger = logging.getLogger(__name__)

class NinjaTraderRepository(TradingRepository):
    """
    NinjaTrader implementation of trading repository using TCP bridge
    """
    
    def __init__(self, config):
        self.config = config
        self.tcp_bridge = TCPBridge()
        self.last_account_data = None
        self.last_positions = {}
    
    def execute_trade(self, trade: Trade) -> ExecutionResult:
        """Execute trade via NinjaTrader TCP bridge"""
        try:
            # Prepare trade signal for NinjaTrader
            trade_signal = {
                'action': trade.action.value,
                'size': trade.size,
                'timestamp': trade.timestamp.isoformat() if trade.timestamp else None,
                'trade_id': trade.id
            }
            
            # Send trade signal
            result = self.tcp_bridge.send_trade_signal(trade_signal)
            
            if result and result.get('success', False):
                return ExecutionResult(
                    success=True,
                    execution_price=result.get('execution_price', 0.0),
                    pnl=result.get('pnl', 0.0),
                    duration=result.get('duration', 0.0)
                )
            else:
                error_msg = result.get('error', 'Unknown execution error') if result else 'No response from NinjaTrader'
                return ExecutionResult(
                    success=False,
                    error=error_msg
                )
                
        except Exception as e:
            logger.error(f"Error executing trade via NinjaTrader: {e}")
            return ExecutionResult(
                success=False,
                error=str(e)
            )
    
    def get_account_data(self) -> Account:
        """Get account data from NinjaTrader"""
        try:
            account_data = self.tcp_bridge.get_account_data()
            
            if account_data:
                self.last_account_data = Account(
                    cash=account_data.get('cash', 0.0),
                    buying_power=account_data.get('buying_power', 0.0),
                    total_value=account_data.get('total_value', 0.0),
                    margin_used=account_data.get('margin_used', 0.0),
                    day_trading_buying_power=account_data.get('day_trading_buying_power', 0.0)
                )
                return self.last_account_data
            else:
                # Return cached data if available
                if self.last_account_data:
                    return self.last_account_data
                else:
                    return Account(0.0, 0.0, 0.0, 0.0, 0.0)
                    
        except Exception as e:
            logger.error(f"Error getting account data: {e}")
            return Account(0.0, 0.0, 0.0, 0.0, 0.0)
    
    def get_current_positions(self) -> Dict[str, Position]:
        """Get current positions from NinjaTrader"""
        try:
            positions_data = self.tcp_bridge.get_positions()
            
            if positions_data:
                positions = {}
                for symbol, pos_data in positions_data.items():
                    positions[symbol] = Position(
                        symbol=symbol,
                        size=pos_data.get('size', 0.0),
                        entry_price=pos_data.get('entry_price', 0.0),
                        current_price=pos_data.get('current_price', 0.0),
                        unrealized_pnl=pos_data.get('unrealized_pnl', 0.0),
                        realized_pnl=pos_data.get('realized_pnl', 0.0)
                    )
                
                self.last_positions = positions
                return positions
            else:
                return self.last_positions
                
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return self.last_positions
    
    def get_market_data(self) -> Dict:
        """Get market data from NinjaTrader"""
        try:
            return self.tcp_bridge.get_market_data()
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {}