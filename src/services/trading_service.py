"""
Trading Domain Services - Core business logic
"""

import logging
from typing import Dict, Optional
from datetime import datetime

from src.shared.types import TradeDecision, TradeOutcome, AccountInfo
from src.data_models.trading_domain_models import Trade, Position, Account, TradeAction, TradeStatus
from src.repositories.trading_repository import TradingRepository

logger = logging.getLogger(__name__)

class TradingService:
    """
    Core trading service that coordinates trade execution and account management
    """
    
    def __init__(self, trading_repository: TradingRepository):
        self.repository = trading_repository
        self.positions = {}
        self.trade_history = []
        
    def execute_trade(self, decision: TradeDecision) -> TradeOutcome:
        """Execute a trading decision"""
        try:
            # Validate decision
            if not self._validate_trade_decision(decision):
                return TradeOutcome(
                    pnl=0.0,
                    duration=0.0,
                    success=False,
                    context={'error': 'Invalid trade decision'}
                )
            
            # Create trade entity
            trade = Trade(
                id=f"trade_{datetime.now().timestamp()}",
                action=TradeAction(decision.action),
                size=decision.size,
                timestamp=datetime.now()
            )
            
            # Execute through repository (NinjaTrader)
            execution_result = self.repository.execute_trade(trade)
            
            if execution_result.success:
                trade.status = TradeStatus.EXECUTED
                trade.price = execution_result.execution_price
                
                # Update positions
                self._update_positions(trade)
                
                # Record trade
                self.trade_history.append(trade)
                
                return TradeOutcome(
                    pnl=execution_result.pnl,
                    duration=execution_result.duration,
                    success=True,
                    context={
                        'trade_id': trade.id,
                        'execution_price': trade.price,
                        'action': trade.action.value,
                        'size': trade.size
                    }
                )
            else:
                trade.status = TradeStatus.FAILED
                return TradeOutcome(
                    pnl=0.0,
                    duration=0.0,
                    success=False,
                    context={'error': execution_result.error}
                )
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return TradeOutcome(
                pnl=0.0,
                duration=0.0,
                success=False,
                context={'error': str(e)}
            )
    
    def get_account_info(self) -> AccountInfo:
        """Get current account information"""
        try:
            account_data = self.repository.get_account_data()
            
            return AccountInfo(
                cash=account_data.cash,
                buying_power=account_data.buying_power,
                position_size=self._calculate_total_position_size(),
                unrealized_pnl=self._calculate_unrealized_pnl(),
                realized_pnl=self._calculate_realized_pnl()
            )
            
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return AccountInfo(0.0, 0.0, 0.0, 0.0, 0.0)
    
    def manage_positions(self) -> Dict:
        """Get current position management data"""
        try:
            # Update current positions from repository
            current_positions = self.repository.get_current_positions()
            
            # Update internal position tracking
            for symbol, position_data in current_positions.items():
                if symbol in self.positions:
                    self.positions[symbol].current_price = position_data.current_price
                    self.positions[symbol].unrealized_pnl = position_data.unrealized_pnl
                else:
                    self.positions[symbol] = position_data
            
            return {
                'positions': {symbol: {
                    'size': pos.size,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'realized_pnl': pos.realized_pnl
                } for symbol, pos in self.positions.items()},
                'total_unrealized_pnl': self._calculate_unrealized_pnl(),
                'total_realized_pnl': self._calculate_realized_pnl(),
                'position_count': len(self.positions)
            }
            
        except Exception as e:
            logger.error(f"Error managing positions: {e}")
            return {'positions': {}, 'total_unrealized_pnl': 0.0, 'total_realized_pnl': 0.0, 'position_count': 0}
    
    def _validate_trade_decision(self, decision: TradeDecision) -> bool:
        """Validate trade decision parameters"""
        if decision.action not in ['buy', 'sell', 'hold']:
            return False
        
        if decision.action == 'hold':
            return True
            
        if decision.size <= 0:
            return False
            
        if decision.confidence < 0.1:  # Minimum confidence threshold
            return False
            
        return True
    
    def _update_positions(self, trade: Trade):
        """Update position tracking after trade execution"""
        symbol = "MNQ"  # Default symbol as per prompt.txt
        
        if symbol not in self.positions:
            if trade.action == TradeAction.BUY:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    size=trade.size,
                    entry_price=trade.price,
                    current_price=trade.price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0
                )
        else:
            position = self.positions[symbol]
            
            if trade.action == TradeAction.BUY:
                # Adding to position
                new_size = position.size + trade.size
                new_entry = ((position.entry_price * position.size) + (trade.price * trade.size)) / new_size
                position.size = new_size
                position.entry_price = new_entry
                
            elif trade.action == TradeAction.SELL:
                # Reducing position
                if trade.size >= position.size:
                    # Closing entire position
                    realized_pnl = (trade.price - position.entry_price) * position.size
                    position.realized_pnl += realized_pnl
                    position.size = 0
                else:
                    # Partial close
                    realized_pnl = (trade.price - position.entry_price) * trade.size
                    position.realized_pnl += realized_pnl
                    position.size -= trade.size
    
    def _calculate_total_position_size(self) -> float:
        """Calculate total position size across all positions"""
        return sum(abs(pos.size) for pos in self.positions.values())
    
    def _calculate_unrealized_pnl(self) -> float:
        """Calculate total unrealized PnL"""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    def _calculate_realized_pnl(self) -> float:
        """Calculate total realized PnL"""
        return sum(pos.realized_pnl for pos in self.positions.values())