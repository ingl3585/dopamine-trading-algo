"""
Trading Repository Interfaces - Abstract data access
"""

from abc import ABC, abstractmethod
from typing import Dict
from src.data_models.trading_domain_models import Trade, Position, Account

class ExecutionResult:
    def __init__(self, success: bool, execution_price: float = 0.0, pnl: float = 0.0, 
                 duration: float = 0.0, error: str = ""):
        self.success = success
        self.execution_price = execution_price
        self.pnl = pnl
        self.duration = duration
        self.error = error

class TradingRepository(ABC):
    """Abstract repository for trading operations"""
    
    @abstractmethod
    def execute_trade(self, trade: Trade) -> ExecutionResult:
        """Execute a trade and return execution result"""
        pass
    
    @abstractmethod
    def get_account_data(self) -> Account:
        """Get current account data"""
        pass
    
    @abstractmethod
    def get_current_positions(self) -> Dict[str, Position]:
        """Get current positions"""
        pass
    
    @abstractmethod
    def get_market_data(self) -> Dict:
        """Get current market data"""
        pass