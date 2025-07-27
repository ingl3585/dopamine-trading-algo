# analysis_trigger_manager.py

import logging
import time
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum

from src.core.market_data_processor import MarketData

logger = logging.getLogger(__name__)

class AnalysisType(Enum):
    """Types of analysis that can be triggered"""
    REGIME_ANALYSIS = "regime_analysis"
    TREND_ANALYSIS = "trend_analysis"
    VOLATILITY_ANALYSIS = "volatility_analysis"
    MOMENTUM_ANALYSIS = "momentum_analysis"
    MICROSTRUCTURE_ANALYSIS = "microstructure_analysis"

@dataclass
class AnalysisTrigger:
    """Configuration for an analysis trigger"""
    name: str
    analysis_type: AnalysisType
    timeframe: str
    callback: Callable[[MarketData], None]
    enabled: bool = True
    last_triggered: float = 0.0
    trigger_count: int = 0

class AnalysisTriggerManager:
    """
    Manages timeframe-based analysis triggers.
    
    Responsibilities:
    - Register analysis callbacks for different timeframes
    - Monitor bar completions and trigger appropriate analysis
    - Track trigger frequency and performance
    - Handle enhanced analysis scheduling
    - Coordinate between different analysis types
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.triggers = {}
        self.analysis_history = []
        
        # Trigger management
        self.max_history_size = 1000
        self.min_trigger_interval = 5.0  # Minimum seconds between same trigger
        
        logger.info("Analysis trigger manager initialized")
    
    def register_trigger(self, name: str, analysis_type: AnalysisType, 
                        timeframe: str, callback: Callable[[MarketData], None],
                        enabled: bool = True):
        """
        Register an analysis trigger
        
        Args:
            name: Unique name for the trigger
            analysis_type: Type of analysis to perform
            timeframe: Timeframe that triggers the analysis
            callback: Function to call when triggered
            enabled: Whether trigger is active
        """
        trigger = AnalysisTrigger(
            name=name,
            analysis_type=analysis_type,
            timeframe=timeframe,
            callback=callback,
            enabled=enabled
        )
        
        self.triggers[name] = trigger
        logger.info(f"Registered analysis trigger: {name} ({analysis_type.value}) for {timeframe}")
    
    def trigger_15m_analysis(self, market_data: MarketData):
        """
        Trigger 15-minute bar analysis
        
        Args:
            market_data: Current market data
        """
        self._execute_triggers_for_timeframe("15m", market_data)
        
        # Log 15m bar completion
        logger.info(f"15m bar completed: {market_data.price:.2f} "
                   f"(Account: ${market_data.account_balance:.0f}, "
                   f"Daily P&L: ${market_data.daily_pnl:.2f})")
    
    def trigger_1h_analysis(self, market_data: MarketData):
        """
        Trigger 1-hour bar analysis
        
        Args:
            market_data: Current market data
        """
        self._execute_triggers_for_timeframe("1h", market_data)
        
        # Log 1h bar completion with enhanced regime analysis
        logger.info(f"1H bar completed: {market_data.price:.2f} - "
                   f"Triggering enhanced regime analysis")
    
    def trigger_4h_analysis(self, market_data: MarketData):
        """
        Trigger 4-hour bar analysis
        
        Args:
            market_data: Current market data
        """
        self._execute_triggers_for_timeframe("4h", market_data)
        
        # Log 4h bar completion with major trend analysis
        logger.info(f"4H bar completed: {market_data.price:.2f} - "
                   f"Triggering major trend analysis")
    
    def trigger_account_adaptation(self, market_data: MarketData):
        """
        Trigger account-based adaptation analysis
        
        Args:
            market_data: Current market data
        """
        self._execute_triggers_for_timeframe("account", market_data)
    
    def _execute_triggers_for_timeframe(self, timeframe: str, market_data: MarketData):
        """
        Execute all triggers for a specific timeframe
        
        Args:
            timeframe: Timeframe to trigger
            market_data: Current market data
        """
        current_time = time.time()
        triggered_count = 0
        
        for trigger_name, trigger in self.triggers.items():
            if not trigger.enabled or trigger.timeframe != timeframe:
                continue
            
            # Check minimum interval between triggers
            if current_time - trigger.last_triggered < self.min_trigger_interval:
                continue
            
            try:
                # Execute trigger callback
                trigger.callback(market_data)
                
                # Update trigger stats
                trigger.last_triggered = current_time
                trigger.trigger_count += 1
                triggered_count += 1
                
                # Record in history
                self._record_trigger_execution(trigger_name, market_data)
                
                logger.debug(f"Executed trigger: {trigger_name} ({trigger.analysis_type.value})")
                
            except Exception as e:
                logger.error(f"Error executing trigger {trigger_name}: {e}")
        
        if triggered_count > 0:
            logger.info(f"Executed {triggered_count} triggers for {timeframe} timeframe")
    
    def _record_trigger_execution(self, trigger_name: str, market_data: MarketData):
        """
        Record trigger execution in history
        
        Args:
            trigger_name: Name of executed trigger
            market_data: Market data at execution
        """
        execution_record = {
            'timestamp': time.time(),
            'trigger_name': trigger_name,
            'price': market_data.price,
            'account_balance': market_data.account_balance,
            'daily_pnl': market_data.daily_pnl
        }
        
        self.analysis_history.append(execution_record)
        
        # Trim history if too large
        if len(self.analysis_history) > self.max_history_size:
            self.analysis_history = self.analysis_history[-self.max_history_size:]
    
    def enable_trigger(self, trigger_name: str):
        """Enable a specific trigger"""
        if trigger_name in self.triggers:
            self.triggers[trigger_name].enabled = True
            logger.info(f"Enabled trigger: {trigger_name}")
    
    def disable_trigger(self, trigger_name: str):
        """Disable a specific trigger"""
        if trigger_name in self.triggers:
            self.triggers[trigger_name].enabled = False
            logger.info(f"Disabled trigger: {trigger_name}")
    
    def get_trigger_stats(self) -> Dict[str, Any]:
        """Get statistics for all triggers"""
        stats = {}
        current_time = time.time()
        
        for trigger_name, trigger in self.triggers.items():
            stats[trigger_name] = {
                'analysis_type': trigger.analysis_type.value,
                'timeframe': trigger.timeframe,
                'enabled': trigger.enabled,
                'trigger_count': trigger.trigger_count,
                'last_triggered': trigger.last_triggered,
                'seconds_since_last': current_time - trigger.last_triggered,
                'average_interval': self._calculate_average_interval(trigger_name)
            }
        
        return stats
    
    def _calculate_average_interval(self, trigger_name: str) -> float:
        """Calculate average interval between triggers"""
        trigger_executions = [
            record for record in self.analysis_history
            if record['trigger_name'] == trigger_name
        ]
        
        if len(trigger_executions) < 2:
            return 0.0
        
        intervals = []
        for i in range(1, len(trigger_executions)):
            interval = trigger_executions[i]['timestamp'] - trigger_executions[i-1]['timestamp']
            intervals.append(interval)
        
        return sum(intervals) / len(intervals)
    
    def get_recent_analysis_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get recent analysis history
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent analysis executions
        """
        cutoff_time = time.time() - (hours * 3600)
        
        return [
            record for record in self.analysis_history
            if record['timestamp'] >= cutoff_time
        ]
    
    def reset_trigger_stats(self):
        """Reset all trigger statistics"""
        for trigger in self.triggers.values():
            trigger.trigger_count = 0
            trigger.last_triggered = 0.0
        
        self.analysis_history.clear()
        logger.info("All trigger statistics reset")
    
    def get_trigger_summary(self) -> Dict[str, Any]:
        """Get summary of trigger manager state"""
        enabled_triggers = sum(1 for t in self.triggers.values() if t.enabled)
        total_triggers = len(self.triggers)
        total_executions = sum(t.trigger_count for t in self.triggers.values())
        
        return {
            'total_triggers': total_triggers,
            'enabled_triggers': enabled_triggers,
            'disabled_triggers': total_triggers - enabled_triggers,
            'total_executions': total_executions,
            'history_size': len(self.analysis_history),
            'timeframes': list(set(t.timeframe for t in self.triggers.values())),
            'analysis_types': list(set(t.analysis_type.value for t in self.triggers.values()))
        }