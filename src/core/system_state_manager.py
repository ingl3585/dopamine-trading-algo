# system_state_manager.py

import logging
import time
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path

from src.core.state_coordinator import state_coordinator, register_state_component

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """Core system metrics and counters"""
    total_decisions: int = 0
    data_updates_received: int = 0
    successful_trades: int = 0
    failed_trades: int = 0
    last_account_balance: float = 0.0
    last_save_time: float = 0.0
    last_account_update_time: float = 0.0
    system_start_time: float = 0.0

@dataclass
class SystemStatus:
    """Current system operational status"""
    running: bool = False
    ready_for_trading: bool = False
    historical_data_loaded: bool = False
    tcp_server_connected: bool = False
    intelligence_ready: bool = False
    agent_ready: bool = False

class SystemStateManager:
    """
    Manages system state, metrics, and persistence.
    
    Responsibilities:
    - Track system metrics and performance counters
    - Manage operational status flags
    - Handle state persistence and recovery
    - Coordinate state saving across components
    - Monitor system health and readiness
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics = SystemMetrics()
        self.status = SystemStatus()
        
        # State management parameters
        self.save_interval = config.get('model_save_interval', 300)  # 5 minutes
        self.account_change_threshold = config.get('account_change_threshold', 0.05)  # 5%
        
        # State callbacks for components
        self.state_callbacks = {}
        
        # Initialize system start time
        self.metrics.system_start_time = time.time()
        
        logger.info("System state manager initialized")
    
    def register_component_state(self, component_name: str, 
                                save_callback: Callable[[], Dict[str, Any]],
                                load_callback: Callable[[Dict[str, Any]], None],
                                priority: int = 10):
        """
        Register a component for state management
        
        Args:
            component_name: Name of the component
            save_callback: Function to save component state
            load_callback: Function to load component state
            priority: Priority for state operations (higher = earlier)
        """
        self.state_callbacks[component_name] = {
            'save': save_callback,
            'load': load_callback,
            'priority': priority
        }
        
        # Register with state coordinator
        register_state_component(component_name, save_callback, load_callback, priority)
        
        logger.info(f"Registered component state: {component_name}")
    
    def update_system_status(self, **kwargs):
        """
        Update system status flags
        
        Args:
            **kwargs: Status fields to update
        """
        for key, value in kwargs.items():
            if hasattr(self.status, key):
                setattr(self.status, key, value)
                logger.info(f"System status updated: {key} = {value}")
    
    def increment_decision_count(self):
        """Increment decision counter"""
        self.metrics.total_decisions += 1
    
    def increment_data_update_count(self):
        """Increment data update counter"""
        self.metrics.data_updates_received += 1
    
    def record_trade_outcome(self, success: bool):
        """Record trade outcome"""
        if success:
            self.metrics.successful_trades += 1
        else:
            self.metrics.failed_trades += 1
    
    def update_account_balance(self, balance: float):
        """
        Update account balance and check for significant changes
        
        Args:
            balance: Current account balance
        """
        if self.metrics.last_account_balance > 0:
            change_ratio = abs(balance - self.metrics.last_account_balance) / self.metrics.last_account_balance
            if change_ratio > self.account_change_threshold:
                logger.info(f"Significant account change: {self.metrics.last_account_balance:.2f} -> {balance:.2f}")
        
        self.metrics.last_account_balance = balance
        self.metrics.last_account_update_time = time.time()
    
    def should_save_state(self) -> bool:
        """Check if it's time to save system state"""
        return time.time() - self.metrics.last_save_time >= self.save_interval
    
    def save_all_state(self) -> bool:
        """
        Save state for all registered components
        
        Returns:
            bool: True if save succeeded
        """
        try:
            # Save using state coordinator
            state_coordinator.save_all_states()
            
            # Update our save time
            self.metrics.last_save_time = time.time()
            
            logger.info("All component states saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save all states: {e}")
            return False
    
    def load_all_state(self) -> bool:
        """
        Load state for all registered components
        
        Returns:
            bool: True if load succeeded
        """
        try:
            # Load using state coordinator
            state_coordinator.load_all_states()
            
            logger.info("All component states loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load all states: {e}")
            return False
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        current_time = time.time()
        uptime = current_time - self.metrics.system_start_time
        
        return {
            **asdict(self.metrics),
            'uptime_seconds': uptime,
            'uptime_hours': uptime / 3600,
            'trade_success_rate': (
                self.metrics.successful_trades / 
                max(1, self.metrics.successful_trades + self.metrics.failed_trades)
            ),
            'decisions_per_minute': (
                self.metrics.total_decisions / max(1, uptime / 60)
            ),
            'data_updates_per_minute': (
                self.metrics.data_updates_received / max(1, uptime / 60)
            )
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return asdict(self.status)
    
    def get_full_system_state(self) -> Dict[str, Any]:
        """Get complete system state including metrics and status"""
        return {
            'metrics': self.get_system_metrics(),
            'status': self.get_system_status(),
            'components_registered': list(self.state_callbacks.keys()),
            'last_save_age_seconds': time.time() - self.metrics.last_save_time,
            'ready_for_trading': self.is_ready_for_trading()
        }
    
    def is_ready_for_trading(self) -> bool:
        """Check if system is ready for live trading"""
        return (
            self.status.running and
            self.status.ready_for_trading and
            self.status.historical_data_loaded and
            self.status.tcp_server_connected and
            self.status.intelligence_ready and
            self.status.agent_ready
        )
    
    def reset_metrics(self):
        """Reset system metrics"""
        self.metrics = SystemMetrics()
        self.metrics.system_start_time = time.time()
        logger.info("System metrics reset")
    
    def shutdown(self):
        """Shutdown state manager and perform final save"""
        try:
            logger.info("Shutting down system state manager...")
            
            # Final state save
            self.save_all_state()
            
            # Update status
            self.status.running = False
            self.status.ready_for_trading = False
            
            logger.info("System state manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during state manager shutdown: {e}")
    
    def check_system_health(self) -> Dict[str, Any]:
        """
        Check system health and return status report
        
        Returns:
            Dict containing health status and issues
        """
        issues = []
        warnings = []
        
        # Check for stale data
        if self.metrics.data_updates_received == 0:
            issues.append("No data updates received")
        elif time.time() - self.metrics.last_account_update_time > 300:  # 5 minutes
            warnings.append("Account balance not updated recently")
        
        # Check save frequency
        if time.time() - self.metrics.last_save_time > self.save_interval * 2:
            warnings.append("State save overdue")
        
        # Check trading readiness
        if not self.is_ready_for_trading() and self.status.running:
            issues.append("System running but not ready for trading")
        
        # Check component states
        missing_components = []
        for component in ['intelligence', 'agent', 'portfolio', 'risk_manager']:
            if component not in self.state_callbacks:
                missing_components.append(component)
        
        if missing_components:
            warnings.append(f"Missing component state registration: {missing_components}")
        
        health_status = "healthy" if not issues else "critical" if issues else "warning"
        
        return {
            'status': health_status,
            'issues': issues,
            'warnings': warnings,
            'uptime_hours': (time.time() - self.metrics.system_start_time) / 3600,
            'ready_for_trading': self.is_ready_for_trading()
        }