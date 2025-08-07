"""
Cleanup Coordinator - Integration layer for seamless StateCoordinator integration

This module provides the main integration point between the state cleanup system
and the existing StateCoordinator, ensuring seamless operation and coordination.

Key Features:
- Seamless integration with existing StateCoordinator
- Automatic cleanup scheduling and execution  
- Event-driven cleanup triggers
- System lifecycle management
- Health monitoring and status reporting
- Configuration management integration

Integration Points:
- StateCoordinator save operations
- System startup and shutdown
- Periodic cleanup scheduling
- Manual cleanup triggers
- Configuration updates
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, Callable
import asyncio
from dataclasses import asdict

from .cleanup_service import CleanupService, CleanupPlan, CleanupResult
from .configuration import ConfigurationManager, CleanupConfiguration, CleanupSchedule
from .file_manager import StateFileManager
from .retention_policies import create_default_retention_policies, RetentionPolicySet

logger = logging.getLogger(__name__)


class CleanupCoordinator:
    """
    Main coordinator that integrates the cleanup system with StateCoordinator
    
    This class serves as the primary interface for the cleanup system,
    managing scheduling, execution, and integration with the existing state management.
    """
    
    def __init__(self, 
                 base_path: str = "data",
                 config_path: Optional[str] = None,
                 state_coordinator=None):
        """
        Initialize the cleanup coordinator
        
        Args:
            base_path: Base directory for state files
            config_path: Path to cleanup configuration file
            state_coordinator: Reference to existing StateCoordinator instance
        """
        self.base_path = Path(base_path)
        self.state_coordinator = state_coordinator
        
        # Initialize configuration management
        self.config_manager = ConfigurationManager(config_path)
        self.config = self.config_manager.load_config()
        
        # Initialize core components
        self.file_manager = StateFileManager(self.base_path)
        self.retention_policies = self._create_retention_policies()
        self.cleanup_service = CleanupService(
            file_manager=self.file_manager,
            retention_policies=self.retention_policies,
            max_files_per_cleanup=self.config.safety.max_files_per_cleanup,
            max_size_per_cleanup_mb=self.config.safety.max_size_per_cleanup_mb
        )
        
        # Scheduling and lifecycle management
        self._scheduler_thread: Optional[threading.Thread] = None
        self._scheduler_stop_event = threading.Event()
        self._initialized = False
        self._running = False
        
        # Status tracking
        self._last_cleanup_time: Optional[float] = None
        self._last_orphaned_cleanup_time: Optional[float] = None
        self._cleanup_count = 0
        self._error_count = 0
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Hook into StateCoordinator if provided
        if self.state_coordinator:
            self._integrate_with_state_coordinator()
    
    def initialize(self) -> bool:
        """
        Initialize the cleanup coordinator
        
        Returns:
            True if initialization was successful
        """
        if self._initialized:
            self.logger.warning("Cleanup coordinator already initialized")
            return True
        
        try:
            self.logger.info("Initializing cleanup coordinator")
            
            # Validate configuration
            if not self.config.enabled:
                self.logger.info("Cleanup system is disabled in configuration")
                return True
            
            config_errors = self.config.validate()
            if config_errors:
                self.logger.error(f"Configuration validation failed: {config_errors}")
                return False
            
            # Initialize file manager
            self.logger.debug("Initializing file manager")
            
            # Run initial orphaned file cleanup if enabled
            if self.config.scheduling.auto_cleanup_orphaned:
                self.logger.info("Running initial orphaned file cleanup")
                try:
                    orphaned_count = self.cleanup_service.cleanup_orphaned_files()
                    if orphaned_count > 0:
                        self.logger.info(f"Cleaned up {orphaned_count} orphaned files during initialization")
                except Exception as e:
                    self.logger.warning(f"Initial orphaned cleanup failed: {e}")
            
            # Start scheduler if needed
            if self.config.scheduling.schedule != CleanupSchedule.DISABLED:
                self._start_scheduler()
            
            # Run startup cleanup if configured
            if self.config.scheduling.run_on_startup:
                self.logger.info("Running startup cleanup")
                try:
                    self._run_cleanup_cycle()
                except Exception as e:
                    self.logger.error(f"Startup cleanup failed: {e}")
            
            self._initialized = True
            self._running = True
            
            self.logger.info("Cleanup coordinator initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize cleanup coordinator: {e}")
            return False
    
    def shutdown(self) -> bool:
        """
        Shutdown the cleanup coordinator gracefully
        
        Returns:
            True if shutdown was successful
        """
        if not self._initialized:
            return True
        
        try:
            self.logger.info("Shutting down cleanup coordinator")
            self._running = False
            
            # Run shutdown cleanup if configured
            if self.config.scheduling.run_on_shutdown:
                self.logger.info("Running shutdown cleanup")
                try:
                    self._run_cleanup_cycle()
                except Exception as e:
                    self.logger.error(f"Shutdown cleanup failed: {e}")
            
            # Stop scheduler
            if self._scheduler_thread:
                self.logger.debug("Stopping cleanup scheduler")
                self._scheduler_stop_event.set()
                self._scheduler_thread.join(timeout=10)
                if self._scheduler_thread.is_alive():
                    self.logger.warning("Cleanup scheduler did not stop gracefully")
            
            self._initialized = False
            self.logger.info("Cleanup coordinator shutdown complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during cleanup coordinator shutdown: {e}")
            return False
    
    def run_cleanup(self, dry_run: bool = False) -> Tuple[CleanupPlan, Optional[CleanupResult]]:
        """
        Run a complete cleanup cycle
        
        Args:
            dry_run: If True, only plan cleanup without executing
            
        Returns:
            Tuple of (CleanupPlan, CleanupResult or None)
        """
        if not self._initialized:
            raise RuntimeError("Cleanup coordinator not initialized")
        
        if not self.config.enabled:
            raise RuntimeError("Cleanup system is disabled")
        
        self.logger.info(f"Running cleanup cycle (dry_run={dry_run})")
        
        try:
            plan, result = self.cleanup_service.run_full_cleanup(dry_run=dry_run)
            
            if not dry_run and result:
                self._last_cleanup_time = time.time()
                self._cleanup_count += 1
                
                if not result.success:
                    self._error_count += 1
                    self.logger.error(f"Cleanup cycle failed: {result.errors}")
                else:
                    self.logger.info(f"Cleanup cycle completed successfully: "
                                   f"deleted {result.files_deleted} files "
                                   f"({result.size_freed_mb:.1f}MB)")
            
            return plan, result
            
        except Exception as e:
            self._error_count += 1
            self.logger.error(f"Cleanup cycle failed with exception: {e}")
            raise
    
    def cleanup_orphaned_files(self) -> int:
        """
        Clean up orphaned temporary files
        
        Returns:
            Number of files cleaned up
        """
        if not self._initialized:
            raise RuntimeError("Cleanup coordinator not initialized")
        
        try:
            count = self.cleanup_service.cleanup_orphaned_files()
            self._last_orphaned_cleanup_time = time.time()
            
            if count > 0:
                self.logger.info(f"Cleaned up {count} orphaned files")
            
            return count
            
        except Exception as e:
            self.logger.error(f"Orphaned file cleanup failed: {e}")
            raise
    
    def update_configuration(self, **updates) -> bool:
        """
        Update cleanup configuration
        
        Args:
            **updates: Configuration fields to update
            
        Returns:
            True if update was successful
        """
        try:
            if self.config_manager.update_config(**updates):
                # Reload configuration
                self.config = self.config_manager.get_config()
                
                # Update component configurations
                self._update_component_configs()
                
                # Restart scheduler if schedule changed
                if any(key.startswith('scheduling.') for key in updates.keys()):
                    if self._scheduler_thread:
                        self._restart_scheduler()
                
                self.logger.info(f"Configuration updated: {list(updates.keys())}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to update configuration: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of the cleanup system
        
        Returns:
            Dictionary with status information
        """
        service_status = self.cleanup_service.get_service_status()
        file_manager_status = self.file_manager.get_manager_status()
        cleanup_stats = self.cleanup_service.get_cleanup_statistics()
        
        return {
            'coordinator': {
                'initialized': self._initialized,
                'running': self._running,
                'enabled': self.config.enabled,
                'cleanup_count': self._cleanup_count,
                'error_count': self._error_count,
                'last_cleanup_time': self._last_cleanup_time,
                'last_orphaned_cleanup_time': self._last_orphaned_cleanup_time,
                'scheduler_active': self._scheduler_thread is not None and self._scheduler_thread.is_alive(),
                'configuration_valid': self.config.is_valid(),
            },
            'service': service_status,
            'file_manager': file_manager_status,
            'statistics': cleanup_stats,
            'configuration': self.config_manager.get_config_summary()
        }
    
    def force_cleanup(self, max_files: Optional[int] = None, 
                     max_size_mb: Optional[float] = None) -> Tuple[CleanupPlan, Optional[CleanupResult]]:
        """
        Force an immediate cleanup with optional limits
        
        Args:
            max_files: Override max files per cleanup
            max_size_mb: Override max size per cleanup
            
        Returns:
            Tuple of (CleanupPlan, CleanupResult)
        """
        if not self._initialized:
            raise RuntimeError("Cleanup coordinator not initialized")
        
        # Temporarily override limits if provided
        original_max_files = self.cleanup_service.max_files_per_cleanup
        original_max_size = self.cleanup_service.max_size_per_cleanup_mb
        
        try:
            if max_files is not None:
                self.cleanup_service.max_files_per_cleanup = max_files
            if max_size_mb is not None:
                self.cleanup_service.max_size_per_cleanup_mb = max_size_mb
            
            self.logger.info(f"Force cleanup requested (max_files={max_files}, max_size_mb={max_size_mb})")
            
            return self.run_cleanup(dry_run=False)
            
        finally:
            # Restore original limits
            self.cleanup_service.max_files_per_cleanup = original_max_files
            self.cleanup_service.max_size_per_cleanup_mb = original_max_size
    
    def _create_retention_policies(self) -> RetentionPolicySet:
        """Create retention policies based on configuration"""
        from .retention_policies import (
            HierarchicalRetentionPolicy, CountBasedRetentionPolicy, 
            SizeBasedRetentionPolicy, RetentionPolicySet, FilePriority
        )
        
        policy_set = RetentionPolicySet("configured_policies")
        
        # Hierarchical policy
        if self.config.retention.hierarchical_enabled:
            hierarchical = HierarchicalRetentionPolicy(
                name="hierarchical_retention",
                recent_hours=self.config.retention.hierarchical_recent_hours,
                hourly_days=self.config.retention.hierarchical_hourly_days,
                daily_days=self.config.retention.hierarchical_daily_days,
                enabled=True
            )
            policy_set.add_policy(hierarchical)
        
        # Count-based policy
        if self.config.retention.count_based_enabled:
            count_limits = {}
            priority_map = {
                "critical": FilePriority.CRITICAL,
                "high": FilePriority.HIGH,
                "medium": FilePriority.MEDIUM,
                "low": FilePriority.LOW
            }
            
            for priority_str, limit in self.config.retention.count_limits.items():
                if priority_str in priority_map:
                    count_limits[priority_map[priority_str]] = limit
            
            count_based = CountBasedRetentionPolicy(
                name="count_based_limits",
                max_count_per_priority=count_limits,
                enabled=True
            )
            policy_set.add_policy(count_based)
        
        # Size-based policy
        if self.config.retention.size_based_enabled:
            size_based = SizeBasedRetentionPolicy(
                name="size_limit",
                max_total_size_mb=self.config.retention.size_limit_mb,
                enabled=True
            )
            policy_set.add_policy(size_based)
        
        return policy_set
    
    def _integrate_with_state_coordinator(self):
        """Integrate with the existing StateCoordinator"""
        if not self.state_coordinator:
            return
        
        try:
            # Hook into state save operations to trigger cleanup
            original_save_state = self.state_coordinator.save_state
            
            def wrapped_save_state(*args, **kwargs):
                result = original_save_state(*args, **kwargs)
                
                # Trigger orphaned cleanup after state saves
                if result and self.config.scheduling.auto_cleanup_orphaned:
                    try:
                        self._check_orphaned_cleanup()
                    except Exception as e:
                        self.logger.warning(f"Post-save orphaned cleanup failed: {e}")
                
                return result
            
            self.state_coordinator.save_state = wrapped_save_state
            self.logger.info("Integrated with StateCoordinator save operations")
            
        except Exception as e:
            self.logger.warning(f"Failed to integrate with StateCoordinator: {e}")
    
    def _start_scheduler(self):
        """Start the cleanup scheduler thread"""
        if self._scheduler_thread:
            return
        
        self._scheduler_stop_event.clear()
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            name="CleanupScheduler",
            daemon=True
        )
        self._scheduler_thread.start()
        
        self.logger.info(f"Cleanup scheduler started with {self.config.scheduling.schedule.value} schedule")
    
    def _restart_scheduler(self):
        """Restart the scheduler with new configuration"""
        if self._scheduler_thread:
            self._scheduler_stop_event.set()
            self._scheduler_thread.join(timeout=5)
        
        self._scheduler_thread = None
        
        if self.config.scheduling.schedule != CleanupSchedule.DISABLED:
            self._start_scheduler()
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        self.logger.debug("Cleanup scheduler loop started")
        
        while not self._scheduler_stop_event.is_set():
            try:
                # Check if it's time for regular cleanup
                if self._should_run_cleanup():
                    self.logger.info("Scheduled cleanup triggered")
                    self._run_cleanup_cycle()
                
                # Check for orphaned file cleanup
                if self._should_run_orphaned_cleanup():
                    self.logger.debug("Scheduled orphaned cleanup triggered")
                    try:
                        self.cleanup_orphaned_files()
                    except Exception as e:
                        self.logger.warning(f"Scheduled orphaned cleanup failed: {e}")
                
                # Sleep for a reasonable interval (check every 10 minutes)
                self._scheduler_stop_event.wait(600)  # 10 minutes
                
            except Exception as e:
                self.logger.error(f"Scheduler loop error: {e}")
                self._scheduler_stop_event.wait(60)  # Wait 1 minute on error
        
        self.logger.debug("Cleanup scheduler loop ended")
    
    def _should_run_cleanup(self) -> bool:
        """Check if it's time to run a cleanup"""
        if self.config.scheduling.schedule == CleanupSchedule.DISABLED:
            return False
        
        if self.config.scheduling.schedule == CleanupSchedule.MANUAL:
            return False
        
        current_time = time.time()
        
        # Check interval-based scheduling
        if self._last_cleanup_time is None:
            return True  # First run
        
        interval_seconds = self.config.scheduling.interval_hours * 3600
        if current_time - self._last_cleanup_time >= interval_seconds:
            return True
        
        # For daily/weekly scheduling, also check preferred time
        if self.config.scheduling.schedule in [CleanupSchedule.DAILY, CleanupSchedule.WEEKLY]:
            now = datetime.now()
            preferred_time = now.replace(
                hour=self.config.scheduling.preferred_hour,
                minute=0,
                second=0,
                microsecond=0
            )
            
            # Check if we've passed the preferred time and haven't run today
            if now >= preferred_time:
                last_run = datetime.fromtimestamp(self._last_cleanup_time) if self._last_cleanup_time else None
                if not last_run or last_run < preferred_time:
                    if self.config.scheduling.schedule == CleanupSchedule.WEEKLY:
                        # For weekly, also check day of week
                        if now.weekday() == self.config.scheduling.preferred_day_of_week:
                            return True
                    else:
                        return True
        
        return False
    
    def _should_run_orphaned_cleanup(self) -> bool:
        """Check if it's time to run orphaned file cleanup"""
        if not self.config.scheduling.auto_cleanup_orphaned:
            return False
        
        if self._last_orphaned_cleanup_time is None:
            return True
        
        interval_seconds = self.config.scheduling.orphaned_cleanup_interval_hours * 3600
        return time.time() - self._last_orphaned_cleanup_time >= interval_seconds
    
    def _check_orphaned_cleanup(self):
        """Check and run orphaned cleanup if needed"""
        if self._should_run_orphaned_cleanup():
            self.cleanup_orphaned_files()
    
    def _run_cleanup_cycle(self):
        """Run a cleanup cycle with error handling"""
        try:
            plan, result = self.run_cleanup(dry_run=False)
            
            if result and self.config.monitoring.report_cleanup_statistics:
                self.logger.info(f"Cleanup statistics: {self.cleanup_service.get_cleanup_statistics()}")
                
        except Exception as e:
            self.logger.error(f"Cleanup cycle failed: {e}")
    
    def _update_component_configs(self):
        """Update component configurations from main config"""
        # Update cleanup service limits
        self.cleanup_service.max_files_per_cleanup = self.config.safety.max_files_per_cleanup
        self.cleanup_service.max_size_per_cleanup_mb = self.config.safety.max_size_per_cleanup_mb
        
        # Recreate retention policies if retention config changed
        self.retention_policies = self._create_retention_policies()
        self.cleanup_service.retention_policies = self.retention_policies
        
        self.logger.debug("Component configurations updated")


# Global cleanup coordinator instance
_cleanup_coordinator: Optional[CleanupCoordinator] = None


def get_cleanup_coordinator() -> Optional[CleanupCoordinator]:
    """Get the global cleanup coordinator instance"""
    return _cleanup_coordinator


def initialize_cleanup_system(base_path: str = "data", 
                            config_path: Optional[str] = None,
                            state_coordinator=None) -> CleanupCoordinator:
    """
    Initialize the global cleanup system
    
    Args:
        base_path: Base directory for state files
        config_path: Path to cleanup configuration file
        state_coordinator: Reference to StateCoordinator instance
        
    Returns:
        Initialized CleanupCoordinator instance
    """
    global _cleanup_coordinator
    
    if _cleanup_coordinator is not None:
        logger.warning("Cleanup system already initialized")
        return _cleanup_coordinator
    
    logger.info("Initializing global cleanup system")
    
    _cleanup_coordinator = CleanupCoordinator(
        base_path=base_path,
        config_path=config_path,
        state_coordinator=state_coordinator
    )
    
    if not _cleanup_coordinator.initialize():
        logger.error("Failed to initialize cleanup system")
        _cleanup_coordinator = None
        raise RuntimeError("Cleanup system initialization failed")
    
    logger.info("Global cleanup system initialized successfully")
    return _cleanup_coordinator


def shutdown_cleanup_system():
    """Shutdown the global cleanup system"""
    global _cleanup_coordinator
    
    if _cleanup_coordinator is None:
        return
    
    logger.info("Shutting down global cleanup system")
    
    try:
        _cleanup_coordinator.shutdown()
    except Exception as e:
        logger.error(f"Error during cleanup system shutdown: {e}")
    finally:
        _cleanup_coordinator = None
    
    logger.info("Global cleanup system shutdown complete")