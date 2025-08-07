"""
Cleanup Configuration - JSON-based configuration system with validation

This module provides comprehensive configuration management for the state cleanup system,
including validation, defaults, and runtime configuration updates.

Key Features:
- JSON-based configuration with schema validation
- Default configuration templates
- Runtime configuration updates
- Environment-specific settings (development/production)
- Configuration validation and error reporting
- Automatic configuration migration

Configuration Structure:
- Cleanup intervals and scheduling
- Retention policy settings
- Safety limits and thresholds
- File operation parameters  
- Monitoring and logging settings
"""

import json
import logging
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import time

from .retention_policies import FilePriority, create_default_retention_policies

logger = logging.getLogger(__name__)


class CleanupSchedule(Enum):
    """Cleanup scheduling options"""
    DISABLED = "disabled"
    MANUAL = "manual"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"


@dataclass
class RetentionPolicyConfig:
    """Configuration for retention policies"""
    enabled: bool = True
    hierarchical_enabled: bool = True
    hierarchical_recent_hours: float = 24.0
    hierarchical_hourly_days: int = 7
    hierarchical_daily_days: int = 30
    
    count_based_enabled: bool = True
    count_limits: Dict[str, int] = field(default_factory=lambda: {
        "critical": 100,
        "high": 50,
        "medium": 30,
        "low": 10
    })
    
    size_based_enabled: bool = True
    size_limit_mb: float = 500.0
    
    def validate(self) -> List[str]:
        """Validate retention policy configuration"""
        errors = []
        
        if self.hierarchical_recent_hours < 0:
            errors.append("hierarchical_recent_hours must be non-negative")
        
        if self.hierarchical_hourly_days < 0:
            errors.append("hierarchical_hourly_days must be non-negative")
        
        if self.hierarchical_daily_days < 0:
            errors.append("hierarchical_daily_days must be non-negative")
        
        if self.size_limit_mb <= 0:
            errors.append("size_limit_mb must be positive")
        
        # Validate count limits
        valid_priorities = {"critical", "high", "medium", "low"}
        for priority, limit in self.count_limits.items():
            if priority not in valid_priorities:
                errors.append(f"Invalid priority in count_limits: {priority}")
            if limit < 0:
                errors.append(f"Count limit for {priority} must be non-negative")
        
        return errors


@dataclass
class SafetyConfig:
    """Configuration for safety limits and thresholds"""
    max_files_per_cleanup: int = 1000
    max_size_per_cleanup_mb: float = 1000.0
    max_deletion_percentage: float = 80.0
    require_recent_files: bool = True
    recent_files_threshold_hours: float = 24.0
    
    # Validation thresholds
    warn_deletion_percentage: float = 50.0
    warn_size_threshold_mb: float = 500.0
    
    def validate(self) -> List[str]:
        """Validate safety configuration"""
        errors = []
        
        if self.max_files_per_cleanup <= 0:
            errors.append("max_files_per_cleanup must be positive")
        
        if self.max_size_per_cleanup_mb <= 0:
            errors.append("max_size_per_cleanup_mb must be positive")
        
        if not 0 <= self.max_deletion_percentage <= 100:
            errors.append("max_deletion_percentage must be between 0 and 100")
        
        if not 0 <= self.warn_deletion_percentage <= 100:
            errors.append("warn_deletion_percentage must be between 0 and 100")
        
        if self.warn_deletion_percentage > self.max_deletion_percentage:
            errors.append("warn_deletion_percentage cannot exceed max_deletion_percentage")
        
        if self.recent_files_threshold_hours < 0:
            errors.append("recent_files_threshold_hours must be non-negative")
        
        if self.warn_size_threshold_mb < 0:
            errors.append("warn_size_threshold_mb must be non-negative")
        
        return errors


@dataclass
class SchedulingConfig:   
    """Configuration for cleanup scheduling"""
    schedule: CleanupSchedule = CleanupSchedule.DAILY
    interval_hours: float = 24.0
    run_on_startup: bool = False
    run_on_shutdown: bool = False
    
    # Time-based scheduling (for daily/weekly)
    preferred_hour: int = 2  # 2 AM
    preferred_day_of_week: int = 0  # Monday (0=Monday, 6=Sunday)
    
    # Automatic orphaned file cleanup
    auto_cleanup_orphaned: bool = True
    orphaned_cleanup_interval_hours: float = 1.0
    
    def validate(self) -> List[str]:
        """Validate scheduling configuration"""
        errors = []
        
        if self.interval_hours <= 0:
            errors.append("interval_hours must be positive")
        
        if not 0 <= self.preferred_hour <= 23:
            errors.append("preferred_hour must be between 0 and 23")
        
        if not 0 <= self.preferred_day_of_week <= 6:
            errors.append("preferred_day_of_week must be between 0 and 6")
        
        if self.orphaned_cleanup_interval_hours <= 0:
            errors.append("orphaned_cleanup_interval_hours must be positive")
        
        return errors


@dataclass
class MonitoringConfig:
    """Configuration for monitoring and logging"""
    enable_detailed_logging: bool = True
    log_cleanup_plans: bool = True
    log_file_operations: bool = False  # Can be verbose
    
    # Metrics and reporting
    keep_cleanup_history_count: int = 100
    report_cleanup_statistics: bool = True
    
    # Alerting thresholds
    alert_on_cleanup_failure: bool = True
    alert_on_large_deletions: bool = True
    alert_threshold_files: int = 500
    alert_threshold_size_mb: float = 1000.0
    
    def validate(self) -> List[str]:
        """Validate monitoring configuration"""
        errors = []
        
        if self.keep_cleanup_history_count < 0:
            errors.append("keep_cleanup_history_count must be non-negative")
        
        if self.alert_threshold_files < 0:
            errors.append("alert_threshold_files must be non-negative")
        
        if self.alert_threshold_size_mb < 0:
            errors.append("alert_threshold_size_mb must be non-negative")
        
        return errors


@dataclass
class CleanupConfiguration:
    """Complete cleanup system configuration"""
    enabled: bool = True
    base_path: str = "data"
    
    retention: RetentionPolicyConfig = field(default_factory=RetentionPolicyConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    scheduling: SchedulingConfig = field(default_factory=SchedulingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Metadata
    version: str = "1.0.0"
    created_at: Optional[float] = None
    last_modified: Optional[float] = None
    
    def __post_init__(self):
        """Initialize timestamps if not provided"""
        current_time = time.time()
        if self.created_at is None:
            self.created_at = current_time
        if self.last_modified is None:
            self.last_modified = current_time
    
    def validate(self) -> List[str]:
        """Validate the complete configuration"""
        errors = []
        
        # Validate base path
        if not self.base_path:
            errors.append("base_path cannot be empty")
        
        # Validate sub-configurations
        errors.extend([f"retention.{e}" for e in self.retention.validate()])
        errors.extend([f"safety.{e}" for e in self.safety.validate()])
        errors.extend([f"scheduling.{e}" for e in self.scheduling.validate()])
        errors.extend([f"monitoring.{e}" for e in self.monitoring.validate()])
        
        return errors
    
    def is_valid(self) -> bool:
        """Check if configuration is valid"""
        return len(self.validate()) == 0
    
    def update_modified_time(self):
        """Update the last modified timestamp"""
        self.last_modified = time.time()


class ConfigurationManager:
    """
    Manages cleanup configuration loading, saving, and validation
    
    This class handles configuration persistence, validation, and provides
    a clean interface for configuration management throughout the system.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path) if config_path else None
        self._config: Optional[CleanupConfiguration] = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> CleanupConfiguration:
        """
        Load configuration from file or create default
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Loaded or default CleanupConfiguration
        """
        if config_path:
            self.config_path = Path(config_path)
        
        # Try to load existing configuration
        if self.config_path and self.config_path.exists():
            try:
                return self._load_from_file(self.config_path)
            except Exception as e:
                self.logger.error(f"Failed to load config from {self.config_path}: {e}")
                self.logger.info("Using default configuration")
        
        # Create default configuration
        config = self._create_default_config()
        self._config = config
        return config
    
    def save_config(self, config: Optional[CleanupConfiguration] = None,
                   config_path: Optional[Union[str, Path]] = None) -> bool:
        """
        Save configuration to file
        
        Args:
            config: Configuration to save (uses current if None)
            config_path: Path to save to (uses current if None)
            
        Returns:
            True if save was successful
        """
        if config is None:
            config = self._config
        if config is None:
            self.logger.error("No configuration to save")
            return False
        
        if config_path:
            self.config_path = Path(config_path)
        
        if not self.config_path:
            self.logger.error("No config path specified for save")
            return False
        
        try:
            # Update modification time
            config.update_modified_time()
            
            # Validate before saving
            errors = config.validate()
            if errors:
                self.logger.error(f"Configuration validation failed: {errors}")
                return False
            
            # Ensure directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to temporary file first, then rename (atomic operation)
            temp_path = self.config_path.with_suffix('.tmp')
            
            with open(temp_path, 'w') as f:
                json.dump(asdict(config), f, indent=2, default=self._json_encoder)
            
            # Atomic rename
            temp_path.rename(self.config_path)
            
            self.logger.info(f"Configuration saved to {self.config_path}")
            self._config = config
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save config to {self.config_path}: {e}")
            return False
    
    def get_config(self) -> CleanupConfiguration:
        """
        Get current configuration
        
        Returns:
            Current CleanupConfiguration
        """
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def update_config(self, **updates) -> bool:
        """
        Update configuration with new values
        
        Args:
            **updates: Configuration fields to update
            
        Returns:
            True if update was successful
        """
        config = self.get_config()
        
        try:
            # Apply updates
            for key, value in updates.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    # Handle nested updates (e.g., retention.size_limit_mb)
                    if '.' in key:
                        parts = key.split('.')
                        obj = config
                        for part in parts[:-1]:
                            obj = getattr(obj, part)
                        setattr(obj, parts[-1], value)
                    else:
                        self.logger.warning(f"Unknown configuration key: {key}")
            
            # Validate updated configuration
            errors = config.validate()
            if errors:
                self.logger.error(f"Updated configuration is invalid: {errors}")
                return False
            
            config.update_modified_time()
            self._config = config
            
            self.logger.info(f"Configuration updated: {list(updates.keys())}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update configuration: {e}")
            return False
    
    def _load_from_file(self, config_path: Path) -> CleanupConfiguration:
        """Load configuration from JSON file"""
        self.logger.info(f"Loading configuration from {config_path}")
        
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        # Handle nested dataclass construction
        config_data = {}
        
        # Basic fields
        for field_name in ['enabled', 'base_path', 'version', 'created_at', 'last_modified']:
            if field_name in data:
                config_data[field_name] = data[field_name]
        
        # Nested configurations
        if 'retention' in data:
            config_data['retention'] = RetentionPolicyConfig(**data['retention'])
        
        if 'safety' in data:
            config_data['safety'] = SafetyConfig(**data['safety'])
        
        if 'scheduling' in data:
            # Handle enum conversion
            sched_data = data['scheduling'].copy()
            if 'schedule' in sched_data:
                sched_data['schedule'] = CleanupSchedule(sched_data['schedule'])
            config_data['scheduling'] = SchedulingConfig(**sched_data)
        
        if 'monitoring' in data:
            config_data['monitoring'] = MonitoringConfig(**data['monitoring'])
        
        config = CleanupConfiguration(**config_data)
        
        # Validate loaded configuration
        errors = config.validate()
        if errors:
            self.logger.warning(f"Loaded configuration has validation errors: {errors}")
        
        self._config = config
        self.logger.info("Configuration loaded successfully")
        return config
    
    def _create_default_config(self) -> CleanupConfiguration:
        """Create default configuration"""
        self.logger.info("Creating default cleanup configuration")
        
        config = CleanupConfiguration(
            enabled=True,
            base_path="data",
            retention=RetentionPolicyConfig(
                enabled=True,
                hierarchical_enabled=True,
                hierarchical_recent_hours=24.0,
                hierarchical_hourly_days=7,
                hierarchical_daily_days=30,
                count_based_enabled=True,
                count_limits={
                    "critical": 100,
                    "high": 50,
                    "medium": 30,
                    "low": 10
                },
                size_based_enabled=True,
                size_limit_mb=500.0
            ),
            safety=SafetyConfig(
                max_files_per_cleanup=1000,
                max_size_per_cleanup_mb=1000.0,
                max_deletion_percentage=80.0,
                require_recent_files=True,
                recent_files_threshold_hours=24.0,
                warn_deletion_percentage=50.0,
                warn_size_threshold_mb=500.0
            ),
            scheduling=SchedulingConfig(
                schedule=CleanupSchedule.DAILY,
                interval_hours=24.0,
                run_on_startup=False,
                run_on_shutdown=False,
                preferred_hour=2,
                preferred_day_of_week=0,
                auto_cleanup_orphaned=True,
                orphaned_cleanup_interval_hours=1.0
            ),
            monitoring=MonitoringConfig(
                enable_detailed_logging=True,
                log_cleanup_plans=True,
                log_file_operations=False,
                keep_cleanup_history_count=100,
                report_cleanup_statistics=True,
                alert_on_cleanup_failure=True,
                alert_on_large_deletions=True,
                alert_threshold_files=500,
                alert_threshold_size_mb=1000.0
            )
        )
        
        return config
    
    def _json_encoder(self, obj):
        """JSON encoder for special objects"""
        if isinstance(obj, CleanupSchedule):
            return obj.value
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration"""
        config = self.get_config()
        
        return {
            'enabled': config.enabled,
            'base_path': config.base_path,
            'version': config.version,
            'is_valid': config.is_valid(),
            'validation_errors': config.validate(),
            'retention_policies_enabled': config.retention.enabled,
            'scheduling': config.scheduling.schedule.value,
            'safety_limits': {
                'max_files': config.safety.max_files_per_cleanup,
                'max_size_mb': config.safety.max_size_per_cleanup_mb,
                'max_deletion_pct': config.safety.max_deletion_percentage
            },
            'last_modified': config.last_modified
        }


def create_development_config() -> CleanupConfiguration:
    """Create a development-oriented configuration"""
    return CleanupConfiguration(
        enabled=True,
        base_path="data",
        retention=RetentionPolicyConfig(
            hierarchical_recent_hours=1.0,  # Shorter retention for development
            hierarchical_hourly_days=1,
            hierarchical_daily_days=7,
            size_limit_mb=100.0  # Smaller size limit
        ),
        safety=SafetyConfig(
            max_files_per_cleanup=100,  # Smaller batches
            max_size_per_cleanup_mb=100.0,
        ),
        scheduling=SchedulingConfig(
            schedule=CleanupSchedule.HOURLY,
            interval_hours=1.0,  # More frequent cleanup
            auto_cleanup_orphaned=True,
            orphaned_cleanup_interval_hours=0.5
        ),
        monitoring=MonitoringConfig(
            enable_detailed_logging=True,
            log_file_operations=True,  # More verbose logging for dev
            keep_cleanup_history_count=50
        )
    )


def create_production_config() -> CleanupConfiguration:
    """Create a production-oriented configuration"""
    return CleanupConfiguration(
        enabled=True,
        base_path="data",
        retention=RetentionPolicyConfig(
            hierarchical_recent_hours=24.0,
            hierarchical_hourly_days=7,
            hierarchical_daily_days=30,
            size_limit_mb=1000.0  # Larger size limit for production
        ),
        safety=SafetyConfig(
            max_files_per_cleanup=1000,
            max_size_per_cleanup_mb=1000.0,
            max_deletion_percentage=70.0,  # More conservative in production
        ),
        scheduling=SchedulingConfig(
            schedule=CleanupSchedule.DAILY,
            interval_hours=24.0,
            preferred_hour=2,  # Run at 2 AM
            auto_cleanup_orphaned=True,
            orphaned_cleanup_interval_hours=2.0
        ),
        monitoring=MonitoringConfig(
            enable_detailed_logging=False,  # Less verbose in production
            log_file_operations=False,
            keep_cleanup_history_count=200,
            alert_on_cleanup_failure=True,
            alert_on_large_deletions=True
        )
    )