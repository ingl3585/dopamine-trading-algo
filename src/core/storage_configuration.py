"""
Unified Storage Configuration System

This module provides centralized configuration management for all storage-related
operations in the trading system. It implements the Single Responsibility Principle
by managing storage configuration separate from storage operations themselves.

Features:
- Centralized storage configuration management
- Component-specific storage settings
- Environment-based configuration profiles
- Dynamic configuration updates with validation
- Configuration versioning and migration
- Storage policy enforcement
- Performance optimization settings
"""

import logging
import json
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path

from src.core.unified_serialization import SerializationFormat
from src.core.data_integrity_validator import ValidationLevel

logger = logging.getLogger(__name__)


class StorageProfile(Enum):
    """Predefined storage profiles for different environments"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"
    HIGH_PERFORMANCE = "high_performance"
    LOW_RESOURCE = "low_resource"


class CompressionLevel(Enum):
    """Compression levels for storage operations"""
    NONE = 0
    FAST = 1
    BALANCED = 6
    MAXIMUM = 9


@dataclass
class ComponentStorageConfig:
    """Storage configuration for a specific component"""
    component_name: str
    enabled: bool = True
    serialization_format: SerializationFormat = SerializationFormat.JSON
    compression_enabled: bool = False
    compression_level: CompressionLevel = CompressionLevel.BALANCED
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    auto_save_enabled: bool = True
    save_interval_seconds: int = 300  # 5 minutes
    max_history_size: int = 10
    backup_enabled: bool = True
    max_file_size_mb: int = 100
    priority: int = 50  # 0-100, higher = more important
    custom_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GlobalStorageConfig:
    """Global storage configuration settings"""
    base_storage_path: str = "data"
    temp_storage_path: str = "data/temp"
    backup_storage_path: str = "data/backups"
    log_storage_path: str = "logs"
    default_serialization_format: SerializationFormat = SerializationFormat.JSON
    default_compression_threshold_kb: int = 10  # 10KB
    default_validation_level: ValidationLevel = ValidationLevel.STANDARD
    max_concurrent_operations: int = 10
    operation_timeout_seconds: int = 120
    auto_cleanup_enabled: bool = True
    cleanup_interval_hours: int = 24
    max_storage_size_gb: float = 10.0
    storage_warning_threshold_percent: float = 80.0
    enable_performance_monitoring: bool = True
    enable_integrity_checking: bool = True
    enable_automatic_backups: bool = True
    backup_retention_days: int = 30


class StorageConfigurationManager:
    """
    Centralized storage configuration management system.
    
    This class manages all storage-related configuration for the trading system,
    providing a single source of truth for storage policies and settings.
    """
    
    def __init__(self, config_file: Optional[str] = None, profile: StorageProfile = StorageProfile.DEVELOPMENT):
        self.config_file = Path(config_file) if config_file else None
        self.current_profile = profile
        
        # Initialize configurations
        self.global_config = GlobalStorageConfig()
        self.component_configs: Dict[str, ComponentStorageConfig] = {}
        self.profile_configs: Dict[StorageProfile, Dict[str, Any]] = {}
        
        # Configuration history for rollback
        self.config_history: List[Dict[str, Any]] = []
        self.max_config_history = 10
        
        # Statistics
        self.config_changes = 0
        self.last_update_time = time.time()
        
        # Initialize default profiles
        self._initialize_default_profiles()
        
        # Load configuration from file if provided
        if self.config_file and self.config_file.exists():
            self.load_configuration()
        else:
            # Apply default profile settings
            self.apply_profile(profile)
        
        logger.info(f"StorageConfigurationManager initialized with profile: {profile.value}")
    
    def _initialize_default_profiles(self):
        """Initialize default configuration profiles"""
        
        # Development profile - convenience over performance
        self.profile_configs[StorageProfile.DEVELOPMENT] = {
            'global': {
                'default_serialization_format': SerializationFormat.JSON,
                'default_validation_level': ValidationLevel.STANDARD,
                'enable_performance_monitoring': True,
                'enable_integrity_checking': True,
                'auto_cleanup_enabled': False,  # Manual cleanup in dev
                'max_storage_size_gb': 5.0
            },
            'components': {
                'default': {
                    'compression_enabled': False,  # Easier debugging
                    'validation_level': ValidationLevel.STANDARD,
                    'save_interval_seconds': 600,  # Less frequent saves
                    'max_history_size': 5
                }
            }
        }
        
        # Testing profile - reliability and validation
        self.profile_configs[StorageProfile.TESTING] = {
            'global': {
                'default_serialization_format': SerializationFormat.JSON,
                'default_validation_level': ValidationLevel.STRICT,
                'enable_performance_monitoring': True,
                'enable_integrity_checking': True,
                'auto_cleanup_enabled': True,
                'max_storage_size_gb': 2.0
            },
            'components': {
                'default': {
                    'compression_enabled': False,
                    'validation_level': ValidationLevel.STRICT,
                    'save_interval_seconds': 300,
                    'max_history_size': 3,
                    'backup_enabled': True
                }
            }
        }
        
        # Production profile - performance and efficiency
        self.profile_configs[StorageProfile.PRODUCTION] = {
            'global': {
                'default_serialization_format': SerializationFormat.COMPRESSED_JSON,
                'default_validation_level': ValidationLevel.STANDARD,
                'enable_performance_monitoring': True,
                'enable_integrity_checking': True,
                'auto_cleanup_enabled': True,
                'max_storage_size_gb': 50.0
            },
            'components': {
                'default': {
                    'compression_enabled': True,
                    'compression_level': CompressionLevel.BALANCED,
                    'validation_level': ValidationLevel.STANDARD,
                    'save_interval_seconds': 300,
                    'max_history_size': 10,
                    'backup_enabled': True
                },
                'experience_manager': {
                    'compression_enabled': True,
                    'compression_level': CompressionLevel.MAXIMUM,
                    'save_interval_seconds': 600,  # Less frequent for large data
                    'max_file_size_mb': 500,
                    'priority': 90
                },
                'portfolio_manager': {
                    'validation_level': ValidationLevel.STRICT,
                    'backup_enabled': True,
                    'priority': 95  # Highest priority for portfolio data
                }
            }
        }
        
        # High performance profile - speed over safety
        self.profile_configs[StorageProfile.HIGH_PERFORMANCE] = {
            'global': {
                'default_serialization_format': SerializationFormat.PICKLE,
                'default_validation_level': ValidationLevel.BASIC,
                'enable_performance_monitoring': False,  # Reduce overhead
                'enable_integrity_checking': False,
                'auto_cleanup_enabled': True,
                'max_concurrent_operations': 20
            },
            'components': {
                'default': {
                    'compression_enabled': False,  # Skip compression for speed
                    'validation_level': ValidationLevel.BASIC,
                    'save_interval_seconds': 600,
                    'max_history_size': 3,
                    'backup_enabled': False  # Skip backups for speed
                }
            }
        }
        
        # Low resource profile - minimal storage usage
        self.profile_configs[StorageProfile.LOW_RESOURCE] = {
            'global': {
                'default_serialization_format': SerializationFormat.COMPRESSED_PICKLE,
                'default_validation_level': ValidationLevel.BASIC,
                'max_storage_size_gb': 1.0,
                'auto_cleanup_enabled': True,
                'cleanup_interval_hours': 6  # More frequent cleanup
            },
            'components': {
                'default': {
                    'compression_enabled': True,
                    'compression_level': CompressionLevel.MAXIMUM,
                    'validation_level': ValidationLevel.BASIC,
                    'save_interval_seconds': 900,  # Less frequent saves
                    'max_history_size': 2,
                    'max_file_size_mb': 10
                }
            }
        }
    
    def apply_profile(self, profile: StorageProfile):
        """
        Apply a configuration profile to the current settings
        
        Args:
            profile: The storage profile to apply
        """
        try:
            if profile not in self.profile_configs:
                logger.error(f"Unknown profile: {profile}")
                return False
            
            # Save current configuration to history
            self._save_config_to_history()
            
            profile_config = self.profile_configs[profile]
            
            # Apply global configuration
            if 'global' in profile_config:
                global_settings = profile_config['global']
                for key, value in global_settings.items():
                    if hasattr(self.global_config, key):
                        setattr(self.global_config, key, value)
                        logger.debug(f"Applied global setting: {key} = {value}")
            
            # Apply component configurations
            if 'components' in profile_config:
                component_settings = profile_config['components']
                
                # Apply default component settings to all components
                if 'default' in component_settings:
                    default_settings = component_settings['default']
                    for component_name in self.component_configs:
                        self._update_component_config(component_name, default_settings)
                
                # Apply specific component settings
                for component_name, settings in component_settings.items():
                    if component_name != 'default':
                        if component_name not in self.component_configs:
                            # Create new component config
                            self.component_configs[component_name] = ComponentStorageConfig(component_name)
                        self._update_component_config(component_name, settings)
            
            self.current_profile = profile
            self.config_changes += 1
            self.last_update_time = time.time()
            
            logger.info(f"Applied storage profile: {profile.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error applying profile {profile}: {e}")
            return False
    
    def register_component(self, 
                          component_name: str,
                          config: Optional[ComponentStorageConfig] = None) -> ComponentStorageConfig:
        """
        Register a component with the storage configuration system
        
        Args:
            component_name: Name of the component to register
            config: Optional custom configuration for the component
            
        Returns:
            The component's storage configuration
        """
        try:
            if config is None:
                # Create default configuration
                config = ComponentStorageConfig(
                    component_name=component_name,
                    serialization_format=self.global_config.default_serialization_format,
                    validation_level=self.global_config.default_validation_level
                )
            
            self.component_configs[component_name] = config
            
            logger.info(f"Registered component storage config: {component_name}")
            logger.debug(f"Component config: {asdict(config)}")
            
            return config
            
        except Exception as e:
            logger.error(f"Error registering component {component_name}: {e}")
            # Return minimal default config
            return ComponentStorageConfig(component_name=component_name)
    
    def get_component_config(self, component_name: str) -> ComponentStorageConfig:
        """
        Get storage configuration for a specific component
        
        Args:
            component_name: Name of the component
            
        Returns:
            Component storage configuration
        """
        if component_name in self.component_configs:
            return self.component_configs[component_name]
        
        # Create and register default configuration
        logger.debug(f"Creating default config for component: {component_name}")
        return self.register_component(component_name)
    
    def update_component_config(self, 
                               component_name: str, 
                               **kwargs) -> bool:
        """
        Update configuration for a specific component
        
        Args:
            component_name: Name of the component
            **kwargs: Configuration parameters to update
            
        Returns:
            True if update was successful
        """
        try:
            if component_name not in self.component_configs:
                self.register_component(component_name)
            
            # Save current configuration to history
            self._save_config_to_history()
            
            config = self.component_configs[component_name]
            
            # Update configuration fields
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                    logger.debug(f"Updated {component_name}.{key} = {value}")
                else:
                    # Add to custom settings
                    config.custom_settings[key] = value
                    logger.debug(f"Added custom setting {component_name}.{key} = {value}")
            
            self.config_changes += 1
            self.last_update_time = time.time()
            
            logger.info(f"Updated configuration for component: {component_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating component {component_name} config: {e}")
            return False
    
    def _update_component_config(self, component_name: str, settings: Dict[str, Any]):
        """Internal method to update component configuration from settings dict"""
        if component_name not in self.component_configs:
            self.component_configs[component_name] = ComponentStorageConfig(component_name)
        
        config = self.component_configs[component_name]
        
        for key, value in settings.items():
            if hasattr(config, key):
                # Handle enum conversions
                if key == 'serialization_format' and isinstance(value, str):
                    try:
                        setattr(config, key, SerializationFormat(value))
                    except ValueError:
                        logger.warning(f"Invalid serialization format: {value}")
                elif key == 'compression_level' and isinstance(value, (str, int)):
                    try:
                        if isinstance(value, str):
                            setattr(config, key, CompressionLevel[value.upper()])
                        else:
                            setattr(config, key, CompressionLevel(value))
                    except (ValueError, KeyError):
                        logger.warning(f"Invalid compression level: {value}")
                elif key == 'validation_level' and isinstance(value, str):
                    try:
                        setattr(config, key, ValidationLevel(value))
                    except ValueError:
                        logger.warning(f"Invalid validation level: {value}")
                else:
                    setattr(config, key, value)
            else:
                config.custom_settings[key] = value
    
    def save_configuration(self, filepath: Optional[str] = None) -> bool:
        """
        Save current configuration to file
        
        Args:
            filepath: Optional path to save configuration
            
        Returns:
            True if save was successful
        """
        try:
            if filepath:
                save_path = Path(filepath)
            elif self.config_file:
                save_path = self.config_file
            else:
                save_path = Path("data/storage_config.json")
            
            # Prepare configuration data
            config_data = {
                'version': '2.0',
                'profile': self.current_profile.value,
                'last_updated': time.time(),
                'global_config': asdict(self.global_config),
                'component_configs': {
                    name: asdict(config) 
                    for name, config in self.component_configs.items()
                },
                'statistics': {
                    'config_changes': self.config_changes,
                    'last_update_time': self.last_update_time
                }
            }
            
            # Ensure directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save configuration
            with open(save_path, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
            
            logger.info(f"Storage configuration saved to: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def load_configuration(self, filepath: Optional[str] = None) -> bool:
        """
        Load configuration from file
        
        Args:
            filepath: Optional path to load configuration from
            
        Returns:
            True if load was successful
        """
        try:
            if filepath:
                load_path = Path(filepath)
            elif self.config_file:
                load_path = self.config_file
            else:
                load_path = Path("data/storage_config.json")
            
            if not load_path.exists():
                logger.info(f"Configuration file not found: {load_path}")
                return False
            
            # Load configuration data
            with open(load_path, 'r') as f:
                config_data = json.load(f)
            
            # Validate version compatibility
            version = config_data.get('version', '1.0')
            if version != '2.0':
                logger.warning(f"Configuration version mismatch: {version} (expected 2.0)")
            
            # Load global configuration
            if 'global_config' in config_data:
                global_config_data = config_data['global_config']
                for key, value in global_config_data.items():
                    if hasattr(self.global_config, key):
                        # Handle enum conversions
                        if key.endswith('_format') and isinstance(value, str):
                            try:
                                setattr(self.global_config, key, SerializationFormat(value))
                            except ValueError:
                                logger.warning(f"Invalid format value: {value}")
                        elif key.endswith('_level') and isinstance(value, str):
                            try:
                                setattr(self.global_config, key, ValidationLevel(value))
                            except ValueError:
                                logger.warning(f"Invalid level value: {value}")
                        else:
                            setattr(self.global_config, key, value)
            
            # Load component configurations
            if 'component_configs' in config_data:
                for name, config_dict in config_data['component_configs'].items():
                    try:
                        # Reconstruct component config
                        config = ComponentStorageConfig(component_name=name)
                        
                        for key, value in config_dict.items():
                            if hasattr(config, key):
                                # Handle enum conversions
                                if key == 'serialization_format' and isinstance(value, str):
                                    try:
                                        setattr(config, key, SerializationFormat(value))
                                    except ValueError:
                                        logger.warning(f"Invalid serialization format: {value}")
                                elif key == 'compression_level' and isinstance(value, (str, int)):
                                    try:
                                        if isinstance(value, str):
                                            setattr(config, key, CompressionLevel[value.upper()])
                                        else:
                                            setattr(config, key, CompressionLevel(value))
                                    except (ValueError, KeyError):
                                        logger.warning(f"Invalid compression level: {value}")
                                elif key == 'validation_level' and isinstance(value, str):
                                    try:
                                        setattr(config, key, ValidationLevel(value))
                                    except ValueError:
                                        logger.warning(f"Invalid validation level: {value}")
                                else:
                                    setattr(config, key, value)
                        
                        self.component_configs[name] = config
                        
                    except Exception as e:
                        logger.error(f"Error loading config for component {name}: {e}")
            
            # Load profile
            if 'profile' in config_data:
                try:
                    self.current_profile = StorageProfile(config_data['profile'])
                except ValueError:
                    logger.warning(f"Invalid profile: {config_data['profile']}")
            
            # Load statistics
            if 'statistics' in config_data:
                stats = config_data['statistics']
                self.config_changes = stats.get('config_changes', 0)
                self.last_update_time = stats.get('last_update_time', time.time())
            
            logger.info(f"Storage configuration loaded from: {load_path}")
            logger.info(f"Loaded {len(self.component_configs)} component configurations")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return False
    
    def _save_config_to_history(self):
        """Save current configuration to history for rollback"""
        try:
            config_snapshot = {
                'timestamp': time.time(),
                'profile': self.current_profile.value,
                'global_config': asdict(self.global_config),
                'component_configs': {
                    name: asdict(config) 
                    for name, config in self.component_configs.items()
                }
            }
            
            self.config_history.append(config_snapshot)
            
            # Maintain history size limit
            if len(self.config_history) > self.max_config_history:
                self.config_history.pop(0)
                
        except Exception as e:
            logger.error(f"Error saving config to history: {e}")
    
    def rollback_configuration(self, steps: int = 1) -> bool:
        """
        Rollback configuration to a previous state
        
        Args:
            steps: Number of steps to rollback (default: 1)
            
        Returns:
            True if rollback was successful
        """
        try:
            if len(self.config_history) < steps:
                logger.error(f"Cannot rollback {steps} steps, only {len(self.config_history)} available")
                return False
            
            # Get target configuration
            target_config = self.config_history[-(steps + 1)]
            
            # Restore global configuration
            global_config_data = target_config['global_config']
            for key, value in global_config_data.items():
                if hasattr(self.global_config, key):
                    setattr(self.global_config, key, value)
            
            # Restore component configurations
            self.component_configs.clear()
            for name, config_dict in target_config['component_configs'].items():
                config = ComponentStorageConfig(component_name=name)
                for key, value in config_dict.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                self.component_configs[name] = config
            
            # Restore profile
            self.current_profile = StorageProfile(target_config['profile'])
            
            # Remove rolled-back entries from history
            self.config_history = self.config_history[:-steps]
            
            self.config_changes += 1
            self.last_update_time = time.time()
            
            logger.info(f"Configuration rolled back {steps} step(s)")
            return True
            
        except Exception as e:
            logger.error(f"Error rolling back configuration: {e}")
            return False
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration"""
        try:
            return {
                'profile': self.current_profile.value,
                'global_config': asdict(self.global_config),
                'component_count': len(self.component_configs),
                'components': list(self.component_configs.keys()),
                'statistics': {
                    'config_changes': self.config_changes,
                    'last_update_time': self.last_update_time,
                    'history_size': len(self.config_history)
                },
                'storage_paths': {
                    'base': self.global_config.base_storage_path,
                    'temp': self.global_config.temp_storage_path,
                    'backup': self.global_config.backup_storage_path,
                    'log': self.global_config.log_storage_path
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting configuration summary: {e}")
            return {'error': str(e)}


# Global storage configuration manager instance
storage_config_manager = StorageConfigurationManager()


def get_component_storage_config(component_name: str) -> ComponentStorageConfig:
    """
    Convenience function to get component storage configuration
    
    Args:
        component_name: Name of the component
        
    Returns:
        Component storage configuration
    """
    return storage_config_manager.get_component_config(component_name)


def get_global_storage_config() -> GlobalStorageConfig:
    """
    Convenience function to get global storage configuration
    
    Returns:
        Global storage configuration
    """
    return storage_config_manager.global_config