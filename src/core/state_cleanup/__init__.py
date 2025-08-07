"""
State Cleanup System - Intelligent state file management and retention

This module provides comprehensive state file cleanup capabilities to prevent
storage bloat while maintaining data integrity and rollback capabilities.

Key Features:
- Intelligent retention policies (time, count, size-based)
- Safe file operations with validation and rollback
- Configurable cleanup strategies
- Seamless integration with StateCoordinator
- Thread-safe operations

Components:
- RetentionPolicies: Define how long to keep different types of state files
- FileManager: Safe file discovery, validation, and cleanup operations
- CleanupService: Orchestrates cleanup operations with planning and execution
- Configuration: JSON-based configuration with validation
- Coordinator: Integration layer with existing StateCoordinator

Usage:
    from src.core.state_cleanup import CleanupCoordinator
    
    coordinator = CleanupCoordinator()
    coordinator.initialize()
    coordinator.run_cleanup()
"""

from .coordinator import CleanupCoordinator
from .cleanup_service import CleanupService
from .configuration import CleanupConfiguration
from .file_manager import StateFileManager
from .retention_policies import RetentionPolicy, RetentionPolicySet

__all__ = [
    'CleanupCoordinator',
    'CleanupService', 
    'CleanupConfiguration',
    'StateFileManager',
    'RetentionPolicy',
    'RetentionPolicySet'
]

__version__ = '1.0.0'