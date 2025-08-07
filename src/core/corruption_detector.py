"""
Corruption Detection and Recovery System

This module provides comprehensive corruption detection and automatic recovery
mechanisms for the trading system. It implements clean architecture principles
with proper separation of concerns and automated healing capabilities.

Features:
- Multi-level corruption detection (file, data, state)
- Automatic recovery mechanisms with fallback strategies
- Quarantine system for corrupted data
- Repair strategies for recoverable corruption
- Health monitoring and alerting
- Comprehensive audit logging
"""

import hashlib
import logging
import json
import time
import shutil
from typing import Dict, Any, List, Optional, Union, Callable, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from collections import deque
import numpy as np

from src.core.unified_serialization import unified_serializer, SerializationFormat
from src.core.data_integrity_validator import data_integrity_validator, ValidationLevel, ValidationResult
from src.core.storage_error_handler import storage_error_handler, StorageErrorType

logger = logging.getLogger(__name__)


class CorruptionSeverity(Enum):
    """Severity levels for corruption detection"""
    MINOR = "minor"          # Recoverable corruption with minimal impact
    MODERATE = "moderate"    # Corruption requiring attention but not critical
    SEVERE = "severe"        # Serious corruption affecting system functionality
    CRITICAL = "critical"    # System-threatening corruption requiring immediate action


class CorruptionType(Enum):
    """Types of corruption that can be detected"""
    FILE_CORRUPTION = auto()
    DATA_STRUCTURE_CORRUPTION = auto()
    CHECKSUM_MISMATCH = auto()
    INCOMPLETE_WRITE = auto()
    METADATA_CORRUPTION = auto()
    TIMESTAMP_ANOMALY = auto()
    STATE_INCONSISTENCY = auto()
    MEMORY_CORRUPTION = auto()
    SERIALIZATION_CORRUPTION = auto()


class RecoveryStrategy(Enum):
    """Available recovery strategies"""
    RESTORE_FROM_BACKUP = "restore_from_backup"
    REPAIR_IN_PLACE = "repair_in_place"
    RECREATE_FROM_DEFAULTS = "recreate_from_defaults"
    QUARANTINE_AND_ALERT = "quarantine_and_alert"
    ROLLBACK_TRANSACTION = "rollback_transaction"
    PARTIAL_RECOVERY = "partial_recovery"


@dataclass
class CorruptionDetection:
    """Details of detected corruption"""
    corruption_type: CorruptionType
    severity: CorruptionSeverity
    component: str
    file_path: Optional[str] = None
    description: str = ""
    detected_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    recovery_possible: bool = True
    recommended_strategy: RecoveryStrategy = RecoveryStrategy.RESTORE_FROM_BACKUP


@dataclass
class RecoveryResult:
    """Result of corruption recovery attempt"""
    success: bool
    strategy_used: RecoveryStrategy
    recovery_time: float
    data_recovered: bool = False
    backup_created: bool = False
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ICorruptionDetector(ABC):
    """Interface for corruption detectors"""
    
    @abstractmethod
    def detect(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> List[CorruptionDetection]:
        """Detect corruption in data"""
        pass
    
    @abstractmethod
    def can_handle(self, data_type: str) -> bool:
        """Check if this detector can handle the data type"""
        pass


class FileCorruptionDetector(ICorruptionDetector):
    """Detects file-level corruption"""
    
    def can_handle(self, data_type: str) -> bool:
        return data_type == "file"
    
    def detect(self, file_path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> List[CorruptionDetection]:
        """Detect file corruption"""
        detections = []
        file_path = Path(file_path)
        
        try:
            if not file_path.exists():
                detections.append(CorruptionDetection(
                    corruption_type=CorruptionType.FILE_CORRUPTION,
                    severity=CorruptionSeverity.SEVERE,
                    component="file_system",
                    file_path=str(file_path),
                    description="File does not exist",
                    recovery_possible=False
                ))
                return detections
            
            # Check file size
            file_size = file_path.stat().st_size
            if file_size == 0:
                detections.append(CorruptionDetection(
                    corruption_type=CorruptionType.INCOMPLETE_WRITE,
                    severity=CorruptionSeverity.MODERATE,
                    component="file_system",
                    file_path=str(file_path),
                    description="File is empty",
                    metadata={"file_size": file_size}
                ))
            
            # Check file permissions
            if not file_path.is_readable():
                detections.append(CorruptionDetection(
                    corruption_type=CorruptionType.FILE_CORRUPTION,
                    severity=CorruptionSeverity.MODERATE,
                    component="file_system",
                    file_path=str(file_path),
                    description="File is not readable"
                ))
            
            # Checksum validation if expected checksum provided
            if metadata and 'expected_checksum' in metadata:
                try:
                    with open(file_path, 'rb') as f:
                        actual_checksum = hashlib.sha256(f.read()).hexdigest()
                    
                    if actual_checksum != metadata['expected_checksum']:
                        detections.append(CorruptionDetection(
                            corruption_type=CorruptionType.CHECKSUM_MISMATCH,
                            severity=CorruptionSeverity.SEVERE,
                            component="file_system",
                            file_path=str(file_path),
                            description="File checksum mismatch",
                            metadata={
                                "expected_checksum": metadata['expected_checksum'],
                                "actual_checksum": actual_checksum
                            }
                        ))
                except Exception as e:
                    detections.append(CorruptionDetection(
                        corruption_type=CorruptionType.FILE_CORRUPTION,
                        severity=CorruptionSeverity.SEVERE,
                        component="file_system",
                        file_path=str(file_path),
                        description=f"Error reading file for checksum: {e}"
                    ))
            
            # Check file modification time for anomalies
            if metadata and 'expected_mtime_range' in metadata:
                mtime = file_path.stat().st_mtime
                min_time, max_time = metadata['expected_mtime_range']
                
                if not (min_time <= mtime <= max_time):
                    detections.append(CorruptionDetection(
                        corruption_type=CorruptionType.TIMESTAMP_ANOMALY,
                        severity=CorruptionSeverity.MINOR,
                        component="file_system",
                        file_path=str(file_path),
                        description="File modification time outside expected range",
                        metadata={
                            "actual_mtime": mtime,
                            "expected_range": (min_time, max_time)
                        }
                    ))
            
        except Exception as e:
            detections.append(CorruptionDetection(
                corruption_type=CorruptionType.FILE_CORRUPTION,
                severity=CorruptionSeverity.CRITICAL,
                component="file_system",
                file_path=str(file_path),
                description=f"Critical error accessing file: {e}",
                recovery_possible=False
            ))
        
        return detections


class DataStructureCorruptionDetector(ICorruptionDetector):
    """Detects corruption in data structures"""
    
    def can_handle(self, data_type: str) -> bool:
        return data_type in ["dict", "list", "trading_data", "neural_data"]
    
    def detect(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> List[CorruptionDetection]:
        """Detect data structure corruption"""
        detections = []
        
        try:
            # Check for None data
            if data is None:
                detections.append(CorruptionDetection(
                    corruption_type=CorruptionType.DATA_STRUCTURE_CORRUPTION,
                    severity=CorruptionSeverity.SEVERE,
                    component="data_validator",
                    description="Data is None",
                    recovery_possible=True,
                    recommended_strategy=RecoveryStrategy.RECREATE_FROM_DEFAULTS
                ))
                return detections
            
            # Check for NaN/inf in numeric data
            if isinstance(data, dict):
                nan_inf_detections = self._detect_nan_inf_in_dict(data)
                detections.extend(nan_inf_detections)
            
            # Check for circular references
            try:
                json.dumps(data, default=str)
            except (ValueError, TypeError) as e:
                if "circular reference" in str(e).lower():
                    detections.append(CorruptionDetection(
                        corruption_type=CorruptionType.DATA_STRUCTURE_CORRUPTION,
                        severity=CorruptionSeverity.MODERATE,
                        component="data_validator",
                        description="Circular reference detected in data structure"
                    ))
            
            # Check data size for anomalies
            if isinstance(data, (dict, list)):
                size = len(data)
                if metadata and 'expected_size_range' in metadata:
                    min_size, max_size = metadata['expected_size_range']
                    if not (min_size <= size <= max_size):
                        detections.append(CorruptionDetection(
                            corruption_type=CorruptionType.DATA_STRUCTURE_CORRUPTION,
                            severity=CorruptionSeverity.MINOR,
                            component="data_validator",
                            description=f"Data size {size} outside expected range [{min_size}, {max_size}]",
                            metadata={"actual_size": size, "expected_range": (min_size, max_size)}
                        ))
            
            # Validate using data integrity validator
            if metadata and 'component' in metadata:
                validation_report = data_integrity_validator.validate_data(
                    data=data,
                    component=metadata['component'],
                    level=ValidationLevel.STRICT,
                    store_report=False
                )
                
                if validation_report.result == ValidationResult.CORRUPTED:
                    detections.append(CorruptionDetection(
                        corruption_type=CorruptionType.DATA_STRUCTURE_CORRUPTION,
                        severity=CorruptionSeverity.CRITICAL,
                        component=metadata['component'],
                        description="Data structure marked as corrupted by integrity validator",
                        metadata={"validation_issues": len(validation_report.issues)}
                    ))
                elif validation_report.result == ValidationResult.INVALID:
                    detections.append(CorruptionDetection(
                        corruption_type=CorruptionType.DATA_STRUCTURE_CORRUPTION,
                        severity=CorruptionSeverity.MODERATE,
                        component=metadata['component'],
                        description="Data structure validation failed",
                        metadata={"validation_issues": len(validation_report.issues)}
                    ))
            
        except Exception as e:
            detections.append(CorruptionDetection(
                corruption_type=CorruptionType.DATA_STRUCTURE_CORRUPTION,
                severity=CorruptionSeverity.CRITICAL,
                component="data_validator",
                description=f"Critical error during corruption detection: {e}",
                recovery_possible=False
            ))
        
        return detections
    
    def _detect_nan_inf_in_dict(self, data: Dict[str, Any], path: str = "") -> List[CorruptionDetection]:
        """Recursively detect NaN/inf values in dictionary"""
        detections = []
        
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            
            if isinstance(value, (int, float)):
                if np.isnan(value):
                    detections.append(CorruptionDetection(
                        corruption_type=CorruptionType.DATA_STRUCTURE_CORRUPTION,
                        severity=CorruptionSeverity.MODERATE,
                        component="data_validator",
                        description=f"NaN value found at {current_path}",
                        metadata={"field_path": current_path, "value": "NaN"}
                    ))
                elif np.isinf(value):
                    detections.append(CorruptionDetection(
                        corruption_type=CorruptionType.DATA_STRUCTURE_CORRUPTION,
                        severity=CorruptionSeverity.MODERATE,
                        component="data_validator",
                        description=f"Infinite value found at {current_path}",
                        metadata={"field_path": current_path, "value": "inf"}
                    ))
            elif isinstance(value, dict):
                detections.extend(self._detect_nan_inf_in_dict(value, current_path))
            elif isinstance(value, list):
                try:
                    arr = np.array(value)
                    if np.any(np.isnan(arr)):
                        detections.append(CorruptionDetection(
                            corruption_type=CorruptionType.DATA_STRUCTURE_CORRUPTION,
                            severity=CorruptionSeverity.MODERATE,
                            component="data_validator",
                            description=f"NaN values found in array at {current_path}",
                            metadata={"field_path": current_path}
                        ))
                    if np.any(np.isinf(arr)):
                        detections.append(CorruptionDetection(
                            corruption_type=CorruptionType.DATA_STRUCTURE_CORRUPTION,
                            severity=CorruptionSeverity.MODERATE,
                            component="data_validator",
                            description=f"Infinite values found in array at {current_path}",
                            metadata={"field_path": current_path}
                        ))
                except (ValueError, TypeError):
                    pass  # Skip if can't convert to array
        
        return detections


class StateConsistencyDetector(ICorruptionDetector):
    """Detects state consistency issues"""
    
    def can_handle(self, data_type: str) -> bool:
        return data_type == "state_consistency"
    
    def detect(self, components_data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> List[CorruptionDetection]:
        """Detect state consistency issues across components"""
        detections = []
        
        try:
            # Check timestamp consistency
            timestamps = {}
            for component, data in components_data.items():
                if isinstance(data, dict) and 'saved_at' in data:
                    timestamps[component] = data['saved_at']
            
            if len(timestamps) > 1:
                min_time = min(timestamps.values())
                max_time = max(timestamps.values())
                time_diff = max_time - min_time
                
                # If timestamps differ by more than 5 minutes, flag as inconsistent
                if time_diff > 300:
                    detections.append(CorruptionDetection(
                        corruption_type=CorruptionType.STATE_INCONSISTENCY,
                        severity=CorruptionSeverity.MINOR,
                        component="state_coordinator",
                        description=f"Component timestamps inconsistent (diff: {time_diff:.1f}s)",
                        metadata={"timestamp_diff": time_diff, "timestamps": timestamps}
                    ))
            
            # Check version consistency
            versions = {}
            for component, data in components_data.items():
                if isinstance(data, dict) and 'version' in data:
                    versions[component] = data['version']
            
            if len(set(versions.values())) > 1:
                detections.append(CorruptionDetection(
                    corruption_type=CorruptionType.STATE_INCONSISTENCY,
                    severity=CorruptionSeverity.MINOR,
                    component="state_coordinator",
                    description="Component versions inconsistent",
                    metadata={"versions": versions}
                ))
            
            # Check for missing expected components
            if metadata and 'expected_components' in metadata:
                expected = set(metadata['expected_components'])
                actual = set(components_data.keys())
                missing = expected - actual
                
                if missing:
                    detections.append(CorruptionDetection(
                        corruption_type=CorruptionType.STATE_INCONSISTENCY,
                        severity=CorruptionSeverity.MODERATE,
                        component="state_coordinator",
                        description=f"Missing expected components: {missing}",
                        metadata={"missing_components": list(missing)}
                    ))
            
        except Exception as e:
            detections.append(CorruptionDetection(
                corruption_type=CorruptionType.STATE_INCONSISTENCY,
                severity=CorruptionSeverity.CRITICAL,
                component="state_coordinator",
                description=f"Error checking state consistency: {e}"
            ))
        
        return detections


class CorruptionRecoveryManager:
    """Manages corruption recovery operations"""
    
    def __init__(self, quarantine_dir: str = "data/quarantine", backup_dir: str = "data/backups"):
        self.quarantine_dir = Path(quarantine_dir)
        self.backup_dir = Path(backup_dir)
        
        # Ensure directories exist
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Recovery strategies
        self.recovery_strategies = {
            RecoveryStrategy.RESTORE_FROM_BACKUP: self._restore_from_backup,
            RecoveryStrategy.REPAIR_IN_PLACE: self._repair_in_place,
            RecoveryStrategy.RECREATE_FROM_DEFAULTS: self._recreate_from_defaults,
            RecoveryStrategy.QUARANTINE_AND_ALERT: self._quarantine_and_alert,
            RecoveryStrategy.PARTIAL_RECOVERY: self._partial_recovery
        }
    
    def recover(self, detection: CorruptionDetection, 
               original_data: Any = None,
               backup_data: Any = None) -> RecoveryResult:
        """Attempt to recover from corruption"""
        start_time = time.time()
        
        try:
            strategy = detection.recommended_strategy
            if strategy not in self.recovery_strategies:
                strategy = RecoveryStrategy.QUARANTINE_AND_ALERT
            
            result = self.recovery_strategies[strategy](detection, original_data, backup_data)
            result.recovery_time = time.time() - start_time
            
            logger.info(f"Corruption recovery {'succeeded' if result.success else 'failed'} "
                       f"for {detection.component} using {strategy.value}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during corruption recovery: {e}")
            return RecoveryResult(
                success=False,
                strategy_used=detection.recommended_strategy,
                recovery_time=time.time() - start_time,
                errors=[str(e)]
            )
    
    def _restore_from_backup(self, detection: CorruptionDetection, 
                           original_data: Any, backup_data: Any) -> RecoveryResult:
        """Restore data from backup"""
        if backup_data is None:
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.RESTORE_FROM_BACKUP,
                recovery_time=0.0,
                errors=["No backup data available"]
            )
        
        try:
            # If file corruption, restore file
            if detection.file_path and Path(detection.file_path).exists():
                backup_file = self.backup_dir / f"{detection.component}_backup.dat"
                if backup_file.exists():
                    shutil.copy2(backup_file, detection.file_path)
                    return RecoveryResult(
                        success=True,
                        strategy_used=RecoveryStrategy.RESTORE_FROM_BACKUP,
                        recovery_time=0.0,
                        data_recovered=True
                    )
            
            # For data corruption, backup_data contains the restored data
            return RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.RESTORE_FROM_BACKUP,
                recovery_time=0.0,
                data_recovered=True,
                metadata={"restored_data_size": len(str(backup_data))}
            )
            
        except Exception as e:
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.RESTORE_FROM_BACKUP,
                recovery_time=0.0,
                errors=[f"Backup restore failed: {e}"]
            )
    
    def _repair_in_place(self, detection: CorruptionDetection, 
                        original_data: Any, backup_data: Any) -> RecoveryResult:
        """Attempt to repair corruption in place"""
        try:
            if detection.corruption_type == CorruptionType.DATA_STRUCTURE_CORRUPTION:
                # Repair NaN/inf values
                if isinstance(original_data, dict):
                    repaired_data = self._repair_nan_inf_values(original_data)
                    return RecoveryResult(
                        success=True,
                        strategy_used=RecoveryStrategy.REPAIR_IN_PLACE,
                        recovery_time=0.0,
                        data_recovered=True,
                        metadata={"repaired_data": repaired_data}
                    )
            
            # For other types, fall back to quarantine
            return self._quarantine_and_alert(detection, original_data, backup_data)
            
        except Exception as e:
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.REPAIR_IN_PLACE,
                recovery_time=0.0,
                errors=[f"Repair failed: {e}"]
            )
    
    def _repair_nan_inf_values(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Repair NaN/inf values in data structure"""
        def repair_value(value):
            if isinstance(value, float):
                if np.isnan(value):
                    return 0.0
                elif np.isinf(value):
                    return 1000.0 if value > 0 else -1000.0
            elif isinstance(value, dict):
                return {k: repair_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [repair_value(v) for v in value]
            return value
        
        return {k: repair_value(v) for k, v in data.items()}
    
    def _recreate_from_defaults(self, detection: CorruptionDetection, 
                               original_data: Any, backup_data: Any) -> RecoveryResult:
        """Recreate data using default values"""
        try:
            # Component-specific default data
            default_data = self._get_default_data(detection.component)
            
            return RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.RECREATE_FROM_DEFAULTS,
                recovery_time=0.0,
                data_recovered=True,
                metadata={"default_data": default_data}
            )
            
        except Exception as e:
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.RECREATE_FROM_DEFAULTS,
                recovery_time=0.0,
                errors=[f"Default recreation failed: {e}"]
            )
    
    def _get_default_data(self, component: str) -> Dict[str, Any]:
        """Get default data for a component"""
        defaults = {
            'portfolio_manager': {
                'positions': {},
                'trade_history': [],
                'performance_history': [],
                'daily_returns': [],
                'pending_orders': {}
            },
            'experience_manager': {
                'experience_buffer': [],
                'priority_buffer': [],
                'previous_task_buffer': [],
                'stats': {
                    'total_experiences_stored': 0,
                    'priority_experiences_stored': 0,
                    'validation_failures': 0
                }
            },
            'intelligence_engine': {
                'patterns': {},
                'recent_outcomes': [],
                'recent_signals': [],
                'historical_processed': False,
                'bootstrap_stats': {
                    'total_bars_processed': 0,
                    'patterns_discovered': 0
                }
            }
        }
        
        return defaults.get(component, {'status': 'initialized', 'data': {}})
    
    def _quarantine_and_alert(self, detection: CorruptionDetection, 
                             original_data: Any, backup_data: Any) -> RecoveryResult:
        """Quarantine corrupted data and alert"""
        try:
            quarantine_file = self.quarantine_dir / f"{detection.component}_{int(time.time())}.quarantine"
            
            # Save corrupted data for analysis
            quarantine_data = {
                'detection': {
                    'corruption_type': detection.corruption_type.name,
                    'severity': detection.severity.value,
                    'component': detection.component,
                    'description': detection.description,
                    'detected_at': detection.detected_at,
                    'metadata': detection.metadata
                },
                'original_data': original_data,
                'timestamp': time.time()
            }
            
            with open(quarantine_file, 'w') as f:
                json.dump(quarantine_data, f, indent=2, default=str)
            
            logger.warning(f"Corrupted data quarantined: {quarantine_file}")
            
            return RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.QUARANTINE_AND_ALERT,
                recovery_time=0.0,
                metadata={"quarantine_file": str(quarantine_file)}
            )
            
        except Exception as e:
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.QUARANTINE_AND_ALERT,
                recovery_time=0.0,
                errors=[f"Quarantine failed: {e}"]
            )
    
    def _partial_recovery(self, detection: CorruptionDetection, 
                         original_data: Any, backup_data: Any) -> RecoveryResult:
        """Attempt partial recovery of uncorrupted data"""
        try:
            if isinstance(original_data, dict) and isinstance(backup_data, dict):
                # Merge valid data from original with backup
                recovered_data = {}
                
                # Start with backup data as base
                recovered_data.update(backup_data)
                
                # Overlay non-corrupted parts from original
                for key, value in original_data.items():
                    if not self._is_value_corrupted(value):
                        recovered_data[key] = value
                
                return RecoveryResult(
                    success=True,
                    strategy_used=RecoveryStrategy.PARTIAL_RECOVERY,
                    recovery_time=0.0,
                    data_recovered=True,
                    metadata={
                        "recovered_keys": list(recovered_data.keys()),
                        "partial_recovery": True
                    }
                )
            
            # Fall back to backup if partial recovery not possible
            return self._restore_from_backup(detection, original_data, backup_data)
            
        except Exception as e:
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.PARTIAL_RECOVERY,
                recovery_time=0.0,
                errors=[f"Partial recovery failed: {e}"]
            )
    
    def _is_value_corrupted(self, value: Any) -> bool:
        """Check if a value appears to be corrupted"""
        try:
            if isinstance(value, float):
                return np.isnan(value) or np.isinf(value)
            elif isinstance(value, (list, tuple)):
                return any(self._is_value_corrupted(v) for v in value)
            elif isinstance(value, dict):
                return any(self._is_value_corrupted(v) for v in value.values())
            return False
        except Exception:
            return True  # If we can't check, assume corrupted


class CorruptionDetectionSystem:
    """
    Comprehensive corruption detection and recovery system.
    
    This class serves as the main interface for corruption detection and
    automatic recovery in the trading system. It coordinates multiple
    detectors and recovery strategies to maintain system integrity.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize detectors
        self.detectors: List[ICorruptionDetector] = [
            FileCorruptionDetector(),
            DataStructureCorruptionDetector(),
            StateConsistencyDetector()
        ]
        
        # Initialize recovery manager
        self.recovery_manager = CorruptionRecoveryManager(
            quarantine_dir=self.config.get('quarantine_dir', 'data/quarantine'),
            backup_dir=self.config.get('backup_dir', 'data/backups')
        )
        
        # Detection history
        self.detection_history = deque(maxlen=1000)
        self.recovery_history = deque(maxlen=500)
        
        # Statistics
        self.stats = {
            'total_detections': 0,
            'total_recoveries': 0,
            'successful_recoveries': 0,
            'quarantined_items': 0,
            'last_scan_time': 0.0
        }
        
        # Health monitoring
        self.health_monitors: Dict[str, Callable] = {}
        
        logger.info("CorruptionDetectionSystem initialized")
    
    def detect_corruption(self, 
                         data: Any, 
                         data_type: str,
                         component: str,
                         metadata: Optional[Dict[str, Any]] = None) -> List[CorruptionDetection]:
        """
        Detect corruption in data using appropriate detectors
        
        Args:
            data: Data to check for corruption
            data_type: Type of data being checked
            component: Component name for context
            metadata: Additional metadata for detection
            
        Returns:
            List of detected corruption issues
        """
        detections = []
        
        try:
            # Add component to metadata
            detection_metadata = metadata or {}
            detection_metadata['component'] = component
            
            # Run appropriate detectors
            for detector in self.detectors:
                if detector.can_handle(data_type):
                    try:
                        detector_results = detector.detect(data, detection_metadata)
                        detections.extend(detector_results)
                    except Exception as e:
                        logger.error(f"Error in {detector.__class__.__name__}: {e}")
                        # Create error detection
                        detections.append(CorruptionDetection(
                            corruption_type=CorruptionType.MEMORY_CORRUPTION,
                            severity=CorruptionSeverity.CRITICAL,
                            component=component,
                            description=f"Detector error: {e}",
                            recovery_possible=False
                        ))
            
            # Update statistics
            self.stats['total_detections'] += len(detections)
            self.stats['last_scan_time'] = time.time()
            
            # Store in history
            for detection in detections:
                self.detection_history.append(detection)
            
            if detections:
                logger.warning(f"Detected {len(detections)} corruption issues in {component}")
                
        except Exception as e:
            logger.error(f"Critical error in corruption detection: {e}")
            detections.append(CorruptionDetection(
                corruption_type=CorruptionType.MEMORY_CORRUPTION,
                severity=CorruptionSeverity.CRITICAL,
                component=component,
                description=f"Detection system error: {e}",
                recovery_possible=False
            ))
        
        return detections
    
    def detect_and_recover(self, 
                          data: Any, 
                          data_type: str,
                          component: str,
                          backup_data: Any = None,
                          metadata: Optional[Dict[str, Any]] = None,
                          auto_recover: bool = True) -> Dict[str, Any]:
        """
        Detect corruption and automatically attempt recovery
        
        Args:
            data: Data to check and potentially recover
            data_type: Type of data being processed
            component: Component name for context
            backup_data: Backup data for recovery
            metadata: Additional metadata
            auto_recover: Whether to automatically attempt recovery
            
        Returns:
            Dictionary containing detection results and recovery status
        """
        result = {
            'detections': [],
            'recoveries': [],
            'recovered_data': data,
            'recovery_successful': True,
            'needs_attention': False
        }
        
        try:
            # Detect corruption
            detections = self.detect_corruption(data, data_type, component, metadata)
            result['detections'] = [
                {
                    'type': d.corruption_type.name,
                    'severity': d.severity.value,
                    'description': d.description,
                    'recoverable': d.recovery_possible
                }
                for d in detections
            ]
            
            if not detections:
                return result
            
            # Attempt recovery if enabled
            if auto_recover:
                for detection in detections:
                    if detection.recovery_possible:
                        recovery_result = self.recovery_manager.recover(
                            detection, data, backup_data
                        )
                        
                        result['recoveries'].append({
                            'strategy': recovery_result.strategy_used.value,
                            'success': recovery_result.success,
                            'errors': recovery_result.errors
                        })
                        
                        # Update statistics
                        self.stats['total_recoveries'] += 1
                        if recovery_result.success:
                            self.stats['successful_recoveries'] += 1
                            if 'quarantine' in recovery_result.strategy_used.value:
                                self.stats['quarantined_items'] += 1
                        
                        # Store in history
                        self.recovery_history.append(recovery_result)
                        
                        # Update recovered data if available
                        if (recovery_result.success and recovery_result.data_recovered and 
                            'restored_data' in recovery_result.metadata):
                            result['recovered_data'] = recovery_result.metadata['restored_data']
                    else:
                        result['recovery_successful'] = False
                        result['needs_attention'] = True
            else:
                result['recovery_successful'] = False
                result['needs_attention'] = any(d.severity in [CorruptionSeverity.SEVERE, CorruptionSeverity.CRITICAL] 
                                              for d in detections)
            
        except Exception as e:
            logger.error(f"Error in detect_and_recover: {e}")
            result['recovery_successful'] = False
            result['needs_attention'] = True
            result['error'] = str(e)
        
        return result
    
    def register_health_monitor(self, component: str, monitor_func: Callable):
        """Register a health monitoring function for a component"""
        self.health_monitors[component] = monitor_func
        logger.debug(f"Registered health monitor for {component}")
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run health checks on all monitored components"""
        health_results = {}
        
        for component, monitor_func in self.health_monitors.items():
            try:
                health_results[component] = monitor_func()
            except Exception as e:
                logger.error(f"Health check failed for {component}: {e}")
                health_results[component] = {'status': 'error', 'error': str(e)}
        
        return health_results
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health including corruption statistics"""
        try:
            # Calculate health metrics
            total_detections = self.stats['total_detections']
            total_recoveries = self.stats['total_recoveries']
            
            recovery_rate = (
                (self.stats['successful_recoveries'] / total_recoveries * 100) 
                if total_recoveries > 0 else 100.0
            )
            
            # Recent detection trends
            recent_detections = [
                d for d in self.detection_history 
                if time.time() - d.detected_at < 3600  # Last hour
            ]
            
            critical_detections = [
                d for d in recent_detections 
                if d.severity == CorruptionSeverity.CRITICAL
            ]
            
            health_status = "healthy"
            if len(critical_detections) > 0:
                health_status = "critical"
            elif len(recent_detections) > 10:
                health_status = "degraded"
            elif len(recent_detections) > 5:
                health_status = "warning"
            
            return {
                'overall_status': health_status,
                'statistics': self.stats.copy(),
                'recovery_rate': recovery_rate,
                'recent_detections': len(recent_detections),
                'critical_issues': len(critical_detections),
                'component_health': self.run_health_checks(),
                'last_scan_age': time.time() - self.stats['last_scan_time'],
                'quarantine_items': self.stats['quarantined_items']
            }
            
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {
                'overall_status': 'error',
                'error': str(e)
            }
    
    def cleanup_quarantine(self, older_than_days: int = 30):
        """Clean up old quarantined files"""
        try:
            cutoff_time = time.time() - (older_than_days * 24 * 3600)
            cleaned_count = 0
            
            for file_path in self.recovery_manager.quarantine_dir.glob("*.quarantine"):
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} old quarantine files")
            
        except Exception as e:
            logger.error(f"Error cleaning up quarantine: {e}")


# Global corruption detection system instance
corruption_detector = CorruptionDetectionSystem()


def detect_corruption(data: Any, data_type: str, component: str, 
                     metadata: Optional[Dict[str, Any]] = None) -> List[CorruptionDetection]:
    """
    Convenience function to detect corruption
    
    Args:
        data: Data to check for corruption
        data_type: Type of data being checked
        component: Component name for context
        metadata: Additional metadata for detection
        
    Returns:
        List of detected corruption issues
    """
    return corruption_detector.detect_corruption(data, data_type, component, metadata)


def detect_and_recover(data: Any, data_type: str, component: str,
                      backup_data: Any = None, metadata: Optional[Dict[str, Any]] = None,
                      auto_recover: bool = True) -> Dict[str, Any]:
    """
    Convenience function to detect corruption and attempt recovery
    """
    return corruption_detector.detect_and_recover(
        data, data_type, component, backup_data, metadata, auto_recover
    )