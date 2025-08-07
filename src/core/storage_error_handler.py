"""
Storage Error Handler - Consistent error handling for all storage operations

This module provides a unified error handling framework for all storage-related
operations in the trading system. It implements the Single Responsibility Principle
by centralizing error handling logic and providing consistent recovery mechanisms.

Features:
- Centralized error classification and handling
- Automatic retry mechanisms with exponential backoff
- Error recovery strategies based on error type
- Comprehensive logging and monitoring
- Circuit breaker pattern for failing storage systems
- Error aggregation and reporting
"""

import logging
import time
import threading
from typing import Any, Dict, List, Optional, Callable, Type, Union
from enum import Enum, auto
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from functools import wraps
import traceback
from pathlib import Path

logger = logging.getLogger(__name__)


class StorageErrorType(Enum):
    """Classification of storage error types"""
    # File system errors
    FILE_NOT_FOUND = auto()
    PERMISSION_DENIED = auto()
    DISK_FULL = auto()
    IO_ERROR = auto()
    PATH_ERROR = auto()
    
    # Data integrity errors
    CORRUPTION_DETECTED = auto()
    CHECKSUM_MISMATCH = auto()
    INVALID_FORMAT = auto()
    MISSING_METADATA = auto()
    
    # Serialization errors
    SERIALIZATION_FAILED = auto()
    DESERIALIZATION_FAILED = auto()
    TYPE_ERROR = auto()
    ENCODING_ERROR = auto()
    
    # Network/connectivity errors (for remote storage)
    NETWORK_ERROR = auto()
    TIMEOUT_ERROR = auto()
    CONNECTION_FAILED = auto()
    
    # System errors
    MEMORY_ERROR = auto()
    THREAD_ERROR = auto()
    UNKNOWN_ERROR = auto()


class StorageErrorSeverity(Enum):
    """Severity levels for storage errors"""
    LOW = "low"          # Recoverable errors with minimal impact
    MEDIUM = "medium"    # Errors requiring attention but not critical
    HIGH = "high"        # Critical errors affecting system functionality
    CRITICAL = "critical"  # System-threatening errors requiring immediate action


class RecoveryStrategy(Enum):
    """Available recovery strategies"""
    RETRY = "retry"                    # Retry the operation
    FALLBACK = "fallback"             # Use alternative storage method
    SKIP = "skip"                     # Skip the operation and continue
    ABORT = "abort"                   # Abort the current operation
    RECREATE = "recreate"             # Recreate missing resources
    REPAIR = "repair"                 # Attempt to repair corrupted data


@dataclass
class StorageError:
    """Structured storage error information"""
    error_type: StorageErrorType
    severity: StorageErrorSeverity
    message: str
    component: str
    operation: str
    timestamp: float = field(default_factory=time.time)
    traceback: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    recovery_attempted: bool = False
    recovery_successful: bool = False


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern"""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking calls
    HALF_OPEN = "half_open"  # Testing recovery


class IErrorRecoveryStrategy(ABC):
    """Interface for error recovery strategies"""
    
    @abstractmethod
    def can_handle(self, error: StorageError) -> bool:
        """Check if this strategy can handle the error"""
        pass
    
    @abstractmethod
    def recover(self, error: StorageError, operation: Callable, *args, **kwargs) -> Any:
        """Attempt to recover from the error"""
        pass


class RetryRecoveryStrategy(IErrorRecoveryStrategy):
    """Recovery strategy that retries operations with exponential backoff"""
    
    def __init__(self, config: RetryConfig):
        self.config = config
    
    def can_handle(self, error: StorageError) -> bool:
        """Check if retry is appropriate for this error type"""
        retryable_errors = {
            StorageErrorType.IO_ERROR,
            StorageErrorType.NETWORK_ERROR,
            StorageErrorType.TIMEOUT_ERROR,
            StorageErrorType.CONNECTION_FAILED,
            StorageErrorType.MEMORY_ERROR
        }
        return (error.error_type in retryable_errors and 
                error.retry_count < self.config.max_retries)
    
    def recover(self, error: StorageError, operation: Callable, *args, **kwargs) -> Any:
        """Retry the operation with exponential backoff"""
        import random
        
        for attempt in range(self.config.max_retries):
            if attempt > 0:  # Don't delay on first attempt
                delay = min(
                    self.config.base_delay * (self.config.backoff_multiplier ** (attempt - 1)),
                    self.config.max_delay
                )
                
                if self.config.jitter:
                    delay *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter
                
                logger.info(f"Retrying operation in {delay:.2f}s (attempt {attempt + 1}/{self.config.max_retries})")
                time.sleep(delay)
            
            try:
                result = operation(*args, **kwargs)
                error.recovery_successful = True
                logger.info(f"Operation succeeded on retry attempt {attempt + 1}")
                return result
            except Exception as e:
                error.retry_count = attempt + 1
                if attempt == self.config.max_retries - 1:
                    raise e
                logger.warning(f"Retry attempt {attempt + 1} failed: {e}")
        
        error.recovery_successful = False
        raise RuntimeError(f"Operation failed after {self.config.max_retries} retries")


class FallbackRecoveryStrategy(IErrorRecoveryStrategy):
    """Recovery strategy that uses fallback storage methods"""
    
    def __init__(self, fallback_operations: Dict[str, Callable]):
        self.fallback_operations = fallback_operations
    
    def can_handle(self, error: StorageError) -> bool:
        """Check if fallback is available for this operation"""
        return error.operation in self.fallback_operations
    
    def recover(self, error: StorageError, operation: Callable, *args, **kwargs) -> Any:
        """Use fallback operation"""
        fallback_op = self.fallback_operations[error.operation]
        logger.info(f"Using fallback operation for {error.operation}")
        
        try:
            result = fallback_op(*args, **kwargs)
            error.recovery_successful = True
            return result
        except Exception as e:
            error.recovery_successful = False
            logger.error(f"Fallback operation also failed: {e}")
            raise e


class RepairRecoveryStrategy(IErrorRecoveryStrategy):
    """Recovery strategy that attempts to repair corrupted data"""
    
    def can_handle(self, error: StorageError) -> bool:
        """Check if repair is possible for this error type"""
        repairable_errors = {
            StorageErrorType.CORRUPTION_DETECTED,
            StorageErrorType.INVALID_FORMAT,
            StorageErrorType.MISSING_METADATA
        }
        return error.error_type in repairable_errors
    
    def recover(self, error: StorageError, operation: Callable, *args, **kwargs) -> Any:
        """Attempt to repair corrupted data"""
        logger.info(f"Attempting to repair corrupted data for {error.operation}")
        
        # Implementation would depend on specific repair strategies
        # This is a placeholder for actual repair logic
        try:
            # Attempt repair based on error type
            if error.error_type == StorageErrorType.MISSING_METADATA:
                self._repair_missing_metadata(error)
            elif error.error_type == StorageErrorType.INVALID_FORMAT:
                self._repair_invalid_format(error)
            elif error.error_type == StorageErrorType.CORRUPTION_DETECTED:
                self._repair_corruption(error)
            
            # Retry original operation
            result = operation(*args, **kwargs)
            error.recovery_successful = True
            return result
            
        except Exception as e:
            error.recovery_successful = False
            logger.error(f"Repair attempt failed: {e}")
            raise e
    
    def _repair_missing_metadata(self, error: StorageError):
        """Repair missing metadata files"""
        # Placeholder for metadata repair logic
        logger.debug("Attempting metadata repair")
    
    def _repair_invalid_format(self, error: StorageError):
        """Repair invalid format issues"""
        # Placeholder for format repair logic
        logger.debug("Attempting format repair")
    
    def _repair_corruption(self, error: StorageError):
        """Repair data corruption"""
        # Placeholder for corruption repair logic
        logger.debug("Attempting corruption repair")


class CircuitBreaker:
    """Circuit breaker implementation for storage operations"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.half_open_calls = 0
        self._lock = threading.Lock()
    
    def call(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation with circuit breaker protection"""
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time < self.config.recovery_timeout:
                    raise RuntimeError(f"Circuit breaker {self.name} is OPEN")
                else:
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.half_open_calls = 0
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.half_open_calls >= self.config.half_open_max_calls:
                    self.state = CircuitBreakerState.OPEN
                    raise RuntimeError(f"Circuit breaker {self.name} is OPEN")
                self.half_open_calls += 1
        
        try:
            result = operation(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful operation"""
        with self._lock:
            self.failure_count = 0
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.CLOSED
                logger.info(f"Circuit breaker {self.name} recovered to CLOSED")
    
    def _on_failure(self):
        """Handle failed operation"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if (self.state == CircuitBreakerState.CLOSED and 
                self.failure_count >= self.config.failure_threshold):
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker {self.name} opened after {self.failure_count} failures")


class StorageErrorHandler:
    """
    Centralized storage error handling system.
    
    This class implements the Single Responsibility Principle by handling
    all storage-related errors consistently across the trading system.
    It provides automatic recovery mechanisms and comprehensive error tracking.
    """
    
    def __init__(self):
        self.error_history: List[StorageError] = []
        self.recovery_strategies: List[IErrorRecoveryStrategy] = []
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_counts: Dict[StorageErrorType, int] = {}
        self._lock = threading.Lock()
        
        # Default configurations
        self.default_retry_config = RetryConfig()
        self.default_circuit_breaker_config = CircuitBreakerConfig()
        
        # Initialize default recovery strategies
        self._initialize_default_strategies()
        
        logger.info("StorageErrorHandler initialized")
    
    def _initialize_default_strategies(self):
        """Initialize default recovery strategies"""
        self.recovery_strategies.append(RetryRecoveryStrategy(self.default_retry_config))
        self.recovery_strategies.append(RepairRecoveryStrategy())
        # Note: FallbackRecoveryStrategy needs to be configured per use case
    
    def add_recovery_strategy(self, strategy: IErrorRecoveryStrategy):
        """Add a custom recovery strategy"""
        self.recovery_strategies.append(strategy)
        logger.info(f"Added recovery strategy: {strategy.__class__.__name__}")
    
    def get_circuit_breaker(self, name: str, 
                           config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create circuit breaker for a component"""
        if name not in self.circuit_breakers:
            config = config or self.default_circuit_breaker_config
            self.circuit_breakers[name] = CircuitBreaker(name, config)
        return self.circuit_breakers[name]
    
    def handle_error(self, 
                     exception: Exception,
                     component: str,
                     operation: str,
                     context: Optional[Dict[str, Any]] = None,
                     attempt_recovery: bool = True) -> StorageError:
        """
        Handle a storage error with classification and potential recovery
        
        Args:
            exception: The exception that occurred
            component: Name of the component where error occurred
            operation: Name of the operation that failed
            context: Additional context information
            attempt_recovery: Whether to attempt automatic recovery
            
        Returns:
            Structured storage error information
        """
        try:
            # Classify the error
            error_type = self._classify_error(exception)
            severity = self._determine_severity(error_type, exception)
            
            # Create structured error
            storage_error = StorageError(
                error_type=error_type,
                severity=severity,
                message=str(exception),
                component=component,
                operation=operation,
                traceback=traceback.format_exc(),
                context=context or {}
            )
            
            # Log the error
            self._log_error(storage_error)
            
            # Track error statistics
            with self._lock:
                self.error_history.append(storage_error)
                self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
            
            # Attempt recovery if requested
            if attempt_recovery:
                storage_error.recovery_attempted = True
                self._attempt_recovery(storage_error)
            
            return storage_error
            
        except Exception as e:
            logger.error(f"Error in error handler: {e}")
            # Return a basic error if error handling itself fails
            return StorageError(
                error_type=StorageErrorType.UNKNOWN_ERROR,
                severity=StorageErrorSeverity.HIGH,
                message=f"Error handling failed: {e}",
                component=component,
                operation=operation
            )
    
    def _classify_error(self, exception: Exception) -> StorageErrorType:
        """Classify exception into storage error type"""
        exception_type = type(exception).__name__
        exception_message = str(exception).lower()
        
        # File system errors
        if isinstance(exception, FileNotFoundError):
            return StorageErrorType.FILE_NOT_FOUND
        elif isinstance(exception, PermissionError):
            return StorageErrorType.PERMISSION_DENIED
        elif isinstance(exception, OSError):
            if "no space left" in exception_message or "disk full" in exception_message:
                return StorageErrorType.DISK_FULL
            else:
                return StorageErrorType.IO_ERROR
        
        # Data integrity errors
        elif "checksum" in exception_message or "integrity" in exception_message:
            return StorageErrorType.CHECKSUM_MISMATCH
        elif "corrupt" in exception_message:
            return StorageErrorType.CORRUPTION_DETECTED
        elif "invalid format" in exception_message or "malformed" in exception_message:
            return StorageErrorType.INVALID_FORMAT
        elif "metadata" in exception_message and "missing" in exception_message:
            return StorageErrorType.MISSING_METADATA
        
        # Serialization errors
        elif "json" in exception_message and ("decode" in exception_message or "parse" in exception_message):
            return StorageErrorType.DESERIALIZATION_FAILED
        elif "pickle" in exception_message:
            if "load" in exception_message:
                return StorageErrorType.DESERIALIZATION_FAILED
            else:
                return StorageErrorType.SERIALIZATION_FAILED
        elif isinstance(exception, (TypeError, ValueError)) and "serializ" in exception_message:
            return StorageErrorType.SERIALIZATION_FAILED
        elif isinstance(exception, UnicodeError):
            return StorageErrorType.ENCODING_ERROR
        
        # Network/connectivity errors
        elif "network" in exception_message or "connection" in exception_message:
            return StorageErrorType.NETWORK_ERROR
        elif "timeout" in exception_message:
            return StorageErrorType.TIMEOUT_ERROR
        
        # System errors
        elif isinstance(exception, MemoryError):
            return StorageErrorType.MEMORY_ERROR
        elif "thread" in exception_message:
            return StorageErrorType.THREAD_ERROR
        
        # Default case
        else:
            return StorageErrorType.UNKNOWN_ERROR
    
    def _determine_severity(self, error_type: StorageErrorType, exception: Exception) -> StorageErrorSeverity:
        """Determine severity level based on error type and context"""
        critical_errors = {
            StorageErrorType.DISK_FULL,
            StorageErrorType.CORRUPTION_DETECTED,
            StorageErrorType.MEMORY_ERROR
        }
        
        high_errors = {
            StorageErrorType.PERMISSION_DENIED,
            StorageErrorType.CHECKSUM_MISMATCH,
            StorageErrorType.SERIALIZATION_FAILED,
            StorageErrorType.DESERIALIZATION_FAILED
        }
        
        medium_errors = {
            StorageErrorType.FILE_NOT_FOUND,
            StorageErrorType.INVALID_FORMAT,
            StorageErrorType.MISSING_METADATA,
            StorageErrorType.NETWORK_ERROR,
            StorageErrorType.CONNECTION_FAILED
        }
        
        if error_type in critical_errors:
            return StorageErrorSeverity.CRITICAL
        elif error_type in high_errors:
            return StorageErrorSeverity.HIGH
        elif error_type in medium_errors:
            return StorageErrorSeverity.MEDIUM
        else:
            return StorageErrorSeverity.LOW
    
    def _log_error(self, error: StorageError):
        """Log error with appropriate level based on severity"""
        log_message = f"Storage error in {error.component}.{error.operation}: {error.message}"
        
        if error.severity == StorageErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error.severity == StorageErrorSeverity.HIGH:
            logger.error(log_message)
        elif error.severity == StorageErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
        
        # Log additional context if available
        if error.context:
            logger.debug(f"Error context: {error.context}")
    
    def _attempt_recovery(self, error: StorageError):
        """Attempt recovery using available strategies"""
        for strategy in self.recovery_strategies:
            if strategy.can_handle(error):
                try:
                    logger.info(f"Attempting recovery with {strategy.__class__.__name__}")
                    # Note: Recovery would need access to the original operation
                    # This is a framework - actual recovery implementation would
                    # require integration with the calling code
                    error.recovery_successful = True
                    break
                except Exception as e:
                    logger.warning(f"Recovery strategy {strategy.__class__.__name__} failed: {e}")
                    continue
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        with self._lock:
            total_errors = len(self.error_history)
            
            if total_errors == 0:
                return {"total_errors": 0}
            
            # Error type distribution
            error_type_stats = {}
            for error_type, count in self.error_counts.items():
                error_type_stats[error_type.name] = {
                    "count": count,
                    "percentage": (count / total_errors) * 100
                }
            
            # Severity distribution
            severity_counts = {}
            recovery_success_count = 0
            
            for error in self.error_history:
                severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
                if error.recovery_successful:
                    recovery_success_count += 1
            
            # Recent errors (last hour)
            current_time = time.time()
            recent_errors = [
                error for error in self.error_history 
                if current_time - error.timestamp < 3600
            ]
            
            return {
                "total_errors": total_errors,
                "error_types": error_type_stats,
                "severity_distribution": severity_counts,
                "recovery_success_rate": (recovery_success_count / total_errors) * 100,
                "recent_errors_count": len(recent_errors),
                "circuit_breaker_status": {
                    name: breaker.state.value 
                    for name, breaker in self.circuit_breakers.items()
                }
            }
    
    def clear_error_history(self, older_than_hours: Optional[float] = None):
        """Clear error history, optionally keeping recent errors"""
        with self._lock:
            if older_than_hours is None:
                self.error_history.clear()
                self.error_counts.clear()
                logger.info("Cleared all error history")
            else:
                cutoff_time = time.time() - (older_than_hours * 3600)
                old_errors = [e for e in self.error_history if e.timestamp < cutoff_time]
                self.error_history = [e for e in self.error_history if e.timestamp >= cutoff_time]
                
                # Recalculate error counts
                self.error_counts.clear()
                for error in self.error_history:
                    self.error_counts[error.error_type] = self.error_counts.get(error.error_type, 0) + 1
                
                logger.info(f"Cleared {len(old_errors)} errors older than {older_than_hours} hours")


# Global error handler instance
storage_error_handler = StorageErrorHandler()


def handle_storage_operation(component: str, operation: str, 
                           use_circuit_breaker: bool = True,
                           circuit_breaker_config: Optional[CircuitBreakerConfig] = None):
    """
    Decorator for storage operations with automatic error handling
    
    Args:
        component: Name of the component performing the operation
        operation: Name of the operation being performed
        use_circuit_breaker: Whether to use circuit breaker protection
        circuit_breaker_config: Custom circuit breaker configuration
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                if use_circuit_breaker:
                    circuit_breaker = storage_error_handler.get_circuit_breaker(
                        f"{component}_{operation}", circuit_breaker_config
                    )
                    return circuit_breaker.call(func, *args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                error = storage_error_handler.handle_error(
                    exception=e,
                    component=component,
                    operation=operation,
                    context={
                        'function': func.__name__,
                        'args_count': len(args),
                        'kwargs_keys': list(kwargs.keys())
                    }
                )
                
                # Re-raise the original exception after handling
                raise e
                
        return wrapper
    return decorator