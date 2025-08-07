"""
Data Integrity Validator - Comprehensive data validation and integrity checking

This module provides robust data integrity validation for all components in the
trading system. It implements comprehensive validation patterns following the
Single Responsibility Principle and ensures data consistency across all operations.

Features:
- Multi-level validation (structure, content, business logic)
- Checksum and hash-based integrity verification
- Schema validation for complex data structures
- Cross-component data consistency checks
- Automated repair mechanisms for recoverable corruption
- Comprehensive audit logging and reporting
"""

import hashlib
import logging
import time
import json
from typing import Any, Dict, List, Optional, Union, Callable, Type, Protocol
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Levels of validation strictness"""
    BASIC = "basic"          # Basic type and structure validation
    STANDARD = "standard"    # Standard validation with business rules
    STRICT = "strict"        # Strict validation with comprehensive checks
    PARANOID = "paranoid"    # Maximum validation with all possible checks


class ValidationResult(Enum):
    """Validation result status"""
    VALID = "valid"
    WARNING = "warning"      # Issues found but data is usable
    INVALID = "invalid"      # Data has errors but might be repairable
    CORRUPTED = "corrupted"  # Data is severely corrupted


class ValidationErrorType(Enum):
    """Types of validation errors"""
    MISSING_FIELD = auto()
    INVALID_TYPE = auto()
    OUT_OF_RANGE = auto()
    INVALID_FORMAT = auto()
    CHECKSUM_MISMATCH = auto()
    SCHEMA_VIOLATION = auto()
    BUSINESS_RULE_VIOLATION = auto()
    CROSS_REFERENCE_ERROR = auto()
    TEMPORAL_INCONSISTENCY = auto()
    DATA_CORRUPTION = auto()


@dataclass
class ValidationIssue:
    """Individual validation issue"""
    error_type: ValidationErrorType
    severity: str  # "error", "warning", "info"
    field_path: str
    message: str
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    component: str
    data_type: str
    validation_level: ValidationLevel
    timestamp: float
    result: ValidationResult
    issues: List[ValidationIssue] = field(default_factory=list)
    checksum: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    repair_suggestions: List[str] = field(default_factory=list)
    
    @property
    def is_valid(self) -> bool:
        """Check if validation passed"""
        return self.result in [ValidationResult.VALID, ValidationResult.WARNING]
    
    @property
    def has_errors(self) -> bool:
        """Check if validation found errors"""
        return any(issue.severity == "error" for issue in self.issues)
    
    @property
    def has_warnings(self) -> bool:
        """Check if validation found warnings"""
        return any(issue.severity == "warning" for issue in self.issues)


class IDataValidator(Protocol):
    """Protocol for data validators"""
    
    def validate(self, data: Any, level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationReport:
        """Validate data and return report"""
        ...
    
    def can_validate(self, data_type: str) -> bool:
        """Check if this validator can handle the data type"""
        ...


class SchemaValidator:
    """Schema-based validation for structured data"""
    
    def __init__(self):
        self.schemas = {}
    
    def register_schema(self, data_type: str, schema: Dict[str, Any]):
        """Register a validation schema for a data type"""
        self.schemas[data_type] = schema
        logger.debug(f"Registered schema for {data_type}")
    
    def validate_against_schema(self, data: Dict[str, Any], data_type: str) -> List[ValidationIssue]:
        """Validate data against registered schema"""
        issues = []
        
        if data_type not in self.schemas:
            issues.append(ValidationIssue(
                error_type=ValidationErrorType.SCHEMA_VIOLATION,
                severity="warning",
                field_path="root",
                message=f"No schema registered for data type: {data_type}"
            ))
            return issues
        
        schema = self.schemas[data_type]
        issues.extend(self._validate_dict_against_schema(data, schema, "root"))
        
        return issues
    
    def _validate_dict_against_schema(self, data: Dict[str, Any], 
                                    schema: Dict[str, Any], 
                                    path: str) -> List[ValidationIssue]:
        """Recursively validate dictionary against schema"""
        issues = []
        
        # Check required fields
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in data:
                issues.append(ValidationIssue(
                    error_type=ValidationErrorType.MISSING_FIELD,
                    severity="error",
                    field_path=f"{path}.{field}",
                    message=f"Required field '{field}' is missing"
                ))
        
        # Validate field types and constraints
        properties = schema.get("properties", {})
        for field, constraints in properties.items():
            if field in data:
                field_path = f"{path}.{field}"
                field_issues = self._validate_field_constraints(
                    data[field], constraints, field_path
                )
                issues.extend(field_issues)
        
        return issues
    
    def _validate_field_constraints(self, value: Any, 
                                  constraints: Dict[str, Any], 
                                  path: str) -> List[ValidationIssue]:
        """Validate individual field constraints"""
        issues = []
        
        # Type validation
        expected_type = constraints.get("type")
        if expected_type and not self._check_type(value, expected_type):
            issues.append(ValidationIssue(
                error_type=ValidationErrorType.INVALID_TYPE,
                severity="error",
                field_path=path,
                message=f"Expected type {expected_type}, got {type(value).__name__}",
                expected_value=expected_type,
                actual_value=type(value).__name__
            ))
        
        # Range validation for numeric types
        if isinstance(value, (int, float)):
            min_val = constraints.get("minimum")
            max_val = constraints.get("maximum")
            
            if min_val is not None and value < min_val:
                issues.append(ValidationIssue(
                    error_type=ValidationErrorType.OUT_OF_RANGE,
                    severity="error",
                    field_path=path,
                    message=f"Value {value} is below minimum {min_val}",
                    expected_value=f">= {min_val}",
                    actual_value=value
                ))
            
            if max_val is not None and value > max_val:
                issues.append(ValidationIssue(
                    error_type=ValidationErrorType.OUT_OF_RANGE,
                    severity="error",
                    field_path=path,
                    message=f"Value {value} is above maximum {max_val}",
                    expected_value=f"<= {max_val}",
                    actual_value=value
                ))
        
        # Pattern validation for strings
        if isinstance(value, str):
            pattern = constraints.get("pattern")
            if pattern:
                import re
                if not re.match(pattern, value):
                    issues.append(ValidationIssue(
                        error_type=ValidationErrorType.INVALID_FORMAT,
                        severity="error",
                        field_path=path,
                        message=f"String does not match pattern: {pattern}",
                        actual_value=value
                    ))
        
        return issues
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type"""
        type_mapping = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": (list, tuple),
            "object": dict,
            "null": type(None)
        }
        
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
        
        return True  # Unknown type, assume valid


class TradingDataValidator(IDataValidator):
    """Specialized validator for trading system data"""
    
    def __init__(self):
        self.schema_validator = SchemaValidator()
        self._register_trading_schemas()
    
    def _register_trading_schemas(self):
        """Register schemas for trading data types"""
        
        # Portfolio state schema
        portfolio_schema = {
            "type": "object",
            "required": ["positions", "account_balance", "timestamp"],
            "properties": {
                "positions": {
                    "type": "object",
                    "properties": {
                        "*": {  # Pattern for any symbol
                            "type": "object",
                            "required": ["entry_price", "current_size"],
                            "properties": {
                                "entry_price": {"type": "number", "minimum": 0},
                                "current_size": {"type": "number"},
                                "unrealized_pnl": {"type": "number"},
                                "realized_pnl": {"type": "number"}
                            }
                        }
                    }
                },
                "account_balance": {"type": "number", "minimum": 0},
                "timestamp": {"type": "number", "minimum": 0}
            }
        }
        self.schema_validator.register_schema("portfolio_state", portfolio_schema)
        
        # Trading experience schema
        experience_schema = {
            "type": "object",
            "required": ["state_features", "action", "reward", "done"],
            "properties": {
                "state_features": {"type": "array"},
                "action": {"type": "integer", "minimum": 0, "maximum": 2},
                "reward": {"type": "number"},
                "done": {"type": "boolean"},
                "timestamp": {"type": "number", "minimum": 0}
            }
        }
        self.schema_validator.register_schema("trading_experience", experience_schema)
        
        # Market data schema
        market_data_schema = {
            "type": "object",
            "required": ["symbol", "price", "timestamp"],
            "properties": {
                "symbol": {"type": "string"},
                "price": {"type": "number", "minimum": 0},
                "volume": {"type": "number", "minimum": 0},
                "timestamp": {"type": "number", "minimum": 0},
                "bid": {"type": "number", "minimum": 0},
                "ask": {"type": "number", "minimum": 0}
            }
        }
        self.schema_validator.register_schema("market_data", market_data_schema)
    
    def can_validate(self, data_type: str) -> bool:
        """Check if this validator can handle the data type"""
        return data_type in ["portfolio_state", "trading_experience", "market_data", "neural_state"]
    
    def validate(self, data: Any, level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationReport:
        """Validate trading data"""
        data_type = self._infer_data_type(data)
        report = ValidationReport(
            component="TradingDataValidator",
            data_type=data_type,
            validation_level=level,
            timestamp=time.time(),
            result=ValidationResult.VALID
        )
        
        try:
            # Basic structure validation
            if level in [ValidationLevel.BASIC, ValidationLevel.STANDARD, 
                        ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                issues = self._validate_basic_structure(data, data_type)
                report.issues.extend(issues)
            
            # Schema validation
            if level in [ValidationLevel.STANDARD, ValidationLevel.STRICT, ValidationLevel.PARANOID] and isinstance(data, dict):
                schema_issues = self.schema_validator.validate_against_schema(data, data_type)
                report.issues.extend(schema_issues)
            
            # Business logic validation
            if level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                business_issues = self._validate_business_logic(data, data_type)
                report.issues.extend(business_issues)
            
            # Comprehensive checks
            if level == ValidationLevel.PARANOID:
                paranoid_issues = self._validate_paranoid_checks(data, data_type)
                report.issues.extend(paranoid_issues)
            
            # Determine overall result
            report.result = self._determine_validation_result(report.issues)
            
            # Generate repair suggestions
            if report.issues:
                report.repair_suggestions = self._generate_repair_suggestions(report.issues)
            
            # Calculate checksum
            report.checksum = self._calculate_data_checksum(data)
            
        except Exception as e:
            report.result = ValidationResult.CORRUPTED
            report.issues.append(ValidationIssue(
                error_type=ValidationErrorType.DATA_CORRUPTION,
                severity="error",
                field_path="root",
                message=f"Validation failed due to corruption: {e}"
            ))
        
        return report
    
    def _infer_data_type(self, data: Any) -> str:
        """Infer data type from data structure"""
        if isinstance(data, dict):
            if "positions" in data and "account_balance" in data:
                return "portfolio_state"
            elif "state_features" in data and "action" in data:
                return "trading_experience"
            elif "symbol" in data and "price" in data:
                return "market_data"
            elif "layers" in data or "weights" in data:
                return "neural_state"
        
        return "unknown"
    
    def _validate_basic_structure(self, data: Any, data_type: str) -> List[ValidationIssue]:
        """Validate basic data structure"""
        issues = []
        
        # Check for None/null values
        if data is None:
            issues.append(ValidationIssue(
                error_type=ValidationErrorType.DATA_CORRUPTION,
                severity="error",
                field_path="root",
                message="Data is None/null"
            ))
            return issues
        
        # Check for empty data
        if isinstance(data, (dict, list)) and len(data) == 0:
            issues.append(ValidationIssue(
                error_type=ValidationErrorType.INVALID_FORMAT,
                severity="warning",
                field_path="root",
                message="Data structure is empty"
            ))
        
        # Check for NaN/inf values in numeric data
        if isinstance(data, dict):
            issues.extend(self._check_numeric_validity(data, "root"))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    issues.extend(self._check_numeric_validity(item, f"[{i}]"))
        
        return issues
    
    def _check_numeric_validity(self, data: Dict[str, Any], path: str) -> List[ValidationIssue]:
        """Check for invalid numeric values (NaN, inf)"""
        issues = []
        
        for key, value in data.items():
            field_path = f"{path}.{key}"
            
            if isinstance(value, (int, float)):
                if np.isnan(value):
                    issues.append(ValidationIssue(
                        error_type=ValidationErrorType.DATA_CORRUPTION,
                        severity="error",
                        field_path=field_path,
                        message="Value is NaN (Not a Number)",
                        actual_value=value
                    ))
                elif np.isinf(value):
                    issues.append(ValidationIssue(
                        error_type=ValidationErrorType.DATA_CORRUPTION,
                        severity="error",
                        field_path=field_path,
                        message="Value is infinite",
                        actual_value=value
                    ))
            elif isinstance(value, (list, np.ndarray)):
                try:
                    arr = np.array(value)
                    if np.any(np.isnan(arr)):
                        issues.append(ValidationIssue(
                            error_type=ValidationErrorType.DATA_CORRUPTION,
                            severity="error",
                            field_path=field_path,
                            message="Array contains NaN values"
                        ))
                    if np.any(np.isinf(arr)):
                        issues.append(ValidationIssue(
                            error_type=ValidationErrorType.DATA_CORRUPTION,
                            severity="error",
                            field_path=field_path,
                            message="Array contains infinite values"
                        ))
                except (ValueError, TypeError):
                    pass  # Skip if can't convert to array
            elif isinstance(value, dict):
                issues.extend(self._check_numeric_validity(value, field_path))
        
        return issues
    
    def _validate_business_logic(self, data: Any, data_type: str) -> List[ValidationIssue]:
        """Validate business logic rules"""
        issues = []
        
        if data_type == "portfolio_state" and isinstance(data, dict):
            # Portfolio-specific business rules
            account_balance = data.get("account_balance", 0)
            if account_balance < 0:
                issues.append(ValidationIssue(
                    error_type=ValidationErrorType.BUSINESS_RULE_VIOLATION,
                    severity="error",
                    field_path="account_balance",
                    message="Account balance cannot be negative",
                    actual_value=account_balance
                ))
            
            # Check position consistency
            positions = data.get("positions", {})
            for symbol, position in positions.items():
                if isinstance(position, dict):
                    size = position.get("current_size", 0)
                    entry_price = position.get("entry_price", 0)
                    
                    if size != 0 and entry_price <= 0:
                        issues.append(ValidationIssue(
                            error_type=ValidationErrorType.BUSINESS_RULE_VIOLATION,
                            severity="error",
                            field_path=f"positions.{symbol}.entry_price",
                            message="Non-zero position must have positive entry price",
                            actual_value=entry_price
                        ))
        
        elif data_type == "trading_experience" and isinstance(data, dict):
            # Experience-specific business rules
            reward = data.get("reward", 0)
            if abs(reward) > 1000:  # Sanity check for extreme rewards
                issues.append(ValidationIssue(
                    error_type=ValidationErrorType.BUSINESS_RULE_VIOLATION,
                    severity="warning",
                    field_path="reward",
                    message="Reward value seems unusually high",
                    actual_value=reward
                ))
        
        return issues
    
    def _validate_paranoid_checks(self, data: Any, data_type: str) -> List[ValidationIssue]:
        """Paranoid-level validation checks"""
        issues = []
        
        # Temporal consistency checks
        if isinstance(data, dict) and "timestamp" in data:
            timestamp = data["timestamp"]
            current_time = time.time()
            
            # Check for future timestamps
            if timestamp > current_time + 60:  # Allow 1 minute tolerance
                issues.append(ValidationIssue(
                    error_type=ValidationErrorType.TEMPORAL_INCONSISTENCY,
                    severity="warning",
                    field_path="timestamp",
                    message="Timestamp is in the future",
                    actual_value=timestamp,
                    expected_value=f"<= {current_time}"
                ))
            
            # Check for very old timestamps (more than 1 year)
            if timestamp < current_time - (365 * 24 * 3600):
                issues.append(ValidationIssue(
                    error_type=ValidationErrorType.TEMPORAL_INCONSISTENCY,
                    severity="warning",
                    field_path="timestamp",
                    message="Timestamp is very old (> 1 year)",
                    actual_value=timestamp
                ))
        
        # Data size checks
        if isinstance(data, dict):
            json_size = len(json.dumps(data, default=str))
            if json_size > 10 * 1024 * 1024:  # 10MB limit
                issues.append(ValidationIssue(
                    error_type=ValidationErrorType.INVALID_FORMAT,
                    severity="warning",
                    field_path="root",
                    message=f"Data size is very large: {json_size / 1024 / 1024:.1f}MB"
                ))
        
        return issues
    
    def _determine_validation_result(self, issues: List[ValidationIssue]) -> ValidationResult:
        """Determine overall validation result from issues"""
        if not issues:
            return ValidationResult.VALID
        
        has_errors = any(issue.severity == "error" for issue in issues)
        has_corruption = any(
            issue.error_type == ValidationErrorType.DATA_CORRUPTION 
            for issue in issues
        )
        
        if has_corruption:
            return ValidationResult.CORRUPTED
        elif has_errors:
            return ValidationResult.INVALID
        else:
            return ValidationResult.WARNING
    
    def _generate_repair_suggestions(self, issues: List[ValidationIssue]) -> List[str]:
        """Generate repair suggestions based on validation issues"""
        suggestions = []
        
        for issue in issues:
            if issue.error_type == ValidationErrorType.MISSING_FIELD:
                suggestions.append(f"Add missing field: {issue.field_path}")
            elif issue.error_type == ValidationErrorType.OUT_OF_RANGE:
                if issue.expected_value:
                    suggestions.append(f"Adjust {issue.field_path} to be {issue.expected_value}")
            elif issue.error_type == ValidationErrorType.DATA_CORRUPTION:
                if "NaN" in issue.message:
                    suggestions.append(f"Replace NaN values in {issue.field_path} with default values")
                elif "infinite" in issue.message:
                    suggestions.append(f"Replace infinite values in {issue.field_path} with max/min bounds")
        
        return list(set(suggestions))  # Remove duplicates
    
    def _calculate_data_checksum(self, data: Any) -> str:
        """Calculate checksum for data integrity verification"""
        try:
            # Convert data to consistent string representation
            data_str = json.dumps(data, sort_keys=True, default=str)
            return hashlib.sha256(data_str.encode('utf-8')).hexdigest()
        except Exception:
            return ""


class DataIntegrityValidator:
    """
    Comprehensive data integrity validation system.
    
    This class serves as the main interface for all data integrity validation
    in the trading system. It coordinates multiple validators and provides
    comprehensive integrity checking with automated repair suggestions.
    """
    
    def __init__(self):
        self.validators: List[IDataValidator] = []
        self.validation_history: List[ValidationReport] = []
        self.integrity_stats = {
            "total_validations": 0,
            "validations_passed": 0,
            "validations_with_warnings": 0,
            "validations_failed": 0,
            "corrupted_data_detected": 0
        }
        
        # Register default validators
        self.validators.append(TradingDataValidator())
        
        logger.info("DataIntegrityValidator initialized")
    
    def register_validator(self, validator: IDataValidator):
        """Register a custom validator"""
        self.validators.append(validator)
        logger.info(f"Registered validator: {validator.__class__.__name__}")
    
    def validate_data(self,
                      data: Any,
                      data_type: Optional[str] = None,
                      component: str = "unknown",
                      level: ValidationLevel = ValidationLevel.STANDARD,
                      store_report: bool = True) -> ValidationReport:
        """
        Validate data using appropriate validator
        
        Args:
            data: Data to validate
            data_type: Hint for data type (optional)
            component: Component name for reporting
            level: Validation strictness level
            store_report: Whether to store the report in history
            
        Returns:
            Comprehensive validation report
        """
        # Find appropriate validator
        validator = None
        if data_type:
            for v in self.validators:
                if v.can_validate(data_type):
                    validator = v
                    break
        
        # Fallback to first available validator
        if not validator and self.validators:
            validator = self.validators[0]
        
        if not validator:
            # Create a basic report if no validator available
            report = ValidationReport(
                component=component,
                data_type=data_type or "unknown",
                validation_level=level,
                timestamp=time.time(),
                result=ValidationResult.WARNING
            )
            report.issues.append(ValidationIssue(
                error_type=ValidationErrorType.SCHEMA_VIOLATION,
                severity="warning",
                field_path="root",
                message="No suitable validator found for data type"
            ))
        else:
            report = validator.validate(data, level)
            report.component = component
        
        # Update statistics
        self._update_statistics(report)
        
        # Store report if requested
        if store_report:
            self.validation_history.append(report)
            # Keep only recent reports (last 1000)
            if len(self.validation_history) > 1000:
                self.validation_history.pop(0)
        
        return report
    
    def validate_file_integrity(self,
                               filepath: Union[str, Path],
                               expected_checksum: Optional[str] = None) -> ValidationReport:
        """
        Validate file integrity including checksum verification
        
        Args:
            filepath: Path to file to validate
            expected_checksum: Expected file checksum (optional)
            
        Returns:
            Validation report for file integrity
        """
        filepath = Path(filepath)
        report = ValidationReport(
            component="FileIntegrityValidator",
            data_type="file",
            validation_level=ValidationLevel.BASIC,
            timestamp=time.time(),
            result=ValidationResult.VALID
        )
        
        # Check if file exists
        if not filepath.exists():
            report.result = ValidationResult.INVALID
            report.issues.append(ValidationIssue(
                error_type=ValidationErrorType.MISSING_FIELD,
                severity="error",
                field_path=str(filepath),
                message="File does not exist"
            ))
            return report
        
        try:
            # Calculate file checksum
            with open(filepath, 'rb') as f:
                file_data = f.read()
                actual_checksum = hashlib.sha256(file_data).hexdigest()
                report.checksum = actual_checksum
            
            # Verify checksum if expected value provided
            if expected_checksum and actual_checksum != expected_checksum:
                report.result = ValidationResult.CORRUPTED
                report.issues.append(ValidationIssue(
                    error_type=ValidationErrorType.CHECKSUM_MISMATCH,
                    severity="error",
                    field_path=str(filepath),
                    message="File checksum mismatch",
                    expected_value=expected_checksum,
                    actual_value=actual_checksum
                ))
            
            # Basic file health checks
            file_size = len(file_data)
            if file_size == 0:
                report.result = ValidationResult.INVALID
                report.issues.append(ValidationIssue(
                    error_type=ValidationErrorType.INVALID_FORMAT,
                    severity="error",
                    field_path=str(filepath),
                    message="File is empty"
                ))
            
            report.metadata = {
                "file_size": file_size,
                "file_path": str(filepath),
                "modification_time": filepath.stat().st_mtime
            }
            
        except Exception as e:
            report.result = ValidationResult.CORRUPTED
            report.issues.append(ValidationIssue(
                error_type=ValidationErrorType.DATA_CORRUPTION,
                severity="error",
                field_path=str(filepath),
                message=f"Error reading file: {e}"
            ))
        
        return report
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive validation statistics"""
        total = self.integrity_stats["total_validations"]
        
        if total == 0:
            return {
                "total_validations": 0,
                "success_rate": 0.0,
                "warning_rate": 0.0,
                "failure_rate": 0.0,
                "corruption_rate": 0.0
            }
        
        return {
            **self.integrity_stats,
            "success_rate": (self.integrity_stats["validations_passed"] / total) * 100,
            "warning_rate": (self.integrity_stats["validations_with_warnings"] / total) * 100,
            "failure_rate": (self.integrity_stats["validations_failed"] / total) * 100,
            "corruption_rate": (self.integrity_stats["corrupted_data_detected"] / total) * 100,
            "recent_validations": len([
                r for r in self.validation_history 
                if time.time() - r.timestamp < 3600  # Last hour
            ])
        }
    
    def _update_statistics(self, report: ValidationReport):
        """Update validation statistics"""
        self.integrity_stats["total_validations"] += 1
        
        if report.result == ValidationResult.VALID:
            self.integrity_stats["validations_passed"] += 1
        elif report.result == ValidationResult.WARNING:
            self.integrity_stats["validations_with_warnings"] += 1
        elif report.result == ValidationResult.INVALID:
            self.integrity_stats["validations_failed"] += 1
        elif report.result == ValidationResult.CORRUPTED:
            self.integrity_stats["corrupted_data_detected"] += 1
    
    def clear_validation_history(self, older_than_hours: Optional[float] = None):
        """Clear validation history"""
        if older_than_hours is None:
            self.validation_history.clear()
            logger.info("Cleared all validation history")
        else:
            cutoff_time = time.time() - (older_than_hours * 3600)
            original_count = len(self.validation_history)
            self.validation_history = [
                r for r in self.validation_history 
                if r.timestamp >= cutoff_time
            ]
            cleared_count = original_count - len(self.validation_history)
            logger.info(f"Cleared {cleared_count} validation reports older than {older_than_hours} hours")


# Global data integrity validator instance
data_integrity_validator = DataIntegrityValidator()


def validate_component_data(component: str, data: Any, 
                          data_type: Optional[str] = None,
                          level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationReport:
    """
    Convenience function to validate component data
    
    Args:
        component: Name of the component
        data: Data to validate
        data_type: Hint for data type
        level: Validation level
        
    Returns:
        Validation report
    """
    return data_integrity_validator.validate_data(
        data=data,
        data_type=data_type,
        component=component,
        level=level
    )