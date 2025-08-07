"""
Unified Data Serialization Framework

This module provides a consistent, robust serialization system for all components
in the trading system. It handles complex data types, ensures data integrity,
and provides standardized serialization patterns.

Features:
- Unified serialization interface for all components
- Support for complex data types (numpy arrays, PyTorch tensors, datetime objects)
- Data integrity validation and checksums
- Compression support for large datasets
- Error handling and recovery mechanisms
- Type-safe serialization with proper validation
"""

import json
import pickle
import gzip
import hashlib
import logging
import time
from typing import Any, Dict, List, Optional, Union, Protocol, Type, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


class SerializationFormat(Enum):
    """Supported serialization formats"""
    JSON = "json"
    PICKLE = "pickle"
    COMPRESSED_JSON = "json.gz"
    COMPRESSED_PICKLE = "pickle.gz"


class DataType(Enum):
    """Data type categories for serialization strategy"""
    PRIMITIVE = "primitive"  # Basic types: int, float, str, bool
    COLLECTION = "collection"  # Lists, dicts, sets
    NUMPY_ARRAY = "numpy_array"
    PYTORCH_TENSOR = "pytorch_tensor"
    DATETIME = "datetime"
    DEQUE = "deque"
    MAPPING = "mapping"
    COMPLEX_OBJECT = "complex_object"


@dataclass
class SerializationMetadata:
    """Metadata for serialized data"""
    format: SerializationFormat
    data_type: DataType
    checksum: str
    timestamp: float
    version: str
    size_bytes: int
    compression_ratio: Optional[float] = None


class SerializationError(Exception):
    """Base exception for serialization errors"""
    pass


class DeserializationError(Exception):
    """Base exception for deserialization errors"""
    pass


class DataIntegrityError(Exception):
    """Exception for data integrity validation failures"""
    pass


class IDataSerializer(Protocol):
    """Protocol for data serializers"""
    
    def serialize(self, data: Any) -> bytes:
        """Serialize data to bytes"""
        ...
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to data"""
        ...
    
    def get_metadata(self, data: Any) -> SerializationMetadata:
        """Get metadata for data"""
        ...


class JSONSerializer:
    """JSON-based serializer with complex type support"""
    
    def __init__(self, indent: Optional[int] = None):
        self.indent = indent
    
    def serialize(self, data: Any) -> bytes:
        """Serialize data to JSON bytes"""
        try:
            json_str = json.dumps(data, indent=self.indent, default=self._json_encoder)
            return json_str.encode('utf-8')
        except Exception as e:
            raise SerializationError(f"JSON serialization failed: {e}")
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize JSON bytes to data"""
        try:
            json_str = data.decode('utf-8')
            return json.loads(json_str, object_hook=self._json_decoder)
        except Exception as e:
            raise DeserializationError(f"JSON deserialization failed: {e}")
    
    def get_metadata(self, data: Any) -> SerializationMetadata:
        """Get metadata for JSON serialization"""
        serialized = self.serialize(data)
        return SerializationMetadata(
            format=SerializationFormat.JSON,
            data_type=self._detect_data_type(data),
            checksum=self._calculate_checksum(serialized),
            timestamp=time.time(),
            version="1.0",
            size_bytes=len(serialized)
        )
    
    def _json_encoder(self, obj: Any) -> Any:
        """Custom JSON encoder for complex objects"""
        try:
            # Handle numpy types
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return {
                    '__type__': 'numpy_array',
                    'data': obj.tolist(),
                    'dtype': str(obj.dtype),
                    'shape': obj.shape
                }
            
            # Handle collections
            elif isinstance(obj, deque):
                return {
                    '__type__': 'deque',
                    'data': list(obj),
                    'maxlen': obj.maxlen
                }
            elif isinstance(obj, set):
                return {
                    '__type__': 'set',
                    'data': list(obj)
                }
            elif hasattr(obj, 'keys') and hasattr(obj, 'values') and hasattr(obj, '__getitem__'):
                # Handle mappingproxy and other mapping-like objects
                return {
                    '__type__': 'mapping',
                    'data': dict(obj)
                }
            
            # Handle datetime objects
            elif hasattr(obj, 'isoformat'):
                return {
                    '__type__': 'datetime',
                    'data': obj.isoformat()
                }
            
            # Handle PyTorch tensors
            elif hasattr(obj, 'detach') and hasattr(obj, 'cpu'):
                return {
                    '__type__': 'pytorch_tensor',
                    'data': obj.detach().cpu().numpy().tolist(),
                    'shape': list(obj.shape),
                    'dtype': str(obj.dtype),
                    'device': str(obj.device)
                }
            
            # Handle objects with __dict__
            elif hasattr(obj, '__dict__'):
                return {
                    '__type__': 'object',
                    'class': obj.__class__.__name__,
                    'data': obj.__dict__
                }
            
            # Fallback to string representation
            else:
                return str(obj)
                
        except Exception as e:
            logger.warning(f"Error encoding object {type(obj)}: {e}")
            return str(obj)
    
    def _json_decoder(self, obj: Dict[str, Any]) -> Any:
        """Custom JSON decoder for complex objects"""
        if not isinstance(obj, dict) or '__type__' not in obj:
            return obj
        
        try:
            obj_type = obj['__type__']
            
            if obj_type == 'numpy_array':
                return np.array(obj['data'], dtype=obj['dtype']).reshape(obj['shape'])
            elif obj_type == 'deque':
                return deque(obj['data'], maxlen=obj.get('maxlen'))
            elif obj_type == 'set':
                return set(obj['data'])
            elif obj_type == 'mapping':
                return obj['data']  # Return as regular dict
            elif obj_type == 'datetime':
                from datetime import datetime
                return datetime.fromisoformat(obj['data'])
            elif obj_type == 'pytorch_tensor':
                import torch
                tensor = torch.tensor(obj['data'], dtype=getattr(torch, obj['dtype'].split('.')[-1]))
                return tensor.reshape(obj['shape'])
            
            return obj
            
        except Exception as e:
            logger.warning(f"Error decoding object type {obj.get('__type__')}: {e}")
            return obj
    
    def _detect_data_type(self, data: Any) -> DataType:
        """Detect the primary data type for optimization"""
        if isinstance(data, (int, float, str, bool, type(None))):
            return DataType.PRIMITIVE
        elif isinstance(data, np.ndarray):
            return DataType.NUMPY_ARRAY
        elif hasattr(data, 'detach') and hasattr(data, 'cpu'):
            return DataType.PYTORCH_TENSOR
        elif hasattr(data, 'isoformat'):
            return DataType.DATETIME
        elif isinstance(data, deque):
            return DataType.DEQUE
        elif hasattr(data, 'keys') and hasattr(data, 'values') and hasattr(data, '__getitem__') and not isinstance(data, dict):
            return DataType.MAPPING
        elif isinstance(data, (list, dict, tuple, set)):
            return DataType.COLLECTION
        else:
            return DataType.COMPLEX_OBJECT
    
    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate SHA256 checksum of data"""
        return hashlib.sha256(data).hexdigest()


class PickleSerializer:
    """Pickle-based serializer for complex Python objects"""
    
    def __init__(self, protocol: int = pickle.HIGHEST_PROTOCOL):
        self.protocol = protocol
    
    def serialize(self, data: Any) -> bytes:
        """Serialize data to pickle bytes"""
        try:
            return pickle.dumps(data, protocol=self.protocol)
        except Exception as e:
            raise SerializationError(f"Pickle serialization failed: {e}")
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize pickle bytes to data"""
        try:
            return pickle.loads(data)
        except Exception as e:
            raise DeserializationError(f"Pickle deserialization failed: {e}")
    
    def get_metadata(self, data: Any) -> SerializationMetadata:
        """Get metadata for pickle serialization"""
        serialized = self.serialize(data)
        return SerializationMetadata(
            format=SerializationFormat.PICKLE,
            data_type=DataType.COMPLEX_OBJECT,
            checksum=self._calculate_checksum(serialized),
            timestamp=time.time(),
            version="1.0",
            size_bytes=len(serialized)
        )
    
    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate SHA256 checksum of data"""
        return hashlib.sha256(data).hexdigest()


class CompressedSerializer:
    """Wrapper for compressed serialization"""
    
    def __init__(self, base_serializer: Union[JSONSerializer, PickleSerializer], 
                 compression_level: int = 6):
        self.base_serializer = base_serializer
        self.compression_level = compression_level
    
    def serialize(self, data: Any) -> bytes:
        """Serialize and compress data"""
        try:
            base_data = self.base_serializer.serialize(data)
            compressed = gzip.compress(base_data, compresslevel=self.compression_level)
            return compressed
        except Exception as e:
            raise SerializationError(f"Compressed serialization failed: {e}")
    
    def deserialize(self, data: bytes) -> Any:
        """Decompress and deserialize data"""
        try:
            decompressed = gzip.decompress(data)
            return self.base_serializer.deserialize(decompressed)
        except Exception as e:
            raise DeserializationError(f"Compressed deserialization failed: {e}")
    
    def get_metadata(self, data: Any) -> SerializationMetadata:
        """Get metadata for compressed serialization"""
        base_data = self.base_serializer.serialize(data)
        compressed = self.serialize(data)
        
        base_format = self.base_serializer.get_metadata(data).format
        compressed_format = (SerializationFormat.COMPRESSED_JSON 
                           if base_format == SerializationFormat.JSON 
                           else SerializationFormat.COMPRESSED_PICKLE)
        
        return SerializationMetadata(
            format=compressed_format,
            data_type=self.base_serializer.get_metadata(data).data_type,
            checksum=self._calculate_checksum(compressed),
            timestamp=time.time(),
            version="1.0",
            size_bytes=len(compressed),
            compression_ratio=len(base_data) / len(compressed)
        )
    
    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate SHA256 checksum of data"""
        return hashlib.sha256(data).hexdigest()


class UnifiedDataSerializer:
    """
    Unified data serialization system for the trading platform.
    
    This class provides a single interface for all serialization needs,
    automatically selecting the optimal serialization strategy based on
    data characteristics and requirements.
    
    Features:
    - Automatic format selection based on data type and size
    - Data integrity validation with checksums
    - Compression for large datasets
    - Comprehensive error handling and recovery
    - Metadata tracking for debugging and optimization
    """
    
    def __init__(self, 
                 default_format: SerializationFormat = SerializationFormat.JSON,
                 auto_compress_threshold: int = 10240,  # 10KB
                 compression_level: int = 6):
        """
        Initialize the unified serializer
        
        Args:
            default_format: Default serialization format
            auto_compress_threshold: Size threshold for automatic compression
            compression_level: Compression level (1-9)
        """
        self.default_format = default_format
        self.auto_compress_threshold = auto_compress_threshold
        self.compression_level = compression_level
        
        # Initialize serializers
        self.json_serializer = JSONSerializer(indent=None)
        self.pickle_serializer = PickleSerializer()
        self.compressed_json = CompressedSerializer(self.json_serializer, compression_level)
        self.compressed_pickle = CompressedSerializer(self.pickle_serializer, compression_level)
        
        # Statistics tracking
        self.stats = {
            'serializations': 0,
            'deserializations': 0,
            'compression_used': 0,
            'integrity_failures': 0,
            'total_bytes_saved': 0,
            'total_bytes_loaded': 0
        }
        
        logger.info(f"UnifiedDataSerializer initialized with format: {default_format.value}")
    
    def serialize_to_file(self, 
                          data: Any, 
                          filepath: Union[str, Path],
                          format_override: Optional[SerializationFormat] = None,
                          validate_integrity: bool = True) -> SerializationMetadata:
        """
        Serialize data to file with automatic format selection
        
        Args:
            data: Data to serialize
            filepath: Target file path
            format_override: Override automatic format selection
            validate_integrity: Perform integrity validation
            
        Returns:
            Serialization metadata
            
        Raises:
            SerializationError: If serialization fails
        """
        try:
            filepath = Path(filepath)
            
            # Select optimal serialization format
            selected_format = format_override or self._select_format(data)
            serializer = self._get_serializer(selected_format)
            
            # Serialize data
            serialized_data = serializer.serialize(data)
            metadata = serializer.get_metadata(data)
            
            # Validate integrity if requested
            if validate_integrity:
                self._validate_integrity(serialized_data, metadata.checksum)
            
            # Ensure directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to file atomically
            temp_filepath = filepath.with_suffix(filepath.suffix + '.tmp')
            
            with open(temp_filepath, 'wb') as f:
                f.write(serialized_data)
            
            # Write metadata file
            metadata_filepath = filepath.with_suffix(filepath.suffix + '.meta')
            with open(metadata_filepath, 'w') as f:
                json.dump(asdict(metadata), f, indent=2)
            
            # Atomic rename
            if filepath.exists():
                filepath.unlink()
            temp_filepath.rename(filepath)
            
            # Update statistics
            self.stats['serializations'] += 1
            self.stats['total_bytes_saved'] += len(serialized_data)
            if 'compressed' in selected_format.value:
                self.stats['compression_used'] += 1
            
            logger.debug(f"Serialized data to {filepath} using {selected_format.value} "
                        f"({len(serialized_data)} bytes)")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to serialize data to {filepath}: {e}")
            raise SerializationError(f"Serialization failed: {e}")
    
    def deserialize_from_file(self, 
                              filepath: Union[str, Path],
                              validate_integrity: bool = True) -> Any:
        """
        Deserialize data from file with integrity validation
        
        Args:
            filepath: Source file path
            validate_integrity: Perform integrity validation
            
        Returns:
            Deserialized data
            
        Raises:
            DeserializationError: If deserialization fails
            DataIntegrityError: If integrity validation fails
        """
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                raise DeserializationError(f"File not found: {filepath}")
            
            # Load metadata if available
            metadata_filepath = filepath.with_suffix(filepath.suffix + '.meta')
            metadata = None
            
            if metadata_filepath.exists():
                with open(metadata_filepath, 'r') as f:
                    metadata_dict = json.load(f)
                    metadata = SerializationMetadata(**metadata_dict)
            
            # Read file data
            with open(filepath, 'rb') as f:
                file_data = f.read()
            
            # Validate integrity if metadata available and requested
            if validate_integrity and metadata:
                self._validate_integrity(file_data, metadata.checksum)
            
            # Determine format from metadata or file extension
            if metadata:
                serializer = self._get_serializer(metadata.format)
            else:
                # Fallback to guessing from file extension or content
                serializer = self._guess_serializer_from_file(filepath, file_data)
            
            # Deserialize data
            data = serializer.deserialize(file_data)
            
            # Update statistics
            self.stats['deserializations'] += 1
            self.stats['total_bytes_loaded'] += len(file_data)
            
            logger.debug(f"Deserialized data from {filepath} "
                        f"({len(file_data)} bytes)")
            
            return data
            
        except DataIntegrityError:
            self.stats['integrity_failures'] += 1
            raise
        except Exception as e:
            logger.error(f"Failed to deserialize data from {filepath}: {e}")
            raise DeserializationError(f"Deserialization failed: {e}")
    
    def serialize_to_bytes(self, 
                           data: Any,
                           format_override: Optional[SerializationFormat] = None) -> Tuple[bytes, SerializationMetadata]:
        """
        Serialize data to bytes with metadata
        
        Args:
            data: Data to serialize
            format_override: Override automatic format selection
            
        Returns:
            Tuple of (serialized_bytes, metadata)
        """
        try:
            selected_format = format_override or self._select_format(data)
            serializer = self._get_serializer(selected_format)
            
            serialized_data = serializer.serialize(data)
            metadata = serializer.get_metadata(data)
            
            self.stats['serializations'] += 1
            self.stats['total_bytes_saved'] += len(serialized_data)
            
            return serialized_data, metadata
            
        except Exception as e:
            raise SerializationError(f"Byte serialization failed: {e}")
    
    def deserialize_from_bytes(self, 
                               data: bytes, 
                               metadata: SerializationMetadata,
                               validate_integrity: bool = True) -> Any:
        """
        Deserialize data from bytes using metadata
        
        Args:
            data: Serialized bytes
            metadata: Serialization metadata
            validate_integrity: Perform integrity validation
            
        Returns:
            Deserialized data
        """
        try:
            if validate_integrity:
                self._validate_integrity(data, metadata.checksum)
            
            serializer = self._get_serializer(metadata.format)
            result = serializer.deserialize(data)
            
            self.stats['deserializations'] += 1
            self.stats['total_bytes_loaded'] += len(data)
            
            return result
            
        except DataIntegrityError:
            self.stats['integrity_failures'] += 1
            raise
        except Exception as e:
            raise DeserializationError(f"Byte deserialization failed: {e}")
    
    def _select_format(self, data: Any) -> SerializationFormat:
        """
        Automatically select optimal serialization format
        
        Args:
            data: Data to analyze
            
        Returns:
            Optimal serialization format
        """
        try:
            # Try JSON first for simple data types
            json_serializer = self.json_serializer
            json_data = json_serializer.serialize(data)
            
            # Use compression for large JSON data
            if len(json_data) > self.auto_compress_threshold:
                return SerializationFormat.COMPRESSED_JSON
            else:
                return SerializationFormat.JSON
                
        except (SerializationError, TypeError, ValueError):
            # Fall back to pickle for complex objects
            pickle_serializer = self.pickle_serializer
            pickle_data = pickle_serializer.serialize(data)
            
            if len(pickle_data) > self.auto_compress_threshold:
                return SerializationFormat.COMPRESSED_PICKLE
            else:
                return SerializationFormat.PICKLE
    
    def _get_serializer(self, format: SerializationFormat) -> Union[JSONSerializer, PickleSerializer, CompressedSerializer]:
        """Get serializer instance for format"""
        if format == SerializationFormat.JSON:
            return self.json_serializer
        elif format == SerializationFormat.PICKLE:
            return self.pickle_serializer
        elif format == SerializationFormat.COMPRESSED_JSON:
            return self.compressed_json
        elif format == SerializationFormat.COMPRESSED_PICKLE:
            return self.compressed_pickle
        else:
            raise ValueError(f"Unsupported serialization format: {format}")
    
    def _validate_integrity(self, data: bytes, expected_checksum: str):
        """
        Validate data integrity using checksum
        
        Args:
            data: Data bytes to validate
            expected_checksum: Expected SHA256 checksum
            
        Raises:
            DataIntegrityError: If checksum validation fails
        """
        actual_checksum = hashlib.sha256(data).hexdigest()
        if actual_checksum != expected_checksum:
            raise DataIntegrityError(
                f"Data integrity validation failed. Expected: {expected_checksum}, "
                f"Got: {actual_checksum}"
            )
    
    def _guess_serializer_from_file(self, filepath: Path, data: bytes) -> Union[JSONSerializer, PickleSerializer, CompressedSerializer]:
        """
        Guess serializer from file extension and content
        
        Args:
            filepath: File path for extension analysis
            data: File data for content analysis
            
        Returns:
            Best guess serializer
        """
        # Check file extension
        if filepath.suffix == '.json':
            return self.json_serializer
        elif filepath.suffix == '.pkl' or filepath.suffix == '.pickle':
            return self.pickle_serializer
        elif filepath.suffix == '.gz':
            # Check if it's compressed JSON or pickle
            try:
                decompressed = gzip.decompress(data)
                # Try to parse as JSON
                json.loads(decompressed.decode('utf-8'))
                return self.compressed_json
            except:
                return self.compressed_pickle
        
        # Content-based detection
        try:
            # Try JSON first
            json.loads(data.decode('utf-8'))
            return self.json_serializer
        except:
            # Default to pickle
            return self.pickle_serializer
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get serialization statistics"""
        return {
            **self.stats,
            'compression_ratio': (
                self.stats['compression_used'] / max(1, self.stats['serializations'])
            ),
            'integrity_failure_rate': (
                self.stats['integrity_failures'] / max(1, self.stats['deserializations'])
            ),
            'avg_bytes_per_serialization': (
                self.stats['total_bytes_saved'] / max(1, self.stats['serializations'])
            ),
            'avg_bytes_per_deserialization': (
                self.stats['total_bytes_loaded'] / max(1, self.stats['deserializations'])
            )
        }
    
    def reset_statistics(self):
        """Reset serialization statistics"""
        self.stats = {
            'serializations': 0,
            'deserializations': 0,
            'compression_used': 0,
            'integrity_failures': 0,
            'total_bytes_saved': 0,
            'total_bytes_loaded': 0
        }
        logger.info("Serialization statistics reset")


# Global instance for system-wide use
unified_serializer = UnifiedDataSerializer()


def serialize_component_state(component_name: str, data: Any, 
                            base_path: Union[str, Path] = "data") -> SerializationMetadata:
    """
    Convenience function to serialize component state
    
    Args:
        component_name: Name of the component
        data: Data to serialize
        base_path: Base directory for storage
        
    Returns:
        Serialization metadata
    """
    base_path = Path(base_path)
    filepath = base_path / f"{component_name}_state.dat"
    return unified_serializer.serialize_to_file(data, filepath)


def deserialize_component_state(component_name: str, 
                               base_path: Union[str, Path] = "data") -> Optional[Any]:
    """
    Convenience function to deserialize component state
    
    Args:
        component_name: Name of the component
        base_path: Base directory for storage
        
    Returns:
        Deserialized data or None if file doesn't exist
    """
    base_path = Path(base_path)
    filepath = base_path / f"{component_name}_state.dat"
    
    if not filepath.exists():
        return None
    
    try:
        return unified_serializer.deserialize_from_file(filepath)
    except Exception as e:
        logger.error(f"Failed to deserialize {component_name} state: {e}")
        return None