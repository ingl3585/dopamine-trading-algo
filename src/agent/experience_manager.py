"""
Experience Manager - Centralized experience storage and sampling

Enhanced with unified state management, error handling, and data integrity validation.
This module handles all experience-related operations following clean architecture principles:

1. Experience buffer management (regular and priority)
2. Batch sampling for training
3. Previous task memory for catastrophic forgetting prevention
4. Experience validation and preprocessing
5. Memory-efficient storage and retrieval
6. Unified state management integration
7. Comprehensive error handling and recovery

Extracted from TradingAgent to centralize memory management and improve performance.
"""

import numpy as np
import torch
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
from queue import Queue
import threading

from src.core.state_coordinator import register_state_component
from src.core.unified_serialization import unified_serializer, serialize_component_state, deserialize_component_state
from src.core.storage_error_handler import storage_error_handler, handle_storage_operation
from src.core.data_integrity_validator import data_integrity_validator, validate_component_data, ValidationLevel

logger = logging.getLogger(__name__)


class ExperienceBuffer:
    """Thread-safe experience buffer with priority support"""
    
    def __init__(self, maxsize: int, buffer_type: str = "regular"):
        self.maxsize = maxsize
        self.buffer_type = buffer_type
        self._data = []
        self._lock = threading.Lock()
        
    def append(self, item: Dict[str, Any]) -> bool:
        """
        Add item to buffer
        
        Returns:
            True if item was added successfully
        """
        with self._lock:
            try:
                self._data.append(item)
                if len(self._data) > self.maxsize:
                    self._data.pop(0)  # Remove oldest
                return True
            except Exception as e:
                logger.error(f"Error appending to {self.buffer_type} buffer: {e}")
                return False
    
    def sample(self, n: int, replace: bool = False) -> List[Dict[str, Any]]:
        """Sample n items from buffer"""
        with self._lock:
            if not self._data:
                return []
            
            n = min(n, len(self._data))
            if n <= 0:
                return []
            
            try:
                if replace or n >= len(self._data):
                    return np.random.choice(self._data, size=n, replace=replace).tolist()
                else:
                    indices = np.random.choice(len(self._data), size=n, replace=False)
                    return [self._data[i] for i in indices]
            except Exception as e:
                logger.error(f"Error sampling from {self.buffer_type} buffer: {e}")
                return self._data[-n:]  # Return most recent as fallback
    
    def get_recent(self, n: int) -> List[Dict[str, Any]]:
        """Get n most recent items"""
        with self._lock:
            return self._data[-n:] if self._data else []
    
    def get_all(self) -> List[Dict[str, Any]]:
        """Get all items (thread-safe copy)"""
        with self._lock:
            return self._data.copy()
    
    def size(self) -> int:
        """Get current buffer size"""
        with self._lock:
            return len(self._data)
    
    def clear(self):
        """Clear all items from buffer"""
        with self._lock:
            self._data.clear()


class ExperienceValidator:
    """Validates and preprocesses experiences before storage"""
    
    @staticmethod
    def validate_experience(experience: Dict[str, Any]) -> bool:
        """
        Validate experience dictionary
        
        Args:
            experience: Experience dictionary to validate
            
        Returns:
            True if experience is valid
        """
        try:
            required_fields = ['state_features', 'action', 'reward', 'done']
            
            # Check required fields
            for field in required_fields:
                if field not in experience:
                    logger.warning(f"Missing required field: {field}")
                    return False
            
            # Validate field types and ranges
            if not isinstance(experience['state_features'], (list, np.ndarray)):
                logger.warning("state_features must be list or numpy array")
                return False
            
            if not isinstance(experience['action'], (int, np.integer)):
                logger.warning("action must be integer")
                return False
            
            if experience['action'] not in [0, 1, 2]:  # hold, buy, sell
                logger.warning(f"Invalid action: {experience['action']}")
                return False
            
            if not isinstance(experience['reward'], (int, float, np.number)):
                logger.warning("reward must be numeric")
                return False
            
            if not isinstance(experience['done'], bool):
                logger.warning("done must be boolean")
                return False
            
            # Validate state features
            state_features = experience['state_features']
            if len(state_features) == 0:
                logger.warning("Empty state_features")
                return False
            
            # Check for NaN or infinite values
            if isinstance(state_features, (list, np.ndarray)):
                state_array = np.array(state_features)
                if np.any(np.isnan(state_array)) or np.any(np.isinf(state_array)):
                    logger.warning("state_features contains NaN or infinite values")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating experience: {e}")
            return False
    
    @staticmethod
    def preprocess_experience(experience: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess experience for storage
        
        Args:
            experience: Raw experience dictionary
            
        Returns:
            Preprocessed experience dictionary
        """
        try:
            processed = experience.copy()
            
            # Ensure state_features is a list
            if isinstance(processed['state_features'], np.ndarray):
                processed['state_features'] = processed['state_features'].tolist()
            
            # Ensure reward is float
            processed['reward'] = float(processed['reward'])
            
            # Ensure action is int
            processed['action'] = int(processed['action'])
            
            # Add timestamp if not present
            if 'timestamp' not in processed:
                processed['timestamp'] = time.time()
            
            # Convert tensors to lists for storage
            if 'trade_data' in processed:
                trade_data = processed['trade_data']
                for key, value in trade_data.items():
                    if hasattr(value, 'detach'):  # PyTorch tensor
                        trade_data[key] = value.detach().cpu().numpy().tolist()
            
            return processed
            
        except Exception as e:
            logger.error(f"Error preprocessing experience: {e}")
            return experience  # Return original on error


class ExperienceManager:
    """
    Centralized experience management for the trading system.
    
    This class handles all experience storage, sampling, and memory management
    that was previously scattered throughout TradingAgent. It provides efficient
    storage with priority support and catastrophic forgetting prevention.
    """
    
    def __init__(self, 
                 config_or_maxsize = None,
                 priority_maxsize: int = 5000,
                 previous_task_maxsize: int = 1000):
        """
        Initialize the experience manager
        
        Args:
            config_or_maxsize: Either config dict or experience maxsize int for backward compatibility
            priority_maxsize: Maximum size of priority experience buffer  
            previous_task_maxsize: Maximum size of previous task buffer
        """
        # Handle config or maxsize parameter
        if isinstance(config_or_maxsize, dict):
            # Config dict passed - extract buffer sizes
            experience_maxsize = config_or_maxsize.get('experience_buffer_size', 20000)
            priority_maxsize = config_or_maxsize.get('priority_buffer_size', priority_maxsize)
            previous_task_maxsize = config_or_maxsize.get('previous_task_buffer_size', previous_task_maxsize)
        elif config_or_maxsize is not None:
            # Integer maxsize passed for backward compatibility
            experience_maxsize = config_or_maxsize
        else:
            # Default values
            experience_maxsize = 20000
        
        # Initialize buffers
        self.experience_buffer = ExperienceBuffer(experience_maxsize, "experience")
        self.priority_buffer = ExperienceBuffer(priority_maxsize, "priority")
        self.previous_task_buffer = ExperienceBuffer(previous_task_maxsize, "previous_task")
        
        # Experience validator
        self.validator = ExperienceValidator()
        
        # Statistics tracking
        self.stats = {
            'total_experiences_stored': 0,
            'priority_experiences_stored': 0,
            'previous_task_experiences_stored': 0,
            'validation_failures': 0,
            'last_cleanup_time': time.time()
        }
        
        # Configuration
        self.cleanup_interval = 3600  # 1 hour
        self.max_reward_threshold = 10.0  # For priority classification
        self.min_uncertainty_threshold = 0.7  # For priority classification
        
        # Register with state coordinator for unified state management
        self._register_state_management()
        
        # Load existing state if available
        self._load_persisted_state() 
        
        logger.info(f"ExperienceManager initialized with buffers: "
                   f"experience={experience_maxsize}, priority={priority_maxsize}, "
                   f"previous_task={previous_task_maxsize}")
    
    def store_experience(self, 
                        experience: Dict[str, Any],
                        force_priority: bool = False) -> bool:
        """
        Store experience in appropriate buffer
        
        Args:
            experience: Experience dictionary
            force_priority: Force storage in priority buffer
            
        Returns:
            True if experience was stored successfully
        """
        try:
            # Validate experience
            if not self.validator.validate_experience(experience):
                self.stats['validation_failures'] += 1
                logger.warning("Experience validation failed, skipping storage")
                return False
            
            # Preprocess experience
            processed_experience = self.validator.preprocess_experience(experience)
            
            # Determine buffer based on importance
            if force_priority or self._is_priority_experience(processed_experience):
                success = self.priority_buffer.append(processed_experience)
                if success:
                    self.stats['priority_experiences_stored'] += 1
                    logger.debug("Experience stored in priority buffer")
            else:
                success = self.experience_buffer.append(processed_experience)
                if success:
                    self.stats['total_experiences_stored'] += 1
                    logger.debug("Experience stored in regular buffer")
            
            # Periodic cleanup
            self._periodic_cleanup()
            
            return success
            
        except Exception as e:
            logger.error(f"Error storing experience: {e}")
            return False
    
    def store_previous_task_experience(self, experience: Dict[str, Any]) -> bool:
        """
        Store experience in previous task buffer for catastrophic forgetting prevention
        
        Args:
            experience: Experience dictionary
            
        Returns:
            True if experience was stored successfully
        """
        try:
            if not self.validator.validate_experience(experience):
                return False
            
            processed_experience = self.validator.preprocess_experience(experience)
            success = self.previous_task_buffer.append(processed_experience)
            
            if success:
                self.stats['previous_task_experiences_stored'] += 1
                logger.debug("Experience stored in previous task buffer")
            
            return success
            
        except Exception as e:
            logger.error(f"Error storing previous task experience: {e}")
            return False
    
    def sample_training_batch(self, 
                             batch_size: int = 32,
                             priority_ratio: float = 0.25,
                             include_previous_tasks: bool = True,
                             previous_task_ratio: float = 0.25) -> List[Dict[str, Any]]:
        """
        Sample a training batch from all buffers
        
        Args:
            batch_size: Total batch size
            priority_ratio: Ratio of priority experiences in batch
            include_previous_tasks: Whether to include previous task experiences
            previous_task_ratio: Ratio of previous task experiences in batch
            
        Returns:
            List of experience dictionaries
        """
        try:
            batch = []
            
            # Calculate sample sizes
            priority_size = int(batch_size * priority_ratio)
            previous_task_size = int(batch_size * previous_task_ratio) if include_previous_tasks else 0
            regular_size = batch_size - priority_size - previous_task_size
            
            # Sample from priority buffer
            if priority_size > 0 and self.priority_buffer.size() > 0:
                priority_samples = self.priority_buffer.sample(
                    min(priority_size, self.priority_buffer.size())
                )
                batch.extend(priority_samples)
                logger.debug(f"Sampled {len(priority_samples)} priority experiences")
            
            # Sample from previous task buffer
            if previous_task_size > 0 and self.previous_task_buffer.size() > 0:
                previous_task_samples = self.previous_task_buffer.sample(
                    min(previous_task_size, self.previous_task_buffer.size())
                )
                batch.extend(previous_task_samples)
                logger.debug(f"Sampled {len(previous_task_samples)} previous task experiences")
            
            # Sample from regular buffer to fill remaining slots
            remaining_slots = batch_size - len(batch)
            if remaining_slots > 0 and self.experience_buffer.size() > 0:
                regular_samples = self.experience_buffer.sample(
                    min(remaining_slots, self.experience_buffer.size())
                )
                batch.extend(regular_samples)
                logger.debug(f"Sampled {len(regular_samples)} regular experiences")
            
            # Shuffle batch
            if batch:
                np.random.shuffle(batch)
            
            logger.debug(f"Training batch created: {len(batch)} total experiences")
            return batch
            
        except Exception as e:
            logger.error(f"Error sampling training batch: {e}")
            return []
    
    def get_recent_experiences(self, 
                              n: int = 20,
                              buffer_type: str = "experience") -> List[Dict[str, Any]]:
        """
        Get recent experiences from specified buffer
        
        Args:
            n: Number of recent experiences to retrieve
            buffer_type: Type of buffer ("experience", "priority", "previous_task")
            
        Returns:
            List of recent experience dictionaries
        """
        try:
            if buffer_type == "experience":
                return self.experience_buffer.get_recent(n)
            elif buffer_type == "priority":
                return self.priority_buffer.get_recent(n)
            elif buffer_type == "previous_task":
                return self.previous_task_buffer.get_recent(n)
            else:
                logger.warning(f"Unknown buffer type: {buffer_type}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting recent experiences: {e}")
            return []
    
    def get_reward_statistics(self, 
                             buffer_type: str = "experience",
                             n_recent: int = 100) -> Dict[str, float]:
        """
        Get reward statistics from specified buffer
        
        Args:
            buffer_type: Type of buffer to analyze
            n_recent: Number of recent experiences to analyze
            
        Returns:
            Dictionary of reward statistics
        """
        try:
            experiences = self.get_recent_experiences(n_recent, buffer_type)
            
            if not experiences:
                return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0}
            
            rewards = [exp['reward'] for exp in experiences if 'reward' in exp]
            
            if not rewards:
                return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0}
            
            return {
                'mean': float(np.mean(rewards)),
                'std': float(np.std(rewards)),
                'min': float(np.min(rewards)),
                'max': float(np.max(rewards)),
                'count': len(rewards)
            }
            
        except Exception as e:
            logger.error(f"Error calculating reward statistics: {e}")
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0}
    
    def _is_priority_experience(self, experience: Dict[str, Any]) -> bool:
        """
        Determine if experience should be stored in priority buffer
        
        Args:
            experience: Experience dictionary
            
        Returns:
            True if experience should be prioritized
        """
        try:
            reward = experience.get('reward', 0.0)
            uncertainty = experience.get('uncertainty', 0.5)
            
            # High impact experiences (high absolute reward)
            if abs(reward) > self.max_reward_threshold:
                return True
            
            # High uncertainty experiences (potential learning opportunities)
            if uncertainty > self.min_uncertainty_threshold:
                return True
            
            # Rare action experiences (exploration outcomes)
            action = experience.get('action', 0)
            if action != 0:  # Non-hold actions are rarer and more valuable
                return True
            
            # Check if experience has regime confidence data
            trade_data = experience.get('trade_data', {})
            regime_confidence = trade_data.get('regime_confidence', 1.0)
            if regime_confidence < 0.3:  # Low confidence regime changes
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error determining priority: {e}")
            return False
    
    def _periodic_cleanup(self):
        """Perform periodic cleanup and maintenance"""
        try:
            current_time = time.time()
            
            if current_time - self.stats['last_cleanup_time'] > self.cleanup_interval:
                logger.debug("Performing periodic cleanup")
                
                # Could add cleanup logic here:
                # - Remove very old experiences
                # - Compress experiences
                # - Validate buffer integrity
                
                self.stats['last_cleanup_time'] = current_time
                
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {e}")
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get comprehensive buffer statistics"""
        try:
            return {
                'buffer_sizes': {
                    'experience': self.experience_buffer.size(),
                    'priority': self.priority_buffer.size(),
                    'previous_task': self.previous_task_buffer.size()
                },
                'buffer_capacities': {
                    'experience': self.experience_buffer.maxsize,
                    'priority': self.priority_buffer.maxsize,
                    'previous_task': self.previous_task_buffer.maxsize
                },
                'utilization': {
                    'experience': self.experience_buffer.size() / self.experience_buffer.maxsize,
                    'priority': self.priority_buffer.size() / self.priority_buffer.maxsize,
                    'previous_task': self.previous_task_buffer.size() / self.previous_task_buffer.maxsize
                },
                'statistics': self.stats.copy(),
                'reward_stats': {
                    'experience': self.get_reward_statistics("experience"),
                    'priority': self.get_reward_statistics("priority"),
                    'previous_task': self.get_reward_statistics("previous_task")
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting buffer stats: {e}")
            return {'error': str(e)}
    
    def clear_all_buffers(self):
        """Clear all experience buffers"""
        try:
            self.experience_buffer.clear()
            self.priority_buffer.clear()
            self.previous_task_buffer.clear()
            
            # Reset statistics
            self.stats = {
                'total_experiences_stored': 0,
                'priority_experiences_stored': 0,
                'previous_task_experiences_stored': 0,
                'validation_failures': 0,
                'last_cleanup_time': time.time()
            }
            
            logger.info("All experience buffers cleared")
            
        except Exception as e:
            logger.error(f"Error clearing buffers: {e}")
    
    def _register_state_management(self):
        """Register with state coordinator for unified state management"""
        try:
            register_state_component(
                name="experience_manager", 
                save_method=self._save_state_data,
                load_method=self._load_state_data,
                priority=90  # Very high priority for experience data
            )
            logger.debug("ExperienceManager registered with state coordinator")
        except Exception as e:
            logger.error(f"Failed to register experience manager state management: {e}")
    
    @handle_storage_operation("experience_manager", "save_state")
    def _save_state_data(self) -> Dict[str, Any]:
        """
        Save experience manager state data for state coordinator.
        
        Returns:
            Dictionary containing all experience manager state data
        """
        try:
            # Prepare state data
            state_data = {
                'experience_buffer': self.experience_buffer.get_all(),
                'priority_buffer': self.priority_buffer.get_all(),
                'previous_task_buffer': self.previous_task_buffer.get_all(),
                'buffer_sizes': {
                    'experience': self.experience_buffer.maxsize,
                    'priority': self.priority_buffer.maxsize,  
                    'previous_task': self.previous_task_buffer.maxsize
                },
                'stats': dict(self.stats),
                'config': {
                    'cleanup_interval': self.cleanup_interval,
                    'max_reward_threshold': self.max_reward_threshold,
                    'min_uncertainty_threshold': self.min_uncertainty_threshold
                },
                'saved_at': time.time(),
                'version': "2.0"  # Updated version for new state format
            }
            
            # Validate data integrity before saving
            validation_report = validate_component_data(
                component="experience_manager",
                data=state_data,
                data_type="experience_data",
                level=ValidationLevel.STANDARD
            )
            
            if not validation_report.is_valid:
                logger.warning(f"Experience manager state validation issues: {len(validation_report.issues)} issues found")
                for issue in validation_report.issues:
                    if issue.severity == "error":
                        logger.error(f"Experience manager state error: {issue.message}")
            
            logger.debug("Experience manager state data prepared for saving")
            return state_data
            
        except Exception as e:
            logger.error(f"Error preparing experience manager state data: {e}")
            raise
    
    @handle_storage_operation("experience_manager", "load_state")
    def _load_state_data(self, state_data: Dict[str, Any]):
        """
        Load experience manager state data from state coordinator.
        
        Args:
            state_data: Dictionary containing experience manager state data
        """
        try:
            # Validate loaded data
            validation_report = validate_component_data(
                component="experience_manager",
                data=state_data,
                data_type="experience_data", 
                level=ValidationLevel.STANDARD
            )
            
            if validation_report.result.value == "corrupted":
                logger.error("Experience manager state data is corrupted, cannot load")
                return
            
            if not validation_report.is_valid:
                logger.warning(f"Experience manager state validation issues during load: {len(validation_report.issues)} issues")
            
            # Load experience buffer
            if 'experience_buffer' in state_data and isinstance(state_data['experience_buffer'], list):
                experiences = state_data['experience_buffer']
                for exp in experiences:
                    if self.validator.validate_experience(exp):
                        self.experience_buffer.append(exp)
                logger.debug(f"Loaded {len(experiences)} experience buffer entries")
            
            # Load priority buffer  
            if 'priority_buffer' in state_data and isinstance(state_data['priority_buffer'], list):
                priority_experiences = state_data['priority_buffer']
                for exp in priority_experiences:
                    if self.validator.validate_experience(exp):
                        self.priority_buffer.append(exp)
                logger.debug(f"Loaded {len(priority_experiences)} priority buffer entries")
            
            # Load previous task buffer
            if 'previous_task_buffer' in state_data and isinstance(state_data['previous_task_buffer'], list):
                previous_task_experiences = state_data['previous_task_buffer']
                for exp in previous_task_experiences:
                    if self.validator.validate_experience(exp):
                        self.previous_task_buffer.append(exp)
                logger.debug(f"Loaded {len(previous_task_experiences)} previous task buffer entries")
            
            # Load statistics
            if 'stats' in state_data and isinstance(state_data['stats'], dict):
                self.stats.update(state_data['stats'])
                logger.debug("Loaded experience manager statistics")
            
            # Load configuration
            if 'config' in state_data and isinstance(state_data['config'], dict):
                config = state_data['config']
                self.cleanup_interval = config.get('cleanup_interval', self.cleanup_interval)
                self.max_reward_threshold = config.get('max_reward_threshold', self.max_reward_threshold)
                self.min_uncertainty_threshold = config.get('min_uncertainty_threshold', self.min_uncertainty_threshold)
                logger.debug("Loaded experience manager configuration")
            
            logger.info("Experience manager state loaded successfully from state coordinator")
            
        except Exception as e:
            logger.error(f"Error loading experience manager state data: {e}")
            raise
    
    def _load_persisted_state(self):
        """Load persisted state if available"""
        try:
            # Try to load from unified serialization system
            state_data = deserialize_component_state("experience_manager")
            if state_data:
                self._load_state_data(state_data)
                logger.info("Experience manager state loaded from persistent storage")
        except Exception as e:
            logger.debug(f"No existing experience manager state found or failed to load: {e}")
    
    @handle_storage_operation("experience_manager", "save_experiences_legacy")
    def save_experiences(self, filepath: str) -> bool:
        """
        Save experiences to file using unified serialization system.
        
        This method provides backward compatibility while using the new
        unified serialization framework with error handling and validation.
        
        Args:
            filepath: Path to save experiences
            
        Returns:
            True if save was successful
        """
        try:
            # Get current state data
            state_data = self._save_state_data()
            
            # Use unified serializer to save to specific file
            metadata = unified_serializer.serialize_to_file(
                data=state_data,
                filepath=filepath,
                validate_integrity=True
            )
            
            logger.info(f"Experiences saved to {filepath} using unified serialization")
            logger.debug(f"Saved data size: {metadata.size_bytes} bytes, format: {metadata.format.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving experiences to {filepath}: {e}")
            # Log error through error handler for tracking
            storage_error_handler.handle_error(
                exception=e,
                component="experience_manager",
                operation="save_experiences_legacy",
                context={"filepath": str(filepath)}
            )
            return False
    
    @handle_storage_operation("experience_manager", "load_experiences_legacy")
    def load_experiences(self, filepath: str) -> bool:
        """
        Load experiences from file using unified serialization system.
        
        This method provides backward compatibility while using the new
        unified serialization framework with error handling and validation.
        
        Args:
            filepath: Path to load experiences from
            
        Returns:
            True if load was successful
        """
        try:
            # Use unified serializer to load from specific file
            state_data = unified_serializer.deserialize_from_file(
                filepath=filepath,
                validate_integrity=True
            )
            
            if state_data:
                # Clear existing buffers before loading
                self.clear_all_buffers()
                
                # Load state data
                self._load_state_data(state_data)
                
                logger.info(f"Experiences loaded from {filepath} using unified serialization")
                logger.info(f"Loaded {self.experience_buffer.size()} regular, "
                           f"{self.priority_buffer.size()} priority, "
                           f"{self.previous_task_buffer.size()} previous task experiences")
                return True
            else:
                logger.warning(f"No data found in {filepath}")
                return False
            
        except FileNotFoundError:
            logger.info("No existing experience file found, starting fresh")
            return False
        except Exception as e:
            logger.error(f"Error loading experiences from {filepath}: {e}")
            # Log error through error handler for tracking
            error_info = storage_error_handler.handle_error(
                exception=e,
                component="experience_manager",
                operation="load_experiences_legacy",
                context={"filepath": str(filepath)}
            )
            
            # Don't fail completely for minor errors
            if error_info.severity.value in ["low", "medium"]:
                logger.warning("Continuing with empty experience buffers due to load error")
                return False
            else:
                return False