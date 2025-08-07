"""
State Coordinator - Atomic state management across all system components
"""

import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class StateComponent:
    """Definition of a component that participates in state management"""
    name: str
    save_method: Callable[[], Dict[str, Any]]
    load_method: Callable[[Dict[str, Any]], None]
    priority: int = 0  # Higher priority components save/load first
    enabled: bool = True

@dataclass 
class StateSnapshot:
    """Complete system state snapshot"""
    timestamp: float
    version: str
    components: Dict[str, Any]
    metadata: Dict[str, Any]

class StateCoordinator:
    """
    Coordinates atomic state saves/loads across all system components
    Prevents race conditions and ensures state consistency
    """
    
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        self._components: Dict[str, StateComponent] = {}
        self._state_lock = threading.RLock()  # Reentrant lock for nested calls
        self._save_in_progress = False
        self._load_in_progress = False
        
        # State versioning
        self._current_version = "1.0"
        self._state_history: List[StateSnapshot] = []
        self._max_history = 10
        
        # Auto-save configuration
        self._auto_save_enabled = False
        self._auto_save_interval = 300  # 5 minutes
        self._last_auto_save = 0
        
        # Transactional state management
        self._transaction_active = False
        self._transaction_backup = None
        self._transaction_id = None
        
    def register_component(self, component: StateComponent):
        """Register a component for coordinated state management"""
        with self._state_lock:
            self._components[component.name] = component
            logger.info(f"Registered state component: {component.name} (priority: {component.priority})")
    
    def unregister_component(self, name: str):
        """Unregister a component"""
        with self._state_lock:
            if name in self._components:
                del self._components[name]
                logger.info(f"Unregistered state component: {name}")
    
    def save_state(self, filepath: Optional[str] = None) -> bool:
        """
        Atomically save state from all registered components
        Returns True if successful, False otherwise
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.base_path / f"system_state_{timestamp}.json"
        else:
            filepath = Path(filepath)
        
        with self._state_lock:
            if self._save_in_progress:
                logger.warning("State save already in progress, skipping")
                return False
            
            self._save_in_progress = True
            temp_filepath = None
            
            try:
                # Get sorted components by priority (higher first)
                components = sorted(
                    [(name, comp) for name, comp in self._components.items() if comp.enabled],
                    key=lambda x: x[1].priority,
                    reverse=True
                )
                
                state_data = {}
                failed_components = []
                
                # Save each component's state
                for name, component in components:
                    try:
                        logger.debug(f"Saving state for component: {name}")
                        component_state = component.save_method()
                        state_data[name] = component_state
                    except Exception as e:
                        logger.error(f"Failed to save state for component {name}: {e}")
                        failed_components.append(name)
                
                # Create state snapshot
                snapshot = StateSnapshot(
                    timestamp=time.time(),
                    version=self._current_version,
                    components=state_data,
                    metadata={
                        'saved_at': datetime.now().isoformat(),
                        'failed_components': failed_components,
                        'total_components': len(components),
                        'successful_components': len(components) - len(failed_components)
                    }
                )
                
                # Save to file with atomic write - improved temp file handling
                temp_filepath = filepath.with_suffix('.tmp')
                
                # Clean up any existing temp file
                if temp_filepath.exists():
                    try:
                        temp_filepath.unlink()
                        logger.debug(f"Removed existing temp file: {temp_filepath}")
                    except Exception as e:
                        logger.warning(f"Failed to remove existing temp file {temp_filepath}: {e}")
                
                # Write to temp file
                try:
                    with open(temp_filepath, 'w') as f:
                        json.dump(asdict(snapshot), f, indent=2, default=self._json_encoder)
                    logger.debug(f"State data written to temp file: {temp_filepath}")
                except Exception as e:
                    logger.error(f"Failed to write to temp file {temp_filepath}: {e}")
                    raise
                
                # Atomic rename - handle existing file
                try:
                    if filepath.exists():
                        filepath.unlink()
                    temp_filepath.rename(filepath)
                    logger.debug(f"Temp file successfully renamed to: {filepath}")
                    temp_filepath = None  # Mark as successfully renamed
                except Exception as e:
                    logger.error(f"Failed to rename temp file {temp_filepath} to {filepath}: {e}")
                    raise
                
                # Update history
                self._state_history.append(snapshot)
                if len(self._state_history) > self._max_history:
                    self._state_history.pop(0)
                
                # Update auto-save timestamp
                self._last_auto_save = time.time()
                
                logger.info(f"State saved successfully to {filepath} "
                           f"({len(state_data)}/{len(components)} components)")
                
                if failed_components:
                    logger.warning(f"Failed to save components: {failed_components}")
                
                return len(failed_components) == 0
                
            except Exception as e:
                logger.error(f"Critical error during state save: {e}")
                
                # Clean up temp file on error
                if temp_filepath and temp_filepath.exists():
                    try:
                        temp_filepath.unlink()
                        logger.debug(f"Cleaned up temp file after error: {temp_filepath}")
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to clean up temp file {temp_filepath}: {cleanup_error}")
                
                return False
            
            finally:
                self._save_in_progress = False
    
    def load_state(self, filepath: str) -> bool:
        """
        Atomically load state to all registered components
        Returns True if successful, False otherwise
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.error(f"State file not found: {filepath}")
            return False
        
        with self._state_lock:
            if self._load_in_progress:
                logger.warning("State load already in progress, skipping")
                return False
            
            self._load_in_progress = True
            
            try:
                # Load state data
                with open(filepath, 'r') as f:
                    snapshot_data = json.load(f)
                
                # Validate structure
                if not self._validate_state_structure(snapshot_data):
                    logger.error("Invalid state file structure")
                    return False
                
                snapshot = StateSnapshot(**snapshot_data)
                
                # Get sorted components by priority (higher first for loading too)
                components = sorted(
                    [(name, comp) for name, comp in self._components.items() if comp.enabled],
                    key=lambda x: x[1].priority,
                    reverse=True
                )
                
                failed_components = []
                loaded_components = []
                
                # Load each component's state
                for name, component in components:
                    if name in snapshot.components:
                        try:
                            logger.debug(f"Loading state for component: {name}")
                            component.load_method(snapshot.components[name])
                            loaded_components.append(name)
                        except Exception as e:
                            logger.error(f"Failed to load state for component {name}: {e}")
                            failed_components.append(name)
                    else:
                        logger.warning(f"No state data found for component: {name}")
                
                logger.info(f"State loaded from {filepath} "
                           f"({len(loaded_components)}/{len(components)} components)")
                
                if failed_components:
                    logger.warning(f"Failed to load components: {failed_components}")
                
                # Add to history
                self._state_history.append(snapshot)
                if len(self._state_history) > self._max_history:
                    self._state_history.pop(0)
                
                return len(failed_components) == 0
                
            except Exception as e:
                logger.error(f"Critical error during state load: {e}")
                return False
            
            finally:
                self._load_in_progress = False
    
    def enable_auto_save(self, interval_seconds: int = 300):
        """Enable automatic state saving"""
        self._auto_save_enabled = True
        self._auto_save_interval = interval_seconds
        logger.info(f"Auto-save enabled with {interval_seconds}s interval")
    
    def disable_auto_save(self):
        """Disable automatic state saving"""
        self._auto_save_enabled = False
        logger.info("Auto-save disabled")
    
    def check_auto_save(self):
        """Check if auto-save should trigger (call from main loop)"""
        if not self._auto_save_enabled:
            return
        
        current_time = time.time()
        if current_time - self._last_auto_save >= self._auto_save_interval:
            logger.debug("Triggering auto-save")
            self.save_state()
    
    def begin_transaction(self, transaction_id: Optional[str] = None) -> str:
        """
        Begin a transactional state save operation.
        
        This creates a backup of the current state that can be rolled back to
        if the transaction fails or is explicitly aborted.
        
        Args:
            transaction_id: Optional transaction identifier
            
        Returns:
            Transaction ID for tracking
            
        Raises:
            RuntimeError: If a transaction is already active
        """
        with self._state_lock:
            if self._transaction_active:
                raise RuntimeError(f"Transaction {self._transaction_id} is already active")
            
            # Generate transaction ID if not provided
            if transaction_id is None:
                import uuid
                transaction_id = str(uuid.uuid4())[:8]
            
            try:
                logger.info(f"Beginning transaction: {transaction_id}")
                
                # Create backup of current state
                self._transaction_backup = self._create_state_backup()
                self._transaction_active = True
                self._transaction_id = transaction_id
                
                logger.debug(f"Transaction {transaction_id} backup created successfully")
                return transaction_id
                
            except Exception as e:
                logger.error(f"Failed to begin transaction {transaction_id}: {e}")
                self._cleanup_transaction()
                raise
    
    def commit_transaction(self, transaction_id: str) -> bool:
        """
        Commit the current transaction and clear the backup.
        
        Args:
            transaction_id: Transaction ID to commit
            
        Returns:
            True if commit was successful
            
        Raises:
            RuntimeError: If no transaction is active or ID mismatch
        """
        with self._state_lock:
            if not self._transaction_active:
                raise RuntimeError("No active transaction to commit")
            
            if self._transaction_id != transaction_id:
                raise RuntimeError(f"Transaction ID mismatch: expected {self._transaction_id}, got {transaction_id}")
            
            try:
                logger.info(f"Committing transaction: {transaction_id}")
                
                # Clear the backup as we're committing the changes
                self._cleanup_transaction()
                
                # Optionally save the current state to disk
                success = self.save_state()
                
                logger.info(f"Transaction {transaction_id} committed successfully")
                return success
                
            except Exception as e:
                logger.error(f"Failed to commit transaction {transaction_id}: {e}")
                return False
    
    def rollback_transaction(self, transaction_id: str) -> bool:
        """
        Rollback the current transaction and restore the backup state.
        
        Args:
            transaction_id: Transaction ID to rollback
            
        Returns:
            True if rollback was successful
            
        Raises:
            RuntimeError: If no transaction is active or ID mismatch
        """
        with self._state_lock:
            if not self._transaction_active:
                raise RuntimeError("No active transaction to rollback")
            
            if self._transaction_id != transaction_id:
                raise RuntimeError(f"Transaction ID mismatch: expected {self._transaction_id}, got {transaction_id}")
            
            try:
                logger.warning(f"Rolling back transaction: {transaction_id}")
                
                if self._transaction_backup is None:
                    logger.error("No backup available for rollback")
                    return False
                
                # Restore state from backup
                success = self._restore_state_backup(self._transaction_backup)
                
                # Clear transaction state
                self._cleanup_transaction()
                
                if success:
                    logger.info(f"Transaction {transaction_id} rolled back successfully")
                else:
                    logger.error(f"Failed to rollback transaction {transaction_id}")
                
                return success
                
            except Exception as e:
                logger.error(f"Error during rollback of transaction {transaction_id}: {e}")
                self._cleanup_transaction()
                return False
    
    def abort_transaction(self, transaction_id: str):
        """
        Abort the current transaction without rollback.
        
        This is used when you want to cancel a transaction but keep
        any changes that have been made.
        
        Args:
            transaction_id: Transaction ID to abort
        """
        with self._state_lock:
            if not self._transaction_active:
                logger.warning("No active transaction to abort")
                return
            
            if self._transaction_id != transaction_id:
                logger.warning(f"Transaction ID mismatch during abort: expected {self._transaction_id}, got {transaction_id}")
                return
            
            logger.info(f"Aborting transaction: {transaction_id}")
            self._cleanup_transaction()
    
    def _create_state_backup(self) -> Dict[str, Any]:
        """Create a backup of the current state from all components"""
        try:
            # Get sorted components by priority (higher first)
            components = sorted(
                [(name, comp) for name, comp in self._components.items() if comp.enabled],
                key=lambda x: x[1].priority,
                reverse=True
            )
            
            backup_data = {}
            failed_components = []
            
            # Save each component's state
            for name, component in components:
                try:
                    logger.debug(f"Creating backup for component: {name}")
                    component_state = component.save_method()
                    backup_data[name] = component_state
                except Exception as e:
                    logger.error(f"Failed to create backup for component {name}: {e}")
                    failed_components.append(name)
            
            # Create backup metadata
            backup = {
                'timestamp': time.time(),
                'version': self._current_version,
                'components': backup_data,
                'failed_components': failed_components,
                'total_components': len(components),
                'successful_components': len(components) - len(failed_components)
            }
            
            logger.debug(f"State backup created: {len(backup_data)}/{len(components)} components")
            return backup
            
        except Exception as e:
            logger.error(f"Critical error creating state backup: {e}")
            raise
    
    def _restore_state_backup(self, backup: Dict[str, Any]) -> bool:
        """Restore state from backup to all components"""
        try:
            if not backup or 'components' not in backup:
                logger.error("Invalid backup data structure")
                return False
            
            # Get sorted components by priority (higher first for loading too)
            components = sorted(
                [(name, comp) for name, comp in self._components.items() if comp.enabled],
                key=lambda x: x[1].priority,
                reverse=True
            )
            
            failed_components = []
            restored_components = []
            
            # Restore each component's state
            for name, component in components:
                if name in backup['components']:
                    try:
                        logger.debug(f"Restoring backup for component: {name}")
                        component.load_method(backup['components'][name])
                        restored_components.append(name)
                    except Exception as e:
                        logger.error(f"Failed to restore backup for component {name}: {e}")
                        failed_components.append(name)
                else:
                    logger.warning(f"No backup data found for component: {name}")
            
            logger.info(f"State backup restored: {len(restored_components)}/{len(components)} components")
            
            if failed_components:
                logger.warning(f"Failed to restore components: {failed_components}")
            
            return len(failed_components) == 0
            
        except Exception as e:
            logger.error(f"Critical error restoring state backup: {e}")
            return False
    
    def _cleanup_transaction(self):
        """Clean up transaction state"""
        self._transaction_active = False
        self._transaction_backup = None
        self._transaction_id = None
    
    def get_transaction_status(self) -> Dict[str, Any]:
        """Get current transaction status"""
        with self._state_lock:
            return {
                'active': self._transaction_active,
                'transaction_id': self._transaction_id,
                'has_backup': self._transaction_backup is not None,
                'backup_timestamp': (
                    self._transaction_backup.get('timestamp') 
                    if self._transaction_backup else None
                ),
                'backup_component_count': (
                    self._transaction_backup.get('successful_components') 
                    if self._transaction_backup else 0
                )
            }
    
    def get_latest_state_file(self) -> Optional[Path]:
        """Get the most recent state file"""
        state_files = list(self.base_path.glob("system_state_*.json"))
        if not state_files:
            return None
        
        # Sort by modification time
        return max(state_files, key=lambda f: f.stat().st_mtime)
    
    def _validate_state_structure(self, data: Dict[str, Any]) -> bool:
        """Validate state file structure"""
        required_fields = ['timestamp', 'version', 'components', 'metadata']
        return all(field in data for field in required_fields)
    
    def _json_encoder(self, obj):
        """JSON encoder for complex objects"""
        import numpy as np
        from collections import deque
        
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, deque):
            return list(obj)
        elif hasattr(obj, 'keys') and hasattr(obj, 'values') and hasattr(obj, '__getitem__'):
            # Handle mappingproxy and other mapping-like objects
            return dict(obj)
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        elif callable(obj):
            # Handle functions, methods, and other callable objects
            if hasattr(obj, '__name__'):
                return f"<callable:{obj.__name__}>"
            else:
                return f"<callable:{type(obj).__name__}>"
        elif hasattr(obj, '__dict__'):  # Objects with attributes
            # Filter out callable attributes to prevent function serialization
            filtered_dict = {}
            for key, value in obj.__dict__.items():
                if not callable(value):
                    filtered_dict[key] = value
                else:
                    filtered_dict[key] = f"<callable:{getattr(value, '__name__', type(value).__name__)}>"
            return filtered_dict
        
        # Last resort: convert to string representation
        logger.warning(f"Converting non-serializable object {type(obj)} to string: {obj}")
        return str(obj)
    
    def cleanup_orphaned_temp_files(self) -> int:
        """
        Clean up orphaned .tmp files in the base directory
        
        Returns:
            Number of orphaned temp files cleaned up
        """
        with self._state_lock:
            try:
                cleaned_count = 0
                
                # Find all .tmp files in base directory
                temp_files = list(self.base_path.glob("*.tmp"))
                
                for temp_file in temp_files:
                    try:
                        # Check if temp file is old (> 1 hour)
                        stat = temp_file.stat()
                        age_hours = (time.time() - stat.st_mtime) / 3600
                        
                        if age_hours > 1:
                            # Check if corresponding main file exists
                            main_file = temp_file.with_suffix('.json')
                            
                            # If main file doesn't exist or temp file is much older, it's orphaned
                            if not main_file.exists() or age_hours > 24:
                                temp_file.unlink()
                                cleaned_count += 1
                                logger.debug(f"Cleaned up orphaned temp file: {temp_file.name}")
                    
                    except Exception as e:
                        logger.warning(f"Failed to process temp file {temp_file}: {e}")
                
                if cleaned_count > 0:
                    logger.info(f"Cleaned up {cleaned_count} orphaned temp files")
                
                return cleaned_count
                
            except Exception as e:
                logger.error(f"Error during orphaned temp file cleanup: {e}")
                return 0
    
    def initialize_cleanup_system(self):
        """
        Initialize the state cleanup system integration
        
        This method sets up the cleanup system to work with this StateCoordinator
        """
        try:
            # Import here to avoid circular imports
            from .state_cleanup.coordinator import initialize_cleanup_system
            
            cleanup_coordinator = initialize_cleanup_system(
                base_path=str(self.base_path),
                state_coordinator=self
            )
            
            logger.info("State cleanup system initialized")
            return cleanup_coordinator
            
        except Exception as e:
            logger.error(f"Failed to initialize cleanup system: {e}")
            return None
    
    def get_coordinator_status(self) -> Dict[str, Any]:
        """Get coordinator status for monitoring"""
        with self._state_lock:
            return {
                'components_registered': len(self._components),
                'components_enabled': sum(1 for c in self._components.values() if c.enabled),
                'auto_save_enabled': self._auto_save_enabled,
                'auto_save_interval': self._auto_save_interval,
                'last_auto_save': self._last_auto_save,
                'save_in_progress': self._save_in_progress,
                'load_in_progress': self._load_in_progress,
                'state_history_size': len(self._state_history),
                'current_version': self._current_version,
                'transaction_active': self._transaction_active,
                'transaction_id': self._transaction_id
            }

# Global state coordinator instance
state_coordinator = StateCoordinator()

def register_state_component(name: str, save_method: Callable, load_method: Callable, priority: int = 0):
    """Convenience function to register a state component"""
    component = StateComponent(
        name=name,
        save_method=save_method,
        load_method=load_method,
        priority=priority
    )
    state_coordinator.register_component(component)