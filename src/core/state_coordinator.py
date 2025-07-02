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
                
                # Save to file with atomic write
                temp_filepath = filepath.with_suffix('.tmp')
                
                # Remove existing temp file if it exists
                if temp_filepath.exists():
                    temp_filepath.unlink()
                
                with open(temp_filepath, 'w') as f:
                    json.dump(asdict(snapshot), f, indent=2, default=self._json_encoder)
                
                # Atomic rename - handle existing file
                if filepath.exists():
                    filepath.unlink()
                temp_filepath.rename(filepath)
                
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
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):  # Objects with attributes
            return obj.__dict__
        
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
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
                'current_version': self._current_version
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