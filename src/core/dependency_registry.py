"""
Dependency Registry - Centralized dependency injection to break circular imports
"""

import logging
from typing import Dict, Any, Type, Optional, Callable
from dataclasses import dataclass
import threading

logger = logging.getLogger(__name__)

@dataclass
class ServiceDefinition:
    """Definition of a service for dependency injection"""
    service_type: Type
    factory: Callable
    singleton: bool = True
    initialized: bool = False
    instance: Optional[Any] = None

class DependencyRegistry:
    """
    Centralized registry for dependency injection
    Breaks circular import cycles by delaying instantiation
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self._services: Dict[str, ServiceDefinition] = {}
            self._instances: Dict[str, Any] = {}
            self._initialization_lock = threading.Lock()
            self.initialized = True
    
    def register_factory(self, name: str, service_type: Type, factory: Callable, singleton: bool = True):
        """Register a service factory"""
        self._services[name] = ServiceDefinition(
            service_type=service_type,
            factory=factory,
            singleton=singleton
        )
        logger.debug(f"Registered service factory: {name}")
    
    def register_instance(self, name: str, instance: Any):
        """Register an existing instance"""
        self._instances[name] = instance
        logger.debug(f"Registered service instance: {name}")
    
    def get(self, name: str) -> Any:
        """Get service instance, creating if necessary"""
        # Check if we have a pre-registered instance
        if name in self._instances:
            return self._instances[name]
        
        # Check if we have a service definition
        if name not in self._services:
            raise ValueError(f"Service '{name}' not registered")
        
        service_def = self._services[name]
        
        # For singletons, use locking to ensure single creation
        if service_def.singleton:
            with self._initialization_lock:
                # Double-check after acquiring lock
                if name in self._instances:
                    return self._instances[name]
                
                if not service_def.initialized:
                    logger.debug(f"Creating singleton service: {name}")
                    instance = service_def.factory()
                    self._instances[name] = instance
                    service_def.initialized = True
                    service_def.instance = instance
                    return instance
                else:
                    return service_def.instance
        else:
            # Non-singleton, create new instance each time
            logger.debug(f"Creating new service instance: {name}")
            return service_def.factory()
    
    def has(self, name: str) -> bool:
        """Check if service is registered"""
        return name in self._services or name in self._instances
    
    def clear(self):
        """Clear all registrations (for testing)"""
        with self._initialization_lock:
            self._services.clear()
            self._instances.clear()
    
    def get_registry_status(self) -> Dict[str, Any]:
        """Get status of all registered services"""
        status = {
            'registered_services': list(self._services.keys()),
            'active_instances': list(self._instances.keys()),
            'service_details': {}
        }
        
        for name, service_def in self._services.items():
            status['service_details'][name] = {
                'type': service_def.service_type.__name__,
                'singleton': service_def.singleton,
                'initialized': service_def.initialized
            }
        
        return status

# Global registry instance
registry = DependencyRegistry()

def get_service(name: str) -> Any:
    """Convenience function to get service from global registry"""
    return registry.get(name)

def register_service(name: str, service_type: Type, factory: Callable, singleton: bool = True):
    """Convenience function to register service factory"""
    registry.register_factory(name, service_type, factory, singleton)

def register_instance(name: str, instance: Any):
    """Convenience function to register service instance"""
    registry.register_instance(name, instance)