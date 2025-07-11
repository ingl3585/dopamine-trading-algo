# Core module - removing conflicting imports that cause dependency issues
from .config import Config

# Modernized components available via direct imports
# from .system_integration_orchestrator import SystemIntegrationOrchestrator
# from .component_integrator import ComponentIntegrator
# from .event_bus import EventBus

__all__ = ['Config']