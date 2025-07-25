# system_integration_orchestrator.py

import logging
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .configuration_manager import ConfigurationManager
from .component_integrator import ComponentIntegrator, ModernizedTradingAgent
from .event_bus import EventBus, EventDrivenTradingSystem, EventType, EventDrivenComponent

logger = logging.getLogger(__name__)

@dataclass
class SystemHealth:
    """System health status"""
    overall_health: str  # healthy, warning, critical
    component_health: Dict[str, str]
    event_bus_health: str
    integration_health: str
    performance_health: str
    issues: list
    recommendations: list

class SystemIntegrationOrchestrator:
    """
    Master orchestrator that integrates all system components with event-driven architecture.
    
    Responsibilities:
    - Coordinate component integration with event-driven architecture
    - Manage system lifecycle and health monitoring
    - Provide unified interface to the modernized trading system
    - Handle system-wide error recovery and resilience
    - Monitor and optimize system performance
    """
    
    def __init__(self, config_file: Optional[str] = None):
        # Initialize configuration
        self.config_manager = ConfigurationManager(config_file)
        self.config = self.config_manager.get_section('')
        
        # Initialize core systems
        self.component_integrator = ComponentIntegrator(self.config_manager)
        self.event_driven_system = EventDrivenTradingSystem(self.config)
        self.event_bus = self.event_driven_system.event_bus
        
        # System state
        self.system_initialized = False
        self.system_running = False
        self.modernized_agent = None
        
        # Health monitoring
        self.health_monitor = SystemHealthMonitor(self.event_bus)
        
        logger.info("System integration orchestrator initialized")
    
    async def initialize_system(self) -> bool:
        """
        Initialize the complete modernized trading system
        
        Returns:
            bool: True if initialization successful
        """
        try:
            logger.info("Initializing modernized trading system...")
            
            # Step 1: Start event bus
            await self.event_driven_system.start_system()
            
            # Step 2: Integrate all components
            if not self.component_integrator.integrate_all_components():
                logger.error("Component integration failed")
                return False
            
            # Step 3: Create modernized trading agent
            self.modernized_agent = self.component_integrator.create_modernized_trading_agent()
            if not self.modernized_agent:
                logger.error("Failed to create modernized trading agent")
                return False
            
            # Step 4: Register components with event system
            await self._register_components_with_event_system()
            
            # Step 5: Start health monitoring
            await self.health_monitor.start_monitoring()
            
            # Step 6: Publish system initialization event
            await self._publish_system_event(EventType.SYSTEM_STARTED, {
                'initialization_time': asyncio.get_event_loop().time(),
                'components_integrated': True,
                'event_system_active': True
            })
            
            self.system_initialized = True
            logger.info("Modernized trading system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing system: {e}")
            await self._publish_system_event(EventType.SYSTEM_ERROR, {
                'error': str(e),
                'stage': 'initialization'
            })
            return False
    
    async def start_system(self) -> bool:
        """
        Start the modernized trading system
        
        Returns:
            bool: True if start successful
        """
        try:
            if not self.system_initialized:
                logger.error("System not initialized")
                return False
            
            if self.system_running:
                logger.warning("System already running")
                return True
            
            logger.info("Starting modernized trading system...")
            
            # Start all integrated components
            integrated_system = self.component_integrator.get_integrated_system()
            
            # Start orchestrator
            if integrated_system['orchestrator']:
                if not integrated_system['orchestrator'].start_system():
                    logger.error("Failed to start system orchestrator")
                    return False
            
            # Publish system start event
            await self._publish_system_event(EventType.SYSTEM_STARTED, {
                'start_time': asyncio.get_event_loop().time(),
                'components_active': True
            })
            
            self.system_running = True
            logger.info("Modernized trading system started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting system: {e}")
            await self._publish_system_event(EventType.SYSTEM_ERROR, {
                'error': str(e),
                'stage': 'startup'
            })
            return False
    
    async def stop_system(self):
        """Stop the modernized trading system"""
        try:
            if not self.system_running:
                logger.info("System not running")
                return
            
            logger.info("Stopping modernized trading system...")
            
            # Publish system stop event
            await self._publish_system_event(EventType.SYSTEM_STOPPED, {
                'stop_time': asyncio.get_event_loop().time()
            })
            
            # Stop health monitoring
            await self.health_monitor.stop_monitoring()
            
            # Stop integrated components
            self.component_integrator.shutdown_integrated_system()
            
            # Stop event system
            await self.event_driven_system.stop_system()
            
            self.system_running = False
            logger.info("Modernized trading system stopped")
            
        except Exception as e:
            logger.error(f"Error stopping system: {e}")
    
    async def _register_components_with_event_system(self):
        """Register all components with the event system"""
        try:
            integrated_system = self.component_integrator.get_integrated_system()
            
            # Register key components as event-driven components
            components_to_register = [
                ('market_processor', integrated_system['market_processor']),
                ('decision_engine', integrated_system['decision_engine']),
                ('reward_engine', integrated_system['reward_engine']),
                ('portfolio_optimizer', integrated_system['portfolio_optimizer']),
                ('neural_manager', integrated_system['neural_manager'])
            ]
            
            for component_name, component in components_to_register:
                if component:
                    # Wrap component in event-driven wrapper
                    event_component = EventDrivenComponentWrapper(component_name, component, self.event_bus)
                    self.event_driven_system.register_component(component_name, event_component)
            
            logger.info("Components registered with event system")
            
        except Exception as e:
            logger.error(f"Error registering components with event system: {e}")
    
    async def _publish_system_event(self, event_type: EventType, data: Dict[str, Any]):
        """Publish a system-level event"""
        try:
            event = self.event_bus.create_event(
                event_type=event_type,
                source='system_orchestrator',
                data=data,
                priority=8  # High priority for system events
            )
            await self.event_bus.publish(event)
            
        except Exception as e:
            logger.error(f"Error publishing system event: {e}")
    
    def get_modernized_agent(self) -> Optional[ModernizedTradingAgent]:
        """Get the modernized trading agent"""
        return self.modernized_agent
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            integration_status = self.component_integrator.get_integration_status()
            event_system_status = self.event_driven_system.get_system_status()
            health_status = self.health_monitor.get_health_status()
            
            return {
                'system_initialized': self.system_initialized,
                'system_running': self.system_running,
                'integration_status': integration_status,
                'event_system_status': event_system_status,
                'health_status': health_status,
                'configuration_summary': self.config_manager.get_configuration_summary(),
                'modernized_agent_available': self.modernized_agent is not None
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}
    
    def get_system_health(self) -> SystemHealth:
        """Get detailed system health assessment"""
        try:
            return self.health_monitor.get_detailed_health()
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return SystemHealth(
                overall_health='critical',
                component_health={},
                event_bus_health='critical',
                integration_health='critical',
                performance_health='critical',
                issues=[f"Health check failed: {e}"],
                recommendations=["Investigate system health monitoring"]
            )
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        try:
            stats = {
                'event_bus_stats': self.event_bus.get_event_statistics(),
                'integration_stats': self.component_integrator.get_integration_status(),
                'health_stats': self.health_monitor.get_health_statistics()
            }
            
            # Add component-specific statistics
            if self.modernized_agent:
                stats['agent_stats'] = self.modernized_agent.get_agent_statistics()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting system statistics: {e}")
            return {'error': str(e)}
    
    async def handle_system_error(self, error: Exception, context: str):
        """Handle system-level errors"""
        try:
            logger.error(f"System error in {context}: {error}")
            
            # Publish error event
            await self._publish_system_event(EventType.SYSTEM_ERROR, {
                'error': str(error),
                'context': context,
                'timestamp': asyncio.get_event_loop().time()
            })
            
            # Attempt recovery based on error type
            await self._attempt_error_recovery(error, context)
            
        except Exception as e:
            logger.error(f"Error handling system error: {e}")
    
    async def _attempt_error_recovery(self, error: Exception, context: str):
        """Attempt to recover from system errors"""
        try:
            # Simple recovery strategies
            if "connection" in str(error).lower():
                logger.info("Attempting connection recovery...")
                # Could implement connection retry logic
                
            elif "memory" in str(error).lower():
                logger.info("Attempting memory cleanup...")
                # Could implement memory cleanup
                
            elif "timeout" in str(error).lower():
                logger.info("Attempting timeout recovery...")
                # Could implement timeout handling
                
            else:
                logger.warning(f"No specific recovery strategy for error: {error}")
                
        except Exception as e:
            logger.error(f"Error during recovery attempt: {e}")


class EventDrivenComponentWrapper(EventDrivenComponent):
    """Wrapper to make existing components event-driven"""
    
    def __init__(self, component_name: str, component: Any, event_bus: EventBus):
        super().__init__(component_name, event_bus)
        self.wrapped_component = component
        
        # Register relevant event handlers based on component type
        self._register_component_handlers()
    
    def _register_component_handlers(self):
        """Register event handlers based on component type"""
        try:
            component_type = type(self.wrapped_component).__name__
            
            if 'MarketData' in component_type:
                self.register_event_handler(
                    'market_data_handler',
                    self._handle_market_data_events,
                    [EventType.MARKET_DATA_RECEIVED, EventType.PRICE_CHANGE]
                )
            
            elif 'Decision' in component_type:
                self.register_event_handler(
                    'decision_handler',
                    self._handle_decision_events,
                    [EventType.TRADE_SIGNAL_GENERATED]
                )
            
            elif 'Reward' in component_type:
                self.register_event_handler(
                    'reward_handler',
                    self._handle_reward_events,
                    [EventType.TRADE_COMPLETED, EventType.SURPRISE_DETECTED]
                )
            
            elif 'Portfolio' in component_type:
                self.register_event_handler(
                    'portfolio_handler',
                    self._handle_portfolio_events,
                    [EventType.POSITION_OPENED, EventType.POSITION_CLOSED]
                )
            
            elif 'Neural' in component_type:
                self.register_event_handler(
                    'neural_handler',
                    self._handle_neural_events,
                    [EventType.NEURAL_TRAINING_COMPLETE, EventType.ARCHITECTURE_EVOLVED]
                )
            
        except Exception as e:
            logger.error(f"Error registering component handlers: {e}")
    
    def _handle_market_data_events(self, event):
        """Handle market data events"""
        try:
            if hasattr(self.wrapped_component, 'process_market_data'):
                self.wrapped_component.process_market_data(event.data)
        except Exception as e:
            logger.error(f"Error handling market data event: {e}")
    
    def _handle_decision_events(self, event):
        """Handle decision events"""
        try:
            if hasattr(self.wrapped_component, 'process_decision'):
                self.wrapped_component.process_decision(event.data)
        except Exception as e:
            logger.error(f"Error handling decision event: {e}")
    
    def _handle_reward_events(self, event):
        """Handle reward events"""
        try:
            if hasattr(self.wrapped_component, 'process_reward'):
                self.wrapped_component.process_reward(event.data)
        except Exception as e:
            logger.error(f"Error handling reward event: {e}")
    
    def _handle_portfolio_events(self, event):
        """Handle portfolio events"""
        try:
            if hasattr(self.wrapped_component, 'process_portfolio_event'):
                self.wrapped_component.process_portfolio_event(event.data)
        except Exception as e:
            logger.error(f"Error handling portfolio event: {e}")
    
    def _handle_neural_events(self, event):
        """Handle neural network events"""
        try:
            if hasattr(self.wrapped_component, 'process_neural_event'):
                self.wrapped_component.process_neural_event(event.data)
        except Exception as e:
            logger.error(f"Error handling neural event: {e}")


class SystemHealthMonitor:
    """Monitor system health and performance"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.monitoring_active = False
        self.health_data = {}
        
        # Register health monitoring handlers
        self._register_health_handlers()
    
    def _register_health_handlers(self):
        """Register health monitoring event handlers"""
        self.event_bus.register_handler(
            'health_monitor',
            self._handle_health_events,
            [EventType.SYSTEM_ERROR, EventType.COMPONENT_FAILED, EventType.PERFORMANCE_MILESTONE],
            priority=9
        )
    
    def _handle_health_events(self, event):
        """Handle health-related events"""
        try:
            if event.event_type == EventType.SYSTEM_ERROR:
                self.health_data['last_error'] = event.data
                self.health_data['error_count'] = self.health_data.get('error_count', 0) + 1
            
            elif event.event_type == EventType.COMPONENT_FAILED:
                failed_components = self.health_data.get('failed_components', [])
                failed_components.append(event.data)
                self.health_data['failed_components'] = failed_components
            
            elif event.event_type == EventType.PERFORMANCE_MILESTONE:
                self.health_data['last_milestone'] = event.data
            
        except Exception as e:
            logger.error(f"Error handling health event: {e}")
    
    async def start_monitoring(self):
        """Start health monitoring"""
        self.monitoring_active = True
        logger.info("Health monitoring started")
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring_active = False
        logger.info("Health monitoring stopped")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get basic health status"""
        return {
            'monitoring_active': self.monitoring_active,
            'error_count': self.health_data.get('error_count', 0),
            'failed_components': len(self.health_data.get('failed_components', [])),
            'last_error': self.health_data.get('last_error'),
            'last_milestone': self.health_data.get('last_milestone')
        }
    
    def get_detailed_health(self) -> SystemHealth:
        """Get detailed health assessment"""
        try:
            error_count = self.health_data.get('error_count', 0)
            failed_components = self.health_data.get('failed_components', [])
            
            # Determine overall health
            if error_count > 10 or len(failed_components) > 2:
                overall_health = 'critical'
            elif error_count > 5 or len(failed_components) > 0:
                overall_health = 'warning'
            else:
                overall_health = 'healthy'
            
            return SystemHealth(
                overall_health=overall_health,
                component_health={},  # Would be populated with actual component health
                event_bus_health='healthy' if self.monitoring_active else 'warning',
                integration_health='healthy',
                performance_health='healthy',
                issues=[],
                recommendations=[]
            )
            
        except Exception as e:
            logger.error(f"Error getting detailed health: {e}")
            return SystemHealth(
                overall_health='critical',
                component_health={},
                event_bus_health='critical',
                integration_health='critical',
                performance_health='critical',
                issues=[f"Health assessment failed: {e}"],
                recommendations=["Investigate health monitoring system"]
            )
    
    def get_health_statistics(self) -> Dict[str, Any]:
        """Get health monitoring statistics"""
        return {
            'monitoring_active': self.monitoring_active,
            'health_data': self.health_data,
            'events_processed': len(self.health_data),
            'monitoring_uptime': 0  # Would track actual uptime
        }