# event_bus.py

import logging
import asyncio
import threading
import time
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Standard event types for the trading system"""
    # Market Data Events
    MARKET_DATA_RECEIVED = "market_data_received"
    NEW_BAR_15M = "new_bar_15m"
    NEW_BAR_1H = "new_bar_1h"
    NEW_BAR_4H = "new_bar_4h"
    PRICE_CHANGE = "price_change"
    VOLATILITY_SPIKE = "volatility_spike"
    VOLUME_SURGE = "volume_surge"
    
    # Trading Events
    TRADE_SIGNAL_GENERATED = "trade_signal_generated"
    TRADE_EXECUTED = "trade_executed"
    TRADE_COMPLETED = "trade_completed"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_UPDATED = "position_updated"
    
    # System Events
    SYSTEM_STARTED = "system_started"
    SYSTEM_STOPPED = "system_stopped"
    SYSTEM_ERROR = "system_error"
    COMPONENT_INITIALIZED = "component_initialized"
    COMPONENT_FAILED = "component_failed"
    
    # Analysis Events
    REGIME_CHANGE_DETECTED = "regime_change_detected"
    TREND_CHANGE_DETECTED = "trend_change_detected"
    SURPRISE_DETECTED = "surprise_detected"
    NOVELTY_DETECTED = "novelty_detected"
    
    # Performance Events
    PERFORMANCE_MILESTONE = "performance_milestone"
    RISK_THRESHOLD_EXCEEDED = "risk_threshold_exceeded"
    DRAWDOWN_ALERT = "drawdown_alert"
    
    # Neural Events
    NEURAL_TRAINING_COMPLETE = "neural_training_complete"
    ARCHITECTURE_EVOLVED = "architecture_evolved"
    NETWORK_PRUNED = "network_pruned"
    
    # Reward Events
    REWARD_COMPUTED = "reward_computed"
    DOPAMINE_BURST = "dopamine_burst"
    LEARNING_MILESTONE = "learning_milestone"

@dataclass
class Event:
    """Event data structure"""
    event_type: EventType
    timestamp: float
    source: str
    data: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher values = higher priority
    correlation_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)

@dataclass
class EventHandler:
    """Event handler registration"""
    handler_id: str
    handler_func: Callable[[Event], Any]
    event_types: List[EventType]
    priority: int = 0
    async_handler: bool = False
    filter_func: Optional[Callable[[Event], bool]] = None

class EventBus:
    """
    Event-driven architecture implementation for the trading system.
    
    Responsibilities:
    - Route events between components
    - Handle synchronous and asynchronous event processing
    - Provide event filtering and prioritization
    - Track event statistics and performance
    - Support event correlation and tracing
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Event handling
        self.handlers: Dict[EventType, List[EventHandler]] = defaultdict(list)
        self.global_handlers: List[EventHandler] = []
        self.event_queue = asyncio.Queue(maxsize=config.get('max_event_queue_size', 1000))
        self.priority_queue = deque()
        
        # Threading and async support
        self.thread_pool = ThreadPoolExecutor(max_workers=config.get('event_thread_pool_size', 4))
        self.event_loop = None
        self.event_processing_task = None
        self.running = False
        
        # Event statistics
        self.event_stats = {
            'total_events': 0,
            'events_by_type': defaultdict(int),
            'events_by_source': defaultdict(int),
            'processing_times': deque(maxlen=1000),
            'failed_events': 0,
            'handler_errors': 0
        }
        
        # Event history for debugging
        self.event_history = deque(maxlen=config.get('event_history_size', 10000))
        
        # Event correlation tracking
        self.correlation_chains = {}
        
        logger.info("Event bus initialized")
    
    async def start(self):
        """Start the event bus"""
        try:
            if self.running:
                logger.warning("Event bus already running")
                return
            
            self.running = True
            self.event_loop = asyncio.get_event_loop()
            
            # Start event processing task
            self.event_processing_task = asyncio.create_task(self._process_events())
            
            logger.info("Event bus started")
            
        except Exception as e:
            logger.error(f"Error starting event bus: {e}")
            raise
    
    async def stop(self):
        """Stop the event bus"""
        try:
            if not self.running:
                return
            
            self.running = False
            
            # Cancel event processing task
            if self.event_processing_task:
                self.event_processing_task.cancel()
                try:
                    await self.event_processing_task
                except asyncio.CancelledError:
                    pass
            
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)
            
            logger.info("Event bus stopped")
            
        except Exception as e:
            logger.error(f"Error stopping event bus: {e}")
    
    def register_handler(self, handler_id: str, handler_func: Callable[[Event], Any],
                        event_types: Union[EventType, List[EventType]],
                        priority: int = 0, async_handler: bool = False,
                        filter_func: Optional[Callable[[Event], bool]] = None):
        """
        Register an event handler
        
        Args:
            handler_id: Unique identifier for the handler
            handler_func: Function to handle events
            event_types: Event type(s) to handle
            priority: Handler priority (higher = earlier execution)
            async_handler: Whether handler is async
            filter_func: Optional filter function
        """
        try:
            # Normalize event types to list
            if isinstance(event_types, EventType):
                event_types = [event_types]
            
            handler = EventHandler(
                handler_id=handler_id,
                handler_func=handler_func,
                event_types=event_types,
                priority=priority,
                async_handler=async_handler,
                filter_func=filter_func
            )
            
            # Register for specific event types
            for event_type in event_types:
                self.handlers[event_type].append(handler)
                # Sort by priority (higher first)
                self.handlers[event_type].sort(key=lambda h: h.priority, reverse=True)
            
            logger.info(f"Registered handler '{handler_id}' for events: {[et.value for et in event_types]}")
            
        except Exception as e:
            logger.error(f"Error registering handler: {e}")
    
    def register_global_handler(self, handler_id: str, handler_func: Callable[[Event], Any],
                               priority: int = 0, async_handler: bool = False,
                               filter_func: Optional[Callable[[Event], bool]] = None):
        """Register a global handler that receives all events"""
        try:
            handler = EventHandler(
                handler_id=handler_id,
                handler_func=handler_func,
                event_types=[],  # Empty for global
                priority=priority,
                async_handler=async_handler,
                filter_func=filter_func
            )
            
            self.global_handlers.append(handler)
            self.global_handlers.sort(key=lambda h: h.priority, reverse=True)
            
            logger.info(f"Registered global handler '{handler_id}'")
            
        except Exception as e:
            logger.error(f"Error registering global handler: {e}")
    
    def unregister_handler(self, handler_id: str):
        """Unregister a handler by ID"""
        try:
            # Remove from specific event handlers
            for event_type, handlers in self.handlers.items():
                self.handlers[event_type] = [h for h in handlers if h.handler_id != handler_id]
            
            # Remove from global handlers
            self.global_handlers = [h for h in self.global_handlers if h.handler_id != handler_id]
            
            logger.info(f"Unregistered handler '{handler_id}'")
            
        except Exception as e:
            logger.error(f"Error unregistering handler: {e}")
    
    async def publish(self, event: Event):
        """Publish an event to the bus"""
        try:
            # Add timestamp if not set
            if event.timestamp == 0:
                event.timestamp = time.time()
            
            # Update statistics
            self.event_stats['total_events'] += 1
            self.event_stats['events_by_type'][event.event_type] += 1
            self.event_stats['events_by_source'][event.source] += 1
            
            # Add to history
            self.event_history.append(event)
            
            # Handle priority events immediately
            if event.priority > 5:
                await self._handle_event_immediately(event)
            else:
                # Add to queue for processing
                try:
                    await self.event_queue.put(event)
                except asyncio.QueueFull:
                    logger.warning(f"Event queue full, dropping event: {event.event_type}")
                    self.event_stats['failed_events'] += 1
            
        except Exception as e:
            logger.error(f"Error publishing event: {e}")
            self.event_stats['failed_events'] += 1
    
    def publish_sync(self, event: Event):
        """Publish an event synchronously"""
        try:
            if self.event_loop and self.running:
                asyncio.run_coroutine_threadsafe(self.publish(event), self.event_loop)
            else:
                logger.warning("Event bus not running, cannot publish event")
                
        except Exception as e:
            logger.error(f"Error publishing event synchronously: {e}")
    
    async def _process_events(self):
        """Main event processing loop"""
        try:
            while self.running:
                try:
                    # Get next event with timeout
                    event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                    
                    # Process the event
                    await self._handle_event(event)
                    
                except asyncio.TimeoutError:
                    # Normal timeout, continue loop
                    continue
                except Exception as e:
                    logger.error(f"Error processing event: {e}")
                    
        except asyncio.CancelledError:
            logger.info("Event processing cancelled")
        except Exception as e:
            logger.error(f"Error in event processing loop: {e}")
    
    async def _handle_event(self, event: Event):
        """Handle a single event"""
        start_time = time.time()
        
        try:
            # Get handlers for this event type
            handlers = self.handlers.get(event.event_type, [])
            
            # Add global handlers
            handlers.extend(self.global_handlers)
            
            # Sort by priority
            handlers.sort(key=lambda h: h.priority, reverse=True)
            
            # Execute handlers
            for handler in handlers:
                try:
                    # Apply filter if present
                    if handler.filter_func and not handler.filter_func(event):
                        continue
                    
                    # Execute handler
                    if handler.async_handler:
                        await handler.handler_func(event)
                    else:
                        # Run sync handler in thread pool
                        await self.event_loop.run_in_executor(
                            self.thread_pool, handler.handler_func, event
                        )
                        
                except Exception as e:
                    logger.error(f"Error in handler '{handler.handler_id}': {e}")
                    self.event_stats['handler_errors'] += 1
            
            # Record processing time
            processing_time = time.time() - start_time
            self.event_stats['processing_times'].append(processing_time)
            
        except Exception as e:
            logger.error(f"Error handling event: {e}")
    
    async def _handle_event_immediately(self, event: Event):
        """Handle high-priority events immediately"""
        try:
            await self._handle_event(event)
        except Exception as e:
            logger.error(f"Error handling high-priority event: {e}")
    
    def create_event(self, event_type: EventType, source: str, 
                    data: Dict[str, Any] = None, priority: int = 0,
                    correlation_id: Optional[str] = None,
                    tags: List[str] = None) -> Event:
        """Create a new event"""
        return Event(
            event_type=event_type,
            timestamp=time.time(),
            source=source,
            data=data or {},
            priority=priority,
            correlation_id=correlation_id,
            tags=tags or []
        )
    
    def get_event_statistics(self) -> Dict[str, Any]:
        """Get event processing statistics"""
        try:
            processing_times = list(self.event_stats['processing_times'])
            
            stats = {
                'total_events': self.event_stats['total_events'],
                'failed_events': self.event_stats['failed_events'],
                'handler_errors': self.event_stats['handler_errors'],
                'events_by_type': dict(self.event_stats['events_by_type']),
                'events_by_source': dict(self.event_stats['events_by_source']),
                'queue_size': self.event_queue.qsize() if self.event_queue else 0,
                'registered_handlers': sum(len(handlers) for handlers in self.handlers.values()),
                'global_handlers': len(self.global_handlers),
                'running': self.running
            }
            
            if processing_times:
                import numpy as np
                stats['processing_performance'] = {
                    'avg_processing_time': np.mean(processing_times),
                    'max_processing_time': np.max(processing_times),
                    'min_processing_time': np.min(processing_times),
                    'events_per_second': len(processing_times) / max(1, sum(processing_times))
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting event statistics: {e}")
            return {}
    
    def get_recent_events(self, limit: int = 100, 
                         event_type: Optional[EventType] = None) -> List[Event]:
        """Get recent events"""
        try:
            events = list(self.event_history)
            
            if event_type:
                events = [e for e in events if e.event_type == event_type]
            
            return events[-limit:]
            
        except Exception as e:
            logger.error(f"Error getting recent events: {e}")
            return []


class EventDrivenComponent:
    """
    Base class for components that participate in event-driven architecture
    """
    
    def __init__(self, component_name: str, event_bus: EventBus):
        self.component_name = component_name
        self.event_bus = event_bus
        self.registered_handlers = []
        
        # Register component initialization event
        self._publish_event(EventType.COMPONENT_INITIALIZED, {
            'component_name': component_name,
            'timestamp': time.time()
        })
    
    def register_event_handler(self, handler_id: str, handler_func: Callable[[Event], Any],
                              event_types: Union[EventType, List[EventType]],
                              priority: int = 0, async_handler: bool = False):
        """Register an event handler for this component"""
        full_handler_id = f"{self.component_name}_{handler_id}"
        
        self.event_bus.register_handler(
            full_handler_id, handler_func, event_types, priority, async_handler
        )
        
        self.registered_handlers.append(full_handler_id)
    
    def _publish_event(self, event_type: EventType, data: Dict[str, Any] = None, 
                      priority: int = 0):
        """Publish an event from this component"""
        event = self.event_bus.create_event(
            event_type=event_type,
            source=self.component_name,
            data=data or {},
            priority=priority
        )
        
        self.event_bus.publish_sync(event)
    
    def cleanup_handlers(self):
        """Clean up registered handlers"""
        for handler_id in self.registered_handlers:
            self.event_bus.unregister_handler(handler_id)
        self.registered_handlers.clear()


class EventDrivenTradingSystem:
    """
    Event-driven trading system that coordinates all components
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.event_bus = EventBus(config)
        self.components = {}
        self.system_running = False
        
        # Register system-level event handlers
        self._register_system_handlers()
        
        logger.info("Event-driven trading system initialized")
    
    def _register_system_handlers(self):
        """Register system-level event handlers"""
        # System monitoring handler
        self.event_bus.register_handler(
            'system_monitor',
            self._handle_system_events,
            [EventType.SYSTEM_ERROR, EventType.COMPONENT_FAILED, EventType.RISK_THRESHOLD_EXCEEDED],
            priority=10
        )
        
        # Performance monitoring handler
        self.event_bus.register_handler(
            'performance_monitor',
            self._handle_performance_events,
            [EventType.PERFORMANCE_MILESTONE, EventType.DRAWDOWN_ALERT],
            priority=5
        )
        
        # Trading event handler
        self.event_bus.register_handler(
            'trading_monitor',
            self._handle_trading_events,
            [EventType.TRADE_EXECUTED, EventType.TRADE_COMPLETED, EventType.POSITION_CLOSED],
            priority=7
        )
    
    def _handle_system_events(self, event: Event):
        """Handle system-level events"""
        try:
            if event.event_type == EventType.SYSTEM_ERROR:
                logger.error(f"System error event: {event.data}")
                
            elif event.event_type == EventType.COMPONENT_FAILED:
                logger.error(f"Component failed: {event.data}")
                
            elif event.event_type == EventType.RISK_THRESHOLD_EXCEEDED:
                logger.warning(f"Risk threshold exceeded: {event.data}")
                
        except Exception as e:
            logger.error(f"Error handling system event: {e}")
    
    def _handle_performance_events(self, event: Event):
        """Handle performance events"""
        try:
            if event.event_type == EventType.PERFORMANCE_MILESTONE:
                logger.info(f"Performance milestone: {event.data}")
                
            elif event.event_type == EventType.DRAWDOWN_ALERT:
                logger.warning(f"Drawdown alert: {event.data}")
                
        except Exception as e:
            logger.error(f"Error handling performance event: {e}")
    
    def _handle_trading_events(self, event: Event):
        """Handle trading events"""
        try:
            if event.event_type == EventType.TRADE_EXECUTED:
                logger.info(f"Trade executed: {event.data}")
                
            elif event.event_type == EventType.TRADE_COMPLETED:
                logger.info(f"Trade completed: {event.data}")
                
            elif event.event_type == EventType.POSITION_CLOSED:
                logger.info(f"Position closed: {event.data}")
                
        except Exception as e:
            logger.error(f"Error handling trading event: {e}")
    
    def register_component(self, component_name: str, component: EventDrivenComponent):
        """Register a component with the system"""
        self.components[component_name] = component
        logger.info(f"Registered component: {component_name}")
    
    async def start_system(self):
        """Start the event-driven system"""
        try:
            # Start event bus
            await self.event_bus.start()
            
            # Publish system start event
            start_event = self.event_bus.create_event(
                EventType.SYSTEM_STARTED,
                'system',
                {'start_time': time.time(), 'components': list(self.components.keys())}
            )
            await self.event_bus.publish(start_event)
            
            self.system_running = True
            logger.info("Event-driven trading system started")
            
        except Exception as e:
            logger.error(f"Error starting system: {e}")
            raise
    
    async def stop_system(self):
        """Stop the event-driven system"""
        try:
            # Publish system stop event
            stop_event = self.event_bus.create_event(
                EventType.SYSTEM_STOPPED,
                'system',
                {'stop_time': time.time()}
            )
            await self.event_bus.publish(stop_event)
            
            # Clean up components
            for component in self.components.values():
                if hasattr(component, 'cleanup_handlers'):
                    component.cleanup_handlers()
            
            # Stop event bus
            await self.event_bus.stop()
            
            self.system_running = False
            logger.info("Event-driven trading system stopped")
            
        except Exception as e:
            logger.error(f"Error stopping system: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            'system_running': self.system_running,
            'components': list(self.components.keys()),
            'event_bus_stats': self.event_bus.get_event_statistics()
        }