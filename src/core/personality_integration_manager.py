# personality_integration_manager.py

import logging
import time
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

from src.data_models.trading_domain_models import MarketData

logger = logging.getLogger(__name__)

# Handle optional personality imports
try:
    from src.personality.config_manager import PersonalityConfigManager
    from src.personality.trading_personality import TradingPersonality, TriggerEvent
    from src.personality.personality_integration import PersonalityIntegration, PersonalityIntegrationConfig
    PERSONALITY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Personality system not available: {e}")
    PERSONALITY_AVAILABLE = False
    PersonalityConfigManager = None
    TradingPersonality = None
    TriggerEvent = None
    PersonalityIntegration = None
    PersonalityIntegrationConfig = None

@dataclass
class PersonalityStats:
    """Statistics for personality system usage"""
    commentary_generated: int = 0
    commentary_failures: int = 0
    last_commentary_time: float = 0.0
    system_initialized: bool = False
    initialization_attempts: int = 0

class PersonalityIntegrationManager:
    """
    Manages AI Trading Personality system integration.
    
    Responsibilities:
    - Initialize personality system if available
    - Handle personality configuration
    - Generate trading commentary
    - Manage personality triggers and events
    - Track personality system performance
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.stats = PersonalityStats()
        
        # Personality system components
        self.personality = None
        self.personality_integration = None
        self.personality_config_manager = None
        
        # Commentary management
        self.commentary_callbacks = []
        self.min_commentary_interval = 30.0  # 30 seconds minimum between commentary
        
        # Initialize personality system if available
        self._initialize_personality_system()
        
        logger.info("Personality integration manager initialized")
    
    def _initialize_personality_system(self):
        """Initialize the personality system if available"""
        if not PERSONALITY_AVAILABLE:
            logger.info("Personality system not available - skipping initialization")
            return
        
        try:
            self.stats.initialization_attempts += 1
            
            # Initialize personality configuration manager
            self.personality_config_manager = PersonalityConfigManager()
            integration_config = self.personality_config_manager.get_integration_config()
            
            if not integration_config.enabled:
                logger.info("AI Trading Personality disabled in configuration")
                return
            
            # Create enhanced personality integration
            integration_config_obj = PersonalityIntegrationConfig(
                enabled=True,
                personality_name=integration_config.personality_name,
                auto_commentary=integration_config.auto_commentary,
                llm_model=integration_config.llm_model,
                llm_api_key=integration_config.llm_api_key
            )
            
            self.personality_integration = PersonalityIntegration(integration_config_obj)
            self.personality = self.personality_integration.personality
            
            self.stats.system_initialized = True
            logger.info(f"Enhanced AI Trading Personality '{integration_config.personality_name}' initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI Trading Personality: {e}")
            self.personality = None
            self.personality_integration = None
            self.personality_config_manager = None
    
    def is_available(self) -> bool:
        """Check if personality system is available and initialized"""
        return PERSONALITY_AVAILABLE and self.stats.system_initialized
    
    def register_commentary_callback(self, callback: Callable[[str], None]):
        """
        Register a callback to handle generated commentary
        
        Args:
            callback: Function to call with generated commentary
        """
        self.commentary_callbacks.append(callback)
        logger.info("Commentary callback registered")
    
    def generate_initial_commentary(self, market_data: Optional[MarketData] = None):
        """
        Generate initial personality commentary after system startup
        
        Args:
            market_data: Optional market data for context
        """
        if not self.is_available():
            return
        
        try:
            if self.personality_integration:
                # Generate initial commentary about system readiness
                context = {
                    'event': 'system_startup',
                    'timestamp': time.time(),
                    'ready_for_trading': True
                }
                
                if market_data:
                    context.update({
                        'price': market_data.price,
                        'account_balance': market_data.account_balance,
                        'daily_pnl': market_data.daily_pnl
                    })
                
                commentary = self.personality_integration.generate_commentary(
                    TriggerEvent.SYSTEM_START,
                    context
                )
                
                if commentary:
                    self._handle_commentary(commentary)
                    logger.info("Initial personality commentary generated")
                
        except Exception as e:
            logger.error(f"Failed to generate initial commentary: {e}")
            self.stats.commentary_failures += 1
    
    def generate_15m_commentary(self, market_data: MarketData):
        """
        Generate commentary for 15-minute bar completion
        
        Args:
            market_data: Current market data
        """
        if not self.is_available():
            return
        
        # Check minimum interval
        current_time = time.time()
        if current_time - self.stats.last_commentary_time < self.min_commentary_interval:
            return
        
        try:
            context = {
                'event': '15m_bar_completion',
                'timestamp': current_time,
                'price': market_data.price,
                'account_balance': market_data.account_balance,
                'daily_pnl': market_data.daily_pnl,
                'volume': getattr(market_data, 'volume', 0)
            }
            
            commentary = self.personality_integration.generate_commentary(
                TriggerEvent.BAR_COMPLETION,
                context
            )
            
            if commentary:
                self._handle_commentary(commentary)
                self.stats.last_commentary_time = current_time
                logger.info("15m bar commentary generated")
                
        except Exception as e:
            logger.error(f"Failed to generate 15m commentary: {e}")
            self.stats.commentary_failures += 1
    
    def generate_trade_commentary(self, trade_data: Dict[str, Any]):
        """
        Generate commentary for trade events
        
        Args:
            trade_data: Trade information
        """
        if not self.is_available():
            return
        
        try:
            context = {
                'event': 'trade_execution',
                'timestamp': time.time(),
                **trade_data
            }
            
            commentary = self.personality_integration.generate_commentary(
                TriggerEvent.TRADE_EXECUTION,
                context
            )
            
            if commentary:
                self._handle_commentary(commentary)
                logger.info("Trade commentary generated")
                
        except Exception as e:
            logger.error(f"Failed to generate trade commentary: {e}")
            self.stats.commentary_failures += 1
    
    def generate_market_event_commentary(self, event_type: str, event_data: Dict[str, Any]):
        """
        Generate commentary for market events
        
        Args:
            event_type: Type of market event
            event_data: Event data
        """
        if not self.is_available():
            return
        
        try:
            context = {
                'event': event_type,
                'timestamp': time.time(),
                **event_data
            }
            
            # Map event types to personality triggers
            trigger_map = {
                'volatility_spike': TriggerEvent.VOLATILITY_SPIKE,
                'regime_change': TriggerEvent.REGIME_CHANGE,
                'significant_move': TriggerEvent.SIGNIFICANT_MOVE,
                'account_change': TriggerEvent.ACCOUNT_CHANGE
            }
            
            trigger = trigger_map.get(event_type, TriggerEvent.MARKET_EVENT)
            
            commentary = self.personality_integration.generate_commentary(trigger, context)
            
            if commentary:
                self._handle_commentary(commentary)
                logger.info(f"Market event commentary generated: {event_type}")
                
        except Exception as e:
            logger.error(f"Failed to generate market event commentary: {e}")
            self.stats.commentary_failures += 1
    
    def _handle_commentary(self, commentary: str):
        """
        Handle generated commentary by calling registered callbacks
        
        Args:
            commentary: Generated commentary text
        """
        self.stats.commentary_generated += 1
        
        # Call all registered callbacks
        for callback in self.commentary_callbacks:
            try:
                callback(commentary)
            except Exception as e:
                logger.error(f"Error in commentary callback: {e}")
        
        # Also log the commentary
        logger.info(f"AI Commentary: {commentary}")
    
    def get_personality_stats(self) -> Dict[str, Any]:
        """Get personality system statistics"""
        return {
            'available': self.is_available(),
            'initialized': self.stats.system_initialized,
            'initialization_attempts': self.stats.initialization_attempts,
            'commentary_generated': self.stats.commentary_generated,
            'commentary_failures': self.stats.commentary_failures,
            'success_rate': (
                self.stats.commentary_generated / 
                max(1, self.stats.commentary_generated + self.stats.commentary_failures)
            ),
            'last_commentary_time': self.stats.last_commentary_time,
            'seconds_since_last': time.time() - self.stats.last_commentary_time,
            'personality_name': (
                self.personality_config_manager.get_personality_config().personality_name
                if self.personality_config_manager else None
            )
        }
    
    def get_available_personalities(self) -> Dict[str, str]:
        """Get list of available personality configurations"""
        if not self.personality_config_manager:
            return {}
        
        return self.personality_config_manager.get_available_personalities()
    
    def reinitialize_personality(self):
        """Reinitialize the personality system"""
        logger.info("Reinitializing personality system...")
        
        # Reset state
        self.personality = None
        self.personality_integration = None
        self.personality_config_manager = None
        self.stats.system_initialized = False
        
        # Reinitialize
        self._initialize_personality_system()
    
    def shutdown(self):
        """Shutdown personality integration manager"""
        try:
            if self.personality_integration:
                # Perform any necessary cleanup
                pass
            
            logger.info("Personality integration manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during personality manager shutdown: {e}")
    
    def get_personality_summary(self) -> Dict[str, Any]:
        """Get summary of personality integration manager state"""
        return {
            'personality_available': PERSONALITY_AVAILABLE,
            'system_initialized': self.stats.system_initialized,
            'commentary_callbacks': len(self.commentary_callbacks),
            'stats': self.get_personality_stats(),
            'available_personalities': self.get_available_personalities()
        }