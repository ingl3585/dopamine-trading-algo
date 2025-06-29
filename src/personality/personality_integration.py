"""
Personality Integration Module

Connects the AI Trading Personality system with the main trading infrastructure
"""

import asyncio
import logging
import time
from typing import Dict, Optional, Any
from dataclasses import dataclass

from .trading_personality import TradingPersonality, TriggerEvent, PersonalityConfig
from .voice_synthesis import VoiceSynthesizer
from src.shared.types import Features
from src.market_analysis.data_processor import MarketData

logger = logging.getLogger(__name__)

@dataclass
class PersonalityIntegrationConfig:
    enabled: bool = True
    personality_name: str = "Rex"
    voice_enabled: bool = False
    auto_commentary: bool = True
    commentary_interval: float = 120.0  # seconds
    log_commentary: bool = True
    save_commentary_history: bool = True
    
    # LLM Configuration
    llm_model: str = "gpt-4"
    llm_api_key: str = ""
    
    # Voice Configuration
    voice_service: str = "elevenlabs"
    voice_api_key: str = ""

class PersonalityIntegration:
    """
    Integration layer between AI Trading Personality and main trading system
    """
    
    def __init__(self, config: PersonalityIntegrationConfig = None):
        self.config = config or PersonalityIntegrationConfig()
        
        if not self.config.enabled:
            logger.info("Personality system disabled")
            self.personality = None
            self.voice_synthesizer = None
            return
        
        # Initialize personality system
        personality_config = PersonalityConfig(
            personality_name=self.config.personality_name,
            voice_enabled=self.config.voice_enabled,
            llm_model=self.config.llm_model
        )
        
        self.personality = TradingPersonality(personality_config)
        
        # Initialize voice synthesis if enabled
        if self.config.voice_enabled:
            voice_config = {
                'tts_service': self.config.voice_service,
                'api_key': self.config.voice_api_key
            }
            self.voice_synthesizer = VoiceSynthesizer(voice_config)
        else:
            self.voice_synthesizer = None
        
        # State tracking
        self.last_decision_context = {}
        self.last_features = None
        self.last_market_data = None
        self.last_commentary_time = 0.0
        self.active_position_info = {}
        
        # Setup callbacks
        self._setup_callbacks()
        
        logger.info(f"Personality integration initialized for '{self.config.personality_name}'")
    
    def is_enabled(self) -> bool:
        """Check if personality system is enabled"""
        return self.config.enabled and self.personality is not None
    
    async def process_trading_decision(self, decision: Any, features: Features, 
                                     market_data: MarketData) -> Optional[str]:
        """
        Process a trading decision through the personality system
        
        Args:
            decision: Trading decision object from agent
            features: Market features including all subsystem signals
            market_data: Current market data
            
        Returns:
            Optional[str]: Commentary text if generated
        """
        
        if not self.is_enabled():
            return None
        
        try:
            # Store current context
            self.last_features = features
            self.last_market_data = market_data
            
            # Determine trigger event based on decision
            trigger_event = self._determine_trigger_event(decision)
            
            # Build comprehensive context
            context = self._build_trading_context(decision, features, market_data)
            
            # Store decision context for future reference
            self.last_decision_context = context['decision_context']
            
            # Process through personality system
            commentary_response = await self.personality.process_trading_event(trigger_event, context)
            
            if commentary_response:
                # Log commentary if enabled
                if self.config.log_commentary:
                    logger.info(f"Personality Commentary: {commentary_response.text}")
                
                # Speak commentary if voice enabled
                if self.config.voice_enabled and self.voice_synthesizer:
                    emotional_state = self.personality.get_current_emotional_state()
                    await self.voice_synthesizer.speak_commentary(
                        commentary_response.text,
                        emotional_state.primary_emotion.value,
                        emotional_state.emotional_intensity
                    )
                
                return commentary_response.text
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing trading decision through personality: {e}")
            return None
    
    async def process_trade_completion(self, trade_outcome: float, trade_context: Dict) -> Optional[str]:
        """
        Process completed trade through personality system
        
        Args:
            trade_outcome: P&L from completed trade
            trade_context: Context of the completed trade
            
        Returns:
            Optional[str]: Commentary about the trade outcome
        """
        
        if not self.is_enabled():
            return None
        
        try:
            # Update personality with performance feedback
            self.personality.update_performance_feedback(trade_outcome, trade_context)
            
            # Determine appropriate trigger event
            if trade_outcome > 0:
                if trade_context.get('exit_reason') == 'profit_target':
                    trigger_event = TriggerEvent.PROFIT_TARGET
                else:
                    trigger_event = TriggerEvent.POSITION_EXIT
            else:
                if trade_context.get('exit_reason') == 'stop_loss':
                    trigger_event = TriggerEvent.STOP_LOSS
                else:
                    trigger_event = TriggerEvent.POSITION_EXIT
            
            # Build context for trade completion
            context = {
                'trade_outcome': trade_outcome,
                'trade_context': trade_context,
                'subsystem_signals': self._get_current_subsystem_signals(),
                'market_data': self._get_current_market_context(),
                'portfolio_state': self._get_current_portfolio_state(),
                'decision_context': {
                    'decision_type': 'trade_completion',
                    'outcome': trade_outcome,
                    'exit_reason': trade_context.get('exit_reason', 'unknown')
                }
            }
            
            # Process through personality
            commentary_response = await self.personality.process_trading_event(trigger_event, context)
            
            if commentary_response:
                if self.config.log_commentary:
                    logger.info(f"Trade Completion Commentary: {commentary_response.text}")
                
                # Speak if voice enabled
                if self.config.voice_enabled and self.voice_synthesizer:
                    emotional_state = self.personality.get_current_emotional_state()
                    await self.voice_synthesizer.speak_commentary(
                        commentary_response.text,
                        emotional_state.primary_emotion.value,
                        emotional_state.emotional_intensity
                    )
                
                return commentary_response.text
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing trade completion through personality: {e}")
            return None
    
    async def process_market_event(self, event_type: str, market_context: Dict) -> Optional[str]:
        """
        Process market events through personality system
        
        Args:
            event_type: Type of market event
            market_context: Market context data
            
        Returns:
            Optional[str]: Commentary about the market event
        """
        
        if not self.is_enabled():
            return None
        
        try:
            # Map event types to trigger events
            event_mapping = {
                'market_open': TriggerEvent.MARKET_OPEN,
                'market_close': TriggerEvent.MARKET_CLOSE,
                'volatility_spike': TriggerEvent.VOLATILITY_SPIKE,
                'regime_change': TriggerEvent.REGIME_CHANGE,
                'immune_warning': TriggerEvent.IMMUNE_WARNING,
                'dopamine_spike': TriggerEvent.DOPAMINE_SPIKE
            }
            
            trigger_event = event_mapping.get(event_type, TriggerEvent.PERIODIC_UPDATE)
            
            # Build context
            context = {
                'event_type': event_type,
                'market_context': market_context,
                'subsystem_signals': self._get_current_subsystem_signals(),
                'market_data': self._get_current_market_context(),
                'portfolio_state': self._get_current_portfolio_state(),
                'decision_context': {
                    'decision_type': 'market_event',
                    'event_type': event_type
                }
            }
            
            # Process through personality
            commentary_response = await self.personality.process_trading_event(trigger_event, context)
            
            if commentary_response:
                if self.config.log_commentary:
                    logger.info(f"Market Event Commentary ({event_type}): {commentary_response.text}")
                
                return commentary_response.text
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing market event through personality: {e}")
            return None
    
    async def manual_query(self, query: str) -> Optional[str]:
        """
        Handle manual query from user
        
        Args:
            query: User's question or request
            
        Returns:
            Optional[str]: AI's response to the query
        """
        
        if not self.is_enabled():
            return "Personality system is not enabled."
        
        try:
            context = {
                'user_query': query,
                'subsystem_signals': self._get_current_subsystem_signals(),
                'market_data': self._get_current_market_context(),
                'portfolio_state': self._get_current_portfolio_state(),
                'decision_context': self.last_decision_context
            }
            
            commentary_response = await self.personality.manual_query(query, context)
            
            if commentary_response:
                if self.config.log_commentary:
                    logger.info(f"Manual Query Response: {commentary_response.text}")
                
                # Speak response if voice enabled
                if self.config.voice_enabled and self.voice_synthesizer:
                    emotional_state = self.personality.get_current_emotional_state()
                    await self.voice_synthesizer.speak_commentary(
                        commentary_response.text,
                        emotional_state.primary_emotion.value,
                        emotional_state.emotional_intensity
                    )
                
                return commentary_response.text
            
            return "I'm not sure how to respond to that right now."
            
        except Exception as e:
            logger.error(f"Error processing manual query: {e}")
            return "Sorry, I encountered an error processing your question."
    
    async def periodic_commentary(self) -> Optional[str]:
        """
        Generate periodic commentary about current market conditions
        
        Returns:
            Optional[str]: Periodic commentary if appropriate
        """
        
        if not self.is_enabled() or not self.config.auto_commentary:
            return None
        
        # Check if enough time has passed
        time_since_last = time.time() - self.last_commentary_time
        if time_since_last < self.config.commentary_interval:
            return None
        
        try:
            context = {
                'subsystem_signals': self._get_current_subsystem_signals(),
                'market_data': self._get_current_market_context(),
                'portfolio_state': self._get_current_portfolio_state(),
                'decision_context': {
                    'decision_type': 'periodic_update',
                    'time_since_last': time_since_last
                }
            }
            
            commentary_response = await self.personality.process_trading_event(
                TriggerEvent.PERIODIC_UPDATE, context
            )
            
            if commentary_response:
                self.last_commentary_time = time.time()
                
                if self.config.log_commentary:
                    logger.info(f"Periodic Commentary: {commentary_response.text}")
                
                return commentary_response.text
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating periodic commentary: {e}")
            return None
    
    def get_personality_status(self) -> Dict:
        """Get current personality system status"""
        
        if not self.is_enabled():
            return {'enabled': False}
        
        try:
            emotional_state = self.personality.get_current_emotional_state()
            personality_summary = self.personality.get_personality_summary()
            
            status = {
                'enabled': True,
                'personality_name': self.config.personality_name,
                'voice_enabled': self.config.voice_enabled,
                'current_emotion': emotional_state.primary_emotion.value,
                'emotional_metrics': {
                    'confidence': emotional_state.confidence,
                    'fear': emotional_state.fear,
                    'excitement': emotional_state.excitement,
                    'intensity': emotional_state.emotional_intensity,
                    'stability': emotional_state.emotional_stability
                },
                'recent_activity': personality_summary.get('recent_performance', {}),
                'voice_status': self.voice_synthesizer.get_voice_info() if self.voice_synthesizer else None
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting personality status: {e}")
            return {'enabled': True, 'error': str(e)}
    
    def _determine_trigger_event(self, decision: Any) -> TriggerEvent:
        """Determine appropriate trigger event from trading decision"""
        
        if hasattr(decision, 'action'):
            action = decision.action.lower()
            
            if action in ['buy', 'long']:
                return TriggerEvent.POSITION_ENTRY
            elif action in ['sell', 'short']:
                if self.active_position_info:  # Exiting existing position
                    return TriggerEvent.POSITION_EXIT
                else:  # New short position
                    return TriggerEvent.POSITION_ENTRY
            elif action == 'hold':
                return TriggerEvent.PERIODIC_UPDATE
        
        # Check for special conditions
        if hasattr(decision, 'primary_tool'):
            if 'immune' in decision.primary_tool:
                return TriggerEvent.IMMUNE_WARNING
            elif 'dopamine' in decision.primary_tool:
                return TriggerEvent.DOPAMINE_SPIKE
            elif 'dna' in decision.primary_tool:
                return TriggerEvent.DNA_PATTERN_LOCK
        
        return TriggerEvent.PERIODIC_UPDATE
    
    def _build_trading_context(self, decision: Any, features: Features, market_data: MarketData) -> Dict:
        """Build comprehensive trading context"""
        
        # Extract subsystem signals
        subsystem_signals = {
            'dna': features.dna_signal,
            'temporal': features.temporal_signal,
            'immune': features.immune_signal,
            'microstructure': features.microstructure_signal,
            'dopamine': features.dopamine_signal
        }
        
        # Extract market context
        market_context = {
            'volatility': features.volatility,
            'price_momentum': features.price_momentum,
            'volume_momentum': features.volume_momentum,
            'price_position': features.price_position,
            'time_of_day': features.time_of_day,
            'regime_confidence': features.regime_confidence,
            'trend_strength': abs(features.price_momentum),
            'volume_regime': features.volume_momentum + 0.5
        }
        
        # Extract portfolio state
        portfolio_state = {
            'unrealized_pnl': getattr(market_data, 'unrealized_pnl', 0.0),
            'daily_pnl': getattr(market_data, 'daily_pnl', 0.0),
            'account_balance': getattr(market_data, 'account_balance', 25000.0),
            'open_positions': getattr(market_data, 'open_positions', 0),
            'margin_utilization': getattr(market_data, 'margin_utilization', 0.0)
        }
        
        # Extract decision context
        decision_context = {
            'decision_type': 'trading_decision',
            'action': getattr(decision, 'action', 'unknown'),
            'confidence': getattr(decision, 'confidence', 0.5),
            'size': getattr(decision, 'size', 0.0),
            'primary_tool': getattr(decision, 'primary_tool', 'unknown'),
            'exploration': getattr(decision, 'exploration', False),
            'uncertainty_estimate': getattr(decision, 'uncertainty_estimate', 0.5)
        }
        
        return {
            'subsystem_signals': subsystem_signals,
            'market_data': market_context,
            'portfolio_state': portfolio_state,
            'decision_context': decision_context
        }
    
    def _get_current_subsystem_signals(self) -> Dict:
        """Get current subsystem signals from last known features"""
        
        if self.last_features:
            return {
                'dna': self.last_features.dna_signal,
                'temporal': self.last_features.temporal_signal,
                'immune': self.last_features.immune_signal,
                'microstructure': self.last_features.microstructure_signal,
                'dopamine': self.last_features.dopamine_signal
            }
        
        return {
            'dna': 0.0,
            'temporal': 0.0,
            'immune': 0.0,
            'microstructure': 0.0,
            'dopamine': 0.0
        }
    
    def _get_current_market_context(self) -> Dict:
        """Get current market context from last known data"""
        
        if self.last_features and self.last_market_data:
            return {
                'volatility': self.last_features.volatility,
                'price_momentum': self.last_features.price_momentum,
                'volume_momentum': self.last_features.volume_momentum,
                'price_position': self.last_features.price_position,
                'regime_confidence': self.last_features.regime_confidence,
                'current_price': self.last_market_data.prices_1m[-1] if self.last_market_data.prices_1m else 0.0
            }
        
        return {
            'volatility': 0.02,
            'price_momentum': 0.0,
            'volume_momentum': 0.0,
            'price_position': 0.5,
            'regime_confidence': 0.5,
            'current_price': 0.0
        }
    
    def _get_current_portfolio_state(self) -> Dict:
        """Get current portfolio state from last known data"""
        
        if self.last_market_data:
            return {
                'unrealized_pnl': getattr(self.last_market_data, 'unrealized_pnl', 0.0),
                'daily_pnl': getattr(self.last_market_data, 'daily_pnl', 0.0),
                'account_balance': getattr(self.last_market_data, 'account_balance', 25000.0),
                'open_positions': getattr(self.last_market_data, 'open_positions', 0),
                'margin_utilization': getattr(self.last_market_data, 'margin_utilization', 0.0)
            }
        
        return {
            'unrealized_pnl': 0.0,
            'daily_pnl': 0.0,
            'account_balance': 25000.0,
            'open_positions': 0,
            'margin_utilization': 0.0
        }
    
    def _setup_callbacks(self):
        """Setup personality system callbacks"""
        
        if not self.personality:
            return
        
        # Add commentary callback
        def on_commentary(commentary_response):
            if self.config.save_commentary_history:
                # Save to file or database
                pass
        
        # Add emotional state callback
        def on_emotional_state_change(emotional_state):
            logger.debug(f"Emotional state changed to: {emotional_state.primary_emotion.value} "
                        f"(intensity: {emotional_state.emotional_intensity:.2f})")
        
        self.personality.add_commentary_callback(on_commentary)
        self.personality.add_emotional_state_callback(on_emotional_state_change)
    
    def save_state(self):
        """Save personality state"""
        if self.personality:
            self.personality.save_state()
    
    def shutdown(self):
        """Shutdown personality integration"""
        if self.personality:
            self.personality.shutdown()
        
        logger.info("Personality integration shutdown complete")