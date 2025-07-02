"""
AI Trading Personality - Main Orchestration Class

The personality and commentary system for the sophisticated trading AI
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass
import numpy as np

from .emotional_engine import EmotionalStateEngine, EmotionalMetrics, EmotionalState
from .llm_client import LLMClient, CommentaryRequest, CommentaryResponse, CommentaryStyle, CommentaryTone
from .personality_memory import PersonalityMemory

logger = logging.getLogger(__name__)

class TriggerEvent(Enum):
    SYSTEM_START = "system_start"
    POSITION_ENTRY = "position_entry"
    POSITION_EXIT = "position_exit"
    MARKET_OPEN = "market_open"
    MARKET_CLOSE = "market_close"
    VOLATILITY_SPIKE = "volatility_spike"
    REGIME_CHANGE = "regime_change"
    PROFIT_TARGET = "profit_target"
    STOP_LOSS = "stop_loss"
    WINNING_STREAK = "winning_streak"
    LOSING_STREAK = "losing_streak"
    DRAWDOWN = "drawdown"
    NEW_HIGH = "new_high"
    SUBSYSTEM_CONFLICT = "subsystem_conflict"
    DOPAMINE_SPIKE = "dopamine_spike"
    IMMUNE_WARNING = "immune_warning"
    DNA_PATTERN_LOCK = "dna_pattern_lock"
    TEMPORAL_CYCLE = "temporal_cycle"
    MICROSTRUCTURE_ANOMALY = "microstructure_anomaly"
    MANUAL_QUERY = "manual_query"
    PERIODIC_UPDATE = "periodic_update"

@dataclass
class PersonalityConfig:
    personality_name: str = "Alex"
    base_confidence: float = 0.6
    emotional_sensitivity: float = 0.8
    memory_weight: float = 0.3
    consistency_preference: float = 0.8
    default_style: CommentaryStyle = CommentaryStyle.ANALYTICAL
    default_tone: CommentaryTone = CommentaryTone.PROFESSIONAL
    max_commentary_length: int = 10000
    min_commentary_interval: float = 30.0  # seconds
    
    # LLM Configuration
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 8000
    llm_base_url: str = "http://localhost:11434"
    llm_api_key: str = ""
    mock_llm: bool = False

class TradingPersonality:
    """
    Main AI Trading Personality class that orchestrates emotional states,
    memory, and LLM-based commentary generation
    """
    
    def __init__(self, config: PersonalityConfig = None, 
                 memory_file: str = "data/personality_memory.json"):
        
        self.config = config or PersonalityConfig()
        
        # Core components
        self.emotional_engine = EmotionalStateEngine({
            'base_confidence': self.config.base_confidence,
            'fear_sensitivity': self.config.emotional_sensitivity,
            'emotional_dampening': 0.7
        })
        
        self.llm_client = LLMClient({
            'model_name': self.config.llm_model,
            'temperature': self.config.llm_temperature,
            'max_tokens': self.config.llm_max_tokens,
            'personality_name': self.config.personality_name,
            'base_url': getattr(self.config, 'llm_base_url', 'http://localhost:11434'),
            'api_key': getattr(self.config, 'llm_api_key', ''),
            'mock_mode': getattr(self.config, 'mock_llm', False)
        })
        
        self.memory = PersonalityMemory(memory_file=memory_file)
        
        # State tracking
        self.current_emotional_state = EmotionalMetrics()
        self.last_commentary_time = 0.0
        self.current_trading_context = {}
        self.active_triggers = set()
        
        # Commentary tracking
        self.commentary_history = []
        self.performance_feedback = []
        
        # Event callbacks
        self.commentary_callbacks: List[Callable[[CommentaryResponse], None]] = []
        self.emotional_state_callbacks: List[Callable[[EmotionalMetrics], None]] = []
        
        # Voice synthesis (placeholder for future implementation)
        
        logger.info(f"AI Trading Personality '{self.config.personality_name}' initialized")
    
    async def process_trading_event(self, event: TriggerEvent, context: Dict) -> Optional[CommentaryResponse]:
        """
        Main entry point for processing trading events and generating commentary
        
        Args:
            event: Type of trading event that occurred
            context: Complete trading context including:
                - subsystem_signals: Dict of all 5 subsystem signals
                - market_data: Current market conditions
                - portfolio_state: Positions, P&L, performance
                - decision_context: Recent decisions and reasoning
        
        Returns:
            CommentaryResponse: Generated commentary with metadata
        """
        
        try:
            # Update current context
            self.current_trading_context = context.copy()
            
            # Check if we should generate commentary
            if not self._should_generate_commentary(event):
                return None
            
            # Aggregate comprehensive personality context
            personality_context = self._aggregate_personality_context(context)
            
            # Update emotional state
            emotional_state = self.emotional_engine.update_emotional_state(personality_context)
            self.current_emotional_state = emotional_state
            
            # Get memory guidance
            memory_guidance = self.memory.get_contextual_guidance(personality_context)
            
            # Determine commentary style and tone
            style, tone = self._determine_commentary_style(emotional_state, event, context)
            
            # Create commentary request with enhanced context
            commentary_request = CommentaryRequest(
                trigger_event=event.value,
                market_context=context.get('market_data', {}),
                emotional_context=self.emotional_engine.get_emotional_context_for_llm(),
                subsystem_context=context.get('subsystem_signals', {}),
                portfolio_context=context.get('portfolio_state', {}),
                style=style,
                tone=tone,
                max_length=self.config.max_commentary_length,
                urgency=self._calculate_urgency(event, emotional_state),
                context_data=context  # Pass full enhanced context including agent insights
            )
            
            # Generate commentary
            commentary_response = await self.llm_client.generate_commentary(commentary_request)
            
            # Add memory entry
            memory_id = self.memory.add_memory(
                event_type=event.value,
                emotional_state=emotional_state.primary_emotion.value,
                market_context=context.get('market_data', {}),
                decision_context=context.get('decision_context', {}),
                commentary=commentary_response.text,
                confidence=commentary_response.confidence,
                key_themes=commentary_response.key_themes
            )
            
            # Store commentary with memory reference
            commentary_response.memory_id = memory_id
            self.commentary_history.append(commentary_response)
            
            # Update tracking
            self.last_commentary_time = time.time()
            self._notify_callbacks(commentary_response, emotional_state)
            
            
            # Commentary will be logged by the trading system
            
            return commentary_response
            
        except Exception as e:
            logger.error(f"Error processing trading event {event.value}: {e}")
            return self._create_fallback_commentary(event, context)
    
    def update_performance_feedback(self, trade_outcome: float, trade_context: Dict):
        """
        Update personality with actual trading performance feedback
        
        Args:
            trade_outcome: Actual P&L from completed trade
            trade_context: Context of the trade that completed
        """
        
        try:
            # Find matching memory entry
            trade_timestamp = trade_context.get('timestamp', time.time())
            
            # Update memory with outcome
            for commentary in self.commentary_history[-10:]:  # Check recent commentary
                commentary_time = getattr(commentary, 'timestamp', 0)
                if abs(commentary_time - trade_timestamp) < 300:  # Within 5 minutes
                    if hasattr(commentary, 'memory_id'):
                        self.memory.update_memory_outcome(commentary.memory_id, trade_outcome)
                        break
            
            # Track performance feedback
            self.performance_feedback.append({
                'timestamp': time.time(),
                'outcome': trade_outcome,
                'context': trade_context,
                'emotional_state': self.current_emotional_state.primary_emotion.value
            })
            
            # Keep only recent feedback
            if len(self.performance_feedback) > 100:
                self.performance_feedback = self.performance_feedback[-100:]
            
            logger.debug(f"Updated performance feedback: {trade_outcome}")
            
        except Exception as e:
            logger.error(f"Error updating performance feedback: {e}")
    
    def get_current_emotional_state(self) -> EmotionalMetrics:
        """Get current emotional state"""
        return self.current_emotional_state
    
    def get_personality_summary(self) -> Dict:
        """Get comprehensive personality summary"""
        
        emotional_description = self.emotional_engine.get_emotional_description()
        memory_stats = self.memory.get_memory_stats()
        
        # Analyze recent performance
        recent_commentary = self.commentary_history[-10:] if self.commentary_history else []
        
        avg_confidence = np.mean([c.confidence for c in recent_commentary]) if recent_commentary else 0.5
        common_themes = {}
        for commentary in recent_commentary:
            for theme in commentary.key_themes:
                common_themes[theme] = common_themes.get(theme, 0) + 1
        
        return {
            'personality_name': self.config.personality_name,
            'current_emotional_state': emotional_description,
            'emotional_metrics': {
                'confidence': self.current_emotional_state.confidence,
                'fear': self.current_emotional_state.fear,
                'excitement': self.current_emotional_state.excitement,
                'confusion': self.current_emotional_state.confusion,
                'optimism': self.current_emotional_state.optimism,
                'primary_emotion': self.current_emotional_state.primary_emotion.value,
                'emotional_intensity': self.current_emotional_state.emotional_intensity,
                'emotional_stability': self.current_emotional_state.emotional_stability
            },
            'recent_performance': {
                'commentary_count': len(recent_commentary),
                'avg_confidence': avg_confidence,
                'common_themes': sorted(common_themes.items(), key=lambda x: x[1], reverse=True)[:5],
                'performance_feedback_count': len(self.performance_feedback)
            },
            'memory_stats': memory_stats,
            'personality_traits': {
                'base_confidence': self.config.base_confidence,
                'emotional_sensitivity': self.config.emotional_sensitivity,
                'consistency_preference': self.config.consistency_preference
            }
        }
    
    async def manual_query(self, query: str, context: Dict = None) -> CommentaryResponse:
        """
        Handle manual queries from user
        
        Args:
            query: User's question or request
            context: Optional additional context
        
        Returns:
            CommentaryResponse: AI's response to the query
        """
        
        # Use current context if none provided
        if context is None:
            context = self.current_trading_context
        
        # Add query to context
        enhanced_context = context.copy()
        enhanced_context['user_query'] = query
        
        # Process as manual query event
        return await self.process_trading_event(TriggerEvent.MANUAL_QUERY, enhanced_context)
    
    def add_commentary_callback(self, callback: Callable[[CommentaryResponse], None]):
        """Add callback for when commentary is generated"""
        self.commentary_callbacks.append(callback)
    
    def add_emotional_state_callback(self, callback: Callable[[EmotionalMetrics], None]):
        """Add callback for when emotional state changes"""
        self.emotional_state_callbacks.append(callback)
    
    def _aggregate_personality_context(self, trading_context: Dict) -> Dict:
        """Aggregate all context needed for personality processing"""
        
        # Extract core components
        subsystem_signals = trading_context.get('subsystem_signals', {})
        market_data = trading_context.get('market_data', {})
        portfolio_state = trading_context.get('portfolio_state', {})
        decision_context = trading_context.get('decision_context', {})
        
        # Build comprehensive context
        personality_context = {
            'subsystem_signals': subsystem_signals,
            'portfolio_state': portfolio_state,
            'market_context': market_data,
            'recent_performance': self._get_recent_performance(),
            'system_confidence': decision_context.get('confidence', 0.5),
            'decision_context': decision_context
        }
        
        return personality_context
    
    def _get_recent_performance(self) -> List[float]:
        """Get recent trading performance for emotional calculation"""
        
        if not self.performance_feedback:
            return []
        
        # Return recent P&L outcomes
        return [feedback['outcome'] for feedback in self.performance_feedback[-10:]]
    
    def _should_generate_commentary(self, event: TriggerEvent) -> bool:
        """Determine if commentary should be generated for this event"""
        
        # Always generate for high-priority events
        high_priority_events = {
            TriggerEvent.POSITION_ENTRY,
            TriggerEvent.POSITION_EXIT,
            TriggerEvent.STOP_LOSS,
            TriggerEvent.PROFIT_TARGET,
            TriggerEvent.IMMUNE_WARNING,
            TriggerEvent.MANUAL_QUERY
        }
        
        if event in high_priority_events:
            return True
        
        # Rate limiting for lower priority events
        time_since_last = time.time() - self.last_commentary_time
        if time_since_last < self.config.min_commentary_interval:
            return False
        
        # Generate for significant emotional state changes
        if self.current_emotional_state.emotional_intensity > 0.7:
            return True
        
        # Generate periodically for ongoing situations
        periodic_events = {
            TriggerEvent.PERIODIC_UPDATE,
            TriggerEvent.DOPAMINE_SPIKE,
            TriggerEvent.DNA_PATTERN_LOCK
        }
        
        if event in periodic_events and time_since_last > 120:  # 2 minutes
            return True
        
        return False
    
    def _determine_commentary_style(self, emotional_state: EmotionalMetrics, 
                                  event: TriggerEvent, context: Dict) -> tuple:
        """Determine appropriate commentary style and tone"""
        
        # Default style based on emotional state
        style_mapping = {
            EmotionalState.CONFIDENT: CommentaryStyle.CONFIDENT,
            EmotionalState.FEARFUL: CommentaryStyle.CAUTIOUS,
            EmotionalState.EXCITED: CommentaryStyle.EXCITED,
            EmotionalState.CONFUSED: CommentaryStyle.ANALYTICAL,
            EmotionalState.ANALYTICAL: CommentaryStyle.ANALYTICAL,
            EmotionalState.CAUTIOUS: CommentaryStyle.CAUTIOUS,
            EmotionalState.DEFENSIVE: CommentaryStyle.CAUTIOUS,
            EmotionalState.OPTIMISTIC: CommentaryStyle.CONFIDENT
        }
        
        style = style_mapping.get(emotional_state.primary_emotion, CommentaryStyle.ANALYTICAL)
        
        # Override style for specific events
        event_style_overrides = {
            TriggerEvent.STOP_LOSS: CommentaryStyle.REFLECTIVE,
            TriggerEvent.PROFIT_TARGET: CommentaryStyle.EXCITED,
            TriggerEvent.IMMUNE_WARNING: CommentaryStyle.CAUTIOUS,
            TriggerEvent.MANUAL_QUERY: CommentaryStyle.ANALYTICAL
        }
        
        if event in event_style_overrides:
            style = event_style_overrides[event]
        
        # Determine tone based on context
        tone = CommentaryTone.PROFESSIONAL
        
        # More emotional tone for high-intensity situations
        if emotional_state.emotional_intensity > 0.7:
            tone = CommentaryTone.EMOTIONAL
        
        # Technical tone for analysis-heavy situations
        elif emotional_state.primary_emotion == EmotionalState.ANALYTICAL:
            tone = CommentaryTone.TECHNICAL
        
        # Casual tone for routine updates
        elif event == TriggerEvent.PERIODIC_UPDATE:
            tone = CommentaryTone.CASUAL
        
        return style, tone
    
    def _calculate_urgency(self, event: TriggerEvent, emotional_state: EmotionalMetrics) -> float:
        """Calculate urgency level for commentary generation"""
        
        # Base urgency by event type
        event_urgency = {
            TriggerEvent.STOP_LOSS: 0.9,
            TriggerEvent.IMMUNE_WARNING: 0.8,
            TriggerEvent.POSITION_ENTRY: 0.7,
            TriggerEvent.POSITION_EXIT: 0.7,
            TriggerEvent.VOLATILITY_SPIKE: 0.6,
            TriggerEvent.PROFIT_TARGET: 0.5,
            TriggerEvent.DOPAMINE_SPIKE: 0.4,
            TriggerEvent.PERIODIC_UPDATE: 0.2,
            TriggerEvent.MANUAL_QUERY: 0.6
        }
        
        base_urgency = event_urgency.get(event, 0.3)
        
        # Adjust based on emotional intensity
        emotional_urgency = emotional_state.emotional_intensity * 0.3
        
        # Adjust based on fear level
        fear_urgency = emotional_state.fear * 0.2
        
        total_urgency = min(1.0, base_urgency + emotional_urgency + fear_urgency)
        
        return total_urgency
    
    def _notify_callbacks(self, commentary: CommentaryResponse, emotional_state: EmotionalMetrics):
        """Notify registered callbacks"""
        
        try:
            # Commentary callbacks
            for callback in self.commentary_callbacks:
                try:
                    callback(commentary)
                except Exception as e:
                    logger.error(f"Error in commentary callback: {e}")
            
            # Emotional state callbacks
            for callback in self.emotional_state_callbacks:
                try:
                    callback(emotional_state)
                except Exception as e:
                    logger.error(f"Error in emotional state callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error notifying callbacks: {e}")
    
    def _create_fallback_commentary(self, event: TriggerEvent, context: Dict) -> CommentaryResponse:
        """Create fallback commentary when main generation fails"""
        
        fallback_texts = {
            TriggerEvent.POSITION_ENTRY: "Taking a position based on current analysis.",
            TriggerEvent.POSITION_EXIT: "Closing position as planned.",
            TriggerEvent.STOP_LOSS: "Stop loss hit - managing risk as designed.",
            TriggerEvent.PROFIT_TARGET: "Profit target reached - taking gains.",
            TriggerEvent.IMMUNE_WARNING: "Risk systems showing caution signals.",
            TriggerEvent.MANUAL_QUERY: "Let me analyze the current situation.",
            TriggerEvent.PERIODIC_UPDATE: "Monitoring market conditions."
        }
        
        fallback_text = fallback_texts.get(event, "Processing current market conditions.")
        
        return CommentaryResponse(
            text=fallback_text,
            confidence=0.3,
            emotional_intensity=0.2,
            key_themes=['fallback'],
            follow_up_suggested=False
        )
    
    
    def save_state(self):
        """Save personality state"""
        try:
            self.memory.save_memory()
            logger.info("Personality state saved")
        except Exception as e:
            logger.error(f"Error saving personality state: {e}")
    
    def shutdown(self):
        """Shutdown personality system gracefully"""
        try:
            # Save current state
            self.save_state()
            
            # Clear callbacks
            self.commentary_callbacks.clear()
            self.emotional_state_callbacks.clear()
            
            logger.info(f"AI Trading Personality '{self.config.personality_name}' shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during personality shutdown: {e}")
    
    # Context manager support
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()