"""
AI Trading Personality System

A revolutionary conversational layer for sophisticated trading AI that provides
human-relatable commentary, emotional states, and decision explanations.
"""

from .trading_personality import TradingPersonality
from .emotional_engine import EmotionalStateEngine
from .personality_memory import PersonalityMemory
from .llm_client import LLMClient

__all__ = [
    'TradingPersonality',
    'EmotionalStateEngine', 
    'PersonalityMemory',
    'LLMClient'
]