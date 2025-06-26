"""
AI Domain - Artificial Intelligence for Trading

Public Interface:
- IntelligenceEngine: Main orchestrator for all AI subsystems
- TradingAgent: RL agent for making trading decisions  
- Neural architectures: Advanced neural networks for pattern recognition
"""

from .intelligence.engine import IntelligenceEngine

# Clean public API for the AI domain
def create_intelligence_engine():
    """Factory method to create configured intelligence engine"""
    return IntelligenceEngine()

__all__ = ['IntelligenceEngine', 'create_intelligence_engine']