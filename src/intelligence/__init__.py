"""
Intelligence Domain - AI/ML components and pattern recognition

Public Interface:
- create_intelligence_engine: Factory to create configured intelligence engine
- analyze_market: Analyze market with all AI subsystems
- learn_from_outcome: Learn from trading outcomes
"""

from .intelligence_engine import IntelligenceEngine
from .subsystems import DNASubsystem, FFTTemporalSubsystem, EvolvingImmuneSystem

def create_intelligence_engine(config):
    """Factory to create configured intelligence engine"""
    return IntelligenceEngine(config)

__all__ = ['IntelligenceEngine', 'DNASubsystem', 'FFTTemporalSubsystem', 'EvolvingImmuneSystem', 'create_intelligence_engine']