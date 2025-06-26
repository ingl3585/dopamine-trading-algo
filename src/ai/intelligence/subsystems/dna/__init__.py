"""
DNA Subsystem - 16-base market pattern encoding and evolution

Public Interface:
- encode_market_state: Convert market data to DNA sequence
- analyze_sequence: Get signal strength from DNA pattern
- learn_from_outcome: Update DNA patterns from trade results
"""

from .domain import DNASubsystem

# Public API - Clean interface for external use
def encode_market_state(subsystem: DNASubsystem, prices, volumes, volatility=None, momentum=None):
    return subsystem.encode_market_state(prices, volumes, volatility, momentum)

def analyze_sequence(subsystem: DNASubsystem, sequence: str):
    return subsystem.analyze_sequence(sequence)

def learn_from_outcome(subsystem: DNASubsystem, sequence: str, outcome: float):
    return subsystem.learn_from_outcome(sequence, outcome)

__all__ = ['DNASubsystem', 'encode_market_state', 'analyze_sequence', 'learn_from_outcome']