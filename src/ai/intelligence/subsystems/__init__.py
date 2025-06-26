"""
AI Subsystems - Modular intelligence components

Each subsystem is a bounded context implementing sophisticated AI per prompt.txt:
- DNA: 16-base encoding, genetic breeding, pattern evolution
- Temporal: FFT cycle detection, interference modeling, seasonal analysis
- Immune: Adaptive antibodies, T-cell memory, threat evolution tracking  
- Microstructure: Smart money detection, regime classification, order flow analysis
"""

# Import subsystem domains
from .dna.domain import DNASubsystem
from .temporal.domain import FFTTemporalSubsystem  
from .immune.domain import EvolvingImmuneSystem
from .microstructure.domain import MarketMicrostructureEngine

# Public APIs for each subsystem
from .dna import encode_market_state as dna_encode, analyze_sequence as dna_analyze
from .temporal import analyze_cycles as temporal_analyze  
from .immune import detect_threats as immune_detect, evolve_antibodies as immune_evolve
from .microstructure import analyze_order_flow as micro_analyze_flow

__all__ = [
    # Domain classes
    'DNASubsystem', 'FFTTemporalSubsystem', 'EvolvingImmuneSystem', 'MarketMicrostructureEngine',
    # Public APIs  
    'dna_encode', 'dna_analyze', 'temporal_analyze', 'immune_detect', 'immune_evolve', 'micro_analyze_flow'
]