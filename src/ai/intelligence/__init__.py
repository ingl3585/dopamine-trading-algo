"""
Intelligence Module - Coordinates all AI subsystems

Provides clean interface to the four main subsystems:
- DNA: 16-base market pattern encoding and evolution
- Temporal: FFT-based cycle detection and seasonal analysis  
- Immune: Adaptive threat detection and antibody evolution
- Microstructure: Order flow analysis and regime detection
"""

from .engine import IntelligenceEngine

__all__ = ['IntelligenceEngine']