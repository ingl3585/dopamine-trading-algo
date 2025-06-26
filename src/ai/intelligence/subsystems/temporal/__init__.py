"""
Temporal Subsystem - FFT-based cycle detection and seasonal analysis

Public Interface:
- analyze_cycles: Detect market cycles using FFT analysis
- learn_from_outcome: Update cycle performance from trade results
- get_cycle_stats: Get comprehensive cycle statistics
"""

from .domain import FFTTemporalSubsystem

# Public API
def analyze_cycles(subsystem: FFTTemporalSubsystem, prices, timestamps=None):
    return subsystem.analyze_cycles(prices, timestamps)

def learn_from_outcome(subsystem: FFTTemporalSubsystem, cycles_info, outcome):
    return subsystem.learn_from_outcome(cycles_info, outcome)

__all__ = ['FFTTemporalSubsystem', 'analyze_cycles', 'learn_from_outcome']