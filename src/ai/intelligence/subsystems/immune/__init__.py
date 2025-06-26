"""
Immune Subsystem - Evolving threat detection and antibody systems

Public Interface:
- detect_threats: Analyze market state for threat patterns
- learn_threat: Update immune system from trade outcomes
- evolve_antibodies: Evolve antibody patterns for better detection
"""

from .domain import EvolvingImmuneSystem

# Public API
def detect_threats(subsystem: EvolvingImmuneSystem, market_state):
    return subsystem.detect_threats(market_state)

def learn_threat(subsystem: EvolvingImmuneSystem, market_state, threat_level, is_bootstrap=False):
    return subsystem.learn_threat(market_state, threat_level, is_bootstrap)

def evolve_antibodies(subsystem: EvolvingImmuneSystem):
    return subsystem.evolve_antibodies()

__all__ = ['EvolvingImmuneSystem', 'detect_threats', 'learn_threat', 'evolve_antibodies']