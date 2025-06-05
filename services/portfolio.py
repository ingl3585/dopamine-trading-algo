# services/portfolio.py

import logging

log = logging.getLogger(__name__)

class Portfolio:
    """
    Simplified Portfolio class for pure ML signal generation.
    All position management, sizing, and risk management moved to NinjaScript.
    This class now only tracks position for reward calculation purposes.
    """
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.position = 0  # Track for reward calculation only
        
        log.info("Portfolio initialized in pure ML mode - all position management in NinjaScript")

    def update_position(self, pos):
        """Update current position (for reward calculation only)"""
        if pos != self.position:
            log.debug(f"Position updated: {self.position} -> {pos}")
            self.position = pos

    def get_current_position(self):
        """Get current position for logging/monitoring"""
        return self.position
    
    def get_portfolio_status(self):
        """Get basic portfolio status for monitoring"""
        return {
            'current_position': self.position,
            'mode': 'pure_ml_signal_generation',
            'position_management': 'handled_by_ninjaScript',
            'utilization_pct': 0  # Not calculated in Python anymore
        }