"""
Risk Domain - Risk management and portfolio optimization

Public Interface:
- assess_risk: Assess trade risk
- size_position: Calculate optimal position size
- manage_portfolio: Portfolio management and tracking
"""

from .management.service import RiskManagementService
from .portfolio.manager import PortfolioManager

def create_risk_manager(config):
    """Factory to create configured risk manager"""
    return RiskManagementService(config)

__all__ = ['RiskManagementService', 'PortfolioManager', 'create_risk_manager']