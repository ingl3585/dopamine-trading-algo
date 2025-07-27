"""
Risk Domain - Risk management and portfolio optimization

Public Interface:
- assess_risk: Assess trade risk  
- validate_order: Validate and size orders
- process_trade_outcome: Learn from trade outcomes
- get_risk_summary: Get comprehensive risk metrics

Consolidated risk management with advanced features including:
- Monte Carlo analysis
- Tail risk detection
- Adaptive position sizing
- Risk learning from outcomes
"""

from .risk_manager import RiskManager, Order
from .advanced_risk_manager import AdvancedRiskManager, RiskScenario, TailRiskMetrics
from .risk_engine import RiskLearningEngine, RiskEvent
from ..portfolio.portfolio_manager import PortfolioManager

def create_risk_manager(portfolio, meta_learner, agent=None):
    """
    Factory to create configured consolidated risk manager
    
    Args:
        portfolio: Portfolio management component
        meta_learner: Meta learning component for risk adaptation
        agent: Optional trading agent reference for feedback
        
    Returns:
        RiskManager: Consolidated risk management system
    """
    return RiskManager(portfolio, meta_learner, agent)

# Backward compatibility alias
RiskManagementService = RiskManager

__all__ = [
    'RiskManager', 'Order', 'AdvancedRiskManager', 'RiskLearningEngine', 
    'RiskScenario', 'TailRiskMetrics', 'RiskEvent', 'PortfolioManager',
    'create_risk_manager', 'RiskManagementService'  # Backward compatibility
]