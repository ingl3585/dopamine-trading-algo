# advanced_risk.py

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import time
import json

# Import dependency manager for clean dependency handling
try:
    from ..core.dependency_manager import dependency_manager
    
    # Conditional SciPy imports with fallbacks
    if dependency_manager.is_available('scipy'):
        from scipy import stats
        from scipy.optimize import minimize_scalar
    else:
        # Use fallback implementations
        stats = dependency_manager.get_fallback('scipy_stats')
        optimize = dependency_manager.get_fallback('scipy_optimize')
        minimize_scalar = optimize.minimize_scalar if optimize else None
        logging.getLogger(__name__).info("Using NumPy-based statistical and optimization fallbacks for SciPy functionality")
        
except ImportError:
    # Fallback if dependency manager is not available
    try:
        from scipy import stats
        from scipy.optimize import minimize_scalar
    except ImportError:
        # Create basic fallbacks inline
        import numpy as np
        
        class StatsFallback:
            @staticmethod
            def percentileofscore(data, score):
                return (np.sum(np.array(data) <= score) / len(data)) * 100
            
            @staticmethod
            def genpareto():
                class GenParetoFallback:
                    @staticmethod
                    def fit(data):
                        mean_val = np.mean(data)
                        std_val = np.std(data)
                        return 0.1, mean_val, std_val
                return GenParetoFallback()
        
        class OptimizeFallback:
            @staticmethod
            def minimize_scalar(func, bounds=None, method='bounded'):
                if bounds is None:
                    bounds = (-10, 10)
                x_values = np.linspace(bounds[0], bounds[1], 100)
                y_values = [func(x) for x in x_values]
                min_idx = np.argmin(y_values)
                
                class OptimizeResult:
                    def __init__(self, x, fun):
                        self.x = x
                        self.fun = fun
                        self.success = True
                
                return OptimizeResult(x_values[min_idx], y_values[min_idx])
        
        stats = StatsFallback()
        minimize_scalar = OptimizeFallback.minimize_scalar
        logging.getLogger(__name__).warning("SciPy not available, using basic NumPy statistical and optimization fallbacks")

logger = logging.getLogger(__name__)

@dataclass
class RiskScenario:
    scenario_id: str
    probability: float
    expected_pnl: float
    max_drawdown: float
    var_95: float
    var_99: float
    expected_shortfall: float
    stress_factor: float

@dataclass
class TailRiskMetrics:
    var_95: float
    var_99: float
    expected_shortfall_95: float
    expected_shortfall_99: float
    tail_ratio: float
    extreme_value_threshold: float
    black_swan_probability: float

class AdvancedRiskManager:
    def __init__(self, meta_learner):
        self.meta_learner = meta_learner
        
        # Monte Carlo parameters
        self.num_simulations = 1000
        self.simulation_horizon = 100  # Number of periods to simulate
        
        # Tail risk detection
        self.return_history = deque(maxlen=1000)
        self.volatility_history = deque(maxlen=100)
        self.correlation_history = deque(maxlen=500)
        
        # Extreme value theory parameters
        self.evt_threshold_percentile = 0.95
        self.evt_parameters = {'shape': 0.0, 'scale': 1.0, 'threshold': 0.0}
        
        # Dynamic correlation tracking
        self.correlation_matrix = np.eye(4)  # For 4 subsystems
        self.correlation_breakdown_threshold = 0.3
        
        # Black swan detection
        self.black_swan_indicators = {
            'volatility_spike': False,
            'correlation_breakdown': False,
            'extreme_move': False,
            'liquidity_crisis': False
        }
        
        # Adaptive drawdown prevention - let AI discover optimal levels
        self.max_portfolio_heat = 0.001  # Very low initial, adaptive discovery
        self.emergency_stop_threshold = 0.001  # Very low initial, adaptive discovery
        self.current_portfolio_heat = 0.0
        
        # Regime detection for risk adjustment
        self.current_regime = 'normal'  # normal, volatile, crisis
        self.regime_history = deque(maxlen=50)
        
        # Adaptive Kelly criterion optimization
        self.kelly_lookback = 100
        self.kelly_adjustment_factor = 0.01  # Very conservative initial, adaptive discovery
        
    def run_monte_carlo_simulation(self, current_position_size: int, 
                                 market_data, intelligence_data: Dict) -> List[RiskScenario]:
        """Run Monte Carlo simulations for position sizing optimization"""
        
        scenarios = []
        
        # Extract market parameters for simulation
        current_volatility = self._estimate_current_volatility(market_data)
        expected_return = self._estimate_expected_return(intelligence_data)
        
        # Generate multiple scenarios with different stress levels
        stress_levels = [1.0, 1.5, 2.0, 3.0, 5.0]  # Normal to extreme stress
        
        for stress_level in stress_levels:
            scenario_results = []
            
            for _ in range(self.num_simulations):
                # Simulate price path with stress adjustments
                returns = self._simulate_returns(
                    expected_return, 
                    current_volatility * stress_level,
                    self.simulation_horizon
                )
                
                # Calculate P&L for this simulation
                pnl_path = self._calculate_pnl_path(returns, current_position_size, market_data.price)
                
                scenario_results.append({
                    'final_pnl': pnl_path[-1],
                    'max_drawdown': self._calculate_max_drawdown(pnl_path),
                    'var_95': np.percentile(pnl_path, 5),
                    'var_99': np.percentile(pnl_path, 1)
                })
            
            # Aggregate scenario results
            final_pnls = [r['final_pnl'] for r in scenario_results]
            max_drawdowns = [r['max_drawdown'] for r in scenario_results]
            
            scenario = RiskScenario(
                scenario_id=f"stress_{stress_level}x",
                probability=self._estimate_scenario_probability(stress_level),
                expected_pnl=np.mean(final_pnls),
                max_drawdown=np.mean(max_drawdowns),
                var_95=np.percentile(final_pnls, 5),
                var_99=np.percentile(final_pnls, 1),
                expected_shortfall=np.mean([pnl for pnl in final_pnls if pnl <= np.percentile(final_pnls, 5)]),
                stress_factor=stress_level
            )
            
            scenarios.append(scenario)
        
        return scenarios
    
    def detect_tail_risk(self, market_data) -> TailRiskMetrics:
        """Detect tail risk using extreme value theory"""
        
        if len(self.return_history) < 100:
            # Not enough data for reliable tail risk estimation
            return TailRiskMetrics(0, 0, 0, 0, 0, 0, 0)
        
        returns = np.array(list(self.return_history))
        
        # Fit extreme value distribution to tail
        threshold = np.percentile(np.abs(returns), self.evt_threshold_percentile * 100)
        exceedances = returns[np.abs(returns) > threshold]
        
        if len(exceedances) < 10:
            threshold = np.percentile(np.abs(returns), 90)
            exceedances = returns[np.abs(returns) > threshold]
        
        # Fit Generalized Pareto Distribution to exceedances
        try:
            shape, loc, scale = stats.genpareto.fit(np.abs(exceedances) - threshold)
            self.evt_parameters = {'shape': shape, 'scale': scale, 'threshold': threshold}
        except:
            # Fallback to empirical estimates
            shape, scale = 0.1, np.std(exceedances)
            self.evt_parameters = {'shape': shape, 'scale': scale, 'threshold': threshold}
        
        # Calculate VaR and Expected Shortfall
        var_95 = self._calculate_var(0.95, returns)
        var_99 = self._calculate_var(0.99, returns)
        es_95 = self._calculate_expected_shortfall(0.95, returns)
        es_99 = self._calculate_expected_shortfall(0.99, returns)
        
        # Tail ratio (extreme tail vs moderate tail)
        tail_ratio = abs(var_99) / max(abs(var_95), 0.001)
        
        # Black swan probability (probability of 5+ sigma event)
        black_swan_prob = self._estimate_black_swan_probability(returns)
        
        return TailRiskMetrics(
            var_95=var_95,
            var_99=var_99,
            expected_shortfall_95=es_95,
            expected_shortfall_99=es_99,
            tail_ratio=tail_ratio,
            extreme_value_threshold=threshold,
            black_swan_probability=black_swan_prob
        )
    
    def detect_regime_change(self, market_data, intelligence_data: Dict) -> str:
        """Detect market regime changes for risk adjustment"""
        
        current_vol = self._estimate_current_volatility(market_data)
        self.volatility_history.append(current_vol)
        
        if len(self.volatility_history) < 20:
            return 'normal'
        
        vol_array = np.array(list(self.volatility_history))
        vol_percentile = stats.percentileofscore(vol_array[:-10], current_vol)
        
        # Correlation breakdown detection
        correlation_breakdown = self._detect_correlation_breakdown(intelligence_data)
        
        # Regime classification
        if vol_percentile > 95 or correlation_breakdown:
            regime = 'crisis'
        elif vol_percentile > 80:
            regime = 'volatile'
        else:
            regime = 'normal'
        
        self.regime_history.append(regime)
        self.current_regime = regime
        
        # Update black swan indicators
        self._update_black_swan_indicators(market_data, intelligence_data)
        
        return regime
    
    def optimize_kelly_position_size(self, base_size: int, market_data, 
                                   intelligence_data: Dict) -> int:
        """Optimize position size using Kelly criterion with uncertainty adjustments"""
        
        if len(self.return_history) < self.kelly_lookback:
            return base_size
        
        recent_returns = list(self.return_history)[-self.kelly_lookback:]
        
        # Estimate win probability and average win/loss
        positive_returns = [r for r in recent_returns if r > 0]
        negative_returns = [r for r in recent_returns if r < 0]
        
        if not positive_returns or not negative_returns:
            return base_size
        
        win_prob = len(positive_returns) / len(recent_returns)
        avg_win = np.mean(positive_returns)
        avg_loss = abs(np.mean(negative_returns))
        
        # Kelly fraction calculation
        if avg_loss > 0:
            kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
        else:
            kelly_fraction = 0
        
        # Apply conservative adjustment and uncertainty penalty
        uncertainty_penalty = self._calculate_uncertainty_penalty(intelligence_data)
        adjusted_kelly = kelly_fraction * self.kelly_adjustment_factor * (1 - uncertainty_penalty)
        
        # Apply regime-based adjustments
        regime_multiplier = self._get_regime_multiplier()
        final_kelly = adjusted_kelly * regime_multiplier
        
        # Convert to position size
        optimal_size = int(base_size * max(0.1, min(2.0, final_kelly)))
        
        return optimal_size
    
    def check_drawdown_prevention(self, proposed_order_size: int,
                                market_data) -> Tuple[bool, int, str]:
        """Enhanced learning-based risk protection"""
        
        account_balance = getattr(market_data, 'account_balance', 10000)
        
        # 1. Learned drawdown protection based on position sizing parameters
        position_size_factor = self.meta_learner.get_parameter('position_size_factor', 0.1)
        current_drawdown = self._calculate_current_drawdown(market_data)
        
        # Dynamic emergency threshold based on position sizing (removing daily loss dependency)
        dynamic_emergency_threshold = position_size_factor * 0.5  # Conservative emergency threshold
        
        if current_drawdown > dynamic_emergency_threshold:
            return False, 0, f"Learned emergency stop: Drawdown {current_drawdown:.2%} > {dynamic_emergency_threshold:.2%}"
        
        # 2. Intelligent portfolio heat management
        estimated_risk_per_contract = self._estimate_contract_risk(market_data)
        estimated_total_risk = proposed_order_size * estimated_risk_per_contract
        portfolio_heat = estimated_total_risk / account_balance
        
        # Learned maximum heat tolerance
        learned_max_heat = self.meta_learner.get_parameter('position_size_factor') * 0.5  # Half of position factor
        total_heat = self.current_portfolio_heat + portfolio_heat
        
        if total_heat > learned_max_heat:
            # Intelligent heat reduction
            max_additional_heat = learned_max_heat - self.current_portfolio_heat
            if max_additional_heat <= 0:
                return False, 0, f"Learned heat limit reached: {self.current_portfolio_heat:.2%}"
            
            heat_reduction_factor = max_additional_heat / portfolio_heat
            adjusted_size = max(1, int(proposed_order_size * heat_reduction_factor))
            return True, adjusted_size, f"Heat-adjusted size: {heat_reduction_factor:.2f}x"
        
        # 3. Regime-based intelligent scaling
        regime_multiplier = self._get_intelligent_regime_multiplier()
        if regime_multiplier < 1.0:
            adjusted_size = max(1, int(proposed_order_size * regime_multiplier))
            return True, adjusted_size, f"Regime-adjusted: {self.current_regime} ({regime_multiplier:.2f}x)"
        
        # 4. Black swan intelligent response
        if any(self.black_swan_indicators.values()):
            # Learned black swan response (not hardcoded)
            active_indicators = [k for k, v in self.black_swan_indicators.items() if v]
            severity = len(active_indicators) / len(self.black_swan_indicators)
            
            # Dynamic reduction based on severity
            reduction_factor = 1.0 - (severity * 0.7)  # Up to 70% reduction
            adjusted_size = max(1, int(proposed_order_size * reduction_factor))
            return True, adjusted_size, f"Black swan response: {active_indicators} ({reduction_factor:.2f}x)"
        
        # 5. Volatility-based intelligent scaling
        if len(self.volatility_history) > 10:
            current_vol = self.volatility_history[-1]
            avg_vol = np.mean(list(self.volatility_history))
            
            if current_vol > avg_vol * 1.5:  # High volatility
                vol_reduction = 0.8  # 20% reduction
                adjusted_size = max(1, int(proposed_order_size * vol_reduction))
                return True, adjusted_size, f"Volatility-adjusted: {current_vol:.3f} vs {avg_vol:.3f}"
        
        return True, proposed_order_size, "Risk checks passed"
    
    def _estimate_contract_risk(self, market_data) -> float:
        """Intelligently estimate risk per contract based on current conditions"""
        
        # Base risk estimate
        base_risk = 600.0  # Base margin requirement
        
        # Volatility adjustment
        if len(self.volatility_history) > 5:
            current_vol = self.volatility_history[-1]
            avg_vol = np.mean(list(self.volatility_history))
            vol_multiplier = max(0.5, min(2.0, current_vol / avg_vol))
            base_risk *= vol_multiplier
        
        # Regime adjustment
        regime_multipliers = {
            'normal': 1.0,
            'volatile': 1.3,
            'crisis': 1.8
        }
        base_risk *= regime_multipliers.get(self.current_regime, 1.0)
        
        return base_risk
    
    def _get_intelligent_regime_multiplier(self) -> float:
        """Get intelligent regime-based position multiplier"""
        
        # Base multipliers
        base_multipliers = {
            'normal': 1.0,
            'volatile': 0.7,
            'crisis': 0.3
        }
        
        base_multiplier = base_multipliers.get(self.current_regime, 1.0)
        
        # Adjust based on recent performance in this regime
        if len(self.regime_history) > 10:
            recent_regime_count = list(self.regime_history)[-10:].count(self.current_regime)
            if recent_regime_count > 5:  # Regime persistence
                # If we've been in this regime a while, be more conservative
                base_multiplier *= 0.9
        
        return base_multiplier
    
    def update_risk_metrics(self, market_data, trade_outcome: Optional[Dict] = None):
        """Update risk metrics with new market data and trade outcomes"""
        
        # Update return history
        if hasattr(market_data, 'price') and len(self.return_history) > 0:
            # Calculate return from previous price
            prev_price = getattr(self, '_last_price', market_data.price)
            if prev_price > 0:
                return_pct = (market_data.price - prev_price) / prev_price
                self.return_history.append(return_pct)
            self._last_price = market_data.price
        
        # Update portfolio heat if trade completed
        if trade_outcome:
            trade_pnl = trade_outcome.get('pnl', 0)
            # Adjust current portfolio heat based on realized outcome
            # This is simplified - in practice you'd track individual position heat
            
        # Update correlation matrix
        self._update_correlation_matrix(market_data)
        
        # Detect regime changes
        self.detect_regime_change(market_data, {})
    
    def get_risk_summary(self) -> Dict:
        """Get comprehensive risk summary"""
        
        tail_metrics = self.detect_tail_risk(None) if len(self.return_history) > 100 else None
        
        return {
            'current_regime': self.current_regime,
            'portfolio_heat': self.current_portfolio_heat,
            'black_swan_indicators': self.black_swan_indicators,
            'tail_risk_metrics': {
                'var_95': tail_metrics.var_95 if tail_metrics else 0,
                'var_99': tail_metrics.var_99 if tail_metrics else 0,
                'expected_shortfall_95': tail_metrics.expected_shortfall_95 if tail_metrics else 0,
                'black_swan_probability': tail_metrics.black_swan_probability if tail_metrics else 0
            } if tail_metrics else {},
            'evt_parameters': self.evt_parameters,
            'correlation_breakdown_risk': self._assess_correlation_breakdown_risk(),
            'return_history_length': len(self.return_history),
            'volatility_regime': 'high' if len(self.volatility_history) > 0 and 
                               self.volatility_history[-1] > np.mean(list(self.volatility_history)) * 1.5 else 'normal'
        }
    
    # Private helper methods
    
    def _simulate_returns(self, expected_return: float, volatility: float, 
                         horizon: int) -> np.ndarray:
        """Simulate return path using geometric Brownian motion with jumps"""
        
        dt = 1.0 / 252  # Daily time step
        
        # Normal diffusion component
        normal_returns = np.random.normal(
            expected_return * dt, 
            volatility * np.sqrt(dt), 
            horizon
        )
        
        # Add jump component for tail events
        jump_prob = 0.02  # 2% chance of jump per period
        jump_intensity = volatility * 3  # Jumps are 3x normal volatility
        
        jumps = np.random.binomial(1, jump_prob, horizon) * \
                np.random.normal(0, jump_intensity, horizon)
        
        return normal_returns + jumps
    
    def _calculate_pnl_path(self, returns: np.ndarray, position_size: int, 
                           entry_price: float) -> np.ndarray:
        """Calculate P&L path from returns"""
        
        price_path = entry_price * np.cumprod(1 + returns)
        pnl_path = (price_path - entry_price) * position_size * 2.0  # MNQ point value
        
        return pnl_path
    
    def _calculate_max_drawdown(self, pnl_path: np.ndarray) -> float:
        """Calculate maximum drawdown from P&L path"""
        
        cumulative = np.cumsum(pnl_path)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        
        return np.min(drawdown)
    
    def _estimate_scenario_probability(self, stress_level: float) -> float:
        """Estimate probability of stress scenario"""
        
        # Simple exponential decay - higher stress = lower probability
        base_prob = 0.4  # 40% for normal scenario
        return base_prob * np.exp(-0.5 * (stress_level - 1))
    
    def _estimate_current_volatility(self, market_data) -> float:
        """Estimate current market volatility"""
        
        if len(self.return_history) < 20:
            return 0.02  # Default 2% daily volatility
        
        recent_returns = list(self.return_history)[-20:]
        return np.std(recent_returns) * np.sqrt(252)  # Annualized
    
    def _estimate_expected_return(self, intelligence_data: Dict) -> float:
        """Estimate expected return from intelligence data"""
        
        # Use intelligence confidence as proxy for expected return
        confidence = intelligence_data.get('confidence', 0.5)
        direction = intelligence_data.get('direction', 0)  # -1, 0, 1
        
        # Convert to expected daily return
        base_return = 0.001  # 0.1% base daily return
        return base_return * direction * confidence
    
    def _calculate_var(self, confidence_level: float, returns: np.ndarray) -> float:
        """Calculate Value at Risk"""
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def _calculate_expected_shortfall(self, confidence_level: float, 
                                    returns: np.ndarray) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        var = self._calculate_var(confidence_level, returns)
        tail_returns = returns[returns <= var]
        return np.mean(tail_returns) if len(tail_returns) > 0 else var
    
    def _estimate_black_swan_probability(self, returns: np.ndarray) -> float:
        """Estimate probability of black swan event (5+ sigma)"""
        
        if len(returns) < 50:
            return 0.01
        
        std_dev = np.std(returns)
        extreme_threshold = 5 * std_dev
        extreme_events = np.sum(np.abs(returns) > extreme_threshold)
        
        return extreme_events / len(returns)
    
    def _detect_correlation_breakdown(self, intelligence_data: Dict) -> bool:
        """Detect if correlations are breaking down"""
        
        # Simplified correlation breakdown detection
        # In practice, you'd track correlations between different market factors
        
        subsystem_signals = intelligence_data.get('subsystem_signals', {})
        if len(subsystem_signals) < 2:
            return False
        
        signals = list(subsystem_signals.values())
        correlation_matrix = np.corrcoef(signals)
        
        # Check if any correlations have broken down significantly
        off_diagonal = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
        breakdown_count = np.sum(np.abs(off_diagonal) < self.correlation_breakdown_threshold)
        
        return breakdown_count > len(off_diagonal) * 0.5
    
    def _update_black_swan_indicators(self, market_data, intelligence_data: Dict):
        """Update black swan warning indicators"""
        
        # Volatility spike detection
        if len(self.volatility_history) > 10:
            current_vol = self.volatility_history[-1]
            avg_vol = np.mean(list(self.volatility_history)[:-1])
            self.black_swan_indicators['volatility_spike'] = current_vol > avg_vol * 2
        
        # Correlation breakdown
        self.black_swan_indicators['correlation_breakdown'] = \
            self._detect_correlation_breakdown(intelligence_data)
        
        # Extreme price move
        if len(self.return_history) > 1:
            latest_return = self.return_history[-1]
            vol_estimate = np.std(list(self.return_history)[-20:]) if len(self.return_history) >= 20 else 0.02
            self.black_swan_indicators['extreme_move'] = abs(latest_return) > vol_estimate * 4
        
        # Liquidity crisis (simplified - would need bid/ask spread data)
        self.black_swan_indicators['liquidity_crisis'] = False
    
    def _calculate_uncertainty_penalty(self, intelligence_data: Dict) -> float:
        """Calculate uncertainty penalty for Kelly sizing"""
        
        # Higher uncertainty = larger penalty
        confidence = intelligence_data.get('confidence', 0.5)
        disagreement = intelligence_data.get('subsystem_disagreement', 0.0)
        
        uncertainty = (1 - confidence) + disagreement
        return min(0.5, uncertainty)  # Cap at 50% penalty
    
    def _get_regime_multiplier(self) -> float:
        """Get position size multiplier based on current regime"""
        
        multipliers = {
            'normal': 1.0,
            'volatile': 0.7,
            'crisis': 0.3
        }
        
        return multipliers.get(self.current_regime, 1.0)
    
    def _calculate_current_drawdown(self, market_data) -> float:
        """Calculate current drawdown from peak"""
        
        # Simplified - would need actual account balance tracking
        account_balance = getattr(market_data, 'account_balance', 10000)
        peak_balance = getattr(self, '_peak_balance', account_balance)
        
        if account_balance > peak_balance:
            self._peak_balance = account_balance
            return 0.0
        
        return (peak_balance - account_balance) / peak_balance
    
    def _update_correlation_matrix(self, market_data):
        """Update correlation matrix for different market factors"""
        
        # Simplified correlation tracking
        # In practice, you'd track correlations between:
        # - Different timeframes
        # - Different subsystem signals  
        # - Market factors (VIX, rates, etc.)
        
        pass
    
    def _assess_correlation_breakdown_risk(self) -> float:
        """Assess risk of correlation breakdown"""
        
        # Return risk score between 0 and 1
        if len(self.regime_history) < 10:
            return 0.0
        
        recent_regimes = list(self.regime_history)[-10:]
        crisis_count = recent_regimes.count('crisis')
        volatile_count = recent_regimes.count('volatile')
        
        return (crisis_count * 0.8 + volatile_count * 0.4) / 10