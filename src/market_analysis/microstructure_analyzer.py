# market_microstructure.py

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Simple progress tracking - matches subsystem_evolution.py
class SimpleProgressTracker:
    def __init__(self):
        self.tasks = {}
        
    def add_task(self, description, **kwargs):
        task_id = f"task_{len(self.tasks)}"
        self.tasks[task_id] = {
            'description': description.replace('[magenta]', '').replace('[/magenta]', ''),
            'count': 0,
            'kwargs': kwargs
        }
        return task_id
    
    def advance(self, task_id, amount=1):
        if task_id in self.tasks:
            self.tasks[task_id]['count'] += amount
            # Only log every 25 items to avoid spam
            if self.tasks[task_id]['count'] % 25 == 0:
                desc = self.tasks[task_id]['description']
                count = self.tasks[task_id]['count']
                logger.info(f"{desc}: {count} patterns learned")
    
    def update(self, task_id, **kwargs):
        if task_id in self.tasks:
            self.tasks[task_id]['kwargs'].update(kwargs)
    
    def stop(self):
        # Log final summary
        for task_id, task in self.tasks.items():
            desc = task['description']
            count = task['count']
            if count > 0:
                logger.info(f"{desc} completed: {count} total patterns")

progress = None

def _get_progress():
    """Get simple progress tracker"""
    global progress
    if progress is None:
        progress = SimpleProgressTracker()
    return progress


@dataclass
class OrderFlowSignals:
    smart_money_flow: float
    retail_flow: float
    market_maker_activity: float
    liquidity_depth: float
    momentum_strength: float
    tape_reading_signal: float


@dataclass
class RegimeState:
    volatility_regime: str  # 'low', 'medium', 'high', 'crisis'
    trend_regime: str      # 'trending', 'ranging', 'transitional'
    correlation_regime: str # 'normal', 'breakdown', 'extreme'
    liquidity_regime: str  # 'abundant', 'normal', 'scarce'
    confidence: float


class OrderFlowAnalyzer:
    def __init__(self):
        self.price_history = deque(maxlen=1000)
        self.volume_history = deque(maxlen=1000)
        self.smart_money_patterns = {}
        self.retail_patterns = {}
        
        # Market maker detection
        self.mm_signatures = deque(maxlen=100)
        self.accumulation_zones = []
        self.distribution_zones = []
        
    def analyze_order_flow(self, prices: List[float], volumes: List[float]) -> OrderFlowSignals:
        if len(prices) < 20 or len(volumes) < 20:
            return OrderFlowSignals(0, 0, 0, 0.5, 0, 0)
        
        # Update history
        self.price_history.extend(prices[-10:])
        self.volume_history.extend(volumes[-10:])
        
        # Smart money vs retail detection
        smart_money_flow = self._detect_smart_money(prices, volumes)
        retail_flow = self._detect_retail_flow(prices, volumes)
        
        # Market maker activity
        mm_activity = self._detect_market_maker_activity(prices, volumes)
        
        # Liquidity analysis
        liquidity_depth = self._analyze_liquidity_depth(prices, volumes)
        
        # Momentum detection
        momentum = self._detect_momentum(prices, volumes)
        
        # Tape reading signals
        tape_signal = self._tape_reading_analysis(prices, volumes)
        
        return OrderFlowSignals(
            smart_money_flow=smart_money_flow,
            retail_flow=retail_flow,
            market_maker_activity=mm_activity,
            liquidity_depth=liquidity_depth,
            momentum_strength=momentum,
            tape_reading_signal=tape_signal
        )
    
    def _detect_smart_money(self, prices: List[float], volumes: List[float]) -> float:
        # Smart money characteristics: large volume with minimal price impact
        signal = 0.0
        
        for i in range(1, min(len(prices), 10)):
            price_change = abs(prices[i] - prices[i-1]) / prices[i-1] if prices[i-1] != 0 else 0
            volume_ratio = volumes[i] / np.mean(volumes[:i]) if i > 0 else 1.0
            
            # High volume with low price impact = smart money
            if volume_ratio > 1.5 and price_change < 0.002:
                signal += 0.3
            
            # Gradual accumulation pattern
            if volume_ratio > 1.2 and price_change > 0 and price_change < 0.001:
                signal += 0.2
        
        return np.tanh(signal)
    
    def _detect_retail_flow(self, prices: List[float], volumes: List[float]) -> float:
        # Retail characteristics: emotional trading, momentum chasing
        signal = 0.0
        
        for i in range(2, min(len(prices), 10)):
            price_change = (prices[i] - prices[i-1]) / prices[i-1] if prices[i-1] != 0 else 0
            prev_change = (prices[i-1] - prices[i-2]) / prices[i-2] if prices[i-2] != 0 else 0
            volume_spike = volumes[i] / volumes[i-1] if volumes[i-1] > 0 else 1.0
            
            # Momentum chasing - buying after price rises
            if price_change > 0.003 and prev_change > 0.002 and volume_spike > 1.3:
                signal += 0.4
            
            # Panic selling after drops
            if price_change < -0.003 and volume_spike > 1.5:
                signal += 0.3
        
        return np.tanh(signal)
    
    def _detect_market_maker_activity(self, prices: List[float], volumes: List[float]) -> float:
        # Market maker signatures: consistent liquidity provision, mean reversion
        signal = 0.0
        
        if len(prices) < 5:
            return 0.0
        
        # Look for mean reversion patterns
        price_array = np.array(prices[-10:])
        mean_price = np.mean(price_array)
        
        # Market makers provide liquidity away from consensus
        distance_from_mean = abs(prices[-1] - mean_price) / mean_price if mean_price > 0 else 0
        
        if distance_from_mean > 0.002:  # Price away from mean
            volume_ratio = volumes[-1] / np.mean(volumes[-5:]) if len(volumes) >= 5 else 1.0
            if volume_ratio > 1.1:  # Increased liquidity provision
                signal += 0.5
        
        # Detect inventory management patterns
        recent_volumes = volumes[-5:]
        volume_consistency = 1.0 - (np.std(recent_volumes) / np.mean(recent_volumes)) if recent_volumes else 0
        signal += volume_consistency * 0.3
        
        return np.tanh(signal)
    
    def _analyze_liquidity_depth(self, prices: List[float], volumes: List[float]) -> float:
        # Estimate market depth based on volume/price relationship
        if len(prices) < 5 or len(volumes) < 5:
            return 0.5
        
        price_moves = [abs(prices[i] - prices[i-1]) / prices[i-1] 
                      for i in range(1, len(prices)) if prices[i-1] != 0]
        
        if not price_moves:
            return 0.5
        
        avg_volume = np.mean(volumes)
        avg_price_move = np.mean(price_moves)
        
        # High liquidity = high volume with small price moves
        if avg_price_move > 0:
            liquidity_ratio = avg_volume / (avg_price_move * 1000000)
            return np.tanh(liquidity_ratio)
        
        return 0.5
    
    def _detect_momentum(self, prices: List[float], volumes: List[float]) -> float:
        if len(prices) < 10:
            return 0.0
        
        # Price momentum
        short_ma = np.mean(prices[-3:])
        long_ma = np.mean(prices[-10:])
        price_momentum = (short_ma - long_ma) / long_ma if long_ma > 0 else 0
        
        # Volume momentum
        recent_vol = np.mean(volumes[-3:])
        avg_vol = np.mean(volumes[-10:])
        volume_momentum = (recent_vol - avg_vol) / avg_vol if avg_vol > 0 else 0
        
        # Combined momentum with volume confirmation
        if price_momentum > 0 and volume_momentum > 0:
            return min(1.0, (price_momentum + volume_momentum) * 2)
        elif price_momentum < 0 and volume_momentum > 0:
            return max(-1.0, price_momentum * 2)
        else:
            return price_momentum
    
    def _tape_reading_analysis(self, prices: List[float], volumes: List[float]) -> float:
        # Traditional tape reading signals
        signal = 0.0
        
        if len(prices) < 5:
            return 0.0
        
        # Look for absorption patterns
        for i in range(1, min(len(prices), 5)):
            price_change = prices[i] - prices[i-1]
            volume_ratio = volumes[i] / volumes[i-1] if volumes[i-1] > 0 else 1.0
            
            # High volume absorption (support/resistance)
            if volume_ratio > 1.5 and abs(price_change) < 0.001:
                signal += 0.3 if price_change >= 0 else -0.3
            
            # Volume climax patterns
            if volume_ratio > 2.0 and abs(price_change) > 0.003:
                signal += -0.2 * np.sign(price_change)  # Exhaustion signal
        
        return np.tanh(signal)


class RegimeDetector:
    def __init__(self):
        self.volatility_history = deque(maxlen=100)
        self.correlation_history = deque(maxlen=50)
        self.trend_strength_history = deque(maxlen=50)
        
        # Regime thresholds adjusted for MNQ futures volatility (23hr trading)
        # With 0.8% daily moves being "high", adjust thresholds accordingly
        self.vol_thresholds = {'low': 0.002, 'medium': 0.005, 'high': 0.012}
        self.trend_thresholds = {'ranging': 0.3, 'trending': 0.7}
        
    def detect_current_regime(self, prices: List[float], volumes: List[float], 
                             external_factors: Dict = None) -> RegimeState:
        
        if len(prices) < 20:
            return RegimeState('unknown', 'unknown', 'unknown', 'unknown', 0.0)
        
        # Volatility regime
        volatility = self._calculate_realized_volatility(prices)
        vol_regime = self._classify_volatility_regime(volatility)
        
        # Trend regime
        trend_strength = self._calculate_trend_strength(prices)
        trend_regime = self._classify_trend_regime(trend_strength)
        
        # Correlation regime (simplified - would need multiple assets)
        correlation_regime = self._estimate_correlation_regime(prices, volumes)
        
        # Liquidity regime
        liquidity_regime = self._assess_liquidity_regime(volumes)
        
        # Overall confidence based on consistency
        confidence = self._calculate_regime_confidence(vol_regime, trend_regime, 
                                                     correlation_regime, liquidity_regime)
        
        return RegimeState(
            volatility_regime=vol_regime,
            trend_regime=trend_regime,
            correlation_regime=correlation_regime,
            liquidity_regime=liquidity_regime,
            confidence=confidence
        )
    
    def _calculate_realized_volatility(self, prices: List[float]) -> float:
        if len(prices) < 2:
            return 0.0
        
        returns = [(prices[i] - prices[i-1]) / prices[i-1] 
                  for i in range(1, len(prices)) if prices[i-1] != 0]
        
        if not returns:
            return 0.0
        
        # Use rolling standard deviation scaled for practical trading decisions
        # This gives us volatility as a percentage that matches our thresholds
        raw_volatility = np.std(returns)
        
        # Scale to approximate daily volatility (sqrt of minutes in trading day)  
        # For MNQ: ~23 hours * 60 minutes = 1380 minutes per day (nearly 24/7 trading)
        daily_vol = raw_volatility * np.sqrt(1380)
        
        return daily_vol
    
    def _classify_volatility_regime(self, volatility: float) -> str:
        self.volatility_history.append(volatility)
        
        if volatility < self.vol_thresholds['low']:
            return 'low'
        elif volatility < self.vol_thresholds['medium']:
            return 'medium'
        elif volatility < self.vol_thresholds['high']:
            return 'high'
        else:
            return 'crisis'
    
    def _calculate_trend_strength(self, prices: List[float]) -> float:
        if len(prices) < 10:
            return 0.0
        
        # Multiple timeframe trend analysis
        short_trend = (prices[-1] - prices[-5]) / prices[-5] if prices[-5] != 0 else 0
        medium_trend = (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 and prices[-10] != 0 else 0
        long_trend = (prices[-1] - prices[-20]) / prices[-20] if len(prices) >= 20 and prices[-20] != 0 else 0
        
        # Trend consistency
        trends = [short_trend, medium_trend, long_trend]
        trend_signs = [np.sign(t) for t in trends if t != 0]
        
        if not trend_signs:
            return 0.0
        
        consistency = abs(sum(trend_signs)) / len(trend_signs)
        magnitude = np.mean([abs(t) for t in trends])
        
        return consistency * magnitude
    
    def _classify_trend_regime(self, trend_strength: float) -> str:
        self.trend_strength_history.append(trend_strength)
        
        if trend_strength < self.trend_thresholds['ranging']:
            return 'ranging'
        elif trend_strength > self.trend_thresholds['trending']:
            return 'trending'
        else:
            return 'transitional'
    
    def _estimate_correlation_regime(self, prices: List[float], volumes: List[float]) -> str:
        # Simplified correlation analysis using price-volume relationship
        if len(prices) < 10 or len(volumes) < 10:
            return 'unknown'
        
        price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        volume_changes = [volumes[i] - volumes[i-1] for i in range(1, len(volumes))]
        
        if len(price_changes) != len(volume_changes):
            return 'unknown'
        
        try:
            # Check for zero variance to avoid divide by zero warning
            if np.std(price_changes) == 0 or np.std(volume_changes) == 0:
                return 'unknown'
            
            correlation = np.corrcoef(price_changes, volume_changes)[0, 1]
            
            if np.isnan(correlation):
                return 'unknown'
            elif abs(correlation) > 0.7:
                return 'extreme'
            elif abs(correlation) < 0.3:
                return 'breakdown'
            else:
                return 'normal'
        except:
            return 'unknown'
    
    def _assess_liquidity_regime(self, volumes: List[float]) -> str:
        if len(volumes) < 10:
            return 'unknown'
        
        recent_volume = np.mean(volumes[-5:])
        historical_volume = np.mean(volumes[-20:-5]) if len(volumes) >= 20 else np.mean(volumes)
        
        if historical_volume == 0:
            return 'unknown'
        
        volume_ratio = recent_volume / historical_volume
        
        if volume_ratio > 1.5:
            return 'abundant'
        elif volume_ratio < 0.7:
            return 'scarce'
        else:
            return 'normal'
    
    def _calculate_regime_confidence(self, vol_regime: str, trend_regime: str, 
                                   corr_regime: str, liquidity_regime: str) -> float:
        # Simple confidence based on how many regimes are in "normal" ranges
        normal_regimes = 0
        total_regimes = 0
        
        regime_scores = {
            'vol': 1.0 if vol_regime in ['low', 'medium'] else 0.5 if vol_regime == 'high' else 0.0,
            'trend': 1.0 if trend_regime in ['ranging', 'trending'] else 0.5,
            'corr': 1.0 if corr_regime == 'normal' else 0.5 if corr_regime == 'breakdown' else 0.0,
            'liquidity': 1.0 if liquidity_regime == 'normal' else 0.7 if liquidity_regime == 'abundant' else 0.3
        }
        
        return np.mean(list(regime_scores.values()))
    


class MarketMicrostructureEngine:
    def __init__(self):
        self.order_flow_analyzer = OrderFlowAnalyzer()
        self.regime_detector = RegimeDetector()
        self.analysis_history = deque(maxlen=100)
        self.patterns = {}
        
        # Progress tracking
        self.learning_progress = None
        self.total_learning_events = 0
        self.learning_batch_size = 30
        self.bootstrap_complete = False
        
    def analyze_market_state(self, prices: List[float], volumes: List[float], 
                           external_data: Dict = None) -> Dict:
        
        # Order flow analysis
        order_flow = self.order_flow_analyzer.analyze_order_flow(prices, volumes)
        
        # Regime detection
        regime_state = self.regime_detector.detect_current_regime(prices, volumes, external_data)
        
        # Combined microstructure signal
        microstructure_signal = self._combine_signals(order_flow, regime_state)
        
        analysis = {
            'order_flow': {
                'smart_money_flow': order_flow.smart_money_flow,
                'retail_flow': order_flow.retail_flow,
                'market_maker_activity': order_flow.market_maker_activity,
                'liquidity_depth': order_flow.liquidity_depth,
                'momentum_strength': order_flow.momentum_strength,
                'tape_reading_signal': order_flow.tape_reading_signal
            },
            'regime_state': {
                'volatility_regime': regime_state.volatility_regime,
                'trend_regime': regime_state.trend_regime,
                'correlation_regime': regime_state.correlation_regime,
                'liquidity_regime': regime_state.liquidity_regime,
                'confidence': regime_state.confidence
            },
            'microstructure_signal': microstructure_signal,
            'regime_adjusted_signal': self._adjust_signal_for_regime(microstructure_signal, regime_state)
        }
        
        self.analysis_history.append(analysis)
        return analysis
    
    def _combine_signals(self, order_flow: OrderFlowSignals, regime_state: RegimeState) -> float:
        # Weight different signals based on regime
        signal = 0.0
        
        # Smart money flow gets higher weight in all regimes
        signal += order_flow.smart_money_flow * 0.3
        
        # Retail flow (contrarian signal)
        signal -= order_flow.retail_flow * 0.15
        
        # Market maker activity (mean reversion signal)
        signal += order_flow.market_maker_activity * 0.2
        
        # Momentum in trending regimes
        if regime_state.trend_regime == 'trending':
            signal += order_flow.momentum_strength * 0.4
        
        # Tape reading always valuable
        signal += order_flow.tape_reading_signal * 0.25
        
        return np.tanh(signal)
    
    def _adjust_signal_for_regime(self, base_signal: float, regime_state: RegimeState) -> float:
        adjusted_signal = base_signal
        
        # Volatility adjustments
        if regime_state.volatility_regime == 'crisis':
            adjusted_signal *= 0.5  # Reduce signal strength in crisis
        elif regime_state.volatility_regime == 'low':
            adjusted_signal *= 1.2  # Boost signal in low vol
        
        # Liquidity adjustments
        if regime_state.liquidity_regime == 'scarce':
            adjusted_signal *= 0.7  # Reduce in illiquid markets
        elif regime_state.liquidity_regime == 'abundant':
            adjusted_signal *= 1.1
        
        # Confidence adjustment
        adjusted_signal *= regime_state.confidence
        
        return adjusted_signal
    
    def learn_from_outcome(self, outcome: float, context: Optional[Dict] = None):
        """Learn from a trade outcome by associating it with a specific microstructure pattern."""
        if not context:
            return

        self.total_learning_events += 1

        # Initialize progress task if needed (only during bootstrap)
        # Skip progress bar if bootstrap is complete OR live trading has started
        if (self.learning_progress is None and 
            not getattr(self, 'bootstrap_complete', False) and 
            not hasattr(self, '_live_trading_started') and
            self.total_learning_events < 100):  # Only during initial bootstrap phase
            try:
                prog = _get_progress()
                if prog is not None:
                    self.learning_progress = prog.add_task(
                        "Microstructure Learning", 
                        patterns=0, 
                        avg_strength=0.0
                    )
            except Exception as e:
                logger.error(f"Error initializing microstructure progress task: {e}")
                self.learning_progress = None

        pattern_id = self._create_pattern_id(context)
        if not pattern_id:
            return

        pattern_added = False
        if pattern_id not in self.patterns:
            self.patterns[pattern_id] = {'outcomes': [], 'strength': 0.0}
            pattern_added = True
        
        self.patterns[pattern_id]['outcomes'].append(outcome)
        # Simple moving average of performance
        self.patterns[pattern_id]['strength'] = np.mean(self.patterns[pattern_id]['outcomes'][-20:])
        
        # Update progress task (only during bootstrap)
        if (pattern_added and self.learning_progress is not None and 
            not getattr(self, 'bootstrap_complete', False) and 
            not hasattr(self, '_live_trading_started')):
            try:
                prog = _get_progress()
                if prog is not None:
                    prog.advance(self.learning_progress, 1)
            except Exception as e:
                logger.warning(f"Error updating microstructure progress task: {e}")
        
        # Update progress task stats every batch (only during bootstrap)
        if (self.total_learning_events % self.learning_batch_size == 0 and 
            self.learning_progress is not None and 
            not getattr(self, 'bootstrap_complete', False) and 
            not hasattr(self, '_live_trading_started')):
            try:
                prog = _get_progress()
                if prog is not None:
                    avg_strength = np.mean([data['strength'] for data in self.patterns.values()]) if self.patterns else 0.0
                    prog.update(
                        self.learning_progress,
                        patterns=len(self.patterns),
                        avg_strength=f"{avg_strength:.3f}"
                    )
            except Exception as e:
                logger.warning(f"Error updating microstructure progress task stats: {e}")

    def _create_pattern_id(self, context: Dict) -> str:
        """Creates a unique, learnable pattern ID from the microstructure context."""
        try:
            order_flow = context.get('order_flow', {})
            regime_state = context.get('regime_state', {})

            smart_money_bucket = int(order_flow.get('smart_money_flow', 0) * 10)
            liquidity_bucket = int(order_flow.get('liquidity_depth', 0) * 10)
            vol_regime = regime_state.get('volatility_regime', 'unknown')[0]
            trend_regime = regime_state.get('trend_regime', 'unknown')[0]

            return f"sm{smart_money_bucket}_liq{liquidity_bucket}_vol{vol_regime}_trd{trend_regime}"
        except (TypeError, KeyError, IndexError):
            return ""
    
    def get_microstructure_features(self) -> Dict:
        """Extract features for use in neural networks"""
        base_features = {}
        
        if self.analysis_history:
            latest = self.analysis_history[-1]
            base_features = {
                'smart_money_signal': latest['order_flow']['smart_money_flow'],
                'liquidity_signal': latest['order_flow']['liquidity_depth'],
                'momentum_signal': latest['order_flow']['momentum_strength'],
                'regime_volatility': self._regime_to_numeric(latest['regime_state']['volatility_regime']),
                'regime_trend': self._regime_to_numeric(latest['regime_state']['trend_regime']),
                'regime_confidence': latest['regime_state']['confidence'],
                'microstructure_signal': latest['microstructure_signal'],
                'regime_adjusted_signal': latest['regime_adjusted_signal']
            }
        
        # Add pattern information for bootstrap stats
        base_features['patterns'] = self.patterns
        base_features['pattern_count'] = len(self.patterns)
        
        return base_features
    
    def _regime_to_numeric(self, regime: str) -> float:
        """Convert regime strings to numeric values for neural networks"""
        regime_mappings = {
            # Volatility regimes
            'low': 0.2, 'medium': 0.5, 'high': 0.8, 'crisis': 1.0,
            # Trend regimes
            'ranging': 0.2, 'transitional': 0.5, 'trending': 0.8,
            # Correlation regimes
            'breakdown': 0.2, 'normal': 0.5, 'extreme': 0.8,
            # Liquidity regimes
            'scarce': 0.2, 'normal': 0.5, 'abundant': 0.8,
            # Unknown
            'unknown': 0.0
        }
        
        return regime_mappings.get(regime, 0.0)