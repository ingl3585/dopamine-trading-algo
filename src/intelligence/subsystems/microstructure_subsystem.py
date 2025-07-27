"""
Microstructure Intelligence Subsystem - Order flow and regime analysis
"""

import logging
import numpy as np
from collections import deque
from typing import Dict

from src.market_analysis.microstructure_analyzer import MarketMicrostructureEngine

logger = logging.getLogger(__name__)

class MicrostructureSubsystem:
    """
    Microstructure intelligence subsystem for order flow analysis and regime detection
    """
    
    def __init__(self, config):
        self.config = config
        self.microstructure_engine = MarketMicrostructureEngine()
        
        # Learning components
        self.pattern_memory = {}
        self.regime_performance = {}
        self.liquidity_predictions = deque(maxlen=100)
        self.total_learning_events = 0
        
        # Adaptive parameters
        self.signal_adaptation_rate = 0.1
        self.regime_confidence_threshold = 0.6
        
    def analyze_order_flow(self, market_data: Dict) -> float:
        """Analyze order flow patterns"""
        try:
            prices = market_data.get('prices', [market_data.get('price', 0.0)])
            volumes = market_data.get('volumes', [market_data.get('volume', 1000.0)])
            
            # Use the underlying engine's analyze_market_state method
            result = self.microstructure_engine.analyze_market_state(prices, volumes)
            return result.get('order_flow', {}).get('smart_money_flow', 0.0)
        except Exception as e:
            logger.error(f"Error in order flow analysis: {e}")
            return 0.0
    
    def detect_regime(self, market_features: Dict) -> str:
        """Detect current market regime"""
        try:
            prices = market_features.get('prices', [market_features.get('price', 0.0)])
            volumes = market_features.get('volumes', [market_features.get('volume', 1000.0)])
            
            # Use the underlying engine's analyze_market_state method
            result = self.microstructure_engine.analyze_market_state(prices, volumes)
            return result.get('regime_state', {}).get('overall_regime', 'unknown')
        except Exception as e:
            logger.error(f"Error in regime detection: {e}")
            return "unknown"
    
    def analyze_liquidity(self, market_data: Dict) -> float:
        """Analyze market liquidity"""
        try:
            prices = market_data.get('prices', [market_data.get('price', 0.0)])
            volumes = market_data.get('volumes', [market_data.get('volume', 1000.0)])
            
            # Use the underlying engine's analyze_market_state method
            result = self.microstructure_engine.analyze_market_state(prices, volumes)
            return result.get('order_flow', {}).get('liquidity_depth', 0.5)
        except Exception as e:
            logger.error(f"Error in liquidity analysis: {e}")
            return 0.5
    
    def get_signal(self, market_data: Dict, market_features: Dict) -> Dict:
        """Get microstructure intelligence signal"""
        try:
            prices = market_data.get('prices', [market_data.get('price', 0.0)])
            volumes = market_data.get('volumes', [market_data.get('volume', 1000.0)])
            
            # Use the underlying engine's analyze_market_state method
            result = self.microstructure_engine.analyze_market_state(prices, volumes)
            
            # Extract microstructure signal with type checking
            microstructure_signal_raw = result.get('microstructure_signal', 0.0)
            if isinstance(microstructure_signal_raw, dict):
                logger.warning(f"microstructure_signal from engine is dict: {microstructure_signal_raw}, using 0.0")
                microstructure_signal = 0.0
            elif isinstance(microstructure_signal_raw, (int, float)):
                microstructure_signal = microstructure_signal_raw
            else:
                logger.warning(f"microstructure_signal has unexpected type {type(microstructure_signal_raw)}: {microstructure_signal_raw}, using 0.0")
                microstructure_signal = 0.0
            
            return {
                'order_flow_signal': result.get('order_flow', {}).get('smart_money_flow', 0.0),
                'market_regime': result.get('regime_state', {}).get('overall_regime', 'unknown'),
                'liquidity_score': result.get('order_flow', {}).get('liquidity_depth', 0.5),
                'microstructure_signal': microstructure_signal,
                'stats': self.microstructure_engine.get_microstructure_features()
            }
        except Exception as e:
            logger.error(f"Error in microstructure signal generation: {e}")
            return {
                'order_flow_signal': 0.0,
                'market_regime': 'unknown',
                'liquidity_score': 0.5,
                'microstructure_signal': 0.0,
                'stats': {}
            }
    
    def learn_from_outcome(self, outcome_or_context, outcome=None):
        """Learn from trading outcomes to improve microstructure analysis"""
        
        # Handle different calling patterns with strict type checking
        if outcome is None:
            # Called with single argument (outcome only)
            if isinstance(outcome_or_context, (int, float)):
                outcome = outcome_or_context
                market_context = {}  # No context available
            else:
                logger.debug(f"Microstructure learn_from_outcome called with single non-numeric argument: {type(outcome_or_context)}")
                return
        else:
            # Called with two arguments (context, outcome)
            market_context = outcome_or_context
            if not isinstance(market_context, dict):
                logger.debug(f"Microstructure market_context is not a dict, skipping context learning: {type(market_context)}")
                market_context = {}  # Use empty context instead of erroring
            
            # Handle outcome - extract numeric value from dict if needed
            if isinstance(outcome, dict):
                # Extract signal value from dict outcome
                numeric_outcome = outcome.get('signal', 0.0)
                if not isinstance(numeric_outcome, (int, float)):
                    logger.warning(f"Microstructure outcome dict has non-numeric signal: {type(numeric_outcome)}, content: {outcome}")
                    return
                outcome = numeric_outcome
                logger.debug(f"Extracted numeric outcome from dict: {outcome}")
            elif not isinstance(outcome, (int, float)):
                logger.warning(f"Microstructure outcome is not numeric: {type(outcome)}, content: {outcome}")
                return
        
        self.total_learning_events += 1
        
        # Delegate to underlying engine first
        try:
            self.microstructure_engine.learn_from_outcome(outcome, market_context if outcome is not None else None)
        except Exception as e:
            logger.error(f"Error in underlying engine learning: {e}")
        
        try:
            # Learn regime performance
            regime = market_context.get('market_regime', 'unknown')
            if regime != 'unknown':
                if regime not in self.regime_performance:
                    self.regime_performance[regime] = {
                        'total_outcomes': [],
                        'confidence_factor': 0.5,
                        'reliability_score': 0.5
                    }
                
                regime_data = self.regime_performance[regime]
                regime_data['total_outcomes'].append(outcome)
                
                # Keep only recent outcomes
                if len(regime_data['total_outcomes']) > 50:
                    regime_data['total_outcomes'] = regime_data['total_outcomes'][-50:]
                
                # Update regime reliability
                if len(regime_data['total_outcomes']) >= 5:
                    recent_performance = np.mean(regime_data['total_outcomes'][-10:])
                    regime_data['reliability_score'] = 0.5 + np.tanh(recent_performance * 5) * 0.4
            
            # Learn order flow patterns
            order_flow_signal = market_context.get('order_flow_signal', 0.0)
            if isinstance(order_flow_signal, (int, float)) and abs(order_flow_signal) > 0.1:  # Significant signal
                pattern_key = f"flow_{int(order_flow_signal * 100)}"
                
                if pattern_key not in self.pattern_memory:
                    self.pattern_memory[pattern_key] = {
                        'outcomes': deque(maxlen=20),
                        'confidence': 0.5,
                        'frequency': 0
                    }
                
                pattern_data = self.pattern_memory[pattern_key]
                pattern_data['outcomes'].append(outcome)
                pattern_data['frequency'] += 1
                
                # Update pattern confidence based on consistency
                if len(pattern_data['outcomes']) >= 3:
                    outcomes_array = np.array(list(pattern_data['outcomes']))
                    consistency = 1.0 - (np.std(outcomes_array) / (np.mean(np.abs(outcomes_array)) + 0.1))
                    pattern_data['confidence'] = 0.3 + consistency * 0.6
            
            # Learn liquidity prediction accuracy
            predicted_liquidity = market_context.get('liquidity_score', 0.5)
            if isinstance(predicted_liquidity, (int, float)) and isinstance(outcome, (int, float)):
                actual_performance = 0.5 + np.tanh(outcome * 3) * 0.5  # Convert outcome to liquidity-like score
                
                self.liquidity_predictions.append({
                    'predicted': predicted_liquidity,
                    'actual': actual_performance,
                    'error': abs(predicted_liquidity - actual_performance)
                })
            
        except Exception as e:
            import traceback
            logger.error(f"Error in microstructure learning: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def get_enhanced_signal(self, market_features: Dict) -> Dict:
        """Get enhanced microstructure signal with learning-based adjustments"""
        try:
            # Extract market data from features
            market_data = {
                'prices': market_features.get('prices', []),
                'volumes': market_features.get('volumes', []),
                'price': market_features.get('prices', [0])[-1] if market_features.get('prices') else 0.0,
                'volume': market_features.get('volumes', [1000])[-1] if market_features.get('volumes') else 1000.0
            }
            
            # Get base signal
            base_signal = self.get_signal(market_data, market_features)
            
            # Apply learning-based enhancements
            regime = base_signal.get('market_regime', 'unknown')
            if regime in self.regime_performance:
                regime_data = self.regime_performance[regime]
                
                # Adjust signal based on regime reliability
                reliability_factor = regime_data['reliability_score']
                
                # Ensure microstructure_signal is numeric before multiplication
                microstructure_signal = base_signal.get('microstructure_signal', 0.0)
                if isinstance(microstructure_signal, dict):
                    logger.warning(f"Microstructure signal is dict instead of number: {microstructure_signal}, using 0.0")
                    base_signal['microstructure_signal'] = 0.0
                elif isinstance(microstructure_signal, (int, float)):
                    base_signal['microstructure_signal'] = microstructure_signal * reliability_factor
                else:
                    logger.warning(f"Microstructure signal has unexpected type {type(microstructure_signal)}: {microstructure_signal}, using 0.0")
                    base_signal['microstructure_signal'] = 0.0
                
                # Add regime confidence to signal
                base_signal['regime_confidence'] = reliability_factor
            
            # Enhance order flow signal with pattern memory
            order_flow = base_signal.get('order_flow_signal', 0.0)
            if isinstance(order_flow, (int, float)) and abs(order_flow) > 0.1:
                pattern_key = f"flow_{int(order_flow * 100)}"
                if pattern_key in self.pattern_memory:
                    pattern_confidence = self.pattern_memory[pattern_key]['confidence']
                    base_signal['order_flow_signal'] = order_flow * pattern_confidence
            
            # Add learning statistics
            base_signal['learning_stats'] = {
                'total_learning_events': self.total_learning_events,
                'known_regimes': len(self.regime_performance),
                'pattern_memory_size': len(self.pattern_memory),
                'liquidity_prediction_accuracy': self._calculate_liquidity_accuracy()
            }
            
            return base_signal
            
        except Exception as e:
            logger.error(f"Error in enhanced microstructure signal: {e}")
            return self.get_signal(market_data, market_features)
    
    def _calculate_liquidity_accuracy(self) -> float:
        """Calculate accuracy of liquidity predictions"""
        if len(self.liquidity_predictions) < 5:
            return 0.5
        
        recent_predictions = list(self.liquidity_predictions)[-20:]
        errors = [pred['error'] for pred in recent_predictions]
        avg_error = np.mean(errors)
        
        # Convert error to accuracy (lower error = higher accuracy)
        accuracy = 1.0 / (1.0 + avg_error * 2.0)
        return accuracy
    
    def get_microstructure_features(self) -> Dict:
        """Get microstructure features for compatibility with intelligence engine"""
        try:
            return {
                'pattern_count': len(self.pattern_memory),
                'regime_count': len(self.regime_performance),
                'total_events': self.total_learning_events,
                'liquidity_accuracy': self._calculate_liquidity_accuracy()
            }
        except Exception as e:
            logger.error(f"Error getting microstructure features: {e}")
            return {'pattern_count': 0, 'regime_count': 0, 'total_events': 0, 'liquidity_accuracy': 0.5}
    
    def analyze_market_state(self, market_features: Dict) -> Dict:
        """Analyze market state for compatibility - wrapper around get_enhanced_signal"""
        try:
            result = self.get_enhanced_signal(market_features)
            
            # Format for intelligence engine compatibility
            return {
                'signal': result.get('microstructure_signal', 0.0),
                'confidence': result.get('regime_confidence', 0.5),
                'reasoning': f"Regime: {result.get('market_regime', 'unknown')}, Flow: {result.get('order_flow_signal', 0.0):.3f}",
                'context': {
                    'smart_money_flow': result.get('order_flow_signal', 0.0),
                    'liquidity_depth': result.get('liquidity_score', 0.5),
                    'market_regime': result.get('market_regime', 'unknown'),
                    'regime_confidence': result.get('regime_confidence', 0.5)
                }
            }
        except Exception as e:
            logger.error(f"Error in analyze_market_state: {e}")
            return {
                'signal': 0.0,
                'confidence': 0.5,
                'reasoning': 'Error in analysis',
                'context': {
                    'smart_money_flow': 0.0,
                    'liquidity_depth': 0.5,
                    'market_regime': 'unknown',
                    'regime_confidence': 0.5
                }
            }
    
    def analyze_market_microstructure(self, features, market_data) -> Optional[Dict]:
        """
        Analyze market microstructure and return intelligence signal data
        
        This method provides the interface for the intelligence engine to get
        microstructure analysis results.
        
        Args:
            features: Market features
            market_data: Current market data
            
        Returns:
            Dict with signal, confidence, and microstructure information
        """
        try:
            # Extract market data
            prices = getattr(market_data, 'prices_1m', []) or []
            volumes = getattr(market_data, 'volumes_1m', []) or []
            
            if len(prices) < 5:
                return None
            
            # Create market features dict for analysis
            market_features = {
                'prices': prices,
                'volumes': volumes,
                'volatility': getattr(features, 'volatility', 0.02),
                'price_momentum': getattr(features, 'price_momentum', 0.0),
                'volume_momentum': getattr(features, 'volume_momentum', 0.0),
                'price': prices[-1] if prices else 0.0,
                'volume': volumes[-1] if volumes else 0.0
            }
            
            # Get enhanced microstructure signal
            microstructure_result = self.get_enhanced_signal(market_features)
            
            if not microstructure_result:
                return None
            
            # Extract signal and confidence
            signal_strength = microstructure_result.get('microstructure_signal', 0.0)
            confidence = microstructure_result.get('regime_confidence', 0.5)
            liquidity_state = self._classify_liquidity_state(microstructure_result)
            
            return {
                'signal': float(signal_strength),
                'confidence': float(confidence),
                'liquidity_state': liquidity_state,
                'market_regime': microstructure_result.get('market_regime', 'unknown'),
                'order_flow_signal': microstructure_result.get('order_flow_signal', 0.0),
                'liquidity_score': microstructure_result.get('liquidity_score', 0.5)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market microstructure: {e}")
            return None
    
    def _classify_liquidity_state(self, microstructure_result: Dict) -> str:
        """Classify the current liquidity state"""
        try:
            liquidity_score = microstructure_result.get('liquidity_score', 0.5)
            order_flow = microstructure_result.get('order_flow_signal', 0.0)
            
            if liquidity_score > 0.7 and abs(order_flow) < 0.3:
                return 'high_liquidity'
            elif liquidity_score < 0.3 or abs(order_flow) > 0.7:
                return 'low_liquidity'
            elif abs(order_flow) > 0.5:
                return 'volatile_flow'
            else:
                return 'normal'
                
        except Exception:
            return 'unknown'