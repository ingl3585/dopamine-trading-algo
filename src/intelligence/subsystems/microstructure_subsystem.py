"""
Microstructure Intelligence Subsystem - Order flow and regime analysis
"""

import logging
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
            return self.microstructure_engine.analyze_order_flow(market_data)
        except Exception as e:
            logger.error(f"Error in order flow analysis: {e}")
            return 0.0
    
    def detect_regime(self, market_features: Dict) -> str:
        """Detect current market regime"""
        try:
            return self.microstructure_engine.detect_regime(market_features)
        except Exception as e:
            logger.error(f"Error in regime detection: {e}")
            return "unknown"
    
    def analyze_liquidity(self, market_data: Dict) -> float:
        """Analyze market liquidity"""
        try:
            return self.microstructure_engine.analyze_liquidity(market_data)
        except Exception as e:
            logger.error(f"Error in liquidity analysis: {e}")
            return 0.5
    
    def get_signal(self, market_data: Dict, market_features: Dict) -> Dict:
        """Get microstructure intelligence signal"""
        try:
            return {
                'order_flow_signal': self.analyze_order_flow(market_data),
                'market_regime': self.detect_regime(market_features),
                'liquidity_score': self.analyze_liquidity(market_data),
                'microstructure_signal': self.microstructure_engine.get_microstructure_signal(
                    market_data, market_features
                ),
                'stats': self.microstructure_engine.get_microstructure_stats()
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
    
    def learn_from_outcome(self, market_context: Dict, outcome: float):
        """Learn from trading outcomes to improve microstructure analysis"""
        if not isinstance(market_context, dict):
            logger.error(f"Microstructure market_context is not a dict: type={type(market_context)}")
            return
        
        self.total_learning_events += 1
        
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
            if abs(order_flow_signal) > 0.1:  # Significant signal
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
            actual_performance = 0.5 + np.tanh(outcome * 3) * 0.5  # Convert outcome to liquidity-like score
            
            self.liquidity_predictions.append({
                'predicted': predicted_liquidity,
                'actual': actual_performance,
                'error': abs(predicted_liquidity - actual_performance)
            })
            
        except Exception as e:
            logger.error(f"Error in microstructure learning: {e}")
    
    def get_enhanced_signal(self, market_data: Dict, market_features: Dict) -> Dict:
        """Get enhanced microstructure signal with learning-based adjustments"""
        try:
            # Get base signal
            base_signal = self.get_signal(market_data, market_features)
            
            # Apply learning-based enhancements
            regime = base_signal.get('market_regime', 'unknown')
            if regime in self.regime_performance:
                regime_data = self.regime_performance[regime]
                
                # Adjust signal based on regime reliability
                reliability_factor = regime_data['reliability_score']
                base_signal['microstructure_signal'] *= reliability_factor
                
                # Add regime confidence to signal
                base_signal['regime_confidence'] = reliability_factor
            
            # Enhance order flow signal with pattern memory
            order_flow = base_signal.get('order_flow_signal', 0.0)
            if abs(order_flow) > 0.1:
                pattern_key = f"flow_{int(order_flow * 100)}"
                if pattern_key in self.pattern_memory:
                    pattern_confidence = self.pattern_memory[pattern_key]['confidence']
                    base_signal['order_flow_signal'] *= pattern_confidence
            
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