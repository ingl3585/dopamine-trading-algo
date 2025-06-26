"""
Threat Analysis Engine - Pattern recognition and evolution tracking
"""

import numpy as np
import logging
from collections import deque
from datetime import datetime
from typing import Dict

logger = logging.getLogger(__name__)

class ThreatAnalyzer:
    """
    Advanced threat pattern analysis and evolution tracking
    """
    
    def create_pattern_signature(self, market_state: Dict) -> str:
        """Enhanced pattern signature creation"""
        signature_parts = []
        
        # Enhanced volatility signature with multiple buckets
        if 'volatility' in market_state:
            vol_bucket = min(99, int(market_state['volatility'] * 1000) // 10)
            signature_parts.append(f"vol_{vol_bucket}")
        
        # Enhanced volume momentum with trend detection
        if 'volume_momentum' in market_state:
            vol_mom = market_state['volume_momentum']
            vol_mom_bucket = min(99, max(0, int((vol_mom + 1.0) * 50)))
            signature_parts.append(f"vmom_{vol_mom_bucket}")
        
        # Enhanced price momentum with acceleration
        if 'price_momentum' in market_state:
            price_mom = market_state['price_momentum']
            price_mom_bucket = min(99, max(0, int((price_mom + 1.0) * 50)))
            signature_parts.append(f"pmom_{price_mom_bucket}")
        
        # Time-based signature with market session awareness
        if 'time_of_day' in market_state:
            time_val = market_state['time_of_day']
            hour = int(time_val * 24)
            
            # Market session classification
            if 9 <= hour <= 16:
                session = "regular"
            elif 4 <= hour <= 9:
                session = "premarket"
            elif 16 <= hour <= 20:
                session = "afterhours"
            else:
                session = "overnight"
            
            signature_parts.append(f"session_{session}")
            signature_parts.append(f"hour_{hour}")
        
        # Market regime signature
        if 'regime' in market_state:
            regime = market_state['regime']
            signature_parts.append(f"regime_{regime}")
        
        return "_".join(signature_parts)

    def pattern_similarity(self, pattern1: str, pattern2: str) -> float:
        """Enhanced pattern similarity with weighted components"""
        if not pattern1 or not pattern2:
            return 0.0
        
        parts1 = pattern1.split("_")
        parts2 = pattern2.split("_")
        
        # Component-wise similarity with weights
        component_weights = {
            'vol': 0.3,
            'vmom': 0.2,
            'pmom': 0.2,
            'session': 0.15,
            'hour': 0.1,
            'regime': 0.05
        }
        
        total_similarity = 0.0
        total_weight = 0.0
        
        # Create component dictionaries
        comp1 = {part.split('_')[0]: part for part in parts1 if '_' in part}
        comp2 = {part.split('_')[0]: part for part in parts2 if '_' in part}
        
        for component, weight in component_weights.items():
            if component in comp1 and component in comp2:
                if comp1[component] == comp2[component]:
                    total_similarity += weight
                elif component in ['vol', 'vmom', 'pmom']:
                    # Numerical similarity for continuous components
                    try:
                        val1 = int(comp1[component].split('_')[1])
                        val2 = int(comp2[component].split('_')[1])
                        numerical_sim = 1.0 - abs(val1 - val2) / 100.0
                        total_similarity += weight * max(0, numerical_sim)
                    except:
                        pass
                total_weight += weight
        
        return total_similarity / max(total_weight, 1e-8)

    def detect_evolved_threats(self, pattern_signature: str, market_state: Dict, 
                              threat_evolution_tracker: Dict) -> float:
        """Detect evolved or novel threat patterns"""
        evolved_threat_level = 0.0
        
        # Check for pattern mutations and evolutions
        current_volatility = market_state.get('volatility', 0.0)
        current_momentum = market_state.get('price_momentum', 0.0)
        
        # Detect extreme market conditions that might indicate evolved threats
        if current_volatility > 0.1:  # Very high volatility
            evolved_threat_level -= 0.2
        
        if abs(current_momentum) > 0.05:  # Strong momentum
            evolved_threat_level -= 0.1
        
        # Check for threat evolution patterns
        if pattern_signature in threat_evolution_tracker:
            evolution_data = threat_evolution_tracker[pattern_signature]
            severity_history = list(evolution_data['severity_history'])
            
            if len(severity_history) >= 3:
                # Check if threat is getting worse over time
                recent_trend = np.mean(severity_history[-3:]) - np.mean(severity_history[:-3])
                if recent_trend < -0.1:  # Getting more threatening
                    evolved_threat_level -= 0.3
        
        return evolved_threat_level

    def track_threat_evolution(self, pattern: str, market_state: Dict, threat_level: float,
                              threat_evolution_tracker: Dict):
        """Track how threats evolve over time"""
        if pattern not in threat_evolution_tracker:
            threat_evolution_tracker[pattern] = {
                'first_seen': datetime.now(),
                'severity_history': deque(maxlen=20),
                'volatility_trend': market_state.get('volatility', 0),
                'mutation_indicators': set()
            }
        
        evolution_data = threat_evolution_tracker[pattern]
        evolution_data['severity_history'].append(threat_level)
        
        # Update volatility trend
        current_vol = market_state.get('volatility', 0)
        evolution_data['volatility_trend'] = (evolution_data['volatility_trend'] * 0.9 + current_vol * 0.1)
        
        # Detect mutation indicators
        pattern_parts = pattern.split('_')
        for part in pattern_parts:
            if 'extreme' in part or 'spike' in part or 'crash' in part:
                evolution_data['mutation_indicators'].add(part)