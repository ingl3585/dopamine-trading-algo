"""
Seasonal Analysis Engine - Lunar/seasonal integration per prompt.txt
"""

import numpy as np
import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)

class SeasonalAnalyzer:
    """
    Seasonal pattern analysis with lunar/seasonal integration
    """
    
    def analyze_seasonal_patterns(self, prices: List[float], timestamps: List[float], 
                                 seasonal_patterns: Dict) -> float:
        """Analyze seasonal market patterns"""
        if len(timestamps) != len(prices):
            return 0.0
        
        signal = 0.0
        
        try:
            # Convert timestamps to datetime objects
            datetimes = [datetime.fromtimestamp(ts) for ts in timestamps[-20:]]
            recent_prices = prices[-20:]
            
            # Hour of day patterns
            hour_patterns = defaultdict(list)
            for dt, price in zip(datetimes, recent_prices):
                hour_patterns[dt.hour].append(price)
            
            # Day of week patterns
            dow_patterns = defaultdict(list)
            for dt, price in zip(datetimes, recent_prices):
                dow_patterns[dt.weekday()].append(price)
            
            # Calculate seasonal signals
            current_hour = datetimes[-1].hour
            current_dow = datetimes[-1].weekday()
            
            if current_hour in seasonal_patterns:
                expected_performance = seasonal_patterns[current_hour].get('performance', 0.0)
                signal += expected_performance * 0.5
            
            if f"dow_{current_dow}" in seasonal_patterns:
                expected_performance = seasonal_patterns[f"dow_{current_dow}"].get('performance', 0.0)
                signal += expected_performance * 0.3
            
        except Exception as e:
            logger.warning(f"Seasonal analysis error: {e}")
        
        return float(np.tanh(signal))

    def analyze_lunar_influence(self, prices: List[float], timestamps: List[float], 
                               lunar_cycle_data) -> float:
        """Analyze lunar cycle influence on markets per prompt.txt"""
        if len(timestamps) < 10:
            return 0.0
        
        try:
            # Simplified lunar cycle approximation (29.5 days)
            lunar_cycle_length = 29.5 * 24 * 3600  # seconds
            
            current_time = timestamps[-1]
            lunar_phase = (current_time % lunar_cycle_length) / lunar_cycle_length
            
            # Store lunar data
            lunar_cycle_data.append({
                'phase': lunar_phase,
                'price_change': (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 and prices[-10] != 0 else 0
            })
            
            # Analyze lunar correlation if we have enough data
            if len(lunar_cycle_data) >= 20:
                phases = [data['phase'] for data in lunar_cycle_data]
                changes = [data['price_change'] for data in lunar_cycle_data]
                
                # Simple correlation analysis
                correlation = np.corrcoef(phases, changes)[0, 1] if len(phases) > 1 else 0
                
                # Current lunar influence
                lunar_signal = correlation * np.sin(2 * np.pi * lunar_phase)
                return float(np.tanh(lunar_signal))
        
        except Exception as e:
            logger.warning(f"Lunar analysis error: {e}")
        
        return 0.0

    def update_seasonal_patterns(self, outcome: float, seasonal_patterns: Dict):
        """Update seasonal pattern performance"""
        try:
            current_time = datetime.now()
            hour_key = current_time.hour
            dow_key = f"dow_{current_time.weekday()}"
            
            # Update hour pattern
            if hour_key not in seasonal_patterns:
                seasonal_patterns[hour_key] = {'performance': 0.0, 'count': 0}
            
            data = seasonal_patterns[hour_key]
            data['performance'] = (data['performance'] * data['count'] + outcome) / (data['count'] + 1)
            data['count'] += 1
            
            # Update day of week pattern
            if dow_key not in seasonal_patterns:
                seasonal_patterns[dow_key] = {'performance': 0.0, 'count': 0}
            
            dow_data = seasonal_patterns[dow_key]
            dow_data['performance'] = (dow_data['performance'] * dow_data['count'] + outcome) / (dow_data['count'] + 1)
            dow_data['count'] += 1
            
        except Exception as e:
            logger.warning(f"Seasonal pattern update error: {e}")