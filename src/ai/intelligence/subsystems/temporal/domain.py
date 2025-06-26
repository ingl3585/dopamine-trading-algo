"""
FFT Temporal Subsystem Domain - Core cycle detection and temporal analysis
Implements advanced FFT analysis per prompt.txt requirements
"""

import numpy as np
import logging
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Optional

from .cycles import CycleDetector
from .seasonal import SeasonalAnalyzer

logger = logging.getLogger(__name__)

class FFTTemporalSubsystem:
    """
    FFT-Based Temporal Subsystem per prompt.txt:
    - FFT cycle detection to find dominant market frequencies
    - Cycle interference modeling that predicts when multiple cycles align/cancel
    - Adaptive cycle tracking that adjusts to changing market rhythms
    - Lunar/seasonal integration for longer-term pattern recognition
    """
    
    def __init__(self):
        self.cycle_memory = {}
        self.dominant_cycles = deque(maxlen=100)
        self.interference_patterns = {}
        self.cycle_predictions = deque(maxlen=50)
        self.cycle_importance_weights = {}
        self.seasonal_patterns = {}
        self.lunar_cycle_data = deque(maxlen=30)
        self.total_learning_events = 0
        self.learning_batch_size = 25
        
        # Domain services
        self.cycle_detector = CycleDetector()
        self.seasonal_analyzer = SeasonalAnalyzer()

    def analyze_cycles(self, prices: List[float], timestamps: Optional[List[float]] = None) -> float:
        """Analyze market cycles using FFT and return temporal signal strength"""
        if len(prices) < 32:
            logger.debug(f"Insufficient data for temporal analysis: {len(prices)} prices (need 32+)")
            return 0.0
        
        try:
            signals = []
            logger.debug(f"Starting temporal analysis with {len(prices)} prices")
            
            # Enhanced FFT analysis with multiple window sizes
            for window_size in [64, 128, 256]:
                if len(prices) >= window_size:
                    try:
                        signal = self.cycle_detector.fft_analysis(
                            prices[-window_size:], window_size, self.cycle_memory
                        )
                        if not (np.isnan(signal) or np.isinf(signal)):
                            signals.append(signal)
                            logger.debug(f"Temporal window {window_size}: signal={signal:.4f}")
                    except Exception as e:
                        logger.warning(f"FFT analysis failed for window size {window_size}: {e}")
                        continue
            
            if not signals:
                logger.debug("No valid temporal signals found, using baseline")
                return 0.02
            
            combined_signal = float(np.mean(signals))
            
            if np.isnan(combined_signal) or np.isinf(combined_signal):
                logger.warning("Combined temporal signal is invalid")
                return 0.0
            
            # Add seasonal and lunar analysis if timestamps available
            if timestamps:
                try:
                    seasonal_signal = self.seasonal_analyzer.analyze_seasonal_patterns(
                        prices, timestamps, self.seasonal_patterns
                    )
                    lunar_signal = self.seasonal_analyzer.analyze_lunar_influence(
                        prices, timestamps, self.lunar_cycle_data
                    )
                    
                    if not (np.isnan(seasonal_signal) or np.isinf(seasonal_signal)):
                        if not (np.isnan(lunar_signal) or np.isinf(lunar_signal)):
                            combined_signal = combined_signal * 0.7 + seasonal_signal * 0.2 + lunar_signal * 0.1
                            logger.debug(f"Enhanced temporal signal with seasonal/lunar: {combined_signal:.4f}")
                except Exception as e:
                    logger.warning(f"Seasonal/lunar analysis failed: {e}")
            
            if np.isnan(combined_signal) or np.isinf(combined_signal):
                logger.warning("Final temporal signal is invalid")
                return 0.0
            
            logger.debug(f"Temporal analysis result: {combined_signal:.4f}")
            return combined_signal
            
        except Exception as e:
            logger.error(f"Error in temporal cycle analysis: {e}")
            return 0.0

    def learn_from_outcome(self, cycles_info: List[Dict], outcome: float):
        """Learn from trade outcome to improve cycle patterns"""
        if not isinstance(cycles_info, list) or not cycles_info:
            return
        
        self.total_learning_events += 1
        
        # Update performance for observed cycles
        for cycle in cycles_info:
            freq = cycle['frequency']
            window_size = cycle.get('window_size', 64)
            cycle_key = f"freq_{freq:.6f}_w{window_size}"
            
            if cycle_key in self.cycle_memory:
                data = self.cycle_memory[cycle_key]
                
                # Adaptive learning rate based on confidence
                learning_rate = 0.05 * (1.0 + data['confidence'])
                
                # Update performance
                new_performance = data['performance'] + learning_rate * (outcome - data['performance'])
                
                # Update confidence based on prediction accuracy
                prediction_error = abs(outcome - data['performance'])
                confidence_update = 0.02 * (1.0 - prediction_error)
                new_confidence = max(0.1, min(1.0, data['confidence'] + confidence_update))
                
                self.cycle_memory[cycle_key] = {
                    'strength': cycle['amplitude'],
                    'phase': cycle['phase'],
                    'performance': new_performance,
                    'confidence': new_confidence
                }
            else:
                self.cycle_memory[cycle_key] = {
                    'strength': cycle['amplitude'],
                    'phase': cycle['phase'],
                    'performance': outcome * 0.3,
                    'confidence': 0.5
                }
        
        # Update seasonal patterns
        self.seasonal_analyzer.update_seasonal_patterns(outcome, self.seasonal_patterns)
        
        # Store interference patterns
        if len(cycles_info) >= 2:
            pattern_key = self._create_interference_pattern_key(cycles_info)
            if pattern_key not in self.interference_patterns:
                self.interference_patterns[pattern_key] = deque(maxlen=20)
            self.interference_patterns[pattern_key].append(outcome)

    def _create_interference_pattern_key(self, cycles_info: List[Dict]) -> str:
        """Create a key for interference pattern storage"""
        if len(cycles_info) < 2:
            return ""
        
        sorted_cycles = sorted(cycles_info, key=lambda x: x['frequency'])
        
        key_parts = []
        for cycle in sorted_cycles[:3]:  # Use top 3 cycles
            freq_bucket = int(cycle['frequency'] * 1000) / 1000
            key_parts.append(f"f{freq_bucket}")
        
        return "_".join(key_parts)

    def get_cycle_stats(self) -> Dict:
        """Get comprehensive cycle statistics"""
        if not self.cycle_memory:
            return {}
        
        performances = [data['performance'] for data in self.cycle_memory.values()]
        confidences = [data['confidence'] for data in self.cycle_memory.values()]
        
        return {
            'total_cycles': len(self.cycle_memory),
            'avg_performance': np.mean(performances),
            'avg_confidence': np.mean(confidences),
            'seasonal_patterns': len(self.seasonal_patterns),
            'interference_patterns': len(self.interference_patterns),
            'dominant_cycles': len(self.dominant_cycles)
        }