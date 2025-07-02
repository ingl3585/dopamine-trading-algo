import numpy as np
import logging
from scipy import fft
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class FFTTemporalSubsystem:
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

    def analyze_cycles(self, prices: List[float], timestamps: Optional[List[float]] = None) -> float:
        if len(prices) < 8:  # Much lower minimum for adaptive discovery
            logger.debug(f"Insufficient data for temporal analysis: {len(prices)} prices (need 8+)")
            return 0.0
        
        try:
            signals = []
            logger.debug(f"Starting temporal analysis with {len(prices)} prices")
            
            for window_size in [8, 16, 32, 64, 128, 256]:  # More adaptive window sizes
                if len(prices) >= window_size:
                    try:
                        signal = self._fft_analysis(prices[-window_size:], window_size)
                        if not (np.isnan(signal) or np.isinf(signal)):
                            signals.append(signal)
                            logger.debug(f"Temporal window {window_size}: signal={signal:.4f}")
                        else:
                            logger.warning(f"Invalid signal from window {window_size}: {signal}")
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
            
            if timestamps:
                try:
                    seasonal_signal = self._analyze_seasonal_patterns(prices, timestamps)
                    lunar_signal = self._analyze_lunar_influence(prices, timestamps)
                    
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

    def _fft_analysis(self, prices: List[float], window_size: int) -> float:
        price_array = np.array(prices)
        
        detrended = price_array - np.linspace(price_array[0], price_array[-1], len(price_array))
        windowed = detrended * np.hanning(len(detrended))
        
        fft_result = fft.fft(windowed)
        frequencies = fft.fftfreq(len(windowed))
        power_spectrum = np.abs(fft_result)
        
        valid_indices = np.where((frequencies > 0) & (frequencies < 0.5))[0]
        if len(valid_indices) == 0:
            return 0.0
        
        valid_power = power_spectrum[valid_indices]
        valid_frequencies = frequencies[valid_indices]
        
        top_indices = np.argsort(valid_power)[-5:]
        
        signal_strength = 0.0
        cycle_info = []
        
        for idx in top_indices:
            freq = valid_frequencies[idx]
            amplitude = valid_power[idx]
            phase = np.angle(fft_result[valid_indices[idx]])
            
            cycle_period = 1.0 / abs(freq) if freq != 0 else float('inf')
            
            cycle_key = f"freq_{freq:.6f}_w{window_size}"
            cycle_info.append({
                'frequency': freq,
                'amplitude': amplitude,
                'phase': phase,
                'period': cycle_period,
                'window_size': window_size
            })
            
            if cycle_key in self.cycle_memory:
                data = self.cycle_memory[cycle_key]
                performance = data['performance']
                confidence = data['confidence']
                signal_strength += amplitude * performance * confidence
            else:
                signal_strength += amplitude * 0.05
        
        interference_signal = self._analyze_cycle_interference(cycle_info)
        self.dominant_cycles.append(cycle_info)
        
        normalized_signal = float(np.tanh(signal_strength / len(prices)))
        final_signal = normalized_signal + interference_signal
        
        return float(final_signal)

    def _analyze_cycle_interference(self, cycles: List[Dict]) -> float:
        if len(cycles) < 2:
            return 0.0
        
        interference_score = 0.0
        
        for i in range(len(cycles)):
            for j in range(i + 1, len(cycles)):
                cycle1, cycle2 = cycles[i], cycles[j]
                
                phase_diff = abs(cycle1['phase'] - cycle2['phase'])
                phase_diff = min(phase_diff, 2 * np.pi - phase_diff)
                
                freq_ratio = cycle1['frequency'] / (cycle2['frequency'] + 1e-8)
                
                harmonic_bonus = 0.0
                if abs(freq_ratio - round(freq_ratio)) < 0.1:
                    harmonic_bonus = 0.2
                
                if phase_diff < np.pi / 4:
                    amplitude_product = cycle1['amplitude'] * cycle2['amplitude']
                    interference_score += (amplitude_product * 0.1 + harmonic_bonus)
                
                elif phase_diff > 3 * np.pi / 4:
                    amplitude_diff = abs(cycle1['amplitude'] - cycle2['amplitude'])
                    interference_score -= amplitude_diff * 0.05
        
        return float(np.tanh(interference_score))

    def _analyze_seasonal_patterns(self, prices: List[float], timestamps: List[float]) -> float:
        if len(timestamps) != len(prices):
            return 0.0
        
        signal = 0.0
        
        try:
            datetimes = [datetime.fromtimestamp(ts) for ts in timestamps[-20:]]
            recent_prices = prices[-20:]
            
            hour_patterns = defaultdict(list)
            for dt, price in zip(datetimes, recent_prices):
                hour_patterns[dt.hour].append(price)
            
            dow_patterns = defaultdict(list)
            for dt, price in zip(datetimes, recent_prices):
                dow_patterns[dt.weekday()].append(price)
            
            current_hour = datetimes[-1].hour
            current_dow = datetimes[-1].weekday()
            
            if current_hour in self.seasonal_patterns:
                expected_performance = self.seasonal_patterns[current_hour].get('performance', 0.0)
                signal += expected_performance * 0.5
            
            if f"dow_{current_dow}" in self.seasonal_patterns:
                expected_performance = self.seasonal_patterns[f"dow_{current_dow}"].get('performance', 0.0)
                signal += expected_performance * 0.3
            
        except Exception as e:
            logger.warning(f"Seasonal analysis error: {e}")
        
        return float(np.tanh(signal))

    def _analyze_lunar_influence(self, prices: List[float], timestamps: List[float]) -> float:
        if len(timestamps) < 10:
            return 0.0
        
        try:
            lunar_cycle_length = 29.5 * 24 * 3600
            
            current_time = timestamps[-1]
            lunar_phase = (current_time % lunar_cycle_length) / lunar_cycle_length
            
            self.lunar_cycle_data.append({
                'phase': lunar_phase,
                'price_change': (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 and prices[-10] != 0 else 0
            })
            
            if len(self.lunar_cycle_data) >= 20:
                phases = [data['phase'] for data in self.lunar_cycle_data]
                changes = [data['price_change'] for data in self.lunar_cycle_data]
                
                correlation = np.corrcoef(phases, changes)[0, 1] if len(phases) > 1 else 0
                lunar_signal = correlation * np.sin(2 * np.pi * lunar_phase)
                return float(np.tanh(lunar_signal))
        
        except Exception as e:
            logger.warning(f"Lunar analysis error: {e}")
        
        return 0.0

    def learn_from_outcome(self, cycles_info: List[Dict], outcome: float):
        if not isinstance(cycles_info, list):
            logger.error(f"Temporal cycles_info is not a list: type={type(cycles_info)}, content={str(cycles_info)[:200]}")
            return
        if not cycles_info:
            return
        
        self.total_learning_events += 1
        
        cycles_learned = 0
        for cycle in cycles_info:
            freq = cycle['frequency']
            window_size = cycle.get('window_size', 64)
            cycle_key = f"freq_{freq:.6f}_w{window_size}"
            
            if cycle_key in self.cycle_memory:
                data = self.cycle_memory[cycle_key]
                
                # Adaptive learning rate based on cycle stability and market volatility
                cycle_stability = 1.0 / (1.0 + abs(cycle['amplitude'] - data['strength']))
                market_volatility = min(2.0, 1.0 + abs(outcome) * 3.0)  # Higher volatility = more conservative
                adaptive_learning_rate = (0.02 + 0.08 * data['confidence'] * cycle_stability) / market_volatility
                
                old_performance = data['performance']
                new_performance = data['performance'] + adaptive_learning_rate * (outcome - data['performance'])
                
                # Enhanced confidence update considering prediction accuracy and cycle consistency
                prediction_error = abs(outcome - data['performance'])
                accuracy_factor = 1.0 / (1.0 + prediction_error * 2.0)
                
                # Bonus confidence for cycles that consistently perform
                consistency_bonus = 0.0
                if hasattr(self, '_cycle_performance_history'):
                    if cycle_key in self._cycle_performance_history:
                        recent_performances = self._cycle_performance_history[cycle_key][-5:]
                        if len(recent_performances) >= 3:
                            consistency = 1.0 - np.std(recent_performances) / (np.mean(np.abs(recent_performances)) + 0.1)
                            consistency_bonus = max(0, consistency * 0.1)
                
                confidence_update = (0.01 + 0.04 * accuracy_factor + consistency_bonus) - (prediction_error * 0.05)
                new_confidence = max(0.1, min(1.0, data['confidence'] + confidence_update))
                
                # Track performance history for consistency analysis
                if not hasattr(self, '_cycle_performance_history'):
                    self._cycle_performance_history = {}
                if cycle_key not in self._cycle_performance_history:
                    self._cycle_performance_history[cycle_key] = deque(maxlen=10)
                self._cycle_performance_history[cycle_key].append(outcome)
                
                self.cycle_memory[cycle_key] = {
                    'strength': cycle['amplitude'],
                    'phase': cycle['phase'],
                    'performance': new_performance,
                    'confidence': new_confidence
                }
                cycles_learned += 1
            else:
                self.cycle_memory[cycle_key] = {
                    'strength': cycle['amplitude'],
                    'phase': cycle['phase'],
                    'performance': outcome * 0.3,
                    'confidence': 0.5
                }
                cycles_learned += 1
        
        self._update_seasonal_patterns(outcome)
        
        if len(cycles_info) >= 2:
            pattern_key = self._create_interference_pattern_key(cycles_info)
            if pattern_key not in self.interference_patterns:
                self.interference_patterns[pattern_key] = deque(maxlen=20)
            self.interference_patterns[pattern_key].append(outcome)

    def _update_seasonal_patterns(self, outcome: float):
        try:
            current_time = datetime.now()
            hour_key = current_time.hour
            dow_key = f"dow_{current_time.weekday()}"
            
            if hour_key not in self.seasonal_patterns:
                self.seasonal_patterns[hour_key] = {'performance': 0.0, 'count': 0}
            
            data = self.seasonal_patterns[hour_key]
            data['performance'] = (data['performance'] * data['count'] + outcome) / (data['count'] + 1)
            data['count'] += 1
            
            if dow_key not in self.seasonal_patterns:
                self.seasonal_patterns[dow_key] = {'performance': 0.0, 'count': 0}
            
            dow_data = self.seasonal_patterns[dow_key]
            dow_data['performance'] = (dow_data['performance'] * dow_data['count'] + outcome) / (dow_data['count'] + 1)
            dow_data['count'] += 1
            
        except Exception as e:
            logger.warning(f"Seasonal pattern update error: {e}")

    def _create_interference_pattern_key(self, cycles_info: List[Dict]) -> str:
        if len(cycles_info) < 2:
            return ""
        
        sorted_cycles = sorted(cycles_info, key=lambda x: x['frequency'])
        
        key_parts = []
        for cycle in sorted_cycles[:3]:
            freq_bucket = int(cycle['frequency'] * 1000) / 1000
            key_parts.append(f"f{freq_bucket}")
        
        return "_".join(key_parts)

    def get_cycle_stats(self) -> Dict:
        if not self.cycle_memory:
            return {}
        
        performances = [data['performance'] for data in self.cycle_memory.values()]
        confidences = [data['confidence'] for data in self.cycle_memory.values()]
        
        return {
            'total_cycles': len(self.cycle_memory),
            'avg_performance': np.mean(performances),
            'avg_confidence': np.mean(confidences),
            'seasonal_patterns': len(self.seasonal_patterns),
            'interference_patterns': len(self.interference_patterns)
        }