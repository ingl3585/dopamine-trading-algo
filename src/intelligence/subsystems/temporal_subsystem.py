import numpy as np
import logging
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Optional

# Import dependency manager for clean dependency handling
try:
    from ...core.dependency_manager import dependency_manager
    
    # Conditional SciPy import with fallback
    if dependency_manager.is_available('scipy'):
        from scipy import fft
    else:
        # Use NumPy-based fallback for FFT operations
        fft = dependency_manager.get_fallback('scipy_fft')
        logging.getLogger(__name__).info("Using NumPy-based FFT fallback for SciPy functionality")
        
except ImportError:
    # Fallback if dependency manager is not available
    try:
        from scipy import fft
    except ImportError:
        # Create basic NumPy FFT fallback inline
        import numpy as np
        
        class FFTFallback:
            @staticmethod
            def fft(data):
                return np.fft.fft(data)
            
            @staticmethod
            def fftfreq(n, d=1.0):
                return np.fft.fftfreq(n, d)
        
        fft = FFTFallback()
        logging.getLogger(__name__).warning("SciPy not available, using basic NumPy FFT fallback")

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
                        # Convert deque to list for safe slicing
                        performance_history = list(self._cycle_performance_history[cycle_key])
                        recent_performances = performance_history[-5:] if len(performance_history) >= 5 else performance_history
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

    def analyze_temporal_patterns(self, prices: List[float], timestamps: List[float]) -> float:
        """
        Enhanced temporal pattern analysis using FFT across multiple timeframes.
        
        This method extends the existing analyze_cycles functionality by:
        1. Performing comprehensive FFT analysis across multiple time windows
        2. Integrating seasonal and lunar cycle detection
        3. Analyzing cycle interference patterns
        4. Returning a normalized temporal signal for orchestrator processing
        
        Args:
            prices: List of price values for analysis
            timestamps: List of corresponding timestamps
            
        Returns:
            float: Normalized temporal signal between -1.0 and 1.0
                  Positive values indicate favorable temporal patterns
                  Negative values indicate unfavorable patterns
                  
        Raises:
            ValueError: If prices and timestamps have different lengths
            TypeError: If inputs are not lists or contain invalid data types
        """
        # Input validation following clean architecture principles
        if not isinstance(prices, list) or not isinstance(timestamps, list):
            raise TypeError("Prices and timestamps must be lists")
        
        if len(prices) != len(timestamps):
            raise ValueError(f"Prices ({len(prices)}) and timestamps ({len(timestamps)}) must have same length")
        
        if len(prices) < 8:
            logger.debug(f"Insufficient data for temporal pattern analysis: {len(prices)} prices (need 8+)")
            return 0.0
        
        try:
            # Delegate to existing analyze_cycles method for core FFT functionality
            # This follows the Single Responsibility Principle - reuse existing validated logic
            base_temporal_signal = self.analyze_cycles(prices, timestamps)
            
            # Enhanced pattern detection using multiple analytical approaches
            cycle_coherence_signal = self._analyze_cycle_coherence(prices)
            phase_alignment_signal = self._analyze_phase_alignment()
            temporal_momentum_signal = self._analyze_temporal_momentum(prices, timestamps)
            
            # Weighted combination of signals using domain expertise
            # Weights based on empirical analysis of signal reliability
            combined_signal = (
                base_temporal_signal * 0.5 +        # Core FFT analysis - highest weight
                cycle_coherence_signal * 0.2 +      # Cycle stability assessment
                phase_alignment_signal * 0.2 +      # Phase relationship analysis
                temporal_momentum_signal * 0.1      # Temporal trend detection
            )
            
            # Apply normalization to ensure output is in [-1.0, 1.0] range
            # Using tanh for smooth normalization that preserves signal strength
            normalized_signal = float(np.tanh(combined_signal * 2.0))
            
            # Ensure output is within expected bounds (defensive programming)
            if np.isnan(normalized_signal) or np.isinf(normalized_signal):
                logger.warning("Invalid temporal pattern signal detected, returning baseline")
                return 0.0
            
            # Clamp to exact bounds for safety
            normalized_signal = max(-1.0, min(1.0, normalized_signal))
            
            logger.debug(f"Temporal pattern analysis complete: signal={normalized_signal:.4f}")
            return normalized_signal
            
        except Exception as e:
            logger.error(f"Error in temporal pattern analysis: {e}")
            return 0.0
    
    def _analyze_cycle_coherence(self, prices: List[float]) -> float:
        """
        Analyze coherence between different cycle patterns.
        
        Coherence measures how well different cycles align and reinforce each other,
        providing insight into pattern reliability and strength.
        """
        try:
            if len(self.dominant_cycles) < 2:
                return 0.0
            
            recent_cycles = list(self.dominant_cycles)[-5:]  # Analyze recent cycle patterns
            coherence_scores = []
            
            for cycle_group in recent_cycles:
                if len(cycle_group) >= 2:
                    # Calculate inter-cycle coherence within each group
                    frequencies = [cycle['frequency'] for cycle in cycle_group]
                    amplitudes = [cycle['amplitude'] for cycle in cycle_group]
                    
                    # Coherence based on frequency relationships
                    freq_coherence = self._calculate_frequency_coherence(frequencies)
                    
                    # Coherence based on amplitude consistency
                    amp_coherence = self._calculate_amplitude_coherence(amplitudes)
                    
                    # Combined coherence score
                    group_coherence = (freq_coherence + amp_coherence) / 2.0
                    coherence_scores.append(group_coherence)
            
            if coherence_scores:
                return float(np.mean(coherence_scores))
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Error in cycle coherence analysis: {e}")
            return 0.0
    
    def _calculate_frequency_coherence(self, frequencies: List[float]) -> float:
        """Calculate coherence between cycle frequencies"""
        if len(frequencies) < 2:
            return 0.0
        
        # Look for harmonic relationships between frequencies
        coherence = 0.0
        for i in range(len(frequencies)):
            for j in range(i + 1, len(frequencies)):
                ratio = frequencies[i] / (frequencies[j] + 1e-8)
                # Check if ratio is close to a simple harmonic (2:1, 3:1, etc.)
                nearest_harmonic = round(ratio)
                if nearest_harmonic > 0:
                    harmonic_error = abs(ratio - nearest_harmonic) / nearest_harmonic
                    if harmonic_error < 0.1:  # Within 10% of perfect harmonic
                        coherence += (1.0 - harmonic_error)
        
        # Normalize by number of frequency pairs
        num_pairs = len(frequencies) * (len(frequencies) - 1) / 2
        return coherence / max(1, num_pairs)
    
    def _calculate_amplitude_coherence(self, amplitudes: List[float]) -> float:
        """Calculate coherence between cycle amplitudes"""
        if len(amplitudes) < 2:
            return 0.0
        
        # Coherence based on amplitude consistency (lower variance = higher coherence)
        amplitude_variance = np.var(amplitudes)
        amplitude_mean = np.mean(amplitudes)
        
        if amplitude_mean > 0:
            # Coefficient of variation (normalized variance)
            cv = np.sqrt(amplitude_variance) / amplitude_mean
            # Convert to coherence score (lower CV = higher coherence)
            coherence = 1.0 / (1.0 + cv)
        else:
            coherence = 0.0
        
        return coherence
    
    def _analyze_phase_alignment(self) -> float:
        """
        Analyze phase alignment between recent cycle patterns.
        
        Well-aligned phases indicate strong, reinforcing temporal patterns.
        """
        try:
            if len(self.dominant_cycles) < 2:
                return 0.0
            
            recent_cycles = list(self.dominant_cycles)[-3:]  # Recent cycle data
            phase_alignment_scores = []
            
            for cycle_group in recent_cycles:
                if len(cycle_group) >= 2:
                    phases = [cycle['phase'] for cycle in cycle_group]
                    
                    # Calculate phase clustering using circular statistics
                    # Convert phases to unit vectors and find average direction
                    cos_phases = [np.cos(phase) for phase in phases]
                    sin_phases = [np.sin(phase) for phase in phases]
                    
                    mean_cos = np.mean(cos_phases)
                    mean_sin = np.mean(sin_phases)
                    
                    # Vector strength indicates phase alignment
                    vector_strength = np.sqrt(mean_cos**2 + mean_sin**2)
                    phase_alignment_scores.append(vector_strength)
            
            if phase_alignment_scores:
                return float(np.mean(phase_alignment_scores))
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Error in phase alignment analysis: {e}")
            return 0.0
    
    def _analyze_temporal_momentum(self, prices: List[float], timestamps: List[float]) -> float:
        """
        Analyze temporal momentum in cycle patterns.
        
        Examines whether cycle patterns are strengthening or weakening over time.
        """
        try:
            if len(prices) < 16:  # Need sufficient data for trend analysis
                return 0.0
            
            # Split data into two halves to analyze temporal change
            mid_point = len(prices) // 2
            early_prices = prices[:mid_point + 4]  # Overlap for continuity
            late_prices = prices[mid_point:]
            
            # Perform FFT analysis on both periods
            early_signal = self._fft_analysis(early_prices, len(early_prices))
            late_signal = self._fft_analysis(late_prices, len(late_prices))
            
            # Calculate momentum as change in signal strength
            signal_change = late_signal - early_signal
            
            # Apply momentum decay factor based on time difference
            time_span = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 1.0
            decay_factor = np.exp(-time_span / 3600.0)  # Decay over 1 hour
            
            momentum = signal_change * decay_factor
            
            # Normalize momentum signal
            return float(np.tanh(momentum))
            
        except Exception as e:
            logger.warning(f"Error in temporal momentum analysis: {e}")
            return 0.0

    def detect_cycles(self, prices: List[float]) -> List[Dict]:
        """
        Extract detailed cycle information from FFT analysis.
        
        This method provides comprehensive cycle detection and analysis,
        extracting specific cycle patterns from price data. Following clean
        architecture principles, it builds upon existing FFT functionality
        to provide detailed cycle characterization.
        
        Args:
            prices: List of price values for cycle detection
            
        Returns:
            List[Dict]: List of detected cycles, each containing:
                frequency: Cycle frequency (cycles per sample)
                period: Cycle period in samples
                amplitude: Cycle strength/amplitude
                phase: Cycle phase offset
                confidence: Detection confidence (0.0 to 1.0)
                harmonic_order: Harmonic relationship indicator
                stability: Cycle stability measure
                
        Raises:
            TypeError: If prices is not a list
            ValueError: If prices contains invalid data
        """
        # Input validation following defensive programming principles
        if not isinstance(prices, list):
            raise TypeError(f"Prices must be a list, got {type(prices)}")
        
        if len(prices) < 8:
            logger.debug(f"Insufficient data for cycle detection: {len(prices)} prices (need 8+)")
            return []
        
        if not all(isinstance(p, (int, float)) for p in prices):
            raise ValueError("All prices must be numeric values")
        
        try:
            # Perform FFT analysis to detect underlying cycles
            cycles = self._perform_comprehensive_cycle_detection(prices)
            
            # Filter and rank cycles by significance
            significant_cycles = self._filter_significant_cycles(cycles)
            
            # Enhance cycle information with additional metrics
            enhanced_cycles = self._enhance_cycle_information(significant_cycles, prices)
            
            # Sort cycles by strength/significance
            enhanced_cycles.sort(key=lambda x: x['amplitude'] * x['confidence'], reverse=True)
            
            logger.debug(f"Detected {len(enhanced_cycles)} significant cycles from {len(prices)} price points")
            return enhanced_cycles
            
        except Exception as e:
            logger.error(f"Error in cycle detection: {e}")
            return []
    
    def _perform_comprehensive_cycle_detection(self, prices: List[float]) -> List[Dict]:
        """Perform comprehensive FFT-based cycle detection"""
        try:
            price_array = np.array(prices)
            
            # Detrend the data to focus on cyclical components
            detrended = price_array - np.linspace(price_array[0], price_array[-1], len(price_array))
            
            # Apply window function to reduce spectral leakage
            windowed = detrended * np.hanning(len(detrended))
            
            # Perform FFT analysis
            fft_result = fft.fft(windowed)
            frequencies = fft.fftfreq(len(windowed))
            power_spectrum = np.abs(fft_result)
            
            # Focus on positive frequencies only (negative frequencies are redundant)
            valid_indices = np.where((frequencies > 0) & (frequencies < 0.5))[0]
            
            if len(valid_indices) == 0:
                return []
            
            cycles = []
            
            # Extract cycle information for each frequency component
            for idx in valid_indices:
                freq = frequencies[idx]
                amplitude = power_spectrum[idx]
                phase = np.angle(fft_result[idx])
                
                # Calculate cycle period
                period = 1.0 / abs(freq) if freq != 0 else float('inf')
                
                # Skip very short or very long cycles
                if period < 2 or period > len(prices) / 2:
                    continue
                
                # Calculate confidence based on amplitude relative to noise floor
                noise_floor = np.median(power_spectrum[valid_indices])
                confidence = min(1.0, amplitude / (noise_floor + 1e-8))
                
                cycle_info = {
                    'frequency': float(freq),
                    'period': float(period),
                    'amplitude': float(amplitude),
                    'phase': float(phase),
                    'confidence': float(confidence),
                    'harmonic_order': 0,  # Will be calculated later
                    'stability': 0.0      # Will be calculated later
                }
                
                cycles.append(cycle_info)
            
            return cycles
            
        except Exception as e:
            logger.warning(f"Error in comprehensive cycle detection: {e}")
            return []
    
    def _filter_significant_cycles(self, cycles: List[Dict]) -> List[Dict]:
        """Filter cycles to keep only statistically significant ones"""
        if not cycles:
            return []
        
        try:
            # Calculate amplitude threshold based on statistical significance
            amplitudes = [cycle['amplitude'] for cycle in cycles]
            median_amplitude = np.median(amplitudes)
            std_amplitude = np.std(amplitudes)
            
            # Threshold: cycles must be at least 1.5 standard deviations above median
            amplitude_threshold = median_amplitude + 1.5 * std_amplitude
            
            # Filter by amplitude and confidence
            significant_cycles = []
            for cycle in cycles:
                if (cycle['amplitude'] > amplitude_threshold and 
                    cycle['confidence'] > 0.3 and
                    cycle['period'] >= 3):  # Minimum meaningful period
                    significant_cycles.append(cycle)
            
            # Limit to top 10 cycles to avoid noise
            if len(significant_cycles) > 10:
                significant_cycles.sort(key=lambda x: x['amplitude'], reverse=True)
                significant_cycles = significant_cycles[:10]
            
            return significant_cycles
            
        except Exception as e:
            logger.warning(f"Error filtering significant cycles: {e}")
            return cycles  # Return all cycles if filtering fails
    
    def _enhance_cycle_information(self, cycles: List[Dict], prices: List[float]) -> List[Dict]:
        """Enhance cycle information with additional metrics"""
        try:
            enhanced_cycles = []
            
            for cycle in cycles:
                enhanced_cycle = cycle.copy()
                
                # Calculate harmonic relationships
                enhanced_cycle['harmonic_order'] = self._determine_harmonic_order(cycle, cycles)
                
                # Calculate stability based on historical performance
                enhanced_cycle['stability'] = self._calculate_cycle_stability(cycle, prices)
                
                # Add cycle strength indicator (normalized amplitude)
                max_amplitude = max(c['amplitude'] for c in cycles) if cycles else 1.0
                enhanced_cycle['strength'] = cycle['amplitude'] / max_amplitude
                
                # Add cycle classification
                enhanced_cycle['classification'] = self._classify_cycle_type(cycle)
                
                # Add predictive power estimate
                enhanced_cycle['predictive_power'] = self._estimate_predictive_power(cycle, prices)
                
                enhanced_cycles.append(enhanced_cycle)
            
            return enhanced_cycles
            
        except Exception as e:
            logger.warning(f"Error enhancing cycle information: {e}")
            return cycles
    
    def _determine_harmonic_order(self, target_cycle: Dict, all_cycles: List[Dict]) -> int:
        """Determine if cycle is a harmonic of another cycle"""
        try:
            target_freq = target_cycle['frequency']
            
            for other_cycle in all_cycles:
                if other_cycle is target_cycle:
                    continue
                
                other_freq = other_cycle['frequency']
                
                # Check if target frequency is a harmonic of other frequency
                if other_freq > 0:
                    ratio = target_freq / other_freq
                    harmonic_order = round(ratio)
                    
                    # If ratio is close to an integer, it's likely a harmonic
                    if abs(ratio - harmonic_order) < 0.1 and harmonic_order > 1:
                        return harmonic_order
            
            return 1  # Fundamental frequency
            
        except Exception:
            return 1
    
    def _calculate_cycle_stability(self, cycle: Dict, prices: List[float]) -> float:
        """Calculate cycle stability based on amplitude consistency"""
        try:
            period = cycle['period']
            
            if period <= 0 or period >= len(prices):
                return 0.0
            
            # Calculate cycle amplitudes across different windows
            window_size = int(period * 3)  # 3 cycles per window
            if window_size >= len(prices):
                return cycle['confidence']  # Fallback to confidence
            
            amplitudes = []
            for start in range(0, len(prices) - window_size, window_size // 2):
                window_prices = prices[start:start + window_size]
                
                try:
                    # Simplified amplitude calculation for this window
                    detrended = np.array(window_prices) - np.mean(window_prices)
                    window_amplitude = np.std(detrended)
                    amplitudes.append(window_amplitude)
                except Exception:
                    continue
            
            if len(amplitudes) < 2:
                return cycle['confidence']
            
            # Stability is inverse of coefficient of variation
            mean_amplitude = np.mean(amplitudes)
            std_amplitude = np.std(amplitudes)
            
            if mean_amplitude > 0:
                cv = std_amplitude / mean_amplitude
                stability = 1.0 / (1.0 + cv)
            else:
                stability = 0.0
            
            return float(np.clip(stability, 0.0, 1.0))
            
        except Exception:
            return 0.5  # Default stability
    
    def _classify_cycle_type(self, cycle: Dict) -> str:
        """Classify cycle based on its characteristics"""
        try:
            period = cycle['period']
            
            if period < 5:
                return 'short_term'
            elif period < 20:
                return 'medium_term'
            elif period < 60:
                return 'long_term'
            else:
                return 'very_long_term'
                
        except Exception:
            return 'unknown'
    
    def _estimate_predictive_power(self, cycle: Dict, prices: List[float]) -> float:
        """Estimate predictive power of the cycle"""
        try:
            # Predictive power based on cycle characteristics
            amplitude_factor = min(1.0, cycle['amplitude'] / np.std(prices))
            confidence_factor = cycle['confidence']
            stability_factor = cycle.get('stability', 0.5)
            
            # Longer cycles generally have more predictive power
            period_factor = min(1.0, cycle['period'] / 20.0)
            
            # Combined predictive power
            predictive_power = (amplitude_factor + confidence_factor + 
                              stability_factor + period_factor) / 4.0
            
            return float(np.clip(predictive_power, 0.0, 1.0))
            
        except Exception:
            return 0.5

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