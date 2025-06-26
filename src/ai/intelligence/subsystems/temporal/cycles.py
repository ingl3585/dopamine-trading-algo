"""
Cycle Detection Engine - FFT analysis and interference modeling
"""

import numpy as np
import logging
from scipy import fft
from collections import deque
from typing import Dict, List

logger = logging.getLogger(__name__)

class CycleDetector:
    """
    Advanced cycle detection using FFT analysis with interference modeling
    """
    
    def fft_analysis(self, prices: List[float], window_size: int, cycle_memory: Dict) -> float:
        """Enhanced FFT analysis with adaptive cycle detection"""
        price_array = np.array(prices)
        
        # Detrend the data
        detrended = price_array - np.linspace(price_array[0], price_array[-1], len(price_array))
        
        # Apply window function to reduce spectral leakage
        windowed = detrended * np.hanning(len(detrended))
        
        # FFT analysis
        fft_result = fft.fft(windowed)
        frequencies = fft.fftfreq(len(windowed))
        power_spectrum = np.abs(fft_result)
        
        # Find dominant frequencies (excluding DC component)
        valid_indices = np.where((frequencies > 0) & (frequencies < 0.5))[0]
        if len(valid_indices) == 0:
            return 0.0
        
        valid_power = power_spectrum[valid_indices]
        valid_frequencies = frequencies[valid_indices]
        
        # Get top 5 frequencies
        top_indices = np.argsort(valid_power)[-5:]
        
        signal_strength = 0.0
        cycle_info = []
        
        for idx in top_indices:
            freq = valid_frequencies[idx]
            amplitude = valid_power[idx]
            phase = np.angle(fft_result[valid_indices[idx]])
            
            cycle_period = 1.0 / abs(freq) if freq != 0 else float('inf')
            
            # Store cycle information
            cycle_key = f"freq_{freq:.6f}_w{window_size}"
            cycle_info.append({
                'frequency': freq,
                'amplitude': amplitude,
                'phase': phase,
                'period': cycle_period,
                'window_size': window_size
            })
            
            # Get historical performance with confidence weighting
            if cycle_key in cycle_memory:
                data = cycle_memory[cycle_key]
                performance = data['performance']
                confidence = data['confidence']
                signal_strength += amplitude * performance * confidence
            else:
                # New cycle, conservative assumption
                signal_strength += amplitude * 0.05
        
        # Analyze cycle interference
        interference_signal = self._analyze_cycle_interference(cycle_info)
        
        # Normalize and combine signals
        normalized_signal = float(np.tanh(signal_strength / len(prices)))
        final_signal = normalized_signal + interference_signal
        
        return float(final_signal)

    def _analyze_cycle_interference(self, cycles: List[Dict]) -> float:
        """Advanced cycle interference analysis per prompt.txt"""
        if len(cycles) < 2:
            return 0.0
        
        interference_score = 0.0
        
        for i in range(len(cycles)):
            for j in range(i + 1, len(cycles)):
                cycle1, cycle2 = cycles[i], cycles[j]
                
                # Calculate phase relationship
                phase_diff = abs(cycle1['phase'] - cycle2['phase'])
                phase_diff = min(phase_diff, 2 * np.pi - phase_diff)  # Normalize to [0, π]
                
                # Calculate frequency relationship
                freq_ratio = cycle1['frequency'] / (cycle2['frequency'] + 1e-8)
                
                # Harmonic relationship bonus
                harmonic_bonus = 0.0
                if abs(freq_ratio - round(freq_ratio)) < 0.1:  # Near harmonic
                    harmonic_bonus = 0.2
                
                # Constructive interference (phases align)
                if phase_diff < np.pi / 4:
                    amplitude_product = cycle1['amplitude'] * cycle2['amplitude']
                    interference_score += (amplitude_product * 0.1 + harmonic_bonus)
                
                # Destructive interference (phases oppose)
                elif phase_diff > 3 * np.pi / 4:
                    amplitude_diff = abs(cycle1['amplitude'] - cycle2['amplitude'])
                    interference_score -= amplitude_diff * 0.05
        
        return float(np.tanh(interference_score))