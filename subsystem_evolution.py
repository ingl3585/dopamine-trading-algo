# subsystem_evolution.py

import numpy as np
import torch
from scipy import fft
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Tuple, Set, Optional
import logging
import random

logger = logging.getLogger(__name__)


class AdvancedDNASubsystem:
    def __init__(self):
        # 16-base DNA encoding with enhanced market signatures
        self.bases = {
            'A': 'price_up_vol_high',      'B': 'price_up_vol_med',
            'C': 'price_up_vol_low',       'D': 'price_up_vol_very_low',
            'E': 'price_down_vol_high',    'F': 'price_down_vol_med', 
            'G': 'price_down_vol_low',     'H': 'price_down_vol_very_low',
            'I': 'price_flat_vol_high',    'J': 'price_flat_vol_med',
            'K': 'price_flat_vol_low',     'L': 'momentum_strong_up',
            'M': 'momentum_strong_down',   'N': 'volatility_spike',
            'O': 'volatility_crush',       'P': 'pattern_continuation'
        }
        
        # Enhanced pattern storage with breeding and aging
        self.sequences = {}  # sequence -> {'performance': float, 'age': int, 'generation': int, 'parents': tuple}
        self.breeding_pool = []
        self.elite_sequences = {}  # Top performing sequences
        self.mutation_rate = 0.05
        self.breeding_frequency = 100
        self.max_age = 1000
        self.generation_count = 0
        
        # DNA aging and evolution tracking
        self.age_decay_factor = 0.999
        self.performance_threshold = 0.6
        self.breeding_success_rate = deque(maxlen=50)
        
    def encode_market_state(self, prices: List[float], volumes: List[float], 
                           volatility: Optional[float] = None, momentum: Optional[float] = None) -> str:
        if len(prices) < 2 or len(volumes) < 2:
            return ""
        
        sequence = ""
        
        # Calculate additional market context
        if volatility is None:
            volatility = np.std(prices[-10:]) / np.mean(prices[-10:]) if len(prices) >= 10 else 0
        if momentum is None:
            momentum = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 and prices[-5] != 0 else 0
        
        for i in range(1, len(prices)):
            price_change = (prices[i] - prices[i-1]) / prices[i-1] if prices[i-1] != 0 else 0
            vol_ratio = volumes[i] / volumes[i-1] if volumes[i-1] > 0 else 1.0
            
            # Enhanced 16-base encoding with market context
            base = self._determine_base(price_change, vol_ratio, volatility, momentum, i, prices, volumes)
            sequence += base
        
        return sequence
    
    def _determine_base(self, price_change: float, vol_ratio: float, volatility: float, 
                       momentum: float, index: int, prices: List[float], volumes: List[float]) -> str:
        """Determine DNA base using enhanced 16-base encoding"""
        
        # Strong momentum patterns override basic price/volume
        if abs(momentum) > 0.02:
            if momentum > 0:
                return 'L'  # Strong upward momentum
            else:
                return 'M'  # Strong downward momentum
        
        # Volatility regime patterns
        if volatility > 0.03:  # High volatility
            return 'N'  # Volatility spike
        elif volatility < 0.005:  # Very low volatility
            return 'O'  # Volatility crush
        
        # Pattern continuation detection
        if index >= 3:
            recent_changes = [(prices[j] - prices[j-1]) / prices[j-1] 
                            for j in range(index-2, index+1) if prices[j-1] != 0]
            if len(recent_changes) >= 3:
                if all(change > 0 for change in recent_changes) or all(change < 0 for change in recent_changes):
                    return 'P'  # Pattern continuation
        
        # Standard price/volume encoding
        if abs(price_change) < 0.0001:  # Flat price
            if vol_ratio > 2.0:
                return 'I'
            elif vol_ratio > 1.2:
                return 'J'
            else:
                return 'K'
        elif price_change > 0:  # Price up
            if vol_ratio > 2.0:
                return 'A'
            elif vol_ratio > 1.2:
                return 'B'
            elif vol_ratio > 0.8:
                return 'C'
            else:
                return 'D'
        else:  # Price down
            if vol_ratio > 2.0:
                return 'E'
            elif vol_ratio > 1.2:
                return 'F'
            elif vol_ratio > 0.8:
                return 'G'
            else:
                return 'H'
    
    def analyze_sequence(self, sequence: str) -> float:
        if not sequence or len(sequence) < 5:
            return 0.0
        
        # Age all sequences
        self._age_sequences()
        
        # Find best matching patterns with breeding consideration
        best_score = 0.0
        best_matches = []
        
        for stored_seq, data in self.sequences.items():
            similarity = self._advanced_sequence_similarity(sequence, stored_seq)
            if similarity > 0.7:
                # Age-adjusted performance
                age_factor = self.age_decay_factor ** data['age']
                adjusted_performance = data['performance'] * age_factor
                score = adjusted_performance * similarity
                
                if abs(score) > abs(best_score):
                    best_score = score
                    best_matches.append((stored_seq, score, similarity))
        
        # Consider breeding if we have good matches
        if len(best_matches) >= 2 and len(self.sequences) % self.breeding_frequency == 0:
            self._attempt_breeding(best_matches)
        
        return best_score
    
    def _advanced_sequence_similarity(self, seq1: str, seq2: str) -> float:
        if not seq1 or not seq2:
            return 0.0
        
        # Multiple similarity metrics
        exact_similarity = self._exact_similarity(seq1, seq2)
        structural_similarity = self._structural_similarity(seq1, seq2)
        functional_similarity = self._functional_similarity(seq1, seq2)
        
        # Weighted combination
        return (exact_similarity * 0.4 + structural_similarity * 0.3 + functional_similarity * 0.3)
    
    def _exact_similarity(self, seq1: str, seq2: str) -> float:
        """Exact character matching similarity"""
        max_len = max(len(seq1), len(seq2))
        min_len = min(len(seq1), len(seq2))
        
        matches = sum(1 for i in range(min_len) if seq1[i] == seq2[i])
        return matches / max_len
    
    def _structural_similarity(self, seq1: str, seq2: str) -> float:
        """Pattern structure similarity (trends, reversals, etc.)"""
        def get_structure(seq):
            structure = []
            for i in range(len(seq) - 1):
                current_type = self._get_base_type(seq[i])
                next_type = self._get_base_type(seq[i+1])
                
                if current_type != next_type:
                    structure.append('TRANSITION')
                elif current_type == 'UP':
                    structure.append('TREND_UP')
                elif current_type == 'DOWN':
                    structure.append('TREND_DOWN')
                else:
                    structure.append('CONTINUATION')
            return structure
        
        struct1 = get_structure(seq1)
        struct2 = get_structure(seq2)
        
        if not struct1 or not struct2:
            return 0.0
        
        matches = sum(1 for i in range(min(len(struct1), len(struct2))) 
                     if struct1[i] == struct2[i])
        
        return matches / max(len(struct1), len(struct2))
    
    def _functional_similarity(self, seq1: str, seq2: str) -> float:
        """Functional similarity based on market behavior"""
        def get_market_signature(seq):
            up_count = sum(1 for base in seq if base in 'ABCDL')
            down_count = sum(1 for base in seq if base in 'EFGHM')
            vol_spike_count = sum(1 for base in seq if base in 'AEIN')
            continuation_count = sum(1 for base in seq if base == 'P')
            
            total = len(seq)
            if total == 0:
                return [0, 0, 0, 0]
            
            return [up_count/total, down_count/total, vol_spike_count/total, continuation_count/total]
        
        sig1 = get_market_signature(seq1)
        sig2 = get_market_signature(seq2)
        
        # Cosine similarity between signatures
        dot_product = sum(a * b for a, b in zip(sig1, sig2))
        norm1 = sum(a * a for a in sig1) ** 0.5
        norm2 = sum(b * b for b in sig2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _get_base_type(self, base: str) -> str:
        """Get the general type of a DNA base"""
        if base in 'ABCDL':
            return 'UP'
        elif base in 'EFGHM':
            return 'DOWN'
        elif base in 'IJK':
            return 'FLAT'
        elif base in 'NO':
            return 'VOLATILITY'
        else:
            return 'SPECIAL'
    
    def _age_sequences(self):
        """Age all sequences and remove very old ones"""
        to_remove = []
        for seq, data in self.sequences.items():
            data['age'] += 1
            if data['age'] > self.max_age and data['performance'] < 0.3:
                to_remove.append(seq)
        
        for seq in to_remove:
            del self.sequences[seq]
    
    def _attempt_breeding(self, matches: List[Tuple[str, float, float]]):
        """Attempt to breed successful sequences"""
        if len(matches) < 2:
            return
        
        # Select top 2 sequences for breeding
        matches.sort(key=lambda x: x[1], reverse=True)
        parent1_seq, parent1_score, _ = matches[0]
        parent2_seq, parent2_score, _ = matches[1]
        
        # Only breed if both parents are successful
        if parent1_score > 0.3 and parent2_score > 0.3:
            offspring = self._breed_sequences(parent1_seq, parent2_seq)
            if offspring and offspring not in self.sequences:
                self.generation_count += 1
                self.sequences[offspring] = {
                    'performance': (parent1_score + parent2_score) / 2 * 0.8,  # Slightly reduced initial performance
                    'age': 0,
                    'generation': self.generation_count,
                    'parents': (parent1_seq, parent2_seq)
                }
                self.breeding_pool.append(offspring)
                self.breeding_success_rate.append(1.0)
                
                logger.debug(f"DNA breeding successful: {offspring[:10]}... (Gen {self.generation_count})")
            else:
                self.breeding_success_rate.append(0.0)
    
    def _breed_sequences(self, parent1: str, parent2: str) -> str:
        """Advanced breeding with multiple crossover strategies"""
        if not parent1 or not parent2:
            return ""
        
        min_len = min(len(parent1), len(parent2))
        if min_len < 4:
            return ""
        
        # Multiple breeding strategies
        strategy = random.choice(['single_point', 'two_point', 'uniform', 'functional'])
        
        if strategy == 'single_point':
            crossover_point = random.randint(1, min_len - 1)
            offspring = parent1[:crossover_point] + parent2[crossover_point:]
        
        elif strategy == 'two_point':
            point1 = random.randint(1, min_len // 2)
            point2 = random.randint(min_len // 2, min_len - 1)
            offspring = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        
        elif strategy == 'uniform':
            offspring = ""
            for i in range(min_len):
                offspring += parent1[i] if random.random() < 0.5 else parent2[i]
        
        else:  # functional breeding
            offspring = self._functional_breeding(parent1, parent2)
        
        # Apply mutation
        if random.random() < self.mutation_rate:
            offspring = self._mutate_sequence(offspring)
        
        return offspring
    
    def _functional_breeding(self, parent1: str, parent2: str) -> str:
        """Breeding based on functional market patterns"""
        # Identify functional blocks in each parent
        blocks1 = self._identify_functional_blocks(parent1)
        blocks2 = self._identify_functional_blocks(parent2)
        
        # Combine best blocks from each parent
        offspring = ""
        for i in range(min(len(blocks1), len(blocks2))):
            # Choose block based on functional strength
            if self._evaluate_block_strength(blocks1[i]) > self._evaluate_block_strength(blocks2[i]):
                offspring += blocks1[i]
            else:
                offspring += blocks2[i]
        
        return offspring
    
    def _identify_functional_blocks(self, sequence: str) -> List[str]:
        """Identify functional blocks within a sequence"""
        blocks = []
        current_block = ""
        current_type = None
        
        for base in sequence:
            base_type = self._get_base_type(base)
            
            if current_type is None or base_type == current_type:
                current_block += base
                current_type = base_type
            else:
                if current_block:
                    blocks.append(current_block)
                current_block = base
                current_type = base_type
        
        if current_block:
            blocks.append(current_block)
        
        return blocks
    
    def _evaluate_block_strength(self, block: str) -> float:
        """Evaluate the strength of a functional block"""
        if not block:
            return 0.0
        
        # Longer consistent blocks are generally stronger
        consistency_score = len(block) / 10.0
        
        # Special patterns get bonus
        special_bonus = 0.0
        if 'L' in block or 'M' in block:  # Momentum patterns
            special_bonus += 0.3
        if 'P' in block:  # Continuation patterns
            special_bonus += 0.2
        if 'N' in block or 'O' in block:  # Volatility patterns
            special_bonus += 0.1
        
        return min(1.0, consistency_score + special_bonus)
    
    def _mutate_sequence(self, sequence: str) -> str:
        """Advanced mutation with context awareness"""
        if not sequence:
            return sequence
        
        mutation_strategies = ['point', 'insertion', 'deletion', 'inversion']
        strategy = random.choice(mutation_strategies)
        
        if strategy == 'point':
            # Point mutation
            mutation_point = random.randint(0, len(sequence) - 1)
            new_base = random.choice(list(self.bases.keys()))
            return sequence[:mutation_point] + new_base + sequence[mutation_point+1:]
        
        elif strategy == 'insertion':
            # Insert a new base
            insertion_point = random.randint(0, len(sequence))
            new_base = random.choice(list(self.bases.keys()))
            return sequence[:insertion_point] + new_base + sequence[insertion_point:]
        
        elif strategy == 'deletion':
            # Delete a base
            if len(sequence) > 1:
                deletion_point = random.randint(0, len(sequence) - 1)
                return sequence[:deletion_point] + sequence[deletion_point+1:]
        
        elif strategy == 'inversion':
            # Invert a segment
            if len(sequence) > 3:
                start = random.randint(0, len(sequence) - 3)
                end = random.randint(start + 2, len(sequence))
                return sequence[:start] + sequence[start:end][::-1] + sequence[end:]
        
        return sequence
    
    def learn_from_outcome(self, sequence: str, outcome: float):
        if not sequence:
            return
        
        # Update or add sequence
        if sequence in self.sequences:
            data = self.sequences[sequence]
            learning_rate = 0.1 / (1 + data['age'] * 0.01)  # Slower learning for older sequences
            data['performance'] += learning_rate * (outcome - data['performance'])
            data['age'] = max(0, data['age'] - 5)  # Rejuvenate successful sequences
        else:
            self.sequences[sequence] = {
                'performance': outcome * 0.5,
                'age': 0,
                'generation': 0,
                'parents': None
            }
        
        # Update elite sequences
        if outcome > self.performance_threshold:
            self.elite_sequences[sequence] = self.sequences[sequence].copy()
        
        # Cleanup poor performers
        self._cleanup_sequences()
    
    def _cleanup_sequences(self):
        """Remove poor performing and very old sequences"""
        if len(self.sequences) > 2000:
            # Sort by adjusted performance (performance * age_factor)
            sequence_scores = []
            for seq, data in self.sequences.items():
                age_factor = self.age_decay_factor ** data['age']
                adjusted_score = data['performance'] * age_factor
                sequence_scores.append((seq, adjusted_score))
            
            # Keep top 1500 sequences
            sequence_scores.sort(key=lambda x: x[1], reverse=True)
            sequences_to_keep = {seq for seq, _ in sequence_scores[:1500]}
            
            # Always keep elite sequences
            sequences_to_keep.update(self.elite_sequences.keys())
            
            # Remove poor performers
            self.sequences = {seq: data for seq, data in self.sequences.items() 
                            if seq in sequences_to_keep}
    
    def get_evolution_stats(self) -> Dict:
        """Get detailed evolution statistics"""
        if not self.sequences:
            return {}
        
        performances = [data['performance'] for data in self.sequences.values()]
        ages = [data['age'] for data in self.sequences.values()]
        generations = [data['generation'] for data in self.sequences.values()]
        
        return {
            'total_sequences': len(self.sequences),
            'elite_sequences': len(self.elite_sequences),
            'avg_performance': np.mean(performances),
            'best_performance': np.max(performances),
            'avg_age': np.mean(ages),
            'max_generation': max(generations) if generations else 0,
            'breeding_success_rate': np.mean(list(self.breeding_success_rate)) if self.breeding_success_rate else 0.0,
            'breeding_pool_size': len(self.breeding_pool)
        }


class FFTTemporalSubsystem:
    def __init__(self):
        self.cycle_memory = {}  # frequency -> {'strength': float, 'phase': float, 'performance': float, 'confidence': float}
        self.dominant_cycles = deque(maxlen=100)
        self.interference_patterns = {}
        self.cycle_predictions = deque(maxlen=50)
        
        # Adaptive cycle tracking
        self.cycle_importance_weights = {}
        self.seasonal_patterns = {}
        self.lunar_cycle_data = deque(maxlen=30)  # Track lunar influence
        
    def analyze_cycles(self, prices: List[float], timestamps: Optional[List[float]] = None) -> float:
        if len(prices) < 32:
            return 0.0
        
        # Enhanced FFT analysis with multiple window sizes
        signals = []
        
        for window_size in [64, 128, 256]:
            if len(prices) >= window_size:
                signal = self._fft_analysis(prices[-window_size:], window_size)
                signals.append(signal)
        
        if not signals:
            return 0.0
        
        # Combine signals from different timeframes
        combined_signal = float(np.mean(signals))
        
        # Add seasonal and lunar analysis if timestamps available
        if timestamps:
            seasonal_signal = self._analyze_seasonal_patterns(prices, timestamps)
            lunar_signal = self._analyze_lunar_influence(prices, timestamps)
            combined_signal = combined_signal * 0.7 + seasonal_signal * 0.2 + lunar_signal * 0.1
        
        return combined_signal
    
    def _fft_analysis(self, prices: List[float], window_size: int) -> float:
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
            if cycle_key in self.cycle_memory:
                data = self.cycle_memory[cycle_key]
                performance = data['performance']
                confidence = data['confidence']
                signal_strength += amplitude * performance * confidence
            else:
                # New cycle, conservative assumption
                signal_strength += amplitude * 0.05
        
        # Analyze cycle interference
        interference_signal = self._analyze_cycle_interference(cycle_info)
        
        # Store cycles for learning
        self.dominant_cycles.append(cycle_info)
        
        # Normalize and combine signals
        normalized_signal = float(np.tanh(signal_strength / len(prices)))
        final_signal = normalized_signal + interference_signal
        
        return float(final_signal)
    
    def _analyze_cycle_interference(self, cycles: List[Dict]) -> float:
        """Advanced cycle interference analysis"""
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
    
    def _analyze_seasonal_patterns(self, prices: List[float], timestamps: List[float]) -> float:
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
        """Analyze lunar cycle influence on markets"""
        if len(timestamps) < 10:
            return 0.0
        
        try:
            # Simplified lunar cycle approximation (29.5 days)
            lunar_cycle_length = 29.5 * 24 * 3600  # seconds
            
            current_time = timestamps[-1]
            lunar_phase = (current_time % lunar_cycle_length) / lunar_cycle_length
            
            # Store lunar data
            self.lunar_cycle_data.append({
                'phase': lunar_phase,
                'price_change': (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 and prices[-10] != 0 else 0
            })
            
            # Analyze lunar correlation if we have enough data
            if len(self.lunar_cycle_data) >= 20:
                phases = [data['phase'] for data in self.lunar_cycle_data]
                changes = [data['price_change'] for data in self.lunar_cycle_data]
                
                # Simple correlation analysis
                correlation = np.corrcoef(phases, changes)[0, 1] if len(phases) > 1 else 0
                
                # Current lunar influence
                lunar_signal = correlation * np.sin(2 * np.pi * lunar_phase)
                return float(np.tanh(lunar_signal))
        
        except Exception as e:
            logger.warning(f"Lunar analysis error: {e}")
        
        return 0.0
    
    def learn_from_outcome(self, cycles_info: List[Dict], outcome: float):
        """Enhanced learning with confidence tracking"""
        if not cycles_info:
            return
        
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
        self._update_seasonal_patterns(outcome)
        
        # Store interference patterns
        if len(cycles_info) >= 2:
            pattern_key = self._create_interference_pattern_key(cycles_info)
            if pattern_key not in self.interference_patterns:
                self.interference_patterns[pattern_key] = deque(maxlen=20)
            self.interference_patterns[pattern_key].append(outcome)
    
    def _update_seasonal_patterns(self, outcome: float):
        """Update seasonal pattern performance"""
        try:
            current_time = datetime.now()
            hour_key = current_time.hour
            dow_key = f"dow_{current_time.weekday()}"
            
            # Update hour pattern
            if hour_key not in self.seasonal_patterns:
                self.seasonal_patterns[hour_key] = {'performance': 0.0, 'count': 0}
            
            data = self.seasonal_patterns[hour_key]
            data['performance'] = (data['performance'] * data['count'] + outcome) / (data['count'] + 1)
            data['count'] += 1
            
            # Update day of week pattern
            if dow_key not in self.seasonal_patterns:
                self.seasonal_patterns[dow_key] = {'performance': 0.0, 'count': 0}
            
            dow_data = self.seasonal_patterns[dow_key]
            dow_data['performance'] = (dow_data['performance'] * dow_data['count'] + outcome) / (dow_data['count'] + 1)
            dow_data['count'] += 1
            
        except Exception as e:
            logger.warning(f"Seasonal pattern update error: {e}")
    
    def _create_interference_pattern_key(self, cycles_info: List[Dict]) -> str:
        """Create a key for interference pattern storage"""
        if len(cycles_info) < 2:
            return ""
        
        # Sort cycles by frequency for consistent key generation
        sorted_cycles = sorted(cycles_info, key=lambda x: x['frequency'])
        
        key_parts = []
        for cycle in sorted_cycles[:3]:  # Use top 3 cycles
            freq_bucket = int(cycle['frequency'] * 1000) / 1000
            key_parts.append(f"f{freq_bucket}")
        
        return "_".join(key_parts)


class EvolvingImmuneSystem:
    def __init__(self):
        self.antibodies = {}  # pattern -> {'strength': float, 'specificity': float, 'memory_count': int, 'generation': int}
        self.t_cell_memory = deque(maxlen=200)
        self.threat_evolution_tracker = {}
        self.autoimmune_prevention = set()
        
        # Enhanced immune system features
        self.antibody_generations = 0
        self.threat_severity_threshold = -0.3
        self.memory_consolidation_threshold = 3
        self.adaptive_response_rate = 0.1
        
    def detect_threats(self, market_state: Dict) -> float:
        threat_level = 0.0
        
        # Create current pattern signature
        pattern_signature = self._create_pattern_signature(market_state)
        
        # Check against evolved antibodies
        for antibody_pattern, data in self.antibodies.items():
            similarity = self._pattern_similarity(pattern_signature, antibody_pattern)
            
            if similarity > 0.7:  # High similarity to known threat
                # Enhanced threat calculation with memory strength
                memory_boost = 1.0 + (data['memory_count'] * 0.1)
                generation_factor = 1.0 + (data['generation'] * 0.05)
                threat_contribution = data['strength'] * similarity * memory_boost * generation_factor
                threat_level += threat_contribution
        
        # Enhanced T-cell memory response
        for past_threat in self.t_cell_memory:
            similarity = self._pattern_similarity(pattern_signature, past_threat['pattern'])
            if similarity > 0.8:
                # Rapid T-cell response with severity weighting
                severity_factor = min(2.0, abs(past_threat['severity']) / 0.1)
                threat_level += past_threat['severity'] * similarity * severity_factor
        
        # Autoimmune prevention with confidence scoring
        if pattern_signature in self.autoimmune_prevention:
            # Reduce threat level but don't eliminate completely
            threat_level *= 0.2
        
        # Adaptive threat evolution detection
        evolved_threat_level = self._detect_evolved_threats(pattern_signature, market_state)
        threat_level += evolved_threat_level
        
        return -min(1.0, threat_level)  # Negative signal for threats
    
    def _create_pattern_signature(self, market_state: Dict) -> str:
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
    
    def _pattern_similarity(self, pattern1: str, pattern2: str) -> float:
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
    
    def _detect_evolved_threats(self, pattern_signature: str, market_state: Dict) -> float:
        """Detect evolved or novel threat patterns"""
        evolved_threat_level = 0.0
        
        # Check for threat evolution patterns
        for threat_pattern, evolution_data in self.threat_evolution_tracker.items():
            base_similarity = self._pattern_similarity(pattern_signature, threat_pattern)
            
            if base_similarity > 0.5:
                # Check for evolution indicators
                evolution_score = 0.0
                
                # Volatility evolution
                if 'volatility_trend' in evolution_data:
                    current_vol = market_state.get('volatility', 0)
                    expected_vol = evolution_data['volatility_trend']
                    if abs(current_vol - expected_vol) > 0.01:
                        evolution_score += 0.3
                
                # Pattern mutation detection
                if 'mutation_indicators' in evolution_data:
                    for indicator in evolution_data['mutation_indicators']:
                        if indicator in pattern_signature:
                            evolution_score += 0.2
                
                evolved_threat_level += evolution_score * base_similarity
        
        return evolved_threat_level
    
    def learn_threat(self, market_state: Dict, threat_level: float):
        """Enhanced threat learning with evolution tracking"""
        pattern = self._create_pattern_signature(market_state)
        
        if threat_level < self.threat_severity_threshold:  # Significant threat
            # Create or strengthen antibody
            if pattern in self.antibodies:
                data = self.antibodies[pattern]
                strength_update = self.adaptive_response_rate * (1.0 + data['memory_count'] * 0.1)
                data['strength'] = min(1.0, data['strength'] + strength_update)
                data['memory_count'] += 1
                
                # Memory consolidation
                if data['memory_count'] >= self.memory_consolidation_threshold:
                    data['specificity'] = min(1.0, data['specificity'] + 0.1)
            else:
                self.antibodies[pattern] = {
                    'strength': 0.5,
                    'specificity': 0.7,
                    'memory_count': 1,
                    'generation': self.antibody_generations
                }
            
            # Enhanced T-cell memory
            self.t_cell_memory.append({
                'pattern': pattern,
                'severity': threat_level,
                'timestamp': datetime.now(),
                'market_context': market_state.copy()
            })
            
            # Track threat evolution
            self._track_threat_evolution(pattern, market_state, threat_level)
        
        elif threat_level > 0.3:  # False positive (good outcome from "threat")
            # Enhanced autoimmune prevention
            self.autoimmune_prevention.add(pattern)
            
            # Weaken antibody if it exists
            if pattern in self.antibodies:
                data = self.antibodies[pattern]
                data['strength'] = max(0.1, data['strength'] - 0.3)
                data['specificity'] = max(0.3, data['specificity'] - 0.1)
    
    def _track_threat_evolution(self, pattern: str, market_state: Dict, threat_level: float):
        """Track how threats evolve over time"""
        if pattern not in self.threat_evolution_tracker:
            self.threat_evolution_tracker[pattern] = {
                'first_seen': datetime.now(),
                'severity_history': deque(maxlen=20),
                'volatility_trend': market_state.get('volatility', 0),
                'mutation_indicators': set()
            }
        
        evolution_data = self.threat_evolution_tracker[pattern]
        evolution_data['severity_history'].append(threat_level)
        
        # Update volatility trend
        current_vol = market_state.get('volatility', 0)
        evolution_data['volatility_trend'] = (evolution_data['volatility_trend'] * 0.9 + current_vol * 0.1)
        
        # Detect mutation indicators
        pattern_parts = pattern.split('_')
        for part in pattern_parts:
            if 'extreme' in part or 'spike' in part or 'crash' in part:
                evolution_data['mutation_indicators'].add(part)
    
    def evolve_antibodies(self):
        """Enhanced antibody evolution with genetic algorithms"""
        if len(self.antibodies) < 10:
            return
        
        self.antibody_generations += 1
        evolved_antibodies = {}
        
        # Sort antibodies by effectiveness
        antibody_scores = []
        for pattern, data in self.antibodies.items():
            effectiveness = data['strength'] * data['specificity'] * (1 + data['memory_count'] * 0.1)
            antibody_scores.append((pattern, effectiveness, data))
        
        antibody_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top performers
        top_antibodies = antibody_scores[:len(antibody_scores)//2]
        for pattern, score, data in top_antibodies:
            evolved_antibodies[pattern] = data.copy()
            evolved_antibodies[pattern]['generation'] = self.antibody_generations
        
        # Generate new antibodies through mutation and crossover
        for i in range(len(top_antibodies)//2):
            parent1_pattern, _, parent1_data = top_antibodies[i]
            parent2_pattern, _, parent2_data = top_antibodies[(i+1) % len(top_antibodies)]
            
            # Crossover: combine pattern components
            offspring_pattern = self._crossover_patterns(parent1_pattern, parent2_pattern)
            
            if offspring_pattern and offspring_pattern not in evolved_antibodies:
                # Inherit traits from parents
                offspring_data = {
                    'strength': (parent1_data['strength'] + parent2_data['strength']) / 2,
                    'specificity': (parent1_data['specificity'] + parent2_data['specificity']) / 2,
                    'memory_count': 0,
                    'generation': self.antibody_generations
                }
                
                # Mutation
                if np.random.random() < 0.1:
                    offspring_data['strength'] += np.random.normal(0, 0.1)
                    offspring_data['specificity'] += np.random.normal(0, 0.05)
                    offspring_data['strength'] = np.clip(offspring_data['strength'], 0.1, 1.0)
                    offspring_data['specificity'] = np.clip(offspring_data['specificity'], 0.1, 1.0)
                
                evolved_antibodies[offspring_pattern] = offspring_data
        
        # Remove weak antibodies
        final_antibodies = {}
        for pattern, data in evolved_antibodies.items():
            if data['strength'] > 0.2 or data['memory_count'] > 2:
                final_antibodies[pattern] = data
        
        self.antibodies = final_antibodies
    
    def _crossover_patterns(self, pattern1: str, pattern2: str) -> str:
        """Create offspring pattern through crossover"""
        parts1 = pattern1.split('_')
        parts2 = pattern2.split('_')
        
        # Create component dictionaries
        comp1 = {}
        comp2 = {}
        
        for part in parts1:
            if '_' in part:
                key = part.split('_')[0]
                comp1[key] = part
        
        for part in parts2:
            if '_' in part:
                key = part.split('_')[0]
                comp2[key] = part
        
        # Combine components randomly
        offspring_parts = []
        all_components = set(comp1.keys()) | set(comp2.keys())
        
        for component in all_components:
            if component in comp1 and component in comp2:
                # Choose randomly from parents
                chosen_part = comp1[component] if np.random.random() < 0.5 else comp2[component]
                offspring_parts.append(chosen_part)
            elif component in comp1:
                offspring_parts.append(comp1[component])
            elif component in comp2:
                offspring_parts.append(comp2[component])
        
        return "_".join(offspring_parts)
    
    def get_immune_stats(self) -> Dict:
        """Get comprehensive immune system statistics"""
        if not self.antibodies:
            return {'total_antibodies': 0}
        
        strengths = [data['strength'] for data in self.antibodies.values()]
        specificities = [data['specificity'] for data in self.antibodies.values()]
        memory_counts = [data['memory_count'] for data in self.antibodies.values()]
        generations = [data['generation'] for data in self.antibodies.values()]
        
        return {
            'total_antibodies': len(self.antibodies),
            'avg_strength': np.mean(strengths),
            'avg_specificity': np.mean(specificities),
            'total_memory_events': sum(memory_counts),
            'max_generation': max(generations) if generations else 0,
            't_cell_memory_size': len(self.t_cell_memory),
            'autoimmune_prevention_patterns': len(self.autoimmune_prevention),
            'threat_evolution_tracking': len(self.threat_evolution_tracker),
            'antibody_generations': self.antibody_generations
        }


class EnhancedIntelligenceOrchestrator:
    def __init__(self):
        self.dna_subsystem = AdvancedDNASubsystem()
        self.temporal_subsystem = FFTTemporalSubsystem()
        self.immune_subsystem = EvolvingImmuneSystem()
        
        # Enhanced swarm intelligence
        self.subsystem_votes = deque(maxlen=200)
        self.consensus_history = deque(maxlen=100)
        self.performance_attribution = defaultdict(lambda: deque(maxlen=50))
        
        # Tool evolution and lifecycle management
        self.tool_lifecycle = {
            'dna': {'birth_time': datetime.now(), 'performance_history': deque(maxlen=100)},
            'temporal': {'birth_time': datetime.now(), 'performance_history': deque(maxlen=100)},
            'immune': {'birth_time': datetime.now(), 'performance_history': deque(maxlen=100)}
        }
        
        # Hybrid tool breeding
        self.hybrid_tools = {}
        self.tool_breeding_frequency = 500
        self.decision_count = 0
        
    def process_market_data(self, prices: List[float], volumes: List[float],
                           market_features: Dict, timestamps: Optional[List[float]] = None) -> Dict[str, float]:
        
        self.decision_count += 1
        
        # Enhanced DNA analysis with 16-base encoding
        volatility = market_features.get('volatility', 0)
        momentum = market_features.get('price_momentum', 0)
        dna_sequence = self.dna_subsystem.encode_market_state(prices[-20:], volumes[-20:], volatility, momentum)
        dna_signal = self.dna_subsystem.analyze_sequence(dna_sequence)
        
        # Enhanced FFT temporal analysis
        temporal_signal = self.temporal_subsystem.analyze_cycles(prices, timestamps)
        
        # Enhanced immune system threat detection
        immune_signal = self.immune_subsystem.detect_threats(market_features)
        
        # Swarm intelligence - enhanced voting with performance attribution
        votes = {
            'dna': dna_signal,
            'temporal': temporal_signal,
            'immune': immune_signal
        }
        
        # Track individual tool performance
        for tool, signal in votes.items():
            self.tool_lifecycle[tool]['performance_history'].append(signal)
        
        # Calculate enhanced consensus with disagreement weighting
        consensus_strength = self._calculate_enhanced_consensus(votes)
        
        # Dynamic tool activation based on market conditions
        active_weights = self._calculate_dynamic_weights(market_features, votes)
        
        # Weighted overall signal with dynamic activation
        overall_signal = sum(votes[tool] * active_weights[tool] for tool in votes.keys())
        
        # Boost signal based on consensus and tool agreement
        if consensus_strength > 0.8:
            overall_signal *= 1.4  # Strong consensus boost
        elif consensus_strength < 0.3:
            overall_signal *= 0.6  # Low consensus penalty
        
        # Check for hybrid tool creation
        if self.decision_count % self.tool_breeding_frequency == 0:
            self._attempt_tool_breeding(votes, market_features)
        
        # Store voting data
        self.subsystem_votes.append({
            'votes': votes.copy(),
            'consensus': consensus_strength,
            'weights': active_weights.copy(),
            'timestamp': datetime.now()
        })
        self.consensus_history.append(consensus_strength)
        
        return {
            'dna_signal': dna_signal,
            'temporal_signal': temporal_signal,
            'immune_signal': immune_signal,
            'overall_signal': overall_signal,
            'consensus_strength': consensus_strength,
            'active_weights': active_weights,
            'current_patterns': {
                'dna_sequence': dna_sequence,
                'dominant_cycles': len(self.temporal_subsystem.dominant_cycles),
                'active_antibodies': len(self.immune_subsystem.antibodies),
                'hybrid_tools': len(self.hybrid_tools)
            }
        }
    
    def _calculate_enhanced_consensus(self, votes: Dict[str, float]) -> float:
        """Enhanced consensus calculation with disagreement analysis"""
        signals = list(votes.values())
        
        if not signals:
            return 0.0
        
        # Directional agreement
        positive_signals = sum(1 for s in signals if s > 0.1)
        negative_signals = sum(1 for s in signals if s < -0.1)
        neutral_signals = sum(1 for s in signals if abs(s) <= 0.1)
        
        total_signals = len(signals)
        directional_consensus = max(positive_signals, negative_signals, neutral_signals) / total_signals
        
        # Magnitude agreement
        signal_magnitudes = [abs(s) for s in signals]
        magnitude_std = np.std(signal_magnitudes) if len(signal_magnitudes) > 1 else 0
        magnitude_consensus = 1.0 / (1.0 + magnitude_std)
        
        # Combined consensus
        return (directional_consensus * 0.7 + magnitude_consensus * 0.3)
    
    def _calculate_dynamic_weights(self, market_features: Dict, votes: Dict[str, float]) -> Dict[str, float]:
        """Calculate dynamic weights based on market conditions and tool performance"""
        weights = {'dna': 0.4, 'temporal': 0.4, 'immune': 0.2}  # Base weights
        
        # Adjust based on market volatility
        volatility = market_features.get('volatility', 0.02)
        if volatility > 0.05:  # High volatility
            weights['immune'] += 0.2  # Boost immune system
            weights['dna'] -= 0.1
            weights['temporal'] -= 0.1
        elif volatility < 0.01:  # Low volatility
            weights['temporal'] += 0.2  # Boost temporal analysis
            weights['immune'] -= 0.1
            weights['dna'] -= 0.1
        
        # Adjust based on recent tool performance
        for tool in weights.keys():
            recent_performance = list(self.tool_lifecycle[tool]['performance_history'])[-10:]
            if len(recent_performance) >= 5:
                avg_performance = np.mean([abs(p) for p in recent_performance])
                if avg_performance > 0.3:
                    weights[tool] *= 1.2  # Boost well-performing tools
                elif avg_performance < 0.1:
                    weights[tool] *= 0.8  # Reduce poorly performing tools
        
        # Normalize weights
        total_weight = sum(weights.values())
        return {tool: weight / total_weight for tool, weight in weights.items()}
    
    def _attempt_tool_breeding(self, votes: Dict[str, float], market_features: Dict):
        """Attempt to create hybrid tools from successful combinations"""
        # Find tools with complementary signals
        tool_pairs = [('dna', 'temporal'), ('dna', 'immune'), ('temporal', 'immune')]
        
        for tool1, tool2 in tool_pairs:
            signal1, signal2 = votes[tool1], votes[tool2]
            
            # Check for complementary behavior
            if abs(signal1) > 0.3 and abs(signal2) > 0.3 and np.sign(signal1) == np.sign(signal2):
                hybrid_name = f"{tool1}_{tool2}_hybrid"
                
                if hybrid_name not in self.hybrid_tools:
                    # Create new hybrid tool
                    self.hybrid_tools[hybrid_name] = {
                        'parents': (tool1, tool2),
                        'birth_time': datetime.now(),
                        'performance_history': deque(maxlen=50),
                        'activation_conditions': self._create_activation_conditions(market_features),
                        'combination_strategy': 'weighted_average'  # Could be evolved
                    }
                    
                    logger.info(f"Created hybrid tool: {hybrid_name}")
    
    def _create_activation_conditions(self, market_features: Dict) -> Dict:
        """Create activation conditions for hybrid tools"""
        return {
            'min_volatility': market_features.get('volatility', 0.02) * 0.8,
            'max_volatility': market_features.get('volatility', 0.02) * 1.2,
            'momentum_threshold': abs(market_features.get('price_momentum', 0)) * 0.5
        }
    
    def learn_from_outcome(self, outcome: float, context: Dict):
        """Enhanced learning with tool evolution and performance attribution"""
        # Extract context
        dna_sequence = context.get('dna_sequence', '')
        cycles_info = context.get('cycles_info', [])
        market_state = context.get('market_state', {})
        
        # Each subsystem learns
        self.dna_subsystem.learn_from_outcome(dna_sequence, outcome)
        
        if cycles_info:
            self.temporal_subsystem.learn_from_outcome(cycles_info, outcome)
        
        self.immune_subsystem.learn_threat(market_state, outcome)
        
        # Performance attribution
        if self.subsystem_votes:
            recent_vote = self.subsystem_votes[-1]
            for tool, signal in recent_vote['votes'].items():
                # Attribute performance based on signal strength and outcome alignment
                attribution_score = outcome * signal * recent_vote['weights'][tool]
                self.performance_attribution[tool].append(attribution_score)
        
        # Update hybrid tool performance
        for hybrid_name, hybrid_data in self.hybrid_tools.items():
            # Simple performance tracking for now
            hybrid_data['performance_history'].append(outcome * 0.5)  # Conservative attribution
        
        # Periodic evolution
        if len(self.subsystem_votes) % 200 == 0:
            self._evolve_tools()
    
    def _evolve_tools(self):
        """Evolve tools based on performance"""
        # Evolve immune system
        self.immune_subsystem.evolve_antibodies()
        
        # Tool lifecycle management
        current_time = datetime.now()
        tools_to_remove = []
        
        for hybrid_name, hybrid_data in self.hybrid_tools.items():
            # Remove poorly performing hybrid tools
            if len(hybrid_data['performance_history']) >= 20:
                avg_performance = np.mean(list(hybrid_data['performance_history']))
                if avg_performance < -0.2:  # Consistently poor performance
                    tools_to_remove.append(hybrid_name)
        
        for tool_name in tools_to_remove:
            del self.hybrid_tools[tool_name]
            logger.info(f"Removed underperforming hybrid tool: {tool_name}")
    
    def get_comprehensive_stats(self) -> Dict:
        """Get comprehensive intelligence statistics"""
        dna_stats = self.dna_subsystem.get_evolution_stats()
        immune_stats = self.immune_subsystem.get_immune_stats()
        
        # Tool performance attribution
        attribution_stats = {}
        for tool, scores in self.performance_attribution.items():
            if scores:
                attribution_stats[tool] = {
                    'avg_attribution': np.mean(list(scores)),
                    'attribution_volatility': np.std(list(scores)),
                    'positive_attributions': sum(1 for s in scores if s > 0),
                    'total_attributions': len(scores)
                }
        
        return {
            'dna_evolution': dna_stats,
            'immune_system': immune_stats,
            'temporal_cycles': len(self.temporal_subsystem.cycle_memory),
            'recent_consensus': np.mean(list(self.consensus_history)) if self.consensus_history else 0.0,
            'performance_attribution': attribution_stats,
            'hybrid_tools': {
                'count': len(self.hybrid_tools),
                'tools': list(self.hybrid_tools.keys())
            },
            'decision_count': self.decision_count
        }