# subsystem_evolution.py

import numpy as np
import torch
from scipy import fft
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
import random
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)  # Reduce logging verbosity, use progress bars instead

# Lazy import rich components
progress = None
console = None

def _get_progress():
    """Lazy initialization of rich progress bar"""
    global progress, console
    if progress is None:
        try:
            from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn
            from rich.console import Console
            
            console = Console()
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[prog.description]{task.description}"),
                BarColumn(bar_width=30),
                MofNCompleteColumn(),
                TextColumn("•"),
                TaskProgressColumn(),
                TextColumn("•"), 
                TimeRemainingColumn(),
                console=console,
                transient=True,  # Hide after completion
                refresh_per_second=2  # Reduce refresh rate
            )
        except ImportError:
            logger.warning("Rich not available, progress bars disabled")
            progress = None
    return progress


class DNASubsystem:
    def __init__(self):
        # 16-base DNA encoding for market patterns
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
        
        # Pattern storage with performance tracking
        self.sequences = {}  # sequence -> {'performance': float, 'age': int, 'count': int}
        self.max_sequences = 8000
        self.age_decay_factor = 0.999
        
        # Evolution and breeding parameters
        self.performance_threshold = 0.3
        self.elite_sequences = {}
        self.max_age = 1000
        self.breeding_frequency = 50
        self.generation_count = 0
        self.breeding_pool = deque(maxlen=100)
        self.breeding_success_rate = deque(maxlen=50)
        self.mutation_rate = 0.1
        
        # Progress tracking
        self.learning_progress = None
        self.total_learning_events = 0
        self.learning_batch_size = 50  # Update progress every N learning events
        
    def encode_market_state(self, prices: List[float], volumes: List[float],
                           volatility: Optional[float] = None, momentum: Optional[float] = None) -> str:
        if len(prices) < 2 or len(volumes) < 2:
            logger.debug(f"Insufficient data for DNA encoding: prices={len(prices)}, volumes={len(volumes)}")
            return ""
        
        sequence = ""
        
        try:
            # Calculate market context if not provided
            if volatility is None:
                volatility = np.std(prices[-10:]) / np.mean(prices[-10:]) if len(prices) >= 10 else 0
            if momentum is None:
                momentum = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 and prices[-5] != 0 else 0
            
            for i in range(1, len(prices)):
                price_change = (prices[i] - prices[i-1]) / prices[i-1] if prices[i-1] != 0 else 0
                vol_ratio = volumes[i] / volumes[i-1] if volumes[i-1] > 0 else 1.0
                
                base = self._determine_base(price_change, vol_ratio, volatility, momentum)
                sequence += base
            
            if not sequence:
                logger.warning("DNA encoding produced empty sequence despite sufficient data")
            else:
                logger.debug(f"DNA sequence encoded: {sequence[:10]}... (length: {len(sequence)})")
                
            return sequence
            
        except Exception as e:
            logger.error(f"Error in DNA encoding: {e}")
            return ""
    
    def _determine_base(self, price_change: float, vol_ratio: float, volatility: float, momentum: float) -> str:
        """Determine DNA base using 16-base encoding"""
        
        # Strong momentum patterns (only for very strong momentum)
        if abs(momentum) > 0.03:
            return 'L' if momentum > 0 else 'M'
        
        # Extreme volatility regime patterns (only for exceptional cases)
        if volatility > 0.05:  # Very high volatility
            return 'N'  # Volatility spike
        elif volatility < 0.002 and abs(price_change) < 0.0001 and vol_ratio < 1.1:
            # Only use volatility crush when price is truly flat AND volume is low
            return 'O'  # Volatility crush
        
        # Standard price/volume encoding (primary encoding method)
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
            logger.debug(f"DNA sequence too short or empty: '{sequence}' (length: {len(sequence) if sequence else 0})")
            return 0.0
        
        try:
            # Age all sequences
            self._age_sequences()
            
            # Find best matching patterns with breeding consideration
            best_score = 0.0
            best_matches = []
            matches_found = 0
            
            logger.debug(f"Analyzing DNA sequence: '{sequence[:20]}...' against {len(self.sequences)} stored sequences")
            
            for stored_seq, data in self.sequences.items():
                try:
                    similarity = self._advanced_sequence_similarity(sequence, stored_seq)
                    if similarity > 0.7:
                        matches_found += 1
                        # Age-adjusted performance
                        age_factor = self.age_decay_factor ** data['age']
                        adjusted_performance = data['performance'] * age_factor
                        score = adjusted_performance * similarity
                        
                        # Validate score
                        if np.isnan(score) or np.isinf(score):
                            continue
                            
                        if abs(score) > abs(best_score):
                            best_score = score
                            best_matches.append((stored_seq, score, similarity))
                            logger.debug(f"New best DNA match: score={score:.4f}, similarity={similarity:.4f}")
                except Exception as e:
                    logger.warning(f"Error analyzing sequence similarity: {e}")
                    continue
            
            # If no good matches found, provide a small baseline signal
            if matches_found == 0 and len(self.sequences) > 0:
                logger.debug("No DNA matches found, using baseline signal")
                best_score = 0.05  # Small baseline signal
            
            # Consider breeding if we have good matches
            if len(best_matches) >= 2 and len(self.sequences) % self.breeding_frequency == 0:
                try:
                    self._attempt_breeding(best_matches)
                except Exception as e:
                    logger.warning(f"Error in breeding attempt: {e}")
            
            # Ensure valid return value
            if np.isnan(best_score) or np.isinf(best_score):
                logger.warning("DNA analysis returned invalid score, using 0.0")
                return 0.0
            
            logger.debug(f"DNA analysis result: {best_score:.4f} (matches: {matches_found})")
            return best_score
            
        except Exception as e:
            logger.error(f"Error in DNA sequence analysis: {e}")
            return 0.0
    
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
        # Debug and validate inputs
        if not isinstance(sequence, str):
            logger.error(f"DNA sequence is not a string: type={type(sequence)}, content={str(sequence)[:200]}")
            return
        if not sequence:
            return
        
        self.total_learning_events += 1
        
        # Initialize progress task if needed (only during bootstrap)
        if self.learning_progress is None and not getattr(self, 'bootstrap_complete', False) and not hasattr(self, '_live_trading_started'):
            try:
                prog = _get_progress()
                if prog is not None:
                    if not prog.live.is_started:
                        prog.start()
                    self.learning_progress = prog.add_task(
                        "[cyan]DNA Learning[/cyan]", 
                        total=None,
                        elite=0, 
                        avg_perf=0.0
                    )
            except Exception as e:
                logger.error(f"Error initializing DNA progress task: {e}")
                self.learning_progress = None
        
        # Update or add sequence
        sequence_added = False
        if sequence in self.sequences:
            data = self.sequences[sequence]
            learning_rate = 0.1 / (1 + data['age'] * 0.01)  # Slower learning for older sequences
            old_performance = data['performance']
            data['performance'] += learning_rate * (outcome - data['performance'])
            data['age'] = max(0, data['age'] - 5)  # Rejuvenate successful sequences
        else:
            # Check if we're at the sequence limit before adding new ones
            if len(self.sequences) >= self.max_sequences:
                self._cleanup_sequences()
            
            # Only add if we still have room after cleanup
            if len(self.sequences) < self.max_sequences:
                self.sequences[sequence] = {
                    'performance': outcome * 0.5,
                    'age': 0,
                    'generation': 0,
                    'parents': None
                }
                sequence_added = True
        
        # Update elite sequences
        if outcome > self.performance_threshold:
            self.elite_sequences[sequence] = self.sequences[sequence].copy()
        
        # Update progress task (only during bootstrap)
        if sequence_added and self.learning_progress is not None and not getattr(self, 'bootstrap_complete', False):
            try:
                prog = _get_progress()
                if prog is not None:
                    prog.advance(self.learning_progress, 1)
            except Exception as e:
                logger.warning(f"Error updating DNA progress task: {e}")
        
        # Update progress task stats every batch
        if self.total_learning_events % self.learning_batch_size == 0 and self.learning_progress is not None:
            try:
                prog = _get_progress()
                if prog is not None:
                    avg_perf = np.mean([data['performance'] for data in self.sequences.values()]) if self.sequences else 0.0
                    prog.update(
                        self.learning_progress,
                        elite=len(self.elite_sequences),
                        avg_perf=f"{avg_perf:.3f}"
                    )
            except Exception as e:
                logger.warning(f"Error updating DNA progress task stats: {e}")
        
        # Cleanup poor performers (periodic cleanup)
        if len(self.sequences) % 100 == 0:  # Cleanup every 100 sequences
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
        
        # Progress tracking
        self.learning_progress = None
        self.total_learning_events = 0
        self.learning_batch_size = 25
        
    def analyze_cycles(self, prices: List[float], timestamps: Optional[List[float]] = None) -> float:
        if len(prices) < 32:
            logger.debug(f"Insufficient data for temporal analysis: {len(prices)} prices (need 32+)")
            return 0.0
        
        try:
            # Enhanced FFT analysis with multiple window sizes
            signals = []
            
            logger.debug(f"Starting temporal analysis with {len(prices)} prices")
            
            for window_size in [64, 128, 256]:
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
                return 0.02  # Small baseline signal
            
            # Combine signals from different timeframes
            combined_signal = float(np.mean(signals))
            
            # Validate combined signal
            if np.isnan(combined_signal) or np.isinf(combined_signal):
                logger.warning("Combined temporal signal is invalid")
                return 0.0
            
            # Add seasonal and lunar analysis if timestamps available
            if timestamps:
                try:
                    seasonal_signal = self._analyze_seasonal_patterns(prices, timestamps)
                    lunar_signal = self._analyze_lunar_influence(prices, timestamps)
                    
                    # Validate additional signals
                    if not (np.isnan(seasonal_signal) or np.isinf(seasonal_signal)):
                        if not (np.isnan(lunar_signal) or np.isinf(lunar_signal)):
                            combined_signal = combined_signal * 0.7 + seasonal_signal * 0.2 + lunar_signal * 0.1
                            logger.debug(f"Enhanced temporal signal with seasonal/lunar: {combined_signal:.4f}")
                except Exception as e:
                    logger.warning(f"Seasonal/lunar analysis failed: {e}")
            
            # Final validation
            if np.isnan(combined_signal) or np.isinf(combined_signal):
                logger.warning("Final temporal signal is invalid")
                return 0.0
            
            logger.debug(f"Temporal analysis result: {combined_signal:.4f}")
            return combined_signal
            
        except Exception as e:
            logger.error(f"Error in temporal cycle analysis: {e}")
            return 0.0
    
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
        # Debug and validate inputs
        if not isinstance(cycles_info, list):
            logger.error(f"Temporal cycles_info is not a list: type={type(cycles_info)}, content={str(cycles_info)[:200]}")
            return
        if not cycles_info:
            return
        
        self.total_learning_events += 1
        
        # Initialize progress task if needed
        if self.learning_progress is None:
            try:
                prog = _get_progress()
                if prog is not None:
                    if not prog.live.is_started:
                        prog.start()
                    self.learning_progress = prog.add_task(
                        "[green]Temporal Learning[/green]", 
                        total=None,
                        avg_conf=0.0, 
                        patterns=0
                    )
            except Exception as e:
                logger.error(f"Error initializing temporal progress task: {e}")
                self.learning_progress = None
        
        cycles_learned = 0
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
                old_performance = data['performance']
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
                cycles_learned += 1
            else:
                self.cycle_memory[cycle_key] = {
                    'strength': cycle['amplitude'],
                    'phase': cycle['phase'],
                    'performance': outcome * 0.3,
                    'confidence': 0.5
                }
                cycles_learned += 1
                if self.learning_progress is not None:
                    try:
                        prog = _get_progress()
                        if prog is not None:
                            prog.advance(self.learning_progress, 1)
                    except Exception as e:
                        logger.warning(f"Error updating temporal progress task: {e}")
        
        # Update progress task stats every batch
        if self.total_learning_events % self.learning_batch_size == 0 and self.learning_progress is not None:
            try:
                prog = _get_progress()
                if prog is not None:
                    avg_conf = np.mean([data['confidence'] for data in self.cycle_memory.values()]) if self.cycle_memory else 0.0
                    prog.update(
                        self.learning_progress,
                        avg_conf=f"{avg_conf:.3f}",
                        patterns=len(self.interference_patterns)
                    )
            except Exception as e:
                logger.warning(f"Error updating temporal progress task stats: {e}")
        
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
        
        # Progress tracking
        self.learning_progress = None
        self.total_learning_events = 0
        self.learning_batch_size = 20
        
    def detect_threats(self, market_state: Dict) -> float:
        try:
            threat_level = 0.0
            
            # Create current pattern signature
            pattern_signature = self._create_pattern_signature(market_state)
            
            if not pattern_signature:
                logger.debug("Empty pattern signature for immune analysis")
                return 0.0
            
            logger.debug(f"Immune analysis: pattern='{pattern_signature[:50]}...', antibodies={len(self.antibodies)}")
            
            antibody_matches = 0
            # Check against evolved antibodies
            for antibody_pattern, data in self.antibodies.items():
                try:
                    similarity = self._pattern_similarity(pattern_signature, antibody_pattern)
                    
                    if similarity > 0.7:  # High similarity to known threat
                        antibody_matches += 1
                        # Enhanced threat calculation with memory strength
                        memory_boost = 1.0 + (data.get('memory_count', 0) * 0.1)
                        generation_factor = 1.0 + (data.get('generation', 0) * 0.05)
                        threat_contribution = data.get('strength', 0.0) * similarity * memory_boost * generation_factor
                        
                        # Validate threat contribution
                        if not (np.isnan(threat_contribution) or np.isinf(threat_contribution)):
                            threat_level += threat_contribution
                            logger.debug(f"Antibody match: similarity={similarity:.3f}, contribution={threat_contribution:.4f}")
                except Exception as e:
                    logger.warning(f"Error processing antibody {antibody_pattern}: {e}")
                    continue
            
            tcell_matches = 0
            # Enhanced T-cell memory response
            for past_threat in self.t_cell_memory:
                try:
                    similarity = self._pattern_similarity(pattern_signature, past_threat['pattern'])
                    if similarity > 0.8:
                        tcell_matches += 1
                        # Rapid T-cell response with severity weighting
                        severity = past_threat.get('severity', 0.0)
                        severity_factor = min(2.0, abs(severity) / 0.1) if severity != 0 else 1.0
                        threat_contribution = severity * similarity * severity_factor
                        
                        # Validate threat contribution
                        if not (np.isnan(threat_contribution) or np.isinf(threat_contribution)):
                            threat_level += threat_contribution
                            logger.debug(f"T-cell match: similarity={similarity:.3f}, contribution={threat_contribution:.4f}")
                except Exception as e:
                    logger.warning(f"Error processing T-cell memory: {e}")
                    continue
            
            # Autoimmune prevention with confidence scoring
            if pattern_signature in self.autoimmune_prevention:
                # Reduce threat level but don't eliminate completely
                threat_level *= 0.2
                logger.debug("Autoimmune prevention activated")
            
            # Adaptive threat evolution detection
            try:
                evolved_threat_level = self._detect_evolved_threats(pattern_signature, market_state)
                if not (np.isnan(evolved_threat_level) or np.isinf(evolved_threat_level)):
                    threat_level += evolved_threat_level
            except Exception as e:
                logger.warning(f"Error in evolved threat detection: {e}")
            
            # If no threats detected but we have antibodies, provide small baseline
            if threat_level == 0.0 and len(self.antibodies) > 0:
                threat_level = -0.01  # Small negative baseline
                logger.debug("No threats detected, using baseline negative signal")
            
            # Validate final threat level
            if np.isnan(threat_level) or np.isinf(threat_level):
                logger.warning("Invalid final threat level")
                return 0.0
            
            final_signal = -min(1.0, max(-1.0, threat_level))  # Negative signal for threats, bounded
            logger.debug(f"Immune result: {final_signal:.4f} (antibody_matches={antibody_matches}, tcell_matches={tcell_matches})")
            
            return final_signal
            
        except Exception as e:
            logger.error(f"Error in immune threat detection: {e}")
            return 0.0
    
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
            
            # Check for pattern mutations and evolutions
            current_volatility = market_state.get('volatility', 0.0)
            current_momentum = market_state.get('price_momentum', 0.0)
            
            # Detect extreme market conditions that might indicate evolved threats
            if current_volatility > 0.1:  # Very high volatility
                evolved_threat_level -= 0.2
            
            if abs(current_momentum) > 0.05:  # Strong momentum
                evolved_threat_level -= 0.1
            
            # Check for threat evolution patterns
            if pattern_signature in self.threat_evolution_tracker:
                evolution_data = self.threat_evolution_tracker[pattern_signature]
                severity_history = list(evolution_data['severity_history'])
                
                if len(severity_history) >= 3:
                    # Check if threat is getting worse over time
                    recent_trend = np.mean(severity_history[-3:]) - np.mean(severity_history[:-3])
                    if recent_trend < -0.1:  # Getting more threatening
                        evolved_threat_level -= 0.3
            
            return evolved_threat_level
    
    def learn_threat(self, market_state: Dict, threat_level: float, is_bootstrap: bool = False):
        """Enhanced threat learning with evolution tracking and bootstrap mode."""
        if not isinstance(market_state, dict):
            logger.error(f"Immune market_state is not a dict: type={type(market_state)}, content={str(market_state)[:200]}")
            return

        pattern = self._create_pattern_signature(market_state)
        if not pattern:
            return

        self.total_learning_events += 1

        # Initialize progress task if needed
        if self.learning_progress is None:
            try:
                prog = _get_progress()
                if prog is not None:
                    if not prog.live.is_started:
                        prog.start()
                    self.learning_progress = prog.add_task(
                        "[red]Immune Learning[/red]", 
                        total=None,
                        tcells=0, 
                        prevention=0
                    )
            except Exception as e:
                logger.error(f"Error initializing immune progress task: {e}")
                self.learning_progress = None

        # Use a more sensitive threshold during historical bootstrapping
        learning_threshold = -0.15 if is_bootstrap else self.threat_severity_threshold

        antibody_created = False
        if threat_level < learning_threshold:  # A significant threat is detected
            if pattern in self.antibodies:
                data = self.antibodies[pattern]
                strength_update = self.adaptive_response_rate * (1.0 + data['memory_count'] * 0.1)
                data['strength'] = min(1.0, data['strength'] + strength_update)
                data['memory_count'] += 1
                if data['memory_count'] >= self.memory_consolidation_threshold:
                    data['specificity'] = min(1.0, data['specificity'] + 0.1)
            else:
                self.antibodies[pattern] = {
                    'strength': 0.5,
                    'specificity': 0.7,
                    'memory_count': 1,
                    'generation': self.antibody_generations
                }
                antibody_created = True
            
            # Enhanced T-cell memory
            self.t_cell_memory.append({
                'pattern': pattern,
                'severity': threat_level,
                'timestamp': datetime.now(),
                'market_context': market_state.copy()
            })
            
            # Track threat evolution
            self._track_threat_evolution(pattern, market_state, threat_level)

        elif threat_level > 0.3:  # False positive (good outcome from supposed threat)
            # Enhanced autoimmune prevention
            self.autoimmune_prevention.add(pattern)
            
            # Weaken antibody if it exists
            if pattern in self.antibodies:
                data = self.antibodies[pattern]
                data['strength'] = max(0.1, data['strength'] - 0.3)
                data['specificity'] = max(0.3, data['specificity'] - 0.1)
        
        # Update progress task
        if antibody_created and self.learning_progress is not None:
            try:
                prog = _get_progress()
                if prog is not None:
                    prog.advance(self.learning_progress, 1)
            except Exception as e:
                logger.warning(f"Error updating immune progress task: {e}")
        
        # Update progress task stats every batch
        if self.total_learning_events % self.learning_batch_size == 0 and self.learning_progress is not None:
            try:
                prog = _get_progress()
                if prog is not None:
                    prog.update(
                        self.learning_progress,
                        tcells=len(self.t_cell_memory),
                        prevention=len(self.autoimmune_prevention)
                    )
            except Exception as e:
                logger.warning(f"Error updating immune progress task stats: {e}")
    
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
        self.dna_subsystem = DNASubsystem()
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
            'immune': {'birth_time': datetime.now(), 'performance_history': deque(maxlen=100)},
            'microstructure': {'birth_time': datetime.now(), 'performance_history': deque(maxlen=100)}
        }
        
        # Hybrid tool breeding
        self.hybrid_tools = {}
        self.tool_breeding_frequency = 500
        self.decision_count = 0
        
        # Progress tracking
        self.orchestrator_progress = None
        self.bootstrap_complete = False
        
    def process_market_data(self, prices: List[float], volumes: List[float],
                           market_features: Dict, timestamps: Optional[List[float]] = None) -> Dict[str, float]:
        
        try:
            self.decision_count += 1
            
            # Enhanced DNA analysis with 16-base encoding
            volatility = market_features.get('volatility', 0)
            momentum = market_features.get('price_momentum', 0)
            
            try:
                dna_sequence = self.dna_subsystem.encode_market_state(prices[-20:], volumes[-20:], volatility, momentum)
                dna_signal = self.dna_subsystem.analyze_sequence(dna_sequence)
                
                # Validate DNA signal and ensure minimum baseline
                if np.isnan(dna_signal) or np.isinf(dna_signal):
                    logger.warning("Invalid DNA signal detected, using baseline")
                    dna_signal = 0.05
                elif dna_signal == 0.0 and len(self.dna_subsystem.sequences) > 0:
                    dna_signal = 0.05  # Ensure baseline when sequences exist
                    
                logger.debug(f"Final DNA signal: {dna_signal:.4f}")
                    
            except Exception as e:
                logger.error(f"DNA analysis failed: {e}")
                dna_sequence = ""
                dna_signal = 0.05  # Use baseline instead of 0.0
            
            # Enhanced FFT temporal analysis
            try:
                temporal_signal = self.temporal_subsystem.analyze_cycles(prices, timestamps)
                
                # Validate temporal signal and ensure minimum baseline
                if np.isnan(temporal_signal) or np.isinf(temporal_signal):
                    logger.warning("Invalid temporal signal detected, using baseline")
                    temporal_signal = 0.02
                elif temporal_signal == 0.0:
                    temporal_signal = 0.02  # Ensure baseline
                    
                logger.debug(f"Final temporal signal: {temporal_signal:.4f}")
                    
            except Exception as e:
                logger.error(f"Temporal analysis failed: {e}")
                temporal_signal = 0.02  # Use baseline instead of 0.0
            
            # Enhanced immune system threat detection
            try:
                immune_signal = self.immune_subsystem.detect_threats(market_features)
                
                # Validate immune signal and ensure minimum baseline
                if np.isnan(immune_signal) or np.isinf(immune_signal):
                    logger.warning("Invalid immune signal detected, using baseline")
                    immune_signal = -0.01
                elif immune_signal == 0.0 and len(self.immune_subsystem.antibodies) == 0:
                    immune_signal = -0.01  # Small negative baseline when no antibodies
                    
                logger.debug(f"Final immune signal: {immune_signal:.4f}")
                    
            except Exception as e:
                logger.error(f"Immune analysis failed: {e}")
                immune_signal = -0.01  # Use baseline instead of 0.0
            
            # Swarm intelligence - enhanced voting with performance attribution
            votes = {
                'dna': dna_signal,
                'temporal': temporal_signal,
                'immune': immune_signal
            }
            
            # Track individual tool performance
            for tool, signal in votes.items():
                if not (np.isnan(signal) or np.isinf(signal)):
                    self.tool_lifecycle[tool]['performance_history'].append(signal)
            
            # Calculate enhanced consensus with disagreement weighting
            try:
                consensus_strength = self._calculate_enhanced_consensus(votes)
                if np.isnan(consensus_strength) or np.isinf(consensus_strength):
                    consensus_strength = 0.5
            except Exception as e:
                logger.warning(f"Consensus calculation failed: {e}")
                consensus_strength = 0.5
            
            # Dynamic tool activation based on market conditions
            try:
                active_weights = self._calculate_dynamic_weights(market_features, votes)
            except Exception as e:
                logger.warning(f"Weight calculation failed: {e}")
                active_weights = {'dna': 0.4, 'temporal': 0.4, 'immune': 0.2}
            
            # Weighted overall signal with dynamic activation
            try:
                overall_signal = sum(votes[tool] * active_weights.get(tool, 0.0) for tool in votes.keys())
                
                logger.debug(f"Raw overall signal: {overall_signal:.4f} from votes: {votes}")
                
                # Validate overall signal
                if np.isnan(overall_signal) or np.isinf(overall_signal):
                    logger.warning("Invalid overall signal detected, using baseline")
                    overall_signal = 0.02  # Use baseline instead of 0.0
                
                # Ensure minimum signal when we have baseline signals
                if overall_signal == 0.0:
                    # Calculate baseline from individual signals
                    baseline_signal = (votes['dna'] * 0.4 + votes['temporal'] * 0.4 + abs(votes['immune']) * 0.2)
                    if baseline_signal > 0:
                        overall_signal = baseline_signal
                        logger.debug(f"Applied baseline overall signal: {overall_signal:.4f}")
                
                # Boost signal based on consensus and tool agreement
                if consensus_strength > 0.8:
                    overall_signal *= 1.4  # Strong consensus boost
                elif consensus_strength < 0.3:
                    overall_signal *= 0.6  # Low consensus penalty
                
                logger.debug(f"Final overall signal: {overall_signal:.4f}")
                    
            except Exception as e:
                logger.error(f"Overall signal calculation failed: {e}")
                overall_signal = 0.02  # Use baseline instead of 0.0
            
            # Check for hybrid tool creation
            if self.decision_count % self.tool_breeding_frequency == 0:
                try:
                    self._attempt_tool_breeding(votes, market_features)
                except Exception as e:
                    logger.warning(f"Tool breeding failed: {e}")
            
            # Store voting data
            try:
                self.subsystem_votes.append({
                    'votes': votes.copy(),
                    'consensus': consensus_strength,
                    'weights': active_weights.copy(),
                    'timestamp': datetime.now()
                })
                self.consensus_history.append(consensus_strength)
            except Exception as e:
                logger.warning(f"Failed to store voting data: {e}")
            
            return {
                'dna_signal': dna_signal,
                'temporal_signal': temporal_signal,
                'immune_signal': immune_signal,
                'overall_signal': overall_signal,
                'consensus_strength': consensus_strength,
                'active_weights': active_weights,
                'current_patterns': {
                    'dna_sequence': dna_sequence if 'dna_sequence' in locals() else "",
                    'dominant_cycles': len(self.temporal_subsystem.dominant_cycles),
                    'active_antibodies': len(self.immune_subsystem.antibodies),
                    'hybrid_tools': len(self.hybrid_tools)
                }
            }
            
        except Exception as e:
            logger.error(f"Critical error in orchestrator processing: {e}")
            # Return safe fallback values
            return {
                'dna_signal': 0.0,
                'temporal_signal': 0.0,
                'immune_signal': 0.0,
                'overall_signal': 0.0,
                'consensus_strength': 0.5,
                'active_weights': {'dna': 0.4, 'temporal': 0.4, 'immune': 0.2},
                'current_patterns': {
                    'dna_sequence': "",
                    'dominant_cycles': 0,
                    'active_antibodies': 0,
                    'hybrid_tools': 0
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
        # Debug and validate context
        if not isinstance(context, dict):
            logger.error(f"Context is not a dictionary: type={type(context)}, content={str(context)[:200]}")
            return
        
        # Initialize orchestrator progress task if needed
        if self.orchestrator_progress is None:
            try:
                prog = _get_progress()
                if prog is not None:
                    if not prog.live.is_started:
                        prog.start()
                    self.orchestrator_progress = prog.add_task(
                        "[blue]Orchestrator Learning[/blue]", 
                        total=None,
                        tools=4, 
                        consensus=0.0
                    )
            except Exception as e:
                logger.error(f"Error initializing orchestrator progress task: {e}")
                self.orchestrator_progress = None
        
        # Extract context with additional validation
        dna_sequence = context.get('dna_sequence', '')
        if not isinstance(dna_sequence, str):
            logger.error(f"DNA sequence is not a string: type={type(dna_sequence)}, content={str(dna_sequence)[:100]}")
            dna_sequence = str(dna_sequence) if dna_sequence else ''
        
        cycles_info = context.get('cycles_info', [])
        if not isinstance(cycles_info, list):
            logger.error(f"Cycles info is not a list: type={type(cycles_info)}, content={str(cycles_info)[:100]}")
            cycles_info = []
        
        market_state = context.get('market_state', {})
        if not isinstance(market_state, dict):
            logger.error(f"Market state is not a dict: type={type(market_state)}, content={str(market_state)[:100]}")
            market_state = {}
        
        microstructure_signal = context.get('microstructure_signal', 0.0)
        if not isinstance(microstructure_signal, (int, float)):
            logger.error(f"Microstructure signal is not numeric: type={type(microstructure_signal)}, content={str(microstructure_signal)[:100]}")
            microstructure_signal = 0.0
        
        # Each subsystem learns with additional error handling
        if dna_sequence:
            try:
                self.dna_subsystem.learn_from_outcome(dna_sequence, outcome)
            except Exception as e:
                logger.error(f"Error in DNA subsystem learning: {e}")
                logger.error(f"DNA sequence type: {type(dna_sequence)}, content: {str(dna_sequence)[:100]}")
        
        # Extract cycles from temporal subsystem's recent analysis
        recent_cycles = []
        try:
            if len(self.temporal_subsystem.dominant_cycles) > 0:
                recent_cycles = list(self.temporal_subsystem.dominant_cycles)[-1] if self.temporal_subsystem.dominant_cycles else []
        except Exception as e:
            logger.error(f"Error extracting recent cycles: {e}")
            recent_cycles = []
        
        if recent_cycles or cycles_info:
            try:
                cycles_to_learn = cycles_info if cycles_info else recent_cycles
                self.temporal_subsystem.learn_from_outcome(cycles_to_learn, outcome)
            except Exception as e:
                logger.error(f"Error in temporal subsystem learning: {e}")
                logger.error(f"Cycles to learn type: {type(cycles_to_learn)}, content: {str(cycles_to_learn)[:100]}")
        
        if market_state:
            try:
                self.immune_subsystem.learn_threat(market_state, outcome)
            except Exception as e:
                logger.error(f"Error in immune subsystem learning: {e}")
                logger.error(f"Market state type: {type(market_state)}, content: {str(market_state)[:100]}")
        
        # Track microstructure learning
        if microstructure_signal != 0.0:
            self.tool_lifecycle['microstructure']['performance_history'].append(microstructure_signal)
        
        # Performance attribution with error handling
        try:
            if self.subsystem_votes:
                recent_vote = self.subsystem_votes[-1]
                
                # Validate recent_vote structure
                if not isinstance(recent_vote, dict):
                    logger.error(f"Recent vote is not a dict: type={type(recent_vote)}, content={str(recent_vote)[:100]}")
                else:
                    votes = recent_vote.get('votes', {})
                    weights = recent_vote.get('weights', {})
                    
                    if not isinstance(votes, dict):
                        logger.error(f"Votes is not a dict: type={type(votes)}, content={str(votes)[:100]}")
                        votes = {}
                    
                    if not isinstance(weights, dict):
                        logger.error(f"Weights is not a dict: type={type(weights)}, content={str(weights)[:100]}")
                        weights = {}
                    
                    for tool, signal in votes.items():
                        try:
                            if tool in weights:
                                # Attribute performance based on signal strength and outcome alignment
                                attribution_score = outcome * signal * weights[tool]
                                self.performance_attribution[tool].append(attribution_score)
                        except Exception as e:
                            logger.error(f"Error in performance attribution for tool {tool}: {e}")
                            logger.error(f"Tool: {tool}, Signal: {signal}, Weight: {weights.get(tool, 'N/A')}")
                
                # Add microstructure attribution
                if microstructure_signal != 0.0:
                    try:
                        micro_attribution = outcome * microstructure_signal * 0.2  # 20% weight for microstructure
                        self.performance_attribution['microstructure'].append(micro_attribution)
                    except Exception as e:
                        logger.error(f"Error in microstructure attribution: {e}")
        except Exception as e:
            logger.error(f"Error in performance attribution section: {e}")
        
        # Update hybrid tool performance
        for hybrid_name, hybrid_data in self.hybrid_tools.items():
            # Simple performance tracking for now
            hybrid_data['performance_history'].append(outcome * 0.5)  # Conservative attribution
        
        # Update orchestrator progress task
        if self.orchestrator_progress is not None:
            try:
                prog = _get_progress()
                if prog is not None:
                    prog.advance(self.orchestrator_progress, 1)
                    
                    # Update progress stats
                    recent_consensus = np.mean(list(self.consensus_history)) if self.consensus_history else 0.0
                    prog.update(
                        self.orchestrator_progress,
                        tools=4 + len(self.hybrid_tools),  # DNA, Temporal, Immune, Microstructure + hybrids
                        consensus=f"{recent_consensus:.3f}"
                    )
            except Exception as e:
                logger.warning(f"Error updating orchestrator progress task: {e}")
        
        # Periodic evolution
        if len(self.subsystem_votes) % 100 == 0:
            try:
                self._evolve_tools()
            except Exception as e:
                logger.warning(f"Tool evolution failed: {e}")

    def close_progress_bars(self, microstructure_engine=None):
        """Graceful shutdown helper"""
        try:
            prog = _get_progress()
            if prog is not None:
                if prog.live.is_started:
                    prog.stop()
            
            # Reset task IDs and mark bootstrap complete
            self.dna_subsystem.learning_progress = None
            self.dna_subsystem.bootstrap_complete = True
            self.temporal_subsystem.learning_progress = None  
            self.temporal_subsystem.bootstrap_complete = True
            self.immune_subsystem.learning_progress = None
            self.immune_subsystem.bootstrap_complete = True
            
            # Handle microstructure engine if provided
            if microstructure_engine:
                microstructure_engine.learning_progress = None
                microstructure_engine.bootstrap_complete = True
            
            self.orchestrator_progress = None
            self.bootstrap_complete = True
            
        except Exception as e:
            logger.warning(f"Error closing progress display: {e}")
    
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