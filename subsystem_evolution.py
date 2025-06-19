# subsystem_evolution.py

import numpy as np
import torch
from scipy import fft
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Tuple, Set
import logging

logger = logging.getLogger(__name__)


class DNASubsystem:
    def __init__(self):
        # 16-base DNA encoding
        self.bases = {
            'A': 'price_up_vol_high',    'B': 'price_up_vol_med',
            'C': 'price_up_vol_low',     'D': 'price_up_vol_very_low',
            'E': 'price_down_vol_high',  'F': 'price_down_vol_med', 
            'G': 'price_down_vol_low',   'H': 'price_down_vol_very_low',
            'I': 'price_flat_vol_high',  'J': 'price_flat_vol_med',
            'K': 'price_flat_vol_low',   'L': 'momentum_strong_up',
            'M': 'momentum_strong_down', 'N': 'volatility_spike',
            'O': 'volatility_crush',     'P': 'pattern_continuation'
        }
        
        # Pattern storage and breeding
        self.sequences = {}  # sequence -> performance
        self.breeding_pool = []
        self.sequence_ages = {}
        self.mutation_rate = 0.05
        
    def encode_market_state(self, prices: List[float], volumes: List[float]) -> str:
        if len(prices) < 2 or len(volumes) < 2:
            return ""
        
        sequence = ""
        
        for i in range(1, len(prices)):
            price_change = (prices[i] - prices[i-1]) / prices[i-1] if prices[i-1] != 0 else 0
            vol_ratio = volumes[i] / volumes[i-1] if volumes[i-1] > 0 else 1.0
            
            # Enhanced encoding logic
            if abs(price_change) < 0.0001:  # Flat price
                if vol_ratio > 2.0:
                    base = 'I'
                elif vol_ratio > 1.2:
                    base = 'J'
                else:
                    base = 'K'
            elif price_change > 0:  # Price up
                if vol_ratio > 2.0:
                    base = 'A'
                elif vol_ratio > 1.2:
                    base = 'B'
                elif vol_ratio > 0.8:
                    base = 'C'
                else:
                    base = 'D'
            else:  # Price down
                if vol_ratio > 2.0:
                    base = 'E'
                elif vol_ratio > 1.2:
                    base = 'F'
                elif vol_ratio > 0.8:
                    base = 'G'
                else:
                    base = 'H'
            
            # Add momentum and volatility context
            if i >= 3:
                momentum = (prices[i] - prices[i-3]) / prices[i-3] if prices[i-3] != 0 else 0
                if abs(momentum) > 0.01:
                    base = 'L' if momentum > 0 else 'M'
                
                vol_change = np.std(prices[i-3:i+1])
                if vol_change > np.mean(prices[i-3:i+1]) * 0.02:
                    base = 'N'
                elif vol_change < np.mean(prices[i-3:i+1]) * 0.005:
                    base = 'O'
            
            sequence += base
        
        return sequence
    
    def analyze_sequence(self, sequence: str) -> float:
        if not sequence or len(sequence) < 5:
            return 0.0
        
        # Find best matching patterns
        best_score = 0.0
        best_match = None
        
        for stored_seq, performance in self.sequences.items():
            similarity = self._sequence_similarity(sequence, stored_seq)
            if similarity > 0.7:
                score = performance * similarity
                if abs(score) > abs(best_score):
                    best_score = score
                    best_match = stored_seq
        
        # Breeding consideration
        if best_match and abs(best_score) > 0.3:
            self._consider_breeding(sequence, best_match)
        
        return best_score
    
    def _sequence_similarity(self, seq1: str, seq2: str) -> float:
        if not seq1 or not seq2:
            return 0.0
        
        # Advanced similarity with partial matching
        max_len = max(len(seq1), len(seq2))
        min_len = min(len(seq1), len(seq2))
        
        # Exact substring matching
        exact_matches = 0
        for i in range(min_len):
            if seq1[i] == seq2[i]:
                exact_matches += 1
        
        # Pattern structure similarity
        pattern_similarity = self._pattern_structure_similarity(seq1, seq2)
        
        return (exact_matches / max_len) * 0.7 + pattern_similarity * 0.3
    
    def _pattern_structure_similarity(self, seq1: str, seq2: str) -> float:
        # Compare structural patterns (trends, reversals, etc.)
        def get_structure(seq):
            structure = []
            for i in range(len(seq) - 1):
                if seq[i] in 'ABCD' and seq[i+1] in 'EFGH':
                    structure.append('REVERSAL')
                elif seq[i] in 'EFGH' and seq[i+1] in 'ABCD':
                    structure.append('REVERSAL')
                elif seq[i] in 'ABCD' and seq[i+1] in 'ABCD':
                    structure.append('TREND_UP')
                elif seq[i] in 'EFGH' and seq[i+1] in 'EFGH':
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
    
    def _consider_breeding(self, seq1: str, seq2: str):
        # DNA breeding: combine successful sequences
        if len(self.breeding_pool) < 100:
            offspring = self._breed_sequences(seq1, seq2)
            if offspring and offspring not in self.sequences:
                self.sequences[offspring] = 0.5  # Neutral start
                self.breeding_pool.append(offspring)
    
    def _breed_sequences(self, parent1: str, parent2: str) -> str:
        if not parent1 or not parent2:
            return ""
        
        # Crossover breeding
        min_len = min(len(parent1), len(parent2))
        crossover_point = np.random.randint(1, min_len)
        
        offspring = parent1[:crossover_point] + parent2[crossover_point:]
        
        # Mutation
        if np.random.random() < self.mutation_rate:
            offspring = self._mutate_sequence(offspring)
        
        return offspring
    
    def _mutate_sequence(self, sequence: str) -> str:
        if not sequence:
            return sequence
        
        mutation_point = np.random.randint(0, len(sequence))
        new_base = np.random.choice(list(self.bases.keys()))
        
        return sequence[:mutation_point] + new_base + sequence[mutation_point+1:]
    
    def learn_from_outcome(self, sequence: str, outcome: float):
        if not sequence:
            return
        
        # Age existing sequences
        for seq in self.sequence_ages:
            self.sequence_ages[seq] += 1
        
        # Update or add sequence
        if sequence in self.sequences:
            current_perf = self.sequences[sequence]
            learning_rate = 0.1
            self.sequences[sequence] = current_perf + learning_rate * (outcome - current_perf)
            self.sequence_ages[sequence] = 0
        else:
            self.sequences[sequence] = outcome * 0.5
            self.sequence_ages[sequence] = 0
        
        # Clean old sequences
        self._cleanup_old_sequences()
    
    def _cleanup_old_sequences(self):
        if len(self.sequences) > 1000:
            # Remove oldest 20% with poor performance
            old_sequences = [(seq, age) for seq, age in self.sequence_ages.items() 
                           if age > 100 and abs(self.sequences.get(seq, 0)) < 0.1]
            
            old_sequences.sort(key=lambda x: x[1], reverse=True)
            
            for seq, _ in old_sequences[:200]:
                self.sequences.pop(seq, None)
                self.sequence_ages.pop(seq, None)


class FFTTemporalSubsystem:
    def __init__(self):
        self.cycle_memory = {}  # frequency -> (strength, phase, performance)
        self.dominant_cycles = deque(maxlen=50)
        self.interference_patterns = {}
        
    def analyze_cycles(self, prices: List[float]) -> float:
        if len(prices) < 32:
            return 0.0
        
        # FFT analysis
        price_array = np.array(prices[-128:] if len(prices) >= 128 else prices)
        fft_result = fft.fft(price_array)
        frequencies = fft.fftfreq(len(price_array))
        
        # Find dominant frequencies
        power_spectrum = np.abs(fft_result)
        dominant_freq_idx = np.argsort(power_spectrum)[-5:]  # Top 5 frequencies
        
        signal_strength = 0.0
        cycle_info = []
        
        for idx in dominant_freq_idx:
            if frequencies[idx] != 0:  # Skip DC component
                freq = frequencies[idx]
                amplitude = power_spectrum[idx]
                phase = np.angle(fft_result[idx])
                
                cycle_period = 1.0 / abs(freq) if freq != 0 else float('inf')
                
                # Store cycle information
                cycle_key = f"freq_{freq:.4f}"
                cycle_info.append({
                    'frequency': freq,
                    'amplitude': amplitude,
                    'phase': phase,
                    'period': cycle_period
                })
                
                # Get historical performance of this cycle
                if cycle_key in self.cycle_memory:
                    _, _, performance = self.cycle_memory[cycle_key]
                    signal_strength += amplitude * performance
                else:
                    # New cycle, neutral assumption
                    signal_strength += amplitude * 0.1
        
        # Analyze cycle interference
        interference_signal = self._analyze_interference(cycle_info)
        
        # Combine signals
        final_signal = np.tanh(signal_strength / len(prices)) + interference_signal
        
        # Store dominant cycles for learning
        self.dominant_cycles.append(cycle_info)
        
        return final_signal
    
    def _analyze_interference(self, cycles: List[Dict]) -> float:
        if len(cycles) < 2:
            return 0.0
        
        interference_score = 0.0
        
        for i in range(len(cycles)):
            for j in range(i + 1, len(cycles)):
                cycle1, cycle2 = cycles[i], cycles[j]
                
                # Calculate phase difference
                phase_diff = abs(cycle1['phase'] - cycle2['phase'])
                
                # Constructive interference (phases align)
                if phase_diff < np.pi / 4 or phase_diff > 7 * np.pi / 4:
                    interference_score += (cycle1['amplitude'] + cycle2['amplitude']) * 0.1
                
                # Destructive interference (phases oppose)
                elif np.pi / 2 < phase_diff < 3 * np.pi / 2:
                    interference_score -= abs(cycle1['amplitude'] - cycle2['amplitude']) * 0.05
        
        return np.tanh(interference_score)
    
    def learn_from_outcome(self, cycles_info: List[Dict], outcome: float):
        if not cycles_info:
            return
        
        # Update performance for observed cycles
        for cycle in cycles_info:
            freq = cycle['frequency']
            cycle_key = f"freq_{freq:.4f}"
            
            if cycle_key in self.cycle_memory:
                strength, phase, performance = self.cycle_memory[cycle_key]
                new_performance = performance + 0.05 * (outcome - performance)
                self.cycle_memory[cycle_key] = (cycle['amplitude'], cycle['phase'], new_performance)
            else:
                self.cycle_memory[cycle_key] = (cycle['amplitude'], cycle['phase'], outcome * 0.3)
        
        # Store interference patterns
        if len(cycles_info) >= 2:
            pattern_key = self._create_interference_pattern_key(cycles_info)
            if pattern_key not in self.interference_patterns:
                self.interference_patterns[pattern_key] = deque(maxlen=20)
            self.interference_patterns[pattern_key].append(outcome)


class EvolvingImmuneSystem:
    def __init__(self):
        self.antibodies = {}  # pattern -> (strength, specificity, memory_count)
        self.t_cell_memory = deque(maxlen=200)
        self.threat_evolution_tracker = {}
        self.autoimmune_prevention = set()
        
    def detect_threats(self, market_state: Dict) -> float:
        threat_level = 0.0
        
        # Create current pattern signature
        pattern_signature = self._create_pattern_signature(market_state)
        
        # Check against known antibodies
        for antibody_pattern, (strength, specificity, memory) in self.antibodies.items():
            similarity = self._pattern_similarity(pattern_signature, antibody_pattern)
            
            if similarity > 0.7:  # High similarity to known threat
                threat_level += strength * similarity * (1.0 + memory * 0.1)
        
        # Check T-cell memory for rapid response
        for past_threat in self.t_cell_memory:
            if self._pattern_similarity(pattern_signature, past_threat['pattern']) > 0.8:
                threat_level += past_threat['severity'] * 1.5  # Quick response
        
        # Autoimmune prevention check
        if pattern_signature in self.autoimmune_prevention:
            threat_level *= 0.1  # Reduce false positive
        
        return -min(1.0, threat_level)  # Negative signal for threats
    
    def _create_pattern_signature(self, market_state: Dict) -> str:
        # Create a unique signature for current market conditions
        signature_parts = []
        
        # Price volatility signature
        if 'volatility' in market_state:
            vol_bucket = int(market_state['volatility'] * 100) // 10
            signature_parts.append(f"vol_{vol_bucket}")
        
        # Volume signature
        if 'volume_momentum' in market_state:
            vol_mom_bucket = int(market_state['volume_momentum'] * 50) + 50
            signature_parts.append(f"vmom_{vol_mom_bucket}")
        
        # Price momentum signature
        if 'price_momentum' in market_state:
            price_mom_bucket = int(market_state['price_momentum'] * 50) + 50
            signature_parts.append(f"pmom_{price_mom_bucket}")
        
        # Time signature
        if 'time_of_day' in market_state:
            time_bucket = int(market_state['time_of_day'] * 24)
            signature_parts.append(f"time_{time_bucket}")
        
        return "_".join(signature_parts)
    
    def _pattern_similarity(self, pattern1: str, pattern2: str) -> float:
        if not pattern1 or not pattern2:
            return 0.0
        
        parts1 = pattern1.split("_")
        parts2 = pattern2.split("_")
        
        matches = sum(1 for p1, p2 in zip(parts1, parts2) if p1 == p2)
        total_parts = max(len(parts1), len(parts2))
        
        return matches / total_parts if total_parts > 0 else 0.0
    
    def learn_threat(self, market_state: Dict, threat_level: float):
        pattern = self._create_pattern_signature(market_state)
        
        if threat_level < -0.5:  # Significant threat
            # Create or strengthen antibody
            if pattern in self.antibodies:
                strength, specificity, memory = self.antibodies[pattern]
                new_strength = min(1.0, strength + 0.2)
                new_memory = memory + 1
                self.antibodies[pattern] = (new_strength, specificity, new_memory)
            else:
                self.antibodies[pattern] = (0.5, 0.8, 1)
            
            # Add to T-cell memory
            self.t_cell_memory.append({
                'pattern': pattern,
                'severity': abs(threat_level),
                'timestamp': datetime.now()
            })
        
        elif threat_level > 0.3:  # False positive (good outcome from "threat")
            # Autoimmune prevention
            self.autoimmune_prevention.add(pattern)
            
            # Weaken antibody if it exists
            if pattern in self.antibodies:
                strength, specificity, memory = self.antibodies[pattern]
                new_strength = max(0.1, strength - 0.3)
                self.antibodies[pattern] = (new_strength, specificity, memory)
    
    def evolve_antibodies(self):
        # Evolve antibodies based on effectiveness
        evolved_antibodies = {}
        
        for pattern, (strength, specificity, memory) in self.antibodies.items():
            # Keep effective antibodies
            if memory > 2 and strength > 0.3:
                # Possible evolution through mutation
                if np.random.random() < 0.1:  # 10% mutation chance
                    evolved_pattern = self._mutate_pattern(pattern)
                    evolved_antibodies[evolved_pattern] = (strength * 0.8, specificity, 1)
                
                evolved_antibodies[pattern] = (strength, specificity, memory)
            elif memory <= 2 and strength < 0.2:
                # Remove weak antibodies
                continue
            else:
                evolved_antibodies[pattern] = (strength, specificity, memory)
        
        self.antibodies = evolved_antibodies
    
    def _mutate_pattern(self, pattern: str) -> str:
        parts = pattern.split("_")
        if len(parts) < 2:
            return pattern
        
        # Randomly mutate one component
        mutation_idx = np.random.randint(0, len(parts) // 2) * 2 + 1  # Only mutate values, not keys
        if mutation_idx < len(parts):
            try:
                current_val = int(parts[mutation_idx])
                new_val = max(0, current_val + np.random.randint(-5, 6))
                parts[mutation_idx] = str(new_val)
            except:
                pass
        
        return "_".join(parts)


class EnhancedIntelligenceOrchestrator:
    def __init__(self):
        self.dna_subsystem = DNASubsystem()
        self.temporal_subsystem = FFTTemporalSubsystem()
        self.immune_subsystem = EvolvingImmuneSystem()
        
        # Swarm intelligence
        self.subsystem_votes = deque(maxlen=100)
        self.consensus_history = deque(maxlen=50)
        
    def process_market_data(self, prices: List[float], volumes: List[float], 
                          market_features: Dict) -> Dict[str, float]:
        
        # Get subsystem signals
        dna_sequence = self.dna_subsystem.encode_market_state(prices[-20:], volumes[-20:])
        dna_signal = self.dna_subsystem.analyze_sequence(dna_sequence)
        
        temporal_signal = self.temporal_subsystem.analyze_cycles(prices)
        
        immune_signal = self.immune_subsystem.detect_threats(market_features)
        
        # Swarm intelligence - subsystems vote
        votes = {
            'dna': dna_signal,
            'temporal': temporal_signal,
            'immune': immune_signal
        }
        
        # Calculate consensus
        consensus_strength = self._calculate_consensus(votes)
        
        # Weighted overall signal
        overall_signal = (dna_signal * 0.4 + temporal_signal * 0.4 + immune_signal * 0.2)
        
        # Boost signal if high consensus
        if consensus_strength > 0.7:
            overall_signal *= 1.3
        elif consensus_strength < 0.3:
            overall_signal *= 0.7
        
        self.subsystem_votes.append(votes)
        self.consensus_history.append(consensus_strength)
        
        return {
            'dna_signal': dna_signal,
            'temporal_signal': temporal_signal,
            'immune_signal': immune_signal,
            'overall_signal': overall_signal,
            'consensus_strength': consensus_strength,
            'current_patterns': {
                'dna_sequence': dna_sequence,
                'dominant_cycles': len(self.temporal_subsystem.dominant_cycles),
                'active_antibodies': len(self.immune_subsystem.antibodies)
            }
        }
    
    def _calculate_consensus(self, votes: Dict[str, float]) -> float:
        signals = list(votes.values())
        
        if not signals:
            return 0.0
        
        # Check directional agreement
        positive_signals = sum(1 for s in signals if s > 0.1)
        negative_signals = sum(1 for s in signals if s < -0.1)
        neutral_signals = sum(1 for s in signals if abs(s) <= 0.1)
        
        total_signals = len(signals)
        
        # Consensus is high when most subsystems agree on direction
        agreement = max(positive_signals, negative_signals, neutral_signals) / total_signals
        
        return agreement
    
    def learn_from_outcome(self, outcome: float, context: Dict):
        # Extract context
        dna_sequence = context.get('dna_sequence', '')
        cycles_info = context.get('cycles_info', [])
        market_state = context.get('market_state', {})
        
        # Each subsystem learns
        self.dna_subsystem.learn_from_outcome(dna_sequence, outcome)
        
        if cycles_info:
            self.temporal_subsystem.learn_from_outcome(cycles_info, outcome)
        
        self.immune_subsystem.learn_threat(market_state, outcome)
        
        # Periodic evolution
        if len(self.subsystem_votes) % 100 == 0:
            self.immune_subsystem.evolve_antibodies()
    
    def get_intelligence_stats(self) -> Dict:
        return {
            'dna_sequences': len(self.dna_subsystem.sequences),
            'breeding_pool_size': len(self.dna_subsystem.breeding_pool),
            'dominant_cycles': len(self.temporal_subsystem.cycle_memory),
            'antibodies': len(self.immune_subsystem.antibodies),
            'threat_memory': len(self.immune_subsystem.t_cell_memory),
            'recent_consensus': np.mean(list(self.consensus_history)) if self.consensus_history else 0.0
        }