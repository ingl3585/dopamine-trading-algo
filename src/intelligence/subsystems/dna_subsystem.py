import numpy as np
import logging
import random
from collections import deque
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class DNASubsystem:
    def __init__(self):
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
        
        self.sequences = {}
        self.max_sequences = 8000
        self.age_decay_factor = 0.999
        self.performance_threshold = 0.3
        self.elite_sequences = {}
        self.max_age = 1000
        self.breeding_frequency = 50
        self.generation_count = 0
        self.breeding_pool = deque(maxlen=100)
        self.breeding_success_rate = deque(maxlen=50)
        self.mutation_rate = 0.1
        self.total_learning_events = 0
        self.learning_batch_size = 50

    def encode_market_state(self, prices: List[float], volumes: List[float],
                           volatility: Optional[float] = None, momentum: Optional[float] = None) -> str:
        if len(prices) < 2 or len(volumes) < 2:
            logger.debug(f"Insufficient data for DNA encoding: prices={len(prices)}, volumes={len(volumes)}")
            return ""
        
        sequence = ""
        
        try:
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
        if abs(momentum) > 0.03:
            return 'L' if momentum > 0 else 'M'
        
        if volatility > 0.05:
            return 'N'
        elif volatility < 0.002 and abs(price_change) < 0.0001 and vol_ratio < 1.1:
            return 'O'
        
        if abs(price_change) < 0.0001:
            if vol_ratio > 2.0:
                return 'I'
            elif vol_ratio > 1.2:
                return 'J'
            else:
                return 'K'
        elif price_change > 0:
            if vol_ratio > 2.0:
                return 'A'
            elif vol_ratio > 1.2:
                return 'B'
            elif vol_ratio > 0.8:
                return 'C'
            else:
                return 'D'
        else:
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
            self._age_sequences()
            
            best_score = 0.0
            best_matches = []
            matches_found = 0
            
            logger.debug(f"Analyzing DNA sequence: '{sequence[:20]}...' against {len(self.sequences)} stored sequences")
            
            for stored_seq, data in self.sequences.items():
                try:
                    similarity = self._advanced_sequence_similarity(sequence, stored_seq)
                    if similarity > 0.7:
                        matches_found += 1
                        age_factor = self.age_decay_factor ** data['age']
                        adjusted_performance = data['performance'] * age_factor
                        score = adjusted_performance * similarity
                        
                        if np.isnan(score) or np.isinf(score):
                            continue
                            
                        if abs(score) > abs(best_score):
                            best_score = score
                            best_matches.append((stored_seq, score, similarity))
                            logger.debug(f"New best DNA match: score={score:.4f}, similarity={similarity:.4f}")
                except Exception as e:
                    logger.warning(f"Error analyzing sequence similarity: {e}")
                    continue
            
            if matches_found == 0 and len(self.sequences) > 0:
                logger.debug("No DNA matches found, using baseline signal")
                best_score = 0.05
            
            if len(best_matches) >= 2 and len(self.sequences) % self.breeding_frequency == 0:
                try:
                    self._attempt_breeding(best_matches)
                except Exception as e:
                    logger.warning(f"Error in breeding attempt: {e}")
            
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
        
        exact_similarity = self._exact_similarity(seq1, seq2)
        structural_similarity = self._structural_similarity(seq1, seq2)
        functional_similarity = self._functional_similarity(seq1, seq2)
        
        return (exact_similarity * 0.4 + structural_similarity * 0.3 + functional_similarity * 0.3)

    def _exact_similarity(self, seq1: str, seq2: str) -> float:
        max_len = max(len(seq1), len(seq2))
        min_len = min(len(seq1), len(seq2))
        
        matches = sum(1 for i in range(min_len) if seq1[i] == seq2[i])
        return matches / max_len

    def _structural_similarity(self, seq1: str, seq2: str) -> float:
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
        
        dot_product = sum(a * b for a, b in zip(sig1, sig2))
        norm1 = sum(a * a for a in sig1) ** 0.5
        norm2 = sum(b * b for b in sig2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

    def _get_base_type(self, base: str) -> str:
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
        to_remove = []
        for seq, data in self.sequences.items():
            data['age'] += 1
            if data['age'] > self.max_age and data['performance'] < 0.3:
                to_remove.append(seq)
        
        for seq in to_remove:
            del self.sequences[seq]

    def _attempt_breeding(self, matches: List[Tuple[str, float, float]]):
        if len(matches) < 2:
            return
        
        matches.sort(key=lambda x: x[1], reverse=True)
        parent1_seq, parent1_score, _ = matches[0]
        parent2_seq, parent2_score, _ = matches[1]
        
        if parent1_score > 0.3 and parent2_score > 0.3:
            offspring = self._breed_sequences(parent1_seq, parent2_seq)
            if offspring and offspring not in self.sequences:
                self.generation_count += 1
                self.sequences[offspring] = {
                    'performance': (parent1_score + parent2_score) / 2 * 0.8,
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
        if not parent1 or not parent2:
            return ""
        
        min_len = min(len(parent1), len(parent2))
        if min_len < 4:
            return ""
        
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
        
        else:
            offspring = self._functional_breeding(parent1, parent2)
        
        if random.random() < self.mutation_rate:
            offspring = self._mutate_sequence(offspring)
        
        return offspring

    def _functional_breeding(self, parent1: str, parent2: str) -> str:
        blocks1 = self._identify_functional_blocks(parent1)
        blocks2 = self._identify_functional_blocks(parent2)
        
        offspring = ""
        for i in range(min(len(blocks1), len(blocks2))):
            if self._evaluate_block_strength(blocks1[i]) > self._evaluate_block_strength(blocks2[i]):
                offspring += blocks1[i]
            else:
                offspring += blocks2[i]
        
        return offspring

    def _identify_functional_blocks(self, sequence: str) -> List[str]:
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
        if not block:
            return 0.0
        
        consistency_score = len(block) / 10.0
        
        special_bonus = 0.0
        if 'L' in block or 'M' in block:
            special_bonus += 0.3
        if 'P' in block:
            special_bonus += 0.2
        if 'N' in block or 'O' in block:
            special_bonus += 0.1
        
        return min(1.0, consistency_score + special_bonus)

    def _mutate_sequence(self, sequence: str) -> str:
        if not sequence:
            return sequence
        
        mutation_strategies = ['point', 'insertion', 'deletion', 'inversion']
        strategy = random.choice(mutation_strategies)
        
        if strategy == 'point':
            mutation_point = random.randint(0, len(sequence) - 1)
            new_base = random.choice(list(self.bases.keys()))
            return sequence[:mutation_point] + new_base + sequence[mutation_point+1:]
        
        elif strategy == 'insertion':
            insertion_point = random.randint(0, len(sequence))
            new_base = random.choice(list(self.bases.keys()))
            return sequence[:insertion_point] + new_base + sequence[insertion_point:]
        
        elif strategy == 'deletion':
            if len(sequence) > 1:
                deletion_point = random.randint(0, len(sequence) - 1)
                return sequence[:deletion_point] + sequence[deletion_point+1:]
        
        elif strategy == 'inversion':
            if len(sequence) > 3:
                start = random.randint(0, len(sequence) - 3)
                end = random.randint(start + 2, len(sequence))
                return sequence[:start] + sequence[start:end][::-1] + sequence[end:]
        
        return sequence

    def learn_from_outcome(self, sequence: str, outcome: float):
        if not isinstance(sequence, str):
            logger.error(f"DNA sequence is not a string: type={type(sequence)}, content={str(sequence)[:200]}")
            return
        if not sequence:
            return
        
        self.total_learning_events += 1
        
        sequence_added = False
        if sequence in self.sequences:
            data = self.sequences[sequence]
            learning_rate = 0.1 / (1 + data['age'] * 0.01)
            data['performance'] += learning_rate * (outcome - data['performance'])
            data['age'] = max(0, data['age'] - 5)
        else:
            if len(self.sequences) >= self.max_sequences:
                self._cleanup_sequences()
            
            if len(self.sequences) < self.max_sequences:
                self.sequences[sequence] = {
                    'performance': outcome * 0.5,
                    'age': 0,
                    'generation': 0,
                    'parents': None
                }
                sequence_added = True
        
        if outcome > self.performance_threshold:
            self.elite_sequences[sequence] = self.sequences[sequence].copy()
        
        if len(self.sequences) % 100 == 0:
            self._cleanup_sequences()

    def _cleanup_sequences(self):
        if len(self.sequences) > 2000:
            sequence_scores = []
            for seq, data in self.sequences.items():
                age_factor = self.age_decay_factor ** data['age']
                adjusted_score = data['performance'] * age_factor
                sequence_scores.append((seq, adjusted_score))
            
            sequence_scores.sort(key=lambda x: x[1], reverse=True)
            sequences_to_keep = {seq for seq, _ in sequence_scores[:1500]}
            sequences_to_keep.update(self.elite_sequences.keys())
            
            self.sequences = {seq: data for seq, data in self.sequences.items() 
                            if seq in sequences_to_keep}

    def get_evolution_stats(self) -> Dict:
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