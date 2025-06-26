"""
DNA Evolution Engine - Genetic algorithms for pattern breeding and mutation
"""

import numpy as np
import random
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class EvolutionEngine:
    """
    Advanced evolution engine implementing genetic algorithms for DNA sequences
    """
    
    def __init__(self, bases: Dict[str, str]):
        self.bases = bases
        
    def attempt_breeding(self, current_sequence: str, sequences: Dict, generation_count: int):
        """Attempt to breed successful sequences"""
        # Find sequences with high performance for breeding
        breeding_candidates = []
        for seq, data in sequences.items():
            if data['performance'] > 0.3:  # Only breed successful patterns
                breeding_candidates.append((seq, data['performance'], data))
        
        if len(breeding_candidates) >= 2:
            # Sort by performance and select top candidates
            breeding_candidates.sort(key=lambda x: x[1], reverse=True)
            parent1_seq, parent1_score, parent1_data = breeding_candidates[0]
            parent2_seq, parent2_score, parent2_data = breeding_candidates[1]
            
            # Breed new offspring
            offspring = self._breed_sequences(parent1_seq, parent2_seq)
            if offspring and offspring not in sequences:
                sequences[offspring] = {
                    'performance': (parent1_score + parent2_score) / 2 * 0.8,
                    'age': 0,
                    'generation': generation_count,
                    'parents': (parent1_seq, parent2_seq)
                }
                logger.debug(f"DNA breeding successful: {offspring[:10]}... (Gen {generation_count})")

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
        if random.random() < 0.1:  # 10% mutation rate
            offspring = self._mutate_sequence(offspring)
        
        return offspring

    def _functional_breeding(self, parent1: str, parent2: str) -> str:
        """Breeding based on functional market patterns"""
        blocks1 = self._identify_functional_blocks(parent1)
        blocks2 = self._identify_functional_blocks(parent2)
        
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