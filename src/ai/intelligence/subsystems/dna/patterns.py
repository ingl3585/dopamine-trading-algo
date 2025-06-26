"""
DNA Pattern Matching - Sophisticated similarity algorithms for DNA sequences
"""

import numpy as np
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class PatternMatcher:
    """
    Advanced pattern matching for DNA sequences with multiple similarity metrics
    """
    
    def find_best_matches(self, sequence: str, sequences: Dict, age_decay_factor: float) -> float:
        """Find best matching patterns with breeding consideration"""
        best_score = 0.0
        best_matches = []
        matches_found = 0
        
        logger.debug(f"Analyzing DNA sequence: '{sequence[:20]}...' against {len(sequences)} stored sequences")
        
        for stored_seq, data in sequences.items():
            try:
                similarity = self._advanced_sequence_similarity(sequence, stored_seq)
                if similarity > 0.7:
                    matches_found += 1
                    # Age-adjusted performance
                    age_factor = age_decay_factor ** data['age']
                    adjusted_performance = data['performance'] * age_factor
                    score = adjusted_performance * similarity
                    
                    if not (np.isnan(score) or np.isinf(score)):
                        if abs(score) > abs(best_score):
                            best_score = score
                            best_matches.append((stored_seq, score, similarity))
                            logger.debug(f"New best DNA match: score={score:.4f}, similarity={similarity:.4f}")
            except Exception as e:
                logger.warning(f"Error analyzing sequence similarity: {e}")
                continue
        
        # If no good matches found, provide a small baseline signal
        if matches_found == 0 and len(sequences) > 0:
            logger.debug("No DNA matches found, using baseline signal")
            best_score = 0.05
        
        logger.debug(f"DNA analysis result: {best_score:.4f} (matches: {matches_found})")
        return best_score

    def _advanced_sequence_similarity(self, seq1: str, seq2: str) -> float:
        """Multiple similarity metrics combined"""
        if not seq1 or not seq2:
            return 0.0
        
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