"""
DNA Subsystem Domain - Core 16-base market pattern encoding logic
Implements sophisticated genetic algorithms for pattern evolution
"""

import numpy as np
import logging
from collections import deque
from typing import Dict, List, Tuple, Optional

from .patterns import PatternMatcher
from .evolution import EvolutionEngine

logger = logging.getLogger(__name__)

class DNASubsystem:
    """
    Advanced DNA Subsystem with 16-base encoding per prompt.txt:
    - 16-base encoding including volume signatures, volatility patterns, momentum directions
    - DNA breeding where successful sequences combine to create offspring patterns
    - DNA aging where old patterns lose influence unless reinforced by recent success
    - Adaptive mutation rates that change based on market volatility
    """
    
    def __init__(self):
        # 16-base DNA encoding for market patterns (from prompt.txt)
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
        self.sequences = {}
        self.max_sequences = 8000
        self.age_decay_factor = 0.999
        self.performance_threshold = 0.3
        self.elite_sequences = {}
        self.max_age = 1000
        
        # Evolution and breeding parameters
        self.breeding_frequency = 50
        self.generation_count = 0
        self.breeding_pool = deque(maxlen=100)
        self.breeding_success_rate = deque(maxlen=50)
        self.mutation_rate = 0.1
        self.total_learning_events = 0
        self.learning_batch_size = 50
        
        # Domain services
        self.pattern_matcher = PatternMatcher()
        self.evolution_engine = EvolutionEngine(self.bases)

    def encode_market_state(self, prices: List[float], volumes: List[float],
                           volatility: Optional[float] = None, momentum: Optional[float] = None) -> str:
        """Convert market data into 16-base DNA sequence"""
        if len(prices) < 2 or len(volumes) < 2:
            logger.debug(f"Insufficient data for DNA encoding: prices={len(prices)}, volumes={len(volumes)}")
            return ""
        
        try:
            # Calculate market context if not provided
            if volatility is None:
                volatility = np.std(prices[-10:]) / np.mean(prices[-10:]) if len(prices) >= 10 else 0
            if momentum is None:
                momentum = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 and prices[-5] != 0 else 0
            
            sequence = ""
            for i in range(1, len(prices)):
                price_change = (prices[i] - prices[i-1]) / prices[i-1] if prices[i-1] != 0 else 0
                vol_ratio = volumes[i] / volumes[i-1] if volumes[i-1] > 0 else 1.0
                
                base = self._determine_base(price_change, vol_ratio, volatility, momentum)
                sequence += base
            
            if sequence:
                logger.debug(f"DNA sequence encoded: {sequence[:10]}... (length: {len(sequence)})")
            return sequence
            
        except Exception as e:
            logger.error(f"Error in DNA encoding: {e}")
            return ""

    def analyze_sequence(self, sequence: str) -> float:
        """Analyze DNA sequence and return signal strength"""
        if not sequence or len(sequence) < 5:
            return 0.0
        
        try:
            self._age_sequences()
            
            # Find best matching patterns using sophisticated similarity algorithms
            best_score = self.pattern_matcher.find_best_matches(
                sequence, self.sequences, self.age_decay_factor
            )
            
            # Consider breeding if we have good matches
            if len(self.sequences) % self.breeding_frequency == 0:
                self.evolution_engine.attempt_breeding(
                    sequence, self.sequences, self.generation_count
                )
            
            # Ensure valid return value
            if np.isnan(best_score) or np.isinf(best_score):
                return 0.0
            
            return best_score
            
        except Exception as e:
            logger.error(f"Error in DNA sequence analysis: {e}")
            return 0.0

    def learn_from_outcome(self, sequence: str, outcome: float):
        """Learn from trade outcome to improve DNA patterns"""
        if not isinstance(sequence, str) or not sequence:
            return
        
        self.total_learning_events += 1
        
        # Update or add sequence
        if sequence in self.sequences:
            data = self.sequences[sequence]
            learning_rate = 0.1 / (1 + data['age'] * 0.01)
            data['performance'] += learning_rate * (outcome - data['performance'])
            data['age'] = max(0, data['age'] - 5)  # Rejuvenate successful sequences
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
        
        # Update elite sequences
        if outcome > self.performance_threshold:
            self.elite_sequences[sequence] = self.sequences[sequence].copy()
        
        # Periodic cleanup
        if len(self.sequences) % 100 == 0:
            self._cleanup_sequences()

    def _determine_base(self, price_change: float, vol_ratio: float, volatility: float, momentum: float) -> str:
        """Determine DNA base using 16-base encoding logic"""
        # Strong momentum patterns (priority)
        if abs(momentum) > 0.03:
            return 'L' if momentum > 0 else 'M'
        
        # Extreme volatility patterns
        if volatility > 0.05:
            return 'N'  # Volatility spike
        elif volatility < 0.002 and abs(price_change) < 0.0001 and vol_ratio < 1.1:
            return 'O'  # Volatility crush
        
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

    def _age_sequences(self):
        """Age all sequences and remove very old ones"""
        to_remove = []
        for seq, data in self.sequences.items():
            data['age'] += 1
            if data['age'] > self.max_age and data['performance'] < 0.3:
                to_remove.append(seq)
        
        for seq in to_remove:
            del self.sequences[seq]

    def _cleanup_sequences(self):
        """Remove poor performing and very old sequences"""
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