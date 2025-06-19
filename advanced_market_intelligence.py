# intelligence_engine.py

import json
import logging
import numpy as np
from datetime import datetime
from collections import defaultdict
from typing import Dict, List

log = logging.getLogger(__name__)

class SimpleIntelligenceEngine:
    """
    Simplified intelligence engine with 4 subsystems:
    - DNA: Pattern sequences in price/volume
    - Micro: Short-term patterns  
    - Temporal: Time-based patterns
    - Immune: Remember bad patterns
    """
    
    def __init__(self, memory_file="data/intelligence_memory.json"):
        self.memory_file = memory_file
        
        # Subsystem memories
        self.dna_patterns = {}  # sequence -> success_rate
        self.micro_patterns = {}  # pattern_id -> success_rate
        self.temporal_patterns = {}  # time_key -> success_rate
        self.immune_patterns = set()  # patterns to avoid
        
        # Simple statistics
        self.pattern_outcomes = defaultdict(list)
        
        # Load existing memory
        self.load_memory()
        
        log.info(f"Intelligence engine loaded: {len(self.dna_patterns)} DNA patterns, "
                f"{len(self.micro_patterns)} micro patterns, "
                f"{len(self.temporal_patterns)} temporal patterns, "
                f"{len(self.immune_patterns)} immune patterns")
    
    def process_market_data(self, prices: List[float], volumes: List[float], timestamp: datetime = None) -> Dict:
        """Process market data through all subsystems"""
        if timestamp is None:
            timestamp = datetime.now()
        
        if len(prices) < 10:
            return self._empty_result()
        
        # Extract features from each subsystem
        dna_signal = self._process_dna(prices, volumes)
        micro_signal = self._process_micro(prices, volumes)
        temporal_signal = self._process_temporal(timestamp)
        immune_signal = self._process_immune(prices, volumes, timestamp)
        
        # Combine signals (simple weighted average)
        subsystem_signals = {
            'dna': dna_signal,
            'micro': micro_signal, 
            'temporal': temporal_signal,
            'immune': immune_signal
        }
        
        # Calculate overall signal
        weights = [0.3, 0.3, 0.2, 0.2]  # DNA, Micro, Temporal, Immune
        signals = list(subsystem_signals.values())
        overall_signal = sum(w * s for w, s in zip(weights, signals))
        
        return {
            'overall_signal': overall_signal,
            'subsystem_signals': subsystem_signals,
            'confidence': abs(overall_signal),
            'current_patterns': self._get_current_patterns(prices, volumes, timestamp)
        }
    
    def _process_dna(self, prices: List[float], volumes: List[float]) -> float:
        """DNA subsystem - encode price/volume as sequences"""
        if len(prices) < 15:
            return 0.0
        
        # Create DNA sequence
        sequence = self._create_dna_sequence(prices[-15:], volumes[-15:])
        
        # Check for similar patterns
        best_match = None
        best_similarity = 0
        
        for stored_seq in self.dna_patterns:
            similarity = self._sequence_similarity(sequence, stored_seq)
            if similarity > best_similarity and similarity > 0.7:
                best_similarity = similarity
                best_match = stored_seq
        
        if best_match:
            success_rate = self.dna_patterns[best_match]
            return (success_rate - 0.5) * 2  # Convert to -1 to +1 range
        
        return 0.0  # No pattern found
    
    def _process_micro(self, prices: List[float], volumes: List[float]) -> float:
        """Micro subsystem - short-term patterns"""
        if len(prices) < 10:
            return 0.0
        
        # Create micro pattern
        recent_prices = prices[-10:]
        price_changes = [recent_prices[i] - recent_prices[i-1] for i in range(1, len(recent_prices))]
        
        # Simple pattern: direction and magnitude
        avg_change = np.mean(price_changes)
        volatility = np.std(price_changes)
        
        pattern_id = f"micro_{int(avg_change * 1000)}_{int(volatility * 1000)}"
        
        if pattern_id in self.micro_patterns:
            success_rate = self.micro_patterns[pattern_id]
            return (success_rate - 0.5) * 2
        
        return 0.0
    
    def _process_temporal(self, timestamp: datetime) -> float:
        """Temporal subsystem - time-based patterns"""
        hour = timestamp.hour
        minute = timestamp.minute // 15 * 15  # Round to 15-min intervals
        day_of_week = timestamp.weekday()
        
        time_key = f"{day_of_week}_{hour}_{minute}"
        
        if time_key in self.temporal_patterns:
            success_rate = self.temporal_patterns[time_key]
            return (success_rate - 0.5) * 2
        
        return 0.0
    
    def _process_immune(self, prices: List[float], volumes: List[float], timestamp: datetime) -> float:
        """Immune subsystem - avoid bad patterns"""
        current_pattern = self._create_immune_pattern(prices, volumes, timestamp)
        
        if current_pattern in self.immune_patterns:
            return -0.8  # Strong avoid signal
        
        return 0.0  # Neutral
    
    def _create_dna_sequence(self, prices: List[float], volumes: List[float]) -> str:
        """Create DNA sequence from price/volume data"""
        sequence = ""
        
        for i in range(1, len(prices)):
            price_change = (prices[i] - prices[i-1]) / prices[i-1]
            vol_ratio = volumes[i] / volumes[i-1] if volumes[i-1] > 0 else 1.0
            
            # Encode as DNA bases
            if price_change > 0.001:
                sequence += 'A' if vol_ratio > 1.2 else 'T'
            else:
                sequence += 'C' if vol_ratio > 1.2 else 'G'
        
        return sequence
    
    def _sequence_similarity(self, seq1: str, seq2: str) -> float:
        """Calculate similarity between two sequences"""
        if not seq1 or not seq2:
            return 0.0
        
        min_len = min(len(seq1), len(seq2))
        matches = sum(1 for i in range(min_len) if seq1[i] == seq2[i])
        
        return matches / max(len(seq1), len(seq2))
    
    def _create_immune_pattern(self, prices: List[float], volumes: List[float], timestamp: datetime) -> str:
        """Create pattern for immune system"""
        if len(prices) < 5:
            return ""
        
        recent_change = (prices[-1] - prices[-5]) / prices[-5]
        volatility = np.std(prices[-10:]) if len(prices) >= 10 else 0
        hour = timestamp.hour
        
        return f"immune_{int(recent_change * 1000)}_{int(volatility * 1000)}_{hour}"
    
    def _get_current_patterns(self, prices: List[float], volumes: List[float], timestamp: datetime) -> Dict:
        """Get current patterns for debugging"""
        return {
            'dna_sequence': self._create_dna_sequence(prices[-15:], volumes[-15:]) if len(prices) >= 15 else "",
            'micro_pattern': f"micro_{len(prices)}",
            'temporal_key': f"{timestamp.weekday()}_{timestamp.hour}_{timestamp.minute // 15 * 15}",
            'immune_pattern': self._create_immune_pattern(prices, volumes, timestamp)
        }
    
    def learn_from_outcome(self, signal_data: Dict, outcome: float):
        """Learn from trading outcome"""
        current_patterns = signal_data.get('current_patterns', {})
        
        # Update DNA patterns
        dna_seq = current_patterns.get('dna_sequence', '')
        if dna_seq:
            if dna_seq not in self.dna_patterns:
                self.dna_patterns[dna_seq] = 0.5  # Start neutral
            
            # Simple learning rule
            current_rate = self.dna_patterns[dna_seq]
            learning_rate = 0.1
            
            if outcome > 0:
                self.dna_patterns[dna_seq] = min(0.95, current_rate + learning_rate)
            else:
                self.dna_patterns[dna_seq] = max(0.05, current_rate - learning_rate)
        
        # Update micro patterns
        micro_pattern = current_patterns.get('micro_pattern', '')
        if micro_pattern:
            if micro_pattern not in self.micro_patterns:
                self.micro_patterns[micro_pattern] = 0.5
            
            current_rate = self.micro_patterns[micro_pattern]
            if outcome > 0:
                self.micro_patterns[micro_pattern] = min(0.95, current_rate + 0.1)
            else:
                self.micro_patterns[micro_pattern] = max(0.05, current_rate - 0.1)
        
        # Update temporal patterns
        temporal_key = current_patterns.get('temporal_key', '')
        if temporal_key:
            if temporal_key not in self.temporal_patterns:
                self.temporal_patterns[temporal_key] = 0.5
            
            current_rate = self.temporal_patterns[temporal_key]
            if outcome > 0:
                self.temporal_patterns[temporal_key] = min(0.95, current_rate + 0.05)
            else:
                self.temporal_patterns[temporal_key] = max(0.05, current_rate - 0.05)
        
        # Update immune system (add bad patterns)
        if outcome < -50:  # Bad outcome
            immune_pattern = current_patterns.get('immune_pattern', '')
            if immune_pattern:
                self.immune_patterns.add(immune_pattern)
                log.info(f"Added immune pattern: {immune_pattern}")
        
        # Clean up old patterns periodically
        self._cleanup_patterns()
        
        log.info(f"Learning: outcome={outcome:.2f}, DNA={len(self.dna_patterns)}, "
                f"Micro={len(self.micro_patterns)}, Temporal={len(self.temporal_patterns)}")
    
    def _cleanup_patterns(self):
        """Remove old or poor-performing patterns"""
        # Limit pattern counts
        max_patterns = 1000
        
        if len(self.dna_patterns) > max_patterns:
            # Remove worst performing patterns
            sorted_patterns = sorted(self.dna_patterns.items(), key=lambda x: abs(x[1] - 0.5))
            keep_patterns = dict(sorted_patterns[-max_patterns//2:])
            self.dna_patterns = keep_patterns
        
        if len(self.micro_patterns) > max_patterns:
            sorted_patterns = sorted(self.micro_patterns.items(), key=lambda x: abs(x[1] - 0.5))
            self.micro_patterns = dict(sorted_patterns[-max_patterns//2:])
        
        if len(self.temporal_patterns) > max_patterns:
            sorted_patterns = sorted(self.temporal_patterns.items(), key=lambda x: abs(x[1] - 0.5))
            self.temporal_patterns = dict(sorted_patterns[-max_patterns//2:])
    
    def _empty_result(self):
        """Return empty result when insufficient data"""
        return {
            'overall_signal': 0.0,
            'subsystem_signals': {'dna': 0.0, 'micro': 0.0, 'temporal': 0.0, 'immune': 0.0},
            'confidence': 0.0,
            'current_patterns': {}
        }
    
    def save_memory(self):
        """Save all patterns to file"""
        try:
            memory_data = {
                'dna_patterns': self.dna_patterns,
                'micro_patterns': self.micro_patterns,
                'temporal_patterns': self.temporal_patterns,
                'immune_patterns': list(self.immune_patterns),
                'saved_at': datetime.now().isoformat()
            }
            
            with open(self.memory_file, 'w') as f:
                json.dump(memory_data, f, indent=2)
            
            log.info(f"Saved intelligence memory to {self.memory_file}")
        except Exception as e:
            log.error(f"Failed to save memory: {e}")
    
    def load_memory(self):
        """Load patterns from file"""
        try:
            with open(self.memory_file, 'r') as f:
                memory_data = json.load(f)
            
            self.dna_patterns = memory_data.get('dna_patterns', {})
            self.micro_patterns = memory_data.get('micro_patterns', {})
            self.temporal_patterns = memory_data.get('temporal_patterns', {})
            self.immune_patterns = set(memory_data.get('immune_patterns', []))
            
            log.info(f"Loaded intelligence memory from {self.memory_file}")
        except Exception as e:
            log.info(f"Starting with empty memory: {e}")
    
    def get_status(self):
        """Get current status"""
        return {
            'dna_patterns': len(self.dna_patterns),
            'micro_patterns': len(self.micro_patterns),
            'temporal_patterns': len(self.temporal_patterns),
            'immune_patterns': len(self.immune_patterns),
            'top_dna_patterns': sorted(self.dna_patterns.items(), 
                                     key=lambda x: abs(x[1] - 0.5), reverse=True)[:5]
        }

# Factory function
def create_intelligence_engine():
    return SimpleIntelligenceEngine()