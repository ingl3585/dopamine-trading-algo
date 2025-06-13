# advanced_market_intelligence.py

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import sqlite3
import hashlib
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
from collections import defaultdict, deque
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

@dataclass
class MarketPattern:
    """Represents a market DNA pattern"""
    sequence: str
    success_rate: float
    occurrences: int
    last_seen: datetime
    outcomes: List[float] = field(default_factory=list)
    confidence: float = 0.0
    
    def update_outcome(self, outcome: float):
        """Update pattern with new outcome"""
        self.outcomes.append(outcome)
        if len(self.outcomes) > 100:  # Keep last 100 outcomes
            self.outcomes = self.outcomes[-100:]
        self.success_rate = sum(1 for x in self.outcomes if x > 0) / len(self.outcomes)
        self.occurrences = len(self.outcomes)
        self.last_seen = datetime.now()
        self.confidence = min(self.occurrences / 10, 1.0)  # Max confidence at 10+ occurrences

@dataclass
class MicroPattern:
    """Micro-pattern from 5-15 minute windows"""
    pattern_id: str
    price_sequence: List[float]
    volume_sequence: List[float]
    timeframe: str
    success_outcomes: List[float] = field(default_factory=list)
    failure_outcomes: List[float] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        total = len(self.success_outcomes) + len(self.failure_outcomes)
        return len(self.success_outcomes) / total if total > 0 else 0.0
    
    @property
    def sample_size(self) -> int:
        return len(self.success_outcomes) + len(self.failure_outcomes)

@dataclass
class TemporalPattern:
    """Time-based pattern tracking"""
    hour: int
    minute: int
    day_of_week: int
    pattern_type: str
    success_count: int = 0
    failure_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0
    
    @property
    def confidence(self) -> float:
        total = self.success_count + self.failure_count
        return min(total / 20, 1.0)  # Max confidence at 20+ samples

class PermanentMemoryDB:
    """SQLite-based permanent memory system"""
    
    def __init__(self, db_path: str = "market_intelligence.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Market DNA patterns
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dna_patterns (
                sequence TEXT PRIMARY KEY,
                success_rate REAL,
                occurrences INTEGER,
                last_seen TEXT,
                outcomes TEXT,
                confidence REAL
            )
        ''')
        
        # Micro patterns
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS micro_patterns (
                pattern_id TEXT PRIMARY KEY,
                price_sequence TEXT,
                volume_sequence TEXT,
                timeframe TEXT,
                success_outcomes TEXT,
                failure_outcomes TEXT,
                created_at TEXT
            )
        ''')
        
        # Temporal patterns
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS temporal_patterns (
                id TEXT PRIMARY KEY,
                hour INTEGER,
                minute INTEGER,
                day_of_week INTEGER,
                pattern_type TEXT,
                success_count INTEGER,
                failure_count INTEGER,
                last_updated TEXT
            )
        ''')
        
        # Immune system (failed patterns)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS immune_patterns (
                pattern_hash TEXT PRIMARY KEY,
                pattern_data TEXT,
                failure_count INTEGER,
                last_failure TEXT,
                pattern_type TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_dna_pattern(self, pattern: MarketPattern):
        """Save DNA pattern to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO dna_patterns 
            (sequence, success_rate, occurrences, last_seen, outcomes, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            pattern.sequence,
            pattern.success_rate,
            pattern.occurrences,
            pattern.last_seen.isoformat(),
            json.dumps(pattern.outcomes),
            pattern.confidence
        ))
        
        conn.commit()
        conn.close()
    
    def load_dna_patterns(self) -> Dict[str, MarketPattern]:
        """Load all DNA patterns from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM dna_patterns')
        rows = cursor.fetchall()
        
        patterns = {}
        for row in rows:
            pattern = MarketPattern(
                sequence=row[0],
                success_rate=row[1],
                occurrences=row[2],
                last_seen=datetime.fromisoformat(row[3]),
                outcomes=json.loads(row[4]),
                confidence=row[5]
            )
            patterns[row[0]] = pattern
        
        conn.close()
        return patterns

class DNASequencingSystem:
    """Encode price/volume patterns as genetic sequences"""
    
    def __init__(self, memory_db: PermanentMemoryDB):
        self.memory_db = memory_db
        self.dna_patterns = memory_db.load_dna_patterns()
        self.base_mapping = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
        self.reverse_mapping = {0: 'A', 1: 'T', 2: 'G', 3: 'C'}
        
    def encode_price_movement(self, prices: List[float], window: int = 10) -> str:
        """Encode price movements as DNA sequence"""
        if len(prices) < window:
            return ""
        
        sequence = ""
        recent_prices = prices[-window:]
        
        for i in range(1, len(recent_prices)):
            change = (recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
            
            if change > 0.002:     # Strong up
                sequence += 'A'
            elif change > 0:       # Weak up
                sequence += 'T'
            elif change > -0.002:  # Weak down
                sequence += 'G'
            else:                  # Strong down
                sequence += 'C'
        
        return sequence
    
    def encode_volume_pattern(self, volumes: List[float], prices: List[float], window: int = 10) -> str:
        """Encode volume patterns relative to price movement"""
        if len(volumes) < window or len(prices) < window:
            return ""
        
        sequence = ""
        recent_volumes = volumes[-window:]
        recent_prices = prices[-window:]
        avg_volume = np.mean(recent_volumes)
        
        for i in range(1, len(recent_volumes)):
            vol_ratio = recent_volumes[i] / avg_volume if avg_volume > 0 else 1
            price_change = (recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
            
            # Combine volume and price direction
            if vol_ratio > 1.5 and price_change > 0:      # High volume up
                sequence += 'A'
            elif vol_ratio > 1.5 and price_change < 0:    # High volume down
                sequence += 'C'
            elif vol_ratio < 0.7 and price_change > 0:    # Low volume up
                sequence += 'T'
            else:                                          # Low volume down or neutral
                sequence += 'G'
        
        return sequence
    
    def create_dna_sequence(self, prices: List[float], volumes: List[float]) -> str:
        """Create combined DNA sequence from price and volume"""
        price_dna = self.encode_price_movement(prices)
        volume_dna = self.encode_volume_pattern(volumes, prices)
        
        # Interleave sequences
        combined = ""
        max_len = max(len(price_dna), len(volume_dna))
        
        for i in range(max_len):
            if i < len(price_dna):
                combined += price_dna[i]
            if i < len(volume_dna):
                combined += volume_dna[i]
        
        return combined
    
    def find_similar_patterns(self, current_sequence: str, similarity_threshold: float = 0.7) -> List[MarketPattern]:
        """Find similar DNA patterns in memory"""
        similar_patterns = []
        
        for seq, pattern in self.dna_patterns.items():
            if len(seq) == 0 or len(current_sequence) == 0:
                continue
                
            similarity = self.calculate_sequence_similarity(current_sequence, seq)
            if similarity >= similarity_threshold:
                similar_patterns.append(pattern)
        
        return sorted(similar_patterns, key=lambda x: x.confidence, reverse=True)
    
    def calculate_sequence_similarity(self, seq1: str, seq2: str) -> float:
        """Calculate similarity between two DNA sequences"""
        if not seq1 or not seq2:
            return 0.0
        
        min_len = min(len(seq1), len(seq2))
        max_len = max(len(seq1), len(seq2))
        
        if max_len == 0:
            return 0.0
        
        matches = sum(1 for i in range(min_len) if seq1[i] == seq2[i])
        return matches / max_len
    
    def update_pattern_outcome(self, sequence: str, outcome: float):
        """Update pattern with trading outcome"""
        if sequence in self.dna_patterns:
            self.dna_patterns[sequence].update_outcome(outcome)
        else:
            pattern = MarketPattern(
                sequence=sequence,
                success_rate=1.0 if outcome > 0 else 0.0,
                occurrences=1,
                last_seen=datetime.now(),
                outcomes=[outcome]
            )
            self.dna_patterns[sequence] = pattern
        
        # Save to permanent memory
        self.memory_db.save_dna_pattern(self.dna_patterns[sequence])

class MicroPatternNetwork:
    """Neural network for detecting micro-patterns"""
    
    def __init__(self, memory_db: PermanentMemoryDB):
        self.memory_db = memory_db
        self.patterns: Dict[str, MicroPattern] = {}
        self.neural_network = MLPClassifier(
            hidden_layer_sizes=(50, 30),
            max_iter=1000,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def extract_micro_pattern(self, prices: List[float], volumes: List[float], 
                            window: int = 15) -> Optional[str]:
        """Extract micro-pattern from recent data"""
        if len(prices) < window or len(volumes) < window:
            return None
        
        recent_prices = prices[-window:]
        recent_volumes = volumes[-window:]
        
        # Normalize to relative changes
        price_changes = [
            (recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1] 
            for i in range(1, len(recent_prices))
        ]
        
        volume_changes = [
            (recent_volumes[i] - recent_volumes[i-1]) / recent_volumes[i-1] if recent_volumes[i-1] > 0 else 0
            for i in range(1, len(recent_volumes))
        ]
        
        # Create pattern signature
        pattern_data = {
            'price_trend': np.mean(price_changes),
            'price_volatility': np.std(price_changes),
            'volume_trend': np.mean(volume_changes),
            'volume_spike': max(volume_changes) if volume_changes else 0,
            'momentum': price_changes[-1] if price_changes else 0
        }
        
        # Hash pattern for unique ID
        pattern_str = json.dumps(pattern_data, sort_keys=True)
        pattern_id = hashlib.md5(pattern_str.encode()).hexdigest()[:16]
        
        return pattern_id
    
    def record_pattern_outcome(self, pattern_id: str, outcome: float, 
                             prices: List[float], volumes: List[float]):
        """Record outcome for a micro-pattern"""
        if pattern_id not in self.patterns:
            self.patterns[pattern_id] = MicroPattern(
                pattern_id=pattern_id,
                price_sequence=prices[-15:] if len(prices) >= 15 else prices,
                volume_sequence=volumes[-15:] if len(volumes) >= 15 else volumes,
                timeframe="15m"
            )
        
        pattern = self.patterns[pattern_id]
        if outcome > 0:
            pattern.success_outcomes.append(outcome)
        else:
            pattern.failure_outcomes.append(outcome)
    
    def get_pattern_prediction(self, pattern_id: str) -> Tuple[float, float]:
        """Get prediction for a pattern (probability, confidence)"""
        if pattern_id not in self.patterns:
            return 0.5, 0.0
        
        pattern = self.patterns[pattern_id]
        if pattern.sample_size < 3:
            return 0.5, 0.0
        
        confidence = min(pattern.sample_size / 20, 1.0)
        return pattern.success_rate, confidence

class TemporalPatternArchaeologist:
    """Track time-based patterns"""
    
    def __init__(self, memory_db: PermanentMemoryDB):
        self.memory_db = memory_db
        self.temporal_patterns: Dict[str, TemporalPattern] = {}
        
    def record_temporal_event(self, timestamp: datetime, pattern_type: str, outcome: float):
        """Record a temporal pattern event"""
        hour = timestamp.hour
        minute = timestamp.minute // 15 * 15  # Round to 15-min intervals
        day_of_week = timestamp.weekday()
        
        pattern_key = f"{hour}_{minute}_{day_of_week}_{pattern_type}"
        
        if pattern_key not in self.temporal_patterns:
            self.temporal_patterns[pattern_key] = TemporalPattern(
                hour=hour,
                minute=minute,
                day_of_week=day_of_week,
                pattern_type=pattern_type
            )
        
        pattern = self.temporal_patterns[pattern_key]
        if outcome > 0:
            pattern.success_count += 1
        else:
            pattern.failure_count += 1
        pattern.last_updated = timestamp
    
    def get_temporal_strength(self, timestamp: datetime, pattern_type: str) -> float:
        """Get strength of temporal pattern at given time"""
        hour = timestamp.hour
        minute = timestamp.minute // 15 * 15
        day_of_week = timestamp.weekday()
        
        pattern_key = f"{hour}_{minute}_{day_of_week}_{pattern_type}"
        
        if pattern_key in self.temporal_patterns:
            pattern = self.temporal_patterns[pattern_key]
            if pattern.confidence > 0.3:  # Require some confidence
                return pattern.success_rate
        
        return 0.5  # Neutral if no pattern found
    
    def get_optimal_times(self, pattern_type: str, min_confidence: float = 0.5) -> List[Tuple[int, int, int, float]]:
        """Get optimal times for a pattern type"""
        optimal_times = []
        
        for pattern in self.temporal_patterns.values():
            if (pattern.pattern_type == pattern_type and 
                pattern.confidence >= min_confidence and
                pattern.success_rate > 0.6):
                optimal_times.append((
                    pattern.hour, 
                    pattern.minute, 
                    pattern.day_of_week, 
                    pattern.success_rate
                ))
        
        return sorted(optimal_times, key=lambda x: x[3], reverse=True)

class MarketImmuneSystem:
    """Remember and avoid losing patterns"""
    
    def __init__(self, memory_db: PermanentMemoryDB):
        self.memory_db = memory_db
        self.immune_patterns: Dict[str, Dict] = {}
        self.beneficial_patterns: Dict[str, Dict] = {}
        
    def add_pathogen(self, pattern_data: Dict, failure_outcome: float):
        """Add a losing pattern as pathogen"""
        pattern_hash = self.hash_pattern(pattern_data)
        
        if pattern_hash not in self.immune_patterns:
            self.immune_patterns[pattern_hash] = {
                'pattern': pattern_data,
                'failure_count': 0,
                'total_loss': 0.0,
                'first_seen': datetime.now(),
                'last_failure': datetime.now()
            }
        
        pathogen = self.immune_patterns[pattern_hash]
        pathogen['failure_count'] += 1
        pathogen['total_loss'] += abs(failure_outcome)
        pathogen['last_failure'] = datetime.now()
    
    def add_beneficial_pattern(self, pattern_data: Dict, success_outcome: float):
        """Add a successful pattern as beneficial"""
        pattern_hash = self.hash_pattern(pattern_data)
        
        if pattern_hash not in self.beneficial_patterns:
            self.beneficial_patterns[pattern_hash] = {
                'pattern': pattern_data,
                'success_count': 0,
                'total_profit': 0.0,
                'first_seen': datetime.now(),
                'last_success': datetime.now()
            }
        
        beneficial = self.beneficial_patterns[pattern_hash]
        beneficial['success_count'] += 1
        beneficial['total_profit'] += success_outcome
        beneficial['last_success'] = datetime.now()
    
    def is_dangerous_pattern(self, pattern_data: Dict, threshold: int = 3) -> bool:
        """Check if pattern is recognized as dangerous"""
        pattern_hash = self.hash_pattern(pattern_data)
        
        if pattern_hash in self.immune_patterns:
            pathogen = self.immune_patterns[pattern_hash]
            return pathogen['failure_count'] >= threshold
        
        return False
    
    def is_beneficial_pattern(self, pattern_data: Dict, threshold: int = 2) -> bool:
        """Check if pattern is recognized as beneficial"""
        pattern_hash = self.hash_pattern(pattern_data)
        
        if pattern_hash in self.beneficial_patterns:
            beneficial = self.beneficial_patterns[pattern_hash]
            return beneficial['success_count'] >= threshold
        
        return False
    
    def hash_pattern(self, pattern_data: Dict) -> str:
        """Create hash for pattern data"""
        pattern_str = json.dumps(pattern_data, sort_keys=True)
        return hashlib.md5(pattern_str.encode()).hexdigest()
    
    def get_immune_strength(self) -> float:
        """Get overall immune system strength"""
        total_patterns = len(self.immune_patterns) + len(self.beneficial_patterns)
        if total_patterns == 0:
            return 0.0
        
        return len(self.beneficial_patterns) / total_patterns

class MetaLearningDirector:
    """Master controller that learns optimal subsystem combinations"""
    
    def __init__(self, memory_db: PermanentMemoryDB):
        super().__init__(memory_db)
        self.disagreement_patterns = defaultdict(list)
        self.consensus_requirements = {
            'min_agreeing_systems': 2,
            'max_disagreement_threshold': 0.4,
            'high_confidence_override': 0.8
        }
    
    def get_consensus_signal(self, subsystem_signals: Dict[str, float], 
                           subsystem_scores: Dict[str, float]) -> Tuple[float, float, str]:
        """Get signal with consensus validation"""
        
        # Check for high-confidence override
        for system, score in subsystem_scores.items():
            if score > self.consensus_requirements['high_confidence_override']:
                signal = subsystem_signals[system]
                if abs(signal) > 0.3:  # Strong signal
                    return signal, score, f"high_confidence_override_{system}"
        
        # Count agreement
        positive_systems = []
        negative_systems = []
        neutral_systems = []
        
        for system, signal in subsystem_signals.items():
            if signal > 0.15:
                positive_systems.append((system, signal, subsystem_scores[system]))
            elif signal < -0.15:
                negative_systems.append((system, signal, subsystem_scores[system]))
            else:
                neutral_systems.append((system, signal, subsystem_scores[system]))
        
        # Require minimum agreement
        if len(positive_systems) >= self.consensus_requirements['min_agreeing_systems']:
            # Weighted average of agreeing systems
            total_weight = sum(score for _, _, score in positive_systems)
            weighted_signal = sum(signal * score for _, signal, score in positive_systems)
            return weighted_signal / total_weight if total_weight > 0 else 0, \
                   min(total_weight / len(positive_systems), 1.0), \
                   f"consensus_buy_{len(positive_systems)}"
                   
        elif len(negative_systems) >= self.consensus_requirements['min_agreeing_systems']:
            total_weight = sum(score for _, _, score in negative_systems)
            weighted_signal = sum(signal * score for _, signal, score in negative_systems)
            return weighted_signal / total_weight if total_weight > 0 else 0, \
                   min(total_weight / len(negative_systems), 1.0), \
                   f"consensus_sell_{len(negative_systems)}"
        
        # Record disagreement for learning
        self.record_disagreement(subsystem_signals, subsystem_scores)
        
        return 0.0, 0.0, "no_consensus"
    
    def record_disagreement(self, signals: Dict[str, float], scores: Dict[str, float]):
        """Record patterns when subsystems disagree"""
        disagreement_data = {
            'timestamp': datetime.now(),
            'signals': signals.copy(),
            'scores': scores.copy(),
            'disagreement_level': self.calculate_disagreement_level(signals)
        }
        self.disagreement_patterns['recent'].append(disagreement_data)
        
        # Keep only recent disagreements
        if len(self.disagreement_patterns['recent']) > 100:
            self.disagreement_patterns['recent'] = self.disagreement_patterns['recent'][-100:]
    
    def calculate_disagreement_level(self, signals: Dict[str, float]) -> float:
        """Calculate how much subsystems disagree"""
        signal_values = list(signals.values())
        return np.std(signal_values) if signal_values else 0.0
    
    def learn_from_disagreements(self, outcome: float):
        """Learn when disagreements predict market moves"""
        if not self.disagreement_patterns['recent']:
            return
            
        recent_disagreement = self.disagreement_patterns['recent'][-1]
        disagreement_level = recent_disagreement['disagreement_level']
        
        # High disagreement that preceded good outcome = market turning point
        if disagreement_level > 0.3 and outcome > 0.01:
            self.disagreement_patterns['turning_points'].append({
                'disagreement_data': recent_disagreement,
                'outcome': outcome,
                'pattern_type': 'successful_chaos'
            })

class AdvancedMarketIntelligence:
    """Main Market Intelligence Engine"""
    
    def __init__(self, db_path: str = "market_intelligence.db"):
        self.memory_db = PermanentMemoryDB(db_path)
        
        # Initialize subsystems
        self.dna_system = DNASequencingSystem(self.memory_db)
        self.micro_system = MicroPatternNetwork(self.memory_db)
        self.temporal_system = TemporalPatternArchaeologist(self.memory_db)
        self.immune_system = MarketImmuneSystem(self.memory_db)
        self.meta_director = MetaLearningDirector(self.memory_db)
        
        # Performance tracking
        self.signal_history = deque(maxlen=1000)
        self.trade_outcomes = deque(maxlen=500)
        
        log.info("Advanced Market Intelligence Engine initialized")
        log.info(f"Loaded {len(self.dna_system.dna_patterns)} DNA patterns from memory")
    
    def process_market_data(self, prices: List[float], volumes: List[float], 
                          timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Process market data through all subsystems"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Generate subsystem signals
        subsystem_signals = {}
        subsystem_scores = {}
        
        # 1. DNA Sequencing Analysis
        current_dna = self.dna_system.create_dna_sequence(prices, volumes)
        similar_patterns = self.dna_system.find_similar_patterns(current_dna)
        
        if similar_patterns:
            dna_signal = np.mean([p.success_rate - 0.5 for p in similar_patterns[:3]])  # Top 3 patterns
            dna_confidence = np.mean([p.confidence for p in similar_patterns[:3]])
        else:
            dna_signal = 0.0
            dna_confidence = 0.0
        
        subsystem_signals['dna'] = dna_signal
        subsystem_scores['dna'] = dna_confidence
        
        # 2. Micro-Pattern Analysis
        micro_pattern_id = self.micro_system.extract_micro_pattern(prices, volumes)
        if micro_pattern_id:
            micro_prob, micro_conf = self.micro_system.get_pattern_prediction(micro_pattern_id)
            micro_signal = micro_prob - 0.5  # Convert to signal (-0.5 to +0.5)
        else:
            micro_signal = 0.0
            micro_conf = 0.0
        
        subsystem_signals['micro'] = micro_signal
        subsystem_scores['micro'] = micro_conf
        
        # 3. Temporal Pattern Analysis
        temporal_strength = self.temporal_system.get_temporal_strength(timestamp, "general")
        temporal_signal = temporal_strength - 0.5
        temporal_conf = 0.5  # Base confidence for temporal
        
        subsystem_signals['temporal'] = temporal_signal
        subsystem_scores['temporal'] = temporal_conf
        
        # 4. Immune System Check
        current_pattern = {
            'prices': prices[-10:] if len(prices) >= 10 else prices,
            'volumes': volumes[-10:] if len(volumes) >= 10 else volumes,
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday()
        }
        
        if self.immune_system.is_dangerous_pattern(current_pattern):
            immune_signal = -0.5  # Strong avoid signal
        elif self.immune_system.is_beneficial_pattern(current_pattern):
            immune_signal = 0.5   # Strong positive signal
        else:
            immune_signal = 0.0   # Neutral
        
        immune_conf = self.immune_system.get_immune_strength()
        
        subsystem_signals['immune'] = immune_signal
        subsystem_scores['immune'] = immune_conf
        
        # 5. Meta-Learning Director combines signals
        final_signal, confidence = self.meta_director.get_weighted_signal(subsystem_signals)
        
        # Generate trading recommendation
        action = self.generate_action(final_signal, confidence)
        
        # Prepare result
        result = {
            'timestamp': timestamp,
            'action': action,
            'signal_strength': final_signal,
            'confidence': confidence,
            'subsystem_signals': subsystem_signals,
            'subsystem_scores': subsystem_scores,
            'current_dna': current_dna,
            'similar_patterns_count': len(similar_patterns),
            'micro_pattern_id': micro_pattern_id,
            'is_dangerous_pattern': self.immune_system.is_dangerous_pattern(current_pattern),
            'is_beneficial_pattern': self.immune_system.is_beneficial_pattern(current_pattern),
            'system_weights': self.meta_director.subsystem_weights.copy()
        }
        
        # Store signal for learning
        self.signal_history.append({
            'timestamp': timestamp,
            'signal': final_signal,
            'confidence': confidence,
            'action': action,
            'subsystem_signals': subsystem_signals.copy()
        })
        
        return result
    
    def generate_action(self, signal: float, confidence: float) -> str:
        """Generate trading action from signal"""
        min_confidence = 0.3
        min_signal_strength = 0.2
        
        if confidence < min_confidence:
            return "HOLD"
        
        if signal > min_signal_strength:
            return "BUY"
        elif signal < -min_signal_strength:
            return "SELL"
        else:
            return "HOLD"
    
    def record_trade_outcome(self, signal_timestamp: datetime, outcome: float, 
                           entry_price: float, exit_price: float):
        """Record outcome of a trade for learning"""
        # Find the corresponding signal
        matching_signal = None
        for signal_data in reversed(self.signal_history):
            if abs((signal_data['timestamp'] - signal_timestamp).total_seconds()) < 300:  # 5 min tolerance
                matching_signal = signal_data
                break
        
        if not matching_signal:
            return
        
        # Update all subsystems with outcome
        self.trade_outcomes.append({
            'timestamp': signal_timestamp,
            'outcome': outcome,
            'signal_strength': matching_signal['signal'],
            'confidence': matching_signal['confidence']
        })
        
        # Update DNA patterns
        if 'current_dna' in matching_signal:
            self.dna_system.update_pattern_outcome(matching_signal['current_dna'], outcome)
        
        # Update micro patterns
        if 'micro_pattern_id' in matching_signal and matching_signal['micro_pattern_id']:
            # Need prices and volumes for micro pattern update - would need to store these
            pass
        
        # Update temporal patterns
        pattern_type = "buy" if matching_signal['action'] == "BUY" else "sell" if matching_signal['action'] == "SELL" else "hold"
        self.temporal_system.record_temporal_event(signal_timestamp, pattern_type, outcome)
        
        # Update immune system
        current_pattern = {
            'signal_strength': matching_signal['signal'],
            'confidence': matching_signal['confidence'],
            'hour': signal_timestamp.hour,
            'day_of_week': signal_timestamp.weekday(),
            'action': matching_signal['action']
        }
        
        if outcome > 0:
            self.immune_system.add_beneficial_pattern(current_pattern, outcome)
        else:
            self.immune_system.add_pathogen(current_pattern, outcome)
        
        # Update meta-learning director
        self.meta_director.update_weights(matching_signal['subsystem_signals'], outcome)
        
        log.info(f"Recorded trade outcome: {outcome:.4f} for signal at {signal_timestamp}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        total_patterns = len(self.dna_system.dna_patterns)
        total_micro_patterns = len(self.micro_system.patterns)
        total_temporal_patterns = len(self.temporal_system.temporal_patterns)
        immune_strength = self.immune_system.get_immune_strength()
        
        # Calculate recent performance
        recent_outcomes = list(self.trade_outcomes)[-20:] if self.trade_outcomes else []
        recent_performance = np.mean([t['outcome'] for t in recent_outcomes]) if recent_outcomes else 0.0
        win_rate = sum(1 for t in recent_outcomes if t['outcome'] > 0) / len(recent_outcomes) if recent_outcomes else 0.0
        
        return {
            'total_dna_patterns': total_patterns,
            'total_micro_patterns': total_micro_patterns,
            'total_temporal_patterns': total_temporal_patterns,
            'immune_strength': immune_strength,
            'recent_performance': recent_performance,
            'win_rate': win_rate,
            'total_trades_recorded': len(self.trade_outcomes),
            'system_weights': self.meta_director.subsystem_weights,
            'memory_utilization': {
                'dna_patterns': len(self.dna_system.dna_patterns),
                'beneficial_patterns': len(self.immune_system.beneficial_patterns),
                'dangerous_patterns': len(self.immune_system.immune_patterns)
            }
        }
    
    def get_pattern_insights(self) -> Dict[str, Any]:
        """Get insights from discovered patterns"""
        insights = {
            'top_dna_patterns': [],
            'best_temporal_windows': [],
            'dangerous_patterns_count': len(self.immune_system.immune_patterns),
            'beneficial_patterns_count': len(self.immune_system.beneficial_patterns)
        }
        
        # Top DNA patterns by confidence and success rate
        top_dna = sorted(
            self.dna_system.dna_patterns.values(),
            key=lambda x: x.confidence * x.success_rate,
            reverse=True
        )[:5]
        
        for pattern in top_dna:
            insights['top_dna_patterns'].append({
                'sequence': pattern.sequence[:20] + "..." if len(pattern.sequence) > 20 else pattern.sequence,
                'success_rate': pattern.success_rate,
                'confidence': pattern.confidence,
                'occurrences': pattern.occurrences
            })
        
        # Best temporal windows
        best_temporal = []
        for pattern in self.temporal_system.temporal_patterns.values():
            if pattern.confidence > 0.5 and pattern.success_rate > 0.6:
                best_temporal.append({
                    'time': f"{pattern.hour:02d}:{pattern.minute:02d}",
                    'day_of_week': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][pattern.day_of_week],
                    'pattern_type': pattern.pattern_type,
                    'success_rate': pattern.success_rate,
                    'confidence': pattern.confidence
                })
        
        insights['best_temporal_windows'] = sorted(best_temporal, key=lambda x: x['success_rate'], reverse=True)[:5]
        
        return insights
    
    def export_knowledge_base(self, filepath: str):
        """Export entire knowledge base to JSON"""
        knowledge = {
            'export_timestamp': datetime.now().isoformat(),
            'system_status': self.get_system_status(),
            'pattern_insights': self.get_pattern_insights(),
            'dna_patterns': {
                seq: {
                    'success_rate': pattern.success_rate,
                    'occurrences': pattern.occurrences,
                    'confidence': pattern.confidence,
                    'last_seen': pattern.last_seen.isoformat()
                }
                for seq, pattern in self.dna_system.dna_patterns.items()
            },
            'system_weights': self.meta_director.subsystem_weights,
            'performance_summary': {
                'total_trades': len(self.trade_outcomes),
                'recent_performance': np.mean([t['outcome'] for t in list(self.trade_outcomes)[-20:]]) if self.trade_outcomes else 0.0
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(knowledge, f, indent=2)
        
        log.info(f"Knowledge base exported to {filepath}")
    
    def continuous_learning_cycle(self):
        """Perform continuous learning and optimization"""
        # This would run in a separate thread
        while True:
            try:
                # Analyze recent performance
                if len(self.trade_outcomes) >= 20:
                    recent_outcomes = list(self.trade_outcomes)[-20:]
                    avg_performance = np.mean([t['outcome'] for t in recent_outcomes])
                    
                    # If performance is declining, adjust weights
                    if avg_performance < -0.01:  # Losing money
                        log.warning("Performance declining, triggering weight adjustment")
                        # Could implement more sophisticated adjustment logic here
                
                # Clean up old patterns (keep memory manageable)
                self.cleanup_old_patterns()
                
                # Sleep for a while before next cycle
                time.sleep(300)  # 5 minutes
                
            except Exception as e:
                log.error(f"Continuous learning cycle error: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def cleanup_old_patterns(self):
        """Remove old, low-confidence patterns to manage memory"""
        cutoff_date = datetime.now() - timedelta(days=30)
        
        # Clean DNA patterns
        patterns_to_remove = []
        for seq, pattern in self.dna_system.dna_patterns.items():
            if (pattern.last_seen < cutoff_date and 
                pattern.confidence < 0.3 and 
                pattern.occurrences < 5):
                patterns_to_remove.append(seq)
        
        for seq in patterns_to_remove:
            del self.dna_system.dna_patterns[seq]
        
        if patterns_to_remove:
            log.info(f"Cleaned up {len(patterns_to_remove)} old DNA patterns")
    
    def start_continuous_learning(self):
        """Start the continuous learning background thread"""
        learning_thread = threading.Thread(
            target=self.continuous_learning_cycle,
            daemon=True,
            name="ContinuousLearning"
        )
        learning_thread.start()
        log.info("Continuous learning thread started")


# Integration wrapper for existing trading system
class IntelligenceIntegration:
    """Integration layer for existing trading system"""
    
    def __init__(self, existing_system):
        self.existing_system = existing_system
        self.intelligence_engine = AdvancedMarketIntelligence()
        self.intelligence_engine.start_continuous_learning()
        
        # Track signals for outcome recording
        self.pending_signals = {}
        
    def enhanced_signal_generation(self, prices: List[float], volumes: List[float]) -> Dict[str, Any]:
        """Enhanced signal generation using intelligence engine"""
        # Get traditional signal
        traditional_result = self.existing_system.process_market_data({'price_5m': prices, 'volume_5m': volumes})
        
        # Get intelligence signal
        intelligence_result = self.intelligence_engine.process_market_data(prices, volumes)
        
        # Combine signals
        traditional_confidence = traditional_result.get('confidence', 0.5)
        intelligence_confidence = intelligence_result['confidence']
        
        # Weighted combination (60% traditional, 40% intelligence for safety)
        combined_confidence = (0.6 * traditional_confidence + 0.4 * intelligence_confidence)
        
        # Use intelligence to filter traditional signals
        if intelligence_result['is_dangerous_pattern']:
            final_action = "HOLD"  # Override with hold if dangerous
            final_confidence = 0.0
        elif intelligence_result['is_beneficial_pattern']:
            final_action = traditional_result.get('action', 'HOLD')
            final_confidence = min(combined_confidence * 1.2, 1.0)  # Boost confidence
        else:
            final_action = traditional_result.get('action', 'HOLD')
            final_confidence = combined_confidence
        
        # Store signal for outcome tracking
        signal_id = f"signal_{datetime.now().timestamp()}"
        self.pending_signals[signal_id] = {
            'timestamp': datetime.now(),
            'traditional_result': traditional_result,
            'intelligence_result': intelligence_result
        }
        
        return {
            'action': final_action,
            'confidence': final_confidence,
            'signal_id': signal_id,
            'intelligence_insights': {
                'dna_patterns_found': intelligence_result['similar_patterns_count'],
                'system_weights': intelligence_result['system_weights'],
                'dangerous_pattern': intelligence_result['is_dangerous_pattern'],
                'beneficial_pattern': intelligence_result['is_beneficial_pattern']
            }
        }
    
    def record_trade_outcome(self, signal_id: str, outcome: float, entry_price: float, exit_price: float):
        """Record trade outcome for learning"""
        if signal_id in self.pending_signals:
            signal_data = self.pending_signals[signal_id]
            
            # Update intelligence engine
            self.intelligence_engine.record_trade_outcome(
                signal_data['timestamp'], 
                outcome, 
                entry_price, 
                exit_price
            )
            
            # Clean up
            del self.pending_signals[signal_id]
    
    def get_system_insights(self) -> str:
        """Get human-readable system insights"""
        status = self.intelligence_engine.get_system_status()
        insights = self.intelligence_engine.get_pattern_insights()
        
        report = f"""
=== MARKET INTELLIGENCE REPORT ===

System Status:
- DNA Patterns Stored: {status['total_dna_patterns']}
- Micro Patterns: {status['total_micro_patterns']}
- Temporal Patterns: {status['total_temporal_patterns']}
- Immune System Strength: {status['immune_strength']:.2%}
- Recent Win Rate: {status['win_rate']:.2%}
- Recent Avg Performance: {status['recent_performance']:.4f}

System Weights:
- DNA Sequencing: {status['system_weights']['dna']:.2%}
- Micro Patterns: {status['system_weights']['micro']:.2%}
- Temporal Analysis: {status['system_weights']['temporal']:.2%}
- Immune System: {status['system_weights']['immune']:.2%}

Top DNA Patterns:
"""
        
        for i, pattern in enumerate(insights['top_dna_patterns'][:3], 1):
            report += f"  {i}. {pattern['sequence']} (Success: {pattern['success_rate']:.2%}, Confidence: {pattern['confidence']:.2f})\n"
        
        report += f"\nBest Trading Windows:\n"
        for i, window in enumerate(insights['best_temporal_windows'][:3], 1):
            report += f"  {i}. {window['day_of_week']} {window['time']} - {window['pattern_type']} (Success: {window['success_rate']:.2%})\n"
        
        report += f"\nPattern Security:\n"
        report += f"  - Beneficial Patterns: {insights['beneficial_patterns_count']}\n"
        report += f"  - Dangerous Patterns: {insights['dangerous_patterns_count']}\n"
        
        return report

# Pattern interpretability and explanation system

class PatternInterpreter:
    """Make AI patterns human-readable"""
    
    def __init__(self):
        self.dna_translations = {
            'AAAA': 'Strong sustained uptrend',
            'CCCC': 'Strong sustained downtrend', 
            'ATGC': 'Classic reversal pattern',
            'ACGT': 'Volatility spike pattern',
            'TTTT': 'Weak momentum up',
            'GGGG': 'Weak momentum down'
        }
        
        self.volume_patterns = {
            'high_volume_breakout': 'Volume confirms price movement',
            'low_volume_drift': 'Weak conviction in price move',
            'volume_divergence': 'Volume contradicts price action'
        }
    
    def explain_dna_pattern(self, dna_sequence: str, success_rate: float) -> str:
        """Convert DNA sequence to human explanation"""
        explanations = []
        
        # Check for known patterns
        for pattern, meaning in self.dna_translations.items():
            if pattern in dna_sequence:
                explanations.append(f"Contains {meaning.lower()}")
        
        # Analyze sequence characteristics
        up_moves = dna_sequence.count('A') + dna_sequence.count('T')
        down_moves = dna_sequence.count('C') + dna_sequence.count('G')
        
        if up_moves > down_moves * 1.5:
            explanations.append("predominantly bullish sequence")
        elif down_moves > up_moves * 1.5:
            explanations.append("predominantly bearish sequence")
        else:
            explanations.append("balanced directional movement")
        
        # Success context
        if success_rate > 0.7:
            confidence_desc = "highly reliable"
        elif success_rate > 0.6:
            confidence_desc = "moderately reliable"
        else:
            confidence_desc = "developing pattern"
        
        base_explanation = ", ".join(explanations)
        return f"{confidence_desc.title()} pattern: {base_explanation} (success rate: {success_rate:.1%})"
    
    def explain_trade_reasoning(self, intelligence_result: Dict) -> str:
        """Explain why the AI made this decision"""
        reasoning_parts = []
        
        # DNA evidence
        if intelligence_result.get('similar_patterns_count', 0) > 0:
            reasoning_parts.append(f"Found {intelligence_result['similar_patterns_count']} similar DNA patterns in memory")
        
        # Subsystem contributions
        signals = intelligence_result.get('subsystem_signals', {})
        for system, signal in signals.items():
            if abs(signal) > 0.2:
                direction = "bullish" if signal > 0 else "bearish"
                strength = "strong" if abs(signal) > 0.4 else "moderate"
                reasoning_parts.append(f"{system} system shows {strength} {direction} signal")
        
        # Risk factors
        if intelligence_result.get('is_dangerous_pattern'):
            reasoning_parts.append("WARNING: IMMUNE SYSTEM detected dangerous pattern - avoiding")
        elif intelligence_result.get('is_beneficial_pattern'):
            reasoning_parts.append("POSITIVE: IMMUNE SYSTEM recognizes beneficial pattern")
        
        # Confidence explanation
        confidence = intelligence_result.get('confidence', 0)
        if confidence > 0.8:
            reasoning_parts.append(f"Very high confidence ({confidence:.2f}) from pattern convergence")
        elif confidence > 0.6:
            reasoning_parts.append(f"Good confidence ({confidence:.2f}) from multiple signals")
        elif confidence > 0.4:
            reasoning_parts.append(f"Moderate confidence ({confidence:.2f}) - proceeding cautiously")
        else:
            reasoning_parts.append(f"Low confidence ({confidence:.2f}) - holding position")
        
        return "; ".join(reasoning_parts)
    
    def generate_pattern_report(self, intelligence_engine) -> str:
        """Generate human-readable intelligence report"""
        status = intelligence_engine.get_system_status()
        insights = intelligence_engine.get_pattern_insights()
        
        report = f"""
MARKET INTELLIGENCE REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PATTERN LIBRARY STATUS:
- DNA Patterns: {status['total_dna_patterns']} sequences learned
- Micro Patterns: {status['total_micro_patterns']} micro-behaviors identified  
- Temporal Patterns: {status['total_temporal_patterns']} time-based patterns
- Win Rate: {status['win_rate']:.1%} (last 20 trades)

TOP DNA DISCOVERIES:
"""
        
        for i, pattern in enumerate(insights['top_dna_patterns'][:3], 1):
            explanation = self.explain_dna_pattern(pattern['sequence'], pattern['success_rate'])
            report += f"  {i}. {explanation}\n"
        
        report += f"""
OPTIMAL TRADING WINDOWS:
"""
        for i, window in enumerate(insights['best_temporal_windows'][:3], 1):
            report += f"  {i}. {window['day_of_week']} {window['time']} - {window['pattern_type']} (Success: {window['success_rate']:.1%})\n"
        
        report += f"""
IMMUNE SYSTEM STATUS:
- Beneficial Patterns: {insights['beneficial_patterns_count']} learned
- Dangerous Patterns: {insights['dangerous_patterns_count']} avoided
- System Health: {status['immune_strength']:.1%}

CURRENT SYSTEM WEIGHTS:
"""
        for system, weight in status['system_weights'].items():
            report += f"  - {system.upper()}: {weight:.1%}\n"
        
        return report


# Example usage and testing
if __name__ == "__main__":
    # Initialize the intelligence engine
    intelligence = AdvancedMarketIntelligence()
    
    # Simulate some market data
    np.random.seed(42)
    base_price = 4000.0
    prices = [base_price]
    volumes = [1000]
    
    # Generate realistic market data
    for i in range(100):
        # Random walk with slight trend
        change = np.random.normal(0, 0.001) + 0.0001  # Slight upward bias
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
        
        # Volume with some correlation to price movement
        vol_change = np.random.normal(0, 0.2) + (0.1 if change > 0 else -0.1)
        new_volume = max(100, volumes[-1] * (1 + vol_change))
        volumes.append(new_volume)
    
    print("Testing Advanced Market Intelligence Engine")
    print("=" * 50)
    
    # Process market data
    for i in range(20, len(prices), 5):  # Process every 5th point
        current_prices = prices[:i]
        current_volumes = volumes[:i]
        
        result = intelligence.process_market_data(current_prices, current_volumes)
        
        print(f"Bar {i}: Action={result['action']}, Signal={result['signal_strength']:.3f}, "
              f"Confidence={result['confidence']:.3f}")
        
        # Simulate trade outcomes (random for demo)
        if result['action'] != 'HOLD':
            outcome = np.random.normal(0.001, 0.01)  # Random outcome
            intelligence.record_trade_outcome(
                result['timestamp'], 
                outcome, 
                current_prices[-1], 
                current_prices[-1] * (1 + outcome)
            )
    
    # Print system status
    print("\n" + "=" * 50)
    print("FINAL SYSTEM STATUS")
    print("=" * 50)
    
    status = intelligence.get_system_status()
    insights = intelligence.get_pattern_insights()
    
    print(f"DNA Patterns: {status['total_dna_patterns']}")
    print(f"Micro Patterns: {status['total_micro_patterns']}")
    print(f"Win Rate: {status['win_rate']:.2%}")
    print(f"System Weights: {status['system_weights']}")
    
    # Export knowledge base
    intelligence.export_knowledge_base("intelligence_knowledge.json")
    print("\nKnowledge base exported to intelligence_knowledge.json")