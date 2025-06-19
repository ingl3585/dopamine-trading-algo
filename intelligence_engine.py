# intelligence_engine.py

import json
import numpy as np
import logging

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

from data_processor import MarketData

logger = logging.getLogger(__name__)

@dataclass
class Features:
    price_momentum: float
    volume_momentum: float
    price_position: float
    volatility: float
    time_of_day: float
    pattern_score: float
    confidence: float
    dna_signal: float
    micro_signal: float
    temporal_signal: float
    immune_signal: float
    overall_signal: float

class IntelligenceEngine:
    def __init__(self, memory_file="data/intelligence_memory.json"):
        self.memory_file = memory_file
        
        # Subsystem memories
        self.dna_patterns = {}  # sequence -> success_rate
        self.micro_patterns = {}  # pattern_id -> success_rate
        self.temporal_patterns = {}  # time_key -> success_rate
        self.immune_patterns = set()  # patterns to avoid
        
        # Legacy pattern storage for compatibility
        self.patterns = defaultdict(list)
        self.recent_outcomes = deque(maxlen=100)
        
        # Historical data processing state
        self.historical_processed = False
        self.bootstrap_stats = {
            'total_bars_processed': 0,
            'patterns_discovered': 0,
            'bootstrap_time': 0
        }
        
        self.load_patterns(self.memory_file)
        
    def bootstrap_from_historical_data(self, historical_data):
        """Bootstrap pattern learning from historical data"""
        logger.info("Starting historical data bootstrap...")
        start_time = datetime.now()
        
        try:
            # Process each timeframe
            total_bars = 0
            
            for timeframe in ['15m', '5m', '1m']:
                bars_key = f'bars_{timeframe}'
                if bars_key in historical_data:
                    bars = historical_data[bars_key]
                    processed = self._process_historical_bars(bars, timeframe)
                    total_bars += processed
                    logger.info(f"Processed {processed} {timeframe} bars")
            
            # Generate synthetic outcomes for pattern initialization
            self._generate_synthetic_outcomes()
            
            # Update bootstrap stats
            self.bootstrap_stats['total_bars_processed'] = total_bars
            self.bootstrap_stats['patterns_discovered'] = (
                len(self.dna_patterns) + 
                len(self.micro_patterns) + 
                len(self.temporal_patterns)
            )
            self.bootstrap_stats['bootstrap_time'] = (datetime.now() - start_time).total_seconds()
            
            self.historical_processed = True
            
            logger.info(f"Bootstrap complete: {total_bars} bars processed, "
                       f"{self.bootstrap_stats['patterns_discovered']} patterns discovered "
                       f"in {self.bootstrap_stats['bootstrap_time']:.1f}s")
            
            # Save bootstrapped patterns
            self.save_patterns(self.memory_file)
            
        except Exception as e:
            logger.error(f"Bootstrap error: {e}")
            self.historical_processed = False
    
    def _process_historical_bars(self, bars, timeframe):
        """Process historical bars for pattern learning"""
        if not bars or len(bars) < 20:
            return 0
        
        prices = [bar['close'] for bar in bars]
        volumes = [bar['volume'] for bar in bars]
        timestamps = [datetime.fromtimestamp(bar['timestamp'] / 10000000 - 62135596800) for bar in bars]
        
        processed_count = 0
        
        # Process with sliding windows
        window_size = min(20, len(bars) // 4)
        
        for i in range(window_size, len(bars)):
            window_prices = prices[i-window_size:i+1]
            window_volumes = volumes[i-window_size:i+1]
            window_timestamp = timestamps[i]
            
            # DNA pattern learning
            if len(window_prices) >= 15:
                dna_seq = self._create_dna_sequence(window_prices[-15:], window_volumes[-15:])
                if dna_seq and dna_seq not in self.dna_patterns:
                    # Initialize with neutral success rate
                    self.dna_patterns[dna_seq] = 0.5
            
            # Micro pattern learning
            if len(window_prices) >= 10:
                micro_pattern = self._create_micro_pattern(window_prices[-10:], window_volumes[-10:])
                if micro_pattern and micro_pattern not in self.micro_patterns:
                    self.micro_patterns[micro_pattern] = 0.5
            
            # Temporal pattern learning
            temporal_key = self._create_temporal_key(window_timestamp)
            if temporal_key not in self.temporal_patterns:
                self.temporal_patterns[temporal_key] = 0.5
            
            processed_count += 1
        
        return processed_count
    
    def _generate_synthetic_outcomes(self):
        """Generate synthetic outcomes based on market structure for initial learning"""
        
        # For DNA patterns - favor patterns with moderate consistency
        for seq, _ in list(self.dna_patterns.items()):
            # Analyze sequence characteristics
            up_moves = seq.count('A') + seq.count('T')
            down_moves = seq.count('C') + seq.count('G')
            total_moves = len(seq)
            
            if total_moves > 0:
                trend_strength = abs(up_moves - down_moves) / total_moves
                # Moderate trends tend to be more reliable
                if 0.3 < trend_strength < 0.7:
                    self.dna_patterns[seq] = 0.5 + (trend_strength - 0.5) * 0.2
                else:
                    self.dna_patterns[seq] = 0.45  # Slightly pessimistic for extreme patterns
        
        # For micro patterns - favor momentum patterns
        for pattern_id, _ in list(self.micro_patterns.items()):
            try:
                # Extract momentum from pattern ID
                parts = pattern_id.split('_')
                if len(parts) >= 3:
                    momentum = int(parts[1])
                    volatility = int(parts[2])
                    
                    # Moderate momentum with low volatility is often good
                    if abs(momentum) > 5 and volatility < 50:
                        self.micro_patterns[pattern_id] = 0.55
                    else:
                        self.micro_patterns[pattern_id] = 0.48
            except:
                self.micro_patterns[pattern_id] = 0.5
        
        # For temporal patterns - use market session knowledge
        for time_key, _ in list(self.temporal_patterns.items()):
            try:
                parts = time_key.split('_')
                if len(parts) >= 3:
                    day_of_week = int(parts[0])
                    hour = int(parts[1])
                    
                    # Market open/close times tend to be more volatile
                    if hour in [9, 10, 15, 16]:  # Market open/close hours
                        self.temporal_patterns[time_key] = 0.52
                    elif day_of_week in [0, 4]:  # Monday/Friday
                        self.temporal_patterns[time_key] = 0.48
                    else:
                        self.temporal_patterns[time_key] = 0.5
            except:
                self.temporal_patterns[time_key] = 0.5
        
        logger.info(f"Generated synthetic outcomes for {len(self.dna_patterns)} DNA, "
                   f"{len(self.micro_patterns)} micro, {len(self.temporal_patterns)} temporal patterns")
    
    def _create_micro_pattern(self, prices, volumes):
        """Create micro pattern from recent price/volume data"""
        if len(prices) < 10:
            return None
        
        price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        avg_change = np.mean(price_changes)
        volatility = np.std(price_changes)
        
        # Add volume characteristics
        volume_trend = (volumes[-1] - volumes[0]) / volumes[0] if volumes[0] > 0 else 0
        
        return f"micro_{int(avg_change * 1000)}_{int(volatility * 1000)}_{int(volume_trend * 100)}"
    
    def _create_temporal_key(self, timestamp):
        """Create temporal key from timestamp"""
        hour = timestamp.hour
        minute = timestamp.minute // 15 * 15  # Round to 15-min intervals
        day_of_week = timestamp.weekday()
        
        return f"{day_of_week}_{hour}_{minute}"
    
    def extract_features(self, data: MarketData) -> Features:
        if len(data.prices_1m) < 20:
            return self._default_features()
            
        prices = np.array(data.prices_1m[-20:])
        volumes = np.array(data.volumes_1m[-20:]) if len(data.volumes_1m) >= 20 else np.ones(20)
        
        # Basic features
        short_ma = np.mean(prices[-5:])
        long_ma = np.mean(prices)
        price_momentum = (short_ma - long_ma) / long_ma
        
        recent_vol = np.mean(volumes[-5:])
        avg_vol = np.mean(volumes)
        volume_momentum = (recent_vol - avg_vol) / avg_vol if avg_vol > 0 else 0
        
        high = np.max(prices)
        low = np.min(prices)
        price_position = (prices[-1] - low) / (high - low) if high > low else 0.5
        
        volatility = np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0
        
        now = datetime.fromtimestamp(data.timestamp)
        time_of_day = (now.hour * 60 + now.minute) / 1440
        
        # Process through subsystems
        subsystem_result = self._process_subsystems(data.prices_1m, data.volumes_1m, now)
        
        # Basic pattern score
        pattern_score = self._recognize_patterns(prices, volumes)
        
        # Overall confidence based on signal strength and consistency
        signal_strength = abs(subsystem_result['overall_signal'])
        momentum_strength = abs(price_momentum) + abs(volume_momentum)
        
        # Boost confidence if we have historical patterns
        pattern_boost = 0.1 if self.historical_processed else 0.0
        confidence = min(1.0, signal_strength + momentum_strength * 0.5 + pattern_boost)
        
        return Features(
            price_momentum=price_momentum,
            volume_momentum=volume_momentum,
            price_position=price_position,
            volatility=volatility,
            time_of_day=time_of_day,
            pattern_score=pattern_score,
            confidence=confidence,
            dna_signal=subsystem_result['subsystem_signals']['dna'],
            micro_signal=subsystem_result['subsystem_signals']['micro'],
            temporal_signal=subsystem_result['subsystem_signals']['temporal'],
            immune_signal=subsystem_result['subsystem_signals']['immune'],
            overall_signal=subsystem_result['overall_signal']
        )
    
    def _process_subsystems(self, prices: List[float], volumes: List[float], 
                          timestamp: datetime) -> Dict:
        """Process market data through all subsystems"""
        if len(prices) < 10:
            return self._empty_subsystem_result()
        
        # Extract features from each subsystem
        dna_signal = self._process_dna(prices, volumes)
        micro_signal = self._process_micro(prices, volumes)
        temporal_signal = self._process_temporal(timestamp)
        immune_signal = self._process_immune(prices, volumes, timestamp)
        
        # Combine signals
        subsystem_signals = {
            'dna': dna_signal,
            'micro': micro_signal, 
            'temporal': temporal_signal,
            'immune': immune_signal
        }
        
        # Calculate raw overall signal
        signals = list(subsystem_signals.values())
        overall_signal = sum(signals) / len(signals) if signals else 0.0
        
        # Boost signal strength if we have good historical patterns
        if self.historical_processed and abs(overall_signal) > 0:
            pattern_count = len(self.dna_patterns) + len(self.micro_patterns) + len(self.temporal_patterns)
            if pattern_count > 100:  # Good pattern base
                overall_signal *= 1.2  # 20% boost
        
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
        
        sequence = self._create_dna_sequence(prices[-15:], volumes[-15:])
        
        if not sequence:
            return 0.0
        
        # Find best matching pattern
        best_match = None
        best_similarity = 0
        
        for stored_seq in self.dna_patterns:
            similarity = self._sequence_similarity(sequence, stored_seq)
            if similarity > best_similarity and similarity > 0.7:
                best_similarity = similarity
                best_match = stored_seq
        
        if best_match:
            success_rate = self.dna_patterns[best_match]
            signal = (success_rate - 0.5) * 2  # Convert to -1 to +1 range
            # Boost signal based on similarity
            return signal * best_similarity
        
        return 0.0
    
    def _process_micro(self, prices: List[float], volumes: List[float]) -> float:
        """Micro subsystem - short-term patterns"""
        if len(prices) < 10:
            return 0.0
        
        pattern_id = self._create_micro_pattern(prices[-10:], volumes[-10:])
        
        if pattern_id and pattern_id in self.micro_patterns:
            success_rate = self.micro_patterns[pattern_id]
            return (success_rate - 0.5) * 2
        
        return 0.0
    
    def _process_temporal(self, timestamp: datetime) -> float:
        """Temporal subsystem - time-based patterns"""
        time_key = self._create_temporal_key(timestamp)
        
        if time_key in self.temporal_patterns:
            success_rate = self.temporal_patterns[time_key]
            return (success_rate - 0.5) * 2
        
        return 0.0
    
    def _process_immune(self, prices: List[float], volumes: List[float], 
                       timestamp: datetime) -> float:
        """Immune subsystem - avoid bad patterns"""
        current_pattern = self._create_immune_pattern(prices, volumes, timestamp)
        
        if current_pattern in self.immune_patterns:
            return -0.8  # Strong avoid signal
        
        return 0.0
    
    def _create_dna_sequence(self, prices: List[float], volumes: List[float]) -> str:
        """Create DNA sequence from price/volume data"""
        if len(prices) < 2 or len(volumes) < 2:
            return ""
        
        sequence = ""
        
        for i in range(1, len(prices)):
            price_change = (prices[i] - prices[i-1]) / prices[i-1] if prices[i-1] != 0 else 0
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
        if min_len == 0:
            return 0.0
        
        matches = sum(1 for i in range(min_len) if seq1[i] == seq2[i])
        return matches / max(len(seq1), len(seq2))
    
    def _create_immune_pattern(self, prices: List[float], volumes: List[float], 
                              timestamp: datetime) -> str:
        """Create pattern for immune system"""
        if len(prices) < 5:
            return ""
        
        recent_change = (prices[-1] - prices[-5]) / prices[-5] if prices[-5] != 0 else 0
        volatility = np.std(prices[-10:]) if len(prices) >= 10 else 0
        hour = timestamp.hour
        
        return f"immune_{int(recent_change * 1000)}_{int(volatility * 1000)}_{hour}"
    
    def _get_current_patterns(self, prices: List[float], volumes: List[float], 
                             timestamp: datetime) -> Dict:
        """Get current patterns for debugging"""
        return {
            'dna_sequence': self._create_dna_sequence(prices[-15:], volumes[-15:]) if len(prices) >= 15 else "",
            'micro_pattern': self._create_micro_pattern(prices[-10:], volumes[-10:]) if len(prices) >= 10 else "",
            'temporal_key': self._create_temporal_key(timestamp),
            'immune_pattern': self._create_immune_pattern(prices, volumes, timestamp)
        }
    
    def _recognize_patterns(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        if len(prices) < 10:
            return 0.0
            
        trend = (prices[-1] - prices[-10]) / prices[-10] if prices[-10] != 0 else 0
        vol_trend = (volumes[-1] - np.mean(volumes[-5:])) / np.mean(volumes[-5:]) if np.mean(volumes[-5:]) > 0 else 0
        
        if (trend > 0 and vol_trend > 0) or (trend < 0 and vol_trend < 0):
            pattern_strength = abs(trend) * (1 + abs(vol_trend))
        else:
            pattern_strength = abs(trend) * 0.5
            
        return np.tanh(pattern_strength * 10)
    
    def learn_from_outcome(self, trade):
        outcome = trade.pnl / abs(trade.entry_price * 0.01) if trade.entry_price != 0 else 0
        
        # Learn subsystem patterns if available
        if hasattr(trade, 'intelligence_data'):
            self._learn_subsystem_patterns(trade.intelligence_data, outcome)
        
        # Legacy pattern learning
        if hasattr(trade, 'features'):
            pattern_id = self._create_pattern_id(trade.features)
            self.patterns[pattern_id].append(outcome)
            
            if len(self.patterns[pattern_id]) > 20:
                self.patterns[pattern_id] = self.patterns[pattern_id][-20:]
        
        self.recent_outcomes.append(outcome)
    
    def _learn_subsystem_patterns(self, intelligence_data: Dict, outcome: float):
        """Learn from trading outcome for subsystems"""
        current_patterns = intelligence_data.get('current_patterns', {})
        
        # Update DNA patterns
        dna_seq = current_patterns.get('dna_sequence', '')
        if dna_seq:
            if dna_seq not in self.dna_patterns:
                self.dna_patterns[dna_seq] = 0.5
            
            current_rate = self.dna_patterns[dna_seq]
            learning_rate = 0.05 if self.historical_processed else 0.1  # Slower learning if bootstrapped
            
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
            learning_rate = 0.05 if self.historical_processed else 0.1
            
            if outcome > 0:
                self.micro_patterns[micro_pattern] = min(0.95, current_rate + learning_rate)
            else:
                self.micro_patterns[micro_pattern] = max(0.05, current_rate - learning_rate)
        
        # Update temporal patterns
        temporal_key = current_patterns.get('temporal_key', '')
        if temporal_key:
            if temporal_key not in self.temporal_patterns:
                self.temporal_patterns[temporal_key] = 0.5
            
            current_rate = self.temporal_patterns[temporal_key]
            learning_rate = 0.02 if self.historical_processed else 0.05
            
            if outcome > 0:
                self.temporal_patterns[temporal_key] = min(0.95, current_rate + learning_rate)
            else:
                self.temporal_patterns[temporal_key] = max(0.05, current_rate - learning_rate)
        
        # Update immune system
        if outcome < -50:  # Bad outcome
            immune_pattern = current_patterns.get('immune_pattern', '')
            if immune_pattern:
                self.immune_patterns.add(immune_pattern)
        
        # Periodic cleanup
        if len(self.dna_patterns) > 1000:
            self._cleanup_patterns()
    
    def _cleanup_patterns(self):
        """Remove old or poor-performing patterns"""
        max_patterns = 800
        
        if len(self.dna_patterns) > max_patterns:
            # Keep patterns that are far from neutral (0.5)
            sorted_patterns = sorted(self.dna_patterns.items(), key=lambda x: abs(x[1] - 0.5), reverse=True)
            self.dna_patterns = dict(sorted_patterns[:max_patterns])
        
        if len(self.micro_patterns) > max_patterns:
            sorted_patterns = sorted(self.micro_patterns.items(), key=lambda x: abs(x[1] - 0.5), reverse=True)
            self.micro_patterns = dict(sorted_patterns[:max_patterns])
        
        if len(self.temporal_patterns) > max_patterns:
            sorted_patterns = sorted(self.temporal_patterns.items(), key=lambda x: abs(x[1] - 0.5), reverse=True)
            self.temporal_patterns = dict(sorted_patterns[:max_patterns])
        
        # Keep most recent immune patterns
        if len(self.immune_patterns) > 200:
            self.immune_patterns = set(list(self.immune_patterns)[-200:])
    
    def _empty_subsystem_result(self):
        return {
            'overall_signal': 0.0,
            'subsystem_signals': {'dna': 0.0, 'micro': 0.0, 'temporal': 0.0, 'immune': 0.0},
            'confidence': 0.0,
            'current_patterns': {}
        }
    
    def _create_pattern_id(self, features: Features) -> str:
        momentum_bucket = int(features.price_momentum * 5) + 5
        position_bucket = int(features.price_position * 4)
        vol_bucket = int(features.volatility * 100) // 10
        
        return f"p{momentum_bucket}_{position_bucket}_{vol_bucket}"
    
    def _default_features(self) -> Features:
        return Features(0, 0, 0.5, 0, 0.5, 0, 0, 0, 0, 0, 0, 0)
    
    def save_patterns(self, filepath: str):
        data = {
            'dna_patterns': self.dna_patterns,
            'micro_patterns': self.micro_patterns,
            'temporal_patterns': self.temporal_patterns,
            'immune_patterns': list(self.immune_patterns),
            'patterns': dict(self.patterns),
            'recent_outcomes': list(self.recent_outcomes),
            'historical_processed': self.historical_processed,
            'bootstrap_stats': self.bootstrap_stats,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    def load_patterns(self, filepath: str):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.dna_patterns = data.get('dna_patterns', {})
            self.micro_patterns = data.get('micro_patterns', {})
            self.temporal_patterns = data.get('temporal_patterns', {})
            self.immune_patterns = set(data.get('immune_patterns', []))
            self.patterns = defaultdict(list, data.get('patterns', {}))
            self.recent_outcomes = deque(data.get('recent_outcomes', []), maxlen=100)
            self.historical_processed = data.get('historical_processed', False)
            self.bootstrap_stats = data.get('bootstrap_stats', {
                'total_bars_processed': 0,
                'patterns_discovered': 0,
                'bootstrap_time': 0
            })
            
            if self.historical_processed:
                logger.info(f"Loaded patterns: DNA={len(self.dna_patterns)}, "
                           f"Micro={len(self.micro_patterns)}, "
                           f"Temporal={len(self.temporal_patterns)}")
            
        except FileNotFoundError:
            logger.info("No existing patterns found, starting fresh")
    
    def get_stats(self) -> Dict:
        stats = {
            'dna_patterns': len(self.dna_patterns),
            'micro_patterns': len(self.micro_patterns),
            'temporal_patterns': len(self.temporal_patterns),
            'immune_patterns': len(self.immune_patterns),
            'total_patterns': len(self.patterns),
            'recent_performance': np.mean(self.recent_outcomes) if self.recent_outcomes else 0,
            'pattern_count': sum(len(outcomes) for outcomes in self.patterns.values()),
            'historical_processed': self.historical_processed,
            'bootstrap_stats': self.bootstrap_stats
        }
        
        return stats