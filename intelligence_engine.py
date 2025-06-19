# intelligence_engine.py

import json
import numpy as np
import logging

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from data_processor import MarketData
from subsystem_evolution import EnhancedIntelligenceOrchestrator
from market_microstructure import MarketMicrostructureEngine
from real_time_adaptation import RealTimeAdaptationEngine

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
    # Enhanced features
    microstructure_signal: float = 0.0
    regime_adjusted_signal: float = 0.0
    adaptation_quality: float = 0.0
    smart_money_flow: float = 0.0
    liquidity_depth: float = 0.5
    regime_confidence: float = 0.5

class IntelligenceEngine:
    def __init__(self, memory_file="data/intelligence_memory.json"):
        self.memory_file = memory_file
        
        # Enhanced subsystem orchestration
        self.orchestrator = EnhancedIntelligenceOrchestrator()
        
        # Market microstructure analysis
        self.microstructure_engine = MarketMicrostructureEngine()
        
        # Real-time adaptation
        self.adaptation_engine = RealTimeAdaptationEngine(model_dim=64)
        
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
        
        # Enhanced feature tracking
        self.feature_importance_history = deque(maxlen=200)
        self.regime_transition_history = deque(maxlen=50)
        self.adaptation_performance = deque(maxlen=100)
        
        self.load_patterns(self.memory_file)
        
    def bootstrap_from_historical_data(self, historical_data):
        """Enhanced bootstrap with microstructure and adaptation learning"""
        logger.info("Starting enhanced historical data bootstrap...")
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
            
            # Enhanced pattern initialization with microstructure
            self._generate_enhanced_synthetic_outcomes()
            
            # Initialize adaptation engine with historical patterns
            self._initialize_adaptation_engine()
            
            # Update bootstrap stats
            self.bootstrap_stats['total_bars_processed'] = total_bars
            self.bootstrap_stats['patterns_discovered'] = self._count_total_patterns()
            self.bootstrap_stats['bootstrap_time'] = (datetime.now() - start_time).total_seconds()
            
            self.historical_processed = True
            
            logger.info(f"Enhanced bootstrap complete: {total_bars} bars processed, "
                       f"{self.bootstrap_stats['patterns_discovered']} patterns discovered "
                       f"in {self.bootstrap_stats['bootstrap_time']:.1f}s")
            
            # Save bootstrapped patterns
            self.save_patterns(self.memory_file)
            
        except Exception as e:
            logger.error(f"Bootstrap error: {e}")
            self.historical_processed = False
    
    def _process_historical_bars(self, bars, timeframe):
        """Enhanced historical bar processing with microstructure analysis"""
        if not bars or len(bars) < 20:
            return 0
        
        prices = [bar['close'] for bar in bars]
        volumes = [bar['volume'] for bar in bars]
        timestamps = [bar['timestamp'] / 10000000 - 62135596800 for bar in bars]  # Convert to Unix timestamp
        
        processed_count = 0
        
        # Process with sliding windows
        window_size = min(50, len(bars) // 4)
        
        for i in range(window_size, len(bars)):
            window_prices = prices[i-window_size:i+1]
            window_volumes = volumes[i-window_size:i+1]
            window_timestamps = timestamps[i-window_size:i+1]
            
            # Enhanced market features for microstructure analysis
            market_features = self._extract_enhanced_market_features(
                window_prices, window_volumes, window_timestamps
            )
            
            # Process through enhanced orchestrator
            orchestrator_result = self.orchestrator.process_market_data(
                window_prices, window_volumes, market_features, window_timestamps
            )
            
            # Microstructure analysis
            microstructure_result = self.microstructure_engine.analyze_market_state(
                window_prices, window_volumes
            )
            
            # Store patterns for learning
            self._store_historical_patterns(orchestrator_result, microstructure_result, market_features)
            
            processed_count += 1
        
        return processed_count
    
    def _extract_enhanced_market_features(self, prices: List[float], volumes: List[float], 
                                        timestamps: List[float]) -> Dict:
        """Extract enhanced market features for analysis"""
        if len(prices) < 10:
            return {}
        
        # Basic features
        price_changes = np.diff(prices)
        returns = price_changes / np.array(prices[:-1])
        
        volatility = np.std(returns) if len(returns) > 1 else 0
        momentum = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 and prices[-5] != 0 else 0
        
        # Volume features
        volume_momentum = (np.mean(volumes[-3:]) - np.mean(volumes[:-3])) / np.mean(volumes[:-3]) if len(volumes) >= 6 else 0
        
        # Time features
        if timestamps:
            dt = datetime.fromtimestamp(timestamps[-1])
            time_of_day = (dt.hour * 60 + dt.minute) / 1440
        else:
            time_of_day = 0.5
        
        # Price position in range
        price_range = max(prices) - min(prices)
        price_position = (prices[-1] - min(prices)) / price_range if price_range > 0 else 0.5
        
        return {
            'volatility': volatility,
            'price_momentum': momentum,
            'volume_momentum': volume_momentum,
            'time_of_day': time_of_day,
            'price_position': price_position,
            'returns_skew': float(np.mean(returns**3)) if len(returns) > 2 else 0,
            'returns_kurtosis': float(np.mean(returns**4)) if len(returns) > 3 else 0
        }
    
    def _store_historical_patterns(self, orchestrator_result: Dict, microstructure_result: Dict, 
                                 market_features: Dict):
        """Store patterns from historical analysis"""
        # Store orchestrator patterns
        current_patterns = orchestrator_result.get('current_patterns', {})
        
        # Create synthetic outcomes for pattern initialization
        signal_strength = abs(orchestrator_result.get('overall_signal', 0))
        microstructure_strength = abs(microstructure_result.get('microstructure_signal', 0))
        
        # Combine signals for synthetic outcome
        synthetic_outcome = (signal_strength + microstructure_strength) / 2
        if orchestrator_result.get('consensus_strength', 0) > 0.7:
            synthetic_outcome *= 1.2  # Boost for high consensus
        
        # Add some noise to make it more realistic
        synthetic_outcome += np.random.normal(0, 0.1)
        synthetic_outcome = np.clip(synthetic_outcome, -1.0, 1.0)
        
        # Learn from synthetic outcome
        self.orchestrator.learn_from_outcome(synthetic_outcome, {
            'dna_sequence': current_patterns.get('dna_sequence', ''),
            'cycles_info': [],  # Would need to extract from temporal subsystem
            'market_state': market_features
        })
        
        # Store for microstructure learning
        self.microstructure_engine.learn_from_outcome(synthetic_outcome)
    
    def _generate_enhanced_synthetic_outcomes(self):
        """Generate enhanced synthetic outcomes with microstructure awareness"""
        logger.info("Generating enhanced synthetic outcomes for pattern initialization...")
        
        # Get orchestrator stats
        orchestrator_stats = self.orchestrator.get_comprehensive_stats()
        
        # Initialize patterns based on market structure knowledge
        dna_stats = orchestrator_stats.get('dna_evolution', {})
        if dna_stats.get('total_sequences', 0) > 0:
            logger.info(f"Initialized {dna_stats['total_sequences']} DNA sequences")
        
        immune_stats = orchestrator_stats.get('immune_system', {})
        if immune_stats.get('total_antibodies', 0) > 0:
            logger.info(f"Initialized {immune_stats['total_antibodies']} immune antibodies")
    
    def _initialize_adaptation_engine(self):
        """Initialize the real-time adaptation engine with historical context"""
        logger.info("Initializing real-time adaptation engine...")
        
        # Process some initialization events
        for i in range(10):
            self.adaptation_engine.process_market_event(
                'initialization',
                {'pattern_count': i * 10, 'bootstrap_complete': True},
                urgency=0.3
            )
    
    def _count_total_patterns(self) -> int:
        """Count total patterns across all subsystems"""
        orchestrator_stats = self.orchestrator.get_comprehensive_stats()
        
        total = 0
        dna_stats = orchestrator_stats.get('dna_evolution', {})
        total += dna_stats.get('total_sequences', 0)
        
        immune_stats = orchestrator_stats.get('immune_system', {})
        total += immune_stats.get('total_antibodies', 0)
        
        total += orchestrator_stats.get('temporal_cycles', 0)
        total += len(self.patterns)
        
        return total
    
    def extract_features(self, data: MarketData) -> Features:
        """Enhanced feature extraction with microstructure and adaptation"""
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
        
        # Enhanced market features
        market_features = {
            'volatility': volatility,
            'price_momentum': price_momentum,
            'volume_momentum': volume_momentum,
            'time_of_day': time_of_day,
            'price_position': price_position,
            'account_balance': data.account_balance,
            'margin_utilization': data.margin_utilization
        }
        
        # Process through enhanced orchestrator
        timestamps = [data.timestamp - i * 60 for i in range(len(prices))][::-1]  # Approximate timestamps
        orchestrator_result = self.orchestrator.process_market_data(
            data.prices_1m, data.volumes_1m, market_features, timestamps
        )
        
        # Microstructure analysis
        microstructure_result = self.microstructure_engine.analyze_market_state(
            data.prices_1m, data.volumes_1m
        )
        
        # Real-time adaptation
        adaptation_context = {
            'volatility': volatility,
            'trend_strength': abs(price_momentum),
            'volume_regime': min(1.0, volume_momentum + 0.5),
            'time_of_day': time_of_day
        }
        
        # Create feature tensor for adaptation
        feature_tensor = self._create_feature_tensor(market_features, orchestrator_result, microstructure_result)
        adaptation_decision = self.adaptation_engine.get_adaptation_decision(feature_tensor, adaptation_context)
        
        # Basic pattern score
        pattern_score = self._recognize_patterns(prices, volumes)
        
        # Enhanced confidence calculation
        signal_strength = abs(orchestrator_result['overall_signal'])
        momentum_strength = abs(price_momentum) + abs(volume_momentum)
        microstructure_strength = abs(microstructure_result.get('microstructure_signal', 0))
        adaptation_quality = adaptation_decision.get('adaptation_quality', 0.5)
        
        # Boost confidence if we have historical patterns and good adaptation
        pattern_boost = 0.15 if self.historical_processed else 0.0
        adaptation_boost = adaptation_quality * 0.1
        
        confidence = min(1.0, signal_strength + momentum_strength * 0.3 + 
                        microstructure_strength * 0.2 + pattern_boost + adaptation_boost)
        
        # Extract microstructure features
        microstructure_features = microstructure_result.get('order_flow', {})
        regime_state = microstructure_result.get('regime_state', {})
        
        return Features(
            price_momentum=price_momentum,
            volume_momentum=volume_momentum,
            price_position=price_position,
            volatility=volatility,
            time_of_day=time_of_day,
            pattern_score=pattern_score,
            confidence=confidence,
            dna_signal=orchestrator_result['dna_signal'],
            micro_signal=orchestrator_result['temporal_signal'],  # Note: this maps to temporal for compatibility
            temporal_signal=orchestrator_result['temporal_signal'],
            immune_signal=orchestrator_result['immune_signal'],
            overall_signal=orchestrator_result['overall_signal'],
            # Enhanced features
            microstructure_signal=microstructure_result.get('microstructure_signal', 0.0),
            regime_adjusted_signal=microstructure_result.get('regime_adjusted_signal', 0.0),
            adaptation_quality=adaptation_quality,
            smart_money_flow=microstructure_features.get('smart_money_flow', 0.0),
            liquidity_depth=microstructure_features.get('liquidity_depth', 0.5),
            regime_confidence=regime_state.get('confidence', 0.5)
        )
    
    def _create_feature_tensor(self, market_features: Dict, orchestrator_result: Dict, 
                             microstructure_result: Dict):
        """Create feature tensor for adaptation engine"""
        import torch
        
        # Combine all features into a tensor
        features = [
            market_features.get('volatility', 0),
            market_features.get('price_momentum', 0),
            market_features.get('volume_momentum', 0),
            market_features.get('time_of_day', 0.5),
            market_features.get('price_position', 0.5),
            orchestrator_result.get('dna_signal', 0),
            orchestrator_result.get('temporal_signal', 0),
            orchestrator_result.get('immune_signal', 0),
            orchestrator_result.get('overall_signal', 0),
            orchestrator_result.get('consensus_strength', 0),
            microstructure_result.get('microstructure_signal', 0),
            microstructure_result.get('regime_adjusted_signal', 0)
        ]
        
        # Pad to 64 dimensions
        while len(features) < 64:
            features.append(0.0)
        
        return torch.tensor(features[:64], dtype=torch.float32)
    
    def _recognize_patterns(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        """Enhanced pattern recognition"""
        if len(prices) < 10:
            return 0.0
            
        trend = (prices[-1] - prices[-10]) / prices[-10] if prices[-10] != 0 else 0
        vol_trend = (volumes[-1] - np.mean(volumes[-5:])) / np.mean(volumes[-5:]) if np.mean(volumes[-5:]) > 0 else 0
        
        # Enhanced pattern strength calculation
        if (trend > 0 and vol_trend > 0) or (trend < 0 and vol_trend < 0):
            pattern_strength = abs(trend) * (1 + abs(vol_trend))
        else:
            pattern_strength = abs(trend) * 0.5
        
        # Add volatility clustering detection
        recent_volatility = np.std(prices[-5:]) if len(prices) >= 5 else 0
        historical_volatility = np.std(prices[:-5]) if len(prices) >= 10 else recent_volatility
        
        if historical_volatility > 0:
            volatility_ratio = recent_volatility / historical_volatility
            if volatility_ratio > 1.5 or volatility_ratio < 0.5:
                pattern_strength *= 1.2  # Boost for volatility clustering
            
        return np.tanh(pattern_strength * 10)
    
    def learn_from_outcome(self, trade):
        """Enhanced learning with microstructure and adaptation feedback"""
        outcome = trade.pnl / abs(trade.entry_price * 0.01) if trade.entry_price != 0 else 0
        
        # Enhanced orchestrator learning
        if hasattr(trade, 'intelligence_data'):
            self.orchestrator.learn_from_outcome(outcome, {
                'dna_sequence': trade.intelligence_data.get('current_patterns', {}).get('dna_sequence', ''),
                'cycles_info': [],  # Would extract from temporal data
                'market_state': getattr(trade, 'market_features', {})
            })
        
        # Microstructure learning
        self.microstructure_engine.learn_from_outcome(outcome)
        
        # Real-time adaptation learning
        if hasattr(trade, 'market_features'):
            adaptation_context = {
                'volatility': trade.market_features.get('volatility', 0.02),
                'trend_strength': abs(trade.market_features.get('price_momentum', 0)),
                'volume_regime': trade.market_features.get('volume_momentum', 0) + 0.5,
                'time_of_day': trade.market_features.get('time_of_day', 0.5),
                'predicted_confidence': getattr(trade, 'confidence', 0.5)
            }
            self.adaptation_engine.update_from_outcome(outcome, adaptation_context)
        
        # Process real-time adaptation event
        urgency = min(1.0, abs(outcome) * 2.0)  # Higher urgency for larger outcomes
        self.adaptation_engine.process_market_event(
            'trade_outcome',
            {'pnl': outcome, 'trade_data': trade},
            urgency=urgency
        )
        
        # Legacy pattern learning for compatibility
        if hasattr(trade, 'features'):
            pattern_id = self._create_pattern_id(trade.features)
            self.patterns[pattern_id].append(outcome)
            
            if len(self.patterns[pattern_id]) > 20:
                self.patterns[pattern_id] = self.patterns[pattern_id][-20:]
        
        self.recent_outcomes.append(outcome)
        
        # Track adaptation performance
        self.adaptation_performance.append(outcome)
    
    def _create_pattern_id(self, features: Features) -> str:
        """Create pattern ID for legacy compatibility"""
        momentum_bucket = int(features.price_momentum * 5) + 5
        position_bucket = int(features.price_position * 4)
        vol_bucket = int(features.volatility * 100) // 10
        
        return f"p{momentum_bucket}_{position_bucket}_{vol_bucket}"
    
    def _default_features(self) -> Features:
        """Default features with enhanced fields"""
        return Features(
            price_momentum=0, volume_momentum=0, price_position=0.5, volatility=0, 
            time_of_day=0.5, pattern_score=0, confidence=0, dna_signal=0, 
            micro_signal=0, temporal_signal=0, immune_signal=0, overall_signal=0,
            microstructure_signal=0.0, regime_adjusted_signal=0.0, adaptation_quality=0.5,
            smart_money_flow=0.0, liquidity_depth=0.5, regime_confidence=0.5
        )
    
    def save_patterns(self, filepath: str):
        """Enhanced pattern saving with all subsystems"""
        # Get comprehensive stats
        orchestrator_stats = self.orchestrator.get_comprehensive_stats()
        microstructure_features = self.microstructure_engine.get_microstructure_features()
        adaptation_stats = self.adaptation_engine.get_comprehensive_stats()
        
        data = {
            # Legacy compatibility
            'patterns': dict(self.patterns),
            'recent_outcomes': list(self.recent_outcomes),
            'historical_processed': self.historical_processed,
            'bootstrap_stats': self.bootstrap_stats,
            
            # Enhanced subsystem data
            'orchestrator_stats': orchestrator_stats,
            'microstructure_features': microstructure_features,
            'adaptation_stats': adaptation_stats,
            'feature_importance_history': list(self.feature_importance_history),
            'adaptation_performance': list(self.adaptation_performance),
            
            'saved_at': datetime.now().isoformat()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving patterns: {e}")
    
    def load_patterns(self, filepath: str):
        """Enhanced pattern loading with all subsystems"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Legacy compatibility
            self.patterns = defaultdict(list, data.get('patterns', {}))
            self.recent_outcomes = deque(data.get('recent_outcomes', []), maxlen=100)
            self.historical_processed = data.get('historical_processed', False)
            self.bootstrap_stats = data.get('bootstrap_stats', {
                'total_bars_processed': 0,
                'patterns_discovered': 0,
                'bootstrap_time': 0
            })
            
            # Enhanced subsystem data
            self.feature_importance_history = deque(
                data.get('feature_importance_history', []), maxlen=200
            )
            self.adaptation_performance = deque(
                data.get('adaptation_performance', []), maxlen=100
            )
            
            if self.historical_processed:
                orchestrator_stats = data.get('orchestrator_stats', {})
                logger.info(f"Loaded enhanced intelligence patterns: "
                           f"DNA={orchestrator_stats.get('dna_evolution', {}).get('total_sequences', 0)}, "
                           f"Immune={orchestrator_stats.get('immune_system', {}).get('total_antibodies', 0)}, "
                           f"Temporal={orchestrator_stats.get('temporal_cycles', 0)}")
            
        except FileNotFoundError:
            logger.info("No existing patterns found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading patterns: {e}")
    
    def get_stats(self) -> Dict:
        """Enhanced statistics with all subsystems"""
        # Get comprehensive stats from all subsystems
        orchestrator_stats = self.orchestrator.get_comprehensive_stats()
        adaptation_stats = self.adaptation_engine.get_comprehensive_stats()
        
        # Legacy stats
        legacy_stats = {
            'total_patterns': len(self.patterns),
            'recent_performance': np.mean(self.recent_outcomes) if self.recent_outcomes else 0,
            'pattern_count': sum(len(outcomes) for outcomes in self.patterns.values()),
            'historical_processed': self.historical_processed,
            'bootstrap_stats': self.bootstrap_stats
        }
        
        # Enhanced stats
        enhanced_stats = {
            'orchestrator': orchestrator_stats,
            'adaptation': adaptation_stats,
            'adaptation_performance': {
                'mean': np.mean(self.adaptation_performance) if self.adaptation_performance else 0,
                'std': np.std(self.adaptation_performance) if self.adaptation_performance else 0,
                'count': len(self.adaptation_performance)
            }
        }
        
        # Combine all stats
        return {**legacy_stats, **enhanced_stats}