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
# logger.setLevel(logging.DEBUG)  # Disable debug logging for cleaner output

@dataclass
class Features:
    # Core market features
    price_momentum: float
    volume_momentum: float
    price_position: float
    volatility: float
    time_of_day: float
    pattern_score: float
    confidence: float
    
    # Subsystem signals
    dna_signal: float
    micro_signal: float
    temporal_signal: float
    immune_signal: float
    microstructure_signal: float
    overall_signal: float
    
    # Enhanced context
    regime_adjusted_signal: float = 0.0
    adaptation_quality: float = 0.0
    smart_money_flow: float = 0.0
    liquidity_depth: float = 0.5
    regime_confidence: float = 0.5

class IntelligenceEngine:
    def __init__(self, memory_file="data/intelligence_memory.json"):
        self.memory_file = memory_file
        
        # Core subsystem orchestration
        self.orchestrator = EnhancedIntelligenceOrchestrator()
        
        # Market microstructure analysis
        self.microstructure_engine = MarketMicrostructureEngine()
        
        # Real-time adaptation
        self.adaptation_engine = RealTimeAdaptationEngine(model_dim=64)
        
        # Pattern storage
        self.patterns = defaultdict(list)
        self.recent_outcomes = deque(maxlen=100)
        
        # Bootstrap state
        self.historical_processed = False
        self.bootstrap_stats = {
            'total_bars_processed': 0,
            'patterns_discovered': 0,
            'bootstrap_time': 0
        }
        
        self.load_patterns(self.memory_file)
        
    def bootstrap_from_historical_data(self, historical_data):
        """Bootstrap subsystems with historical data"""
        logger.info("Starting historical data bootstrap...")
        start_time = datetime.now()
        
        try:
            # Debug the historical data structure
            logger.info(f"Historical data type: {type(historical_data)}")
            logger.info(f"Historical data keys: {list(historical_data.keys()) if isinstance(historical_data, dict) else 'Not a dict'}")
            
            total_bars = 0
            
            for timeframe in ['15m', '5m', '1m']:
                bars_key = f'bars_{timeframe}'
                if bars_key in historical_data:
                    bars = historical_data[bars_key]
                    logger.info(f"Found {bars_key}, type: {type(bars)}, length: {len(bars) if hasattr(bars, '__len__') else 'No length'}")
                    if bars and hasattr(bars, '__len__') and len(bars) > 0:
                        logger.info(f"First item in {bars_key}: type={type(bars[0])}, content={str(bars[0])[:200]}...")
                    processed = self._process_historical_bars(bars, timeframe)
                    total_bars += processed
                    logger.info(f"Processed {processed} {timeframe} bars")
                else:
                    logger.info(f"No {bars_key} found in historical data")
            
            # Initialize subsystems
            self._initialize_adaptation_engine()
            
            # Update bootstrap stats
            self.bootstrap_stats['total_bars_processed'] = total_bars
            self.bootstrap_stats['patterns_discovered'] = self._count_total_patterns()
            self.bootstrap_stats['bootstrap_time'] = (datetime.now() - start_time).total_seconds()
            
            self.historical_processed = True
            
            logger.info(f"Bootstrap complete: {total_bars} bars processed, "
                       f"{self.bootstrap_stats['patterns_discovered']} patterns discovered "
                       f"in {self.bootstrap_stats['bootstrap_time']:.1f}s")
            
            self.save_patterns(self.memory_file)
            
        except Exception as e:
            logger.error(f"Bootstrap error: {e}")
            self.historical_processed = False
    
    def _process_historical_bars(self, bars, timeframe):
        """Process historical bars for pattern learning"""
        if not bars or len(bars) < 20:
            return 0
        
        # Debug the structure of bars data
        logger.debug(f"Processing {len(bars)} bars for {timeframe}")
        if bars:
            logger.debug(f"First bar type: {type(bars[0])}, content: {bars[0]}")
        
        try:
            # Handle different possible data structures
            if isinstance(bars[0], dict):
                # Standard dictionary format
                logger.debug("Processing as dictionary format")
                try:
                    prices = [bar['close'] for bar in bars]
                    logger.debug(f"Extracted {len(prices)} prices")
                except Exception as e:
                    logger.error(f"Error extracting prices: {e}")
                    raise
                
                try:
                    volumes = [bar['volume'] for bar in bars]
                    logger.debug(f"Extracted {len(volumes)} volumes")
                except Exception as e:
                    logger.error(f"Error extracting volumes: {e}")
                    raise
                
                try:
                    timestamps = [bar['timestamp'] / 10000000 - 62135596800 for bar in bars]
                    logger.debug(f"Extracted {len(timestamps)} timestamps")
                except Exception as e:
                    logger.error(f"Error extracting timestamps: {e}")
                    raise
                    
            elif isinstance(bars[0], (list, tuple)):
                # Array format [timestamp, open, high, low, close, volume]
                logger.debug("Processing as array format")
                prices = [bar[4] for bar in bars]  # close price
                volumes = [bar[5] if len(bar) > 5 else 1000 for bar in bars]  # volume
                timestamps = [bar[0] / 10000000 - 62135596800 for bar in bars]  # timestamp
            elif isinstance(bars[0], str):
                # String format - try to parse as JSON
                logger.debug("Processing as string format")
                import json
                parsed_bars = []
                for bar_str in bars:
                    try:
                        bar_dict = json.loads(bar_str)
                        parsed_bars.append(bar_dict)
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse bar string: {bar_str}")
                        continue
                
                if parsed_bars:
                    prices = [bar['close'] for bar in parsed_bars]
                    volumes = [bar['volume'] for bar in parsed_bars]
                    timestamps = [bar['timestamp'] / 10000000 - 62135596800 for bar in parsed_bars]
                else:
                    logger.warning(f"No valid bars parsed from strings for {timeframe}")
                    return 0
            else:
                logger.warning(f"Unknown bar data structure: {type(bars[0])}")
                return 0
                
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Error processing bars structure: {e}")
            logger.error(f"Bar sample: {bars[0] if bars else 'No bars'}")
            return 0
        
        processed_count = 0
        window_size = min(50, len(bars) // 4)
        
        logger.debug(f"Starting processing loop with window_size={window_size}, total_bars={len(bars)}")
        
        for i in range(window_size, len(bars)):
            try:
                window_prices = prices[i-window_size:i+1]
                window_volumes = volumes[i-window_size:i+1]
                window_timestamps = timestamps[i-window_size:i+1]
                
                logger.debug(f"Processing window {i}: prices={len(window_prices)}, volumes={len(window_volumes)}, timestamps={len(window_timestamps)}")
                
                # Extract market features
                try:
                    market_features = self._extract_market_features(
                        window_prices, window_volumes, window_timestamps
                    )
                    logger.debug(f"Extracted market features: {list(market_features.keys())}")
                except Exception as e:
                    logger.error(f"Error extracting market features at window {i}: {e}")
                    continue
                
                # Process through orchestrator
                try:
                    orchestrator_result = self.orchestrator.process_market_data(
                        window_prices, window_volumes, market_features, window_timestamps
                    )
                    logger.debug(f"Orchestrator processing complete for window {i}")
                except Exception as e:
                    logger.error(f"Error in orchestrator processing at window {i}: {e}")
                    continue
                
                # Microstructure analysis
                try:
                    microstructure_result = self.microstructure_engine.analyze_market_state(
                        window_prices, window_volumes
                    )
                    logger.debug(f"Microstructure analysis complete for window {i}")
                except Exception as e:
                    logger.error(f"Error in microstructure analysis at window {i}: {e}")
                    continue
                
                # Store patterns for learning
                try:
                    self._store_historical_patterns(orchestrator_result, microstructure_result, market_features)
                    logger.debug(f"Pattern storage complete for window {i}")
                except Exception as e:
                    logger.error(f"Error storing patterns at window {i}: {e}")
                    continue
                
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Unexpected error processing window {i}: {e}")
                continue
        
        logger.debug(f"Processing complete: {processed_count} windows processed")
        return processed_count
    
    def _extract_market_features(self, prices: List[float], volumes: List[float],
                                timestamps: List[float]) -> Dict:
        """Extract market features for analysis"""
        if len(prices) < 10:
            return {}
        
        # Price features
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
        
        # Price position
        price_range = max(prices) - min(prices)
        price_position = (prices[-1] - min(prices)) / price_range if price_range > 0 else 0.5
        
        return {
            'volatility': volatility,
            'price_momentum': momentum,
            'volume_momentum': volume_momentum,
            'time_of_day': time_of_day,
            'price_position': price_position
        }
    
    def _store_historical_patterns(self, orchestrator_result: Dict, microstructure_result: Dict,
                                 market_features: Dict):
        """Store patterns from historical analysis"""
        try:
            # Debug the result types
            logger.debug(f"Orchestrator result type: {type(orchestrator_result)}")
            logger.debug(f"Microstructure result type: {type(microstructure_result)}")
            
            # Validate orchestrator_result
            if not isinstance(orchestrator_result, dict):
                logger.error(f"Orchestrator result is not a dict: {type(orchestrator_result)}, content: {str(orchestrator_result)[:200]}")
                orchestrator_result = {}
            
            # Validate microstructure_result
            if not isinstance(microstructure_result, dict):
                logger.error(f"Microstructure result is not a dict: {type(microstructure_result)}, content: {str(microstructure_result)[:200]}")
                microstructure_result = {}
            
            # Store orchestrator patterns
            current_patterns = orchestrator_result.get('current_patterns', {})
            
            # Create synthetic outcomes for pattern initialization
            signal_strength = abs(float(orchestrator_result.get('overall_signal', 0)))
            microstructure_strength = abs(float(microstructure_result.get('microstructure_signal', 0)))
            
            # Combine signals for synthetic outcome
            synthetic_outcome = (signal_strength + microstructure_strength) / 2
            consensus_strength = float(orchestrator_result.get('consensus_strength', 0))
            if consensus_strength > 0.7:
                synthetic_outcome *= 1.2  # Boost for high consensus
            
            # Add some noise to make it more realistic
            synthetic_outcome += np.random.normal(0, 0.1)
            synthetic_outcome = float(np.clip(synthetic_outcome, -1.0, 1.0))
            
            # Convert numpy types to regular Python types for learning data
            learning_data = {
                'dna_sequence': str(current_patterns.get('dna_sequence', '')),
                'cycles_info': [],  # Would need to extract from temporal subsystem
                'market_state': {k: float(v) if isinstance(v, (int, float, np.number)) else v
                               for k, v in market_features.items()},
                'microstructure_signal': float(microstructure_result.get('microstructure_signal', 0.0))
            }
            
            logger.debug(f"Learning data prepared: {learning_data}")
            
            # Learn from synthetic outcome
            try:
                self.orchestrator.learn_from_outcome(synthetic_outcome, learning_data)
                logger.debug("Orchestrator learning completed successfully")
            except Exception as e:
                logger.error(f"Error in orchestrator learning: {e}")
                logger.error(f"Synthetic outcome: {synthetic_outcome}")
                logger.error(f"Learning data: {learning_data}")
                # Continue without failing the entire process
            
            # Store for microstructure learning
            try:
                self.microstructure_engine.learn_from_outcome(synthetic_outcome)
                logger.debug("Microstructure learning completed successfully")
            except Exception as e:
                logger.error(f"Error in microstructure learning: {e}")
                # Continue without failing the entire process
            
        except Exception as e:
            logger.error(f"Error in _store_historical_patterns: {e}")
            logger.error(f"Orchestrator result: {orchestrator_result}")
            logger.error(f"Microstructure result: {microstructure_result}")
            raise
    
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
    
    def _bootstrap_microstructure_training(self):
        """Enhanced microstructure training during bootstrap"""
        logger.info("Bootstrap training microstructure engine...")
        
        # Generate synthetic market scenarios for microstructure learning
        scenarios = [
            # High volatility scenarios
            {'volatility': 0.08, 'volume_momentum': 0.5, 'price_momentum': 0.03, 'outcome': -0.3},
            {'volatility': 0.09, 'volume_momentum': 0.8, 'price_momentum': -0.04, 'outcome': -0.4},
            
            # Low volatility scenarios
            {'volatility': 0.005, 'volume_momentum': 0.1, 'price_momentum': 0.001, 'outcome': 0.2},
            {'volatility': 0.008, 'volume_momentum': -0.1, 'price_momentum': -0.002, 'outcome': 0.1},
            
            # Normal market scenarios
            {'volatility': 0.02, 'volume_momentum': 0.3, 'price_momentum': 0.01, 'outcome': 0.15},
            {'volatility': 0.025, 'volume_momentum': -0.2, 'price_momentum': -0.01, 'outcome': -0.1},
            
            # Trending scenarios
            {'volatility': 0.03, 'volume_momentum': 0.6, 'price_momentum': 0.02, 'outcome': 0.4},
            {'volatility': 0.035, 'volume_momentum': 0.7, 'price_momentum': -0.025, 'outcome': 0.3},
        ]
        
        for scenario in scenarios:
            try:
                # Create synthetic price/volume data
                base_price = 100.0
                prices = []
                volumes = []
                
                for i in range(20):
                    price_change = np.random.normal(scenario['price_momentum'], scenario['volatility'])
                    volume_change = np.random.normal(scenario['volume_momentum'], 0.3)
                    
                    new_price = base_price * (1 + price_change)
                    new_volume = max(100, 1000 * (1 + volume_change))
                    
                    prices.append(new_price)
                    volumes.append(new_volume)
                    base_price = new_price
                
                # Train microstructure engine
                self.microstructure_engine.analyze_market_state(prices, volumes)
                self.microstructure_engine.learn_from_outcome(scenario['outcome'])
                
            except Exception as e:
                logger.warning(f"Error in microstructure bootstrap scenario: {e}")
                continue
        
        logger.info("Microstructure bootstrap training completed")
    
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
        if len(data.prices_1m) < 20:
            return self._default_features()

        prices = np.array(data.prices_1m[-20:])
        volumes = np.array(data.volumes_1m[-20:]) if len(data.volumes_1m) >= 20 else np.ones(20)

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

        try:
            now = datetime.fromtimestamp(data.timestamp)
            time_of_day = (now.hour * 60 + now.minute) / 1440
        except (OSError, ValueError, OverflowError):
            try:
                if data.timestamp > 1e15:
                    unix_timestamp = (data.timestamp - 621355968000000000) / 10000000
                    if unix_timestamp < 0 or unix_timestamp > 2147483647:
                        raise ValueError
                    now = datetime.fromtimestamp(unix_timestamp)
                    time_of_day = (now.hour * 60 + now.minute) / 1440
                else:
                    raise ValueError
            except:
                import time as time_module
                now = datetime.fromtimestamp(time_module.time())
                time_of_day = (now.hour * 60 + now.minute) / 1440
                logger.warning(f"Using current time as fallback due to invalid timestamp: {data.timestamp}")

        market_features = {
            'volatility': volatility,
            'price_momentum': price_momentum,
            'volume_momentum': volume_momentum,
            'time_of_day': time_of_day,
            'price_position': price_position,
            'account_balance': data.account_balance,
            'margin_utilization': data.margin_utilization
        }

        timestamps = [data.timestamp - i * 60 for i in range(len(prices))][::-1]
        
        # Add error handling and validation for orchestrator result
        try:
            orchestrator_result = self.orchestrator.process_market_data(
                data.prices_1m, data.volumes_1m, market_features, timestamps
            )
            
            # Validate orchestrator result and provide fallbacks
            if not orchestrator_result or not isinstance(orchestrator_result, dict):
                logger.warning("Orchestrator returned invalid result, using fallback")
                orchestrator_result = {
                    'dna_signal': 0.0,
                    'temporal_signal': 0.0,
                    'immune_signal': 0.0,
                    'overall_signal': 0.0,
                    'consensus_strength': 0.5
                }
            
            # Ensure all required keys exist with valid values
            required_keys = ['dna_signal', 'temporal_signal', 'immune_signal', 'overall_signal']
            for key in required_keys:
                if key not in orchestrator_result or not isinstance(orchestrator_result[key], (int, float)):
                    logger.warning(f"Missing or invalid {key} in orchestrator result, using 0.0")
                    orchestrator_result[key] = 0.0
                elif np.isnan(orchestrator_result[key]) or np.isinf(orchestrator_result[key]):
                    logger.warning(f"NaN or Inf detected in {key}, using 0.0")
                    orchestrator_result[key] = 0.0
                    
        except Exception as e:
            logger.error(f"Orchestrator processing failed: {e}")
            orchestrator_result = {
                'dna_signal': 0.0,
                'temporal_signal': 0.0,
                'immune_signal': 0.0,
                'overall_signal': 0.0,
                'consensus_strength': 0.5
            }

        # Add error handling for microstructure analysis
        try:
            microstructure_result = self.microstructure_engine.analyze_market_state(
                data.prices_1m, data.volumes_1m
            )
            
            if not microstructure_result or not isinstance(microstructure_result, dict):
                logger.warning("Microstructure engine returned invalid result, using fallback")
                microstructure_result = {
                    'microstructure_signal': 0.0,
                    'regime_adjusted_signal': 0.0,
                    'order_flow': {},
                    'regime_state': {'confidence': 0.5}
                }
                
        except Exception as e:
            logger.error(f"Microstructure analysis failed: {e}")
            microstructure_result = {
                'microstructure_signal': 0.0,
                'regime_adjusted_signal': 0.0,
                'order_flow': {},
                'regime_state': {'confidence': 0.5}
            }

        adaptation_context = {
            'volatility': volatility,
            'trend_strength': abs(price_momentum),
            'volume_regime': min(1.0, volume_momentum + 0.5),
            'time_of_day': time_of_day
        }

        # Add error handling for adaptation engine
        try:
            feature_tensor = self._create_feature_tensor(market_features, orchestrator_result, microstructure_result)
            adaptation_decision = self.adaptation_engine.get_adaptation_decision(feature_tensor, adaptation_context)
        except Exception as e:
            logger.error(f"Adaptation engine failed: {e}")
            adaptation_decision = {'adaptation_quality': 0.5}

        pattern_score = self._recognize_patterns(prices, volumes)
        
        # Ensure signal strength calculation is robust
        signal_strength = abs(orchestrator_result.get('overall_signal', 0.0))
        momentum_strength = abs(price_momentum) + abs(volume_momentum)
        microstructure_strength = abs(microstructure_result.get('microstructure_signal', 0.0))
        adaptation_quality = adaptation_decision.get('adaptation_quality', 0.5)
        pattern_boost = 0.15 if self.historical_processed else 0.0
        adaptation_boost = adaptation_quality * 0.1
        
        # Ensure minimum confidence level
        confidence = max(0.1, min(
            1.0,
            signal_strength + momentum_strength * 0.3 + microstructure_strength * 0.2 + pattern_boost + adaptation_boost
        ))

        microstructure_features = microstructure_result.get('order_flow', {})
        regime_state = microstructure_result.get('regime_state', {})

        # Create features with validation
        features = Features(
            price_momentum=price_momentum,
            volume_momentum=volume_momentum,
            price_position=price_position,
            volatility=volatility,
            time_of_day=time_of_day,
            pattern_score=pattern_score,
            confidence=confidence,
            dna_signal=orchestrator_result.get('dna_signal', 0.0),
            micro_signal=orchestrator_result.get('microstructure_signal', 0.0),  # Fixed: use microstructure as micro
            temporal_signal=orchestrator_result.get('temporal_signal', 0.0),
            immune_signal=orchestrator_result.get('immune_signal', 0.0),
            overall_signal=orchestrator_result.get('overall_signal', 0.0),
            microstructure_signal=microstructure_result.get('microstructure_signal', 0.0),
            regime_adjusted_signal=microstructure_result.get('regime_adjusted_signal', 0.0),
            adaptation_quality=adaptation_quality,
            smart_money_flow=microstructure_features.get('smart_money_flow', 0.0),
            liquidity_depth=microstructure_features.get('liquidity_depth', 0.5),
            regime_confidence=regime_state.get('confidence', 0.5)
        )
        
        # Log warning if all signals are zero
        if (features.dna_signal == 0.0 and features.temporal_signal == 0.0 and
            features.immune_signal == 0.0 and features.overall_signal == 0.0):
            logger.warning("All intelligence signals are zero - potential subsystem issue")
            
        return features
    
    def _extract_time_of_day(self, timestamp: float) -> float:
        """Extract time of day from timestamp with error handling"""
        try:
            now = datetime.fromtimestamp(timestamp)
            return (now.hour * 60 + now.minute) / 1440
        except (OSError, ValueError, OverflowError):
            try:
                if timestamp > 1e15:
                    unix_timestamp = (timestamp - 621355968000000000) / 10000000
                    if 0 < unix_timestamp < 2147483647:
                        now = datetime.fromtimestamp(unix_timestamp)
                        return (now.hour * 60 + now.minute) / 1440
            except:
                pass
            
            # Fallback to current time
            import time as time_module
            now = datetime.fromtimestamp(time_module.time())
            logger.warning(f"Using current time as fallback due to invalid timestamp: {timestamp}")
            return (now.hour * 60 + now.minute) / 1440
    
    def _create_feature_tensor(self, market_features: Dict, orchestrator_result: Dict,
                             microstructure_result: Dict):
        """Create feature tensor for adaptation engine"""
        import torch
        
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
        
        return torch.tensor(features[:64], dtype=torch.float64, device='cpu')
    
    def _recognize_patterns(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        """Simple pattern recognition"""
        if len(prices) < 10:
            return 0.0
            
        trend = (prices[-1] - prices[-10]) / prices[-10] if prices[-10] != 0 else 0
        vol_trend = (volumes[-1] - np.mean(volumes[-5:])) / np.mean(volumes[-5:]) if np.mean(volumes[-5:]) > 0 else 0
        
        # Pattern strength based on trend and volume confirmation
        if (trend > 0 and vol_trend > 0) or (trend < 0 and vol_trend < 0):
            pattern_strength = abs(trend) * (1 + abs(vol_trend))
        else:
            pattern_strength = abs(trend) * 0.5
            
        return np.tanh(pattern_strength * 10)
    
    def learn_from_outcome(self, trade):
        """Learn from trade outcomes across all subsystems"""
        outcome = trade.pnl / abs(trade.entry_price * 0.01) if trade.entry_price != 0 else 0
        
        # Orchestrator learning
        if hasattr(trade, 'intelligence_data'):
            self.orchestrator.learn_from_outcome(outcome, {
                'dna_sequence': trade.intelligence_data.get('current_patterns', {}).get('dna_sequence', ''),
                'cycles_info': [],
                'market_state': getattr(trade, 'market_features', {})
            })
        
        # Microstructure learning
        self.microstructure_engine.learn_from_outcome(outcome)
        
        # Adaptation learning
        if hasattr(trade, 'market_features'):
            adaptation_context = {
                'volatility': trade.market_features.get('volatility', 0.02),
                'trend_strength': abs(trade.market_features.get('price_momentum', 0)),
                'volume_regime': trade.market_features.get('volume_momentum', 0) + 0.5,
                'time_of_day': trade.market_features.get('time_of_day', 0.5),
                'predicted_confidence': getattr(trade, 'confidence', 0.5)
            }
            self.adaptation_engine.update_from_outcome(outcome, adaptation_context)
        
        # Process adaptation event
        urgency = min(1.0, abs(outcome) * 2.0)
        self.adaptation_engine.process_market_event(
            'trade_outcome',
            {'pnl': outcome, 'trade_data': trade},
            urgency=urgency
        )
        
        # Pattern learning
        if hasattr(trade, 'features'):
            pattern_id = self._create_pattern_id(trade.features)
            self.patterns[pattern_id].append(outcome)
            
            if len(self.patterns[pattern_id]) > 20:
                self.patterns[pattern_id] = self.patterns[pattern_id][-20:]
        
        self.recent_outcomes.append(outcome)
    
    def _create_pattern_id(self, features: Features) -> str:
        """Create pattern ID for legacy compatibility"""
        momentum_bucket = int(features.price_momentum * 5) + 5
        position_bucket = int(features.price_position * 4)
        vol_bucket = int(features.volatility * 100) // 10
        
        return f"p{momentum_bucket}_{position_bucket}_{vol_bucket}"
    
    def _default_features(self) -> Features:
        """Default features when insufficient data"""
        return Features(
            price_momentum=0, volume_momentum=0, price_position=0.5, volatility=0,
            time_of_day=0.5, pattern_score=0, confidence=0, dna_signal=0,
            micro_signal=0, temporal_signal=0, immune_signal=0, microstructure_signal=0.0, overall_signal=0,
            regime_adjusted_signal=0.0, adaptation_quality=0.5,
            smart_money_flow=0.0, liquidity_depth=0.5, regime_confidence=0.5
        )
    
    def save_patterns(self, filepath: str):
        """Save patterns from all subsystems"""
        orchestrator_stats = self.orchestrator.get_comprehensive_stats()
        microstructure_features = self.microstructure_engine.get_microstructure_features()
        adaptation_stats = self.adaptation_engine.get_comprehensive_stats()
        
        data = {
            'patterns': dict(self.patterns),
            'recent_outcomes': list(self.recent_outcomes),
            'historical_processed': self.historical_processed,
            'bootstrap_stats': self.bootstrap_stats,
            'orchestrator_stats': orchestrator_stats,
            'microstructure_features': microstructure_features,
            'adaptation_stats': adaptation_stats,
            'saved_at': datetime.now().isoformat()
        }
        
        try:
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving patterns: {e}")
    
    def load_patterns(self, filepath: str):
        """Load patterns from all subsystems"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.patterns = defaultdict(list, data.get('patterns', {}))
            self.recent_outcomes = deque(data.get('recent_outcomes', []), maxlen=100)
            self.historical_processed = data.get('historical_processed', False)
            self.bootstrap_stats = data.get('bootstrap_stats', {
                'total_bars_processed': 0,
                'patterns_discovered': 0,
                'bootstrap_time': 0
            })
            
            if self.historical_processed:
                orchestrator_stats = data.get('orchestrator_stats', {})
                logger.info(f"Loaded intelligence patterns: "
                           f"DNA={orchestrator_stats.get('dna_evolution', {}).get('total_sequences', 0)}, "
                           f"Immune={orchestrator_stats.get('immune_system', {}).get('total_antibodies', 0)}, "
                           f"Temporal={orchestrator_stats.get('temporal_cycles', 0)}")
            
        except FileNotFoundError:
            logger.info("No existing patterns found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading patterns: {e}")
    
    def get_stats(self) -> Dict:
        """Get statistics from all subsystems"""
        orchestrator_stats = self.orchestrator.get_comprehensive_stats()
        adaptation_stats = self.adaptation_engine.get_comprehensive_stats()
        microstructure_features = self.microstructure_engine.get_microstructure_features()
        
        # Extract subsystem counts
        dna_stats = orchestrator_stats.get('dna_evolution', {})
        immune_stats = orchestrator_stats.get('immune_system', {})
        temporal_cycles = orchestrator_stats.get('temporal_cycles', 0)
        
        # Count microstructure patterns
        micro_patterns_count = 0
        if isinstance(microstructure_features, dict):
            micro_patterns_count = len(microstructure_features.get('patterns', {}))
        
        return {
            'total_patterns': len(self.patterns),
            'recent_performance': np.mean(self.recent_outcomes) if self.recent_outcomes else 0,
            'pattern_count': sum(len(outcomes) for outcomes in self.patterns.values()),
            'historical_processed': self.historical_processed,
            'bootstrap_stats': self.bootstrap_stats,
            'dna_patterns': dna_stats.get('total_sequences', 0),
            'micro_patterns': micro_patterns_count,
            'temporal_patterns': temporal_cycles,
            'immune_patterns': immune_stats.get('total_antibodies', 0),
            'orchestrator': orchestrator_stats,
            'adaptation': adaptation_stats
        }