# intelligence_engine.py

import json
import numpy as np
import logging
import time

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any

from src.core.market_data_processor import MarketData
from src.intelligence.subsystems.subsystem_orchestrator import EnhancedIntelligenceOrchestrator
from src.intelligence.subsystems.microstructure_subsystem import MicrostructureSubsystem
from src.shared.types import Features
from src.shared.intelligence_types import (
    IntelligenceSignal, IntelligenceUpdate, IntelligenceContext,
    IntelligenceSignalType, create_intelligence_signal, create_intelligence_context
)
from src.core.state_coordinator import register_state_component
from src.core.unified_serialization import unified_serializer, serialize_component_state, deserialize_component_state
from src.core.storage_error_handler import storage_error_handler, handle_storage_operation
from src.core.data_integrity_validator import data_integrity_validator, validate_component_data, ValidationLevel

logger = logging.getLogger(__name__)

class DataTypeNormalizer:
    """Utility class for ensuring consistent data types throughout the intelligence pipeline"""
    
    @staticmethod
    def ensure_list(data) -> list:
        """Convert any array-like data to Python list"""
        if hasattr(data, 'tolist'):  # numpy array
            return data.tolist()
        elif isinstance(data, (list, tuple)):
            return list(data)
        else:
            return [data] if data is not None else []
    
    @staticmethod 
    def safe_slice(data, start_idx: int = None, end_idx: int = None) -> list:
        """Safely slice data and return as Python list"""
        try:
            data_list = DataTypeNormalizer.ensure_list(data)
            if start_idx is None and end_idx is None:
                return data_list
            elif start_idx is None:
                return data_list[:end_idx]
            elif end_idx is None:
                return data_list[start_idx:]
            else:
                return data_list[start_idx:end_idx]
        except Exception as e:
            logger.warning(f"Safe slice failed: {e}, returning empty list")
            return []
    
    @staticmethod
    def safe_numpy_calculation(data, default_value=0.0):
        """Safely perform numpy calculations with fallback"""
        try:
            if not data or len(data) == 0:
                return default_value
            return np.array(DataTypeNormalizer.ensure_list(data))
        except Exception as e:
            logger.warning(f"Numpy calculation failed: {e}, using default")
            return np.array([default_value])

class IntelligenceEngine:
    def __init__(self, config, memory_file="data/intelligence_memory.json"):
        self.config = config
        self.memory_file = memory_file
        
        # Core subsystem orchestration (DNA, Temporal, Immune)
        self.orchestrator = EnhancedIntelligenceOrchestrator(config)
        
        # Microstructure analysis (4th subsystem)
        self.microstructure_subsystem = MicrostructureSubsystem(config)
        
        # Note: Dopamine processing has been moved to the trading agent
        # Intelligence engine now focuses on the 4 core subsystems: DNA, Temporal, Immune, Microstructure
        
        # Real-time adaptation (dependency injection to avoid circular imports)
        self.adaptation_engine = None
        
        # Pattern storage
        self.patterns = defaultdict(list)
        self.recent_outcomes = deque(maxlen=100)
        
        # Bootstrap state tracking
        self.historical_processed = False
        self.bootstrap_stats = {
            'total_bars_processed': 0,
            'patterns_discovered': 0,
            'bootstrap_time': 0,
            'dna_patterns_learned': 0,
            'temporal_cycles_found': 0,
            'immune_threats_learned': 0,
            'microstructure_patterns_learned': 0
        }
        
        # Intelligence signal generation parameters
        self.signal_generation_config = {
            'signal_threshold': 0.1,
            'confidence_threshold': 0.3,
            'consensus_window': 20,
            'signal_decay_factor': 0.95
        }
        
        # Subsystem training progress tracking (4 subsystems)
        self.subsystem_training_progress = {
            'dna': {'sequences_processed': 0, 'learning_events': 0},
            'temporal': {'cycles_analyzed': 0, 'learning_events': 0},
            'immune': {'threats_detected': 0, 'patterns_analyzed': 0, 'learning_events': 0},
            'microstructure': {'patterns_analyzed': 0, 'learning_events': 0}
        }
        
        # Signal history for consensus calculation
        self.recent_signals = deque(maxlen=100)
        
        # Register with state coordinator for unified state management
        self._register_state_management()
        
        # Load existing state if available
        self._load_persisted_state()
        
        # Legacy load for backward compatibility
        self.load_patterns(self.memory_file)

    def _get_adaptation_engine(self):
        """Get adaptation engine via dependency injection"""
        if self.adaptation_engine is None:
            from src.core.dependency_registry import get_service
            try:
                self.adaptation_engine = get_service('adaptation_engine')
            except ValueError:
                # Fallback to direct creation if not registered
                from src.agent.real_time_adaptation import RealTimeAdaptationEngine
                self.adaptation_engine = RealTimeAdaptationEngine(model_dim=64)
                logger.warning("Adaptation engine not in registry, created directly")
        return self.adaptation_engine

    def save_patterns(self, filepath: str):
        """Enhanced save with all subsystem patterns"""
        if not filepath or filepath.strip() == "":
            filepath = "data/intelligence_memory.json"
            logger.warning(f"Empty filepath provided, using default: {filepath}")
        
        orchestrator_stats = self.orchestrator.get_comprehensive_stats()
        microstructure_features = self.microstructure_subsystem.get_microstructure_features()
        adaptation_stats = self._get_adaptation_engine().get_comprehensive_stats()
        
        # Include intelligence signal statistics
        signal_stats = self._get_signal_generation_stats()
        
        data = {
            'patterns': dict(self.patterns),
            'recent_outcomes': list(self.recent_outcomes),
            'historical_processed': self.historical_processed,
            'bootstrap_stats': self.bootstrap_stats,
            'subsystem_training_progress': self.subsystem_training_progress,
            'orchestrator_stats': orchestrator_stats,
            'microstructure_features': microstructure_features,
            'adaptation_stats': adaptation_stats,
            'signal_generation_stats': signal_stats,
            'saved_at': datetime.now().isoformat()
        }
        
        try:
            import os, numpy as np
            
            def _enhanced_json_encoder(obj):
                """Convert NumPy types, Trade objects, and other complex types to JSON-serializable formats."""
                # Handle NumPy types
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, (np.bool_,)):
                    return bool(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                
                # Handle Trade objects
                if hasattr(obj, '__class__') and 'Trade' in str(type(obj)):
                    # Convert Trade to a simple dictionary with basic info
                    try:
                        return {
                            'entry_time': getattr(obj, 'entry_time', 0),
                            'exit_time': getattr(obj, 'exit_time', 0),
                            'entry_price': getattr(obj, 'entry_price', 0.0),
                            'exit_price': getattr(obj, 'exit_price', 0.0),
                            'size': getattr(obj, 'size', 0),
                            'pnl': getattr(obj, 'pnl', 0.0),
                            'type_info': str(type(obj))
                        }
                    except Exception:
                        return {'type_info': str(type(obj)), 'serialization_error': True}
                
                # Handle other complex objects by converting to string representation
                if hasattr(obj, '__dict__'):
                    try:
                        # Try to convert to dict, but filter out complex nested objects
                        simple_dict = {}
                        for key, value in obj.__dict__.items():
                            if isinstance(value, (str, int, float, bool, type(None))):
                                simple_dict[key] = value
                            else:
                                simple_dict[key] = str(value)[:100]  # Truncate long strings
                        return simple_dict
                    except Exception:
                        return {'type_info': str(type(obj)), 'serialization_fallback': True}
                
                # Final fallback - convert to string
                try:
                    return str(obj)
                except Exception:
                    return {'type_info': str(type(obj)), 'serialization_failed': True}
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=_enhanced_json_encoder)
                
            logger.info("Enhanced patterns saved with all four subsystems")
        except Exception as e:
            logger.error(f"Error saving enhanced patterns: {e}")
    
    def load_patterns(self, filepath: str):
        """Enhanced load with all subsystem patterns"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.patterns = defaultdict(list, data.get('patterns', {}))
            self.recent_outcomes = deque(data.get('recent_outcomes', []), maxlen=100)
            self.historical_processed = data.get('historical_processed', False)
            
            self.bootstrap_stats = data.get('bootstrap_stats', {
                'total_bars_processed': 0,
                'patterns_discovered': 0,
                'bootstrap_time': 0,
                'dna_patterns_learned': 0,
                'temporal_cycles_found': 0,
                'immune_threats_learned': 0,
                'microstructure_patterns_learned': 0
            })
            
            self.subsystem_training_progress = data.get('subsystem_training_progress', {
                'dna': {'sequences_processed': 0, 'learning_events': 0},
                'temporal': {'cycles_analyzed': 0, 'learning_events': 0},
                'immune': {'threats_detected': 0, 'learning_events': 0},
                'microstructure': {'patterns_analyzed': 0, 'learning_events': 0}
            })
            
            if self.historical_processed:
                orchestrator_stats = data.get('orchestrator_stats', {})
                logger.info(f"Loaded enhanced intelligence patterns with all subsystems:")
                logger.info(f"  DNA={orchestrator_stats.get('dna_evolution', {}).get('total_sequences', 0)}")
                logger.info(f"  Temporal={orchestrator_stats.get('temporal_cycles', 0)}")
                logger.info(f"  Immune={orchestrator_stats.get('immune_system', {}).get('total_antibodies', 0)}")
                logger.info(f"  Microstructure patterns loaded")
            
        except FileNotFoundError:
            logger.info("No existing enhanced patterns found, starting fresh with all subsystems")
        except Exception as e:
            logger.error(f"Error loading enhanced patterns: {e}")
        
    def bootstrap_from_historical_data(self, historical_data):
        """Enhanced bootstrap with comprehensive subsystem training"""
        logger.info("Starting comprehensive historical data bootstrap for all 4 subsystems...")
        start_time = datetime.now()
        
        try:
            # Debug the historical data structure
            logger.info(f"Historical data type: {type(historical_data)}")
            logger.info(f"Historical data keys: {list(historical_data.keys()) if isinstance(historical_data, dict) else 'Not a dict'}")
            
            total_bars = 0
            
            # Process each timeframe with comprehensive subsystem training
            for timeframe in ['4h', '1h', '15m', '5m', '1m']:
                bars_key = f'bars_{timeframe}'
                if bars_key in historical_data:
                    bars = historical_data[bars_key]
                    logger.info(f"Found {bars_key}, type: {type(bars)}, length: {len(bars) if hasattr(bars, '__len__') else 'No length'}")
                    if bars and hasattr(bars, '__len__') and len(bars) > 0:
                        logger.info(f"First item in {bars_key}: type={type(bars[0])}, content={str(bars[0])[:200]}...")
                    
                    # Enhanced processing with all subsystems
                    processed = self._process_historical_bars_comprehensive(bars, timeframe)
                    total_bars += processed
                    logger.info(f"Processed {processed} {timeframe} bars across all subsystems")
            
            # Comprehensive subsystem training phase
            self._comprehensive_subsystem_training()
            
            # Initialize adaptation engine with historical context
            self._initialize_adaptation_engine()
            
            # Update bootstrap stats with all subsystems
            self.bootstrap_stats['total_bars_processed'] = total_bars
            self.bootstrap_stats['patterns_discovered'] = self._count_total_patterns()
            self.bootstrap_stats['bootstrap_time'] = (datetime.now() - start_time).total_seconds()
            
            # Detailed subsystem stats
            orchestrator_stats = self.orchestrator.get_comprehensive_stats()
            self.bootstrap_stats['dna_patterns_learned'] = orchestrator_stats.get('dna_evolution', {}).get('total_sequences', 0)
            self.bootstrap_stats['temporal_cycles_found'] = orchestrator_stats.get('temporal_cycles', 0)
            self.bootstrap_stats['immune_threats_learned'] = orchestrator_stats.get('immune_system', {}).get('total_antibodies', 0)
            
            # Update microstructure stats from training progress 
            # Note: 'patterns_analyzed' represents data windows processed, not learned patterns stored
            self.bootstrap_stats['microstructure_patterns_learned'] = self.subsystem_training_progress['microstructure']['patterns_analyzed']
            
            # Close progress bars after bootstrap to prevent interference with live trading logs
            self.orchestrator.close_progress_bars()
            
            # Mark all subsystems as live trading to prevent progress bar reinitialization
            self.orchestrator.dna_subsystem._live_trading_started = True
            self.orchestrator.temporal_subsystem._live_trading_started = True
            self.orchestrator.immune_subsystem._live_trading_started = True
            
            self.historical_processed = True
            
            logger.info(f"Enhanced bootstrap complete: {total_bars} bars processed across all subsystems")
            logger.info(f"DNA patterns: {self.bootstrap_stats['dna_patterns_learned']} (stored sequences)")
            logger.info(f"Temporal cycles: {self.bootstrap_stats['temporal_cycles_found']} (detected cycles)")
            logger.info(f"Immune threats: {self.bootstrap_stats['immune_threats_learned']} (learned antibodies)")
            logger.info(f"Microstructure patterns: {self.bootstrap_stats['microstructure_patterns_learned']} (data windows analyzed)")
            logger.info(f"Total time: {self.bootstrap_stats['bootstrap_time']:.1f}s")
            
            self.save_patterns(self.memory_file)
            
        except Exception as e:
            logger.error(f"Enhanced bootstrap error: {e}")
            self.historical_processed = False
    
    def _process_historical_bars_comprehensive(self, bars, timeframe):
        """Process historical bars with training for all four subsystems"""
        if not bars or len(bars) < 5:  # Need more data for comprehensive training
            return 0
        
        logger.debug(f"Comprehensive processing {len(bars)} bars for {timeframe}")
        
        try:
            # Extract price/volume data
            if isinstance(bars[0], dict):
                prices = [bar['close'] for bar in bars]
                volumes = [bar['volume'] for bar in bars]
                timestamps = [bar['timestamp'] / 10000000 - 62135596800 for bar in bars]
            elif isinstance(bars[0], (list, tuple)):
                prices = [bar[4] for bar in bars]
                volumes = [bar[5] if len(bar) > 5 else 1000 for bar in bars]
                timestamps = [bar[0] / 10000000 - 62135596800 for bar in bars]
            else:
                logger.warning(f"Unknown bar data structure: {type(bars[0])}")
                return 0
                
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Error processing bars structure: {e}")
            return 0
        
        processed_count = 0
        window_size = min(100, len(bars) // 4)  # Larger window for comprehensive training
        
        logger.info(f"Starting comprehensive processing with window_size={window_size}")
        
        # Process overlapping windows for comprehensive training
        step_size = max(1, window_size // 4)  # Overlapping windows for better training
        
        for i in range(window_size, len(bars), step_size):
            try:
                window_prices = prices[i-window_size:i+1]
                window_volumes = volumes[i-window_size:i+1]
                window_timestamps = timestamps[i-window_size:i+1]
                
                # Extract comprehensive market features
                market_features = self._extract_comprehensive_market_features(
                    window_prices, window_volumes, window_timestamps
                )
                
                # 1. DNA Subsystem Training
                dna_training_result = self._train_dna_subsystem(
                    window_prices, window_volumes, market_features
                )
                
                # 2. Temporal Subsystem Training
                temporal_training_result = self._train_temporal_subsystem(
                    window_prices, window_timestamps
                )
                
                # 3. Immune Subsystem Training
                immune_training_result = self._train_immune_subsystem(
                    market_features
                )
                
                # 4. Microstructure Subsystem Training
                microstructure_training_result = self._train_microstructure_subsystem(
                    window_prices, window_volumes
                )
                
                # Generate synthetic outcomes for training based on all subsystem signals
                synthetic_outcome = self._generate_comprehensive_synthetic_outcome(
                    dna_training_result, temporal_training_result, 
                    immune_training_result, microstructure_training_result,
                    market_features
                )
                
                # Train all subsystems with the synthetic outcome
                self._apply_comprehensive_learning(
                    dna_training_result, temporal_training_result,
                    immune_training_result, microstructure_training_result,
                    synthetic_outcome, market_features
                )
                
                processed_count += 1
                
                # Progress logging - more frequent for longer timeframes, less for shorter ones
                progress_interval = {
                    '4h': 5, '1h': 10, '15m': 10, '5m': 25, '1m': 50
                }.get(timeframe, 50)
                
                if processed_count % progress_interval == 0:
                    logger.info(f"Comprehensive training progress: {processed_count} windows processed for {timeframe}")
                
            except Exception as e:
                logger.error(f"Error in comprehensive window processing at {i}: {e}")
                continue
        
        logger.info(f"Comprehensive processing complete: {processed_count} windows for {timeframe}")
        
        # Add clean line break after progress bar updates for major processing phases
        if processed_count > 20:  # Show line break for any significant processing
            print()
            
        return processed_count
    
    def _extract_comprehensive_market_features(self, prices: List[float], volumes: List[float],
                                             timestamps: List[float]) -> Dict:
        """Extract comprehensive features for all subsystems"""
        if len(prices) < 20:
            return {}
        
        # Enhanced feature extraction for all subsystems with safe data handling
        safe_prices = DataTypeNormalizer.safe_slice(prices, -50)
        safe_volumes = DataTypeNormalizer.safe_slice(volumes, -50)
        
        price_array = DataTypeNormalizer.safe_numpy_calculation(safe_prices)
        volume_array = DataTypeNormalizer.safe_numpy_calculation(safe_volumes)
        
        # Price dynamics
        returns = np.diff(price_array) / price_array[:-1]
        volatility = np.std(returns) if len(returns) > 1 else 0
        momentum = (price_array[-1] - price_array[-10]) / price_array[-10] if len(price_array) >= 10 and price_array[-10] != 0 else 0
        
        # Volume dynamics
        volume_momentum = (np.mean(volume_array[-5:]) - np.mean(volume_array[-15:-5])) / np.mean(volume_array[-15:-5]) if len(volume_array) >= 15 else 0
        volume_volatility = np.std(volume_array) / np.mean(volume_array) if np.mean(volume_array) > 0 else 0
        
        # Price-volume relationship
        price_volume_corr = np.corrcoef(price_array[-20:], volume_array[-20:])[0,1] if len(price_array) >= 20 else 0
        if np.isnan(price_volume_corr):
            price_volume_corr = 0
        
        # Position and range analysis with safe data access
        safe_prices_list = DataTypeNormalizer.ensure_list(prices)
        if safe_prices_list:
            price_range = max(safe_prices_list) - min(safe_prices_list)
            price_position = (safe_prices_list[-1] - min(safe_prices_list)) / price_range if price_range > 0 else 0.5
        else:
            price_position = 0.5
        
        # Time-based features
        if timestamps:
            try:
                dt = datetime.fromtimestamp(timestamps[-1])
                time_of_day = (dt.hour * 60 + dt.minute) / 1440
                day_of_week = dt.weekday() / 6.0
            except:
                time_of_day = 0.5
                day_of_week = 0.5
        else:
            time_of_day = 0.5
            day_of_week = 0.5
        
        # Advanced technical indicators for microstructure analysis
        rsi = self._calculate_rsi(price_array, 14) if len(price_array) >= 15 else 50
        
        # Bollinger Bands
        if len(price_array) >= 20:
            bb_upper, bb_lower = self._calculate_bollinger_bands(price_array, 20, 2)
            bb_position = (price_array[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) if bb_upper[-1] != bb_lower[-1] else 0.5
        else:
            bb_position = 0.5
        
        # MACD for temporal analysis
        if len(price_array) >= 26:
            ema_12 = self._calculate_ema(price_array, 12)
            ema_26 = self._calculate_ema(price_array, 26)
            macd = (ema_12[-1] - ema_26[-1]) / price_array[-1] if price_array[-1] != 0 else 0
        else:
            macd = 0
        
        # Market stress indicators for immune system
        extreme_moves = sum(1 for r in returns[-10:] if abs(r) > volatility * 2) if volatility > 0 else 0
        stress_indicator = extreme_moves / min(10, len(returns))
        
        # Liquidity indicators for microstructure
        volume_spike_count = sum(1 for i in range(-10, 0) if i < len(volume_array) and 
                               volume_array[i] > np.mean(volume_array[max(0, i-5):i]) * 1.5)
        liquidity_depth = 1.0 - (volume_spike_count / 10.0)
        
        return {
            # Core features
            'volatility': volatility,
            'price_momentum': momentum,
            'volume_momentum': volume_momentum,
            'volume_volatility': volume_volatility,
            'time_of_day': time_of_day,
            'day_of_week': day_of_week,
            'price_position': price_position,
            'price_volume_correlation': price_volume_corr,
            
            # Technical indicators
            'rsi': rsi / 100.0,  # Normalize to 0-1
            'bb_position': bb_position,
            'macd': macd,
            
            # Regime indicators
            'stress_indicator': stress_indicator,
            'liquidity_depth': liquidity_depth,
            'regime': self._classify_market_regime(volatility, momentum, stress_indicator),
            
            # Raw data for subsystems
            'returns': returns.tolist() if len(returns) > 0 else [],
            'price_changes': [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices)) if prices[i-1] != 0],
            'volume_ratios': [volumes[i] / volumes[i-1] for i in range(1, len(volumes)) if volumes[i-1] > 0]
        }
    
    def _train_dna_subsystem(self, prices: List[float], volumes: List[float], 
                           market_features: Dict) -> Dict:
        """Train DNA subsystem with enhanced pattern learning"""
        try:
            # Enhanced DNA encoding with market context
            volatility = market_features.get('volatility', 0)
            momentum = market_features.get('price_momentum', 0)
            
            # Use safe slicing to ensure proper data types
            safe_prices = DataTypeNormalizer.safe_slice(prices, -20)
            safe_volumes = DataTypeNormalizer.safe_slice(volumes, -20)
            
            dna_sequence = self.orchestrator.dna_subsystem.encode_market_state(
                safe_prices, safe_volumes, volatility, momentum
            )
            
            if dna_sequence:
                # Analyze the sequence
                dna_signal = self.orchestrator.dna_subsystem.analyze_sequence(dna_sequence)
                
                # Track training progress
                self.subsystem_training_progress['dna']['sequences_processed'] += 1
                
                return {
                    'subsystem': 'dna',
                    'sequence': dna_sequence,
                    'signal': dna_signal,
                    'strength': abs(dna_signal),
                    'market_context': {
                        'volatility': volatility,
                        'momentum': momentum
                    }
                }
            
        except Exception as e:
            logger.error(f"Error in DNA subsystem training: {e}")
        
        return {'subsystem': 'dna', 'signal': 0.0, 'strength': 0.0, 'sequence': ''}
    
    def _train_temporal_subsystem(self, prices: List[float], timestamps: List[float]) -> Dict:
        """Train temporal subsystem with cycle detection"""
        try:
            # Enhanced temporal analysis
            temporal_signal = self.orchestrator.temporal_subsystem.analyze_cycles(
                prices, timestamps
            )
            
            # Extract cycle information
            cycles_info = []
            if len(self.orchestrator.temporal_subsystem.dominant_cycles) > 0:
                cycles_info = list(self.orchestrator.temporal_subsystem.dominant_cycles)[-1]
            
            # Track training progress
            self.subsystem_training_progress['temporal']['cycles_analyzed'] += 1
            
            return {
                'subsystem': 'temporal',
                'signal': temporal_signal,
                'strength': abs(temporal_signal),
                'cycles_info': cycles_info,
                'cycle_count': len(cycles_info) if isinstance(cycles_info, list) else 0
            }
            
        except Exception as e:
            logger.error(f"Error in temporal subsystem training: {e}")
        
        return {'subsystem': 'temporal', 'signal': 0.0, 'strength': 0.0, 'cycles_info': []}
    
    def _train_immune_subsystem(self, market_features: Dict) -> Dict:
        """Train immune subsystem with threat detection on historical data"""
        try:
            # Enhanced threat detection
            immune_signal = self.orchestrator.immune_subsystem.detect_threats(market_features)
            
            # Track training progress
            self.subsystem_training_progress['immune']['threats_detected'] += 1
            
            # Train on negative signals (threats) from historical data
            if immune_signal < -0.1:  # Significant threat detected
                self.orchestrator.immune_subsystem.learn_threat(
                    market_features, 
                    immune_signal, 
                    is_bootstrap=True
                )
                self.subsystem_training_progress['immune']['patterns_analyzed'] += 1
                
                # Log immune learning progress
                antibody_count = len(self.orchestrator.immune_subsystem.antibodies)
                if antibody_count > 0 and antibody_count % 5 == 0:
                    logger.info(f"Immune Learning: {antibody_count} threats learned")
            
            # Classify threat type
            threat_type = 'none'
            if immune_signal < -0.3:
                threat_type = 'high_threat'
            elif immune_signal < -0.1:
                threat_type = 'moderate_threat'
            elif immune_signal > 0.1:
                threat_type = 'false_positive'
            
            return {
                'subsystem': 'immune',
                'signal': immune_signal,
                'strength': abs(immune_signal),
                'threat_type': threat_type,
                'antibody_count': len(self.orchestrator.immune_subsystem.antibodies)
            }
            
        except Exception as e:
            logger.error(f"Error in immune subsystem training: {e}")
        
        return {'subsystem': 'immune', 'signal': 0.0, 'strength': 0.0, 'threat_type': 'none'}
    
    def _train_microstructure_subsystem(self, prices: List[float], volumes: List[float]) -> Dict:
        """Train microstructure subsystem with order flow analysis"""
        try:
            # Enhanced microstructure analysis using integrated subsystem
            market_features = {
                'prices': prices,
                'volumes': volumes,
                'price_momentum': (prices[-1] - prices[0]) / prices[0] if len(prices) > 1 and prices[0] != 0 else 0.0,
                'volume_ratio': sum(volumes[-5:]) / sum(volumes[:-5]) if len(volumes) > 10 else 1.0
            }
            
            microstructure_result = self.microstructure_subsystem.get_enhanced_signal(market_features)
            
            # Track training progress
            self.subsystem_training_progress['microstructure']['patterns_analyzed'] += 1
            
            # Extract key signals
            microstructure_signal = microstructure_result.get('signal', 0.0)
            regime_adjusted_signal = microstructure_result.get('confidence', 0.0)
            
            # Extract order flow features  
            context = microstructure_result.get('context', {})
            smart_money_flow = context.get('smart_money_flow', 0.0)
            liquidity_depth = context.get('liquidity_depth', 0.5)
            
            # Log microstructure learning progress
            pattern_count = self.subsystem_training_progress['microstructure']['patterns_analyzed']
            if pattern_count > 0 and pattern_count % 25 == 0:
                logger.info(f"Microstructure Learning: {pattern_count} patterns learned")
            
            return {
                'subsystem': 'microstructure',
                'signal': microstructure_signal,
                'regime_adjusted_signal': regime_adjusted_signal,
                'strength': abs(microstructure_signal),
                'smart_money_flow': smart_money_flow,
                'liquidity_depth': liquidity_depth,
                'order_flow': context,
                'regime_state': microstructure_result.get('reasoning', '')
            }
            
        except Exception as e:
            logger.error(f"Error in microstructure subsystem training: {e}")
        
        return {
            'subsystem': 'microstructure', 
            'signal': 0.0, 
            'strength': 0.0,
            'smart_money_flow': 0.0,
            'liquidity_depth': 0.5
        }
    
    def _generate_comprehensive_synthetic_outcome(self, dna_result: Dict, temporal_result: Dict,
                                                immune_result: Dict, microstructure_result: Dict,
                                                market_features: Dict) -> float:
        """Generate synthetic outcome based on all four subsystem signals"""
        
        # Extract signals from all subsystems
        dna_signal = dna_result.get('signal', 0.0)
        temporal_signal = temporal_result.get('signal', 0.0)
        immune_signal = immune_result.get('signal', 0.0)
        microstructure_signal = microstructure_result.get('signal', 0.0)
        
        # Calculate signal strengths
        dna_strength = dna_result.get('strength', 0.0)
        temporal_strength = temporal_result.get('strength', 0.0)
        immune_strength = immune_result.get('strength', 0.0)
        microstructure_strength = microstructure_result.get('strength', 0.0)
        
        # Equal adaptive weights for subsystem discovery
        base_weights = {
            'dna': 0.25,
            'temporal': 0.25,
            'immune': 0.25,
            'microstructure': 0.25
        }
        
        # Adjust weights based on signal strength and market conditions
        total_strength = dna_strength + temporal_strength + immune_strength + microstructure_strength
        
        if total_strength > 0:
            # Weight by signal strength
            strength_weights = {
                'dna': dna_strength / total_strength,
                'temporal': temporal_strength / total_strength,
                'immune': immune_strength / total_strength,
                'microstructure': microstructure_strength / total_strength
            }
            
            # Combine base weights and strength weights
            final_weights = {
                'dna': base_weights['dna'] * 0.7 + strength_weights['dna'] * 0.3,
                'temporal': base_weights['temporal'] * 0.7 + strength_weights['temporal'] * 0.3,
                'immune': base_weights['immune'] * 0.7 + strength_weights['immune'] * 0.3,
                'microstructure': base_weights['microstructure'] * 0.7 + strength_weights['microstructure'] * 0.3
            }
        else:
            final_weights = base_weights
        
        # Calculate weighted signal
        weighted_signal = (
            dna_signal * final_weights['dna'] +
            temporal_signal * final_weights['temporal'] +
            immune_signal * final_weights['immune'] +
            microstructure_signal * final_weights['microstructure']
        )
        
        # Enhance signal based on consensus
        signals = [dna_signal, temporal_signal, immune_signal, microstructure_signal]
        
        # Calculate consensus
        positive_signals = sum(1 for s in signals if s > 0.1)
        negative_signals = sum(1 for s in signals if s < -0.1)
        consensus_strength = max(positive_signals, negative_signals) / len(signals)
        
        # Apply consensus bonus
        if consensus_strength > 0.75:
            weighted_signal *= 1.3  # Strong consensus bonus
        elif consensus_strength < 0.25:
            weighted_signal *= 0.7  # Weak consensus penalty
        
        # Market condition adjustments
        volatility = market_features.get('volatility', 0.02)
        if volatility > 0.05:  # High volatility reduces outcome reliability
            weighted_signal *= 0.8
        
        stress_indicator = market_features.get('stress_indicator', 0.0)
        if stress_indicator > 0.5:  # High stress reduces positive outcomes
            if weighted_signal > 0:
                weighted_signal *= 0.6
        
        # Add some noise for realistic training, but amplify for immune system training
        noise = np.random.normal(0, 0.1)
        synthetic_outcome = weighted_signal + noise

        # Amplify negative outcomes to ensure immune system learns threats
        if synthetic_outcome < -0.2:
            synthetic_outcome *= 2.0
        
        # Bound the outcome
        synthetic_outcome = float(np.clip(synthetic_outcome, -1.0, 1.0))
        
        return synthetic_outcome
    
    def _apply_comprehensive_learning(self, dna_result: Dict, temporal_result: Dict,
                                    immune_result: Dict, microstructure_result: Dict,
                                    outcome: float, market_features: Dict):
        """Apply learning to all subsystems based on synthetic outcome"""
        
        try:
            # 1. DNA Subsystem Learning with type safety
            dna_sequence = dna_result.get('sequence', '')
            if dna_sequence and isinstance(dna_sequence, str):
                self.orchestrator.dna_subsystem.learn_from_outcome(dna_sequence, outcome)
                self.subsystem_training_progress['dna']['learning_events'] += 1
            
            # 2. Temporal Subsystem Learning with type safety
            cycles_info = temporal_result.get('cycles_info', [])
            if cycles_info and isinstance(cycles_info, list):
                self.orchestrator.temporal_subsystem.learn_from_outcome(cycles_info, outcome)
                self.subsystem_training_progress['temporal']['learning_events'] += 1
            
            # 3. Immune Subsystem Learning with safe market features
            if isinstance(market_features, dict):
                self.orchestrator.immune_subsystem.learn_threat(market_features, outcome)
                self.subsystem_training_progress['immune']['learning_events'] += 1
            
            # 4. Microstructure Subsystem Learning with safe context
            if isinstance(microstructure_result, dict):
                self.microstructure_subsystem.learn_from_outcome(outcome, microstructure_result)
                self.subsystem_training_progress['microstructure']['learning_events'] += 1
            
        except Exception as e:
            import traceback
            logger.error(f"Error in comprehensive learning: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _comprehensive_subsystem_training(self):
        """Additional comprehensive training phase for all subsystems"""
        logger.info("Starting comprehensive subsystem training phase...")
        
        try:
            # Cross-subsystem pattern discovery
            self._discover_cross_subsystem_patterns()
            
            # Subsystem evolution
            self._evolve_all_subsystems()
            
            # Generate synthetic market scenarios for additional training
            self._generate_synthetic_scenarios()
            
            logger.info("Comprehensive subsystem training completed")
            
        except Exception as e:
            logger.error(f"Error in comprehensive subsystem training: {e}")

    def _initialize_adaptation_engine(self):
        """Initialize the real-time adaptation engine with historical context"""
        logger.info("Initializing real-time adaptation engine with all subsystems...")
        
        # Process initialization events for each subsystem
        for i in range(10):
            self._get_adaptation_engine().process_market_event(
                'initialization',
                {
                    'pattern_count': i * 10, 
                    'bootstrap_complete': True,
                    'subsystem_count': 4,
                    'dna_patterns': self.subsystem_training_progress['dna']['sequences_processed'],
                    'temporal_cycles': self.subsystem_training_progress['temporal']['cycles_analyzed'],
                    'immune_threats': self.subsystem_training_progress['immune']['threats_detected'],
                    'microstructure_patterns': self.subsystem_training_progress['microstructure']['patterns_analyzed']
                },
                urgency=0.3
            )
    
    def _discover_cross_subsystem_patterns(self):
        """Discover patterns that involve multiple subsystems"""
        logger.info("Discovering cross-subsystem patterns...")
        
        # Get stats from all subsystems
        orchestrator_stats = self.orchestrator.get_comprehensive_stats()
        microstructure_features = self.microstructure_subsystem.get_microstructure_features()
        
        # Look for correlations between subsystem signals
        # This would involve analyzing historical performance of subsystem combinations
        # For now, we'll do basic pattern identification
        
        dna_patterns = orchestrator_stats.get('dna_evolution', {}).get('total_sequences', 0)
        temporal_cycles = orchestrator_stats.get('temporal_cycles', 0)
        immune_antibodies = orchestrator_stats.get('immune_system', {}).get('total_antibodies', 0)
        
        logger.info(f"Cross-pattern analysis: DNA={dna_patterns}, Temporal={temporal_cycles}, Immune={immune_antibodies}")
    
    def _evolve_all_subsystems(self):
        """Trigger evolution in all subsystems"""
        logger.info("Evolving all subsystems...")
        
        try:
            # Evolve immune system antibodies
            self.orchestrator.immune_subsystem.evolve_antibodies()
            
            # The DNA and temporal subsystems evolve automatically during learning
            
            logger.info("Subsystem evolution completed")
            
        except Exception as e:
            logger.error(f"Error in subsystem evolution: {e}")
    
    def _generate_synthetic_scenarios(self):
        """Generate synthetic market scenarios for additional training"""
        logger.info("Generating synthetic training scenarios...")
        
        # Generate various market scenarios
        scenarios = [
            # High volatility scenarios
            {'volatility': 0.08, 'momentum': 0.03, 'volume_momentum': 0.5, 'expected_outcome': -0.2},
            {'volatility': 0.09, 'momentum': -0.04, 'volume_momentum': 0.8, 'expected_outcome': -0.3},
            
            # Trending scenarios
            {'volatility': 0.02, 'momentum': 0.02, 'volume_momentum': 0.4, 'expected_outcome': 0.4},
            {'volatility': 0.025, 'momentum': -0.025, 'volume_momentum': 0.6, 'expected_outcome': 0.3},
            
            # Low volatility scenarios
            {'volatility': 0.005, 'momentum': 0.001, 'volume_momentum': 0.1, 'expected_outcome': 0.1},
            {'volatility': 0.008, 'momentum': -0.002, 'volume_momentum': -0.1, 'expected_outcome': 0.05},
        ]
        
        for scenario in scenarios:
            try:
                # Create synthetic market state
                market_state = {
                    'volatility': scenario['volatility'],
                    'price_momentum': scenario['momentum'],
                    'volume_momentum': scenario['volume_momentum'],
                    'time_of_day': 0.5,
                    'price_position': 0.5,
                    'stress_indicator': scenario['volatility'] * 10,
                    'regime': self._classify_market_regime(scenario['volatility'], scenario['momentum'], scenario['volatility'] * 10)
                }
                
                # Train immune system on synthetic scenarios
                self.orchestrator.immune_subsystem.learn_threat(market_state, scenario['expected_outcome'])
                
                # Train microstructure on synthetic scenarios
                self.microstructure_subsystem.learn_from_outcome(scenario['expected_outcome'])
                
            except Exception as e:
                logger.warning(f"Error in synthetic scenario training: {e}")
                continue
        
        logger.info("Synthetic scenario training completed")
    
    def _classify_market_regime(self, volatility: float, momentum: float, stress: float) -> str:
        """Classify market regime for subsystem training"""
        # Adaptive regime classification - let AI discover market states
        regime_score = stress * 2.0 + volatility * 10.0 + abs(momentum) * 50.0
        if regime_score > 2.0:  # Much higher threshold for adaptive discovery
            return 'crisis'
        elif regime_score > 1.0:  # Higher threshold
            return 'volatile' 
        else:
            return 'normal'
    
    def _calculate_rsi(self, prices: np.ndarray, period: int) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA"""
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int, std_dev: float):
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return np.full_like(prices, prices[-1] * 1.02), np.full_like(prices, prices[-1] * 0.98)
        
        rolling_mean = np.convolve(prices, np.ones(period)/period, mode='same')
        rolling_std = np.array([np.std(prices[max(0, i-period+1):i+1]) for i in range(len(prices))])
        
        upper_band = rolling_mean + (rolling_std * std_dev)
        lower_band = rolling_mean - (rolling_std * std_dev)
        
        return upper_band, lower_band
    
    def _calculate_enhanced_consensus(self, signals: List[float]) -> float:
        """Calculate consensus among all four subsystems"""
        if not signals:
            return 0.5
        
        # Directional agreement
        positive_signals = sum(1 for s in signals if s > 0.1)
        negative_signals = sum(1 for s in signals if s < -0.1)
        neutral_signals = sum(1 for s in signals if abs(s) <= 0.1)
        
        total_signals = len(signals)
        directional_consensus = max(positive_signals, negative_signals, neutral_signals) / total_signals
        
        # Magnitude agreement
        signal_magnitudes = [abs(s) for s in signals]
        if len(signal_magnitudes) > 1:
            magnitude_std = np.std(signal_magnitudes)
            magnitude_consensus = 1.0 / (1.0 + magnitude_std)
        else:
            magnitude_consensus = 1.0
        
        return directional_consensus * 0.7 + magnitude_consensus * 0.3
    
    def learn_from_outcome(self, outcome, learning_context=None):
        """Enhanced learning with all four subsystems and comprehensive error handling"""
        try:
            # Validate outcome parameter first
            if not self._validate_outcome_parameter(outcome):
                logger.error(f"Invalid outcome parameter for learning: {outcome} (type: {type(outcome)})")
                return
            
            # Convert outcome to float for safe numeric operations
            numeric_outcome = self._safe_convert_outcome(outcome)
            
            # Use provided context or create default with consistent structure
            if learning_context is None:
                learning_context = {
                    'dna_sequence': 'DEFAULT_DNA_SEQUENCE',
                    'cycles_info': [{
                        'frequency': 1.0/60.0,
                        'amplitude': 0.1,
                        'phase': 0.0,
                        'period': 60,
                        'window_size': 64
                    }],
                    'market_state': {
                        'volatility': 0.02,
                        'price_momentum': 0.0,
                        'volume_momentum': 0.0,
                        'regime': 'ranging',
                        'regime_confidence': 0.5
                    },
                    'microstructure_signal': 0.0,
                    'is_bootstrap': False
                }
            
            # Extract intelligence data from learning context if available
            dna_sequence = learning_context.get('dna_sequence', '')
            cycles_info = learning_context.get('cycles_info', [])
            market_state = learning_context.get('market_state', {})
            microstructure_signal = learning_context.get('microstructure_signal', 0.0)
            is_bootstrap = learning_context.get('is_bootstrap', False)
            
            # Ensure consistent market state structure
            if not market_state:
                market_state = {
                    'volatility': 0.02,
                    'price_momentum': 0.0,
                    'volume_momentum': 0.0,
                    'regime': 'ranging',
                    'regime_confidence': 0.5,
                    'time_of_day': 0.5,
                    'market_session': 'regular'
                }
            else:
                # Ensure all required keys exist
                required_keys = {
                    'volatility': 0.02,
                    'price_momentum': 0.0,
                    'volume_momentum': 0.0,
                    'regime': 'ranging',
                    'regime_confidence': 0.5,
                    'time_of_day': 0.5,
                    'market_session': 'regular'
                }
                for key, default_value in required_keys.items():
                    if key not in market_state:
                        market_state[key] = default_value
            
            # Learn in all subsystems using numeric outcome
            try:
                self.orchestrator.learn_from_outcome(numeric_outcome, learning_context)
            except Exception as e:
                logger.warning(f"Error in orchestrator learning: {e}")
            
            # Learn in microstructure engine
            try:
                if hasattr(self.microstructure_subsystem, 'learn_from_outcome'):
                    self.microstructure_subsystem.learn_from_outcome(numeric_outcome)
            except Exception as e:
                logger.warning(f"Error in microstructure learning: {e}")
            
            # Learn in enhanced dopamine subsystem with consistent context
            try:
                dopamine_context = {
                    'outcome': numeric_outcome,
                    'market_state': market_state,
                    'is_bootstrap': is_bootstrap,
                    'trade_data': None,  # Will be populated if available
                    'prediction_error': abs(numeric_outcome) if numeric_outcome != 0 else 0.1,
                    'confidence': learning_context.get('confidence', 0.5)
                }
                # Note: Dopamine learning is now handled by the trading agent's dopamine manager
                # Intelligence subsystems focus on providing clean signals only
                pass  # Dopamine processing moved to agent
            except Exception as e:
                logger.warning(f"Error in dopamine learning: {e}")
            
            # Update adaptation engine
            try:
                adaptation_context = {
                    'volatility': market_state.get('volatility', 0.02),
                    'predicted_confidence': 0.5  # Default confidence
                }
                self._get_adaptation_engine().update_from_outcome(numeric_outcome, adaptation_context)
                
                # Process adaptation event
                urgency = min(1.0, abs(numeric_outcome) * 2.0)
                self._get_adaptation_engine().process_market_event(
                    'trade_outcome',
                    {'pnl': numeric_outcome, 'learning_context': learning_context},
                urgency=urgency
            ) 
            except Exception as e:
                logger.warning(f"Error in adaptation engine learning: {e}")
            
            # Pattern learning
            try:
                pattern_id = self._create_pattern_id_from_context(learning_context, market_state)
                self.patterns[pattern_id].append(outcome)
                
                if len(self.patterns[pattern_id]) > 20:
                    self.patterns[pattern_id] = self.patterns[pattern_id][-20:]
            except Exception as e:
                logger.warning(f"Error in pattern learning: {e}")
            
            self.recent_outcomes.append(outcome)
            
        except Exception as e:
            logger.error(f"Error in enhanced learning from outcome: {e}")
    
    def _validate_outcome_parameter(self, outcome) -> bool:
        """Validate outcome parameter for learning operations"""
        try:
            # Check if outcome is a Trade object that needs numeric extraction
            if hasattr(outcome, 'pnl'):
                return True  # Trade object with PnL attribute
            
            # Check if outcome is already numeric
            if isinstance(outcome, (int, float)):
                import numpy as np
                return not (np.isnan(outcome) or np.isinf(outcome))
            
            # Check if outcome can be converted to numeric
            try:
                float(outcome)
                return True
            except (ValueError, TypeError):
                return False
                
        except Exception as e:
            logger.error(f"Error validating outcome parameter: {e}")
            return False
    
    def _safe_convert_outcome(self, outcome):
        """Safely convert Trade object or other outcome types to numeric value"""
        try:
            # Handle Trade objects - extract numeric PnL value
            if hasattr(outcome, 'pnl'):
                pnl_value = getattr(outcome, 'pnl', 0.0)
                try:
                    return float(pnl_value)
                except (ValueError, TypeError) as e:
                    logger.error(f"Error converting Trade.pnl to float: {e}, pnl_value={pnl_value}")
                    return 0.0
            
            # Handle other possible Trade attributes
            if hasattr(outcome, 'profit_loss'):
                try:
                    return float(getattr(outcome, 'profit_loss', 0.0))
                except (ValueError, TypeError):
                    return 0.0
                    
            if hasattr(outcome, 'realized_pnl'):
                try:
                    return float(getattr(outcome, 'realized_pnl', 0.0))
                except (ValueError, TypeError):
                    return 0.0
            
            # Handle already numeric values
            if isinstance(outcome, (int, float)):
                import numpy as np
                if np.isnan(outcome) or np.isinf(outcome):
                    logger.warning(f"Invalid numeric outcome: {outcome}, using 0.0")
                    return 0.0
                return float(outcome)
            
            # Try direct conversion
            try:
                numeric_value = float(outcome)
                import numpy as np
                if np.isnan(numeric_value) or np.isinf(numeric_value):
                    logger.warning(f"Converted outcome is invalid: {numeric_value}, using 0.0")
                    return 0.0
                return numeric_value
            except (ValueError, TypeError) as e:
                logger.error(f"Cannot convert outcome to numeric: {outcome} (type: {type(outcome)}), error: {e}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Critical error in safe outcome conversion: {e}")
            logger.error(f"Outcome type: {type(outcome)}, content: {str(outcome)[:100]}")
            return 0.0
    
    def _create_pattern_id_from_context(self, learning_context: Dict, market_state: Dict) -> str:
        """Create pattern ID from learning context and market state"""
        try:
            volatility = market_state.get('volatility', 0.02)
            price_momentum = market_state.get('price_momentum', 0.0)
            microstructure_signal = learning_context.get('microstructure_signal', 0.0)
            
            # Create simple pattern ID
            vol_bucket = int(volatility * 100) // 5  # Group volatility into buckets
            momentum_bucket = int(price_momentum * 100) // 10  # Group momentum into buckets
            micro_bucket = int(microstructure_signal * 100) // 10  # Group microstructure into buckets
            
            return f"pattern_v{vol_bucket}_m{momentum_bucket}_s{micro_bucket}"
        except Exception as e:
            logger.warning(f"Error creating pattern ID: {e}")
            return "pattern_default"
    
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
            microstructure_result.get('microstructure_signal', 0),
            microstructure_result.get('regime_adjusted_signal', 0),
            # Additional features for enhanced tensor
            market_features.get('margin_utilization', 0),
            market_features.get('buying_power_ratio', 1.0),
            market_features.get('daily_pnl_pct', 0.0)
        ]
        
        # Pad to 64 dimensions
        while len(features) < 64:
            features.append(0.0)
        
        return torch.tensor(features[:64], dtype=torch.float32, device='cpu')
    
    def _recognize_patterns(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        """Enhanced pattern recognition with all subsystems"""
        if len(prices) < 10:
            return 0.0
            
        trend = (prices[-1] - prices[-10]) / prices[-10] if prices[-10] != 0 else 0
        vol_trend = (volumes[-1] - np.mean(volumes[-5:])) / np.mean(volumes[-5:]) if np.mean(volumes[-5:]) > 0 else 0
        
        # Enhanced pattern strength with volume confirmation
        if (trend > 0 and vol_trend > 0) or (trend < 0 and vol_trend < 0):
            pattern_strength = abs(trend) * (1 + abs(vol_trend))
        else:
            pattern_strength = abs(trend) * 0.5
        
        # Add volatility pattern detection using proper annualized volatility
        if len(prices) > 1:
            price_returns = np.diff(prices) / prices[:-1]
            volatility = np.std(price_returns) * np.sqrt(252)  # Annualized volatility
        else:
            volatility = 0.02
        if volatility > 0.04:  # High volatility pattern
            pattern_strength *= 0.8  # Reduce pattern confidence in high volatility
        
        return np.tanh(pattern_strength * 10)
    
    def _create_pattern_id(self, features: Features) -> str:
        """Create pattern ID for legacy compatibility with enhanced features"""
        momentum_bucket = int(features.price_momentum * 5) + 5
        position_bucket = int(features.price_position * 4)
        vol_bucket = int(features.volatility * 100) // 10
        
        # Add subsystem information to pattern ID
        dna_bucket = int((features.dna_signal + 1) * 5)
        micro_bucket = int((features.microstructure_signal + 1) * 5)
        
        return f"p{momentum_bucket}_{position_bucket}_{vol_bucket}_d{dna_bucket}_m{micro_bucket}"
    
    def _alert_timestamp_failure(self, timestamp):
        """Alert system to timestamp validation failure"""
        try:
            # Log critical alert
            logger.critical(f"SYSTEM ALERT: Timestamp validation failed - {timestamp}")
            
            # Track timestamp failures for pattern analysis
            if not hasattr(self, 'timestamp_failures'):
                self.timestamp_failures = []
            
            self.timestamp_failures.append({
                'timestamp': timestamp,
                'timestamp_type': type(timestamp).__name__,
                'time': time.time()
            })
            
            # Keep only last 100 failures
            self.timestamp_failures = self.timestamp_failures[-100:]
            
        except Exception as e:
            logger.error(f"Failed to log timestamp failure alert: {e}")
    
    def _alert_insufficient_data(self, price_count: int, volume_count: int):
        """Alert system to insufficient data for feature extraction"""
        try:
            logger.critical(f"SYSTEM ALERT: Insufficient data - Prices: {price_count}, Volumes: {volume_count}")
            
            # Track data insufficiency patterns
            if not hasattr(self, 'data_insufficiency_events'):
                self.data_insufficiency_events = []
            
            self.data_insufficiency_events.append({
                'price_count': price_count,
                'volume_count': volume_count,
                'time': time.time()
            })
            
            # Keep only last 100 events
            self.data_insufficiency_events = self.data_insufficiency_events[-100:]
            
        except Exception as e:
            logger.error(f"Failed to log data insufficiency alert: {e}")
    
    def _calculate_dynamic_signal_weights(self, signals: list, market_features: dict, consensus_strength: float) -> list:
        """Calculate dynamic weights based on signal quality, reliability, and recent performance"""
        try:
            # Initialize subsystem performance tracking if not exists
            if not hasattr(self, 'subsystem_performance'):
                self.subsystem_performance = {
                    'dna': {'correct_predictions': 0, 'total_predictions': 0, 'recent_accuracy': 0.5},
                    'temporal': {'correct_predictions': 0, 'total_predictions': 0, 'recent_accuracy': 0.5},
                    'immune': {'correct_predictions': 0, 'total_predictions': 0, 'recent_accuracy': 0.5},
                    'microstructure': {'correct_predictions': 0, 'total_predictions': 0, 'recent_accuracy': 0.5},
                    'dopamine': {'correct_predictions': 0, 'total_predictions': 0, 'recent_accuracy': 0.5}
                }
            
            # Signal quality metrics (strength, consistency, reliability)
            signal_names = ['dna', 'temporal', 'immune', 'microstructure', 'dopamine']
            base_weights = []
            
            for i, (signal, name) in enumerate(zip(signals, signal_names)):
                # 1. Signal strength component (0-1)
                signal_strength = min(1.0, abs(signal))
                
                # 2. Signal reliability based on recent performance (0-1)
                performance = self.subsystem_performance[name]
                if performance['total_predictions'] > 10:
                    reliability = performance['recent_accuracy']
                else:
                    reliability = 0.5  # Default for insufficient data
                
                # 3. Market condition suitability (0-1)
                suitability = self._calculate_subsystem_suitability(name, market_features)
                
                # 4. Signal consistency (how well it aligns with consensus)
                if consensus_strength > 0:
                    if signal * consensus_strength > 0:  # Same direction as consensus
                        consistency = min(1.0, abs(signal) / max(0.1, abs(consensus_strength)))
                    else:  # Contrarian signal
                        consistency = 0.3  # Lower weight for contrarian signals
                else:
                    consistency = 0.5  # Neutral when no consensus
                
                # Combine components with adaptive weights
                weight = (
                    signal_strength * 0.3 +      # Current signal strength
                    reliability * 0.4 +          # Historical performance  
                    suitability * 0.2 +          # Market condition fit
                    consistency * 0.1            # Consensus alignment
                )
                
                base_weights.append(max(0.05, weight))  # Minimum 5% weight
            
            # Normalize weights to sum to 1.0
            total_weight = sum(base_weights)
            if total_weight > 0:
                normalized_weights = [w / total_weight for w in base_weights]
            else:
                normalized_weights = [0.2] * 5  # Fallback to equal weights
            
            # Log weight distribution for monitoring
            if hasattr(self, 'weight_logging_counter'):
                self.weight_logging_counter += 1
            else:
                self.weight_logging_counter = 0
            
            # Log every 100 calculations to avoid spam
            if self.weight_logging_counter % 100 == 0:
                weight_str = ", ".join(f"{name}: {w:.3f}" for name, w in zip(signal_names, normalized_weights))
                logger.info(f"Dynamic signal weights - {weight_str}")
            
            return normalized_weights
            
        except Exception as e:
            logger.error(f"Error calculating dynamic signal weights: {e}")
            return [0.2, 0.2, 0.2, 0.2, 0.2]  # Fallback to equal weights
    
    def _calculate_subsystem_suitability(self, subsystem_name: str, market_features: dict) -> float:
        """Calculate how suitable a subsystem is for current market conditions"""
        try:
            volatility = market_features.get('volatility', 0.02)
            price_momentum = market_features.get('price_momentum', 0.0)
            volume_momentum = market_features.get('volume_momentum', 0.0)
            time_of_day = market_features.get('time_of_day', 0.5)
            
            # Each subsystem has different strengths in different market conditions
            if subsystem_name == 'dna':
                # DNA system works well in trending markets
                trend_strength = abs(price_momentum)
                return min(1.0, trend_strength * 3.0 + 0.3)
            
            elif subsystem_name == 'temporal':
                # Temporal system works well during specific time periods
                # Peak performance during market open/close
                time_factor = abs(time_of_day - 0.5) * 2  # Distance from midday
                return min(1.0, time_factor * 0.8 + 0.4)
            
            elif subsystem_name == 'immune':
                # Immune system excels in volatile, risky conditions
                return min(1.0, volatility * 20 + 0.2)
            
            elif subsystem_name == 'microstructure':
                # Microstructure works well with high volume activity
                volume_factor = abs(volume_momentum)
                return min(1.0, volume_factor * 2.0 + 0.3)
            
            elif subsystem_name == 'dopamine':
                # Dopamine system works consistently across conditions
                # Slight preference for moderate volatility
                vol_preference = 1.0 - abs(volatility - 0.03) * 10
                return max(0.3, min(1.0, vol_preference))
            
            else:
                return 0.5  # Default suitability
                
        except Exception as e:
            logger.error(f"Error calculating subsystem suitability for {subsystem_name}: {e}")
            return 0.5
    
    def _coordinate_subsystem_learning(self, learning_data: dict):
        """Coordinate concurrent learning phases across subsystems to prevent conflicts"""
        try:
            # Initialize learning coordination if not exists
            if not hasattr(self, 'learning_coordinator'):
                import threading
                self.learning_coordinator = {
                    'active_learners': set(),
                    'learning_queue': [],
                    'coordination_lock': threading.Lock(),
                    'max_concurrent_learners': 2,  # Limit concurrent learning
                    'learning_round': 0
                }
            
            coordinator = self.learning_coordinator
            
            with coordinator['coordination_lock']:
                # Determine which subsystems need learning updates
                subsystems_to_update = []
                
                # Priority-based learning scheduling
                learning_priorities = {
                    'dopamine': learning_data.get('trade_outcome', 0) * 2,  # Highest priority for trade outcomes
                    'immune': learning_data.get('risk_violation', 0) * 1.8,  # High priority for risk events
                    'dna': learning_data.get('pattern_match', 0) * 1.5,     # Medium-high for patterns
                    'temporal': learning_data.get('time_relevance', 0) * 1.2, # Medium for time-based
                    'microstructure': learning_data.get('volume_significance', 0) * 1.0 # Base priority
                }
                
                # Sort by priority (descending)
                sorted_subsystems = sorted(learning_priorities.items(), key=lambda x: x[1], reverse=True)
                
                # Schedule learning updates
                for subsystem_name, priority in sorted_subsystems:
                    if priority > 0.1:  # Only learn if significant priority
                        if len(coordinator['active_learners']) < coordinator['max_concurrent_learners']:
                            # Start learning immediately
                            coordinator['active_learners'].add(subsystem_name)
                            self._start_subsystem_learning(subsystem_name, learning_data, priority)
                        else:
                            # Queue for later
                            coordinator['learning_queue'].append((subsystem_name, learning_data, priority))
                
                coordinator['learning_round'] += 1
                
                # Log coordination status
                if coordinator['learning_round'] % 50 == 0:
                    active_str = ", ".join(coordinator['active_learners'])
                    queue_len = len(coordinator['learning_queue'])
                    logger.info(f"Learning coordination - Active: [{active_str}], Queued: {queue_len}")
            
        except Exception as e:
            logger.error(f"Error in subsystem learning coordination: {e}")
    
    def _start_subsystem_learning(self, subsystem_name: str, learning_data: dict, priority: float):
        """Start learning process for a specific subsystem"""
        try:
            # Update subsystem performance tracking with learning data
            if hasattr(self, 'subsystem_performance'):
                performance = self.subsystem_performance[subsystem_name]
                
                # Update prediction accuracy if we have outcome data
                if 'actual_outcome' in learning_data and 'predicted_outcome' in learning_data:
                    predicted = learning_data['predicted_outcome']
                    actual = learning_data['actual_outcome']
                    
                    # Simple accuracy: same direction = correct
                    correct = (predicted * actual) > 0
                    
                    performance['total_predictions'] += 1
                    if correct:
                        performance['correct_predictions'] += 1
                    
                    # Update recent accuracy (rolling average)
                    if performance['total_predictions'] > 0:
                        accuracy = performance['correct_predictions'] / performance['total_predictions']
                        # Weighted average: 80% historical, 20% current
                        performance['recent_accuracy'] = (
                            performance['recent_accuracy'] * 0.8 + accuracy * 0.2
                        )
                
                # Trigger actual subsystem learning
                self._trigger_subsystem_learning(subsystem_name, learning_data, priority)
                
        except Exception as e:
            logger.error(f"Error starting learning for {subsystem_name}: {e}")
        finally:
            # Remove from active learners when done
            if hasattr(self, 'learning_coordinator'):
                self.learning_coordinator['active_learners'].discard(subsystem_name)
                self._process_learning_queue()
    
    def _trigger_subsystem_learning(self, subsystem_name: str, learning_data: dict, priority: float):
        """Trigger actual learning update for specific subsystem"""
        try:
            if subsystem_name == 'dopamine':
                # Note: Dopamine subsystem learning is now handled by the agent's dopamine manager
                pass  # Dopamine processing moved to agent
            
            elif subsystem_name == 'immune' and hasattr(self.orchestrator, 'immune_subsystem'):
                # Update immune system
                risk_data = learning_data.get('risk_violation', 0)
                if risk_data > 0:
                    self.orchestrator.immune_subsystem.learn_from_threat(learning_data)
            
            elif subsystem_name == 'dna' and hasattr(self.orchestrator, 'dna_subsystem'):
                # Update DNA patterns
                pattern_data = learning_data.get('pattern_match', 0)
                if abs(pattern_data) > 0.1:
                    self.orchestrator.dna_subsystem.evolve_patterns(learning_data)
            
            elif subsystem_name == 'temporal' and hasattr(self.orchestrator, 'temporal_subsystem'):
                # Update temporal patterns
                time_data = learning_data.get('time_relevance', 0)
                if abs(time_data) > 0.1:
                    self.orchestrator.temporal_subsystem.update_cycles(learning_data)
            
            elif subsystem_name == 'microstructure':
                # Update microstructure analysis
                volume_data = learning_data.get('volume_significance', 0)
                if abs(volume_data) > 0.1:
                    if hasattr(self, 'microstructure_subsystem'):
                        self.microstructure_subsystem.learn_from_outcome(learning_data)
        
        except Exception as e:
            logger.error(f"Error triggering learning for {subsystem_name}: {e}")
    
    def _process_learning_queue(self):
        """Process queued learning requests when capacity becomes available"""
        try:
            coordinator = self.learning_coordinator
            
            while (len(coordinator['active_learners']) < coordinator['max_concurrent_learners'] and 
                   coordinator['learning_queue']):
                
                # Get highest priority item from queue
                coordinator['learning_queue'].sort(key=lambda x: x[2], reverse=True)
                subsystem_name, learning_data, priority = coordinator['learning_queue'].pop(0)
                
                # Start learning
                coordinator['active_learners'].add(subsystem_name)
                self._start_subsystem_learning(subsystem_name, learning_data, priority)
        
        except Exception as e:
            logger.error(f"Error processing learning queue: {e}")

    def _default_features(self) -> Features:
        """Enhanced default features with all subsystems"""
        return Features(
            price_momentum=0, volume_momentum=0, price_position=0.5, volatility=0,
            time_of_day=0.5, pattern_score=0, confidence=0, 
            # All five subsystem signals
            dna_signal=0, micro_signal=0, temporal_signal=0, immune_signal=0, 
            microstructure_signal=0.0, dopamine_signal=0.0, overall_signal=0,
            # Enhanced features
            regime_adjusted_signal=0.0, adaptation_quality=0.5,
            smart_money_flow=0.0, liquidity_depth=0.5, regime_confidence=0.5
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
        
        # Add microstructure patterns
        microstructure_features = self.microstructure_subsystem.get_microstructure_features()
        if isinstance(microstructure_features, dict):
            total += len(microstructure_features.get('patterns', {}))
        
        return total

    def get_stats(self) -> Dict:
        """Enhanced statistics with all four subsystems"""
        orchestrator_stats = self.orchestrator.get_comprehensive_stats()
        adaptation_stats = self._get_adaptation_engine().get_comprehensive_stats()
        microstructure_features = self.microstructure_subsystem.get_microstructure_features()
        
        # Extract subsystem counts
        dna_stats = orchestrator_stats.get('dna_evolution', {})
        immune_stats = orchestrator_stats.get('immune_system', {})
        temporal_cycles = orchestrator_stats.get('temporal_cycles', 0)
        
        # Count microstructure patterns - use bootstrap stats if available for consistency
        micro_patterns_count = 0
        if hasattr(self, 'bootstrap_stats') and 'microstructure_patterns_learned' in self.bootstrap_stats:
            # Use bootstrap stats for consistency with logged values
            micro_patterns_count = self.bootstrap_stats['microstructure_patterns_learned']
        elif isinstance(microstructure_features, dict):
            # Fallback to real-time pattern count
            micro_patterns_count = microstructure_features.get('pattern_count', 0)
        
        # Get dopamine subsystem stats
        dopamine_stats = 0
        try:
            # Note: Dopamine metrics are now provided by the agent's dopamine manager
            dopamine_metrics = {'status': 'moved_to_agent'}
            if isinstance(dopamine_metrics, dict) and dopamine_metrics.get('status') != 'insufficient_data':
                dopamine_stats = dopamine_metrics.get('total_updates', 0)
        except:
            dopamine_stats = 0
        
        return {
            'total_patterns': len(self.patterns),
            'recent_performance': np.mean(self.recent_outcomes) if self.recent_outcomes else 0,
            'pattern_count': sum(len(outcomes) for outcomes in self.patterns.values()),
            'historical_processed': self.historical_processed,
            'bootstrap_stats': self.bootstrap_stats,
            'subsystem_training_progress': self.subsystem_training_progress,
            # Individual subsystem stats - ALL 5 SUBSYSTEMS
            'dna_patterns': dna_stats.get('total_sequences', 0),
            'micro_patterns': micro_patterns_count,
            'temporal_patterns': temporal_cycles,
            'immune_patterns': immune_stats.get('total_antibodies', 0),
            'dopamine_patterns': dopamine_stats,
            # Detailed subsystem stats
            'orchestrator': orchestrator_stats,
            'adaptation': adaptation_stats,
            'microstructure': microstructure_features
        }
    
    def extract_features(self, data: MarketData) -> Features:
        """
        Enhanced and corrected feature extraction with all four subsystems.
        This version ensures data integrity and avoids fabricating data.
        """
        # --- Pre-computation Guards ---
        # Ensure we have enough data points to calculate features meaningfully.
        if len(data.prices_1m) < 20 or len(data.volumes_1m) < 20:
            logger.error(f"CRITICAL: Insufficient data for feature extraction. Prices: {len(data.prices_1m)}, Volumes: {len(data.volumes_1m)}")
            logger.error("ALERT: Returning default features due to insufficient market data")
            self._alert_insufficient_data(len(data.prices_1m), len(data.volumes_1m))
            return self._default_features()

        # Use the most recent 20 data points for stable calculations.
        prices = np.array(data.prices_1m[-20:])
        volumes = np.array(data.volumes_1m[-20:])

        # --- Basic Market Features ---
        # Safely calculate momentum, position, and volatility.
        short_ma = np.mean(prices[-5:])
        long_ma = np.mean(prices)
        price_momentum = (short_ma - long_ma) / long_ma if long_ma > 0 else 0

        recent_vol = np.mean(volumes[-5:])
        avg_vol = np.mean(volumes)
        volume_momentum = (recent_vol - avg_vol) / avg_vol if avg_vol > 0 else 0

        high = np.max(prices)
        low = np.min(prices)
        price_position = (prices[-1] - low) / (high - low) if high > low else 0.5

        # Calculate proper annualized volatility from price returns
        if len(prices) > 1:
            price_returns = np.diff(prices) / prices[:-1]
            volatility = np.std(price_returns) * np.sqrt(252)  # Annualized volatility
        else:
            volatility = 0.02

        # --- Timestamp and Time-based Features ---
        # Safely parse the timestamp from the data source. Avoid fallbacks to current time.
        try:
            # The primary timestamp should be a standard Unix timestamp.
            now = datetime.fromtimestamp(data.timestamp)
            time_of_day = (now.hour * 60 + now.minute) / 1440
        except (OSError, ValueError, OverflowError):
            logger.error(f"CRITICAL: Invalid timestamp format received: {data.timestamp}. Cannot calculate time features.")
            logger.error(f"Timestamp type: {type(data.timestamp)}, Value: {data.timestamp}")
            logger.error("ALERT: Returning default features due to timestamp validation failure")
            # CRITICAL: Alert system to timestamp validation failure
            self._alert_timestamp_failure(data.timestamp)
            return self._default_features()

        # --- Multi-Timeframe Analysis ---
        # Enhanced analysis with higher timeframes for better trend detection
        tf_analysis = self._analyze_multi_timeframe(data, price_momentum)
        
        # --- Subsystem Feature Preparation ---
        # This dictionary will be passed to the subsystems.
        market_features = {
            'volatility': volatility,
            'price_momentum': price_momentum,
            'volume_momentum': volume_momentum,
            'time_of_day': time_of_day,
            'price_position': price_position,
            'account_balance': data.account_balance,
            'margin_utilization': data.margin_utilization,
            'buying_power_ratio': data.buying_power_ratio,
            'daily_pnl_pct': data.daily_pnl_pct,
            'regime': self._classify_market_regime(volatility, price_momentum, volatility * 10),
            # Enhanced with all timeframe features
            'trend_1m': tf_analysis['trend_1m'],
            'trend_5m': tf_analysis['trend_5m'],
            'trend_15m': tf_analysis['trend_15m'],
            'trend_1h': tf_analysis['trend_1h'],
            'trend_4h': tf_analysis['trend_4h'],
            'trend_alignment': tf_analysis['trend_alignment'],
            'higher_tf_bias': tf_analysis['higher_tf_bias'],
            'volatility_1m': tf_analysis['volatility_1m'],
            'volatility_5m': tf_analysis['volatility_5m'],
            'volatility_15m': tf_analysis['volatility_15m'],
            'volatility_1h': tf_analysis['volatility_1h'],
            'volatility_4h': tf_analysis['volatility_4h']
        }
        
        # IMPORTANT: Do not fabricate timestamps for temporal analysis.
        # The temporal subsystem requires real timestamps if available.
        # If not, it should gracefully handle an empty list.
        timestamps = [] # Pass empty list, as MarketData doesn't provide a list of timestamps.

        # --- Subsystem Processing ---
        # Wrap in a try-except block to catch any errors during signal processing.
        try:
            # 1. DNA Subsystem
            # Use safe data type conversion
            safe_prices = DataTypeNormalizer.ensure_list(data.prices_1m)
            safe_volumes = DataTypeNormalizer.ensure_list(data.volumes_1m)
            
            dna_sequence = self.orchestrator.dna_subsystem.encode_market_state(
                safe_prices, safe_volumes, volatility, price_momentum
            )
            dna_signal = self.orchestrator.dna_subsystem.analyze_sequence(dna_sequence) if dna_sequence else 0.0
            
            # 2. Temporal Subsystem
            temporal_signal = self.orchestrator.temporal_subsystem.analyze_cycles(safe_prices, timestamps)
            
            # 3. Immune Subsystem
            immune_signal = self.orchestrator.immune_subsystem.detect_threats(market_features)
            
            # 4. Microstructure Subsystem
            microstructure_result = self.microstructure_subsystem.analyze_market_state(
                data.prices_1m, data.volumes_1m
            )
            microstructure_signal = microstructure_result.get('microstructure_signal', 0.0)
            regime_adjusted_signal = microstructure_result.get('regime_adjusted_signal', 0.0)
            
            # Extract additional microstructure features
            order_flow = microstructure_result.get('order_flow', {})
            regime_state = microstructure_result.get('regime_state', {})
            
            smart_money_flow = order_flow.get('smart_money_flow', 0.0)
            liquidity_depth = order_flow.get('liquidity_depth', 0.5)
            regime_confidence = regime_state.get('confidence', 0.5)
            
            # 5. Dopamine Subsystem - P&L-based reward signal
            dopamine_market_data = {
                'unrealized_pnl': getattr(data, 'unrealized_pnl', 0.0),
                'daily_pnl': getattr(data, 'daily_pnl', 0.0),
                'open_positions': getattr(data, 'open_positions', 0.0),
                'current_price': data.prices_1m[-1] if data.prices_1m else 0.0
            }
            # Note: Dopamine signals are now generated by the agent's dopamine manager
            dopamine_signal = 0.0  # Placeholder - actual dopamine processing is in agent

        except Exception as e:
            logger.error(f"Error during subsystem processing in extract_features: {e}")
            return self._default_features()

        # --- Dynamic Signal Quality Weighting ---
        subsystem_signals = [dna_signal, temporal_signal, immune_signal, microstructure_signal, dopamine_signal]
        consensus_strength = self._calculate_enhanced_consensus(subsystem_signals)
        
        # Calculate dynamic weights based on signal quality, reliability, and recent performance
        weights = self._calculate_dynamic_signal_weights(
            subsystem_signals, 
            market_features, 
            consensus_strength
        )
        overall_signal = sum(signal * weight for signal, weight in zip(subsystem_signals, weights))
        
        signal_strength = sum(abs(s) for s in subsystem_signals)
        pattern_score = self._recognize_patterns(prices, volumes)
        
        confidence = max(0.1, min(1.0, 
            signal_strength * 0.4 + 
            consensus_strength * 0.3 + 
            pattern_score * 0.2 + 
            regime_confidence * 0.1
        ))

        # --- Real-time Adaptation ---
        adaptation_quality = 0.5 # Default value
        try:
            adaptation_context = {
                'volatility': volatility,
                'trend_strength': abs(price_momentum),
                'volume_regime': min(1.0, volume_momentum + 0.5),
                'time_of_day': time_of_day,
                'regime_confidence': regime_confidence
            }
            feature_tensor = self._create_feature_tensor(market_features, 
                                                       {'dna_signal': dna_signal, 'temporal_signal': temporal_signal, 
                                                        'immune_signal': immune_signal, 'overall_signal': overall_signal},
                                                       microstructure_result)
            adaptation_decision = self._get_adaptation_engine().get_adaptation_decision(feature_tensor, adaptation_context)
            adaptation_quality = adaptation_decision.get('adaptation_quality', 0.5)
        except Exception as e:
            logger.error(f"Adaptation engine failed during feature extraction: {e}")

        # --- Final Feature Construction ---
        return Features(
            price_momentum=price_momentum,
            volume_momentum=volume_momentum,
            price_position=price_position,
            volatility=volatility,
            time_of_day=time_of_day,
            pattern_score=pattern_score,
            confidence=confidence,
            dna_signal=dna_signal,
            micro_signal=microstructure_signal,
            temporal_signal=temporal_signal,
            immune_signal=immune_signal,
            microstructure_signal=microstructure_signal,
            dopamine_signal=dopamine_signal,
            overall_signal=overall_signal,
            regime_adjusted_signal=regime_adjusted_signal,
            adaptation_quality=adaptation_quality,
            smart_money_flow=smart_money_flow,
            liquidity_depth=liquidity_depth,
            regime_confidence=regime_confidence,
            # Multi-timeframe features - all timeframes
            trend_1m=tf_analysis['trend_1m'],
            trend_5m=tf_analysis['trend_5m'],
            trend_15m=tf_analysis['trend_15m'],
            trend_1h=tf_analysis['trend_1h'],
            trend_4h=tf_analysis['trend_4h'],
            trend_alignment=tf_analysis['trend_alignment'],
            higher_tf_bias=tf_analysis['higher_tf_bias'],
            volatility_1m=tf_analysis['volatility_1m'],
            volatility_5m=tf_analysis['volatility_5m'],
            volatility_15m=tf_analysis['volatility_15m'],
            volatility_1h=tf_analysis['volatility_1h'],
            volatility_4h=tf_analysis['volatility_4h']
        )
    
    def _analyze_multi_timeframe(self, data: MarketData, price_momentum: float) -> Dict:
        """
        Enhanced multi-timeframe analysis for better trend detection and market bias
        """
        try:
            # Initialize default values for all timeframes
            analysis = {
                'trend_1m': 0.0,
                'trend_5m': 0.0,
                'trend_15m': 0.0,
                'trend_1h': 0.0,
                'trend_4h': 0.0,
                'trend_alignment': 0.0,
                'higher_tf_bias': 0.0,
                'volatility_1m': 0.02,
                'volatility_5m': 0.02,
                'volatility_15m': 0.02,
                'volatility_1h': 0.02,
                'volatility_4h': 0.02
            }
            
            # 1M Timeframe Analysis - REDUCED requirement from 20 to 10
            if hasattr(data, 'prices_1m') and len(data.prices_1m) >= 10:
                prices_1m = np.array(data.prices_1m[-min(20, len(data.prices_1m)):])
                
                # 1M trend strength (percentage change)
                m1_short_ma = np.mean(prices_1m[-min(5, len(prices_1m)):])
                m1_long_ma = np.mean(prices_1m)
                analysis['trend_1m'] = (m1_short_ma - m1_long_ma) / m1_long_ma if m1_long_ma > 0 else 0.0
                
                # 1M volatility - proper annualized calculation
                if len(prices_1m) > 1:
                    price_returns_1m = np.diff(prices_1m) / prices_1m[:-1]
                    analysis['volatility_1m'] = np.std(price_returns_1m) * np.sqrt(252 * 24 * 60)  # Annualized from 1-minute returns
                else:
                    analysis['volatility_1m'] = 0.02
                
                logger.debug(f"1M trend calculated: {analysis['trend_1m']:.4f} from {len(prices_1m)} bars")
            elif hasattr(data, 'prices_1m') and len(data.prices_1m) >= 3:
                # Fallback with minimal data
                prices_1m = np.array(data.prices_1m[-3:])
                analysis['trend_1m'] = (prices_1m[-1] - prices_1m[0]) / prices_1m[0] if prices_1m[0] > 0 else 0.0
                logger.debug(f"1M trend (fallback): {analysis['trend_1m']:.4f} from {len(prices_1m)} bars")
            
            # 5M Timeframe Analysis - REDUCED requirement from 15 to 5
            if hasattr(data, 'prices_5m') and len(data.prices_5m) >= 5:
                prices_5m = np.array(data.prices_5m[-min(15, len(data.prices_5m)):])
                
                # 5M trend strength
                m5_short_ma = np.mean(prices_5m[-min(3, len(prices_5m)):])
                m5_long_ma = np.mean(prices_5m)
                analysis['trend_5m'] = (m5_short_ma - m5_long_ma) / m5_long_ma if m5_long_ma > 0 else 0.0
                
                # 5M volatility - proper annualized calculation
                if len(prices_5m) > 1:
                    price_returns_5m = np.diff(prices_5m) / prices_5m[:-1]
                    analysis['volatility_5m'] = np.std(price_returns_5m) * np.sqrt(252 * 24 * 12)  # Annualized from 5-minute returns
                else:
                    analysis['volatility_5m'] = 0.02
                
                logger.debug(f"5M trend calculated: {analysis['trend_5m']:.4f} from {len(prices_5m)} bars")
            elif hasattr(data, 'prices_5m') and len(data.prices_5m) >= 2:
                # Fallback with minimal data
                prices_5m = np.array(data.prices_5m[-2:])
                analysis['trend_5m'] = (prices_5m[-1] - prices_5m[0]) / prices_5m[0] if prices_5m[0] > 0 else 0.0
                logger.debug(f"5M trend (fallback): {analysis['trend_5m']:.4f} from {len(prices_5m)} bars")
            
            # 15M Timeframe Analysis - REDUCED requirement from 12 to 4
            if hasattr(data, 'prices_15m') and len(data.prices_15m) >= 4:
                prices_15m = np.array(data.prices_15m[-min(12, len(data.prices_15m)):])
                
                # 15M trend strength
                m15_short_ma = np.mean(prices_15m[-min(3, len(prices_15m)):])
                m15_long_ma = np.mean(prices_15m)
                analysis['trend_15m'] = (m15_short_ma - m15_long_ma) / m15_long_ma if m15_long_ma > 0 else 0.0
                
                # 15M volatility - proper annualized calculation
                if len(prices_15m) > 1:
                    price_returns_15m = np.diff(prices_15m) / prices_15m[:-1]
                    analysis['volatility_15m'] = np.std(price_returns_15m) * np.sqrt(252 * 24 * 4)  # Annualized from 15-minute returns
                else:
                    analysis['volatility_15m'] = 0.02
                
                logger.debug(f"15M trend calculated: {analysis['trend_15m']:.4f} from {len(prices_15m)} bars")
            elif hasattr(data, 'prices_15m') and len(data.prices_15m) >= 2:
                # Fallback with minimal data
                prices_15m = np.array(data.prices_15m[-2:])
                analysis['trend_15m'] = (prices_15m[-1] - prices_15m[0]) / prices_15m[0] if prices_15m[0] > 0 else 0.0
                logger.debug(f"15M trend (fallback): {analysis['trend_15m']:.4f} from {len(prices_15m)} bars")
            
            # 1H Timeframe Analysis - REDUCED requirement from 10 to 3
            if hasattr(data, 'prices_1h') and len(data.prices_1h) >= 3:
                prices_1h = np.array(data.prices_1h[-min(10, len(data.prices_1h)):])
                
                # 1H trend strength
                h1_short_ma = np.mean(prices_1h[-min(3, len(prices_1h)):])
                h1_long_ma = np.mean(prices_1h)
                analysis['trend_1h'] = (h1_short_ma - h1_long_ma) / h1_long_ma if h1_long_ma > 0 else 0.0
                
                # 1H volatility - proper annualized calculation
                if len(prices_1h) > 1:
                    price_returns_1h = np.diff(prices_1h) / prices_1h[:-1]
                    analysis['volatility_1h'] = np.std(price_returns_1h) * np.sqrt(252 * 24)  # Annualized from 1-hour returns
                else:
                    analysis['volatility_1h'] = 0.02
                
                logger.debug(f"1H trend calculated: {analysis['trend_1h']:.4f} from {len(prices_1h)} bars")
            elif hasattr(data, 'prices_1h') and len(data.prices_1h) >= 2:
                # Fallback with minimal data
                prices_1h = np.array(data.prices_1h[-2:])
                analysis['trend_1h'] = (prices_1h[-1] - prices_1h[0]) / prices_1h[0] if prices_1h[0] > 0 else 0.0
                logger.debug(f"1H trend (fallback): {analysis['trend_1h']:.4f} from {len(prices_1h)} bars")
            else:
                # Use 15M data to estimate 1H trend if no 1H data available
                if hasattr(data, 'prices_15m') and len(data.prices_15m) >= 4:
                    prices_15m = np.array(data.prices_15m[-4:])
                    analysis['trend_1h'] = (np.mean(prices_15m[-2:]) - np.mean(prices_15m[:2])) / np.mean(prices_15m[:2]) if np.mean(prices_15m[:2]) > 0 else 0.0
                    logger.debug(f"1H trend (15M estimate): {analysis['trend_1h']:.4f}")
            
            # 4H Timeframe Analysis - REDUCED requirement from 6 to 2
            if hasattr(data, 'prices_4h') and len(data.prices_4h) >= 2:
                prices_4h = np.array(data.prices_4h[-min(6, len(data.prices_4h)):])
                
                # 4H trend strength
                h4_short_ma = np.mean(prices_4h[-min(2, len(prices_4h)):])
                h4_long_ma = np.mean(prices_4h)
                analysis['trend_4h'] = (h4_short_ma - h4_long_ma) / h4_long_ma if h4_long_ma > 0 else 0.0
                
                # 4H volatility - proper annualized calculation
                if len(prices_4h) > 1:
                    price_returns_4h = np.diff(prices_4h) / prices_4h[:-1]
                    analysis['volatility_4h'] = np.std(price_returns_4h) * np.sqrt(252 * 6)  # Annualized from 4-hour returns
                else:
                    analysis['volatility_4h'] = 0.02
                
                logger.debug(f"4H trend calculated: {analysis['trend_4h']:.4f} from {len(prices_4h)} bars")
            elif hasattr(data, 'prices_4h') and len(data.prices_4h) >= 1:
                # Single bar - no trend calculation possible, keep default
                logger.debug(f"4H trend: insufficient data ({len(data.prices_4h)} bars)")
            else:
                # Use 1H data to estimate 4H trend if no 4H data available
                if hasattr(data, 'prices_1h') and len(data.prices_1h) >= 4:
                    prices_1h = np.array(data.prices_1h[-4:])
                    analysis['trend_4h'] = (np.mean(prices_1h[-2:]) - np.mean(prices_1h[:2])) / np.mean(prices_1h[:2]) if np.mean(prices_1h[:2]) > 0 else 0.0
                    logger.debug(f"4H trend (1H estimate): {analysis['trend_4h']:.4f}")
                elif hasattr(data, 'prices_15m') and len(data.prices_15m) >= 8:
                    # Use 15M data to estimate 4H trend (rough approximation)
                    prices_15m = np.array(data.prices_15m[-8:])
                    analysis['trend_4h'] = (np.mean(prices_15m[-4:]) - np.mean(prices_15m[:4])) / np.mean(prices_15m[:4]) if np.mean(prices_15m[:4]) > 0 else 0.0
                    logger.debug(f"4H trend (15M estimate): {analysis['trend_4h']:.4f}")
            
            # Multi-Timeframe Alignment
            # Calculate how well all 5 timeframes agree on direction
            trend_1m = analysis['trend_1m']
            trend_5m = analysis['trend_5m']
            trend_15m = analysis['trend_15m']
            trend_1h = analysis['trend_1h']
            trend_4h = analysis['trend_4h']
            
            # All timeframes for comprehensive alignment
            trends = [trend_1m, trend_5m, trend_15m, trend_1h, trend_4h]
            positive_trends = sum(1 for t in trends if t > 0.0005)  # 0.05% threshold
            negative_trends = sum(1 for t in trends if t < -0.0005)  # -0.05% threshold
            
            # Enhanced alignment scoring with all 5 timeframes
            if positive_trends >= 4 and negative_trends == 0:
                analysis['trend_alignment'] = 0.9  # Very strong bullish alignment
            elif negative_trends >= 4 and positive_trends == 0:
                analysis['trend_alignment'] = -0.9  # Very strong bearish alignment
            elif positive_trends >= 3 and negative_trends <= 1:
                analysis['trend_alignment'] = 0.6  # Strong bullish alignment
            elif negative_trends >= 3 and positive_trends <= 1:
                analysis['trend_alignment'] = -0.6  # Strong bearish alignment
            elif positive_trends > negative_trends:
                analysis['trend_alignment'] = 0.3  # Weak bullish alignment
            elif negative_trends > positive_trends:
                analysis['trend_alignment'] = -0.3  # Weak bearish alignment
            else:
                analysis['trend_alignment'] = 0.0  # No clear alignment
            
            # Higher Timeframe Bias (4H has most weight, then 1H, 15m, 5m, 1m)
            # This gives the "big picture" market direction weighted by timeframe importance
            weights = [0.05, 0.1, 0.2, 0.3, 0.35]  # 1m, 5m, 15m, 1h, 4h weights
            analysis['higher_tf_bias'] = np.average(trends, weights=weights)
            
            # Log multi-timeframe analysis for debugging
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Multi-TF Analysis: 1m={trend_1m:.4f}, 5m={trend_5m:.4f}, 15m={trend_15m:.4f}, "
                           f"1H={trend_1h:.4f}, 4H={trend_4h:.4f}, Alignment={analysis['trend_alignment']:.3f}, "
                           f"Bias={analysis['higher_tf_bias']:.4f}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in multi-timeframe analysis: {e}")
            # Return safe defaults for all timeframes
            return {
                'trend_1m': 0.0,
                'trend_5m': 0.0,
                'trend_15m': 0.0,
                'trend_1h': 0.0,
                'trend_4h': 0.0,
                'trend_alignment': 0.0,
                'higher_tf_bias': 0.0,
                'volatility_1m': 0.02,
                'volatility_5m': 0.02,
                'volatility_15m': 0.02,
                'volatility_1h': 0.02,
                'volatility_4h': 0.02
            }
    
    def analyze_market(self, historical_context, market_features):
        """
        Analyze market data and return comprehensive intelligence signals
        
        Args:
            market_data: Current market data
            features: Extracted market features
            
        Returns:
            Dictionary containing intelligence analysis results
        """
        try:
            # Extract price data from historical context
            prices = historical_context.get('prices', [])
            volumes = historical_context.get('volumes', [])
            timestamps = historical_context.get('timestamps', [])
            
            current_price = prices[-1] if prices else 0.0
            current_volume = volumes[-1] if volumes else 0.0
            current_timestamp = timestamps[-1] if timestamps else 0.0
            prices_1m = prices or [current_price]
            
            # Basic market analysis
            price_momentum = market_features.get('price_momentum', 0.0)
            volatility = market_features.get('volatility', 0.02)
            volume_momentum = market_features.get('volume_momentum', 0.0)
            
            # Multi-timeframe analysis
            multi_tf_analysis = self._analyze_multi_timeframe_simple(prices, price_momentum)
            
            # Calculate individual subsystem signals (simplified for reliability)
            dna_signal_value = self._analyze_price_patterns(prices_1m)
            temporal_signal_value = self._analyze_temporal_patterns_simple(current_timestamp)
            immune_signal_value = self._analyze_risk_patterns_simple(current_price, volatility)
            microstructure_signal_value = max(-1.0, min(1.0, volume_momentum * 0.5 + (1.0 - volatility) * 0.5))
            # Note: Dopamine signals are now generated by the agent's dopamine manager  
            dopamine_signal = 0.0  # Placeholder - actual dopamine processing is in agent
            
            # Get dopamine signal value
            dopamine_signal_value = dopamine_signal.signal if hasattr(dopamine_signal, 'signal') else dopamine_signal
            
            # Calculate overall signal as weighted combination of all 5 subsystem signals
            overall_signal_value = (
                dna_signal_value * 0.2 +            # DNA pattern recognition
                temporal_signal_value * 0.2 +       # Temporal cycles  
                immune_signal_value * 0.2 +         # Risk assessment
                microstructure_signal_value * 0.2 + # Market microstructure
                dopamine_signal_value * 0.2         # Dopamine reward optimization
            )
            
            # Calculate individual confidence levels with dynamic range
            base_confidence = max(0.2, min(0.9, 1.0 - volatility * 4.0))
            
            # Each subsystem contributes to confidence based on signal strength and market conditions
            dna_confidence = max(0.15, min(0.95, base_confidence + abs(dna_signal_value) * 0.4))
            temporal_confidence = max(0.2, min(0.9, base_confidence + abs(temporal_signal_value) * 0.35))
            immune_confidence = max(0.25, min(0.85, base_confidence + (1.0 - abs(immune_signal_value)) * 0.3))
            microstructure_confidence = max(0.3, min(0.9, base_confidence + abs(microstructure_signal_value) * 0.25))
            dopamine_confidence = max(0.1, min(0.95, base_confidence + abs(dopamine_signal_value) * 0.45))
            
            # Calculate consensus strength between all 5 subsystem signals
            signal_values = [dna_signal_value, temporal_signal_value, immune_signal_value, microstructure_signal_value, dopamine_signal_value]
            consensus_strength = self._calculate_consensus_strength(signal_values)
            
            # Overall confidence as average of all 5 subsystems
            overall_confidence = (dna_confidence + temporal_confidence + immune_confidence + microstructure_confidence + dopamine_confidence) / 5.0
            
            # Create signal objects with value and confidence (expected format)
            class SignalObject:
                def __init__(self, value, confidence=0.5):
                    self.value = value
                    self.confidence = confidence
            
            # Generate comprehensive signals in expected format
            intelligence_signals = {
                # Core signals with expected .value and .confidence attributes
                'overall': SignalObject(overall_signal_value, overall_confidence),
                'dna': SignalObject(dna_signal_value, dna_confidence),
                'temporal': SignalObject(temporal_signal_value, temporal_confidence),
                'immune': SignalObject(immune_signal_value, immune_confidence),
                'microstructure': SignalObject(microstructure_signal_value, microstructure_confidence),
                'dopamine': SignalObject(dopamine_signal_value, dopamine_confidence),
                
                # Critical: Add consensus_strength to fix 0.000 issue
                'consensus_strength': consensus_strength,
                
                # Backward compatibility - also include flat values
                'overall_signal': overall_signal_value,
                'confidence': overall_confidence,
                'regime_confidence': multi_tf_analysis.get('trend_alignment', 0.5),
                'dna_signal': dna_signal_value,
                'temporal_signal': temporal_signal_value,
                'immune_signal': immune_signal_value,
                'microstructure_signal': microstructure_signal_value,
                'dopamine_signal': dopamine_signal_value,
                'regime_adjusted_signal': self._adjust_signal_for_regime(price_momentum, volatility),
                'adaptation_quality': 0.8,
                'pattern_score': abs(price_momentum) * 0.5 + (1.0 - volatility) * 0.5,
                'smart_money_flow': volume_momentum,
                'liquidity_depth': min(1.0, current_volume / 10000.0) if current_volume > 0 else 0.5,
                'multi_timeframe_analysis': multi_tf_analysis
            }
            
            return intelligence_signals
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error in market analysis: {e}")
            
            # Return safe default signals with proper SignalObject format
            class SignalObject:
                def __init__(self, value, confidence=0.5):
                    self.value = value
                    self.confidence = confidence
            
            return {
                # Core signals with expected .value and .confidence attributes
                'overall': SignalObject(0.0, 0.5),
                'dna': SignalObject(0.0, 0.5),
                'temporal': SignalObject(0.0, 0.5),
                'immune': SignalObject(0.0, 0.5),
                'microstructure': SignalObject(0.0, 0.5),
                'dopamine': SignalObject(0.0, 0.5),
                
                # Critical: Add consensus_strength for error case
                'consensus_strength': 0.5,
                
                # Backward compatibility - also include flat values
                'overall_signal': 0.0,
                'confidence': 0.5,
                'regime_confidence': 0.5,
                'dna_signal': 0.0,
                'temporal_signal': 0.0,
                'immune_signal': 0.0,
                'microstructure_signal': 0.0,
                'dopamine_signal': 0.0,
                'regime_adjusted_signal': 0.0,
                'adaptation_quality': 0.5,
                'pattern_score': 0.0,
                'smart_money_flow': 0.0,
                'liquidity_depth': 0.5,
                'multi_timeframe_analysis': {'trend_alignment': 0.0}
            }
    
    def _calculate_consensus_strength(self, signal_values):
        """Calculate consensus strength between subsystem signals"""
        try:
            if len(signal_values) < 2:
                return 1.0
            
            import numpy as np
            
            # Remove any invalid values
            valid_signals = [s for s in signal_values if not (np.isnan(s) or np.isinf(s))]
            
            if len(valid_signals) < 2:
                return 0.5
            
            # Calculate standard deviation and range
            signal_std = np.std(valid_signals)
            signal_range = max(valid_signals) - min(valid_signals) if len(valid_signals) > 1 else 1.0
            
            # Consensus is higher when signals are closer together
            if signal_range > 0:
                # Normalize standard deviation by range
                consensus = 1.0 - min(signal_std / (signal_range + 0.01), 1.0)
            else:
                consensus = 1.0  # Perfect consensus when all signals identical
            
            # Additional consensus check: count signals in same direction
            positive_signals = sum(1 for s in valid_signals if s > 0.05)
            negative_signals = sum(1 for s in valid_signals if s < -0.05)
            neutral_signals = len(valid_signals) - positive_signals - negative_signals
            
            # Directional consensus bonus
            max_direction = max(positive_signals, negative_signals, neutral_signals)
            directional_consensus = max_direction / len(valid_signals)
            
            # Combine statistical and directional consensus
            final_consensus = (consensus * 0.6) + (directional_consensus * 0.4)
            
            return float(np.clip(final_consensus, 0.0, 1.0))
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error calculating consensus strength: {e}")
            return 0.5
    
    def _analyze_multi_timeframe_simple(self, prices, price_momentum):
        """Simple multi-timeframe analysis using price history"""
        try:
            if len(prices) < 10:
                return {'trend_alignment': 0.0}
            
            # Calculate trend alignment from price momentum
            trend_alignment = max(-1.0, min(1.0, price_momentum * 2.0))
            return {'trend_alignment': trend_alignment}
        except Exception as e:
            logger.warning(f"Error in multi-timeframe analysis: {e}")
            return {'trend_alignment': 0.0}
    
    def _analyze_temporal_patterns_simple(self, timestamp):
        """Simple temporal pattern analysis using timestamp"""
        try:
            import time
            
            # Validate timestamp range (Unix timestamp should be positive and reasonable)
            if timestamp <= 0 or timestamp > 2147483647:  # Max 32-bit Unix timestamp
                timestamp = time.time()
            
            # Convert timestamp to datetime safely
            try:
                current_hour = time.localtime(timestamp).tm_hour
            except (OSError, ValueError) as e:
                logger.warning(f"Invalid timestamp {timestamp}, using current time: {e}")
                current_hour = time.localtime(time.time()).tm_hour
            
            # Market hours bias (9 AM - 4 PM typically active)
            if 9 <= current_hour <= 16:
                return 0.1  # Slight positive bias during market hours
            else:
                return -0.05  # Slight negative bias outside market hours
        except Exception as e:
            logger.warning(f"Error in temporal pattern analysis: {e}")
            return 0.0
    
    def _analyze_risk_patterns_simple(self, current_price, volatility):
        """Simple risk pattern analysis"""
        try:
            # Higher volatility = higher risk = negative signal
            risk_signal = max(-1.0, min(1.0, -volatility * 5.0))
            return risk_signal
        except Exception as e:
            logger.warning(f"Error in risk pattern analysis: {e}")
            return 0.0
    
    def _analyze_price_patterns_enhanced(self, prices):
        """Enhanced DNA-style price pattern analysis"""
        if len(prices) < 5:
            return 0.0
        
        try:
            import numpy as np
            
            # Convert to numpy array for analysis
            price_array = np.array(prices)
            
            # Multiple pattern detection
            signals = []
            
            # 1. Short-term momentum (last 3 bars)
            if len(prices) >= 3:
                short_momentum = (prices[-1] - prices[-3]) / prices[-3] if prices[-3] != 0 else 0.0
                signals.append(short_momentum * 15)  # Amplify small changes
            
            # 2. Medium-term trend (last 5-10 bars)
            if len(prices) >= 5:
                recent_prices = price_array[-5:]
                trend_slope = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
                trend_signal = trend_slope / np.mean(recent_prices) * 1000  # Normalize by price level
                signals.append(trend_signal)
            
            # 3. Volatility-adjusted momentum
            if len(prices) >= 5:
                returns = np.diff(price_array[-5:]) / price_array[-5:-1]
                if len(returns) > 0:
                    volatility = np.std(returns)
                    momentum = np.mean(returns)
                    vol_adj_momentum = momentum / (volatility + 0.0001) * 0.1  # Risk-adjusted momentum
                    signals.append(vol_adj_momentum)
            
            # Combine signals
            if signals:
                combined_signal = np.mean(signals)
                return float(np.clip(combined_signal, -1.0, 1.0))
            else:
                return 0.0
                
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error in enhanced price pattern analysis: {e}")
            return 0.0
    
    def _analyze_temporal_patterns_enhanced(self, timestamp, prices):
        """Enhanced temporal pattern analysis"""
        try:
            import time
            import numpy as np
            
            signals = []
            
            # 1. Market hours effect (existing logic but refined)
            try:
                current_hour = time.localtime(timestamp).tm_hour if timestamp > 0 else time.localtime().tm_hour
            except (OSError, ValueError):
                current_hour = time.localtime().tm_hour
            
            # More nuanced market hours signal
            if 9 <= current_hour <= 16:
                # Peak hours: 10-11 AM and 2-3 PM
                if current_hour in [10, 11, 14, 15]:
                    hour_signal = 0.15
                else:
                    hour_signal = 0.05
            elif 4 <= current_hour <= 9:  # Pre-market
                hour_signal = -0.02
            elif 16 <= current_hour <= 20:  # After hours
                hour_signal = -0.05
            else:  # Overnight
                hour_signal = -0.10
            
            signals.append(hour_signal)
            
            # 2. Price cycle analysis (if enough data)
            if len(prices) >= 10:
                recent_prices = np.array(prices[-10:])
                # Simple cycle detection using autocorrelation
                returns = np.diff(recent_prices) / recent_prices[:-1]
                if len(returns) >= 3:
                    # Check for oscillating pattern
                    oscillation = 0.0
                    for i in range(1, len(returns)):
                        if returns[i] * returns[i-1] < 0:  # Sign change
                            oscillation += abs(returns[i])
                    
                    cycle_signal = min(0.2, oscillation * 2)  # Oscillation strength
                    signals.append(cycle_signal)
            
            # Combine temporal signals
            return float(np.clip(np.mean(signals), -0.5, 0.5))
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error in enhanced temporal analysis: {e}")
            return 0.0
    
    def _analyze_risk_patterns_enhanced(self, current_price, volatility, volume_momentum):
        """Enhanced immune system risk analysis"""
        try:
            import numpy as np
            
            signals = []
            
            # 1. Volatility risk signal (refined)
            vol_risk = -volatility * 8.0  # Stronger volatility penalty
            signals.append(vol_risk)
            
            # 2. Volume divergence risk
            if abs(volume_momentum) > 0.1:
                # High volume activity can signal risk
                volume_risk = -abs(volume_momentum) * 0.5
                signals.append(volume_risk)
            
            # 3. Price level risk (extreme prices)
            if current_price > 0:
                # Simple risk assessment based on recent price action
                price_risk = 0.0
                if current_price > 25000:  # High Bitcoin price territory
                    price_risk = -0.05
                elif current_price < 20000:  # Low Bitcoin price territory  
                    price_risk = -0.03
                signals.append(price_risk)
            
            # Combine risk signals
            combined_risk = np.mean(signals) if signals else 0.0
            return float(np.clip(combined_risk, -1.0, 0.2))  # Mostly negative (risk), small positive allowed
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error in enhanced risk analysis: {e}")
            return 0.0
    
    def _analyze_microstructure_enhanced(self, volume_momentum, volatility, prices):
        """Enhanced microstructure analysis"""
        try:
            import numpy as np
            
            signals = []
            
            # 1. Volume-price relationship
            vol_signal = volume_momentum * 0.6  # Volume momentum base
            signals.append(vol_signal)
            
            # 2. Volatility regime signal
            if volatility < 0.01:  # Low volatility - potential breakout setup
                regime_signal = 0.1
            elif volatility > 0.05:  # High volatility - unstable
                regime_signal = -0.2
            else:  # Normal volatility
                regime_signal = 0.05
            signals.append(regime_signal)
            
            # 3. Price action quality
            if len(prices) >= 5:
                recent_prices = np.array(prices[-5:])
                price_changes = np.diff(recent_prices)
                
                # Smooth price action is positive
                price_smoothness = 1.0 - (np.std(price_changes) / (np.mean(np.abs(price_changes)) + 0.01))
                smoothness_signal = price_smoothness * 0.15
                signals.append(smoothness_signal)
            
            # Combine microstructure signals
            combined_signal = np.mean(signals) if signals else 0.0
            return float(np.clip(combined_signal, -1.0, 1.0))
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error in enhanced microstructure analysis: {e}")
            return 0.0
    
    def _analyze_price_patterns(self, prices):
        """Basic price pattern analysis - kept for backward compatibility"""
        if len(prices) < 3:
            return 0.0
        
        # Simple momentum pattern
        recent_change = (prices[-1] - prices[-3]) / prices[-3] if prices[-3] != 0 else 0.0
        return max(-1.0, min(1.0, recent_change * 10))
    
    def _analyze_temporal_patterns(self, market_data):
        """Basic temporal pattern analysis"""
        import time
        
        # Time-of-day effects
        current_hour = time.localtime(market_data.timestamp).tm_hour
        
        # Market hours bias (9 AM - 4 PM typically active)
        if 9 <= current_hour <= 16:
            return 0.1  # Slight positive bias during market hours
        else:
            return -0.05  # Slight negative bias during off hours
    
    def _analyze_risk_patterns(self, market_data, volatility):
        """Basic risk pattern analysis"""
        # High volatility = higher risk signal
        risk_signal = -volatility * 2.0  # Negative signal for high risk
        return max(-1.0, min(1.0, risk_signal))
    
    def _adjust_signal_for_regime(self, signal, volatility):
        """Adjust signal based on market regime"""
        if volatility > 0.05:  # High volatility regime
            return signal * 0.5  # Reduce signal strength
        elif volatility < 0.01:  # Low volatility regime
            return signal * 1.2  # Increase signal strength
        else:
            return signal  # Normal regime
    
    # ========================================
    # INTELLIGENCE SIGNAL GENERATION METHODS
    # ========================================
    
    def get_intelligence_signals(self, features: Features, market_data) -> List[IntelligenceSignal]:
        """
        Get intelligence signals from all 4 subsystems
        
        This is the main method for generating intelligence signals that the
        dopamine manager will process for trading decisions.
        
        Args:
            features: Processed market features
            market_data: Current market data
            
        Returns:
            List of intelligence signals from each subsystem
        """
        try:
            signals = []
            current_time = time.time()
            
            # Get DNA subsystem signal (pattern recognition)
            dna_signal = self._generate_dna_intelligence_signal(features, market_data, current_time)
            if dna_signal:
                signals.append(dna_signal)
            
            # Get Temporal subsystem signal (time-based analysis)
            temporal_signal = self._generate_temporal_intelligence_signal(features, market_data, current_time)
            if temporal_signal:
                signals.append(temporal_signal)
            
            # Get Immune subsystem signal (anomaly detection)
            immune_signal = self._generate_immune_intelligence_signal(features, market_data, current_time)
            if immune_signal:
                signals.append(immune_signal)
            
            # Get Microstructure subsystem signal (market microstructure)
            microstructure_signal = self._generate_microstructure_intelligence_signal(features, market_data, current_time)
            if microstructure_signal:
                signals.append(microstructure_signal)
            
            # Store signals for consensus tracking
            for signal in signals:
                self.recent_signals.append(signal)
            
            logger.debug(f"Generated {len(signals)} intelligence signals")
            return signals
            
        except Exception as e:
            logger.error(f"Error generating intelligence signals: {e}")
            return []
    
    def _generate_dna_intelligence_signal(self, features: Features, market_data, timestamp: float) -> Optional[IntelligenceSignal]:
        """Generate intelligence signal from DNA subsystem (pattern recognition)"""
        try:
            # Get DNA analysis from orchestrator
            dna_result = self.orchestrator.analyze_dna_patterns(features, market_data)
            
            if not dna_result or not isinstance(dna_result, dict):
                return None
            
            # Extract signal strength and confidence
            raw_signal_strength = dna_result.get('signal', 0.0)
            confidence = dna_result.get('confidence', 0.5)
            pattern_type = dna_result.get('pattern_type', 'unknown')
            
            # Normalize signal strength to [-1.0, 1.0] range using tanh
            signal_strength = float(np.tanh(raw_signal_strength))
            
            # Ensure bounds are strictly enforced
            signal_strength = max(-1.0, min(1.0, signal_strength))
            
            # Determine direction based on signal strength
            if signal_strength > self.signal_generation_config['signal_threshold']:
                direction = 'bullish'
            elif signal_strength < -self.signal_generation_config['signal_threshold']:
                direction = 'bearish'
            else:
                direction = 'neutral'
            
            # Only generate signals above confidence threshold
            if confidence < self.signal_generation_config['confidence_threshold']:
                return None
            
            return create_intelligence_signal(
                signal_type=IntelligenceSignalType.PATTERN_RECOGNITION,
                strength=signal_strength,
                confidence=confidence,
                direction=direction,
                timeframe=getattr(features, 'timeframe', '1m'),
                subsystem_id='dna',
                pattern_type=pattern_type,
                analysis_source='dna_sequences'
            )
            
        except Exception as e:
            logger.error(f"Error generating DNA intelligence signal: {e}")
            return None
    
    def _generate_temporal_intelligence_signal(self, features: Features, market_data, timestamp: float) -> Optional[IntelligenceSignal]:
        """Generate intelligence signal from Temporal subsystem (time-based analysis)"""
        try:
            # Get temporal analysis from orchestrator
            temporal_result = self.orchestrator.analyze_temporal_cycles(features, market_data)
            
            if not temporal_result or not isinstance(temporal_result, dict):
                return None
            
            raw_signal_strength = temporal_result.get('signal', 0.0)
            confidence = temporal_result.get('confidence', 0.5)
            cycle_phase = temporal_result.get('cycle_phase', 'unknown')
            
            # Normalize signal strength to [-1.0, 1.0] range using tanh
            signal_strength = float(np.tanh(raw_signal_strength))
            
            # Ensure bounds are strictly enforced
            signal_strength = max(-1.0, min(1.0, signal_strength))
            
            # Determine direction
            if signal_strength > self.signal_generation_config['signal_threshold']:
                direction = 'bullish'
            elif signal_strength < -self.signal_generation_config['signal_threshold']:
                direction = 'bearish'
            else:
                direction = 'neutral'
            
            if confidence < self.signal_generation_config['confidence_threshold']:
                return None
            
            return create_intelligence_signal(
                signal_type=IntelligenceSignalType.TEMPORAL_ANALYSIS,
                strength=signal_strength,
                confidence=confidence,
                direction=direction,
                timeframe=getattr(features, 'timeframe', '1m'),
                subsystem_id='temporal',
                cycle_phase=cycle_phase,
                analysis_source='temporal_cycles'
            )
            
        except Exception as e:
            logger.error(f"Error generating Temporal intelligence signal: {e}")
            return None
    
    def _generate_immune_intelligence_signal(self, features: Features, market_data, timestamp: float) -> Optional[IntelligenceSignal]:
        """Generate intelligence signal from Immune subsystem (anomaly detection)"""
        try:
            # Get immune analysis from orchestrator
            immune_result = self.orchestrator.analyze_immune_threats(features, market_data)
            
            if not immune_result or not isinstance(immune_result, dict):
                return None
            
            raw_signal_strength = immune_result.get('signal', 0.0)
            confidence = immune_result.get('confidence', 0.5)
            threat_level = immune_result.get('threat_level', 'none')
            
            # Normalize signal strength to [-1.0, 1.0] range using tanh
            signal_strength = float(np.tanh(raw_signal_strength))
            
            # Ensure bounds are strictly enforced
            signal_strength = max(-1.0, min(1.0, signal_strength))
            
            # Immune signals are typically bearish (detecting threats) or neutral
            if signal_strength < -self.signal_generation_config['signal_threshold']:
                direction = 'bearish'  # Threat detected
            elif abs(signal_strength) < self.signal_generation_config['signal_threshold']:
                direction = 'neutral'  # No threat
            else:
                direction = 'neutral'  # Conservative approach for immune signals
            
            if confidence < self.signal_generation_config['confidence_threshold']:
                return None
            
            return create_intelligence_signal(
                signal_type=IntelligenceSignalType.ANOMALY_DETECTION,
                strength=signal_strength,
                confidence=confidence,
                direction=direction,
                timeframe=getattr(features, 'timeframe', '1m'),
                subsystem_id='immune',
                threat_level=threat_level,
                analysis_source='immune_antibodies'
            )
            
        except Exception as e:
            logger.error(f"Error generating Immune intelligence signal: {e}")
            return None
    
    def _generate_microstructure_intelligence_signal(self, features: Features, market_data, timestamp: float) -> Optional[IntelligenceSignal]:
        """Generate intelligence signal from Microstructure subsystem"""
        try:
            # Get microstructure analysis
            microstructure_result = self.microstructure_subsystem.analyze_market_microstructure(features, market_data)
            
            if not microstructure_result or not isinstance(microstructure_result, dict):
                return None
            
            raw_signal_strength = microstructure_result.get('signal', 0.0)
            confidence = microstructure_result.get('confidence', 0.5)
            liquidity_state = microstructure_result.get('liquidity_state', 'normal')
            
            # Normalize signal strength to [-1.0, 1.0] range using tanh
            signal_strength = float(np.tanh(raw_signal_strength))
            
            # Ensure bounds are strictly enforced
            signal_strength = max(-1.0, min(1.0, signal_strength))
            
            # Determine direction based on microstructure conditions
            if signal_strength > self.signal_generation_config['signal_threshold']:
                direction = 'bullish'
            elif signal_strength < -self.signal_generation_config['signal_threshold']:
                direction = 'bearish'
            else:
                direction = 'neutral'
            
            if confidence < self.signal_generation_config['confidence_threshold']:
                return None
            
            return create_intelligence_signal(
                signal_type=IntelligenceSignalType.MICROSTRUCTURE,
                strength=signal_strength,
                confidence=confidence,
                direction=direction,
                timeframe=getattr(features, 'timeframe', '1m'),
                subsystem_id='microstructure',
                liquidity_state=liquidity_state,
                analysis_source='market_microstructure'
            )
            
        except Exception as e:
            logger.error(f"Error generating Microstructure intelligence signal: {e}")
            return None
    
    def _get_signal_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about signal generation"""
        try:
            recent_signals_list = list(self.recent_signals)
            
            if not recent_signals_list:
                return {
                    'total_signals': 0,
                    'subsystem_distribution': {},
                    'direction_distribution': {},
                    'average_confidence': 0.0,
                    'average_strength': 0.0
                }
            
            # Count signals by subsystem
            subsystem_counts = {}
            direction_counts = {}
            confidences = []
            strengths = []
            
            for signal in recent_signals_list:
                subsystem = signal.get('subsystem_id')
                subsystem_counts[subsystem] = subsystem_counts.get(subsystem, 0) + 1

                direction = signal.get('direction')
                direction_counts[direction] = direction_counts.get(direction, 0) + 1

                confidences.append(signal.get('confidence', 0.0))
                strengths.append(abs(signal.get('strength', 0.0)))
            
            return {
                'total_signals': len(recent_signals_list),
                'subsystem_distribution': subsystem_counts,
                'direction_distribution': direction_counts,
                'average_confidence': np.mean(confidences) if confidences else 0.0,
                'average_strength': np.mean(strengths) if strengths else 0.0,
                'signal_generation_config': self.signal_generation_config.copy()
            }
            
        except Exception as e:
            logger.error(f"Error getting signal generation stats: {e}")
            return {'error': str(e)}
    
    def update_signal_generation_config(self, new_config: Dict[str, Any]):
        """Update signal generation configuration"""
        try:
            self.signal_generation_config.update(new_config)
            logger.info(f"Signal generation config updated: {new_config}")
        except Exception as e:
            logger.error(f"Error updating signal generation config: {e}")
    
    def get_intelligence_health(self) -> Dict[str, Any]:
        """Get health status of all intelligence subsystems"""
        try:
            orchestrator_health = self.orchestrator.get_comprehensive_stats()
            microstructure_health = self.microstructure_subsystem.get_microstructure_features()
            
            return {
                'dna_health': orchestrator_health.get('dna_evolution', {}),
                'temporal_health': orchestrator_health.get('temporal_cycles', {}),
                'immune_health': orchestrator_health.get('immune_system', {}),
                'microstructure_health': microstructure_health,
                'signal_generation_health': self._get_signal_generation_stats(),
                'overall_status': 'healthy' if all([
                    orchestrator_health,
                    microstructure_health
                ]) else 'degraded'
            }
            
        except Exception as e:
            logger.error(f"Error getting intelligence health: {e}")
            return {'error': str(e), 'overall_status': 'error'}
    
    def _register_state_management(self):
        """Register with state coordinator for unified state management"""
        try:
            register_state_component(
                name="intelligence_engine",
                save_method=self._save_state_data,
                load_method=self._load_state_data,
                priority=75  # High priority for intelligence data
            )
            logger.debug("IntelligenceEngine registered with state coordinator")
        except Exception as e:
            logger.error(f"Failed to register intelligence engine state management: {e}")
    
    @handle_storage_operation("intelligence_engine", "save_state")
    def _save_state_data(self) -> Dict[str, Any]:
        """
        Save intelligence engine state data for state coordinator.
        
        Returns:
            Dictionary containing all intelligence engine state data
        """
        try:
            # Prepare state data
            state_data = {
                'patterns': dict(self.patterns),  # Convert defaultdict to regular dict
                'recent_outcomes': list(self.recent_outcomes),
                'recent_signals': list(self.recent_signals),
                'historical_processed': self.historical_processed,
                'bootstrap_stats': dict(self.bootstrap_stats),
                'signal_generation_config': dict(self.signal_generation_config),
                'subsystem_training_progress': dict(self.subsystem_training_progress),
                'memory_file': self.memory_file,
                'saved_at': time.time(),
                'version': "2.0"  # Updated version for new state format
            }
            
            # Get orchestrator state if available
            try:
                if hasattr(self.orchestrator, 'get_state_data'):
                    state_data['orchestrator_state'] = self.orchestrator.get_state_data()
                elif hasattr(self.orchestrator, 'get_comprehensive_stats'):
                    state_data['orchestrator_stats'] = self.orchestrator.get_comprehensive_stats()
            except Exception as e:
                logger.warning(f"Error saving orchestrator state: {e}")
            
            # Get microstructure state if available
            try:
                if hasattr(self.microstructure_subsystem, 'get_state_data'):
                    state_data['microstructure_state'] = self.microstructure_subsystem.get_state_data()
                elif hasattr(self.microstructure_subsystem, 'get_microstructure_features'):
                    state_data['microstructure_features'] = self.microstructure_subsystem.get_microstructure_features()
            except Exception as e:
                logger.warning(f"Error saving microstructure state: {e}")
            
            # Validate data integrity before saving
            validation_report = validate_component_data(
                component="intelligence_engine",
                data=state_data,
                data_type="intelligence_data",
                level=ValidationLevel.STANDARD
            )
            
            if not validation_report.is_valid:
                logger.warning(f"Intelligence engine state validation issues: {len(validation_report.issues)} issues found")
                for issue in validation_report.issues:
                    if issue.severity == "error":
                        logger.error(f"Intelligence engine state error: {issue.message}")
            
            logger.debug("Intelligence engine state data prepared for saving")
            return state_data
            
        except Exception as e:
            logger.error(f"Error preparing intelligence engine state data: {e}")
            raise
    
    @handle_storage_operation("intelligence_engine", "load_state")
    def _load_state_data(self, state_data: Dict[str, Any]):
        """
        Load intelligence engine state data from state coordinator.
        
        Args:
            state_data: Dictionary containing intelligence engine state data
        """
        try:
            # Validate loaded data
            validation_report = validate_component_data(
                component="intelligence_engine",
                data=state_data,
                data_type="intelligence_data",
                level=ValidationLevel.STANDARD
            )
            
            if validation_report.result.value == "corrupted":
                logger.error("Intelligence engine state data is corrupted, cannot load")
                return
            
            if not validation_report.is_valid:
                logger.warning(f"Intelligence engine state validation issues during load: {len(validation_report.issues)} issues")
            
            # Load patterns with validation
            if 'patterns' in state_data and isinstance(state_data['patterns'], dict):
                self.patterns = defaultdict(list, state_data['patterns'])
                logger.debug(f"Loaded {len(self.patterns)} pattern types")
            
            # Load recent outcomes
            if 'recent_outcomes' in state_data and isinstance(state_data['recent_outcomes'], list):
                self.recent_outcomes = deque(state_data['recent_outcomes'], maxlen=100)
                logger.debug(f"Loaded {len(self.recent_outcomes)} recent outcomes")
            
            # Load recent signals
            if 'recent_signals' in state_data and isinstance(state_data['recent_signals'], list):
                self.recent_signals = deque(state_data['recent_signals'], maxlen=100)
                logger.debug(f"Loaded {len(self.recent_signals)} recent signals")
            
            # Load bootstrap state
            if 'historical_processed' in state_data:
                self.historical_processed = state_data['historical_processed']
            if 'bootstrap_stats' in state_data and isinstance(state_data['bootstrap_stats'], dict):
                self.bootstrap_stats.update(state_data['bootstrap_stats'])
            
            # Load configuration
            if 'signal_generation_config' in state_data and isinstance(state_data['signal_generation_config'], dict):
                self.signal_generation_config.update(state_data['signal_generation_config'])
            
            # Load training progress
            if 'subsystem_training_progress' in state_data and isinstance(state_data['subsystem_training_progress'], dict):
                for subsystem, progress in state_data['subsystem_training_progress'].items():
                    if subsystem in self.subsystem_training_progress and isinstance(progress, dict):
                        self.subsystem_training_progress[subsystem].update(progress)
            
            # Load orchestrator state if available
            if 'orchestrator_state' in state_data:
                try:
                    if hasattr(self.orchestrator, 'load_state_data'):
                        self.orchestrator.load_state_data(state_data['orchestrator_state'])
                        logger.debug("Loaded orchestrator state")
                except Exception as e:
                    logger.warning(f"Error loading orchestrator state: {e}")
            
            # Load microstructure state if available
            if 'microstructure_state' in state_data:
                try:
                    if hasattr(self.microstructure_subsystem, 'load_state_data'):
                        self.microstructure_subsystem.load_state_data(state_data['microstructure_state'])
                        logger.debug("Loaded microstructure state")
                except Exception as e:
                    logger.warning(f"Error loading microstructure state: {e}")
            
            logger.info("Intelligence engine state loaded successfully from state coordinator")
            
        except Exception as e:
            logger.error(f"Error loading intelligence engine state data: {e}")
            raise
    
    def _load_persisted_state(self):
        """Load persisted state if available"""
        try:
            # Try to load from unified serialization system
            state_data = deserialize_component_state("intelligence_engine")
            if state_data:
                self._load_state_data(state_data)
                logger.info("Intelligence engine state loaded from persistent storage")
        except Exception as e:
            logger.debug(f"No existing intelligence engine state found or failed to load: {e}")