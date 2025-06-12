# core/trading_system.py

import os
import threading
import logging
import numpy as np
from typing import Dict
from datetime import datetime
from config import ResearchConfig
from features.feature_extractor import FeatureExtractor
from models.logistic_model import LogisticSignalModel
from communication.tcp_bridge import TCPBridge
from pattern_learning import PatternLearningSystem, TradeRecord

log = logging.getLogger(__name__)

class TradingSystem:
    """Main trading system with pattern learning - ENHANCED with 1-minute timing"""
    
    def __init__(self):
        # Configure logging first
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        self.config = ResearchConfig()
        self.feature_extractor = FeatureExtractor(self.config)
        self.model = LogisticSignalModel(self.config)
        
        # TCP bridge with trade completion handling
        self.tcp_bridge = TCPBridge(self.config)
        self.tcp_bridge.on_trade_completion = self.on_trade_completed

        # Handle shutdown
        self.shutdown_event = threading.Event()
        
        # Price tracking
        self.last_price = None
        self.trained_on_historical = False
        
        # Pattern learning
        self.pattern_learner = PatternLearningSystem(min_samples=15)
        self.pattern_learner.load_data()
        
        # Track active trades for pattern learning
        self.active_trades = {}  # signal_id -> trade_info
        self.trade_counter = 0
        
        # Simple state tracking for better entry timing
        self.last_signal_time = datetime.min
        self.entry_timing_threshold = 0.55  # Was 0.6
        
        # Statistics
        self.stats = {
            'signals_generated': 0,
            'trades_completed': 0,
            'patterns_found': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'hold_signals': 0,
            'avg_confidence': 0.0,
            'high_confidence_signals': 0,
            'moderate_confidence_signals': 0,
            'low_confidence_signals': 0,
            'entry_quality_avg': 0.0,
            'timeframe_alignment_avg': 0.0
        }
        
        log.info("Trading system with 1-minute entry timing initialized")
    
    def start(self):
        """Start the trading system"""
        try:
            # Set TCP callback
            self.tcp_bridge.on_market_data = self.process_market_data
            
            # Start TCP bridge
            self.tcp_bridge.start()
            
            log.info("Enhanced trading system started with 1-minute entry timing")
            log.info("Strategy: 15m (trend) → 5m (momentum) → 1m (precise entry)")
            
            while not self.shutdown_event.is_set():
                self.shutdown_event.wait(timeout=1)
            
        except KeyboardInterrupt:
            log.info("Shutdown requested via Ctrl+C")
        except Exception as e:
            log.error(f"System error: {e}")
        finally:
            self.stop()
    
    def process_market_data(self, data: Dict):
        """Process incoming market data and generate signals - ENHANCED"""
        try:
            # Extract data for all timeframes
            price_15m = data.get("price_15m", [])
            volume_15m = data.get("volume_15m", [])
            price_5m = data.get("price_5m", [])
            volume_5m = data.get("volume_5m", [])
            # Extract 1-minute data
            price_1m = data.get("price_1m", [])
            volume_1m = data.get("volume_1m", [])
            
            # Extract enhanced features (now includes 1m data)
            features = self.feature_extractor.extract_features(
                price_15m, volume_15m, price_5m, volume_5m, price_1m, volume_1m
            )

            if features is None:
                return

            # Train on historical data
            if not self.trained_on_historical:
                log.info(f"Training on historical data: 15m={len(price_15m)}, 5m={len(price_5m)}, 1m={len(price_1m)}")
                self._train_on_historical_data(data)
                self.trained_on_historical = True
            
            # Generate signal
            action, confidence, quality = self.model.predict(features)
            
            # Apply 1-minute entry timing logic
            should_send, final_confidence = self._apply_entry_timing_logic(
                action, confidence, features
            )
            
            # Determine signal strength
            signal_strength = self._assess_signal_strength(final_confidence)
            
            log.info(f"Signal: Action={self._get_action_name(action)}, "
                    f"Confidence={final_confidence:.3f}, Quality={quality}, "
                    f"Alignment={features.timeframe_alignment:.2f}, "
                    f"EntryQuality={features.entry_timing_quality:.2f}, Send={should_send}")
            
            # Update enhanced statistics
            self._update_enhanced_stats(action, final_confidence, signal_strength, features)
            
            # Track signal for pattern learning
            if should_send and action != 0:
                current_price = price_1m[-1] if price_1m else (price_5m[-1] if price_5m else 0)
                self._track_signal(action, final_confidence, current_price, features)
            
            # Send signal if all conditions met
            if should_send:
                self.tcp_bridge.send_signal(action, final_confidence, quality)
                self.last_signal_time = datetime.now()
            else:
                log.info(f"Signal not sent - timing/confidence filters applied")

            # Add to training samples
            self.model.add_training_sample(features)
                
        except Exception as e:
            log.error(f"Enhanced market data processing error: {e}")
    
    def _apply_entry_timing_logic(self, action: int, confidence: float, features) -> tuple:
        """ENHANCED: Apply 1-minute entry timing logic"""
        
        # Don't modify HOLD signals
        if action == 0:
            return confidence >= self.config.CONFIDENCE_THRESHOLD, confidence
        
        # Check basic confidence threshold first
        if confidence < self.config.CONFIDENCE_THRESHOLD:
            return False, confidence
        
        # 1. Require minimum timeframe alignment for directional signals
        if features.timeframe_alignment < 0.3:
            log.debug(f"Signal filtered: poor alignment {features.timeframe_alignment:.2f}")
            return False, confidence
        
        # 2. Require good entry timing quality
        if features.entry_timing_quality < self.entry_timing_threshold:
            log.debug(f"Signal filtered: poor entry timing {features.entry_timing_quality:.2f}")
            return False, confidence
        
        # 3. Prevent over-trading (simple time-based filter)
        time_since_last = (datetime.now() - self.last_signal_time).total_seconds()
        if time_since_last < 300:  # 5 minutes minimum between signals
            log.debug(f"Signal filtered: too soon after last signal ({time_since_last:.0f}s)")
            return False, confidence
        
        # 4. Boost confidence for excellent entry timing
        final_confidence = confidence
        if features.timeframe_alignment > 0.8 and features.entry_timing_quality > 0.8:
            final_confidence = min(0.95, confidence * 1.1)  # 10% boost, capped at 95%
            log.debug(f"Confidence boosted for excellent timing: {confidence:.3f} → {final_confidence:.3f}")
        
        return True, final_confidence
    
    def _track_signal(self, action, confidence, price, features):
        """Track signal for pattern learning - ENHANCED"""
        self.trade_counter += 1
        signal_id = f"trade_{self.trade_counter}"
        
        self.active_trades[signal_id] = {
            'action': action,
            'confidence': confidence,
            'entry_price': price,
            'entry_time': datetime.now(),
            'direction': 'long' if action == 1 else 'short',
            'features': features,
            'timing_data': {
                'timeframe_alignment': features.timeframe_alignment,
                'entry_timing_quality': features.entry_timing_quality,
                'rsi_1m': features.rsi_1m,
                'bb_position_1m': features.bb_position_1m
            }
        }
        
        log.info(f"Tracking enhanced signal {signal_id} (alignment: {features.timeframe_alignment:.2f}, "
                f"timing: {features.entry_timing_quality:.2f})")
    
    def on_trade_completed(self, completion_data):
        """Handle trade completion with enhanced analysis"""
        try:
            signal_id = completion_data.get('signal_id', '')
            exit_price = completion_data.get('exit_price', 0)
            exit_reason = completion_data.get('exit_reason', 'unknown')
            duration_minutes = completion_data.get('duration_minutes', 0)
            
            if signal_id not in self.active_trades:
                log.warning(f"Unknown trade completed: {signal_id}")
                return
            
            trade_info = self.active_trades[signal_id]
            
            # Calculate PnL
            entry_price = trade_info['entry_price']
            direction = trade_info['direction']
            
            if direction == 'long':
                pnl = exit_price - entry_price
            else:
                pnl = entry_price - exit_price
            
            # Create trade record
            trade_record = TradeRecord(
                entry_time=trade_info['entry_time'],
                exit_time=datetime.now(),
                entry_price=entry_price,
                exit_price=exit_price,
                direction=direction,
                confidence=trade_info['confidence'],
                time_of_day=trade_info['entry_time'].hour,
                day_of_week=trade_info['entry_time'].weekday(),
                pnl=pnl,
                duration_minutes=duration_minutes,
                exit_reason=exit_reason
            )
            
            # Add to pattern learner
            self.pattern_learner.add_trade(trade_record)
            
            # Update stats
            self.stats['trades_completed'] += 1
            self.stats['patterns_found'] = len(self.pattern_learner.insights)
            
            # Log timing analysis
            timing_data = trade_info['timing_data']
            log.info(f"Trade completed: {signal_id}, PnL=${pnl:.2f}, Duration={duration_minutes}min, "
                    f"Alignment={timing_data['timeframe_alignment']:.2f}, "
                    f"TimingQuality={timing_data['entry_timing_quality']:.2f}")
            
            # Remove from active trades
            del self.active_trades[signal_id]
            
        except Exception as e:
            log.error(f"Trade completion error: {e}")
    
    def _assess_signal_strength(self, confidence: float) -> str:
        if confidence >= 0.8:
            return "excellent"
        elif confidence >= 0.7:
            return "good"
        elif confidence >= 0.6:
            return "fair"
        else:
            return "poor"
    
    def _get_action_name(self, action: int) -> str:
        return {0: "HOLD", 1: "BUY", 2: "SELL"}.get(action, "UNKNOWN")
    
    def _train_on_historical_data(self, data):
        """Train using research-aligned feature-based signals - ENHANCED"""
        price_15m = data.get("price_15m", [])
        volume_15m = data.get("volume_15m", [])
        price_5m = data.get("price_5m", [])
        volume_5m = data.get("volume_5m", [])
        price_1m = data.get("price_1m", [])
        volume_1m = data.get("volume_1m", [])
        
        min_samples = max(50, self.config.SMA_PERIOD)
        training_samples = 0
        
        # Use the longest available timeframe for training loop
        max_length = max(len(price_5m), len(price_1m) if price_1m else 0)
        
        # Process historical data for training
        for i in range(min_samples, max_length - 5):
            # Get data up to point i
            hist_price_15m = price_15m[:min(i+1, len(price_15m))]
            hist_vol_15m = volume_15m[:min(i+1, len(volume_15m))] if volume_15m else []
            hist_price_5m = price_5m[:min(i+1, len(price_5m))]
            hist_vol_5m = volume_5m[:min(i+1, len(volume_5m))] if volume_5m else []
            
            # Include 1m data if available
            hist_price_1m = price_1m[:min(i+1, len(price_1m))] if price_1m else []
            hist_vol_1m = volume_1m[:min(i+1, len(volume_1m))] if volume_1m else []
            
            # Extract features for this specific time point
            features = self.feature_extractor.extract_features(
                hist_price_15m, hist_vol_15m, hist_price_5m, hist_vol_5m,
                hist_price_1m, hist_vol_1m
            )

            if features is not None:
                signal = self.model._generate_signal_from_features(features)
                
                # Store for training
                self.model.feature_history.append(features)
                self.model.signal_history.append(signal)
                training_samples += 1

        # Signal distribution
        unique_signals, counts = np.unique(self.model.signal_history, return_counts=True)
        signal_dist = dict(zip(unique_signals, counts))
        log.info(f"Enhanced training completed: {training_samples} samples")
        log.info(f"Signal distribution: {signal_dist}")
        
        # Train if we have enough samples
        if training_samples >= self.config.MIN_TRAINING_SAMPLES:
            self.model.train()
            log.info("Enhanced model training completed with 1-minute features")
        else:
            log.warning(f"Insufficient training samples: {training_samples}/{self.config.MIN_TRAINING_SAMPLES}")
    
    def _update_enhanced_stats(self, action: int, confidence: float, signal_strength: str, features):
        """Update enhanced system statistics"""
        self.stats['signals_generated'] += 1
        
        # Count by action
        if action == 1:
            self.stats['buy_signals'] += 1
        elif action == 2:
            self.stats['sell_signals'] += 1
        else:
            self.stats['hold_signals'] += 1
        
        # Count by signal strength
        if signal_strength == "excellent":
            self.stats['high_confidence_signals'] += 1
        elif signal_strength == "good":
            self.stats['moderate_confidence_signals'] += 1
        elif signal_strength == "fair":
            self.stats['low_confidence_signals'] += 1
        
        # Update average confidence
        total = self.stats['signals_generated']
        self.stats['avg_confidence'] = (
            (self.stats['avg_confidence'] * (total - 1) + confidence) / total
        )
        
        # Update timing stats
        self.stats['entry_quality_avg'] = (
            (self.stats['entry_quality_avg'] * (total - 1) + features.entry_timing_quality) / total
        )
        self.stats['timeframe_alignment_avg'] = (
            (self.stats['timeframe_alignment_avg'] * (total - 1) + features.timeframe_alignment) / total
        )
    
    def stop(self):
        """Stop the trading system"""
        log.info("Stopping enhanced trading system...")
        
        # Stop TCP bridge safely
        try:
            if hasattr(self, 'tcp_bridge') and self.tcp_bridge:
                self.tcp_bridge.stop()
        except Exception as e:
            log.warning(f"Error stopping TCP bridge: {e}")
        
        # Save model safely
        try:
            if hasattr(self, 'model') and self.model:
                self.model.save_model()
        except Exception as e:
            log.warning(f"Error saving model: {e}")
        
        # Print enhanced final statistics
        self._print_enhanced_final_stats()
        
        log.info("Enhanced system stopped")
    
    def _print_enhanced_final_stats(self):
        """Print comprehensive enhanced final statistics"""
        total = self.stats['signals_generated']
        completed = self.stats['trades_completed']
        patterns = self.stats['patterns_found']
        
        if total > 0:
            log.info("=== ENHANCED SYSTEM STATISTICS ===")
            log.info(f"Total Signals Generated: {total}")
            log.info(f"Trades Completed: {completed}")
            log.info(f"Patterns Discovered: {patterns}")
            log.info(f"Signal Distribution:")
            log.info(f"HOLD: {self.stats['hold_signals']} ({self.stats['hold_signals']/total*100:.1f}%)")
            log.info(f"BUY:  {self.stats['buy_signals']} ({self.stats['buy_signals']/total*100:.1f}%)")
            log.info(f"SELL: {self.stats['sell_signals']} ({self.stats['sell_signals']/total*100:.1f}%)")
            log.info(f"Average Confidence: {self.stats['avg_confidence']:.3f}")
            log.info(f"Average Entry Quality: {self.stats['entry_quality_avg']:.3f}")
            log.info(f"Average Timeframe Alignment: {self.stats['timeframe_alignment_avg']:.3f}")
            log.info(f"1-Minute Enhancement: Active")
            log.info("===================================")
        else:
            log.info("No signals generated during session")