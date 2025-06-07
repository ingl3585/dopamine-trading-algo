# core/trading_system.py

import threading
import logging
import numpy as np
from typing import Dict
from datetime import datetime
from config import ResearchConfig
from features.feature_extractor import FeatureExtractor
from models.logistic_model import LogisticSignalModel
from communication.tcp_bridge import TCPBridge
from pattern_learning import PatternLearningSystem, TradeRecord  # NEW

log = logging.getLogger(__name__)

class TradingSystem:
    """Main trading system with pattern learning"""
    
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
        self.tcp_bridge.on_trade_completion = self.on_trade_completed  # NEW

        # Handle shutdown
        self.shutdown_event = threading.Event()
        
        # Price tracking
        self.last_price = None
        self.trained_on_historical = False
        
        # NEW: Pattern learning
        self.pattern_learner = PatternLearningSystem(min_samples=15)
        self.pattern_learner.load_data()
        
        # NEW: Track active trades for pattern learning
        self.active_trades = {}  # signal_id -> trade_info
        self.trade_counter = 0
        
        # Statistics
        self.stats = {
            'signals_generated': 0,
            'trades_completed': 0,  # NEW
            'patterns_found': 0,    # NEW
            'buy_signals': 0,
            'sell_signals': 0,
            'hold_signals': 0,
            'avg_confidence': 0.0,
            'high_confidence_signals': 0,
            'moderate_confidence_signals': 0,
            'low_confidence_signals': 0
        }
        
        log.info("Trading system with pattern learning initialized")
    
    def start(self):
        """Start the trading system"""
        try:
            # Set TCP callback
            self.tcp_bridge.on_market_data = self.process_market_data
            
            # Start TCP bridge
            self.tcp_bridge.start()
            
            # NEW: Schedule weekly pattern reports
            self._schedule_weekly_reports()
            
            log.info("Trading system started - pattern learning active")
            
            while not self.shutdown_event.is_set():
                self.shutdown_event.wait(timeout=1)
            
        except KeyboardInterrupt:
            log.info("Shutdown requested via Ctrl+C")
        except Exception as e:
            log.error(f"System error: {e}")
        finally:
            self.stop()
    
    def process_market_data(self, data: Dict):
        """Process incoming market data and generate signals"""
        try:
            # Extract data (same as before)
            price_15m = data.get("price_15m", [])
            volume_15m = data.get("volume_15m", [])
            price_5m = data.get("price_5m", [])
            volume_5m = data.get("volume_5m", [])
            
            # Extract features
            features = self.feature_extractor.extract_features(
                price_15m, volume_15m, price_5m, volume_5m
            )

            # Train on historical data
            if not self.trained_on_historical:
                log.info(f"Training on {len(price_5m)} historical bars")
                self._train_on_historical_data(data)
                self.trained_on_historical = True
            
            # Generate signal
            action, confidence, quality = self.model.predict(features)
            
            # Determine signal strength and whether to send
            signal_strength = self._assess_signal_strength(confidence)
            should_send = confidence >= self.config.CONFIDENCE_THRESHOLD
            
            log.info(f"Signal: Action={self._get_action_name(action)}, "
                    f"Confidence={confidence:.3f}, Quality={quality}, "
                    f"Strength={signal_strength}, Send={should_send}")
            
            # Update statistics
            self._update_stats(action, confidence, signal_strength)
            
            # NEW: Track signal for pattern learning
            if should_send and action != 0:
                current_price = price_5m[-1] if price_5m else 0
                self._track_signal(action, confidence, current_price, features)
            
            # Send signal if confidence meets threshold
            if should_send:
                self.tcp_bridge.send_signal(action, confidence, quality)
            else:
                log.info(f"Signal below threshold ({self.config.CONFIDENCE_THRESHOLD:.1f}) - not sent")

            # Add to training samples
            self.model.add_training_sample(features)
                
        except Exception as e:
            log.error(f"Market data processing error: {e}")
    
    # NEW: Track signals for pattern learning
    def _track_signal(self, action, confidence, price, features):
        """Track signal for pattern learning"""
        self.trade_counter += 1
        signal_id = f"trade_{self.trade_counter}"
        
        self.active_trades[signal_id] = {
            'action': action,
            'confidence': confidence,
            'entry_price': price,
            'entry_time': datetime.now(),
            'direction': 'long' if action == 1 else 'short',
            'features': features
        }
        
        log.info(f"Tracking signal {signal_id} for pattern analysis")
    
    # NEW: Handle trade completions from NinjaTrader
    def on_trade_completed(self, completion_data):
        """Handle trade completion from NinjaTrader"""
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
            
            # Remove from active trades
            del self.active_trades[signal_id]
            
            log.info(f"Trade completed: {signal_id}, PnL=${pnl:.2f}, Duration={duration_minutes}min")
            
        except Exception as e:
            log.error(f"Trade completion error: {e}")
    
    # NEW: Schedule weekly reports
    def _schedule_weekly_reports(self):
        """Schedule weekly pattern reports"""
        def weekly_report():
            while not self.shutdown_event.is_set():
                # Wait 1 week (or until shutdown)
                if self.shutdown_event.wait(timeout=7*24*3600):
                    break
                
                try:
                    if len(self.pattern_learner.trades) >= 10:
                        report = self.pattern_learner.generate_report()
                        self._save_report(report)
                        
                        # Log top insights
                        insights = self.pattern_learner.get_insights()[:3]
                        if insights:
                            log.info("="*40)
                            log.info("WEEKLY PATTERN INSIGHTS")
                            log.info("="*40)
                            for i, insight in enumerate(insights, 1):
                                log.info(f"{i}. {insight.name}: {insight.impact:+.1%} impact")
                                log.info(f"   Action: {insight.action}")
                            log.info("="*40)
                        
                except Exception as e:
                    log.error(f"Weekly report error: {e}")
        
        # Start weekly report thread
        threading.Thread(target=weekly_report, daemon=True, name="WeeklyReports").start()
        log.info("Weekly pattern reports scheduled")
    
    # NEW: Save reports
    def _save_report(self, report):
        """Save pattern report"""
        try:
            import os
            os.makedirs('reports', exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'reports/pattern_report_{timestamp}.txt'
            
            with open(filename, 'w') as f:
                f.write(report)
            
            log.info(f"Pattern report saved: {filename}")
        except Exception as e:
            log.error(f"Report save error: {e}")
    
    # Rest of methods stay the same...
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
        """Train using research-aligned feature-based signals"""
        price_15m = data.get("price_15m", [])
        volume_15m = data.get("volume_15m", [])
        price_5m = data.get("price_5m", [])
        volume_5m = data.get("volume_5m", [])
        
        min_samples = max(50, self.config.SMA_PERIOD)
        training_samples = 0
        
        # Process historical data for training
        for i in range(min_samples, len(price_5m) - 5):
            # Get data up to point i
            hist_price_15m = price_15m[:i+1]
            hist_vol_15m = volume_15m[:i+1] 
            hist_price_5m = price_5m[:i+1]
            hist_vol_5m = volume_5m[:i+1]
            
            # Extract features for this specific time point
            features = self.feature_extractor.extract_features(
                hist_price_15m, hist_vol_15m, hist_price_5m, hist_vol_5m
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
        log.info(f"Training samples: {training_samples}")
        log.info(f"Signal distribution: {signal_dist}")
        
        # Train if we have enough samples
        if training_samples >= self.config.MIN_TRAINING_SAMPLES:
            self.model.train()
            log.info("Initial model training completed")
        else:
            log.warning(f"Insufficient training samples: {training_samples}/{self.config.MIN_TRAINING_SAMPLES}")
    
    def _update_stats(self, action: int, confidence: float, signal_strength: str):
        """Update system statistics"""
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
    
    def stop(self):
        """Stop the trading system"""
        log.info("Stopping trading system...")
        
        # NEW: Generate final pattern report
        if len(self.pattern_learner.trades) > 0:
            try:
                final_report = self.pattern_learner.generate_report()
                self._save_report(final_report)
                log.info("Final pattern report generated")
            except Exception as e:
                log.error(f"Final report error: {e}")
        
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
        
        # Print final statistics
        self._print_final_stats()
        
        log.info("System stopped")
    
    def _print_final_stats(self):
        """Print comprehensive final statistics"""
        total = self.stats['signals_generated']
        completed = self.stats['trades_completed']
        patterns = self.stats['patterns_found']
        
        if total > 0:
            log.info("=== FINAL SYSTEM STATISTICS ===")
            log.info(f"Total Signals Generated: {total}")
            log.info(f"Trades Completed: {completed}")  # NEW
            log.info(f"Patterns Discovered: {patterns}")  # NEW
            log.info(f"Signal Distribution:")
            log.info(f"  HOLD: {self.stats['hold_signals']} ({self.stats['hold_signals']/total*100:.1f}%)")
            log.info(f"  BUY:  {self.stats['buy_signals']} ({self.stats['buy_signals']/total*100:.1f}%)")
            log.info(f"  SELL: {self.stats['sell_signals']} ({self.stats['sell_signals']/total*100:.1f}%)")
            log.info(f"Average Confidence: {self.stats['avg_confidence']:.3f}")
            log.info(f"Pattern Learning: {'Active' if completed > 0 else 'Waiting for trades'}")  # NEW
            log.info("================================")
        else:
            log.info("No signals generated during session")