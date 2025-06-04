# handlers/live_feature_handler.py

import logging

from model.trainer import Trainer
from model.reward import RewardCalculator
from services.feature_processor import FeatureProcessor
from handlers.live_signal_handler import LiveSignalHandler

log = logging.getLogger(__name__)

class LiveFeatureHandler:
    def __init__(self, cfg, agent, portfolio, logger, tcp, args):
        self.cfg = cfg
        self.rewarder = RewardCalculator()
        self.processor = FeatureProcessor(agent, self.rewarder)
        self.trainer = Trainer(cfg, agent, logger, args)
        self.dispatcher = LiveSignalHandler(cfg, agent, portfolio, tcp, logger)
        self.step_counter = 0
        self.signal_summary_interval = 100

    def handle_live_feature(self, feat, live):
        """
        Handle incoming feature vector with Ichimoku/EMA signals
        
        Args:
            feat: Feature vector [close, volume, tenkan_kijun, price_cloud, future_cloud, 
                                ema_cross, tenkan_momentum, kijun_momentum, lwpe]
            live: 1 if real-time data, 0 if historical
        """
        self.step_counter += 1

        # Validate feature vector
        if not self._validate_feature_vector(feat):
            log.warning(f"Invalid feature vector at step {self.step_counter}: {feat}")
            return

        # Process features and get prediction
        row, action, conf, close = self.processor.process_and_predict(feat)
        
        # Log the processed data
        self.trainer.append(row)
        self.trainer.last_price = close

        # Historical data processing
        if live == 0:
            self._handle_historical_data(action, conf)
            return

        # Live trading logic
        self._handle_live_trading(action, conf, feat)

        # Periodic logging and maintenance
        if self.step_counter % self.signal_summary_interval == 0:
            self._log_periodic_summary()

    def _validate_feature_vector(self, feat):
        """
        Accept both the old 9‑field and the new 27‑field (3×9) vectors.
        For 27‑field input we just sanity‑check the *entry (1‑min) slice*
        – indices 18‑26 – because that’s what drives the RL right now.
        """
        try:
            if not isinstance(feat, (list, tuple)):
                return False

            if len(feat) == 27:                       # ← new multi‑TF vector
                entry_slice = feat[18:27]             # 1‑min part
            elif len(feat) == 9:                      # ← old single‑TF vector
                entry_slice = feat
            else:
                return False

            close, _, *signals, lwpe = entry_slice
            if close <= 0 or not (0 <= lwpe <= 1):
                return False

            # clamp ternary signals in place
            for i in range(2, 8):                     # the six signal slots
                v = round(entry_slice[i])
                entry_slice[i] = float(max(-1, min(1, v)))

            # shove the cleaned slice back into feat if we took one
            if len(feat) == 27:
                feat[18:27] = entry_slice
            else:
                feat[:] = entry_slice

            return True
        except Exception as e:
            log.warning(f"Feature validation error: {e}")
            return False

    def _handle_historical_data(self, action, conf):
        """Handle historical data processing"""
        # During historical processing, focus on learning
        if self.step_counter % 50 == 0:
            log.debug(f"Historical processing: step {self.step_counter}, "
                     f"action={action}, conf={conf:.3f}")

    def _handle_live_trading(self, action, conf, feat):
        """Handle live trading decisions with signal features"""
        # Initial training check
        if self.trainer.should_train_initial():
            log.info("Performing initial training with Ichimoku/EMA features before live trading")
            self.trainer.perform_initial_training()
            return

        # Ready check
        if not self.trainer.is_ready_for_trading():
            log.debug("Not ready for trading yet - waiting for sufficient data")
            return

        # Check confidence threshold
        if conf < self.cfg.CONFIDENCE_THRESHOLD:
            log.debug(f"Confidence {conf:.3f} below threshold {self.cfg.CONFIDENCE_THRESHOLD}, holding")
            action = 0  # Force hold for low confidence

        # Extract signal features for quality analysis
        signal_features = self._extract_signal_features(feat)

        # Dispatch trading signal with features for quality analysis
        self.dispatcher.dispatch_signal(action, conf, signal_features)

        # Online learning
        if self.trainer.should_train_online():
            self.trainer.train_online()

        # Batch training
        if self.trainer.should_train_batch():
            log.info("Performing batch training on accumulated Ichimoku/EMA data")
            self.trainer.train_batch()

    def _extract_signal_features(self, feat):
        """
        Always pull from the 1‑minute slice when we get a 27‑field vector.
        """
        try:
            if len(feat) == 27:
                feat = feat[18:27]

            if len(feat) != 9:
                return None

            return {
                'close':             feat[0],
                'normalized_volume': feat[1],
                'tenkan_kijun':      feat[2],
                'price_cloud':       feat[3],
                'future_cloud':      feat[4],
                'ema_cross':         feat[5],
                'tenkan_momentum':   feat[6],
                'kijun_momentum':    feat[7],
                'lwpe':              feat[8],
            }
        except Exception as e:
            log.warning(f"Signal feature extraction failed: {e}")
            return None

    def _log_periodic_summary(self):
        """Log periodic summary of signals and performance"""
        try:
            # Signal summary
            signal_summary = self.processor.get_signal_summary()
            log.info(f"Step {self.step_counter} - Signals: {signal_summary}")
            
            # Buffer status
            buffer_size = len(self.trainer.agent.experience_buffer)
            log.info(f"Experience buffer: {buffer_size} samples")
            
            # Feature importance (if available)
            importance = self.trainer.agent.get_feature_importance_summary()
            if importance:
                top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
                log.info(f"Top features: {', '.join([f'{k}:{v:.3f}' for k, v in top_features])}")
            
            # Logger statistics
            if hasattr(self.trainer.logger, 'get_feature_statistics'):
                stats = self.trainer.logger.get_feature_statistics()
                if stats:
                    log.info(f"Recent performance: avg_reward={stats.get('avg_reward', 0):.4f}, "
                            f"ichimoku_bull={stats.get('ichimoku_bullish_signals', 0)}, "
                            f"ema_bull={stats.get('ema_bullish', 0)}")
            
            # Signal quality statistics from dispatcher
            if hasattr(self.dispatcher, 'get_signal_quality_stats'):
                quality_stats = self.dispatcher.get_signal_quality_stats()
                if quality_stats:
                    total_signals = quality_stats.get('total_recent_signals', 0)
                    avg_conf = quality_stats.get('overall_avg_confidence', 0)
                    log.info(f"Signal quality: {total_signals} recent, avg_conf={avg_conf:.3f}")
            
        except Exception as e:
            log.warning(f"Periodic summary logging failed: {e}")

    def get_status_report(self):
        """Get comprehensive status report"""
        try:
            report = {
                'step_counter': self.step_counter,
                'is_ready_for_trading': self.trainer.is_ready_for_trading(),
                'experience_buffer_size': len(self.trainer.agent.experience_buffer),
                'current_signals': self.processor.get_signal_summary(),
                'feature_importance': self.trainer.agent.get_feature_importance_summary(),
                'last_price': self.trainer.last_price,
                'architecture_mode': 'pure_ml_signal_generation'
            }
            
            if hasattr(self.trainer.logger, 'get_feature_statistics'):
                report['feature_statistics'] = self.trainer.logger.get_feature_statistics()
            
            # Add signal quality stats
            if hasattr(self.dispatcher, 'get_signal_quality_stats'):
                report['signal_quality_stats'] = self.dispatcher.get_signal_quality_stats()
            
            return report
            
        except Exception as e:
            log.warning(f"Status report generation failed: {e}")
            return {'error': str(e)}