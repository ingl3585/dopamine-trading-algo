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
        self._handle_live_trading(action, conf)

        # Periodic logging and maintenance
        if self.step_counter % self.signal_summary_interval == 0:
            self._log_periodic_summary()

    def _validate_feature_vector(self, feat):
        """Validate incoming feature vector"""
        try:
            if not isinstance(feat, (list, tuple)) or len(feat) != 9:
                return False
            
            # Check for reasonable value ranges
            close = feat[0]
            if close <= 0:
                return False
            
            # Check LWPE is in reasonable range
            lwpe = feat[8]
            if not (0 <= lwpe <= 1):
                log.debug(f"LWPE out of range: {lwpe}")
            
            # Check signals are in expected range (-1, 0, 1)
            signal_indices = [2, 3, 4, 5, 6, 7]  # All signal features
            for i in signal_indices:
                if feat[i] not in [-1, 0, 1]:
                    log.debug(f"Signal {i} out of range: {feat[i]}")
            
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

    def _handle_live_trading(self, action, conf):
        """Handle live trading decisions"""
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

        # Dispatch trading signal
        self.dispatcher.dispatch_signal(action, conf)

        # Online learning
        if self.trainer.should_train_online():
            self.trainer.train_online()

        # Batch training
        if self.trainer.should_train_batch():
            log.info("Performing batch training on accumulated Ichimoku/EMA data")
            self.trainer.train_batch()

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
                'last_price': self.trainer.last_price
            }
            
            if hasattr(self.trainer.logger, 'get_feature_statistics'):
                report['feature_statistics'] = self.trainer.logger.get_feature_statistics()
            
            return report
            
        except Exception as e:
            log.warning(f"Status report generation failed: {e}")
            return {'error': str(e)}