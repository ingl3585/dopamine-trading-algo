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
        Enhanced feature handling with better logging and validation
        """
        self.step_counter += 1

        # Enhanced validation and logging
        if not self._validate_feature_vector(feat):
            log.error(f"Step {self.step_counter}: Feature validation failed for vector: {feat[:5] if len(feat) > 5 else feat}...")
            return

        # Log feature vector info periodically
        if self.step_counter % 100 == 0:
            log.info(f"Step {self.step_counter}: Processing {len(feat)}-feature vector, live={live}")
            
            # Log feature distribution
            if len(feat) == 27:
                log.debug("Multi-timeframe features - 15m: trend context, 5m: momentum, 1m: entry timing")
            else:
                log.debug("Single timeframe features")

        # Process features and get prediction
        try:
            row, action, conf, close = self.processor.process_and_predict(feat)
        except Exception as e:
            log.error(f"Step {self.step_counter}: Feature processing failed: {e}")
            return
        
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
        Enhanced validation for both 9-field and 27-field vectors with better error handling
        """
        try:
            if not isinstance(feat, (list, tuple)):
                log.warning(f"Feature vector is not list/tuple: {type(feat)}")
                return False

            feat_len = len(feat)
            log.debug(f"Validating feature vector with {feat_len} elements")

            if feat_len == 27:
                # Full multi-timeframe vector - validate all three timeframes
                return self._validate_27_feature_vector(feat)
            elif feat_len == 9:
                # Single timeframe vector - validate and expand
                return self._validate_9_feature_vector(feat)
            else:
                log.warning(f"Invalid feature vector length: {feat_len}, expected 9 or 27")
                return False

        except Exception as e:
            log.error(f"Feature validation error: {e}")
            return False
        
    def _validate_27_feature_vector(self, feat):
        """Validate 27-feature multi-timeframe vector"""
        try:
            # Validate each 9-feature timeframe block
            timeframes = [
                (0, 9, "15m"),   # Trend context
                (9, 18, "5m"),   # Momentum context  
                (18, 27, "1m")   # Entry timing
            ]
            
            for start_idx, end_idx, tf_name in timeframes:
                tf_slice = feat[start_idx:end_idx]
                if not self._validate_timeframe_slice(tf_slice, tf_name):
                    log.warning(f"Validation failed for {tf_name} timeframe slice")
                    return False
            
            log.debug("27-feature vector validation passed")
            return True
            
        except Exception as e:
            log.warning(f"27-feature validation error: {e}")
            return False
        
    def _validate_9_feature_vector(self, feat):
        """Validate and clean 9-feature vector"""
        try:
            return self._validate_timeframe_slice(feat, "single")
        except Exception as e:
            log.warning(f"9-feature validation error: {e}")
            return False
    
    def _validate_timeframe_slice(self, tf_slice, tf_name):
        """Validate a 9-element timeframe slice"""
        try:
            if len(tf_slice) != 9:
                log.warning(f"{tf_name} slice has {len(tf_slice)} elements, expected 9")
                return False
            
            # Validate and clean each element
            # [0] close price
            close = float(tf_slice[0])
            if close <= 0:
                log.warning(f"{tf_name} close price invalid: {close}")
                return False
            tf_slice[0] = close
            
            # [1] normalized volume
            norm_vol = float(tf_slice[1])
            if abs(norm_vol) > 10:  # Reasonable bounds
                log.warning(f"{tf_name} normalized volume extreme: {norm_vol}, clamping")
                tf_slice[1] = max(-10, min(10, norm_vol))
            else:
                tf_slice[1] = norm_vol
            
            # [2-7] ternary signals - must be -1, 0, or 1
            signal_names = [
                "tenkan_kijun", "price_cloud", "future_cloud", 
                "ema_cross", "tenkan_momentum", "kijun_momentum"
            ]
            
            for i, signal_name in enumerate(signal_names, start=2):
                try:
                    signal_val = float(tf_slice[i])
                    
                    # Handle NaN/Infinity
                    if not (abs(signal_val) < float('inf')):
                        log.warning(f"{tf_name} {signal_name} signal is NaN/Inf, setting to 0")
                        tf_slice[i] = 0.0
                        continue
                    
                    # Round to nearest integer
                    rounded_signal = round(signal_val)
                    
                    # Clamp to ternary range
                    if rounded_signal > 1:
                        clamped_signal = 1
                    elif rounded_signal < -1:
                        clamped_signal = -1
                    else:
                        clamped_signal = rounded_signal
                    
                    if clamped_signal != signal_val:
                        log.debug(f"{tf_name} {signal_name}: {signal_val} -> {clamped_signal}")
                    
                    tf_slice[i] = float(clamped_signal)
                    
                except (ValueError, TypeError) as e:
                    log.warning(f"{tf_name} {signal_name} signal conversion error: {e}, setting to 0")
                    tf_slice[i] = 0.0
            
            # [8] LWPE - must be 0.0 to 1.0
            try:
                lwpe = float(tf_slice[8])
                if not (0 <= lwpe <= 1):
                    log.warning(f"{tf_name} LWPE out of range: {lwpe}, clamping to [0,1]")
                    lwpe = max(0.0, min(1.0, lwpe))
                tf_slice[8] = lwpe
            except (ValueError, TypeError):
                log.warning(f"{tf_name} LWPE invalid, setting to 0.5")
                tf_slice[8] = 0.5
            
            return True
            
        except Exception as e:
            log.warning(f"{tf_name} timeframe slice validation error: {e}")
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
        Extract signal features, always from 1-minute slice (indices 18-26) for 27-feature vectors
        """
        try:
            # Determine which slice to use
            if len(feat) == 27:
                # Use 1-minute slice (entry timing) - indices 18-26
                feat_slice = feat[18:27]
                log.debug("Extracting signals from 1-minute timeframe (27-feature input)")
            elif len(feat) == 9:
                # Use entire vector
                feat_slice = feat
                log.debug("Extracting signals from single timeframe (9-feature input)")
            else:
                log.warning(f"Cannot extract signals from {len(feat)}-element vector")
                return None

            if len(feat_slice) != 9:
                log.warning(f"Signal extraction slice has {len(feat_slice)} elements, expected 9")
                return None

            return {
                'close':             feat_slice[0],
                'normalized_volume': feat_slice[1],
                'tenkan_kijun':      feat_slice[2],
                'price_cloud':       feat_slice[3],
                'future_cloud':      feat_slice[4],
                'ema_cross':         feat_slice[5],
                'tenkan_momentum':   feat_slice[6],
                'kijun_momentum':    feat_slice[7],
                'lwpe':              feat_slice[8],
            }
            
        except Exception as e:
            log.error(f"Signal feature extraction failed: {e}")
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