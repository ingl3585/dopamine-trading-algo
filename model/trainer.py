# model/trainer.py

import time
import logging
import pandas as pd

from argparse import Namespace
from model.agent import RLAgent
from utils.feature_logger import FeatureLogger

log = logging.getLogger(__name__)

class Trainer:
    def __init__(self, cfg, agent: RLAgent, logger: FeatureLogger, args: Namespace):
        self.cfg = cfg
        self.agent = agent
        self.logger = logger
        self.args = args
        self.trained = False
        self.last_price = None
        self.last_save_time = time.time()
        self.training_in_progress = False
        self.online_training_counter = 0
        self.feature_validation_errors = 0

    def append(self, row):
        """Append processed feature row to logger"""
        try:
            self.logger.append(row)
        except Exception as e:
            log.warning(f"Failed to append row to logger: {e}")

    def should_train_initial(self):
        """Check if initial training should be performed"""
        return not self.trained or self.args.reset

    def perform_initial_training(self):
        """Perform initial training on accumulated data"""
        if self.training_in_progress:
            log.debug("Training already in progress, skipping")
            return
            
        self.training_in_progress = True
        log.info("Starting initial training with multi-timeframe Ichimoku/EMA features")
        
        try:
            # Create DataFrame with updated 27-feature column structure
            columns = [
                "ts",
                # 15-minute features
                "close_15m", "normalized_volume_15m", "tenkan_kijun_15m", 
                "price_cloud_15m", "future_cloud_15m", "ema_cross_15m",
                "tenkan_momentum_15m", "kijun_momentum_15m", "lwpe_15m",
                
                # 5-minute features
                "close_5m", "normalized_volume_5m", "tenkan_kijun_5m", 
                "price_cloud_5m", "future_cloud_5m", "ema_cross_5m",
                "tenkan_momentum_5m", "kijun_momentum_5m", "lwpe_5m",
                
                # 1-minute features
                "close_1m", "normalized_volume_1m", "tenkan_kijun_1m", 
                "price_cloud_1m", "future_cloud_1m", "ema_cross_1m",
                "tenkan_momentum_1m", "kijun_momentum_1m", "lwpe_1m",
                
                "reward"
            ]
            
            if len(self.logger.rows) > 0:
                df = pd.DataFrame(self.logger.rows, columns=columns)
                
                # Validate data integrity
                if not self._validate_training_data(df):
                    log.warning("Training data validation failed, skipping initial training")
                    return
                
                # Enhanced training for multi-timeframe features
                epochs = self._determine_training_epochs(len(df))
                self.agent.train(df, epochs=epochs)
                self.agent.save_model()
                
                log.info(f"Initial multi-timeframe training completed on {len(df)} samples with {epochs} epochs")
                
                # Log feature analysis
                self._log_multiframe_feature_analysis(df)
                
            else:
                log.warning("No data available for initial training")
            
            # Clear training data and update state
            self.logger.rows.clear()
            self.trained = True
            self.args.reset = False
            
        except Exception as e:
            log.error(f"Initial multi-timeframe training failed: {e}")
        finally:
            self.training_in_progress = False

    def _validate_training_data(self, df):
        """Validate training data for multi-timeframe features"""
        try:
            # Check minimum data requirements
            if len(df) < 10:
                log.warning(f"Insufficient training data: {len(df)} samples")
                return False
            
            # Check for required columns
            required_cols = [
                "close_1m", "tenkan_kijun_1m", "price_cloud_1m", 
                "ema_cross_1m", "lwpe_1m", "reward"
            ]
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                log.error(f"Missing required columns: {missing_cols}")
                return False
            
            # Check for data quality - validate all timeframes
            timeframes = ['15m', '5m', '1m']
            for tf in timeframes:
                close_col = f"close_{tf}"
                if close_col in df.columns:
                    if df[close_col].isna().any() or (df[close_col] <= 0).any():
                        log.warning(f"Invalid price data in {close_col}")
                        return False
            
            # Check signal ranges for all timeframes
            signal_types = ['tenkan_kijun', 'price_cloud', 'future_cloud', 'ema_cross', 'tenkan_momentum', 'kijun_momentum']
            
            for tf in timeframes:
                for signal_type in signal_types:
                    col_name = f"{signal_type}_{tf}"
                    if col_name in df.columns:
                        invalid_signals = ~df[col_name].isin([-1, 0, 1])
                        if invalid_signals.any():
                            log.warning(f"Invalid signal values in {col_name}: {df[col_name][invalid_signals].unique()}")
                            # Clip to valid range
                            df[col_name] = df[col_name].clip(-1, 1)
            
            # Check LWPE range for all timeframes
            for tf in timeframes:
                lwpe_col = f"lwpe_{tf}"
                if lwpe_col in df.columns:
                    invalid_lwpe = (df[lwpe_col] < 0) | (df[lwpe_col] > 1)
                    if invalid_lwpe.any():
                        log.warning(f"LWPE values out of range in {lwpe_col}, clipping to [0,1]")
                        df[lwpe_col] = df[lwpe_col].clip(0, 1)
            
            return True
            
        except Exception as e:
            log.error(f"Multi-timeframe training data validation error: {e}")
            return False

    def _determine_training_epochs(self, data_size):
        """Determine optimal number of training epochs based on data size"""
        if data_size < 100:
            return 5
        elif data_size < 500:
            return 3
        elif data_size < 1000:
            return 2
        else:
            return 1

    def _log_multiframe_feature_analysis(self, df):
        """Log analysis of multi-timeframe feature distributions"""
        try:
            log.info("=== Multi-Timeframe Feature Analysis ===")
            
            timeframes = ['15m', '5m', '1m']
            signal_types = ['tenkan_kijun', 'price_cloud', 'ema_cross']
            
            for tf in timeframes:
                log.info(f"\n{tf.upper()} Timeframe Analysis:")
                
                for signal_type in signal_types:
                    col_name = f"{signal_type}_signal_{tf}"
                    if col_name in df.columns:
                        bull = (df[col_name] > 0).sum()
                        bear = (df[col_name] < 0).sum()
                        neutral = (df[col_name] == 0).sum()
                        total = len(df)
                        
                        log.info(f"  {signal_type}: Bull={bull}({bull/total*100:.1f}%) "
                                f"Bear={bear}({bear/total*100:.1f}%) "
                                f"Neutral={neutral}({neutral/total*100:.1f}%)")
            
            # Multi-timeframe performance analysis
            if 'reward' in df.columns:
                log.info("\nMulti-Timeframe Performance:")
                
                for tf in timeframes:
                    # All signals bullish
                    tk_col = f"tenkan_kijun_signal_{tf}"
                    pc_col = f"price_cloud_signal_{tf}"
                    ema_col = f"ema_cross_signal_{tf}"
                    
                    if all(col in df.columns for col in [tk_col, pc_col, ema_col]):
                        all_bull_mask = (df[tk_col] > 0) & (df[pc_col] > 0) & (df[ema_col] > 0)
                        all_bear_mask = (df[tk_col] < 0) & (df[pc_col] < 0) & (df[ema_col] < 0)
                        
                        if all_bull_mask.any():
                            bull_reward = df[all_bull_mask]['reward'].mean()
                            log.info(f"  {tf.upper()} All-Bull: {all_bull_mask.sum()} setups, avg reward: {bull_reward:.4f}")
                        
                        if all_bear_mask.any():
                            bear_reward = df[all_bear_mask]['reward'].mean()
                            log.info(f"  {tf.upper()} All-Bear: {all_bear_mask.sum()} setups, avg reward: {bear_reward:.4f}")
            
        except Exception as e:
            log.warning(f"Multi-timeframe feature analysis logging failed: {e}")

    def should_train_batch(self):
        """Check if batch training should be performed"""
        return (len(self.logger.rows) >= self.cfg.BATCH_SIZE and 
                not self.training_in_progress and
                self.trained)

    def should_train_online(self):
        """Check if online training should be performed"""
        return (len(self.agent.experience_buffer) >= self.cfg.BATCH_SIZE and
                not self.training_in_progress and
                self.trained)

    def train_batch(self):
        """Perform batch training on accumulated data"""
        if self.training_in_progress:
            log.debug("Training in progress, skipping batch training")
            return
            
        try:
            # Create DataFrame with multi-timeframe structure
            columns = [
                "ts",
                # 15-minute features
                "close_15m", "normalized_volume_15m", "tenkan_kijun_signal_15m", 
                "price_cloud_signal_15m", "future_cloud_signal_15m", "ema_cross_signal_15m",
                "tenkan_momentum_15m", "kijun_momentum_15m", "lwpe_15m",
                
                # 5-minute features
                "close_5m", "normalized_volume_5m", "tenkan_kijun_signal_5m", 
                "price_cloud_signal_5m", "future_cloud_signal_5m", "ema_cross_signal_5m",
                "tenkan_momentum_5m", "kijun_momentum_5m", "lwpe_5m",
                
                # 1-minute features
                "close_1m", "normalized_volume_1m", "tenkan_kijun_signal_1m", 
                "price_cloud_signal_1m", "future_cloud_signal_1m", "ema_cross_signal_1m",
                "tenkan_momentum_1m", "kijun_momentum_1m", "lwpe_1m",
                
                "reward"
            ]
            
            df = pd.DataFrame(self.logger.rows, columns=columns)
            
            if not self._validate_training_data(df):
                log.warning("Batch training data validation failed")
                return
            
            self.agent.train(df, epochs=1)
            
            # Save model periodically
            if time.time() - self.last_save_time > 1800:  # 30 minutes
                self.agent.save_model()
                self.last_save_time = time.time()
                log.info("Multi-timeframe model saved during batch training")
                
            self.logger.flush()
            log.info(f"Multi-timeframe batch training completed on {len(df)} samples")
            
            # Log brief performance summary
            avg_reward = df['reward'].mean()
            log.info(f"Batch average reward: {avg_reward:.4f}")
            
        except Exception as e:
            log.error(f"Multi-timeframe batch training failed: {e}")

    def train_online(self):
        """Perform online training on recent experiences"""
        if self.training_in_progress or not self.trained:
            return
            
        try:
            loss = self.agent.train_online()
            self.online_training_counter += 1
            
            if self.online_training_counter % 10 == 0:
                log.debug(f"Online training step {self.online_training_counter}, "
                         f"loss: {loss:.4f}")
                
            if self.online_training_counter % 100 == 0:
                self.agent.save_model()
                log.info(f"Multi-timeframe model saved after {self.online_training_counter} online training steps")
                
                # Log feature importance update
                importance = self.agent.get_feature_importance_summary()
                if importance:
                    log.info(f"Multi-timeframe feature importance update: {importance}")
                
        except Exception as e:
            log.error(f"Online training failed: {e}")

    def is_ready_for_trading(self):
        """Check if system is ready for live trading"""
        return (self.trained and 
                not self.training_in_progress and
                self.feature_validation_errors < 10)

    def get_training_status(self):
        """Get comprehensive training status"""
        return {
            'trained': self.trained,
            'training_in_progress': self.training_in_progress,
            'online_training_steps': self.online_training_counter,
            'experience_buffer_size': len(self.agent.experience_buffer),
            'pending_batch_size': len(self.logger.rows),
            'feature_validation_errors': self.feature_validation_errors,
            'ready_for_trading': self.is_ready_for_trading(),
            'last_save_time': self.last_save_time,
            'multi_timeframe_mode': True,
            'feature_count': 27
        }