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
        log.info("Starting initial training with Ichimoku/EMA features")
        
        try:
            # Create DataFrame with updated column structure
            columns = [
                "ts", "close", "normalized_volume", 
                "tenkan_kijun_signal", "price_cloud_signal", "future_cloud_signal",
                "ema_cross_signal", "tenkan_momentum", "kijun_momentum", 
                "lwpe", "reward"
            ]
            
            if len(self.logger.rows) > 0:
                df = pd.DataFrame(self.logger.rows, columns=columns)
                
                # Validate data integrity
                if not self._validate_training_data(df):
                    log.warning("Training data validation failed, skipping initial training")
                    return
                
                # Enhanced training for Ichimoku/EMA features
                epochs = self._determine_training_epochs(len(df))
                self.agent.train(df, epochs=epochs)
                self.agent.save_model()
                
                log.info(f"Initial training completed on {len(df)} samples with {epochs} epochs")
                
                # Log feature analysis
                self._log_feature_analysis(df)
                
            else:
                log.warning("No data available for initial training")
            
            # Clear training data and update state
            self.logger.rows.clear()
            self.trained = True
            self.args.reset = False
            
        except Exception as e:
            log.error(f"Initial training failed: {e}")
        finally:
            self.training_in_progress = False

    def _validate_training_data(self, df):
        """Validate training data for Ichimoku/EMA features"""
        try:
            # Check minimum data requirements
            if len(df) < 10:
                log.warning(f"Insufficient training data: {len(df)} samples")
                return False
            
            # Check for required columns
            required_cols = [
                "close", "tenkan_kijun_signal", "price_cloud_signal", 
                "ema_cross_signal", "lwpe", "reward"
            ]
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                log.error(f"Missing required columns: {missing_cols}")
                return False
            
            # Check for data quality
            if df['close'].isna().any() or (df['close'] <= 0).any():
                log.warning("Invalid price data detected")
                return False
            
            # Check signal ranges
            signal_cols = [
                "tenkan_kijun_signal", "price_cloud_signal", "future_cloud_signal",
                "ema_cross_signal", "tenkan_momentum", "kijun_momentum"
            ]
            
            for col in signal_cols:
                if col in df.columns:
                    invalid_signals = ~df[col].isin([-1, 0, 1])
                    if invalid_signals.any():
                        log.warning(f"Invalid signal values in {col}: {df[col][invalid_signals].unique()}")
                        # Clip to valid range
                        df[col] = df[col].clip(-1, 1)
            
            # Check LWPE range
            if 'lwpe' in df.columns:
                invalid_lwpe = (df['lwpe'] < 0) | (df['lwpe'] > 1)
                if invalid_lwpe.any():
                    log.warning(f"LWPE values out of range, clipping to [0,1]")
                    df['lwpe'] = df['lwpe'].clip(0, 1)
            
            return True
            
        except Exception as e:
            log.error(f"Training data validation error: {e}")
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

    def _log_feature_analysis(self, df):
        """Log analysis of Ichimoku/EMA feature distributions"""
        try:
            log.info("=== Feature Analysis ===")
            
            # Signal distribution
            signal_cols = [
                "tenkan_kijun_signal", "price_cloud_signal", "ema_cross_signal"
            ]
            
            for col in signal_cols:
                if col in df.columns:
                    bull = (df[col] > 0).sum()
                    bear = (df[col] < 0).sum()
                    neutral = (df[col] == 0).sum()
                    total = len(df)
                    
                    log.info(f"{col}: Bull={bull}({bull/total*100:.1f}%) "
                            f"Bear={bear}({bear/total*100:.1f}%) "
                            f"Neutral={neutral}({neutral/total*100:.1f}%)")
            
            # Performance by signal type
            if 'reward' in df.columns:
                # Ichimoku performance
                ichimoku_bull = df[
                    (df['tenkan_kijun_signal'] > 0) & 
                    (df['price_cloud_signal'] > 0)
                ]['reward'].mean()
                
                ichimoku_bear = df[
                    (df['tenkan_kijun_signal'] < 0) & 
                    (df['price_cloud_signal'] < 0)
                ]['reward'].mean()
                
                # EMA performance
                ema_bull = df[df['ema_cross_signal'] > 0]['reward'].mean()
                ema_bear = df[df['ema_cross_signal'] < 0]['reward'].mean()
                
                log.info(f"Performance - Ichimoku Bull: {ichimoku_bull:.4f}, "
                        f"Bear: {ichimoku_bear:.4f}")
                log.info(f"Performance - EMA Bull: {ema_bull:.4f}, "
                        f"Bear: {ema_bear:.4f}")
            
        except Exception as e:
            log.warning(f"Feature analysis logging failed: {e}")

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
            # Create DataFrame with Ichimoku/EMA structure
            columns = [
                "ts", "close", "normalized_volume", 
                "tenkan_kijun_signal", "price_cloud_signal", "future_cloud_signal",
                "ema_cross_signal", "tenkan_momentum", "kijun_momentum", 
                "lwpe", "reward"
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
                log.info("Model saved during batch training")
                
            self.logger.flush()
            log.info(f"Batch training completed on {len(df)} samples")
            
            # Log brief performance summary
            avg_reward = df['reward'].mean()
            log.info(f"Batch average reward: {avg_reward:.4f}")
            
        except Exception as e:
            log.error(f"Batch training failed: {e}")

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
                log.info(f"Model saved after {self.online_training_counter} online training steps")
                
                # Log feature importance update
                importance = self.agent.get_feature_importance_summary()
                if importance:
                    log.info(f"Feature importance update: {importance}")
                
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
            'last_save_time': self.last_save_time
        }