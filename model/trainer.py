# model/trainer.py

import time
import logging
import pandas as pd
import os

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
        """Perform initial training using ALL available data - ENHANCED VERSION"""
        if self.training_in_progress:
            log.debug("Training already in progress, skipping")
            return
            
        self.training_in_progress = True
        log.info("Starting ENHANCED initial training with ALL multi-timeframe data")
        
        try:
            # Column structure for 27-feature multi-timeframe format
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
            
            df = None
            total_samples = 0
            
            # ENHANCED: Try to load ALL data from multiple sources
            
            # 1. First, preserve any in-memory data
            in_memory_df = None
            if len(self.logger.rows) > 0:
                log.info(f"Preserving {len(self.logger.rows)} in-memory rows")
                in_memory_df = pd.DataFrame(self.logger.rows, columns=columns)
                total_samples += len(in_memory_df)
            
            # 2. Load existing CSV file data
            csv_df = None
            if os.path.exists(self.cfg.FEATURE_FILE):
                try:
                    log.info(f"Loading existing CSV file: {self.cfg.FEATURE_FILE}")
                    csv_df = pd.read_csv(self.cfg.FEATURE_FILE)
                    
                    # Ensure correct column names
                    if len(csv_df.columns) == len(columns):
                        csv_df.columns = columns
                        log.info(f"Loaded {len(csv_df)} rows from CSV file")
                        total_samples += len(csv_df)
                    else:
                        log.warning(f"CSV has {len(csv_df.columns)} columns, expected {len(columns)}")
                        csv_df = None
                        
                except Exception as e:
                    log.warning(f"Could not load CSV file: {e}")
                    csv_df = None
            
            # 3. Combine all available data
            dfs_to_combine = []
            
            if csv_df is not None and len(csv_df) > 0:
                dfs_to_combine.append(csv_df)
                log.info(f"Adding {len(csv_df)} samples from CSV")
            
            if in_memory_df is not None and len(in_memory_df) > 0:
                dfs_to_combine.append(in_memory_df)
                log.info(f"Adding {len(in_memory_df)} samples from memory")
            
            # 4. Create master training dataset
            if len(dfs_to_combine) > 0:
                if len(dfs_to_combine) == 1:
                    df = dfs_to_combine[0].copy()
                else:
                    # Combine and remove duplicates based on timestamp
                    df = pd.concat(dfs_to_combine, ignore_index=True)
                    # Remove duplicates by timestamp, keeping the most recent
                    initial_len = len(df)
                    df = df.drop_duplicates(subset=['ts'], keep='last')
                    df = df.sort_values('ts').reset_index(drop=True)
                    
                    if len(df) < initial_len:
                        log.info(f"Removed {initial_len - len(df)} duplicate samples")
                
                log.info(f"MASTER DATASET: {len(df)} total samples for training")
                
                # Validate the master dataset
                if not self._validate_training_data(df):
                    log.warning("Master dataset validation failed, trying with recent data only")
                    if len(df) > 100:
                        df = df.tail(100)  # Use most recent 100 samples
                        log.info(f"Using most recent {len(df)} samples after validation failure")
                    else:
                        log.error("Insufficient valid data for training")
                        return
                
                # ENHANCED TRAINING with all available data
                epochs = self._determine_enhanced_training_epochs(len(df))
                log.info(f"ENHANCED TRAINING: {len(df)} samples, {epochs} epochs")
                
                # Train the model
                self.agent.train(df, epochs=epochs)
                self.agent.save_model()
                
                log.info(f"ENHANCED training completed on {len(df)} samples with {epochs} epochs")
                
                # ENHANCED feature analysis
                self._log_enhanced_multiframe_analysis(df)
                
                # Flush any remaining in-memory data
                if len(self.logger.rows) > 0:
                    log.info("Flushing remaining in-memory data to CSV")
                    self.logger.force_flush()
                
                # Update training state
                self.trained = True
                self.args.reset = False
                
            else:
                log.warning("No data available for enhanced training")
            
        except Exception as e:
            log.error(f"Enhanced multi-timeframe training failed: {e}")
        finally:
            self.training_in_progress = False

    def _determine_enhanced_training_epochs(self, data_size):
        """Determine optimal epochs for enhanced training with larger datasets"""
        if data_size < 50:
            return 10  # More epochs for small datasets
        elif data_size < 100:
            return 8
        elif data_size < 200:
            return 6
        elif data_size < 400:
            return 4   # Your 483 samples will use 4 epochs
        elif data_size < 1000:
            return 3
        else:
            return 2   # Fewer epochs for very large datasets

    def _log_enhanced_multiframe_analysis(self, df):
        """Enhanced analysis for larger multi-timeframe datasets"""
        try:
            log.info("=== ENHANCED Multi-Timeframe Analysis ===")
            log.info(f"Dataset Size: {len(df)} samples")
            
            # Time range analysis
            if 'ts' in df.columns:
                start_time = pd.to_datetime(df['ts'].min(), unit='s')
                end_time = pd.to_datetime(df['ts'].max(), unit='s')
                duration = end_time - start_time
                log.info(f"Time Range: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}")
                log.info(f"Duration: {duration.total_seconds()/3600:.1f} hours")
            
            timeframes = ['15m', '5m', '1m']
            signal_types = ['tenkan_kijun', 'price_cloud', 'ema_cross']
            
            # Enhanced signal distribution analysis
            for tf in timeframes:
                log.info(f"\n{tf.upper()} Timeframe Analysis:")
                
                for signal_type in signal_types:
                    col_name = f"{signal_type}_{tf}"
                    if col_name in df.columns:
                        bull = (df[col_name] > 0).sum()
                        bear = (df[col_name] < 0).sum()
                        neutral = (df[col_name] == 0).sum()
                        total = len(df)
                        
                        log.info(f"  {signal_type}: Bull={bull}({bull/total*100:.1f}%) "
                                f"Bear={bear}({bear/total*100:.1f}%) "
                                f"Neutral={neutral}({neutral/total*100:.1f}%)")
                
                # LWPE analysis for each timeframe
                lwpe_col = f"lwpe_{tf}"
                if lwpe_col in df.columns:
                    avg_lwpe = df[lwpe_col].mean()
                    extreme_buy = (df[lwpe_col] > 0.7).sum()
                    extreme_sell = (df[lwpe_col] < 0.3).sum()
                    log.info(f"  LWPE: Avg={avg_lwpe:.3f}, Strong Buy={extreme_buy}, Strong Sell={extreme_sell}")
            
            # Enhanced performance analysis
            if 'reward' in df.columns:
                log.info(f"\nReward Analysis:")
                log.info(f"  Average Reward: {df['reward'].mean():.4f}")
                log.info(f"  Reward Std: {df['reward'].std():.4f}")
                log.info(f"  Best Reward: {df['reward'].max():.4f}")
                log.info(f"  Worst Reward: {df['reward'].min():.4f}")
                
                # Profitable vs unprofitable samples
                profitable = (df['reward'] > 0).sum()
                unprofitable = (df['reward'] < 0).sum()
                neutral = (df['reward'] == 0).sum()
                log.info(f"  Profitable: {profitable}({profitable/len(df)*100:.1f}%)")
                log.info(f"  Unprofitable: {unprofitable}({unprofitable/len(df)*100:.1f}%)")
                log.info(f"  Neutral: {neutral}({neutral/len(df)*100:.1f}%)")
                
                # Enhanced timeframe performance
                log.info("\nEnhanced Multi-Timeframe Performance:")
                
                for tf in timeframes:
                    tk_col = f"tenkan_kijun_{tf}"
                    pc_col = f"price_cloud_{tf}"
                    ema_col = f"ema_cross_{tf}"
                    
                    if all(col in df.columns for col in [tk_col, pc_col, ema_col]):
                        # Perfect bullish setups
                        all_bull_mask = (df[tk_col] > 0) & (df[pc_col] > 0) & (df[ema_col] > 0)
                        # Perfect bearish setups
                        all_bear_mask = (df[tk_col] < 0) & (df[pc_col] < 0) & (df[ema_col] < 0)
                        
                        if all_bull_mask.any():
                            bull_reward = df[all_bull_mask]['reward'].mean()
                            bull_count = all_bull_mask.sum()
                            bull_win_rate = (df[all_bull_mask]['reward'] > 0).sum() / bull_count * 100
                            log.info(f"  {tf.upper()} Perfect Bull: {bull_count} setups, "
                                   f"avg reward: {bull_reward:.4f}, win rate: {bull_win_rate:.1f}%")
                        
                        if all_bear_mask.any():
                            bear_reward = df[all_bear_mask]['reward'].mean()
                            bear_count = all_bear_mask.sum()
                            bear_win_rate = (df[all_bear_mask]['reward'] > 0).sum() / bear_count * 100
                            log.info(f"  {tf.upper()} Perfect Bear: {bear_count} setups, "
                                   f"avg reward: {bear_reward:.4f}, win rate: {bear_win_rate:.1f}%")
                
                # Market regime analysis
                recent_data = df.tail(100) if len(df) > 100 else df
                recent_reward = recent_data['reward'].mean()
                overall_reward = df['reward'].mean()
                
                if recent_reward > overall_reward * 1.1:
                    log.info(f"📈 Market Regime: IMPROVING (recent: {recent_reward:.4f} vs overall: {overall_reward:.4f})")
                elif recent_reward < overall_reward * 0.9:
                    log.info(f"📉 Market Regime: DETERIORATING (recent: {recent_reward:.4f} vs overall: {overall_reward:.4f})")
                else:
                    log.info(f"📊 Market Regime: STABLE (recent: {recent_reward:.4f} vs overall: {overall_reward:.4f})")
            
        except Exception as e:
            log.warning(f"Enhanced multi-timeframe analysis failed: {e}")

    def _validate_training_data(self, df):
        """Enhanced validation for larger datasets"""
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
            
            # Enhanced data quality checks
            timeframes = ['15m', '5m', '1m']
            
            # Check for valid price data
            for tf in timeframes:
                close_col = f"close_{tf}"
                if close_col in df.columns:
                    invalid_prices = (df[close_col].isna()) | (df[close_col] <= 0)
                    if invalid_prices.any():
                        invalid_count = invalid_prices.sum()
                        log.warning(f"Invalid price data in {close_col}: {invalid_count} rows")
                        if invalid_count > len(df) * 0.1:  # More than 10% invalid
                            return False
                        # Clean invalid data
                        df.loc[invalid_prices, close_col] = df[close_col].median()
            
            # Enhanced signal validation
            signal_types = ['tenkan_kijun', 'price_cloud', 'future_cloud', 'ema_cross', 'tenkan_momentum', 'kijun_momentum']
            
            for tf in timeframes:
                for signal_type in signal_types:
                    col_name = f"{signal_type}_{tf}"
                    if col_name in df.columns:
                        # Check for extreme values and clip
                        extreme_mask = (df[col_name] < -1) | (df[col_name] > 1)
                        if extreme_mask.any():
                            extreme_count = extreme_mask.sum()
                            log.info(f"Clipping {extreme_count} extreme values in {col_name}")
                            df[col_name] = df[col_name].clip(-1, 1)
                        
                        # Round to nearest valid signal value
                        df[col_name] = df[col_name].round().astype(int)
            
            # Enhanced LWPE validation
            for tf in timeframes:
                lwpe_col = f"lwpe_{tf}"
                if lwpe_col in df.columns:
                    invalid_lwpe = (df[lwpe_col] < 0) | (df[lwpe_col] > 1)
                    if invalid_lwpe.any():
                        invalid_count = invalid_lwpe.sum()
                        log.info(f"Fixing {invalid_count} LWPE values in {lwpe_col}")
                        df[lwpe_col] = df[lwpe_col].clip(0, 1)
            
            log.info(f"Enhanced validation passed for {len(df)} samples")
            return True
            
        except Exception as e:
            log.error(f"Enhanced training data validation error: {e}")
            return False

    # ... [Keep all other methods unchanged] ...
    
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