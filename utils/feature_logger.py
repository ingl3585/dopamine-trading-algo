# utils/feature_logger.py

import os
import logging
import time
import pandas as pd

log = logging.getLogger(__name__)

class FeatureLogger:
    def __init__(self, file_path, batch_size):
        self.file_path = file_path
        self.batch_size = batch_size
        self.rows = []
        
        # Updated column names for 27-feature multi-timeframe format
        self.columns = [
            "ts",
            # 15-minute features (indices 0-8)
            "close_15m", "normalized_volume_15m", "tenkan_kijun_15m", 
            "price_cloud_15m", "future_cloud_15m", "ema_cross_15m",
            "tenkan_momentum_15m", "kijun_momentum_15m", "lwpe_15m",
            
            # 5-minute features (indices 9-17)
            "close_5m", "normalized_volume_5m", "tenkan_kijun_5m", 
            "price_cloud_5m", "future_cloud_5m", "ema_cross_5m",
            "tenkan_momentum_5m", "kijun_momentum_5m", "lwpe_5m",
            
            # 1-minute features (indices 18-26)
            "close_1m", "normalized_volume_1m", "tenkan_kijun_1m", 
            "price_cloud_1m", "future_cloud_1m", "ema_cross_1m",
            "tenkan_momentum_1m", "kijun_momentum_1m", "lwpe_1m",
            
            "reward"
        ]
        
        # Verify we have exactly 29 columns (1 timestamp + 27 features + 1 reward)
        expected_columns = 29
        if len(self.columns) != expected_columns:
            log.error(f"Column count mismatch: have {len(self.columns)}, expected {expected_columns}")
            log.error(f"Columns: {self.columns}")
        else:
            log.info(f"Feature logger initialized with {len(self.columns)} columns for 27-feature multi-timeframe format")

    def append(self, row):
        """Append a new feature row with enhanced validation"""
        expected_length = len(self.columns)
        
        if len(row) != expected_length:
            log.warning(f"Row length mismatch: expected {expected_length}, got {len(row)}")
            
            # Debug: show what we actually received
            if log.isEnabledFor(logging.DEBUG):
                log.debug(f"Expected columns: {self.columns}")
                log.debug(f"Received row length: {len(row)}")
                if len(row) > 0:
                    log.debug(f"First few row elements: {row[:min(5, len(row))]}")
            
            # Try to handle the mismatch gracefully
            if len(row) < expected_length:
                # Pad with default values
                padded_row = list(row) + [0.0] * (expected_length - len(row))
                log.info(f"Padded row from {len(row)} to {len(padded_row)} elements")
                row = padded_row
            elif len(row) > expected_length:
                # Truncate extra elements
                truncated_row = row[:expected_length]
                log.info(f"Truncated row from {len(row)} to {len(truncated_row)} elements")
                row = truncated_row
        
        # Validate the row content
        try:
            # Check timestamp
            if not isinstance(row[0], (int, float)) or row[0] <= 0:
                log.warning(f"Invalid timestamp: {row[0]}")
                row[0] = time.time()
            
            # Check features (elements 1-27)
            for i in range(1, 28):
                if i < len(row):
                    if not isinstance(row[i], (int, float)):
                        log.warning(f"Non-numeric feature at index {i}: {row[i]}")
                        row[i] = 0.0
                    elif i in [1, 10, 19]:  # Price indices (close_15m, close_5m, close_1m)
                        # Price values can be large (e.g., 21000+), so use reasonable bounds
                        if not (0 <= row[i] <= 100000):
                            log.debug(f"Price value out of bounds at index {i}: {row[i]} (clamping)")
                            row[i] = max(0, min(100000, row[i]))
                    elif i in [2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16, 20, 21, 22, 23, 24, 25]:  # Signal indices
                        # Ternary signals should be -1, 0, or 1
                        if row[i] not in [-1, 0, 1]:
                            rounded_signal = max(-1, min(1, round(row[i])))
                            if abs(row[i] - rounded_signal) > 0.1:  # Only log if significant difference
                                log.debug(f"Signal at index {i} clamped: {row[i]} -> {rounded_signal}")
                            row[i] = float(rounded_signal)
                    elif i in [8, 17, 26]:  # LWPE indices
                        # LWPE should be between 0 and 1
                        if not (0 <= row[i] <= 1):
                            log.debug(f"LWPE at index {i} clamped: {row[i]} -> {max(0, min(1, row[i]))}")
                            row[i] = max(0, min(1, row[i]))
                    elif not (-1000 <= row[i] <= 1000):  # Other features (volume, etc.)
                        log.debug(f"Feature value at index {i} clamped: {row[i]}")
                        row[i] = max(-1000, min(1000, row[i]))
            
            # Check reward (last element)
            if len(row) > 28:
                if not isinstance(row[28], (int, float)):
                    log.warning(f"Non-numeric reward: {row[28]}")
                    row[28] = 0.0
        
        except Exception as e:
            log.warning(f"Row validation error: {e}")
        
        self.rows.append(row)
        
        # Auto-flush if we have enough rows
        if len(self.rows) >= self.batch_size:
            self.flush()

    def flush(self):
        """Write accumulated rows to CSV file with enhanced error handling"""
        if not self.rows:
            return
        
        try:
            df = pd.DataFrame(self.rows, columns=self.columns)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            
            # Write with header only if file doesn't exist
            header_needed = not os.path.exists(self.file_path) or os.path.getsize(self.file_path) == 0
            
            # Enhanced error handling for file writing
            try:
                df.to_csv(self.file_path, mode='a', header=header_needed, index=False)
                log.info(f"Successfully flushed {len(df)} multi-timeframe feature rows to {self.file_path}")
                self.rows.clear()
                
                # Verify the file was created
                if os.path.exists(self.file_path):
                    file_size = os.path.getsize(self.file_path)
                    log.info(f"CSV file size: {file_size} bytes")
                else:
                    log.error(f"CSV file was not created: {self.file_path}")
                    
            except PermissionError as e:
                log.error(f"Permission denied writing to CSV file: {e}")
                log.error(f"Check if file is open in another application: {self.file_path}")
            except Exception as e:
                log.error(f"CSV write error: {e}")
                
        except Exception as e:
            log.error(f"DataFrame creation error: {e}")
            log.error(f"Rows data: {self.rows[:2] if self.rows else 'No rows'}")  # Show first 2 rows for debugging

    def force_flush(self):
        """Force flush even with few rows (for debugging)"""
        if self.rows:
            log.info(f"Force flushing {len(self.rows)} rows to CSV")
            self.flush()
        else:
            log.info("No rows to flush")

    def get_feature_statistics(self):
        """Enhanced statistics for multi-timeframe ternary signal analysis"""
        if not self.rows:
            # Try to load from existing CSV file
            try:
                if os.path.exists(self.file_path):
                    df = pd.read_csv(self.file_path)
                    if len(df) > 0:
                        log.info(f"Loaded {len(df)} rows from existing CSV for statistics")
                        return self._calculate_statistics_from_df(df)
            except Exception as e:
                log.warning(f"Could not load existing CSV for statistics: {e}")
            
            return {}
        
        try:
            df = pd.DataFrame(self.rows, columns=self.columns)
            return self._calculate_statistics_from_df(df)
            
        except Exception as e:
            log.warning(f"Multi-timeframe statistics calculation failed: {e}")
            return {}

    def _calculate_statistics_from_df(self, df):
        """Calculate statistics from DataFrame"""
        try:
            # Basic stats
            stats = {
                'total_rows': len(df),
                'avg_reward': df['reward'].mean() if 'reward' in df.columns else 0,
            }
            
            # Multi-timeframe signal distribution analysis
            timeframes = ['15m', '5m', '1m']
            signal_types = ['tenkan_kijun', 'price_cloud', 'future_cloud', 'ema_cross', 'tenkan_momentum', 'kijun_momentum']
            
            for tf in timeframes:
                for signal_type in signal_types:
                    col_name = f"{signal_type}_{tf}"
                    if col_name in df.columns:
                        bullish = sum(df[col_name] > 0)
                        bearish = sum(df[col_name] < 0)
                        neutral = sum(df[col_name] == 0)
                        total = len(df)
                        
                        stats[f'{tf}_{signal_type}_bullish_pct'] = (bullish / total * 100) if total > 0 else 0
                        stats[f'{tf}_{signal_type}_bearish_pct'] = (bearish / total * 100) if total > 0 else 0
                        stats[f'{tf}_{signal_type}_neutral_pct'] = (neutral / total * 100) if total > 0 else 0
                
                # LWPE and volume stats per timeframe
                lwpe_col = f"lwpe_{tf}"
                vol_col = f"normalized_volume_{tf}"
                if lwpe_col in df.columns:
                    stats[f'{tf}_avg_lwpe'] = df[lwpe_col].mean()
                if vol_col in df.columns:
                    stats[f'{tf}_avg_volume_normalized'] = df[vol_col].mean()
            
            return stats
            
        except Exception as e:
            log.warning(f"Statistics calculation error: {e}")
            return {}

    def check_file_status(self):
        """Check CSV file status for debugging"""
        try:
            if os.path.exists(self.file_path):
                file_size = os.path.getsize(self.file_path)
                log.info(f"CSV file exists: {self.file_path}")
                log.info(f"File size: {file_size} bytes")
                
                # Try to read first few rows
                try:
                    df = pd.read_csv(self.file_path, nrows=5)
                    log.info(f"CSV contains {len(df)} rows (showing first 5)")
                    log.info(f"Columns: {list(df.columns)}")
                except Exception as e:
                    log.warning(f"Could not read CSV file: {e}")
            else:
                log.info(f"CSV file does not exist yet: {self.file_path}")
                
            log.info(f"Current buffer has {len(self.rows)} rows")
            
        except Exception as e:
            log.error(f"File status check error: {e}")

    def create_summary_report(self):
        """Enhanced summary report for multi-timeframe analysis"""
        if not self.rows:
            return "No data available for multi-timeframe summary"
        
        try:
            df = pd.DataFrame(self.rows, columns=self.columns)
            
            # Recent data (last 100 rows or all if less)
            recent = df.tail(100)
            
            report = f"""
=== Multi-Timeframe Summary (27 Features) ===
Total samples: {len(recent)}
Time range: {pd.to_datetime(recent['ts'].iloc[0], unit='s').strftime('%Y-%m-%d %H:%M')} to {pd.to_datetime(recent['ts'].iloc[-1], unit='s').strftime('%Y-%m-%d %H:%M')}

Multi-Timeframe Signal Distribution:"""
            
            timeframes = ['15m', '5m', '1m']
            for tf in timeframes:
                report += f"\n\n{tf.upper()} Timeframe:"
                
                # Calculate signal percentages for this timeframe
                tk_col = f"tenkan_kijun_signal_{tf}"
                pc_col = f"price_cloud_signal_{tf}"
                ema_col = f"ema_cross_signal_{tf}"
                
                for col in [tk_col, pc_col, ema_col]:
                    if col in recent.columns:
                        bull_pct = (recent[col] > 0).sum() / len(recent) * 100
                        bear_pct = (recent[col] < 0).sum() / len(recent) * 100
                        neut_pct = (recent[col] == 0).sum() / len(recent) * 100
                        
                        signal_name = col.replace(f'_signal_{tf}', '').replace('_', ' ').title()
                        report += f"\n- {signal_name}: {bull_pct:.1f}% Bull / {bear_pct:.1f}% Bear / {neut_pct:.1f}% Neutral"
                
                # LWPE stats
                lwpe_col = f"lwpe_{tf}"
                if lwpe_col in recent.columns:
                    avg_lwpe = recent[lwpe_col].mean()
                    report += f"\n- Average LWPE: {avg_lwpe:.3f}"
            
            # Performance metrics
            report += f"""

Performance Metrics:
- Average Reward: {recent['reward'].mean():.4f}
- Reward Std: {recent['reward'].std():.4f}
- Best Reward: {recent['reward'].max():.4f}
- Worst Reward: {recent['reward'].min():.4f}"""

            # Timeframe alignment analysis
            if 'reward' in recent.columns:
                report += "\n\nTimeframe Alignment Performance:"
                
                for tf in timeframes:
                    tk_col = f"tenkan_kijun_signal_{tf}"
                    pc_col = f"price_cloud_signal_{tf}"
                    ema_col = f"ema_cross_signal_{tf}"
                    
                    if all(col in recent.columns for col in [tk_col, pc_col, ema_col]):
                        # All bullish
                        all_bull_mask = (recent[tk_col] > 0) & (recent[pc_col] > 0) & (recent[ema_col] > 0)
                        if all_bull_mask.any():
                            bull_reward = recent[all_bull_mask]['reward'].mean()
                            bull_count = all_bull_mask.sum()
                            report += f"\n- {tf.upper()} All Bullish: {bull_count} setups, avg reward {bull_reward:.4f}"
                        
                        # All bearish
                        all_bear_mask = (recent[tk_col] < 0) & (recent[pc_col] < 0) & (recent[ema_col] < 0)
                        if all_bear_mask.any():
                            bear_reward = recent[all_bear_mask]['reward'].mean()
                            bear_count = all_bear_mask.sum()
                            report += f"\n- {tf.upper()} All Bearish: {bear_count} setups, avg reward {bear_reward:.4f}"
            
            return report
            
        except Exception as e:
            log.warning(f"Multi-timeframe summary report generation failed: {e}")
            return "Multi-timeframe summary report generation failed"