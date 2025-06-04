# utils/feature_logger.py

import os
import logging
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
        
        # Verify we have exactly 28 columns (1 timestamp + 27 features + 1 reward)
        expected_columns = 28
        if len(self.columns) != expected_columns:
            log.error(f"Column count mismatch: have {len(self.columns)}, expected {expected_columns}")
            log.error(f"Columns: {self.columns}")

    def append(self, row):
        """Append a new feature row"""
        expected_length = len(self.columns)
        
        if len(row) != expected_length:
            log.warning(f"Row length mismatch: expected {expected_length}, got {len(row)}")
            
            # Try to handle the mismatch gracefully
            if len(row) < expected_length:
                # Pad with default values
                padded_row = list(row) + [0.0] * (expected_length - len(row))
                log.info(f"Padded row from {len(row)} to {len(padded_row)} elements")
                row = padded_row
            else:
                # Truncate extra elements
                row = row[:expected_length]
                log.info(f"Truncated row from {len(row)} to {expected_length} elements")
        
        self.rows.append(row)

    def flush(self):
        """Write accumulated rows to CSV file"""
        if not self.rows:
            return
        try:
            df = pd.DataFrame(self.rows, columns=self.columns)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            
            # Write with header only if file doesn't exist
            header_needed = not os.path.exists(self.file_path) or os.path.getsize(self.file_path) == 0
            
            df.to_csv(self.file_path, mode='a', header=header_needed, index=False)
            
            log.info(f"Flushed {len(df)} multi-timeframe feature rows to disk (27 features)")
            self.rows.clear()
            
        except Exception as e:
            log.warning(f"Flush failed: {e}")

    def get_feature_statistics(self):
        """Enhanced statistics for multi-timeframe ternary signal analysis"""
        if not self.rows:
            return {}
        
        try:
            df = pd.DataFrame(self.rows, columns=self.columns)
            
            # Basic stats
            stats = {
                'total_rows': len(df),
                'avg_reward': df['reward'].mean(),
            }
            
            # Multi-timeframe signal distribution analysis
            timeframes = ['15m', '5m', '1m']
            signal_types = ['tenkan_kijun', 'price_cloud', 'future_cloud', 'ema_cross', 'tenkan_momentum', 'kijun_momentum']
            
            for tf in timeframes:
                for signal_type in signal_types:
                    col_name = f"{signal_type}_signal_{tf}"
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
            
            # Multi-timeframe alignment analysis
            if 'reward' in df.columns and len(df) > 0:
                # Timeframe-specific performance
                for tf in timeframes:
                    tk_col = f"tenkan_kijun_signal_{tf}"
                    pc_col = f"price_cloud_signal_{tf}"
                    ema_col = f"ema_cross_signal_{tf}"
                    
                    if all(col in df.columns for col in [tk_col, pc_col, ema_col]):
                        # All bullish alignment
                        all_bull_mask = (df[tk_col] > 0) & (df[pc_col] > 0) & (df[ema_col] > 0)
                        if all_bull_mask.any():
                            stats[f'{tf}_all_bullish_reward'] = df[all_bull_mask]['reward'].mean()
                            stats[f'{tf}_all_bullish_count'] = all_bull_mask.sum()
                        
                        # All bearish alignment
                        all_bear_mask = (df[tk_col] < 0) & (df[pc_col] < 0) & (df[ema_col] < 0)
                        if all_bear_mask.any():
                            stats[f'{tf}_all_bearish_reward'] = df[all_bear_mask]['reward'].mean()
                            stats[f'{tf}_all_bearish_count'] = all_bear_mask.sum()
            
            return stats
            
        except Exception as e:
            log.warning(f"Multi-timeframe statistics calculation failed: {e}")
            return {}

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