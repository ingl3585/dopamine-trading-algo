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
        
        # Updated column names for Ichimoku/EMA features
        self.columns = [
            "ts", "close", "normalized_volume", 
            "tenkan_kijun_signal", "price_cloud_signal", "future_cloud_signal",
            "ema_cross_signal", "tenkan_momentum", "kijun_momentum", 
            "lwpe", "reward"
        ]

    def append(self, row):
        """Append a new feature row"""
        if len(row) != len(self.columns):
            log.warning(f"Row length mismatch: expected {len(self.columns)}, got {len(row)}")
            # Pad or truncate as needed
            if len(row) < len(self.columns):
                row = list(row) + [0.0] * (len(self.columns) - len(row))
            else:
                row = row[:len(self.columns)]
        
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
            
            log.info(f"Flushed {len(df)} feature rows to disk (Ichimoku/EMA features)")
            self.rows.clear()
            
        except Exception as e:
            log.warning(f"Flush failed: {e}")

    def get_feature_statistics(self):
        """Get statistics about current batch of features"""
        if not self.rows:
            return {}
        
        try:
            df = pd.DataFrame(self.rows, columns=self.columns)
            
            stats = {
                'total_rows': len(df),
                'ichimoku_bullish_signals': sum(
                    (df['tenkan_kijun_signal'] > 0) & 
                    (df['price_cloud_signal'] > 0) & 
                    (df['future_cloud_signal'] > 0)
                ),
                'ichimoku_bearish_signals': sum(
                    (df['tenkan_kijun_signal'] < 0) & 
                    (df['price_cloud_signal'] < 0) & 
                    (df['future_cloud_signal'] < 0)
                ),
                'ema_bullish': sum(df['ema_cross_signal'] > 0),
                'ema_bearish': sum(df['ema_cross_signal'] < 0),
                'avg_reward': df['reward'].mean(),
                'avg_lwpe': df['lwpe'].mean(),
                'avg_volume_normalized': df['normalized_volume'].mean()
            }
            
            return stats
            
        except Exception as e:
            log.warning(f"Statistics calculation failed: {e}")
            return {}

    def create_summary_report(self):
        """Create a summary report of recent trading signals"""
        if not self.rows:
            return "No data available for summary"
        
        try:
            df = pd.DataFrame(self.rows, columns=self.columns)
            
            # Recent data (last 100 rows or all if less)
            recent = df.tail(100)
            
            report = f"""
                === Ichimoku/EMA Feature Summary ===
                Total samples: {len(recent)}
                Time range: {pd.to_datetime(recent['ts'].iloc[0], unit='s').strftime('%Y-%m-%d %H:%M')} to {pd.to_datetime(recent['ts'].iloc[-1], unit='s').strftime('%Y-%m-%d %H:%M')}

                Signal Distribution:
                - Tenkan > Kijun: {sum(recent['tenkan_kijun_signal'] > 0)} ({sum(recent['tenkan_kijun_signal'] > 0)/len(recent)*100:.1f}%)
                - Price above Cloud: {sum(recent['price_cloud_signal'] > 0)} ({sum(recent['price_cloud_signal'] > 0)/len(recent)*100:.1f}%)
                - EMA Bullish: {sum(recent['ema_cross_signal'] > 0)} ({sum(recent['ema_cross_signal'] > 0)/len(recent)*100:.1f}%)

                Performance Metrics:
                - Average Reward: {recent['reward'].mean():.4f}
                - Reward Std: {recent['reward'].std():.4f}
                - Best Reward: {recent['reward'].max():.4f}
                - Worst Reward: {recent['reward'].min():.4f}

                Market Conditions:
                - Average LWPE: {recent['lwpe'].mean():.4f}
                - Volume Activity: {recent['normalized_volume'].abs().mean():.4f}

                Signal Alignment (Strong Setups):
                - Full Bull Setup: {sum((recent['tenkan_kijun_signal'] > 0) & (recent['price_cloud_signal'] > 0) & (recent['ema_cross_signal'] > 0))}
                - Full Bear Setup: {sum((recent['tenkan_kijun_signal'] < 0) & (recent['price_cloud_signal'] < 0) & (recent['ema_cross_signal'] < 0))}
                """
            
            return report
            
        except Exception as e:
            log.warning(f"Summary report generation failed: {e}")
            return "Summary report generation failed"