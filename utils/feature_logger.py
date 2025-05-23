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
        """Enhanced statistics for ternary signal analysis"""
        if not self.rows:
            return {}
        
        try:
            df = pd.DataFrame(self.rows, columns=self.columns)
            
            # Basic stats
            stats = {
                'total_rows': len(df),
                'avg_reward': df['reward'].mean(),
                'avg_lwpe': df['lwpe'].mean(),
                'avg_volume_normalized': df['normalized_volume'].mean()
            }
            
            # Enhanced signal distribution analysis for ternary signals
            signal_columns = [
                'tenkan_kijun_signal', 'price_cloud_signal', 'future_cloud_signal',
                'ema_cross_signal', 'tenkan_momentum', 'kijun_momentum'
            ]
            
            for col in signal_columns:
                if col in df.columns:
                    bullish = sum(df[col] > 0)
                    bearish = sum(df[col] < 0)
                    neutral = sum(df[col] == 0)
                    total = len(df)
                    
                    stats[f'{col}_bullish_pct'] = (bullish / total * 100) if total > 0 else 0
                    stats[f'{col}_bearish_pct'] = (bearish / total * 100) if total > 0 else 0
                    stats[f'{col}_neutral_pct'] = (neutral / total * 100) if total > 0 else 0
            
            # Signal alignment analysis
            ichimoku_bull = sum(
                (df['tenkan_kijun_signal'] > 0) & 
                (df['price_cloud_signal'] > 0) & 
                (df['future_cloud_signal'] > 0)
            )
            
            ichimoku_bear = sum(
                (df['tenkan_kijun_signal'] < 0) & 
                (df['price_cloud_signal'] < 0) & 
                (df['future_cloud_signal'] < 0)
            )
            
            # Mixed signals (contain neutral)
            ichimoku_mixed = sum(
                (df['tenkan_kijun_signal'] == 0) | 
                (df['price_cloud_signal'] == 0) | 
                (df['future_cloud_signal'] == 0)
            )
            
            stats.update({
                'ichimoku_full_bullish': ichimoku_bull,
                'ichimoku_full_bearish': ichimoku_bear,
                'ichimoku_mixed_signals': ichimoku_mixed,
                'ichimoku_full_bullish_pct': (ichimoku_bull / len(df) * 100) if len(df) > 0 else 0,
                'ichimoku_full_bearish_pct': (ichimoku_bear / len(df) * 100) if len(df) > 0 else 0,
                'ichimoku_mixed_signals_pct': (ichimoku_mixed / len(df) * 100) if len(df) > 0 else 0
            })
            
            # EMA analysis
            ema_bull = sum(df['ema_cross_signal'] > 0)
            ema_bear = sum(df['ema_cross_signal'] < 0)
            ema_neutral = sum(df['ema_cross_signal'] == 0)
            
            stats.update({
                'ema_bullish': ema_bull,
                'ema_bearish': ema_bear,
                'ema_neutral': ema_neutral,
                'ema_bullish_pct': (ema_bull / len(df) * 100) if len(df) > 0 else 0,
                'ema_bearish_pct': (ema_bear / len(df) * 100) if len(df) > 0 else 0,
                'ema_neutral_pct': (ema_neutral / len(df) * 100) if len(df) > 0 else 0
            })
            
            # Performance by signal type
            if 'reward' in df.columns and len(df) > 0:
                # Performance when all Ichimoku signals align bullish
                full_bull_mask = (df['tenkan_kijun_signal'] > 0) & (df['price_cloud_signal'] > 0) & (df['future_cloud_signal'] > 0)
                if full_bull_mask.any():
                    stats['ichimoku_full_bull_reward'] = df[full_bull_mask]['reward'].mean()
                
                # Performance when all Ichimoku signals align bearish
                full_bear_mask = (df['tenkan_kijun_signal'] < 0) & (df['price_cloud_signal'] < 0) & (df['future_cloud_signal'] < 0)
                if full_bear_mask.any():
                    stats['ichimoku_full_bear_reward'] = df[full_bear_mask]['reward'].mean()
                
                # Performance with mixed/neutral signals
                mixed_mask = (df['tenkan_kijun_signal'] == 0) | (df['price_cloud_signal'] == 0) | (df['future_cloud_signal'] == 0)
                if mixed_mask.any():
                    stats['ichimoku_mixed_reward'] = df[mixed_mask]['reward'].mean()
                
                # EMA performance
                ema_bull_mask = df['ema_cross_signal'] > 0
                if ema_bull_mask.any():
                    stats['ema_bull_reward'] = df[ema_bull_mask]['reward'].mean()
                
                ema_bear_mask = df['ema_cross_signal'] < 0
                if ema_bear_mask.any():
                    stats['ema_bear_reward'] = df[ema_bear_mask]['reward'].mean()
                
                ema_neutral_mask = df['ema_cross_signal'] == 0
                if ema_neutral_mask.any():
                    stats['ema_neutral_reward'] = df[ema_neutral_mask]['reward'].mean()
            
            return stats
            
        except Exception as e:
            log.warning(f"Enhanced statistics calculation failed: {e}")
            return {}

    def create_summary_report(self):
        """Enhanced summary report for ternary signal analysis"""
        if not self.rows:
            return "No data available for summary"
        
        try:
            df = pd.DataFrame(self.rows, columns=self.columns)
            
            # Recent data (last 100 rows or all if less)
            recent = df.tail(100)
            
            # Calculate signal distributions
            signal_dist = {}
            signal_columns = [
                'tenkan_kijun_signal', 'price_cloud_signal', 'future_cloud_signal',
                'ema_cross_signal', 'tenkan_momentum', 'kijun_momentum'
            ]
            
            for col in signal_columns:
                if col in recent.columns:
                    bull_pct = (recent[col] > 0).sum() / len(recent) * 100
                    bear_pct = (recent[col] < 0).sum() / len(recent) * 100
                    neut_pct = (recent[col] == 0).sum() / len(recent) * 100
                    signal_dist[col] = (bull_pct, bear_pct, neut_pct)
            
            report = f"""
    === Enhanced Ichimoku/EMA Ternary Signal Summary ===
    Total samples: {len(recent)}
    Time range: {pd.to_datetime(recent['ts'].iloc[0], unit='s').strftime('%Y-%m-%d %H:%M')} to {pd.to_datetime(recent['ts'].iloc[-1], unit='s').strftime('%Y-%m-%d %H:%M')}

    Signal Distribution (Bull% / Bear% / Neutral%):"""
            
            for col, (bull, bear, neut) in signal_dist.items():
                clean_name = col.replace('_signal', '').replace('_', ' ').title()
                report += f"\n- {clean_name}: {bull:.1f}% / {bear:.1f}% / {neut:.1f}%"
            
            # Signal alignment analysis
            full_ichimoku_bull = sum(
                (recent['tenkan_kijun_signal'] > 0) & 
                (recent['price_cloud_signal'] > 0) & 
                (recent['future_cloud_signal'] > 0)
            )
            
            full_ichimoku_bear = sum(
                (recent['tenkan_kijun_signal'] < 0) & 
                (recent['price_cloud_signal'] < 0) & 
                (recent['future_cloud_signal'] < 0)
            )
            
            mixed_signals = len(recent) - full_ichimoku_bull - full_ichimoku_bear
            
            report += f"""

    Signal Alignment Analysis:
    - Full Ichimoku Bullish: {full_ichimoku_bull} ({full_ichimoku_bull/len(recent)*100:.1f}%)
    - Full Ichimoku Bearish: {full_ichimoku_bear} ({full_ichimoku_bear/len(recent)*100:.1f}%)
    - Mixed/Neutral Signals: {mixed_signals} ({mixed_signals/len(recent)*100:.1f}%)

    Performance Metrics:
    - Average Reward: {recent['reward'].mean():.4f}
    - Reward Std: {recent['reward'].std():.4f}
    - Best Reward: {recent['reward'].max():.4f}
    - Worst Reward: {recent['reward'].min():.4f}"""

            # Performance by signal alignment
            if 'reward' in recent.columns:
                full_bull_mask = (recent['tenkan_kijun_signal'] > 0) & (recent['price_cloud_signal'] > 0) & (recent['future_cloud_signal'] > 0)
                full_bear_mask = (recent['tenkan_kijun_signal'] < 0) & (recent['price_cloud_signal'] < 0) & (recent['future_cloud_signal'] < 0)
                
                if full_bull_mask.any():
                    full_bull_reward = recent[full_bull_mask]['reward'].mean()
                    report += f"\n- Full Bullish Setup Reward: {full_bull_reward:.4f}"
                
                if full_bear_mask.any():
                    full_bear_reward = recent[full_bear_mask]['reward'].mean()
                    report += f"\n- Full Bearish Setup Reward: {full_bear_reward:.4f}"
                
                # Neutral signal performance
                neutral_mask = (recent['tenkan_kijun_signal'] == 0) | (recent['price_cloud_signal'] == 0) | (recent['ema_cross_signal'] == 0)
                if neutral_mask.any():
                    neutral_reward = recent[neutral_mask]['reward'].mean()
                    report += f"\n- Neutral Signal Reward: {neutral_reward:.4f}"

            report += f"""

    Market Conditions:
    - Average LWPE: {recent['lwpe'].mean():.4f} (0.5=balanced, extremes=directional)
    - Volume Activity: {recent['normalized_volume'].abs().mean():.4f}

    Enhanced Signal Quality:
    - All-Bull Setups: {full_ichimoku_bull}
    - All-Bear Setups: {full_ichimoku_bear}
    - EMA+Ichimoku Bull: {sum((recent['ema_cross_signal'] > 0) & (recent['tenkan_kijun_signal'] > 0) & (recent['price_cloud_signal'] > 0))}
    - EMA+Ichimoku Bear: {sum((recent['ema_cross_signal'] < 0) & (recent['tenkan_kijun_signal'] < 0) & (recent['price_cloud_signal'] < 0))}
    """
            
            return report
            
        except Exception as e:
            log.warning(f"Enhanced summary report generation failed: {e}")
            return "Enhanced summary report generation failed"