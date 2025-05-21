# indicators/lwpe.py

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

def calculate_lwpe(tick_buffer):
    try:
        if len(tick_buffer) < 100:
            return 0.5

        df = pd.DataFrame(tick_buffer[-1000:])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').set_index('timestamp')
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df.dropna(subset=['price'])

        resampled = df['price'].resample('10ms').ohlc().ffill()
        resampled['mid'] = (resampled['high'] + resampled['low']) / 2

        if len(resampled) < 10:
            return 0.5

        merged = pd.merge_asof(df, resampled, left_index=True, right_index=True, direction='nearest')

        if 'price' not in merged.columns:
            log.warning("Merge failed, missing price — returning fallback LWPE")
            return 0.5

        merged['buy_pressure'] = np.where(merged['price'] > merged['mid'], merged['volume'], 0)
        merged['sell_pressure'] = np.where(merged['price'] < merged['mid'], merged['volume'], 0)

        merged['total_pressure'] = merged[['buy_pressure', 'sell_pressure']].sum(axis=1)
        merged['p_buy'] = merged['buy_pressure'] / merged['total_pressure'].replace(0, 1e-6)
        merged['p_buy'] = merged['p_buy'].clip(1e-6, 1 - 1e-6)
        merged['entropy'] = -merged['p_buy'] * np.log2(merged['p_buy']) - (1 - merged['p_buy']) * np.log2(1 - merged['p_buy'])

        liquidity = merged['volume'].rolling('100ms').mean().fillna(0)

        lwpe = merged['entropy'].iloc[-1] * (1 + liquidity.iloc[-1])
        return np.nan_to_num(lwpe, nan=0.5)

    except Exception as e:
        log.warning(f"LWPE error: {e}")
        return 0.5