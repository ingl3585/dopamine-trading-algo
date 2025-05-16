# utils/tick_processor.py

import socket
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

log = logging.getLogger(__name__)

class TickProcessor:
    def __init__(self, host='localhost', tick_port=5558, output_port=5559):
        self.buffer = []
        self.host = host
        
        self.tick_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tick_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.tick_sock.bind((host, tick_port))
        self.tick_sock.listen(5)

        self.output_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.output_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.output_sock.bind((host, output_port))
        self.output_sock.listen(5)

        self._ready = threading.Event()

        threading.Thread(target=self.accept_connections, daemon=True).start()

    def accept_connections(self):
        self._ready.set()
        while True:
            conn, _ = self.output_sock.accept()
            threading.Thread(target=self.handle_client, args=(conn,), daemon=True).start()

    def wait_until_ready(self, timeout=5):
        return self._ready.wait(timeout)

    def handle_client(self, conn):
        tsock, _ = self.tick_sock.accept()
        buffer = ""
        
        while True:
            try:
                data = tsock.recv(4096).decode()
                if not data: break
                
                buffer += data
                while '\n' in buffer:
                    tick, buffer = buffer.split('\n', 1)
                    self.process_tick(tick.strip())
                
                if len(self.buffer) >= 100:
                    lwpe = self.calculate_lwpe()
                    conn.sendall(f"{lwpe:.4f}\n".encode())
                    
            except Exception as e:
                log.warning(f"Client error: {e}")
                break

    def process_tick(self, tick):
        log.debug(f"[tick] {tick}")
        parts = tick.split(',', 3)
        if len(parts) != 4: return
        
        try:
            unix_ms = int(parts[0])
            dt = datetime(1970, 1, 1) + timedelta(milliseconds=unix_ms)
            self.buffer.append({
                'timestamp': dt,
                'price': float(parts[1]),
                'volume': float(parts[2]),
                'type': parts[3].strip()
            })
        except ValueError:
            return

    def calculate_lwpe(self):
        try:
            if len(self.buffer) < 100: return 0.5
            
            df = pd.DataFrame(self.buffer[-1000:])
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