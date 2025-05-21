# services/tick_processor.py

import socket
import threading
import logging

from datetime import datetime, timedelta
from indicators.lwpe import calculate_lwpe

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
                    lwpe = calculate_lwpe(self.buffer)
                    conn.sendall(f"{lwpe:.4f}\n".encode())
                    
            except Exception as e:
                log.warning(f"Client error: {e}")
                break

    def process_tick(self, tick):
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