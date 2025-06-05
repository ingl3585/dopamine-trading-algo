# communication/tcp_bridge.py

import socket
import struct
import json
import threading
import time
import logging
from typing import Callable, Optional, Dict
from config import ResearchConfig

log = logging.getLogger(__name__)

class TCPBridge:
    """TCP communication bridge with NinjaTrader"""
    
    def __init__(self, config: ResearchConfig):
        self.config = config
        self.running = False
        
        # Setup sockets
        self.feature_socket = self._create_socket(config.FEATURE_PORT)
        self.signal_socket = self._create_socket(config.SIGNAL_PORT)
        
        # Connections
        self.feature_conn = None
        self.signal_conn = None
        
        # Callback for market data
        self.on_market_data: Optional[Callable] = None
        
        log.info(f"TCP Bridge initialized on ports {config.FEATURE_PORT} and {config.SIGNAL_PORT}")
    
    def start(self):
        """Start TCP server and wait for connections"""
        try:
            self.feature_socket.listen(1)
            self.signal_socket.listen(1)
            
            log.info("Waiting for NinjaTrader connection...")
            
            # Accept connections
            self.feature_conn, addr1 = self.feature_socket.accept()
            log.info(f"Feature connection from {addr1}")
            
            self.signal_conn, addr2 = self.signal_socket.accept()
            log.info(f"Signal connection from {addr2}")
            
            self.running = True
            
            # Start receiver thread
            threading.Thread(target=self._receive_loop, daemon=True).start()
            
            log.info("TCP Bridge started successfully")
            
        except Exception as e:
            log.error(f"TCP Bridge start error: {e}")
            raise
    
    def send_signal(self, action: int, confidence: float, quality: str):
        """Send trading signal to NinjaTrader"""
        try:
            if not self.signal_conn:
                log.warning("No signal connection available")
                return
            
            signal = {
                "action": action,
                "confidence": round(confidence, 3),
                "quality": quality,
                "timestamp": int(time.time())
            }
            
            self._send_message(self.signal_conn, signal)
            
            action_name = ['Hold', 'Buy', 'Sell'][action]
            log.info(f"Signal sent: {action_name} (conf: {confidence:.3f}, quality: {quality})")
            
        except Exception as e:
            log.error(f"Signal send error: {e}")
    
    def stop(self):
        """Stop TCP bridge"""
        self.running = False
        
        # Close connections
        for conn in [self.feature_conn, self.signal_conn]:
            if conn:
                try:
                    conn.close()
                except:
                    pass
        
        # Close sockets
        for sock in [self.feature_socket, self.signal_socket]:
            try:
                sock.close()
            except:
                pass
        
        log.info("TCP Bridge stopped")
    
    def _create_socket(self, port: int) -> socket.socket:
        """Create and configure socket"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.config.TCP_HOST, port))
        return sock
    
    def _receive_loop(self):
        """Receive market data from NinjaTrader"""
        while self.running:
            try:
                # Read message header
                header = self.feature_conn.recv(4)
                if not header:
                    break
                
                msg_len = struct.unpack('<I', header)[0]
                
                # Read message data
                data = self.feature_conn.recv(msg_len)
                if len(data) != msg_len:
                    continue
                
                # Parse and process message
                message = json.loads(data.decode())
                
                if self.on_market_data and "price_15m" in message:
                    self.on_market_data(message)
                
            except Exception as e:
                log.error(f"Receive error: {e}")
                break
        
        log.info("TCP receive loop stopped")
    
    def _send_message(self, connection: socket.socket, message: Dict):
        """Send JSON message with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                data = json.dumps(message).encode()
                header = struct.pack('<I', len(data))
                connection.sendall(header + data)
                return True
            except (BrokenPipeError, ConnectionResetError) as e:
                log.warning(f"Connection lost on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    # Try to reconnect
                    try:
                        connection.connect((self.config.TCP_HOST, self.config.SIGNAL_PORT))
                    except:
                        pass
                else:
                    log.error("Failed to send message after retries")
                    return False
            except Exception as e:
                log.error(f"Send message error: {e}")
                return False