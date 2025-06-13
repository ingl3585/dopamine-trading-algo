# tcp_bridge.py

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
    """TCP communication bridge with NinjaTrader - enhanced for trade completions"""
    
    def __init__(self, config: ResearchConfig):
        self.config = config
        self.running = False
        
        # Callbacks
        self.on_market_data: Optional[Callable] = None
        self.on_trade_completion: Optional[Callable] = None  # NEW
        
        # Initialize sockets but don't accept yet
        self._feat_srv = socket.socket()
        self._feat_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._feat_srv.bind((config.TCP_HOST, config.FEATURE_PORT))
        self._feat_srv.listen(1)

        self._sig_srv = socket.socket()
        self._sig_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sig_srv.bind((config.TCP_HOST, config.SIGNAL_PORT))
        self._sig_srv.listen(1)

        log.info(f"TCP Bridge initialized on {config.TCP_HOST}:{config.FEATURE_PORT} (features) and {config.TCP_HOST}:{config.SIGNAL_PORT} (signals)")

    def start(self):
        """Start TCP server and wait for connections"""
        log.info("Waiting for NinjaTrader connection...")
        
        # Accept connections when start() is called
        try:
            self.fsock, feat_addr = self._feat_srv.accept()
            log.info(f"Feature connection established from {feat_addr}")
            
            self.ssock, sig_addr = self._sig_srv.accept()
            log.info(f"Signal connection established from {sig_addr}")
            
            log.info("NinjaTrader connected successfully")
            
        except Exception as e:
            log.error(f"Connection establishment failed: {e}")
            raise

        # Start reader thread after connections established
        self.running = True
        threading.Thread(target=self._reader, daemon=True, name="TCPReader").start()
        log.info("TCP reader thread started")

    def _reader(self):
        """Read market data and trade completions from NinjaTrader"""
        log.info("TCP receive loop started - waiting for data...")
        
        while self.running:
            try:
                # Read message header
                hdr = self.fsock.recv(4, socket.MSG_WAITALL)
                if not hdr:
                    log.warning("Connection lost, header read failed")
                    break
                    
                msg_len = struct.unpack('<I', hdr)[0]
                
                if msg_len <= 0 or msg_len > 1000000:
                    log.error(f"Invalid message length: {msg_len}")
                    continue
                
                # Read message data
                data = self.fsock.recv(msg_len, socket.MSG_WAITALL)
                if len(data) != msg_len:
                    log.warning(f"Incomplete data received: {len(data)}/{msg_len}")
                    continue
                
                # Parse and process message
                try:
                    message = json.loads(data.decode())
                    
                    # NEW: Handle different message types
                    if message.get('type') == 'trade_completion':
                        if self.on_trade_completion:
                            self.on_trade_completion(message)
                    elif "price_15m" in message:
                        # Regular market data
                        if self.on_market_data:
                            self.on_market_data(message)
                    
                except json.JSONDecodeError as e:
                    log.error(f"JSON decode error: {e}")
                    continue
                    
            except Exception as e:
                log.error(f"Receive error: {e}")
                break
        
        log.info("TCP receive loop stopped")

    def send_signal(self, action: int, confidence: float, quality: str,
                    stop_atr: float = 0.0, tp_atr: float = 0.0):
        """Send trading signal to NinjaTrader"""
        try:
            if not hasattr(self, 'ssock') or not self.ssock:
                log.warning("No signal connection available")
                return
            
            # Convert to .NET Ticks (100-nanosecond intervals since 0001-01-01)
            # .NET epoch starts 621355968000000000 ticks before Unix epoch
            unix_timestamp = time.time()
            net_ticks = int((unix_timestamp * 10000000) + 621355968000000000)
            
            signal = {
                "action": int(action),
                "confidence": float(confidence),
                "quality": str(quality),
                "stop_atr": float(stop_atr),           
                "tp_atr": float(tp_atr),               
                "timestamp": int(net_ticks)
            }
            
            # Send signal
            data = json.dumps(signal).encode()
            header = struct.pack('<I', len(data))
            self.ssock.sendall(header + data)
            
            action_name = ['Hold', 'Buy', 'Sell'][action]
            log.info(f"Signal sent: {action_name} (conf: {confidence:.3f}, quality: {quality})")
            
        except Exception as e:
            log.error(f"Signal send error: {e}")

    def stop(self):
        """Stop TCP bridge"""
        log.info("Closing TCP bridge connections")
        self.running = False
        
        # Close connections
        for name, sock in [("feature", getattr(self, 'fsock', None)), 
                          ("signal", getattr(self, 'ssock', None)), 
                          ("feature_server", getattr(self, '_feat_srv', None)), 
                          ("signal_server", getattr(self, '_sig_srv', None))]:
            try:
                if sock:
                    sock.close()
                    log.debug(f"Closed {name} socket")
            except Exception as e:
                log.warning(f"Error closing {name} socket: {e}")
        
        log.info("TCP Bridge stopped")