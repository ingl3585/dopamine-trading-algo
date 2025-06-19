# tcp_bridge.py

import socket
import struct
import json
import threading
import time
import logging
from typing import Callable, Optional

log = logging.getLogger(__name__)

class SimpleTCPBridge:
    """
    Simplified TCP bridge for communication with NinjaTrader
    - Receives market data on port 5556
    - Sends trading signals on port 5557
    """
    
    def __init__(self):
        self.host = "localhost"
        self.feature_port = 5556
        self.signal_port = 5557
        
        self.running = False
        self.connected = False
        
        # Callbacks
        self.on_market_data: Optional[Callable] = None
        self.on_trade_completion: Optional[Callable] = None
        
        # Simple statistics
        self.signals_sent = 0
        self.data_received = 0
        
        # Setup sockets
        self.feature_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.feature_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.feature_server.bind((self.host, self.feature_port))
        self.feature_server.listen(1)
        
        self.signal_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.signal_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.signal_server.bind((self.host, self.signal_port))
        self.signal_server.listen(1)
        
        log.info(f"TCP bridge initialized on ports {self.feature_port} and {self.signal_port}")
    
    def start(self):
        """Start TCP server and wait for NinjaTrader connection"""
        log.info("Waiting for NinjaTrader connection...")
        
        try:
            # Set timeout for connection
            self.feature_server.settimeout(60.0)
            self.signal_server.settimeout(60.0)
            
            # Wait for connections
            self.feature_socket, addr1 = self.feature_server.accept()
            log.info(f"Feature connection from {addr1}")
            
            self.signal_socket, addr2 = self.signal_server.accept()
            log.info(f"Signal connection from {addr2}")
            
            # Remove timeout
            self.feature_socket.settimeout(None)
            self.signal_socket.settimeout(None)
            
            self.connected = True
            self.running = True
            
            # Start data receiver thread
            threading.Thread(target=self._data_receiver, daemon=True, name="DataReceiver").start()
            
            log.info("NinjaTrader connected successfully")
            
        except socket.timeout:
            log.error("Connection timeout - make sure NinjaTrader is running")
            raise ConnectionError("NinjaTrader connection timeout")
        except Exception as e:
            log.error(f"Connection failed: {e}")
            raise
    
    def _data_receiver(self):
        """Receive data from NinjaTrader"""
        log.info("Data receiver started")
        
        while self.running:
            try:
                # Read message header (4 bytes for length)
                header = self.feature_socket.recv(4, socket.MSG_WAITALL)
                if not header:
                    log.warning("Connection lost")
                    self.connected = False
                    break
                
                # Get message length
                msg_length = struct.unpack('<I', header)[0]
                
                if msg_length <= 0 or msg_length > 1000000:
                    log.error(f"Invalid message length: {msg_length}")
                    continue
                
                # Read message data
                data = self.feature_socket.recv(msg_length, socket.MSG_WAITALL)
                if len(data) != msg_length:
                    log.warning(f"Incomplete data: {len(data)}/{msg_length}")
                    continue
                
                # Parse JSON
                try:
                    message = json.loads(data.decode())
                    self.data_received += 1
                    
                    # Handle different message types
                    if message.get('type') == 'trade_completion':
                        if self.on_trade_completion:
                            self.on_trade_completion(message)
                            log.info(f"Trade completion: P&L=${message.get('final_pnl', 0):.2f}")
                    
                    elif 'price_1m' in message:  # Market data
                        if self.on_market_data:
                            self.on_market_data(message)
                    
                    else:
                        log.debug(f"Unknown message type: {message.get('type', 'no_type')}")
                
                except json.JSONDecodeError as e:
                    log.error(f"JSON decode error: {e}")
                    continue
                
            except Exception as e:
                log.error(f"Data receiver error: {e}")
                self.connected = False
                break
        
        log.info("Data receiver stopped")
    
    def send_signal(self, action: int, confidence: float, position_size: float = 1.0,
                   stop_price: float = 0.0, target_price: float = 0.0, 
                   tool_name: str = "unknown") -> bool:
        """Send trading signal to NinjaTrader"""
        
        if not self.connected:
            log.warning("Cannot send signal - not connected")
            return False
        
        try:
            # Create signal
            signal = {
                "action": int(action),
                "confidence": float(confidence),
                "position_size": float(position_size),
                "use_stop": stop_price > 0.0,
                "stop_price": float(stop_price),
                "use_target": target_price > 0.0,
                "target_price": float(target_price),
                "tool_used": str(tool_name),
                "quality": f"{tool_name}_signal",
                "timestamp": int(time.time() * 10000000 + 621355968000000000)  # .NET ticks
            }
            
            # Send signal
            data = json.dumps(signal).encode()
            header = struct.pack('<I', len(data))
            self.signal_socket.sendall(header + data)
            
            self.signals_sent += 1
            
            action_name = ['EXIT', 'BUY', 'SELL'][action]
            log.info(f"Signal #{self.signals_sent}: {action_name} "
                    f"(conf: {confidence:.3f}, size: {position_size:.1f}, tool: {tool_name})")
            
            if stop_price > 0:
                log.info(f"  Stop: ${stop_price:.2f}")
            if target_price > 0:
                log.info(f"  Target: ${target_price:.2f}")
            
            return True
            
        except Exception as e:
            log.error(f"Signal send error: {e}")
            self.connected = False
            return False
    
    def stop(self):
        """Stop TCP bridge"""
        log.info("Stopping TCP bridge...")
        
        self.running = False
        self.connected = False
        
        # Close sockets
        for name, sock in [("feature", getattr(self, 'feature_socket', None)),
                          ("signal", getattr(self, 'signal_socket', None)),
                          ("feature_server", self.feature_server),
                          ("signal_server", self.signal_server)]:
            try:
                if sock:
                    sock.close()
                    log.debug(f"Closed {name} socket")
            except Exception as e:
                log.warning(f"Error closing {name} socket: {e}")
        
        log.info(f"TCP bridge stopped - sent {self.signals_sent} signals, "
                f"received {self.data_received} data messages")
    
    def get_status(self):
        """Get connection status"""
        return {
            'connected': self.connected,
            'running': self.running,
            'signals_sent': self.signals_sent,
            'data_received': self.data_received,
            'feature_port': self.feature_port,
            'signal_port': self.signal_port
        }
    
    def is_healthy(self):
        """Check if connection is healthy"""
        return self.connected and self.running

# Factory function
def create_tcp_bridge():
    return SimpleTCPBridge()