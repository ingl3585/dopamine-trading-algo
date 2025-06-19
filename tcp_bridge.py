import json
import logging
import socket
import struct
import threading
import time
from typing import Callable, Optional

from risk_manager import Order

logger = logging.getLogger(__name__)


class TCPServer:
    def __init__(self, data_port=5556, signal_port=5557):
        self.data_port = data_port
        self.signal_port = signal_port
        
        self.running = False
        self.data_socket = None
        self.signal_socket = None
        
        # Callbacks
        self.on_market_data: Optional[Callable] = None
        self.on_trade_completion: Optional[Callable] = None
        
        self.signals_sent = 0
        self.data_received = 0
        
    def start(self):
        logger.info(f"Starting TCP server on ports {self.data_port} and {self.signal_port}")
        
        # Setup sockets
        self.data_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.data_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.data_server.bind(('localhost', self.data_port))
        self.data_server.listen(1)
        
        self.signal_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.signal_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.signal_server.bind(('localhost', self.signal_port))
        self.signal_server.listen(1)
        
        # Wait for connections
        logger.info("Waiting for NinjaTrader connection...")
        
        self.data_socket, _ = self.data_server.accept()
        self.signal_socket, _ = self.signal_server.accept()
        
        logger.info("NinjaTrader connected")
        self.running = True
        
        # Start data receiver thread
        threading.Thread(target=self._receive_data, daemon=True).start()
    
    def _receive_data(self):
        while self.running:
            try:
                # Read message header
                header = self.data_socket.recv(4, socket.MSG_WAITALL)
                if not header:
                    break
                    
                msg_length = struct.unpack('<I', header)[0]
                if msg_length <= 0 or msg_length > 1000000:
                    continue
                
                # Read message data
                data = self.data_socket.recv(msg_length, socket.MSG_WAITALL)
                if len(data) != msg_length:
                    continue
                
                # Parse and handle message
                message = json.loads(data.decode())
                self.data_received += 1
                
                if message.get('type') == 'trade_completion':
                    if self.on_trade_completion:
                        self.on_trade_completion(message)
                elif 'price_1m' in message:
                    if self.on_market_data:
                        self.on_market_data(message)
                        
            except Exception as e:
                logger.error(f"Data receive error: {e}")
                break
    
    def send_signal(self, order: Order) -> bool:
        if not self.running or not self.signal_socket:
            return False
            
        try:
            signal = {
                "action": 1 if order.action == 'buy' else 2,  # NinjaTrader expects 1=buy, 2=sell
                "confidence": order.confidence,
                "position_size": order.size,
                "use_stop": order.stop_price > 0,
                "stop_price": order.stop_price,
                "use_target": order.target_price > 0,
                "target_price": order.target_price,
                "tool_used": "ai_agent",
                "timestamp": int(time.time() * 10000000 + 621355968000000000)  # .NET ticks
            }
            
            data = json.dumps(signal).encode()
            header = struct.pack('<I', len(data))
            self.signal_socket.sendall(header + data)
            
            self.signals_sent += 1
            logger.info(f"Signal sent: {order.action.upper()} {order.size} @ {order.price:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Signal send error: {e}")
            return False
    
    def stop(self):
        logger.info("Stopping TCP server")
        self.running = False
        
        for sock in [self.data_socket, self.signal_socket, self.data_server, self.signal_server]:
            try:
                if sock:
                    sock.close()
            except:
                pass
                
        logger.info(f"TCP server stopped. Sent {self.signals_sent} signals, received {self.data_received} data messages")