# tcp_bridge.py

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
        self.on_historical_data: Optional[Callable] = None  # NEW: Historical data callback
        
        self.signals_sent = 0
        self.data_received = 0
        self.historical_data_received = False  # NEW: Track historical data status
        
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
                if msg_length <= 0 or msg_length > 10000000:  # MODIFIED: Increased for historical data
                    continue
                
                # Read message data
                data = self.data_socket.recv(msg_length, socket.MSG_WAITALL)
                if len(data) != msg_length:
                    continue
                
                # Parse and handle message
                message = json.loads(data.decode())
                self.data_received += 1
                
                # MODIFIED: Enhanced message handling for historical data
                message_type = message.get('type', '')
                
                if message_type == 'historical_data':
                    if self.on_historical_data:
                        logger.info(f"Received historical data: {len(message.get('bars_15m', []))} 15m bars")
                        self.on_historical_data(message)
                        self.historical_data_received = True
                elif message_type == 'trade_completion':
                    if self.on_trade_completion:
                        self.on_trade_completion(message)
                elif self._is_market_data(message):
                    if self.on_market_data:
                        # Add validation and enhancement for account data
                        enhanced_message = self._enhance_market_data(message)
                        self.on_market_data(enhanced_message)
                        
            except Exception as e:
                logger.error(f"Data receive error: {e}")
                break
    
    def _is_market_data(self, message: dict) -> bool:
        """Check if message contains market data"""
        required_fields = ['price_1m', 'account_balance', 'buying_power']
        return any(field in message for field in required_fields)
    
    def _enhance_market_data(self, message: dict) -> dict:
        """Enhance market data with computed fields and validation"""
        enhanced = message.copy()
        
        # Ensure all required account fields exist with defaults
        account_defaults = {
            'account_balance': 25000.0,
            'buying_power': 25000.0,
            'daily_pnl': 0.0,
            'net_liquidation': 25000.0,
            'margin_used': 0.0,
            'available_margin': 25000.0,
            'open_positions': 0
        }
        
        for field, default_value in account_defaults.items():
            if field not in enhanced:
                enhanced[field] = default_value
        
        # Add computed fields for better analysis
        try:
            # Calculate margin utilization percentage
            margin_used = enhanced.get('margin_used', 0)
            net_liquidation = enhanced.get('net_liquidation', enhanced.get('account_balance', 25000))
            
            if net_liquidation > 0:
                enhanced['margin_utilization'] = margin_used / net_liquidation
            else:
                enhanced['margin_utilization'] = 0.0
            
            # Calculate available buying power percentage
            buying_power = enhanced.get('buying_power', 25000)
            if net_liquidation > 0:
                enhanced['buying_power_ratio'] = buying_power / net_liquidation
            else:
                enhanced['buying_power_ratio'] = 1.0
            
            # Calculate daily PnL percentage
            daily_pnl = enhanced.get('daily_pnl', 0)
            if net_liquidation > 0:
                enhanced['daily_pnl_pct'] = daily_pnl / net_liquidation
            else:
                enhanced['daily_pnl_pct'] = 0.0
            
            # Validate and clean price/volume arrays
            for timeframe in ['1m', '5m', '15m']:
                price_key = f'price_{timeframe}'
                volume_key = f'volume_{timeframe}'
                
                if price_key in enhanced and enhanced[price_key]:
                    # Ensure arrays are lists and contain valid numbers
                    prices = enhanced[price_key]
                    if isinstance(prices, list) and len(prices) > 0:
                        # Remove any invalid values
                        valid_prices = [p for p in prices if isinstance(p, (int, float)) and p > 0]
                        enhanced[price_key] = valid_prices
                    else:
                        enhanced[price_key] = []
                
                if volume_key in enhanced and enhanced[volume_key]:
                    volumes = enhanced[volume_key]
                    if isinstance(volumes, list) and len(volumes) > 0:
                        valid_volumes = [v for v in volumes if isinstance(v, (int, float)) and v >= 0]
                        enhanced[volume_key] = valid_volumes
                    else:
                        enhanced[volume_key] = []
        
        except Exception as e:
            logger.warning(f"Error enhancing market data: {e}")
        
        return enhanced
    
    def _json_default(self, obj):
        import numpy as np
        if hasattr(obj, 'item'):
            return obj.item()
        if isinstance(obj, (set, bytes)):
            return list(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError

    def send_signal(self, order: Order) -> bool:
        if not self.running or not self.signal_socket:
            return False
        try:
            signal = {
                "action": 1 if order.action == 'buy' else 2,
                "confidence": float(order.confidence),
                "position_size": int(max(1, order.size)),
                "use_stop": bool(order.stop_price > 0),
                "stop_price": float(order.stop_price) if order.stop_price > 0 else 0.0,
                "use_target": bool(order.target_price > 0),
                "target_price": float(order.target_price) if order.target_price > 0 else 0.0,
                "tool_used": getattr(order, 'primary_tool', 'ai_agent'),
                "risk_adjusted": True,
                "kelly_optimized": True,
                "timestamp": int(time.time() * 1e7 + 621355968000000000),
            }

            if not self._validate_signal(signal, order):
                logger.warning(f"Invalid signal rejected: {signal}")
                return False

            data = json.dumps(signal, default=self._json_default).encode()
            header = struct.pack('<I', len(data))
            self.signal_socket.sendall(header + data)
            self.signals_sent += 1
            logger.info(f"Signal sent: {order.action.upper()} {order.size} @ {order.price:.2f} "
                        f"(Conf: {order.confidence:.2f})")
            return True
        except Exception as e:
            logger.error(f"Signal send error: {e}")
            return False
    
    def _validate_signal(self, signal: dict, order: Order) -> bool:
        """Validate signal before sending to NinjaTrader"""
        try:
            # Basic validation
            if signal['action'] not in [1, 2]:
                return False
            
            if signal['position_size'] <= 0:
                return False
            
            if signal['confidence'] < 0 or signal['confidence'] > 1:
                return False
            
            # Stop/target validation
            if signal['use_stop'] and signal['stop_price'] <= 0:
                return False
            
            if signal['use_target'] and signal['target_price'] <= 0:
                return False
            
            # Price relationship validation
            if hasattr(order, 'price') and order.price > 0:
                if signal['use_stop']:
                    if order.action == 'buy' and signal['stop_price'] >= order.price:
                        return False
                    if order.action == 'sell' and signal['stop_price'] <= order.price:
                        return False
                
                if signal['use_target']:
                    if order.action == 'buy' and signal['target_price'] <= order.price:
                        return False
                    if order.action == 'sell' and signal['target_price'] >= order.price:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Signal validation error: {e}")
            return False
    
    # NEW: Method to check if ready for live trading
    def is_ready_for_live_trading(self) -> bool:
        """Check if historical data has been received and processed"""
        return self.historical_data_received
    
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