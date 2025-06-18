# tcp_bridge.py - UPDATED: Added position_size parameter support

import socket
import struct
import json
import threading
import time
import logging
from typing import Callable, Optional, Dict

log = logging.getLogger(__name__)

class TCPBridge:
    """Enhanced TCP communication bridge with NinjaTrader - AI position sizing support"""
    
    def __init__(self, config):
        self.config = config
        self.running = False
        self.connected = False
        
        # Get TCP configuration from adaptive config
        self.TCP_HOST = "localhost"
        self.FEATURE_PORT = 5556
        self.SIGNAL_PORT = 5557
        
        # Callbacks
        self.on_market_data: Optional[Callable] = None
        self.on_trade_completion: Optional[Callable] = None
        
        # Signal tracking for learning
        self.signals_sent = 0
        self.last_signal_time = 0
        
        # Initialize sockets but don't accept yet
        self._feat_srv = socket.socket()
        self._feat_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._feat_srv.bind((self.TCP_HOST, self.FEATURE_PORT))
        self._feat_srv.listen(1)

        self._sig_srv = socket.socket()
        self._sig_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sig_srv.bind((self.TCP_HOST, self.SIGNAL_PORT))
        self._sig_srv.listen(1)

        log.info(f"Enhanced TCP Bridge initialized on {self.TCP_HOST}:{self.FEATURE_PORT} (features) and {self.TCP_HOST}:{self.SIGNAL_PORT} (signals)")
        log.info("AI will determine optimal position sizing through learning")

    def start(self):
        """Start TCP server and wait for connections"""
        log.info("Waiting for NinjaTrader connection...")
        
        # Accept connections when start() is called
        try:
            # Set socket timeout to avoid hanging indefinitely
            self._feat_srv.settimeout(30.0)  # 30 second timeout
            self._sig_srv.settimeout(30.0)
            
            self.fsock, feat_addr = self._feat_srv.accept()
            log.info(f"Feature connection established from {feat_addr}")
            
            self.ssock, sig_addr = self._sig_srv.accept()
            log.info(f"Signal connection established from {sig_addr}")
            
            # Remove timeout once connected
            self.fsock.settimeout(None)
            self.ssock.settimeout(None)
            
            self.connected = True
            log.info("NinjaTrader connected successfully - AI position sizing active")
            
        except socket.timeout:
            log.error("Connection timeout - NinjaTrader not responding")
            self.connected = False
            raise ConnectionError("NinjaTrader connection timeout")
        except Exception as e:
            log.error(f"Connection establishment failed: {e}")
            self.connected = False
            raise

        # Start reader thread after connections established
        self.running = True
        threading.Thread(target=self._reader, daemon=True, name="TCPReader").start()
        log.info("Enhanced TCP reader thread started")

    def _reader(self):
        """Enhanced reader for market data and trade completions from NinjaTrader"""
        log.info("Enhanced TCP receive loop started - monitoring AI learning...")
        
        while self.running:
            try:
                # Read message header
                hdr = self.fsock.recv(4, socket.MSG_WAITALL)
                if not hdr:
                    log.warning("Connection lost, header read failed")
                    self.connected = False
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
                    
                    # Handle different message types
                    if message.get('type') == 'trade_completion':
                        # Trade completion for AI learning
                        if self.on_trade_completion:
                            self.on_trade_completion(message)
                            log.info(f"Trade completion received: {message.get('trade_id', 'unknown')} - "
                                   f"P&L: ${message.get('final_pnl', 0):.2f}, "
                                   f"Size: {message.get('position_size', 0):.2f}")
                    
                    elif "price_15m" in message:
                        # Regular market data
                        if self.on_market_data:
                            self.on_market_data(message)
                    
                    else:
                        log.debug(f"Unknown message type: {message.get('type', 'no_type')}")
                    
                except json.JSONDecodeError as e:
                    log.error(f"JSON decode error: {e}")
                    continue
                    
            except Exception as e:
                log.error(f"Receive error: {e}")
                self.connected = False
                break
        
        log.info("Enhanced TCP receive loop stopped")

    def send_signal(self, action: int, confidence: float, quality: str,
                   stop_price: float = 0.0, target_price: float = 0.0, 
                   position_size: float = 1.0):
        """
        Send AI trading signal with AI-calculated position size and risk management
        
        Args:
            action: 0=exit, 1=buy, 2=sell
            confidence: AI confidence 0.0-1.0
            quality: AI decision description
            stop_price: Actual stop loss price (0.0 = AI chose no stop)
            target_price: Actual take profit price (0.0 = AI chose no target)
            position_size: AI-calculated position size
        """
        
        if not self.connected:
            log.warning("Cannot send signal - not connected to NinjaTrader")
            return False
        
        try:
            # AI decides whether to use stops/targets based on learned experience
            use_stop = stop_price > 0.0
            use_target = target_price > 0.0
            
            # Convert to .NET Ticks (100-nanosecond intervals since 0001-01-01)
            unix_timestamp = time.time()
            net_ticks = int((unix_timestamp * 10000000) + 621355968000000000)
            
            signal = {
                "action": int(action),
                "confidence": float(confidence),
                "quality": str(quality),
                "position_size": float(position_size),     # AI-calculated position size
                "use_stop": use_stop,                      # AI decision: use stop or not
                "stop_price": float(stop_price),           # Actual price AI wants
                "use_target": use_target,                  # AI decision: use target or not
                "target_price": float(target_price),       # Actual price AI wants
                "timestamp": int(net_ticks)
            }
            
            # Send signal
            data = json.dumps(signal).encode()
            header = struct.pack('<I', len(data))
            self.ssock.sendall(header + data)
            
            # Enhanced logging
            action_name = ['EXIT', 'BUY', 'SELL'][action]
            self.signals_sent += 1
            self.last_signal_time = time.time()
            
            log.info(f"AI Signal #{self.signals_sent}: {action_name} (conf: {confidence:.3f}, size: {position_size:.2f})")
            
            # Log AI's risk management decisions
            risk_decisions = []
            if use_stop:
                risk_decisions.append(f"Stop: ${stop_price:.2f}")
            else:
                risk_decisions.append("No stop (AI choice)")
                
            if use_target:
                risk_decisions.append(f"Target: ${target_price:.2f}")
            else:
                risk_decisions.append("No target (AI choice)")
            
            log.info(f"AI Decisions: Size: {position_size:.2f} | {' | '.join(risk_decisions)}")
            
            # Track AI learning progression
            if self.signals_sent % 10 == 0:
                log.info(f"AI Learning Progress: {self.signals_sent} signals sent with AI position sizing")
            
            return True
            
        except Exception as e:
            log.error(f"Signal send error: {e}")
            self.connected = False
            return False

    def send_position_management_signal(self, action_type: str, confidence: float, 
                                      amount: float, reasoning: str):
        """
        Send advanced position management signals (scaling, partial exits)
        
        Args:
            action_type: "scale_25%", "scale_50%", "exit_25%", "exit_50%", "exit_100%"
            confidence: AI confidence in this decision
            amount: Size multiplier or exit percentage
            reasoning: AI's reasoning for this action
        """
        
        if not self.connected:
            log.warning("Cannot send position management signal - not connected")
            return False
        
        try:
            # Convert to .NET Ticks
            unix_timestamp = time.time()
            net_ticks = int((unix_timestamp * 10000000) + 621355968000000000)
            
            # Determine action code for position management
            if "scale" in action_type:
                action = 1 if "long" in reasoning.lower() else 2  # Scale in same direction
            elif "exit" in action_type:
                action = 0  # Exit signal
            else:
                action = 0  # Default to exit
            
            quality = f"AI_{action_type}_{reasoning.split()[0] if reasoning else 'unknown'}"
            
            signal = {
                "action": action,
                "confidence": float(confidence),
                "quality": quality,
                "position_size": float(amount),  # Amount for scaling/partial exit
                "use_stop": False,      # Position management doesn't set new stops
                "stop_price": 0.0,
                "use_target": False,    # Position management doesn't set new targets
                "target_price": 0.0,
                "timestamp": int(net_ticks)
            }
            
            # Send signal
            data = json.dumps(signal).encode()
            header = struct.pack('<I', len(data))
            self.ssock.sendall(header + data)
            
            self.signals_sent += 1
            log.info(f"AI Position Management #{self.signals_sent}: {action_type} (conf: {confidence:.3f}, amount: {amount:.2f})")
            log.info(f"AI Reasoning: {reasoning}")
            
            return True
            
        except Exception as e:
            log.error(f"Position management signal error: {e}")
            return False

    def get_connection_status(self) -> Dict[str, any]:
        """Get detailed connection status for monitoring"""
        return {
            "connected": self.connected,
            "running": self.running,
            "signals_sent": self.signals_sent,
            "last_signal_ago_seconds": time.time() - self.last_signal_time if self.last_signal_time > 0 else -1,
            "feature_port": self.FEATURE_PORT,
            "signal_port": self.SIGNAL_PORT
        }

    def emergency_close_signal(self):
        """Send emergency close signal"""
        if self.connected:
            log.warning("Sending EMERGENCY CLOSE signal")
            self.send_signal(0, 0.99, "EMERGENCY_CLOSE", position_size=0.0)
            time.sleep(0.5)  # Give time for signal to process

    def stop(self):
        """Enhanced stop with cleanup reporting"""
        log.info("Stopping Enhanced TCP Bridge...")
        
        # Send any final signals if needed
        if self.connected and self.signals_sent > 0:
            log.info(f"AI sent {self.signals_sent} total signals with position sizing during this session")
        
        self.running = False
        self.connected = False
        
        # Close connections with enhanced logging
        connections_closed = 0
        for name, sock in [("feature", getattr(self, 'fsock', None)), 
                          ("signal", getattr(self, 'ssock', None)), 
                          ("feature_server", getattr(self, '_feat_srv', None)), 
                          ("signal_server", getattr(self, '_sig_srv', None))]:
            try:
                if sock:
                    sock.close()
                    connections_closed += 1
                    log.debug(f"Closed {name} socket")
            except Exception as e:
                log.warning(f"Error closing {name} socket: {e}")
        
        log.info(f"Enhanced TCP Bridge stopped - {connections_closed} connections closed")
        
        if self.signals_sent > 0:
            log.info(f"Session Summary: {self.signals_sent} AI signals sent with adaptive position sizing")
            log.info("AI learning data preserved for next session")

    # Utility methods for AI learning
    def get_signal_rate(self) -> float:
        """Get signals per hour rate"""
        if self.last_signal_time == 0:
            return 0.0
        
        session_hours = (time.time() - self.last_signal_time) / 3600
        return self.signals_sent / max(session_hours, 0.1)

    def is_healthy(self) -> bool:
        """Check if connection is healthy"""
        return self.connected and self.running

    def reconnect(self) -> bool:
        """Attempt to reconnect if connection lost"""
        if self.connected:
            return True
        
        try:
            log.info("Attempting TCP reconnection...")
            self.stop()
            time.sleep(2)
            self.start()
            return self.connected
        except Exception as e:
            log.error(f"Reconnection failed: {e}")
            return False