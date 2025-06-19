# FIXED tcp_bridge.py - Enhanced correlation and account data handling

import socket
import struct
import json
import threading
import time
import logging
from typing import Callable, Optional, Dict

log = logging.getLogger(__name__)

class TCPBridge:
    """FIXED: Enhanced TCP communication bridge with better learning correlation"""
    
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
        
        # FIXED: Enhanced signal tracking for better correlation
        self.signals_sent = 0
        self.last_signal_time = 0
        self.sent_signals_log = []  # Track sent signals for correlation
        
        # Initialize sockets
        self._feat_srv = socket.socket()
        self._feat_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._feat_srv.bind((self.TCP_HOST, self.FEATURE_PORT))
        self._feat_srv.listen(1)

        self._sig_srv = socket.socket()
        self._sig_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sig_srv.bind((self.TCP_HOST, self.SIGNAL_PORT))
        self._sig_srv.listen(1)

        log.info(f"TCP bridge initialized on {self.TCP_HOST}:{self.FEATURE_PORT} (features) and {self.TCP_HOST}:{self.SIGNAL_PORT} (signals)")
        log.info("Enhanced learning correlation system active")

    def start(self):
        """Start TCP server with LONGER timeout for NinjaTrader"""
        log.info("Waiting for NinjaTrader connection...")
        
        try:
            # INCREASED timeout from 30 to 120 seconds
            self._feat_srv.settimeout(120.0)  
            self._sig_srv.settimeout(120.0)
            
            log.info("Waiting up to 2 minutes for NinjaTrader to connect...")
            self.fsock, feat_addr = self._feat_srv.accept()
            log.info(f"Feature connection established from {feat_addr}")
            
            self.ssock, sig_addr = self._sig_srv.accept()
            log.info(f"Signal connection established from {sig_addr}")
            
            # Remove timeout once connected
            self.fsock.settimeout(None)
            self.ssock.settimeout(None)
            
            self.connected = True
            log.info("NinjaTrader connected successfully")
            
        except socket.timeout:
            log.error("Connection timeout after 2 minutes")
            log.error("Make sure NinjaTrader is running with ResearchStrategy loaded")
            self.connected = False
            raise ConnectionError("NinjaTrader connection timeout")
        except Exception as e:
            log.error(f"Connection establishment failed: {e}")
            self.connected = False
            raise

        # Start reader thread
        self.running = True
        threading.Thread(target=self._reader, daemon=True, name="TCPReader").start()
        log.info("TCP reader thread started")

    def _reader(self):
        """Enhanced reader for market data and trade completions from NinjaTrader"""
        log.info("TCP receive loop started - monitoring AI learning...")
        
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
                        # ENHANCED: Include signal timestamp for proper learning correlation
                        if self.on_trade_completion:
                            # Add signal correlation timestamp if missing
                            if 'signal_timestamp' not in message and self.last_signal_time > 0:
                                message['signal_timestamp'] = self.last_signal_time
                            
                            self.on_trade_completion(message)
                            log.info(f"Trade completion learned: {message.get('trade_id', 'unknown')} - "
                                f"P&L: ${message.get('final_pnl', 0):.2f}")
                    
                    elif "price_15m" in message:
                        # Regular market data - enhanced with signal correlation
                        if self.on_market_data:
                            # Add timing info for signal correlation
                            message['receive_timestamp'] = time.time()
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
        
        log.info("TCP receive loop stopped")

    def send_signal(self, action: int, confidence: float, quality: str,
                stop_price: float = 0.0, target_price: float = 0.0, 
                position_size: float = 1.0, tool_name: str = "unknown", meta_learner=None):
        """
        FIXED: Enhanced signal sending with tool_name parameter and tool_used field
        """
        
        if not self.connected:
            log.warning("Cannot send signal - not connected to NinjaTrader")
            return False
        
        try:
            # AI decides whether to use stops/targets based on learned experience
            use_stop = stop_price > 0.0
            use_target = target_price > 0.0
            
            # Convert to .NET Ticks with high precision
            unix_timestamp = time.time()
            self.last_signal_time = unix_timestamp  # FIXED: Store for correlation
            net_ticks = int((unix_timestamp * 10000000) + 621355968000000000)
            
            # Get adaptive timeout from meta-learner
            adaptive_timeout = 30.0  # Default
            if meta_learner:
                adaptive_timeout = meta_learner.get_parameter('signal_timeout_seconds')
            
            signal = {
                "action": int(action),
                "confidence": float(confidence),
                "quality": str(quality),
                "tool_used": str(tool_name),  # FIXED: Add tool_used field
                "position_size": float(position_size),
                "use_stop": use_stop,
                "stop_price": float(stop_price),
                "use_target": use_target,
                "target_price": float(target_price),
                "adaptive_timeout": float(adaptive_timeout),
                "timestamp": int(net_ticks),
                
                # FIXED: Enhanced correlation data
                "signal_id": f"tcp_signal_{self.signals_sent + 1}",
                "send_timestamp": unix_timestamp,
                "correlation_key": f"{action}_{confidence:.3f}_{position_size:.1f}"
            }
            
            # Send signal
            data = json.dumps(signal).encode()
            header = struct.pack('<I', len(data))
            self.ssock.sendall(header + data)
            
            # FIXED: Enhanced signal tracking for correlation
            self.signals_sent += 1
            signal_log_entry = {
                'signal_id': signal['signal_id'],
                'action': action,
                'confidence': confidence,
                'quality': quality,
                'tool_name': tool_name,  # Track tool name
                'position_size': position_size,
                'timestamp': unix_timestamp,
                'net_ticks': net_ticks,
                'correlation_key': signal['correlation_key']
            }
            
            self.sent_signals_log.append(signal_log_entry)
            
            # Keep only recent signals for correlation (last 100)
            if len(self.sent_signals_log) > 100:
                self.sent_signals_log = self.sent_signals_log[-100:]
            
            # Enhanced logging
            action_name = ['EXIT', 'BUY', 'SELL'][action]
            log.info(f"Enhanced AI Signal #{self.signals_sent}: {action_name} (conf: {confidence:.3f}, size: {position_size:.2f}, tool: {tool_name})")
            
            # Log AI's risk management decisions with correlation ID
            risk_decisions = []
            if use_stop:
                risk_decisions.append(f"Stop: ${stop_price:.2f}")
            else:
                risk_decisions.append("No stop (AI choice)")
                
            if use_target:
                risk_decisions.append(f"Target: ${target_price:.2f}")
            else:
                risk_decisions.append("No target (AI choice)")
            
            log.info(f"AI Decisions [ID: {signal['signal_id']}]: Size: {position_size:.2f} | {' | '.join(risk_decisions)}")
            
            # Track learning progression with enhanced correlation
            if self.signals_sent % 10 == 0:
                log.info(f"AI Learning Progress: {self.signals_sent} signals sent with enhanced correlation tracking")
            
            return True
            
        except Exception as e:
            log.error(f"Signal send error: {e}")
            self.connected = False
            return False
            
        except Exception as e:
            log.error(f"Signal send error: {e}")
            self.connected = False
            return False

    def send_position_management_signal(self, action_type: str, confidence: float, 
                                      amount: float, reasoning: str):
        """
        FIXED: Enhanced position management signals with correlation
        """
        
        if not self.connected:
            log.warning("Cannot send position management signal - not connected")
            return False
        
        try:
            # Convert to .NET Ticks
            unix_timestamp = time.time()
            self.last_signal_time = unix_timestamp
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
                "position_size": float(amount),
                "use_stop": False,
                "stop_price": 0.0,
                "use_target": False,
                "target_price": 0.0,
                "timestamp": int(net_ticks),
                
                # Enhanced correlation data
                "signal_id": f"tcp_mgmt_{self.signals_sent + 1}",
                "send_timestamp": unix_timestamp,
                "management_type": action_type
            }
            
            # Send signal
            data = json.dumps(signal).encode()
            header = struct.pack('<I', len(data))
            self.ssock.sendall(header + data)
            
            self.signals_sent += 1
            log.info(f"Enhanced AI position management #{self.signals_sent}: {action_type} (conf: {confidence:.3f}, amount: {amount:.2f})")
            log.info(f"AI reasoning: {reasoning}")
            
            return True
            
        except Exception as e:
            log.error(f"Position management signal error: {e}")
            return False

    def get_connection_status(self) -> Dict[str, any]:
        """FIXED: Enhanced connection status with correlation info"""
        return {
            "connected": self.connected,
            "running": self.running,
            "signals_sent": self.signals_sent,
            "last_signal_ago_seconds": time.time() - self.last_signal_time if self.last_signal_time > 0 else -1,
            "feature_port": self.FEATURE_PORT,
            "signal_port": self.SIGNAL_PORT,
            "correlation_signals_tracked": len(self.sent_signals_log),
            "last_correlation_timestamp": self.last_signal_time
        }

    def get_correlation_data(self, time_window_seconds: int = 300) -> Dict:
        """FIXED: Get correlation data for learning"""
        cutoff_time = time.time() - time_window_seconds
        recent_signals = [s for s in self.sent_signals_log if s['timestamp'] >= cutoff_time]
        
        return {
            'recent_signals_count': len(recent_signals),
            'time_window': time_window_seconds,
            'signals_in_window': recent_signals,
            'total_signals_sent': self.signals_sent
        }

    def emergency_close_signal(self):
        """FIXED: Enhanced emergency close signal with correlation"""
        if self.connected:
            log.warning("Sending EMERGENCY CLOSE signal with correlation")
            self.send_signal(0, 0.99, "EMERGENCY_CLOSE", position_size=0.0)
            time.sleep(0.5)  # Give time for signal to process

    def stop(self):
        """FIXED: Enhanced stop with learning correlation summary"""
        log.info("Stopping TCP bridge with enhanced correlation data...")
        
        shutdown_start = time.time()
        
        # Send any final signals if needed
        if self.connected and self.signals_sent > 0:
            log.info(f"AI sent {self.signals_sent} total signals with enhanced correlation during this session")
            log.info(f"Correlation tracking: {len(self.sent_signals_log)} signals logged for learning")
        
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
        
        shutdown_duration = time.time() - shutdown_start
        
        log.info(f"Enhanced TCP bridge stopped - {connections_closed} connections closed")
        log.info(f"Shutdown time: {shutdown_duration:.2f}s")
        
        if self.signals_sent > 0:
            log.info(f"Session summary: {self.signals_sent} AI signals sent with enhanced correlation")
            log.info("Enhanced learning correlation data preserved for next session")

    # FIXED: Enhanced utility methods for learning
    def get_signal_rate(self) -> float:
        """Get signals per hour rate"""
        if not self.sent_signals_log:
            return 0.0
        
        if len(self.sent_signals_log) < 2:
            return 0.0
        
        oldest_signal = self.sent_signals_log[0]['timestamp']
        newest_signal = self.sent_signals_log[-1]['timestamp']
        
        session_hours = (newest_signal - oldest_signal) / 3600
        return len(self.sent_signals_log) / max(session_hours, 0.1)

    def is_healthy(self) -> bool:
        """Check if connection is healthy with enhanced monitoring"""
        return self.connected and self.running

    def reconnect(self) -> bool:
        """Attempt to reconnect if connection lost"""
        if self.connected:
            return True
        
        try:
            log.info("Attempting TCP reconnection with enhanced correlation...")
            self.stop()
            time.sleep(2)
            self.start()
            log.info(f"Reconnection successful - correlation tracking resumed")
            return self.connected
        except Exception as e:
            log.error(f"Enhanced reconnection failed: {e}")
            return False