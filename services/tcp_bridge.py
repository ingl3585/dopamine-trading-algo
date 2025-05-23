# services/tcp_bridge.py

import socket
import struct
import json
import threading
import logging

log = logging.getLogger(__name__)

class TCPBridge:
    def __init__(self, host: str, feat_port: int, sig_port: int):
        self.host = host
        self.feat_port = feat_port
        self.sig_port = sig_port
        
        self.on_features = lambda *args: None
        self._current_position = 0 
        
        # Enhanced tracking for Ichimoku/EMA features
        self.feature_count = 0
        self.signal_count = 0
        self.position_updates = 0
        self.connection_errors = 0
        
        # Feature validation
        self.expected_feature_count = 9  # Updated for Ichimoku/EMA
        self.invalid_features = 0

        # Initialize sockets
        self._feat_srv = socket.socket()
        self._feat_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._feat_srv.bind((host, feat_port))
        self._feat_srv.listen(1)

        self._sig_srv = socket.socket()
        self._sig_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sig_srv.bind((host, sig_port))
        self._sig_srv.listen(1)

        log.info(f"TCP Bridge initialized on {host}:{feat_port} (features) and {host}:{sig_port} (signals)")
        log.info("Waiting for NinjaTrader connection...")
        
        # Accept connections
        try:
            self.fsock, feat_addr = self._feat_srv.accept()
            log.info(f"Feature connection established from {feat_addr}")
            
            self.ssock, sig_addr = self._sig_srv.accept()
            log.info(f"Signal connection established from {sig_addr}")
            
            log.info("NinjaTrader connected successfully")
            
        except Exception as e:
            log.error(f"Connection establishment failed: {e}")
            raise

        # Start reader thread
        self._running = True
        threading.Thread(target=self._reader, daemon=True, name="TCPReader").start()
        log.info("TCP reader thread started")

    def _reader(self):
        """Enhanced reader with better error handling and feature validation"""
        stream = self.fsock
        reconnect_attempts = 0
        max_reconnect_attempts = 5
        
        while self._running:
            try:
                # Read message header
                hdr = stream.recv(4, socket.MSG_WAITALL)
                if not hdr:
                    log.warning("Connection lost, header read failed")
                    break
                    
                n = struct.unpack('<I', hdr)[0]
                
                # Validate message size
                if n > 10000:  # Reasonable limit
                    log.warning(f"Suspicious message size: {n} bytes")
                    continue
                
                # Read message data
                data = stream.recv(n, socket.MSG_WAITALL)
                if len(data) != n:
                    log.warning(f"Incomplete message: expected {n}, got {len(data)}")
                    continue
                
                # Parse and process message
                try:
                    msg = json.loads(data.decode())
                    self._process_message(msg)
                    reconnect_attempts = 0  # Reset on successful message
                    
                except json.JSONDecodeError as e:
                    log.warning(f"JSON decode error: {e}")
                    continue
                    
            except ConnectionResetError:
                log.error("Connection reset by NinjaTrader")
                break
            except Exception as e:
                self.connection_errors += 1
                log.warning(f"Reader error #{self.connection_errors}: {e}")
                
                reconnect_attempts += 1
                if reconnect_attempts >= max_reconnect_attempts:
                    log.error(f"Max reconnection attempts ({max_reconnect_attempts}) reached")
                    break
                
                # Brief pause before retry
                threading.Event().wait(1)

        log.info("TCP reader thread stopped")

    def _process_message(self, msg):
        """Process incoming message with enhanced validation"""
        try:
            if "position" in msg:
                self._handle_position_update(msg)
            elif "features" in msg:
                self._handle_feature_update(msg)
            else:
                log.debug(f"Unknown message type: {list(msg.keys())}")
                
        except Exception as e:
            log.warning(f"Message processing error: {e}")

    def _handle_position_update(self, msg):
        """Handle position update from NinjaTrader"""
        try:
            new_position = int(msg["position"])
            
            if new_position != self._current_position:
                log.debug(f"Position update: {self._current_position} -> {new_position}")
                self._current_position = new_position
                self.position_updates += 1
            
        except (ValueError, KeyError) as e:
            log.warning(f"Invalid position update: {e}")

    def _handle_feature_update(self, msg):
        """Handle feature vector with Ichimoku/EMA validation"""
        try:
            features = msg["features"]
            live_flag = msg.get("live", 0)
            
            # Validate feature vector
            if not self._validate_features(features):
                self.invalid_features += 1
                if self.invalid_features % 10 == 0:
                    log.warning(f"Invalid feature count: {self.invalid_features}")
                return
            
            # Process valid features
            self.feature_count += 1
            self.on_features(features, live_flag)
            
            # Periodic logging
            if self.feature_count % 1000 == 0:
                log.info(f"Processed {self.feature_count} feature vectors")
                self._log_statistics()
            
        except KeyError as e:
            log.warning(f"Missing key in feature message: {e}")
        except Exception as e:
            log.warning(f"Feature processing error: {e}")

    def _validate_features(self, features):
        """Enhanced validation for ternary Ichimoku/EMA signals"""
        try:
            # Check if features is a list/array
            if not isinstance(features, (list, tuple)):
                log.debug("Features not in list/tuple format")
                return False
            
            # Check feature count
            if len(features) != self.expected_feature_count:
                log.debug(f"Feature count mismatch: expected {self.expected_feature_count}, got {len(features)}")
                return False
            
            # Check for numeric values and clean up
            cleaned_features = []
            for i, feat in enumerate(features):
                if not isinstance(feat, (int, float)):
                    log.debug(f"Non-numeric feature at index {i}: {type(feat)}")
                    return False
                cleaned_features.append(float(feat))
            
            # Update the original features list
            features[:] = cleaned_features
            
            # Basic sanity checks for specific features
            close_price = features[0]
            if close_price <= 0:
                log.debug(f"Invalid close price: {close_price}")
                return False
            
            # Enhanced LWPE validation
            lwpe = features[8]
            if not (0 <= lwpe <= 1):
                log.debug(f"LWPE out of range: {lwpe}, clipping to [0,1]")
                features[8] = max(0, min(1, lwpe))  # Clip to valid range
            
            # Enhanced signal validation for ternary signals (-1, 0, 1)
            signal_indices = [2, 3, 4, 5, 6, 7]  # All signal features
            signal_names = ['tenkan_kijun', 'price_cloud', 'future_cloud', 'ema_cross', 'tenkan_momentum', 'kijun_momentum']
            
            for idx, name in zip(signal_indices, signal_names):
                if idx < len(features):
                    signal_val = features[idx]
                    
                    # Round to nearest valid signal value for floating point precision issues
                    rounded_signal = round(signal_val)
                    
                    # Validate range
                    if rounded_signal not in [-1, 0, 1]:
                        log.warning(f"{name} signal out of range: {signal_val} -> clamping")
                        # Clamp to valid range
                        if rounded_signal > 1:
                            rounded_signal = 1
                        elif rounded_signal < -1:
                            rounded_signal = -1
                        else:
                            rounded_signal = 0
                    
                    features[idx] = float(rounded_signal)
            
            return True
            
        except Exception as e:
            log.debug(f"Feature validation error: {e}")
            return False

    def send_signal(self, sig: dict):
        """Send trading signal to NinjaTrader with enhanced error handling"""
        try:
            # Validate signal before sending
            if not self._validate_signal(sig):
                log.warning(f"Invalid signal not sent: {sig}")
                return False
            
            # Serialize and send
            blob = json.dumps(sig, separators=(',', ':')).encode()
            self.ssock.sendall(struct.pack('<I', len(blob)) + blob)
            
            self.signal_count += 1
            log.debug(f"Signal #{self.signal_count} sent successfully")
            
            return True
            
        except BrokenPipeError:
            log.error("Signal socket broken - NinjaTrader disconnected")
            return False
        except Exception as e:
            log.warning(f"Signal send error: {e}")
            return False

    def _validate_signal(self, sig):
        """Validate outgoing signal"""
        try:
            required_fields = ["action", "confidence", "size", "timestamp", "signal_id"]
            
            for field in required_fields:
                if field not in sig:
                    log.warning(f"Missing signal field: {field}")
                    return False
            
            # Validate field values
            if sig["action"] not in [0, 1, 2]:
                log.warning(f"Invalid action: {sig['action']}")
                return False
            
            if not (0 <= sig["confidence"] <= 1):
                log.warning(f"Invalid confidence: {sig['confidence']}")
                return False
            
            if sig["size"] < 0 or sig["size"] > 100:
                log.warning(f"Invalid size: {sig['size']}")
                return False
            
            return True
            
        except Exception as e:
            log.warning(f"Signal validation error: {e}")
            return False

    def _log_statistics(self):
        """Log TCP bridge statistics"""
        try:
            log.info(f"TCP Statistics: Features={self.feature_count}, "
                    f"Signals={self.signal_count}, "
                    f"Position Updates={self.position_updates}, "
                    f"Errors={self.connection_errors}, "
                    f"Invalid Features={self.invalid_features}")
                    
        except Exception as e:
            log.debug(f"Statistics logging error: {e}")

    def get_status(self):
        """Get TCP bridge status"""
        try:
            return {
                'feature_count': self.feature_count,
                'signal_count': self.signal_count,
                'position_updates': self.position_updates,
                'connection_errors': self.connection_errors,
                'invalid_features': self.invalid_features,
                'current_position': self._current_position,
                'connected': self.fsock is not None and self.ssock is not None,
                'running': self._running
            }
        except:
            return {'error': 'Status unavailable'}

    def close(self):
        """Close all connections"""
        log.info("Closing TCP bridge connections")
        self._running = False
        
        for name, sock in [("feature", self.fsock), ("signal", self.ssock), 
                          ("feature_server", self._feat_srv), ("signal_server", self._sig_srv)]:
            try:
                if sock:
                    sock.close()
                    log.debug(f"Closed {name} socket")
            except Exception as e:
                log.warning(f"Error closing {name} socket: {e}")
        
        log.info("TCP bridge closed")