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
        
        # Enhanced tracking for multi-timeframe features
        self.feature_count = 0
        self.signal_count = 0
        self.position_updates = 0
        self.connection_errors = 0
        
        # Multi-timeframe feature validation
        self.expected_feature_count = 27  # Updated for multi-timeframe
        self.invalid_features = 0
        self.timeframe_validation_errors = 0
        
        # Feature distribution tracking
        self.single_timeframe_messages = 0
        self.multi_timeframe_messages = 0

        # Initialize sockets
        self._feat_srv = socket.socket()
        self._feat_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._feat_srv.bind((host, feat_port))
        self._feat_srv.listen(1)

        self._sig_srv = socket.socket()
        self._sig_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sig_srv.bind((host, sig_port))
        self._sig_srv.listen(1)

        log.info(f"Multi-Timeframe TCP Bridge initialized on {host}:{feat_port} (features) and {host}:{sig_port} (signals)")
        log.info("Waiting for NinjaTrader multi-timeframe connection...")
        
        # Accept connections
        try:
            self.fsock, feat_addr = self._feat_srv.accept()
            log.info(f"Feature connection established from {feat_addr}")
            
            self.ssock, sig_addr = self._sig_srv.accept()
            log.info(f"Signal connection established from {sig_addr}")
            
            log.info("NinjaTrader connected successfully - Multi-timeframe ML mode active (27 features)")
            
        except Exception as e:
            log.error(f"Connection establishment failed: {e}")
            raise

        # Start reader thread
        self._running = True
        threading.Thread(target=self._reader, daemon=True, name="MultiFrameTCPReader").start()
        log.info("Multi-timeframe TCP reader thread started")

    def _reader(self):
        """Enhanced reader with multi-timeframe feature validation"""
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
                if n > 15000:  # Increased limit for 27-feature messages
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
                log.warning(f"Multi-timeframe reader error #{self.connection_errors}: {e}")
                
                reconnect_attempts += 1
                if reconnect_attempts >= max_reconnect_attempts:
                    log.error(f"Max reconnection attempts ({max_reconnect_attempts}) reached")
                    break
                
                # Brief pause before retry
                threading.Event().wait(1)

        log.info("Multi-timeframe TCP reader thread stopped")

    def _process_message(self, msg):
        """Process incoming message with multi-timeframe validation"""
        try:
            if "position" in msg:
                self._handle_position_update(msg)
            elif "features" in msg:
                self._handle_multiframe_feature_update(msg)
            else:
                log.debug(f"Unknown message type: {list(msg.keys())}")
                
        except Exception as e:
            log.warning(f"Multi-timeframe message processing error: {e}")

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

    def _handle_multiframe_feature_update(self, msg):
        """Handle multi-timeframe feature vector with 27-feature validation"""
        try:
            features = msg["features"]
            live_flag = msg.get("live", 0)
            
            # Check if we have timeframe alignment data
            timeframe_alignment = msg.get("timeframe_alignment", {})
            
            # Validate feature vector for multi-timeframe
            if not self._validate_multiframe_features(features, timeframe_alignment):
                self.invalid_features += 1
                if self.invalid_features % 10 == 0:
                    log.warning(f"Invalid multi-timeframe feature count: {self.invalid_features}")
                return
            
            # Track message types
            if len(features) == 27:
                self.multi_timeframe_messages += 1
            elif len(features) == 9:
                self.single_timeframe_messages += 1
                log.debug("Received single timeframe message, will expand to multi-timeframe")
            
            # Process valid features
            self.feature_count += 1
            self.on_features(features, live_flag)
            
            # Enhanced periodic logging for multi-timeframe
            if self.feature_count % 1000 == 0:
                log.info(f"Processed {self.feature_count} multi-timeframe feature vectors")
                self._log_multiframe_statistics()
            
        except KeyError as e:
            log.warning(f"Missing key in multi-timeframe feature message: {e}")
        except Exception as e:
            log.warning(f"Multi-timeframe feature processing error: {e}")

    def _validate_multiframe_features(self, features, timeframe_alignment=None):
        """Enhanced validation for 27-feature multi-timeframe signals"""
        try:
            # Check if features is a list/array
            if not isinstance(features, (list, tuple)):
                log.debug("Features not in list/tuple format")
                return False
            
            # Handle both 27-feature and 9-feature (backward compatibility)
            if len(features) == 27:
                # Full multi-timeframe validation
                return self._validate_27_features(features, timeframe_alignment)
            elif len(features) == 9:
                # Single timeframe - will be expanded by agent
                log.debug("Received 9-feature vector, will expand to 27-feature")
                return self._validate_9_features(features)
            else:
                log.debug(f"Unexpected feature count: expected 27 or 9, got {len(features)}")
                return False
                
        except Exception as e:
            log.debug(f"Multi-timeframe feature validation error: {e}")
            return False

    def _validate_27_features(self, features, timeframe_alignment):
        """Validate full 27-feature multi-timeframe vector"""
        try:
            # Check for numeric values and clean up
            cleaned_features = []
            for i, feat in enumerate(features):
                if not isinstance(feat, (int, float)):
                    log.debug(f"Non-numeric feature at index {i}: {type(feat)}")
                    return False
                cleaned_features.append(float(feat))
            
            # Update the original features list
            features[:] = cleaned_features
            
            # Validate each timeframe's features
            timeframes = [
                (0, 9, "15m"),    # Trend context
                (9, 18, "5m"),    # Momentum context  
                (18, 27, "1m")    # Entry timing
            ]
            
            for start_idx, end_idx, timeframe_name in timeframes:
                if not self._validate_timeframe_features(features[start_idx:end_idx], timeframe_name):
                    self.timeframe_validation_errors += 1
                    log.debug(f"Validation failed for {timeframe_name} timeframe")
                    return False
            
            # Validate timeframe alignment if provided
            if timeframe_alignment:
                self._validate_timeframe_alignment(timeframe_alignment)
            
            return True
            
        except Exception as e:
            log.debug(f"27-feature validation error: {e}")
            return False

    def _validate_9_features(self, features):
        """Validate single timeframe 9-feature vector (backward compatibility)"""
        try:
            # Check for numeric values
            cleaned_features = []
            for i, feat in enumerate(features):
                if not isinstance(feat, (int, float)):
                    log.debug(f"Non-numeric feature at index {i}: {type(feat)}")
                    return False
                cleaned_features.append(float(feat))
            
            features[:] = cleaned_features
            
            # Basic sanity checks
            close_price = features[0]
            if close_price <= 0:
                log.debug(f"Invalid close price: {close_price}")
                return False
            
            # LWPE validation
            lwpe = features[8]
            if not (0 <= lwpe <= 1):
                log.debug(f"LWPE out of range: {lwpe}, clipping to [0,1]")
                features[8] = max(0, min(1, lwpe))
            
            # Signal validation for ternary signals (-1, 0, 1)
            signal_indices = [2, 3, 4, 5, 6, 7]
            signal_names = ['tenkan_kijun', 'price_cloud', 'future_cloud', 'ema_cross', 'tenkan_momentum', 'kijun_momentum']
            
            for idx, name in zip(signal_indices, signal_names):
                signal_val = features[idx]
                rounded_signal = round(signal_val)
                
                if rounded_signal not in [-1, 0, 1]:
                    log.warning(f"{name} signal out of range: {signal_val} -> clamping")
                    if rounded_signal > 1:
                        rounded_signal = 1
                    elif rounded_signal < -1:
                        rounded_signal = -1
                    else:
                        rounded_signal = 0
                
                features[idx] = float(rounded_signal)
            
            return True
            
        except Exception as e:
            log.debug(f"9-feature validation error: {e}")
            return False

    def _validate_timeframe_features(self, timeframe_features, timeframe_name):
        """Validate features for a specific timeframe"""
        try:
            if len(timeframe_features) != 9:
                log.debug(f"{timeframe_name} timeframe: expected 9 features, got {len(timeframe_features)}")
                return False
            
            # Basic sanity checks
            close_price = timeframe_features[0]
            if close_price <= 0:
                log.debug(f"{timeframe_name} timeframe: invalid close price: {close_price}")
                return False
            
            # LWPE validation
            lwpe = timeframe_features[8]
            if not (0 <= lwpe <= 1):
                log.debug(f"{timeframe_name} timeframe: LWPE out of range: {lwpe}, clipping")
                timeframe_features[8] = max(0, min(1, lwpe))
            
            # Signal validation
            signal_indices = [2, 3, 4, 5, 6, 7]
            for idx in signal_indices:
                signal_val = timeframe_features[idx]
                rounded_signal = round(signal_val)
                
                if rounded_signal not in [-1, 0, 1]:
                    log.debug(f"{timeframe_name} timeframe: signal at index {idx} out of range: {signal_val}")
                    # Clamp to valid range
                    if rounded_signal > 1:
                        rounded_signal = 1
                    elif rounded_signal < -1:
                        rounded_signal = -1
                    else:
                        rounded_signal = 0
                
                timeframe_features[idx] = float(rounded_signal)
            
            return True
            
        except Exception as e:
            log.debug(f"{timeframe_name} timeframe validation error: {e}")
            return False

    def _validate_timeframe_alignment(self, alignment):
        """Validate timeframe alignment data"""
        try:
            expected_fields = ["trend_15m", "momentum_5m", "entry_1m"]
            
            for field in expected_fields:
                if field in alignment:
                    value = alignment[field]
                    if not isinstance(value, (int, float)) or not (-1 <= value <= 1):
                        log.debug(f"Invalid {field} alignment: {value}")
                        alignment[field] = max(-1, min(1, float(value)))
            
            return True
            
        except Exception as e:
            log.debug(f"Timeframe alignment validation error: {e}")
            return False

    def send_signal(self, sig: dict):
        """Send multi-timeframe ML signal to NinjaTrader"""
        try:
            # Validate signal before sending
            if not self._validate_signal(sig):
                log.warning(f"Invalid multi-timeframe ML signal not sent: {sig}")
                return False
            
            # Serialize and send
            blob = json.dumps(sig, separators=(',', ':')).encode()
            self.ssock.sendall(struct.pack('<I', len(blob)) + blob)
            
            self.signal_count += 1
            log.debug(f"Multi-timeframe ML Signal #{self.signal_count} sent successfully")
            
            return True
            
        except BrokenPipeError:
            log.error("Signal socket broken - NinjaTrader disconnected")
            return False
        except Exception as e:
            log.warning(f"Multi-timeframe signal send error: {e}")
            return False

    def _validate_signal(self, sig):
        """Validate outgoing multi-timeframe ML signal"""
        try:
            # Required fields for multi-timeframe protocol
            required_fields = ["action", "confidence", "signal_quality", "timestamp"]
            
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
            
            # Validate signal quality for multi-timeframe
            valid_qualities = ["excellent", "good", "mixed", "poor", "neutral", "trend_filtered"]
            if sig["signal_quality"] not in valid_qualities:
                log.warning(f"Invalid signal quality: {sig['signal_quality']}")
                return False
            
            # Validate timestamp
            if not isinstance(sig["timestamp"], (int, float)) or sig["timestamp"] <= 0:
                log.warning(f"Invalid timestamp: {sig['timestamp']}")
                return False
            
            return True
            
        except Exception as e:
            log.warning(f"Multi-timeframe signal validation error: {e}")
            return False

    def _log_multiframe_statistics(self):
        """Log multi-timeframe TCP bridge statistics"""
        try:
            log.info(f"Multi-Timeframe TCP Statistics:")
            log.info(f"  Total Features: {self.feature_count}")
            log.info(f"  27-Feature Messages: {self.multi_timeframe_messages}")
            log.info(f"  9-Feature Messages: {self.single_timeframe_messages}")
            log.info(f"  ML Signals: {self.signal_count}")
            log.info(f"  Position Updates: {self.position_updates}")
            log.info(f"  Connection Errors: {self.connection_errors}")
            log.info(f"  Invalid Features: {self.invalid_features}")
            log.info(f"  Timeframe Validation Errors: {self.timeframe_validation_errors}")
                    
        except Exception as e:
            log.debug(f"Multi-timeframe statistics logging error: {e}")

    def get_status(self):
        """Get multi-timeframe TCP bridge status"""
        try:
            return {
                'feature_count': self.feature_count,
                'multi_timeframe_messages': self.multi_timeframe_messages,
                'single_timeframe_messages': self.single_timeframe_messages,
                'signal_count': self.signal_count,
                'position_updates': self.position_updates,
                'connection_errors': self.connection_errors,
                'invalid_features': self.invalid_features,
                'timeframe_validation_errors': self.timeframe_validation_errors,
                'current_position': self._current_position,
                'connected': self.fsock is not None and self.ssock is not None,
                'running': self._running,
                'protocol_version': 'multi_timeframe_v1.0',
                'expected_features': self.expected_feature_count
            }
        except:
            return {'error': 'Multi-timeframe status unavailable'}

    def close(self):
        """Close all connections"""
        log.info("Closing multi-timeframe TCP bridge connections")
        self._running = False
        
        for name, sock in [("feature", self.fsock), ("signal", self.ssock), 
                          ("feature_server", self._feat_srv), ("signal_server", self._sig_srv)]:
            try:
                if sock:
                    sock.close()
                    log.debug(f"Closed {name} socket")
            except Exception as e:
                log.warning(f"Error closing {name} socket: {e}")
        
        log.info("Multi-timeframe TCP bridge closed")

    def get_timeframe_analysis(self):
        """Get analysis of timeframe message distribution"""
        try:
            total_messages = self.multi_timeframe_messages + self.single_timeframe_messages
            
            if total_messages == 0:
                return {
                    'multi_timeframe_percentage': 0,
                    'single_timeframe_percentage': 0,
                    'recommendation': 'No messages received yet'
                }
            
            multi_pct = (self.multi_timeframe_messages / total_messages) * 100
            single_pct = (self.single_timeframe_messages / total_messages) * 100
            
            recommendation = ""
            if multi_pct > 80:
                recommendation = "Excellent multi-timeframe coverage"
            elif multi_pct > 50:
                recommendation = "Good multi-timeframe usage"
            elif single_pct > 80:
                recommendation = "Consider enabling multi-timeframe in NinjaScript"
            else:
                recommendation = "Mixed timeframe usage detected"
            
            return {
                'multi_timeframe_percentage': multi_pct,
                'single_timeframe_percentage': single_pct,
                'total_messages': total_messages,
                'recommendation': recommendation
            }
            
        except Exception as e:
            log.warning(f"Timeframe analysis error: {e}")
            return {'error': 'Analysis unavailable'}