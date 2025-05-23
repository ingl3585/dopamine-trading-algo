# handlers/live_signal_handler.py

import time
import logging

log = logging.getLogger(__name__)

class LiveSignalHandler:
    def __init__(self, cfg, agent, portfolio, tcp, logger):
        self.cfg = cfg
        self.agent = agent
        self.portfolio = portfolio
        self.tcp = tcp
        self.logger = logger
        self.signal_counter = 0
        self.last_signal_time = 0
        
        # Enhanced tracking for Ichimoku/EMA signals
        self.signal_history = []
        self.performance_tracker = {
            'total_signals': 0,
            'ichimoku_signals': 0,
            'ema_signals': 0,
            'mixed_signals': 0,
            'high_confidence_signals': 0
        }

    def dispatch_signal(self, action: int, confidence: float, signal_features=None):
        """
        Dispatch trading signal with enhanced Ichimoku/EMA analysis
        
        Args:
            action: Trading action (0=hold, 1=long, 2=short)
            confidence: Model confidence
            signal_features: Dict with current Ichimoku/EMA signal values
        """
        # Update current position
        self.portfolio.update_position(self.tcp._current_position)

        # Extract signal data for position sizing
        signal_data = self._extract_signal_data(signal_features)
        
        # Calculate position size with signal enhancement
        size = self.portfolio.calculate_position_size(action, confidence, signal_data)

        # Analyze signal quality
        signal_quality = self._analyze_signal_quality(action, confidence, signal_data)

        current_time = time.time()
        self.signal_counter += 1
        timestamp = int(current_time)
        
        # Create signal payload
        sig = {
            "action": action,
            "confidence": round(confidence, 3),
            "size": size,
            "timestamp": timestamp,
            "signal_id": self.signal_counter
        }

        try:
            # Send signal to NinjaTrader
            self.tcp.send_signal(sig)
            self.last_signal_time = current_time
            
            # Enhanced logging with signal analysis
            action_name = "Long" if action == 1 else ("Short" if action == 2 else "Hold")
            
            log.info(f"Signal sent - {action_name}: size={size}, conf={confidence:.3f}, "
                    f"quality={signal_quality}, id={self.signal_counter}, timestamp={timestamp}")
            
            # Log signal details if significant
            if size > 0 and signal_data:
                self._log_signal_details(action_name, signal_data, signal_quality)
            
            # Update performance tracking
            self._update_performance_tracking(action, confidence, signal_data, size)
            
            # Store signal history
            self._store_signal_history(sig, signal_data, signal_quality)
                    
        except Exception as e:
            log.error(f"Failed to send signal: {e}")

    def _extract_signal_data(self, signal_features):
        """Extract signal data from feature processor"""
        if not signal_features:
            return None
        
        try:
            # Map feature processor signals to portfolio format
            return {
                'tenkan_kijun_signal': signal_features.get('tenkan_kijun', 0),
                'price_cloud_signal': signal_features.get('price_cloud', 0),
                'future_cloud_signal': signal_features.get('future_cloud', 0),
                'ema_cross_signal': signal_features.get('ema_cross', 0),
                'tenkan_momentum': signal_features.get('tenkan_momentum', 0),
                'kijun_momentum': signal_features.get('kijun_momentum', 0),
                'normalized_volume': signal_features.get('normalized_volume', 0),
                'lwpe': signal_features.get('lwpe', 0.5)
            }
        except Exception as e:
            log.warning(f"Signal data extraction failed: {e}")
            return None

    def _analyze_signal_quality(self, action, confidence, signal_data):
        """Analyze the quality of the trading signal"""
        try:
            if action == 0 or not signal_data:
                return "neutral"
            
            expected_direction = 1 if action == 1 else -1
            
            # Count aligned signals
            aligned_signals = 0
            total_signals = 0
            
            signal_checks = [
                ('tenkan_kijun_signal', 'Tenkan/Kijun'),
                ('price_cloud_signal', 'Price/Cloud'),
                ('ema_cross_signal', 'EMA Cross'),
                ('tenkan_momentum', 'Tenkan Momentum'),
                ('kijun_momentum', 'Kijun Momentum')
            ]
            
            for signal_key, signal_name in signal_checks:
                if signal_key in signal_data and signal_data[signal_key] != 0:
                    total_signals += 1
                    if signal_data[signal_key] == expected_direction:
                        aligned_signals += 1
            
            if total_signals == 0:
                return "no_signals"
            
            alignment_ratio = aligned_signals / total_signals
            
            # Determine signal quality
            if alignment_ratio >= 0.8 and confidence >= 0.7:
                return "excellent"
            elif alignment_ratio >= 0.6 and confidence >= 0.6:
                return "good"
            elif alignment_ratio >= 0.4:
                return "mixed"
            else:
                return "poor"
                
        except Exception as e:
            log.debug(f"Signal quality analysis failed: {e}")
            return "unknown"

    def _log_signal_details(self, action_name, signal_data, quality):
        """Log detailed signal information"""
        try:
            details = []
            
            # Ichimoku signals
            if signal_data.get('tenkan_kijun_signal', 0) != 0:
                direction = "Bull" if signal_data['tenkan_kijun_signal'] > 0 else "Bear"
                details.append(f"TenkanKijun:{direction}")
            
            if signal_data.get('price_cloud_signal', 0) != 0:
                position = "Above" if signal_data['price_cloud_signal'] > 0 else "Below"
                details.append(f"Cloud:{position}")
            
            # EMA signal
            if signal_data.get('ema_cross_signal', 0) != 0:
                direction = "Bull" if signal_data['ema_cross_signal'] > 0 else "Bear"
                details.append(f"EMA:{direction}")
            
            # Momentum
            momentum_details = []
            if signal_data.get('tenkan_momentum', 0) != 0:
                direction = "+" if signal_data['tenkan_momentum'] > 0 else "-"
                momentum_details.append(f"T{direction}")
            
            if signal_data.get('kijun_momentum', 0) != 0:
                direction = "+" if signal_data['kijun_momentum'] > 0 else "-"
                momentum_details.append(f"K{direction}")
            
            if momentum_details:
                details.append(f"Momentum:{','.join(momentum_details)}")
            
            # Volume and LWPE
            lwpe = signal_data.get('lwpe', 0.5)
            details.append(f"LWPE:{lwpe:.3f}")
            
            if details:
                log.info(f"  Signal details: {' | '.join(details)} | Quality: {quality}")
                
        except Exception as e:
            log.debug(f"Signal detail logging failed: {e}")

    def _update_performance_tracking(self, action, confidence, signal_data, size):
        """Update performance tracking statistics"""
        try:
            self.performance_tracker['total_signals'] += 1
            
            if size > 0:  # Only count actual trades
                # Count signal types
                if signal_data:
                    ichimoku_count = sum(1 for key in ['tenkan_kijun_signal', 'price_cloud_signal'] 
                                       if signal_data.get(key, 0) != 0)
                    ema_count = 1 if signal_data.get('ema_cross_signal', 0) != 0 else 0
                    
                    if ichimoku_count >= 2:
                        self.performance_tracker['ichimoku_signals'] += 1
                    elif ema_count > 0:
                        self.performance_tracker['ema_signals'] += 1
                    else:
                        self.performance_tracker['mixed_signals'] += 1
                
                if confidence >= 0.7:
                    self.performance_tracker['high_confidence_signals'] += 1
            
        except Exception as e:
            log.debug(f"Performance tracking update failed: {e}")

    def _store_signal_history(self, signal, signal_data, quality):
        """Store signal in history for analysis"""
        try:
            history_entry = {
                'timestamp': signal['timestamp'],
                'action': signal['action'],
                'confidence': signal['confidence'],
                'size': signal['size'],
                'quality': quality,
                'signal_data': signal_data.copy() if signal_data else None
            }
            
            self.signal_history.append(history_entry)
            
            # Keep only recent history (last 100 signals)
            if len(self.signal_history) > 100:
                self.signal_history = self.signal_history[-100:]
                
        except Exception as e:
            log.debug(f"Signal history storage failed: {e}")

    def get_performance_summary(self):
        """Get performance summary"""
        try:
            total = self.performance_tracker['total_signals']
            if total == 0:
                return "No signals dispatched yet"
            
            summary = f"""
                Signal Performance Summary:
                Total Signals: {total}
                Ichimoku-based: {self.performance_tracker['ichimoku_signals']} ({self.performance_tracker['ichimoku_signals']/total*100:.1f}%)
                EMA-based: {self.performance_tracker['ema_signals']} ({self.performance_tracker['ema_signals']/total*100:.1f}%)
                Mixed Signals: {self.performance_tracker['mixed_signals']} ({self.performance_tracker['mixed_signals']/total*100:.1f}%)
                High Confidence: {self.performance_tracker['high_confidence_signals']} ({self.performance_tracker['high_confidence_signals']/total*100:.1f}%)
                """
            
            # Recent signal quality distribution
            if self.signal_history:
                recent_qualities = [s['quality'] for s in self.signal_history[-20:]]
                quality_dist = {}
                for q in recent_qualities:
                    quality_dist[q] = quality_dist.get(q, 0) + 1
                
                summary += f"  Recent Quality: {quality_dist}\n"
            
            return summary.strip()
            
        except Exception as e:
            log.warning(f"Performance summary generation failed: {e}")
            return "Performance summary unavailable"

    def get_recent_signals(self, count=10):
        """Get recent signals for analysis"""
        return self.signal_history[-count:] if self.signal_history else []