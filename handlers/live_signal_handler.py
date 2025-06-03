# handlers/live_signal_handler.py

import time
import logging

log = logging.getLogger(__name__)

class LiveSignalHandler:
    """
    Simplified signal handler that sends pure ML signals to NinjaScript.
    All position sizing and risk management moved to NinjaScript.
    """
    
    def __init__(self, cfg, agent, portfolio, tcp, logger):
        self.cfg = cfg
        self.agent = agent
        self.portfolio = portfolio
        self.tcp = tcp
        self.logger = logger
        self.signal_counter = 0
        self.last_signal_time = 0
        
        # Simplified tracking for pure ML signals
        self.signal_history = []
        self.performance_tracker = {
            'total_signals': 0,
            'high_confidence_signals': 0,
            'excellent_quality_signals': 0,
            'good_quality_signals': 0,
            'poor_quality_signals': 0
        }

    def dispatch_signal(self, action: int, confidence: float, signal_features=None):
        """
        Dispatch pure ML signal to NinjaScript with enhanced signal quality analysis.
        NinjaScript handles all position sizing, stops, targets, and risk management.
        
        Args:
            action: Trading action (0=hold/exit, 1=long, 2=short)
            confidence: Model confidence (0.0-1.0)
            signal_features: Dict with current signal values for quality analysis
        """
        # Update position tracking (for reward calculation only)
        self.portfolio.update_position(self.tcp._current_position)

        # Analyze signal quality for NinjaScript decision making
        signal_quality = self._analyze_signal_quality(action, confidence, signal_features)

        current_time = time.time()
        self.signal_counter += 1
        timestamp = int(current_time)
        
        # Create simplified signal payload for NinjaScript
        signal = {
            "action": action,
            "confidence": round(confidence, 3),
            "signal_quality": signal_quality,  # NinjaScript uses this for exit strategy
            "timestamp": timestamp
        }

        try:
            # Send pure ML signal to NinjaScript
            self.tcp.send_signal(signal)
            self.last_signal_time = current_time
            
            # Log the signal
            action_name = "Long" if action == 1 else ("Short" if action == 2 else "Hold/Exit")
            
            log.info(f"ML Signal #{self.signal_counter}: {action_name} | "
                    f"Confidence: {confidence:.3f} | Quality: {signal_quality} | "
                    f"Timestamp: {timestamp}")
            
            # Log signal details if action signal
            if action != 0 and signal_features:
                self._log_signal_details(action_name, signal_features, signal_quality)
            
            # Update performance tracking
            self._update_performance_tracking(action, confidence, signal_quality)
            
            # Store signal history for analysis
            self._store_signal_history(signal, signal_features, signal_quality)
                    
        except Exception as e:
            log.error(f"Failed to send ML signal: {e}")

    def _analyze_signal_quality(self, action, confidence, signal_features):
        """
        Analyze signal quality for NinjaScript position management decisions.
        Returns: "excellent", "good", "mixed", "poor", or "neutral"
        """
        try:
            if action == 0:  # Hold/Exit signal
                return "neutral"
            
            if not signal_features:
                return "poor"
            
            expected_direction = 1 if action == 1 else -1
            
            # Count aligned signals with enhanced weighting
            alignment_score = 0.0
            total_weight = 0.0
            
            # Major signals (higher weight)
            major_signals = [
                ('tenkan_kijun', 0.3),
                ('price_cloud', 0.3),
                ('ema_cross', 0.25)
            ]
            
            # Minor signals (lower weight)
            minor_signals = [
                ('tenkan_momentum', 0.075),
                ('kijun_momentum', 0.075)
            ]
            
            all_signals = major_signals + minor_signals
            
            for signal_key, weight in all_signals:
                if signal_key in signal_features:
                    signal_value = signal_features[signal_key]
                    total_weight += weight
                    
                    if signal_value == expected_direction:
                        alignment_score += weight  # Full alignment
                    elif signal_value == 0:
                        alignment_score += weight * 0.5  # Neutral gets half credit
                    # Opposing signals get 0 credit
            
            # Calculate alignment percentage
            if total_weight > 0:
                alignment_pct = alignment_score / total_weight
            else:
                return "poor"
            
            # Volume and LWPE boost
            volume_boost = 0.0
            if 'normalized_volume' in signal_features:
                vol = abs(signal_features['normalized_volume'])
                if vol > 1.5:  # High volume
                    volume_boost += 0.1
            
            if 'lwpe' in signal_features:
                lwpe = signal_features['lwpe']
                # LWPE extremes indicate strong directional flow
                if abs(lwpe - 0.5) > 0.3:
                    volume_boost += 0.1
            
            # Final quality determination with confidence weighting
            final_score = (alignment_pct * 0.7) + (confidence * 0.2) + (volume_boost * 0.1)
            
            if final_score >= 0.85:
                return "excellent"
            elif final_score >= 0.7:
                return "good"
            elif final_score >= 0.5:
                return "mixed"
            else:
                return "poor"
                
        except Exception as e:
            log.debug(f"Signal quality analysis failed: {e}")
            return "poor"

    def _log_signal_details(self, action_name, signal_features, quality):
        """Log detailed ML signal information"""
        try:
            details = []
            
            # Ichimoku signals
            ichimoku_signals = []
            if signal_features.get('tenkan_kijun', 0) != 0:
                direction = "Bull" if signal_features['tenkan_kijun'] > 0 else "Bear"
                ichimoku_signals.append(f"TK:{direction}")
            
            if signal_features.get('price_cloud', 0) != 0:
                position = "Above" if signal_features['price_cloud'] > 0 else "Below"
                ichimoku_signals.append(f"Cloud:{position}")
            
            if signal_features.get('future_cloud', 0) != 0:
                color = "Green" if signal_features['future_cloud'] > 0 else "Red"
                ichimoku_signals.append(f"Future:{color}")
            
            if ichimoku_signals:
                details.append(f"Ichimoku: {','.join(ichimoku_signals)}")
            
            # EMA signal
            if signal_features.get('ema_cross', 0) != 0:
                direction = "Bull" if signal_features['ema_cross'] > 0 else "Bear"
                details.append(f"EMA:{direction}")
            
            # Momentum
            momentum_details = []
            if signal_features.get('tenkan_momentum', 0) != 0:
                direction = "+" if signal_features['tenkan_momentum'] > 0 else "-"
                momentum_details.append(f"T{direction}")
            
            if signal_features.get('kijun_momentum', 0) != 0:
                direction = "+" if signal_features['kijun_momentum'] > 0 else "-"
                momentum_details.append(f"K{direction}")
            
            if momentum_details:
                details.append(f"Momentum:{','.join(momentum_details)}")
            
            # Market condition indicators
            lwpe = signal_features.get('lwpe', 0.5)
            details.append(f"LWPE:{lwpe:.3f}")
            
            vol = signal_features.get('normalized_volume', 0)
            if abs(vol) > 1.0:
                details.append(f"Vol:{'High' if vol > 0 else 'Low'}")
            
            if details:
                log.info(f"  Signal Details: {' | '.join(details)} | Quality: {quality}")
                
        except Exception as e:
            log.debug(f"Signal detail logging failed: {e}")

    def _update_performance_tracking(self, action, confidence, quality):
        """Update performance tracking statistics"""
        try:
            self.performance_tracker['total_signals'] += 1
            
            if confidence >= 0.7:
                self.performance_tracker['high_confidence_signals'] += 1
            
            # Track quality distribution
            quality_key = f'{quality}_quality_signals'
            if quality_key in self.performance_tracker:
                self.performance_tracker[quality_key] += 1
            
        except Exception as e:
            log.debug(f"Performance tracking update failed: {e}")

    def _store_signal_history(self, signal, signal_features, quality):
        """Store signal in history for analysis"""
        try:
            history_entry = {
                'timestamp': signal['timestamp'],
                'action': signal['action'],
                'confidence': signal['confidence'],
                'quality': quality,
                'signal_features': signal_features.copy() if signal_features else None
            }
            
            self.signal_history.append(history_entry)
            
            # Keep only recent history (last 100 signals)
            if len(self.signal_history) > 100:
                self.signal_history = self.signal_history[-100:]
                
        except Exception as e:
            log.debug(f"Signal history storage failed: {e}")

    def get_performance_summary(self):
        """Get ML signal performance summary"""
        try:
            total = self.performance_tracker['total_signals']
            if total == 0:
                return "No ML signals dispatched yet"
            
            summary = f"""
            ML Signal Performance Summary:
            Total Signals: {total}
            High Confidence (≥0.7): {self.performance_tracker['high_confidence_signals']} ({self.performance_tracker['high_confidence_signals']/total*100:.1f}%)
            
            Signal Quality Distribution:
            - Excellent: {self.performance_tracker['excellent_quality_signals']} ({self.performance_tracker['excellent_quality_signals']/total*100:.1f}%)
            - Good: {self.performance_tracker['good_quality_signals']} ({self.performance_tracker['good_quality_signals']/total*100:.1f}%)
            - Poor: {self.performance_tracker['poor_quality_signals']} ({self.performance_tracker['poor_quality_signals']/total*100:.1f}%)
            
            Note: Position sizing, stops, targets handled by NinjaScript
            """
            
            # Recent signal quality trend
            if self.signal_history:
                recent_qualities = [s['quality'] for s in self.signal_history[-20:]]
                quality_dist = {}
                for q in recent_qualities:
                    quality_dist[q] = quality_dist.get(q, 0) + 1
                
                summary += f"Recent Quality Trend: {quality_dist}\n"
            
            return summary.strip()
            
        except Exception as e:
            log.warning(f"Performance summary generation failed: {e}")
            return "Performance summary unavailable"

    def get_recent_signals(self, count=10):
        """Get recent ML signals for analysis"""
        return self.signal_history[-count:] if self.signal_history else []

    def get_signal_quality_stats(self):
        """Get detailed signal quality statistics for monitoring"""
        try:
            if not self.signal_history:
                return {}
            
            recent_signals = self.signal_history[-50:]  # Last 50 signals
            
            quality_counts = {}
            confidence_by_quality = {}
            
            for signal in recent_signals:
                quality = signal['quality']
                confidence = signal['confidence']
                
                if quality not in quality_counts:
                    quality_counts[quality] = 0
                    confidence_by_quality[quality] = []
                
                quality_counts[quality] += 1
                confidence_by_quality[quality].append(confidence)
            
            # Calculate average confidence by quality
            avg_confidence_by_quality = {}
            for quality, confidences in confidence_by_quality.items():
                avg_confidence_by_quality[quality] = sum(confidences) / len(confidences)
            
            return {
                'total_recent_signals': len(recent_signals),
                'quality_distribution': quality_counts,
                'avg_confidence_by_quality': avg_confidence_by_quality,
                'overall_avg_confidence': sum(s['confidence'] for s in recent_signals) / len(recent_signals)
            }
            
        except Exception as e:
            log.warning(f"Signal quality stats generation failed: {e}")
            return {}