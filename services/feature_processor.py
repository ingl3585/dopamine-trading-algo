# services/feature_processor.py

import time
import logging

from model.reward import RewardCalculator
from model.agent import RLAgent

log = logging.getLogger(__name__)

class FeatureProcessor:
    def __init__(self, agent: RLAgent, rewarder: RewardCalculator):
        self.agent = agent
        self.rewarder = rewarder
        self.last_price = None
        self.last_features = None
        self.last_action = None
        self.last_signals = None
        
        # Multi-timeframe tracking
        self.last_15m_features = None
        self.last_5m_features = None
        self.last_1m_features = None
        
        # Feature statistics for analysis
        self.feature_stats = {
            'total_processed': 0,
            'multi_timeframe_count': 0,
            'single_timeframe_count': 0,
            'timeframe_alignment_count': 0,
            'trend_filter_blocks': 0
        }

    def process_and_predict(self, feat):
        """
        Process new multi-timeframe feature vector with 27 features
        
        Expected feature vector (27 elements):
        [0-8]:   15-minute features (trend context)
        [9-17]:  5-minute features (momentum context)  
        [18-26]: 1-minute features (entry timing)
        """
        
        # Handle both 27-feature and 9-feature inputs
        if len(feat) == 9:
            log.debug("Received 9-feature vector, expanding to 27-feature format")
            feat = self._expand_to_27_features(feat)
        elif len(feat) != 27:
            log.warning(f"Expected 27 features, got {len(feat)}. Normalizing.")
            feat = self._normalize_feature_vector(feat)
        
        close = feat[18]  # Use 1-minute close as primary price
        lwpe_1m = feat[26] if len(feat) > 26 else 0.5
        
        # Extract multi-timeframe signals for reward calculation
        current_signals = self._extract_multiframe_signals(feat)

        reward = 0.0
        if self.last_price is not None and self.last_action is not None:
            price_change = close - self.last_price

            # Use 1-minute LWPE as volatility proxy
            volatility_proxy = max(abs(lwpe_1m - 0.5) * 2, 0.01)

            if self.last_action == 1:  # Long
                reward = self.rewarder.compute_reward(
                    price_change, volatility_proxy, self.last_action, self.last_signals
                )
            elif self.last_action == 2:  # Short
                reward = self.rewarder.compute_reward(
                    -price_change, volatility_proxy, self.last_action, self.last_signals
                )
            else:  # Hold
                reward = self.rewarder.compute_reward(
                    0, volatility_proxy, self.last_action, self.last_signals
                )
            
            # Add experience to agent
            if self.last_features is not None:
                self.agent.add_experience(self.last_features, self.last_action, reward, feat)

        # Get new action and confidence with multi-timeframe analysis
        action, conf = self.agent.predict_single(feat)
        
        # Apply multi-timeframe filters
        action, conf = self._apply_multiframe_filters(action, conf, current_signals)
        
        # Update state
        self.last_price = close
        self.last_features = feat.copy()
        self.last_action = action
        self.last_signals = current_signals
        
        # Update timeframe-specific tracking
        self._update_timeframe_tracking(feat)
        
        # Update statistics
        self._update_statistics(feat)

        # Create row for logging (timestamp + 27 features + reward)
        row = [time.time()] + feat + [reward]

        return row, action, conf, close

    def _expand_to_27_features(self, feat_vec_9):
        """Expand 9-feature vector to 27-feature format"""
        try:
            # This is a fallback for backward compatibility
            # In practice, NinjaScript should send proper 27-feature vectors
            expanded = []
            
            # Replicate across timeframes with slight variations
            # 15-minute (trend context) - use as-is
            expanded.extend(feat_vec_9)
            
            # 5-minute (momentum context) - slight dampening for shorter timeframe
            feat_5m = feat_vec_9.copy()
            for i in [2, 3, 4, 5, 6, 7]:  # Signal indices
                feat_5m[i] *= 0.8  # Dampen signals slightly
            expanded.extend(feat_5m)
            
            # 1-minute (entry timing) - more dampening
            feat_1m = feat_vec_9.copy()
            for i in [2, 3, 4, 5, 6, 7]:  # Signal indices
                feat_1m[i] *= 0.6  # More dampening for shortest timeframe
            expanded.extend(feat_1m)
            
            self.feature_stats['single_timeframe_count'] += 1
            log.debug("Expanded 9-feature vector to 27-feature format")
            return expanded
            
        except Exception as e:
            log.warning(f"Feature expansion error: {e}")
            return feat_vec_9 + [0.0] * (27 - len(feat_vec_9))

    def _normalize_feature_vector(self, feat):
        """Ensure feature vector has exactly 27 elements"""
        if len(feat) < 27:
            # Pad with default values
            padded = list(feat) + [0.0] * (27 - len(feat))
            # Set LWPE defaults if missing
            for lwpe_idx in [8, 17, 26]:
                if len(feat) <= lwpe_idx:
                    padded[lwpe_idx] = 0.5  # Default LWPE
            return padded
        elif len(feat) > 27:
            # Truncate to 27 elements
            return feat[:27]
        return feat

    def _extract_multiframe_signals(self, feat):
        """Extract and validate signal values from all timeframes"""
        try:
            signals = {
                # 15-minute signals (trend context)
                '15m_tenkan_kijun': int(round(feat[2])),
                '15m_price_cloud': int(round(feat[3])),
                '15m_future_cloud': int(round(feat[4])),
                '15m_ema_cross': int(round(feat[5])),
                '15m_tenkan_momentum': int(round(feat[6])),
                '15m_kijun_momentum': int(round(feat[7])),
                '15m_normalized_volume': feat[1],
                '15m_lwpe': feat[8],
                
                # 5-minute signals (momentum context)
                '5m_tenkan_kijun': int(round(feat[11])),
                '5m_price_cloud': int(round(feat[12])),
                '5m_future_cloud': int(round(feat[13])),
                '5m_ema_cross': int(round(feat[14])),
                '5m_tenkan_momentum': int(round(feat[15])),
                '5m_kijun_momentum': int(round(feat[16])),
                '5m_normalized_volume': feat[10],
                '5m_lwpe': feat[17],
                
                # 1-minute signals (entry timing)
                '1m_tenkan_kijun': int(round(feat[20])),
                '1m_price_cloud': int(round(feat[21])),
                '1m_future_cloud': int(round(feat[22])),
                '1m_ema_cross': int(round(feat[23])),
                '1m_tenkan_momentum': int(round(feat[24])),
                '1m_kijun_momentum': int(round(feat[25])),
                '1m_normalized_volume': feat[19],
                '1m_lwpe': feat[26]
            }
            
            # Validate all signals are in valid range
            signal_keys = [k for k in signals.keys() if not any(x in k for x in ['volume', 'lwpe'])]
            for signal_name in signal_keys:
                signal_value = signals[signal_name]
                if signal_value not in [-1, 0, 1]:
                    log.warning(f"Invalid {signal_name} signal: {signal_value}, setting to 0")
                    signals[signal_name] = 0
            
            return signals
            
        except (IndexError, ValueError) as e:
            log.warning(f"Multi-timeframe signal extraction error: {e}")
            return self._get_default_signals()

    def _get_default_signals(self):
        """Get default signal structure"""
        return {
            # 15-minute defaults
            '15m_tenkan_kijun': 0, '15m_price_cloud': 0, '15m_future_cloud': 0,
            '15m_ema_cross': 0, '15m_tenkan_momentum': 0, '15m_kijun_momentum': 0,
            '15m_normalized_volume': 0, '15m_lwpe': 0.5,
            
            # 5-minute defaults
            '5m_tenkan_kijun': 0, '5m_price_cloud': 0, '5m_future_cloud': 0,
            '5m_ema_cross': 0, '5m_tenkan_momentum': 0, '5m_kijun_momentum': 0,
            '5m_normalized_volume': 0, '5m_lwpe': 0.5,
            
            # 1-minute defaults
            '1m_tenkan_kijun': 0, '1m_price_cloud': 0, '1m_future_cloud': 0,
            '1m_ema_cross': 0, '1m_tenkan_momentum': 0, '1m_kijun_momentum': 0,
            '1m_normalized_volume': 0, '1m_lwpe': 0.5
        }

    def _apply_multiframe_filters(self, action, confidence, signals):
        """Apply multi-timeframe filters to prevent trend-fighting"""
        try:
            if action == 0:  # Hold action, no filtering needed
                return action, confidence
            
            # Calculate 15-minute trend strength
            trend_15m_signals = [
                signals.get('15m_tenkan_kijun', 0),
                signals.get('15m_price_cloud', 0),
                signals.get('15m_ema_cross', 0)
            ]
            
            non_zero_trend = [s for s in trend_15m_signals if s != 0]
            
            if len(non_zero_trend) >= 2:  # Sufficient signal for trend determination
                trend_direction = sum(non_zero_trend) / len(non_zero_trend)
                trend_strength = abs(trend_direction)
                
                # Check for trend-fighting scenario
                action_direction = 1 if action == 1 else -1
                
                if trend_strength > 0.6:  # Strong trend
                    if (trend_direction > 0 and action_direction < 0) or \
                       (trend_direction < 0 and action_direction > 0):
                        # Fighting strong trend
                        log.debug(f"Trend filter: Action {action} against 15m trend {trend_direction:.2f}")
                        
                        self.feature_stats['trend_filter_blocks'] += 1
                        
                        # Reduce confidence or block trade
                        if confidence > 0.8:
                            # High confidence against trend - reduce significantly
                            confidence *= 0.4
                            log.debug(f"High confidence against trend reduced to {confidence:.3f}")
                        else:
                            # Lower confidence against trend - block trade
                            action = 0
                            confidence = 0.3
                            log.debug("Trade blocked by trend filter")
            
            # Check for timeframe alignment bonus
            alignment_score = self._calculate_timeframe_alignment_score(signals, action)
            if alignment_score > 0.7:  # Strong alignment across timeframes
                confidence = min(0.9, confidence * 1.2)  # Boost confidence
                self.feature_stats['timeframe_alignment_count'] += 1
                log.debug(f"Timeframe alignment bonus: confidence boosted to {confidence:.3f}")
            
            return action, confidence
            
        except Exception as e:
            log.warning(f"Multi-timeframe filter error: {e}")
            return action, confidence

    def _calculate_timeframe_alignment_score(self, signals, action):
        """Calculate how well timeframes align for the given action"""
        try:
            if action == 0:
                return 0.5
            
            expected_direction = 1 if action == 1 else -1
            
            # Score each timeframe
            timeframe_scores = []
            
            # 15-minute score (trend context)
            trend_signals = [
                signals.get('15m_tenkan_kijun', 0),
                signals.get('15m_price_cloud', 0),
                signals.get('15m_ema_cross', 0)
            ]
            trend_score = sum(1 for s in trend_signals if s == expected_direction) / 3.0
            timeframe_scores.append(trend_score)
            
            # 5-minute score (momentum context)
            momentum_signals = [
                signals.get('5m_tenkan_kijun', 0),
                signals.get('5m_ema_cross', 0),
                signals.get('5m_tenkan_momentum', 0)
            ]
            momentum_score = sum(1 for s in momentum_signals if s == expected_direction) / 3.0
            timeframe_scores.append(momentum_score)
            
            # 1-minute score (entry timing)
            entry_signals = [
                signals.get('1m_tenkan_kijun', 0),
                signals.get('1m_price_cloud', 0),
                signals.get('1m_ema_cross', 0)
            ]
            entry_score = sum(1 for s in entry_signals if s == expected_direction) / 3.0
            timeframe_scores.append(entry_score)
            
            # Weighted average (15m has highest weight)
            weights = [0.5, 0.3, 0.2]  # Emphasize longer timeframes
            alignment_score = sum(score * weight for score, weight in zip(timeframe_scores, weights))
            
            return alignment_score
            
        except Exception as e:
            log.debug(f"Timeframe alignment calculation error: {e}")
            return 0.5

    def _update_timeframe_tracking(self, feat):
        """Update tracking for individual timeframes"""
        try:
            # Extract timeframe-specific features
            self.last_15m_features = feat[0:9]     # 15-minute features
            self.last_5m_features = feat[9:18]    # 5-minute features  
            self.last_1m_features = feat[18:27]   # 1-minute features
            
        except Exception as e:
            log.debug(f"Timeframe tracking update error: {e}")

    def _update_statistics(self, feat):
        """Update processing statistics"""
        try:
            self.feature_stats['total_processed'] += 1
            
            if len(feat) == 27:
                self.feature_stats['multi_timeframe_count'] += 1
            else:
                self.feature_stats['single_timeframe_count'] += 1
                
        except Exception as e:
            log.debug(f"Statistics update error: {e}")

    def get_signal_summary(self):
        """Enhanced signal state summary with multi-timeframe analysis"""
        if self.last_signals is None:
            return "No multi-timeframe signals available"
        
        try:
            summary_parts = []
            
            # 15-minute trend analysis
            trend_15m = []
            if self.last_signals.get('15m_tenkan_kijun', 0) > 0:
                trend_15m.append("TK↑")
            elif self.last_signals.get('15m_tenkan_kijun', 0) < 0:
                trend_15m.append("TK↓")
            else:
                trend_15m.append("TK=")
                
            if self.last_signals.get('15m_price_cloud', 0) > 0:
                trend_15m.append("Cloud↑")
            elif self.last_signals.get('15m_price_cloud', 0) < 0:
                trend_15m.append("Cloud↓")
            else:
                trend_15m.append("InCloud")
                
            if self.last_signals.get('15m_ema_cross', 0) > 0:
                trend_15m.append("EMA↑")
            elif self.last_signals.get('15m_ema_cross', 0) < 0:
                trend_15m.append("EMA↓")
            else:
                trend_15m.append("EMA=")
                
            summary_parts.append(f"15m: {','.join(trend_15m)}")
            
            # 5-minute momentum analysis
            momentum_5m = []
            if self.last_signals.get('5m_ema_cross', 0) > 0:
                momentum_5m.append("EMA↑")
            elif self.last_signals.get('5m_ema_cross', 0) < 0:
                momentum_5m.append("EMA↓")
            else:
                momentum_5m.append("EMA=")
                
            mom_signals = []
            if self.last_signals.get('5m_tenkan_momentum', 0) > 0:
                mom_signals.append("T+")
            elif self.last_signals.get('5m_tenkan_momentum', 0) < 0:
                mom_signals.append("T-")
            else:
                mom_signals.append("T=")
                
            if self.last_signals.get('5m_kijun_momentum', 0) > 0:
                mom_signals.append("K+")
            elif self.last_signals.get('5m_kijun_momentum', 0) < 0:
                mom_signals.append("K-")
            else:
                mom_signals.append("K=")
                
            momentum_5m.extend(mom_signals)
            summary_parts.append(f"5m: {','.join(momentum_5m)}")
            
            # 1-minute entry analysis
            entry_1m = []
            if self.last_signals.get('1m_tenkan_kijun', 0) > 0:
                entry_1m.append("TK↑")
            elif self.last_signals.get('1m_tenkan_kijun', 0) < 0:
                entry_1m.append("TK↓")
            else:
                entry_1m.append("TK=")
                
            if self.last_signals.get('1m_price_cloud', 0) > 0:
                entry_1m.append("Cloud↑")
            elif self.last_signals.get('1m_price_cloud', 0) < 0:
                entry_1m.append("Cloud↓")
            else:
                entry_1m.append("InCloud")
                
            # LWPE condition
            lwpe_1m = self.last_signals.get('1m_lwpe', 0.5)
            if lwpe_1m > 0.7:
                entry_1m.append("LWPE:Buy")
            elif lwpe_1m < 0.3:
                entry_1m.append("LWPE:Sell")
            else:
                entry_1m.append("LWPE:Neutral")
                
            summary_parts.append(f"1m: {','.join(entry_1m)}")
            
            return " | ".join(summary_parts)
            
        except Exception as e:
            log.warning(f"Multi-timeframe signal summary error: {e}")
            return "Multi-timeframe signal analysis error"

    def get_timeframe_analysis(self):
        """Get detailed analysis of timeframe performance"""
        try:
            if self.last_signals is None:
                return "No timeframe data available"
            
            analysis = {}
            
            # 15-minute trend strength
            trend_signals = [
                self.last_signals.get('15m_tenkan_kijun', 0),
                self.last_signals.get('15m_price_cloud', 0),
                self.last_signals.get('15m_ema_cross', 0)
            ]
            non_zero_trend = [s for s in trend_signals if s != 0]
            if non_zero_trend:
                trend_direction = sum(non_zero_trend) / len(non_zero_trend)
                analysis['15m_trend_strength'] = abs(trend_direction)
                analysis['15m_trend_direction'] = "Bullish" if trend_direction > 0 else "Bearish"
            else:
                analysis['15m_trend_strength'] = 0
                analysis['15m_trend_direction'] = "Neutral"
            
            # 5-minute momentum strength
            momentum_signals = [
                self.last_signals.get('5m_tenkan_kijun', 0),
                self.last_signals.get('5m_ema_cross', 0),
                self.last_signals.get('5m_tenkan_momentum', 0)
            ]
            non_zero_momentum = [s for s in momentum_signals if s != 0]
            if non_zero_momentum:
                momentum_direction = sum(non_zero_momentum) / len(non_zero_momentum)
                analysis['5m_momentum_strength'] = abs(momentum_direction)
                analysis['5m_momentum_direction'] = "Bullish" if momentum_direction > 0 else "Bearish"
            else:
                analysis['5m_momentum_strength'] = 0
                analysis['5m_momentum_direction'] = "Neutral"
            
            # 1-minute entry quality
            entry_signals = [
                self.last_signals.get('1m_tenkan_kijun', 0),
                self.last_signals.get('1m_price_cloud', 0),
                self.last_signals.get('1m_ema_cross', 0)
            ]
            non_zero_entry = [s for s in entry_signals if s != 0]
            if non_zero_entry:
                entry_direction = sum(non_zero_entry) / len(non_zero_entry)
                analysis['1m_entry_strength'] = abs(entry_direction)
                analysis['1m_entry_direction'] = "Bullish" if entry_direction > 0 else "Bearish"
            else:
                analysis['1m_entry_strength'] = 0
                analysis['1m_entry_direction'] = "Neutral"
            
            # Overall alignment
            directions = [
                analysis.get('15m_trend_direction', 'Neutral'),
                analysis.get('5m_momentum_direction', 'Neutral'),
                analysis.get('1m_entry_direction', 'Neutral')
            ]
            
            bullish_count = directions.count('Bullish')
            bearish_count = directions.count('Bearish')
            
            if bullish_count >= 2:
                analysis['overall_alignment'] = "Bullish"
            elif bearish_count >= 2:
                analysis['overall_alignment'] = "Bearish"
            else:
                analysis['overall_alignment'] = "Mixed"
            
            analysis['alignment_score'] = max(bullish_count, bearish_count) / 3.0
            
            return analysis
            
        except Exception as e:
            log.warning(f"Timeframe analysis error: {e}")
            return {"error": str(e)}

    def get_processing_statistics(self):
        """Get processing statistics"""
        try:
            total = self.feature_stats['total_processed']
            if total == 0:
                return "No features processed yet"
            
            multi_pct = (self.feature_stats['multi_timeframe_count'] / total) * 100
            single_pct = (self.feature_stats['single_timeframe_count'] / total) * 100
            
            stats = {
                'total_processed': total,
                'multi_timeframe_percentage': multi_pct,
                'single_timeframe_percentage': single_pct,
                'timeframe_alignments': self.feature_stats['timeframe_alignment_count'],
                'trend_filter_blocks': self.feature_stats['trend_filter_blocks'],
                'trend_filter_rate': (self.feature_stats['trend_filter_blocks'] / total * 100) if total > 0 else 0
            }
            
            return stats
            
        except Exception as e:
            log.warning(f"Processing statistics error: {e}")
            return {"error": str(e)}

    def get_recommendation(self):
        """Get trading recommendation based on current multi-timeframe analysis"""
        try:
            if self.last_signals is None:
                return "No signal data available for recommendation"
            
            analysis = self.get_timeframe_analysis()
            if isinstance(analysis, dict) and 'error' not in analysis:
                
                trend_strength = analysis.get('15m_trend_strength', 0)
                momentum_strength = analysis.get('5m_momentum_strength', 0)
                entry_strength = analysis.get('1m_entry_strength', 0)
                alignment = analysis.get('overall_alignment', 'Mixed')
                alignment_score = analysis.get('alignment_score', 0)
                
                # Generate recommendation
                if alignment_score >= 0.67:  # 2/3 timeframes aligned
                    if trend_strength > 0.6:  # Strong trend
                        if alignment == "Bullish":
                            return f"STRONG BUY: All timeframes bullish, trend strength {trend_strength:.2f}"
                        else:
                            return f"STRONG SELL: All timeframes bearish, trend strength {trend_strength:.2f}"
                    else:
                        if alignment == "Bullish":
                            return f"BUY: Timeframes aligned bullish, moderate strength"
                        else:
                            return f"SELL: Timeframes aligned bearish, moderate strength"
                
                elif trend_strength > 0.7:  # Very strong trend overrides
                    direction = analysis.get('15m_trend_direction', 'Neutral')
                    return f"TREND FOLLOW: Strong {direction.lower()} 15m trend ({trend_strength:.2f}), wait for pullback"
                
                elif alignment_score <= 0.33:  # Conflicting signals
                    return "HOLD: Conflicting timeframe signals, wait for clarity"
                
                else:  # Mixed signals
                    if entry_strength > 0.5:
                        direction = analysis.get('1m_entry_direction', 'Neutral')
                        return f"SCALP: Short-term {direction.lower()} entry opportunity"
                    else:
                        return "WAIT: Mixed signals across timeframes"
            
            return "Unable to generate recommendation"
            
        except Exception as e:
            log.warning(f"Recommendation generation error: {e}")
            return "Recommendation unavailable due to error"