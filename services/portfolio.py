# services/portfolio.py

import logging

log = logging.getLogger(__name__)

class Portfolio:
    def __init__(self, cfg):
        self.cfg = cfg
        self.position = 0
        self.max_position = cfg.MAX_SIZE
        
        # Enhanced position sizing for Ichimoku/EMA signals
        self.signal_weights = {
            'ichimoku_alignment': 0.4,
            'ema_alignment': 0.3,
            'momentum_alignment': 0.2,
            'confidence_base': 0.1
        }

    def update_position(self, pos):
        """Update current position"""
        self.position = pos

    def get_available_capacity(self, action):
        """Get available capacity for new positions"""
        if action == 1:  # BUY
            return max(0, self.max_position - self.position)
        elif action == 2:  # SELL  
            return max(0, self.max_position + self.position)
        return 0
    
    def calculate_position_size(self, action: int, confidence: float, signal_data=None) -> int:
        """
        Calculate position size with enhanced logic for Ichimoku/EMA signals
        
        Args:
            action: Trading action (0=hold, 1=long, 2=short)
            confidence: Model confidence
            signal_data: Optional dict with current signal values
        """
        if action == 0:
            return 0
        
        available_capacity = self.get_available_capacity(action)
        
        if available_capacity <= 0:
            log.debug(f"No available capacity for action {action}")
            return 0
        
        # Enhanced confidence calculation with signal alignment
        enhanced_confidence = self._calculate_enhanced_confidence(
            confidence, action, signal_data
        )
        
        # Check minimum confidence threshold
        if enhanced_confidence < self.cfg.CONFIDENCE_THRESHOLD:
            log.debug(f"Enhanced confidence {enhanced_confidence:.3f} below threshold")
            return 0
        
        # Base position sizing
        base_size = self.cfg.BASE_SIZE
        
        # Apply confidence multiplier with signal-based scaling
        confidence_multiplier = self._calculate_confidence_multiplier(enhanced_confidence)
        scaled_size = int(base_size * confidence_multiplier)
        
        # Apply bounds
        scaled_size = max(self.cfg.MIN_SIZE, min(scaled_size, self.cfg.MAX_SIZE))
        
        # Final capacity check
        final_size = min(scaled_size, available_capacity)
        
        log.debug(f"Position sizing: action={action}, conf={confidence:.3f}, "
                 f"enhanced_conf={enhanced_confidence:.3f}, size={final_size}")
        
        return final_size

    def _calculate_enhanced_confidence(self, base_confidence, action, signal_data):
        """Calculate enhanced confidence using signal alignment"""
        if not signal_data:
            return base_confidence
        
        try:
            alignment_scores = {
                'ichimoku_alignment': self._get_ichimoku_alignment_score(action, signal_data),
                'ema_alignment': self._get_ema_alignment_score(action, signal_data),
                'momentum_alignment': self._get_momentum_alignment_score(action, signal_data)
            }
            
            # Calculate weighted alignment score
            total_alignment = sum(
                self.signal_weights[key] * score 
                for key, score in alignment_scores.items()
            )
            
            # Combine with base confidence
            enhanced_confidence = (
                self.signal_weights['confidence_base'] * base_confidence +
                (1 - self.signal_weights['confidence_base']) * total_alignment
            )
            
            return min(enhanced_confidence, 0.95)  # Cap at 95%
            
        except Exception as e:
            log.warning(f"Enhanced confidence calculation failed: {e}")
            return base_confidence

    def _get_ichimoku_alignment_score(self, action, signal_data):
        """Enhanced Ichimoku signal alignment with neutral handling"""
        try:
            if action == 0:  # Hold action
                return 0.5
                
            expected_direction = 1 if action == 1 else -1
            
            alignments = []
            weights = []
            
            # Tenkan/Kijun cross (highest weight)
            if 'tenkan_kijun_signal' in signal_data:
                signal = signal_data['tenkan_kijun_signal']
                if signal == expected_direction:
                    alignments.append(1.0)
                elif signal == -expected_direction:
                    alignments.append(0.0)
                else:  # signal == 0 (neutral)
                    alignments.append(0.5)
                weights.append(0.4)
            
            # Price vs Cloud (high weight)
            if 'price_cloud_signal' in signal_data:
                signal = signal_data['price_cloud_signal']
                if signal == expected_direction:
                    alignments.append(1.0)
                elif signal == -expected_direction:
                    alignments.append(0.0)
                else:  # signal == 0 (neutral/inside cloud)
                    alignments.append(0.3)  # Slight penalty for uncertain position
                weights.append(0.4)
            
            # Future Cloud color (moderate weight)
            if 'future_cloud_signal' in signal_data:
                signal = signal_data['future_cloud_signal']
                if signal == expected_direction:
                    alignments.append(1.0)
                elif signal == -expected_direction:
                    alignments.append(0.0)
                else:  # signal == 0 (neutral cloud)
                    alignments.append(0.5)
                weights.append(0.2)
            
            if alignments:
                # Weighted average
                total_weight = sum(weights)
                weighted_sum = sum(a * w for a, w in zip(alignments, weights))
                return weighted_sum / total_weight
            else:
                return 0.5
                
        except Exception as e:
            log.debug(f"Ichimoku alignment calculation error: {e}")
            return 0.5

    def _get_ema_alignment_score(self, action, signal_data):
        """Enhanced EMA signal alignment with neutral handling"""
        try:
            if action == 0:  # Hold action
                return 0.5
                
            if 'ema_cross_signal' not in signal_data:
                return 0.5
            
            ema_signal = signal_data['ema_cross_signal']
            expected_direction = 1 if action == 1 else -1
            
            if ema_signal == expected_direction:
                return 1.0
            elif ema_signal == -expected_direction:
                return 0.0
            else:  # ema_signal == 0 (neutral)
                return 0.4  # Slight penalty for neutral EMA
                
        except Exception as e:
            log.debug(f"EMA alignment calculation error: {e}")
            return 0.5

    def _get_momentum_alignment_score(self, action, signal_data):
        """Enhanced momentum signal alignment with neutral handling"""
        try:
            if action == 0:  # Hold action
                return 0.5
                
            expected_direction = 1 if action == 1 else -1
            
            alignments = []
            
            # Tenkan momentum
            if 'tenkan_momentum' in signal_data:
                momentum = signal_data['tenkan_momentum']
                if momentum == expected_direction:
                    alignments.append(1.0)
                elif momentum == -expected_direction:
                    alignments.append(0.0)
                else:  # momentum == 0 (flat)
                    alignments.append(0.5)
            
            # Kijun momentum
            if 'kijun_momentum' in signal_data:
                momentum = signal_data['kijun_momentum']
                if momentum == expected_direction:
                    alignments.append(1.0)
                elif momentum == -expected_direction:
                    alignments.append(0.0)
                else:  # momentum == 0 (flat)
                    alignments.append(0.5)
            
            return sum(alignments) / len(alignments) if alignments else 0.5
            
        except Exception as e:
            log.debug(f"Momentum alignment calculation error: {e}")
            return 0.5

    def _calculate_confidence_multiplier(self, enhanced_confidence):
        """Calculate position size multiplier based on enhanced confidence"""
        # More aggressive scaling for high-confidence signals
        if enhanced_confidence >= 0.8:
            return 1.5  # 150% of base size
        elif enhanced_confidence >= 0.7:
            return 1.2  # 120% of base size
        elif enhanced_confidence >= 0.6:
            return 1.0  # 100% of base size
        else:
            return 0.7  # 70% of base size

    def get_portfolio_status(self):
        """Get current portfolio status"""
        return {
            'current_position': self.position,
            'max_position': self.max_position,
            'long_capacity': self.get_available_capacity(1),
            'short_capacity': self.get_available_capacity(2),
            'utilization_pct': abs(self.position) / self.max_position * 100
        }

    def calculate_risk_adjusted_size(self, action, confidence, volatility_measure):
        """Calculate position size with risk adjustment based on market volatility"""
        try:
            base_size = self.calculate_position_size(action, confidence)
            
            if base_size == 0:
                return 0
            
            # Adjust for volatility (using LWPE as volatility proxy)
            if volatility_measure is not None:
                # LWPE close to 0.5 = balanced/low volatility
                # LWPE near 0 or 1 = high volatility/directional
                volatility_score = abs(volatility_measure - 0.5) * 2
                
                # Reduce size in high volatility
                if volatility_score > 0.7:
                    volatility_adjustment = 0.7
                elif volatility_score > 0.5:
                    volatility_adjustment = 0.85
                else:
                    volatility_adjustment = 1.0
                
                adjusted_size = int(base_size * volatility_adjustment)
                return max(self.cfg.MIN_SIZE, adjusted_size)
            
            return base_size
            
        except Exception as e:
            log.warning(f"Risk adjusted sizing failed: {e}")
            return self.cfg.MIN_SIZE