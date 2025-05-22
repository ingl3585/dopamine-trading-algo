# services/portfolio.py

import logging

log = logging.getLogger(__name__)

class Portfolio:
    def __init__(self, max_position):
        self.position = 0
        self.max_position = max_position

    def get_current_position(self):
        return self.position

    def update_position(self, pos):
        self.position = pos

    def can_execute(self, action, size):
        if action == 1:
            return self.position + size <= self.max_position
        elif action == 2:
            return self.position - size >= -self.max_position
        return False

    def adjust_size(self, action, size):
        if action == 1:
            max_buy = self.max_position - self.position
            return min(size, max(0, max_buy))
        elif action == 2:
            max_sell = self.position + self.max_position
            return min(size, max(0, max_sell))
        return 0

    def get_position_utilization(self):
        """Calculate how much of max position is currently used (0.0 to 1.0)"""
        return abs(self.position) / self.max_position if self.max_position > 0 else 0

    def get_available_capacity(self, action):
        """Get remaining capacity for new positions"""
        if action == 1:  # BUY
            return max(0, self.max_position - self.position)
        elif action == 2:  # SELL  
            return max(0, self.max_position + self.position)
        return 0

    def calculate_trade_size(self, action, confidence, base_size, min_size):
        """
        ENHANCED: Position-aware sizing with conservative approach
        """
        if action == 0:
            return 0
        
        current_pos = abs(self.position)
        position_util = self.get_position_utilization()
        available_capacity = self.get_available_capacity(action)
        
        # Log current position state
        log.info(f"[POSITION] Current: {self.position}, Util: {position_util:.1%}, "
                f"Available: {available_capacity}, Action: {action}")
        
        # CONSERVATIVE SIZING BASED ON POSITION AND CONFIDENCE
        
        # 1. Base confidence mapping (more conservative range)
        conf_min = 0.15
        conf_max = 0.90
        conf_clamped = max(conf_min, min(conf_max, confidence))
        conf_ratio = (conf_clamped - conf_min) / (conf_max - conf_min)
        
        # 2. Position-aware size reduction
        if position_util < 0.3:  # Less than 30% of max position
            # Free to trade normal sizes
            size_multiplier = 0.4 + (conf_ratio * 0.6)  # 40% to 100% of base_size
            max_trade_size = base_size
        elif position_util < 0.6:  # 30-60% of max position  
            # Reduce trade sizes
            size_multiplier = 0.3 + (conf_ratio * 0.4)  # 30% to 70% of base_size
            max_trade_size = max(3, base_size // 2)
        elif position_util < 0.8:  # 60-80% of max position
            # Much smaller trades only
            size_multiplier = 0.2 + (conf_ratio * 0.3)  # 20% to 50% of base_size  
            max_trade_size = max(2, base_size // 3)
        else:  # 80%+ of max position
            # Tiny trades only, very selective
            size_multiplier = 0.1 + (conf_ratio * 0.2)  # 10% to 30% of base_size
            max_trade_size = max(1, base_size // 4)
        
        # 3. Calculate desired size
        desired = round(size_multiplier * base_size)
        desired = min(desired, max_trade_size)
        desired = max(min_size, desired)
        
        # 4. Confidence-based micro-adjustments for variation
        if confidence > 0.75:
            desired += 1  # High confidence gets +1
        elif confidence < 0.50:
            desired = max(min_size, desired - 1)  # Low confidence gets -1
            
        # 5. Add small variation to prevent identical sizes
        micro_variation = int((confidence * 1000) % 3)  # 0, 1, or 2
        if micro_variation == 2 and desired < max_trade_size:
            desired += 1
        elif micro_variation == 0 and desired > min_size:
            desired -= 1
        
        # 6. Final position limit check
        adjusted = self.adjust_size(action, desired)
        
        # 7. Ensure we don't exceed capacity
        final_size = min(adjusted, available_capacity)
        final_size = max(0, final_size)
        
        # 8. Safety check
        if not self.trade_makes_sense(action, final_size):
            log.warning(f"Trade rejected: action={action}, size={final_size}, pos={self.position}")
            return 0
        
        # 9. Enhanced logging
        log.info(f"[SIZE] conf={confidence:.3f}, pos_util={position_util:.1%}, "
                f"capacity={available_capacity}, desired={desired}, final={final_size}")
        
        if final_size > 0:
            projected_pos = self.position + (final_size if action == 1 else -final_size)
            projected_util = abs(projected_pos) / self.max_position
            log.info(f"[PROJECTED] New position: {projected_pos}, New util: {projected_util:.1%}")
        
        return final_size

    def trade_makes_sense(self, action, size):
        """Enhanced trade validation"""
        if size <= 0:
            return False
            
        current_pos = self.position
        projected_pos = current_pos + (size if action == 1 else -size)
        
        # Check absolute position limits
        if abs(projected_pos) > self.max_position:
            return False
            
        # Additional safety checks
        if current_pos == 0:
            return True  # Opening positions always OK if within limits
            
        # For existing positions, be more conservative about adding
        current_util = abs(current_pos) / self.max_position
        
        if action == 1 and current_pos > 0:  # Adding to long position
            return current_util < 0.8 and size <= 3  # Max 3 additional
        elif action == 2 and current_pos < 0:  # Adding to short position  
            return current_util < 0.8 and size <= 3  # Max 3 additional
        elif action == 1 and current_pos < 0:  # Covering short
            return True  # Always allow covering
        elif action == 2 and current_pos > 0:  # Covering long
            return True  # Always allow covering
            
        return True

    def get_trade_intent(self, action, size):
        """Enhanced trade intent description"""
        if action == 0 or size == 0:
            return "HOLD"
            
        current_pos = self.position
        util_before = abs(current_pos) / self.max_position
        projected_pos = current_pos + (size if action == 1 else -size)
        util_after = abs(projected_pos) / self.max_position
        
        if action == 1:
            if current_pos == 0:
                return f"OPEN_LONG_{size} (0% → {util_after:.0%})"
            elif current_pos > 0:
                return f"ADD_LONG_{size} ({util_before:.0%} → {util_after:.0%})"
            elif current_pos < 0:
                if size >= abs(current_pos):
                    return f"COVER_SHORT_GO_LONG_{size} ({util_before:.0%} → {util_after:.0%})"
                else:
                    return f"REDUCE_SHORT_{size} ({util_before:.0%} → {util_after:.0%})"
                    
        elif action == 2:
            if current_pos == 0:
                return f"OPEN_SHORT_{size} (0% → {util_after:.0%})"
            elif current_pos < 0:
                return f"ADD_SHORT_{size} ({util_before:.0%} → {util_after:.0%})"
            elif current_pos > 0:
                if size >= current_pos:
                    return f"CLOSE_LONG_GO_SHORT_{size} ({util_before:.0%} → {util_after:.0%})"
                else:
                    return f"REDUCE_LONG_{size} ({util_before:.0%} → {util_after:.0%})"
        
        return f"UNKNOWN_{action}_{size}"

    def get_risk_level(self):
        """Get current risk level description"""
        util = self.get_position_utilization()
        if util < 0.3:
            return "LOW"
        elif util < 0.6:
            return "MEDIUM" 
        elif util < 0.8:
            return "HIGH"
        else:
            return "CRITICAL"