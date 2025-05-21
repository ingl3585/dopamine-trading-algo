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

    def calculate_trade_size(self, action, confidence, base_size, min_size):
        if action == 0:
            return 0
            
        desired = int(confidence * base_size)
        adjusted = self.adjust_size(action, desired)
        
        if not self.trade_makes_sense(action, adjusted):
            log.debug(f"Trade doesn't make sense: action={action}, size={adjusted}, pos={self.position}")
            return 0
        
        final_size = max(min_size, adjusted) if adjusted > 0 else 0
        log.debug(f"Trade calc: action={action}, conf={confidence:.2f}, desired={desired}, adjusted={adjusted}, final={final_size}")
        return final_size

    def trade_makes_sense(self, action, size):
        if size <= 0:
            return False
            
        current_pos = self.position
        
        if current_pos == 0:
            return True
            
        if action == 1:
            if current_pos > 0:
                return size <= self.max_position // 3
            return True
            
        if action == 2:
            if current_pos < 0:
                return size <= self.max_position // 3
            return True
            
        return True

    def get_trade_intent(self, action, size):
        if action == 0 or size == 0:
            return "HOLD"
            
        current_pos = self.position
        
        if action == 1:
            if current_pos == 0:
                return f"OPEN_LONG_{size}"
            elif current_pos > 0:
                return f"ADD_LONG_{size}"
            elif current_pos < 0:
                if size >= abs(current_pos):
                    return f"CLOSE_SHORT_AND_LONG_{size}"
                else:
                    return f"REDUCE_SHORT_{size}"
                    
        elif action == 2:
            if current_pos == 0:
                return f"OPEN_SHORT_{size}"
            elif current_pos < 0:
                return f"ADD_SHORT_{size}"
            elif current_pos > 0:
                if size >= current_pos:
                    return f"CLOSE_LONG_AND_SHORT_{size}"
                else:
                    return f"REDUCE_LONG_{size}"
        
        return "UNKNOWN"