# handlers/live_signal_handler.py

import time
import logging
from datetime import datetime

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

    def dispatch_signal(self, action: int, confidence: float):
        # Simple: just use the position NT sends (now with correct sign!)
        self.portfolio.update_position(self.tcp._current_position)
        
        # Get position info
        risk_level = self.portfolio.get_risk_level()
        position_util = self.portfolio.get_position_utilization()
        
        size = self.portfolio.calculate_trade_size(action, confidence, self.cfg.BASE_SIZE, self.cfg.MIN_SIZE)

        if size == 0:
            log.info(f"Signal REJECTED: action={action}, conf={confidence:.3f}, "
                    f"pos={self.portfolio.position}, util={position_util:.1%}, risk={risk_level}")
            return

        current_time = time.time()
        
        if current_time - self.last_signal_time < 1.0:
            log.debug("Dispatch skipped: too close to last signal")
            return

        self.signal_counter += 1
        timestamp = int(current_time)
        
        sig = {
            "action": action,
            "confidence": round(confidence, 4),
            "size": size,
            "timestamp": timestamp,
            "signal_id": self.signal_counter
        }

        try:
            self.tcp.send_signal(sig)
            self.last_signal_time = current_time
            
            # Simple logging
            action_name = "BUY" if action == 1 else ("SELL" if action == 2 else "HOLD")
            
            projected_pos = self.portfolio.position + (size if action == 1 else -size)
            projected_util = abs(projected_pos) / self.portfolio.max_position if self.portfolio.max_position > 0 else 0
            
            log.info(f"✓ {action_name}: size={size}, conf={confidence:.3f}, "
                    f"pos={self.portfolio.position}→{projected_pos}, "
                    f"util={position_util:.1%}→{projected_util:.1%}, risk={risk_level}, id={self.signal_counter}")
                    
        except Exception as e:
            log.error(f"Failed to send signal: {e}")