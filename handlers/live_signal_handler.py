# handlers/live_signal_handler.py:

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
        # Smart position sign interpretation
        raw_position = self.tcp._current_position
        corrected_position = self.interpret_position_sign(raw_position, action)
        
        log.info(f"[POS_FIX] NT reports: {raw_position}, corrected to: {corrected_position}")
        self.portfolio.update_position(corrected_position)
        
        # Get position info before sizing
        risk_level = self.portfolio.get_risk_level()
        position_util = self.portfolio.get_position_utilization()
        
        size = self.portfolio.calculate_trade_size(action, confidence, self.cfg.BASE_SIZE, self.cfg.MIN_SIZE)

        if size == 0:
            log.info(f"Signal REJECTED: action={action}, conf={confidence:.3f}, "
                    f"pos={self.portfolio.position}, util={position_util:.1%}, risk={risk_level}")
            return

        current_time = time.time()
        
        # Prevent rapid-fire signals
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
            
            # Enhanced logging with position context (NOW USING position_util!)
            action_name = "BUY" if action == 1 else ("SELL" if action == 2 else "HOLD")
            trade_intent = self.portfolio.get_trade_intent(action, size)
            
            # Calculate projected utilization
            projected_pos = self.portfolio.position + (size if action == 1 else -size)
            projected_util = abs(projected_pos) / self.portfolio.max_position if self.portfolio.max_position > 0 else 0
            
            log.info(f"✓ {action_name} signal: {trade_intent}, conf={confidence:.3f}, "
                    f"util={position_util:.1%}→{projected_util:.1%}, risk={risk_level}, id={self.signal_counter}")
                    
        except Exception as e:
            log.error(f"Failed to send signal: {e}")

    def interpret_position_sign(self, raw_position, current_action):
        """
        Simple: Only fix SHORT positions, leave LONG as-is
        """
        if raw_position == 0:
            return 0
            
        # If we're trying to SELL and have a position, assume we're SHORT
        if current_action == 2 and raw_position > 0:
            return -abs(raw_position)
        
        # Otherwise, trust NT's sign (works for LONG positions)
        return raw_position