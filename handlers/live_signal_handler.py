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
        self.last_signal_time = 0  # ADD THIS BACK - needed for rate limiting

    def dispatch_signal(self, action: int, confidence: float):
        self.portfolio.update_position(self.tcp._current_position)
        size = self.portfolio.calculate_trade_size(action, confidence, self.cfg.BASE_SIZE, self.cfg.MIN_SIZE)

        if size == 0:
            log.debug("Dispatch skipped: size = 0")
            return

        current_time = time.time()
        
        # ADD THIS BACK - Prevent rapid-fire signals (aligned with NT bar processing)
        if current_time - self.last_signal_time < 1.0:
            log.debug("Dispatch skipped: too close to last signal")
            return

        self.signal_counter += 1
        timestamp = int(current_time)  # Simple Unix timestamp in seconds
        
        sig = {
            "action": action,
            "confidence": round(confidence, 4),
            "size": size,
            "timestamp": timestamp,
            "signal_id": self.signal_counter
        }

        try:
            self.tcp.send_signal(sig)
            self.last_signal_time = current_time  # ADD THIS BACK
            log.info("Sent signal %s", sig)
        except Exception as e:
            log.error(f"Failed to send signal: {e}")