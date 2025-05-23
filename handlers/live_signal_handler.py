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

    def dispatch_signal(self, action: int, confidence: float):
        self.portfolio.update_position(self.tcp._current_position)

        size = self.portfolio.calculate_position_size(action, confidence)

        current_time = time.time()

        self.signal_counter += 1
        timestamp = int(current_time)
        
        sig = {
            "action": action,
            "confidence": round(confidence, 3),
            "size": size,
            "timestamp": timestamp,
            "signal_id": self.signal_counter
        }

        try:
            self.tcp.send_signal(sig)
            self.last_signal_time = current_time
            
            action_name = "Long" if action == 1 else ("Short" if action == 2 else "Hold")
            
            log.info(f"Signal sent - {action_name}: size={size}, conf={confidence:.3f}, id={self.signal_counter}, timestamp={timestamp}")
                    
        except Exception as e:
            log.error(f"Failed to send signal: {e}")