# Updated handlers/live_signal_handler.py

import time
import logging
from datetime import datetime

from services.portfolio import Portfolio
from model.agent import RLAgent
from utils.feature_logger import FeatureLogger
from services.tcp_bridge import TCPBridge

log = logging.getLogger(__name__)

class LiveSignalHandler:
    def __init__(self, cfg, agent: RLAgent, portfolio: Portfolio, tcp: TCPBridge, logger: FeatureLogger):
        self.cfg = cfg
        self.agent = agent
        self.portfolio = portfolio
        self.tcp = tcp
        self.logger = logger
        self.last_signal_time = 0  # Track last signal to prevent duplicates

    def dispatch_signal(self, action: int, confidence: float):
        self.portfolio.update_position(self.tcp._current_position)
        size = self.portfolio.calculate_trade_size(action, confidence, self.cfg.BASE_SIZE, self.cfg.MIN_SIZE)

        if size == 0:
            log.debug("Dispatch skipped: size = 0")
            return

        # Generate proper timestamp
        current_time = time.time()
        
        # Prevent duplicate signals within 1 second
        if current_time - self.last_signal_time < 1.0:
            log.debug("Dispatch skipped: too close to last signal")
            return
        
        # Convert to .NET DateTime ticks
        # .NET DateTime ticks = (Unix timestamp * 10,000,000) + 621,355,968,000,000,000
        unix_timestamp = int(current_time)
        dotnet_ticks = (unix_timestamp * 10_000_000) + 621_355_968_000_000_000
        
        sig = {
            "action": action,
            "confidence": round(confidence, 4),  # Round confidence to prevent precision issues
            "size": size,
            "timestamp": dotnet_ticks  # Use .NET ticks format
        }

        try:
            self.tcp.send_signal(sig)
            self.last_signal_time = current_time
            log.info("Sent signal %s", sig)
        except Exception as e:
            log.error(f"Failed to send signal: {e}")