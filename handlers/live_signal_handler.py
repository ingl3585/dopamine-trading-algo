# handlers/live_signal_handler.py

import time
import logging

from services.portfolio import Portfolio
from model.agent import RLAgent
from utils.feature_writer import FeatureWriter
from services.tcp_bridge import TCPBridge

log = logging.getLogger(__name__)

class LiveSignalHandler:
    def __init__(self, cfg, agent: RLAgent, portfolio: Portfolio, tcp: TCPBridge, logger: FeatureWriter):
        self.cfg = cfg
        self.agent = agent
        self.portfolio = portfolio
        self.tcp = tcp
        self.logger = logger

    def dispatch_signal(self, action: int, confidence: float):
        self.portfolio.update_position(self.tcp._current_position)
        size = self.portfolio.calculate_trade_size(action, confidence, self.cfg.BASE_SIZE, self.cfg.MIN_SIZE)

        if size == 0:
            log.debug("Dispatch skipped: size = 0")
            return

        sig = {
            "action": action,
            "confidence": confidence,
            "size": size,
            "timestamp": int(time.time())
        }

        self.tcp.send_signal(sig)
        log.info("Sent signal %s", sig)
