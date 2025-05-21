# core/runner.py

import time

from config import Config
from model.agent import RLAgent
from services.portfolio import Portfolio
from services.tcp_bridge import TCPBridge
from services.tick_processor import TickProcessor
from utils.feature_writer import FeatureWriter
from handlers.live_feature_handler import LiveFeatureHandler

class Runner:
    def __init__(self, args):
        self.args = args
        self.cfg = Config()
        self.agent = RLAgent(self.cfg)
        self.portfolio = Portfolio(self.cfg.MAX_SIZE)
        self.logger = FeatureWriter(self.cfg.FEATURE_FILE, self.cfg.BATCH_SIZE)
        self.tcp = TCPBridge("localhost", 5556, 5557)
        self.tick_processor = TickProcessor()
        self.handler = LiveFeatureHandler(self.cfg, self.agent, self.portfolio, self.logger, self.tcp, self.args)

    def run(self):
        self.tick_processor.wait_until_ready()
        self.tcp.on_features = self.handler.handle_live_feature

        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            self.shutdown()

    def shutdown(self):
        self.tcp.close()
        self.logger.flush()
        self.agent.save_model()