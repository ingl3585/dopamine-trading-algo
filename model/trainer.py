# model/trainer.py

import time
import logging
import pandas as pd

from argparse import Namespace
from model.agent import RLAgent
from utils.feature_logger import FeatureLogger

log = logging.getLogger(__name__)

class Trainer:
    def __init__(self, cfg, agent: RLAgent, logger: FeatureLogger, args: Namespace):
        self.cfg = cfg
        self.agent = agent
        self.logger = logger
        self.args = args
        self.trained = False
        self.last_price = None
        self.last_save_time = time.time()
        self.training_in_progress = False

    def append(self, row):
        self.logger.append(row)

    def should_train_initial(self):
        return not self.trained or self.args.reset

    def perform_initial_training(self):
        if self.training_in_progress:
            return
            
        self.training_in_progress = True
        log.info("Initial backfill training")
        df = pd.DataFrame(self.logger.rows, columns=["ts", "close", "volume", "atr", "lwpe", "reward"])
        self.agent.train(df, epochs=3)
        self.agent.save_model()
        self.agent.last_save_time = time.time()  # Reset the save timer
        self.logger.rows.clear()
        self.trained = True
        self.args.reset = False
        self.training_in_progress = False
        log.info("Initial training completed - ready for live trading")

    def should_train_batch(self):
        return len(self.logger.rows) >= self.cfg.BATCH_SIZE and not self.training_in_progress

    def train_batch(self):
        if self.training_in_progress:
            return
            
        df = pd.DataFrame(self.logger.rows, columns=["ts", "close", "volume", "atr", "lwpe", "reward"])
        self.agent.train(df, epochs=1)
        self.agent.save_model()
        self.logger.flush()

    def update_latest_reward(self, reward):
        if self.logger.rows:
            self.logger.rows[-1][-1] = reward

    def is_ready_for_trading(self):
        return self.trained and not self.training_in_progress