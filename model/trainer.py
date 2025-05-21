# model/trainer.py

import time
import logging
import pandas as pd

from argparse import Namespace
from model.agent import RLAgent
from utils.feature_writer import FeatureWriter

log = logging.getLogger(__name__)

class Trainer:
    def __init__(self, cfg, agent: RLAgent, logger: FeatureWriter, args: Namespace):
        self.cfg = cfg
        self.agent = agent
        self.logger = logger
        self.args = args
        self.trained = False
        self.last_price = None
        self.last_save_time = time.time()

    def append(self, row):
        self.logger.append(row)

    def should_train_initial(self):
        return not self.trained or self.args.reset

    def perform_initial_training(self):
        log.info("Initial backfill training")
        df = pd.DataFrame(self.logger.rows, columns=["ts", "close", "volume", "atr", "lwpe", "reward"])
        self.agent.train(df, epochs=3)
        self.agent.save_model()
        self.logger.rows.clear()
        self.trained = True
        self.args.reset = False

    def should_train_batch(self):
        return len(self.logger.rows) >= self.cfg.BATCH_SIZE

    def train_batch(self):
        df = pd.DataFrame(self.logger.rows, columns=["ts", "close", "volume", "atr", "lwpe", "reward"])
        self.agent.train(df, epochs=1)
        self.agent.save_model()
        self.logger.flush()

    def update_latest_reward(self, reward):
        if self.logger.rows:
            self.logger.rows[-1][-1] = reward
