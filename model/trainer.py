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
        self.online_training_counter = 0

    def append(self, row):
        self.logger.append(row)

    def should_train_initial(self):
        return not self.trained or self.args.reset

    def perform_initial_training(self):
        if self.training_in_progress:
            return
            
        self.training_in_progress = True
        log.info("Starting initial training")
        
        try:
            df = pd.DataFrame(self.logger.rows, columns=["ts", "close", "volume", "atr", "lwpe", "reward"])
            if len(df) > 0:
                self.agent.train(df, epochs=3)
                self.agent.save_model()
                log.info(f"Initial training completed on {len(df)} samples")
            else:
                log.warning("No data available for initial training")
            
            self.logger.rows.clear()
            self.trained = True
            self.args.reset = False
            
        except Exception as e:
            log.error(f"Initial training failed: {e}")
        finally:
            self.training_in_progress = False

    def should_train_batch(self):
        return (len(self.logger.rows) >= self.cfg.BATCH_SIZE and 
                not self.training_in_progress and
                self.trained)

    def should_train_online(self):
        return (len(self.agent.experience_buffer) >= self.cfg.BATCH_SIZE and
                not self.training_in_progress and
                self.trained)

    def train_batch(self):
        if self.training_in_progress:
            return
            
        try:
            df = pd.DataFrame(self.logger.rows, columns=["ts", "close", "volume", "atr", "lwpe", "reward"])
            self.agent.train(df, epochs=1)
            
            if time.time() - self.last_save_time > 1800:
                self.agent.save_model()
                self.last_save_time = time.time()
                
            self.logger.flush()
            log.info(f"Batch training completed on {len(df)} samples")
            
        except Exception as e:
            log.error(f"Batch training failed: {e}")

    def train_online(self):
        if self.training_in_progress or not self.trained:
            return
            
        try:
            loss = self.agent.train_online()
            self.online_training_counter += 1
            
            if self.online_training_counter % 10 == 0:
                log.debug(f"Online training step {self.online_training_counter}, loss: {loss:.4f}")
                
            if self.online_training_counter % 100 == 0:
                self.agent.save_model()
                log.info(f"Model saved after {self.online_training_counter} online training steps")
                
        except Exception as e:
            log.error(f"Online training failed: {e}")

    def is_ready_for_trading(self):
        return self.trained and not self.training_in_progress