# services/feature_processor.py

import time
import logging

from model.reward import RewardCalculator
from model.agent import RLAgent

log = logging.getLogger(__name__)

class FeatureProcessor:
    def __init__(self, agent: RLAgent, rewarder: RewardCalculator):
        self.agent = agent
        self.rewarder = rewarder
        self.last_price = None
        self.last_features = None
        self.last_action = None

    def process_and_predict(self, feat):
        close = feat[0]
        atr = feat[2] if len(feat) > 2 else 0.01

        reward = 0.0
        if self.last_price is not None and self.last_action is not None:
            price_change = close - self.last_price

            if self.last_action == 1:
                reward = self.rewarder.compute_reward(price_change, atr, self.last_action)
            elif self.last_action == 2:
                reward = self.rewarder.compute_reward(-price_change, atr, self.last_action)
            else:
                reward = self.rewarder.compute_reward(0, atr, self.last_action)
            
            if self.last_features is not None:
                self.agent.add_experience(self.last_features, self.last_action, reward, feat)

        action, conf = self.agent.predict_single(feat)
        
        self.last_price = close
        self.last_features = feat.copy()
        self.last_action = action

        row = [time.time(), *feat, reward]

        return row, action, conf, close