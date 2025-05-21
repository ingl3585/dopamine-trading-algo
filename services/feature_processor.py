# services/feature_processor.py

import time

from model.reward import RewardCalculator
from model.agent import RLAgent

class FeatureProcessor:
    def __init__(self, agent: RLAgent, rewarder: RewardCalculator):
        self.agent = agent
        self.rewarder = rewarder
        self.last_price = None

    def process_and_predict(self, feat):
        close = feat[0]
        atr = feat[2] if len(feat) > 2 else 0.01

        price_change = 0.0 if self.last_price is None else close - self.last_price
        reward = 0.0 if self.last_price is None else self.rewarder.compute_reward(price_change, atr)
        self.last_price = close

        action, conf = self.agent.predict_single(feat)
        reward = self.rewarder.modify_reward(action, reward)

        self.agent.push_sample(feat, action, reward)
        row = [time.time(), *feat, reward]

        return row, action, conf, close