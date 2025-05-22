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
        """
        Process features and make prediction, handling rewards properly
        """
        close = feat[0]
        atr = feat[2] if len(feat) > 2 else 0.01

        # Calculate reward based on previous action and price change
        reward = 0.0
        if self.last_price is not None and self.last_action is not None:
            price_change = close - self.last_price
            
            # Calculate reward based on what action was taken
            if self.last_action == 1:  # BUY action
                reward = self.rewarder.compute_reward(price_change, atr, self.last_action)
            elif self.last_action == 2:  # SELL action  
                reward = self.rewarder.compute_reward(-price_change, atr, self.last_action)
            else:  # HOLD action
                reward = self.rewarder.compute_reward(0, atr, self.last_action)
            
            # Add experience to agent for online learning
            if self.last_features is not None:
                self.agent.add_experience(self.last_features, self.last_action, reward, feat)

        # Make prediction for current state
        action, conf = self.agent.predict_single(feat)
        
        # Store current state for next iteration
        self.last_price = close
        self.last_features = feat.copy()
        self.last_action = action

        # Create row for logging
        row = [time.time(), *feat, reward]

        return row, action, conf, close