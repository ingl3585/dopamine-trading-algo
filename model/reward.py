# model/reward.py

class RewardCalculator:
    def __init__(self):
        self.recent_rewards = []
        self.prev_volatility = None

    def compute_reward(self, price_change, atr, state_data=None):
        base_reward = price_change / (atr + 1e-6)
        consistency_reward = self._calculate_consistency_bonus(base_reward)
        volatility_reward = self._calculate_volatility_bonus(base_reward, atr, state_data)
        return base_reward + consistency_reward + volatility_reward

    def modify_reward(self, action, reward):
        if action == 1:  # Buy
            reward -= 0.01
        return reward

    def _calculate_consistency_bonus(self, base_reward):
        if len(self.recent_rewards) >= 10:
            win_rate = sum(1 for r in self.recent_rewards if r > 0) / len(self.recent_rewards)
            bonus = 0.2 * (1.0 - 2.0 * abs(win_rate - 0.65))
        else:
            bonus = 0.0

        self.recent_rewards.append(base_reward)
        if len(self.recent_rewards) > 50:
            self.recent_rewards.pop(0)

        return bonus

    def _calculate_volatility_bonus(self, base_reward, atr, state_data):
        if state_data is not None:
            current_volatility = state_data[0, -1, 2].item()
        else:
            current_volatility = atr

        if self.prev_volatility is None:
            self.prev_volatility = current_volatility
            return 0.0

        vol_change = abs(current_volatility - self.prev_volatility)
        self.prev_volatility = current_volatility

        if vol_change > 0.0005:
            return 0.1 * min(base_reward, 1.0)

        return 0.0

