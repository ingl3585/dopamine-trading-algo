# utils/portfolio.py

class Portfolio:
    def __init__(self, max_position=10):
        self.position = 0
        self.max_position = max_position

    def get_current_position(self):
        return self.position

    def update_position(self, pos):
        self.position = pos

    def can_execute(self, action, size):
        if action == 1:  # Buy
            return self.position + size <= self.max_position
        elif action == 2:  # Sell
            return self.position - size >= -self.max_position
        return False # Hold

    def adjust_size(self, action, size):
        if action == 1: # Buy
            return min(size, self.max_position - self.position)
        elif action == 2: # Sell
            return min(size, self.max_position + self.position)
        return 0 # Hold