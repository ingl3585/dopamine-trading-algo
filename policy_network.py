# policy_network.py

import torch
import torch.nn as nn

class PolicyRNN(nn.Module):
    """
    Pure black box LSTM that learns raw price stop/target distances
    • logits (3-way: HOLD / BUY / SELL)
    • stop_points  (raw $ distance for stops)
    • tp_points    (raw $ distance for targets)
    """
    def __init__(self, obs_size: int = 15, hidden_size: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(obs_size, hidden_size, batch_first=True)
        self.action_head = nn.Linear(hidden_size, 3)
        
        # Output raw price distances (scaled for MNQ typical ranges)
        self.stop_head = nn.Linear(hidden_size, 1)   # Will output 0-50 points
        self.tp_head = nn.Linear(hidden_size, 1)     # Will output 0-50 points

    def forward(self, x):
        # x shape: (batch, seq_len, obs_size)
        h, _ = self.lstm(x)
        h_last = h[:, -1]                     # final step
        
        logits = self.action_head(h_last)
        
        # Scale outputs to reasonable MNQ ranges
        # Sigmoid * 50 gives 0-50 point range ($0-$100 for MNQ)
        stop_points = torch.sigmoid(self.stop_head(h_last)) * 50.0
        tp_points = torch.sigmoid(self.tp_head(h_last)) * 50.0
        
        return logits, stop_points, tp_points