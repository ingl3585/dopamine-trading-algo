# policy_network.py

import torch
import torch.nn as nn


class PolicyRNN(nn.Module):
    """
    Tiny LSTM head ⇒
        • logits (3-way: HOLD / BUY / SELL)
        • stop_offset_atr  (≥0)
        • tp_offset_atr    (≥0)
    """
    def __init__(self, obs_size: int = 15, hidden_size: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(obs_size, hidden_size, batch_first=True)
        self.action_head = nn.Linear(hidden_size, 3)
        self.stop_head = nn.Linear(hidden_size, 1)
        self.tp_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, obs_size)
        h, _ = self.lstm(x)
        h_last = h[:, -1]                     # final step
        logits = self.action_head(h_last)
        stop_offset = torch.relu(self.stop_head(h_last))
        tp_offset = torch.relu(self.tp_head(h_last))
        return logits, stop_offset, tp_offset
