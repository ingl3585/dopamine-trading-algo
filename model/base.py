# model/base.py

import torch.nn as nn
import torch.nn.functional as F

class BayesianLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.dropout(lstm_out[:, -1, :])

class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super().__init__()
        self.feature_extractor = BayesianLSTM(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, action_dim))

        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1))

    def forward(self, x, temperature=1.0):
        features = self.feature_extractor(x)
        features = self.dropout(features)
        logits = self.actor(features) / temperature
        probs = F.softmax(logits, dim=-1)
        value = self.critic(features)
        return probs, value