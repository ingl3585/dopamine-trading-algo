# model/base.py

import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super().__init__()
        
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, temperature=1.0):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        elif len(x.shape) == 3:
            x = x.squeeze(1) if x.shape[1] == 1 else x[:, -1, :]
            
        features = self.shared_layers(x)
        
        logits = self.actor(features) / temperature
        probs = F.softmax(logits, dim=-1)
        value = self.critic(features)
        
        return probs, value