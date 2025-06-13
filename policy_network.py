
# File 1: Update policy_network.py - Make it truly black box

import torch
import torch.nn as nn

class BlackBoxPolicyNetwork(nn.Module):
    """
    COMPLETE black box that learns ALL trading decisions:
    - Action (buy/sell/hold)  
    - Position size
    - Whether to use stops/targets
    - Stop/target levels
    - Exit timing
    """
    
    def __init__(self, obs_size: int = 15, hidden_size: int = 128):
        super().__init__()
        
        # Larger network for complex decisions
        self.lstm = nn.LSTM(obs_size, hidden_size, batch_first=True)
        
        # Decision heads - AI learns EVERYTHING
        self.action_head = nn.Linear(hidden_size, 3)        # buy/sell/hold
        self.position_size_head = nn.Linear(hidden_size, 1) # how much to trade
        self.use_stop_head = nn.Linear(hidden_size, 1)      # whether to use stop
        self.stop_distance_head = nn.Linear(hidden_size, 1) # stop distance
        self.use_target_head = nn.Linear(hidden_size, 1)    # whether to use target
        self.target_distance_head = nn.Linear(hidden_size, 1) # target distance
        self.exit_confidence_head = nn.Linear(hidden_size, 1) # exit current position?
        self.overall_confidence_head = nn.Linear(hidden_size, 1) # decision confidence

    def forward(self, x):
        h, _ = self.lstm(x)
        h_last = h[:, -1]  # Last timestep
        
        # All decisions made by AI
        action_logits = self.action_head(h_last)
        position_size = torch.sigmoid(self.position_size_head(h_last))  # 0-1
        use_stop = torch.sigmoid(self.use_stop_head(h_last))            # 0-1 probability
        stop_distance = torch.sigmoid(self.stop_distance_head(h_last)) * 0.05  # 0-5% of price
        use_target = torch.sigmoid(self.use_target_head(h_last))        # 0-1 probability
        target_distance = torch.sigmoid(self.target_distance_head(h_last)) * 0.10  # 0-10% of price
        exit_confidence = torch.sigmoid(self.exit_confidence_head(h_last))  # 0-1
        overall_confidence = torch.sigmoid(self.overall_confidence_head(h_last))  # 0-1
        
        return {
            'action_logits': action_logits,
            'position_size': position_size,
            'use_stop': use_stop,
            'stop_distance': stop_distance,
            'use_target': use_target,
            'target_distance': target_distance,
            'exit_confidence': exit_confidence,
            'overall_confidence': overall_confidence
        }
