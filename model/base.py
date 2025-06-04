# model/base.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiTimeframeActorCritic(nn.Module):
    """
    Enhanced Actor-Critic model with multi-timeframe awareness and attention mechanism
    """
    def __init__(self, input_dim=27, hidden_dim=256, action_dim=3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        
        # Timeframe-specific encoders (9 features each)
        encoder_dim = hidden_dim // 3
        
        # 15-minute encoder (trend context) - deeper network for trend analysis
        self.encoder_15m = nn.Sequential(
            nn.Linear(9, encoder_dim),
            nn.LayerNorm(encoder_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(encoder_dim, encoder_dim),
            nn.LayerNorm(encoder_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 5-minute encoder (momentum context) - balanced depth
        self.encoder_5m = nn.Sequential(
            nn.Linear(9, encoder_dim),
            nn.LayerNorm(encoder_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(encoder_dim, encoder_dim),
            nn.LayerNorm(encoder_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 1-minute encoder (entry timing) - lighter network for responsiveness
        self.encoder_1m = nn.Sequential(
            nn.Linear(9, encoder_dim),
            nn.LayerNorm(encoder_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Cross-timeframe attention mechanism
        self.timeframe_attention = TimeframeAttention(encoder_dim, num_heads=3)
        
        # Timeframe importance weights (learnable)
        self.timeframe_weights = nn.Parameter(torch.tensor([0.4, 0.35, 0.25]))  # 15m, 5m, 1m
        
        # Combined processing layers
        self.shared_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Output heads
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, action_dim)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights with appropriate scaling"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x, temperature=1.0, return_attention=False):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        elif len(x.shape) == 3:
            x = x.squeeze(1) if x.shape[1] == 1 else x[:, -1, :]
        
        batch_size = x.shape[0]
        
        # Split input into timeframes
        feat_15m = x[:, 0:9]    # Trend context
        feat_5m = x[:, 9:18]    # Momentum context  
        feat_1m = x[:, 18:27]   # Entry timing
        
        # Encode each timeframe
        enc_15m = self.encoder_15m(feat_15m)
        enc_5m = self.encoder_5m(feat_5m)
        enc_1m = self.encoder_1m(feat_1m)
        
        # Stack for attention processing
        timeframe_encodings = torch.stack([enc_15m, enc_5m, enc_1m], dim=1)  # [batch, 3, encoder_dim]
        
        # Apply cross-timeframe attention
        attended_features, attention_weights = self.timeframe_attention(timeframe_encodings)
        
        # Apply learnable timeframe importance weights
        weights = F.softmax(self.timeframe_weights, dim=0)
        weighted_features = attended_features * weights.view(1, 3, 1)
        
        # Combine timeframe features
        combined_features = weighted_features.view(batch_size, -1)  # [batch, hidden_dim]
        
        # Shared processing
        shared_output = self.shared_layers(combined_features)
        
        # Actor output (action probabilities)
        actor_logits = self.actor(shared_output) / temperature
        action_probs = F.softmax(actor_logits, dim=-1)
        
        # Critic output (value estimate)
        value = self.critic(shared_output)
        
        if return_attention:
            return action_probs, value, attention_weights, weights
        else:
            return action_probs, value
    
    def get_timeframe_importance(self):
        """Get current timeframe importance weights"""
        weights = F.softmax(self.timeframe_weights, dim=0)
        return {
            '15m_weight': weights[0].item(),
            '5m_weight': weights[1].item(), 
            '1m_weight': weights[2].item()
        }

class TimeframeAttention(nn.Module):
    """
    Attention mechanism for cross-timeframe feature interaction
    """
    def __init__(self, embed_dim, num_heads=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x):
        # x shape: [batch_size, 3, embed_dim] (3 timeframes)
        batch_size, seq_len, embed_dim = x.shape
        
        # Residual connection
        residual = x
        
        # Layer norm
        x = self.layer_norm(x)
        
        # Linear projections
        q = self.q_proj(x)  # [batch, 3, embed_dim]
        k = self.k_proj(x)  # [batch, 3, embed_dim]
        v = self.v_proj(x)  # [batch, 3, embed_dim]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, v)
        
        # Reshape and project
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.out_proj(attended)
        
        # Residual connection
        output = output + residual
        
        # Return attended features and attention weights for analysis
        avg_attention = attention_weights.mean(dim=1)  # Average across heads
        
        return output, avg_attention

# Legacy ActorCritic class for backward compatibility
class ActorCritic(nn.Module):
    """Original ActorCritic - kept for backward compatibility"""
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