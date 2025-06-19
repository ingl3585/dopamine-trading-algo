# enhanced_neural.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention = F.softmax(attention, dim=-1)
        
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out(out)


class CrossTimeframeAttention(nn.Module):
    def __init__(self, feature_dim: int = 64):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Separate encoders for each timeframe
        self.tf_1m_encoder = nn.Linear(20, feature_dim)
        self.tf_5m_encoder = nn.Linear(20, feature_dim)
        self.tf_15m_encoder = nn.Linear(20, feature_dim)
        
        # Cross-timeframe attention
        self.attention = MultiHeadAttention(feature_dim, num_heads=4)
        
        # Output projection
        self.output = nn.Linear(feature_dim * 3, feature_dim)
        
    def forward(self, tf_1m: torch.Tensor, tf_5m: torch.Tensor, tf_15m: torch.Tensor) -> torch.Tensor:
        # Encode each timeframe
        enc_1m = self.tf_1m_encoder(tf_1m).unsqueeze(1)
        enc_5m = self.tf_5m_encoder(tf_5m).unsqueeze(1)
        enc_15m = self.tf_15m_encoder(tf_15m).unsqueeze(1)
        
        # Stack timeframes for attention
        timeframes = torch.cat([enc_1m, enc_5m, enc_15m], dim=1)
        
        # Apply cross-timeframe attention
        attended = self.attention(timeframes)
        
        # Combine all timeframes
        combined = attended.flatten(start_dim=1)
        
        return self.output(combined)


class AdaptiveMemoryLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128):
        super().__init__()
        self.hidden_size = hidden_size
        
        # LSTM with attention
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.memory_attention = MultiHeadAttention(hidden_size, num_heads=2)
        
        # Adaptive memory capacity
        self.memory_gate = nn.Linear(hidden_size, 1)
        
        # Pattern persistence tracking
        self.pattern_memory = nn.Parameter(torch.zeros(100, hidden_size))
        self.pattern_ages = nn.Parameter(torch.zeros(100))
        
    def forward(self, x: torch.Tensor, hidden_state=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # LSTM forward
        lstm_out, hidden = self.lstm(x, hidden_state)
        
        # Apply memory attention
        attended_memory = self.memory_attention(lstm_out)
        
        # Memory gate for adaptive capacity
        memory_importance = torch.sigmoid(self.memory_gate(attended_memory))
        
        # Update pattern memory
        current_pattern = attended_memory[:, -1:, :]
        self._update_pattern_memory(current_pattern)
        
        return attended_memory * memory_importance, hidden
    
    def _update_pattern_memory(self, pattern: torch.Tensor):
        with torch.no_grad():
            # Find oldest pattern slot
            oldest_idx = torch.argmax(self.pattern_ages)
            
            # Update memory
            self.pattern_memory[oldest_idx] = pattern.squeeze()
            self.pattern_ages[oldest_idx] = 0
            self.pattern_ages += 1

class SelfEvolvingNetwork(nn.Module):
    def __init__(self, input_size: int = 64, initial_sizes: List[int] = [128, 64]):
        super().__init__()
        self.input_size = input_size
        self.current_sizes = initial_sizes.copy()
        
        # Cross-timeframe processing
        self.timeframe_attention = CrossTimeframeAttention()
        
        # Memory system
        self.memory_lstm = AdaptiveMemoryLSTM(input_size)
        
        # Build initial network
        self._build_network()
        
        # Evolution tracking
        self.layer_usage = torch.zeros(len(self.current_sizes))
        self.layer_performance = torch.zeros(len(self.current_sizes))
        
    def _build_network(self):
        layers = []
        prev_size = self.input_size
        
        for i, size in enumerate(self.current_sizes):
            layers.extend([
                nn.Linear(prev_size, size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = size
        
        self.backbone = nn.Sequential(*layers)
        
        # Output heads
        self.action_head = nn.Linear(prev_size, 3)
        self.confidence_head = nn.Linear(prev_size, 1)
        self.position_head = nn.Linear(prev_size, 1)
        self.risk_head = nn.Linear(prev_size, 4)
        
    def forward(self, x: torch.Tensor, timeframe_data: Dict = None) -> Dict[str, torch.Tensor]:
        # Process multiple timeframes if available
        if timeframe_data:
            x = self.timeframe_attention(
                timeframe_data.get('1m', x),
                timeframe_data.get('5m', x),
                timeframe_data.get('15m', x)
            )
        
        # Memory processing
        memory_out, _ = self.memory_lstm(x.unsqueeze(1))
        x = memory_out.squeeze(1)
        
        # Track layer usage
        with torch.no_grad():
            for i in range(len(self.current_sizes)):
                self.layer_usage[i] += 1
        
        # Forward through backbone
        features = self.backbone(x)
        
        # Output heads
        return {
            'action_logits': self.action_head(features),
            'confidence': torch.sigmoid(self.confidence_head(features)),
            'position_size': torch.sigmoid(self.position_head(features)) * 3.0 + 0.5,
            'risk_params': torch.sigmoid(self.risk_head(features))
        }
    
    def evolve_architecture(self, performance_history: List[float]) -> List[int]:
        if len(performance_history) < 20:
            return self.current_sizes
        
        recent_performance = np.mean(performance_history[-10:])
        
        # Simple evolution strategy
        if recent_performance < -0.1:
            # Performance is poor, evolve
            new_sizes = []
            for size in self.current_sizes:
                # Random mutation
                mutation = np.random.choice([-16, -8, 0, 8, 16])
                new_size = max(32, min(256, size + mutation))
                new_sizes.append(new_size)
            
            self.current_sizes = new_sizes
            self._build_network()
        
        return self.current_sizes
    
    def prune_unused_layers(self):
        # Simple pruning based on usage
        usage_threshold = torch.median(self.layer_usage) * 0.1
        active_layers = self.layer_usage > usage_threshold
        
        if active_layers.sum() < len(self.current_sizes):
            # Rebuild with active layers only
            new_sizes = [size for i, size in enumerate(self.current_sizes) if active_layers[i]]
            if new_sizes:
                self.current_sizes = new_sizes
                self._build_network()

class AdvancedFeatureExtractor(nn.Module):
    def __init__(self, raw_dim: int = 100, learned_dim: int = 64):
        super().__init__()
        self.raw_dim = raw_dim
        self.learned_dim = learned_dim
        
        # Feature importance learning
        self.importance_net = nn.Sequential(
            nn.Linear(raw_dim, raw_dim // 2),
            nn.ReLU(),
            nn.Linear(raw_dim // 2, raw_dim),
            nn.Sigmoid()
        )
        
        # Feature combination
        self.combiner = nn.Sequential(
            nn.Linear(raw_dim, learned_dim * 2),
            nn.ReLU(),
            nn.Linear(learned_dim * 2, learned_dim)
        )
        
        # Feature evolution tracking
        self.feature_history = torch.zeros(raw_dim, 100)
        self.history_idx = 0
        
    def forward(self, raw_features: torch.Tensor) -> torch.Tensor:
        # Learn feature importance
        importance = self.importance_net(raw_features)
        
        # Apply importance weighting
        weighted_features = raw_features * importance
        
        # Combine features
        learned_features = self.combiner(weighted_features)
        
        # Update feature history
        with torch.no_grad():
            self.feature_history[:, self.history_idx] = importance.squeeze()
            self.history_idx = (self.history_idx + 1) % 100
        
        return learned_features
    
    def get_feature_importance(self) -> torch.Tensor:
        return torch.mean(self.feature_history, dim=1)
    
    def evolve_features(self):
        # Identify consistently unimportant features
        importance = self.get_feature_importance()
        threshold = torch.quantile(importance, 0.2)
        
        # Zero out unimportant features
        with torch.no_grad():
            mask = importance > threshold
            for param in self.importance_net.parameters():
                if param.dim() == 2:
                    param.data *= mask.unsqueeze(0)


def create_enhanced_network(input_size: int = 64) -> SelfEvolvingNetwork:
    """Factory function for creating enhanced network"""
    return SelfEvolvingNetwork(input_size=input_size)