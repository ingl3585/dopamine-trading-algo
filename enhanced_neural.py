# enhanced_neural.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Learnable temperature for attention scaling
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention with learnable temperature
        scores = torch.matmul(q, k.transpose(-2, -1)) / (math.sqrt(self.head_dim) * self.temperature)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out(out)


class CrossTimeframeAttention(nn.Module):
    def __init__(self, feature_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Separate encoders for each timeframe with positional encoding
        self.tf_1m_encoder = nn.Sequential(
            nn.Linear(20, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.tf_5m_encoder = nn.Sequential(
            nn.Linear(20, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.tf_15m_encoder = nn.Sequential(
            nn.Linear(20, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Positional encodings for timeframes
        self.timeframe_embeddings = nn.Embedding(3, feature_dim)
        
        # Cross-timeframe attention
        self.attention = MultiHeadAttention(feature_dim, num_heads)
        
        # Output projection with residual connection
        self.output = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, tf_1m: torch.Tensor, tf_5m: torch.Tensor, tf_15m: torch.Tensor) -> torch.Tensor:
        batch_size = tf_1m.shape[0]
        
        # Encode each timeframe
        enc_1m = self.tf_1m_encoder(tf_1m).unsqueeze(1)
        enc_5m = self.tf_5m_encoder(tf_5m).unsqueeze(1)
        enc_15m = self.tf_15m_encoder(tf_15m).unsqueeze(1)
        
        # Add positional embeddings
        timeframe_ids = torch.arange(3, device=tf_1m.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeddings = self.timeframe_embeddings(timeframe_ids).unsqueeze(1)
        
        # Stack timeframes for attention
        timeframes = torch.cat([enc_1m, enc_5m, enc_15m], dim=1)
        timeframes = timeframes + pos_embeddings
        
        # Apply cross-timeframe attention
        attended = self.attention(timeframes)
        
        # Combine all timeframes
        combined = attended.flatten(start_dim=1)
        
        return self.output(combined)


class AdaptiveMemoryLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Multi-layer LSTM with attention
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.1 if num_layers > 1 else 0)
        
        # Memory attention mechanism
        self.memory_attention = MultiHeadAttention(hidden_size, num_heads=2)
        
        # Adaptive memory capacity with gating
        self.memory_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Pattern persistence tracking with learnable memory bank
        self.pattern_memory = nn.Parameter(torch.randn(100, hidden_size) * 0.1)
        self.pattern_ages = nn.Parameter(torch.zeros(100))
        self.memory_importance = nn.Parameter(torch.ones(100))
        
        # Memory consolidation mechanism
        self.consolidation_threshold = 0.8
        
    def forward(self, x: torch.Tensor, hidden_state=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden_state)
        
        # Apply memory attention
        attended_memory = self.memory_attention(lstm_out)
        
        # Memory gate for adaptive capacity
        memory_importance = self.memory_gate(attended_memory)
        
        # Update pattern memory with consolidation
        current_pattern = attended_memory[:, -1:, :]
        self._update_pattern_memory(current_pattern)
        
        # Apply memory gating
        gated_memory = attended_memory * memory_importance
        
        return gated_memory, hidden
    
    def _update_pattern_memory(self, pattern: torch.Tensor):
        with torch.no_grad():
            # Find least important pattern slot for replacement
            importance_scores = self.memory_importance * (1.0 / (self.pattern_ages + 1))
            replacement_idx = torch.argmin(importance_scores)
            
            # Update memory with consolidation
            if self.memory_importance[replacement_idx] < self.consolidation_threshold:
                self.pattern_memory[replacement_idx] = pattern.squeeze()
                self.pattern_ages[replacement_idx] = 0
                self.memory_importance[replacement_idx] = 0.5
            
            # Age all patterns
            self.pattern_ages += 1
            
            # Decay importance over time
            self.memory_importance *= 0.999


class SelfEvolvingNetwork(nn.Module):
    def __init__(self, input_size: int = 64, initial_sizes: List[int] = [128, 64], 
                 evolution_frequency: int = 1000):
        super().__init__()
        self.input_size = input_size
        self.current_sizes = initial_sizes.copy()
        self.evolution_frequency = evolution_frequency
        self.forward_count = 0
        
        # Cross-timeframe processing
        self.timeframe_attention = CrossTimeframeAttention()
        
        # Enhanced memory system
        self.memory_lstm = AdaptiveMemoryLSTM(input_size, hidden_size=input_size)
        
        # Build initial network
        self._build_network()
        
        # Evolution tracking with performance history
        self.layer_usage = torch.zeros(len(self.current_sizes))
        self.layer_performance = deque(maxlen=200)
        self.architecture_history = []
        
        # Catastrophic forgetting prevention
        self.importance_weights = {}
        self.previous_tasks_data = deque(maxlen=1000)
        
    def _build_network(self):
        layers = []
        prev_size = self.input_size
        
        for i, size in enumerate(self.current_sizes):
            layers.extend([
                nn.Linear(prev_size, size),
                nn.LayerNorm(size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = size
        
        self.backbone = nn.Sequential(*layers)
        
        # Output heads with improved architecture
        self.action_head = nn.Sequential(
            nn.Linear(prev_size, prev_size // 2),
            nn.ReLU(),
            nn.Linear(prev_size // 2, 3)
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(prev_size, prev_size // 2),
            nn.ReLU(),
            nn.Linear(prev_size // 2, 1),
            nn.Sigmoid()
        )
        
        self.position_head = nn.Sequential(
            nn.Linear(prev_size, prev_size // 2),
            nn.ReLU(),
            nn.Linear(prev_size // 2, 1),
            nn.Sigmoid()
        )
        
        self.risk_head = nn.Sequential(
            nn.Linear(prev_size, prev_size // 2),
            nn.ReLU(),
            nn.Linear(prev_size // 2, 4),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, timeframe_data: Dict = None) -> Dict[str, torch.Tensor]:
        self.forward_count += 1
        
        # Process multiple timeframes if available
        if timeframe_data:
            x = self.timeframe_attention(
                timeframe_data.get('1m', x),
                timeframe_data.get('5m', x),
                timeframe_data.get('15m', x)
            )
        
        # Memory processing with sequence handling
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        memory_out, _ = self.memory_lstm(x)
        x = memory_out.squeeze(1)  # Remove sequence dimension
        
        # Track layer usage for evolution
        with torch.no_grad():
            for i in range(len(self.current_sizes)):
                self.layer_usage[i] += 1
        
        # Forward through backbone
        features = self.backbone(x)
        
        # Output heads
        outputs = {
            'action_logits': self.action_head(features),
            'confidence': self.confidence_head(features),
            'position_size': self.position_head(features) * 3.0 + 0.5,
            'risk_params': self.risk_head(features)
        }
        
        # Periodic evolution check
        if self.forward_count % self.evolution_frequency == 0:
            self._consider_evolution()
        
        return outputs
    
    def _consider_evolution(self):
        if len(self.layer_performance) < 50:
            return
        
        recent_performance = np.mean(list(self.layer_performance)[-20:])
        historical_performance = np.mean(list(self.layer_performance)[-50:-20])
        
        # Evolution trigger: performance degradation
        if recent_performance < historical_performance - 0.1:
            self._evolve_architecture()
    
    def _evolve_architecture(self):
        """Evolve network architecture with catastrophic forgetting prevention"""
        # Store current architecture performance
        self.architecture_history.append({
            'sizes': self.current_sizes.copy(),
            'performance': np.mean(list(self.layer_performance)[-20:]) if self.layer_performance else 0.0
        })
        
        # Calculate importance weights for catastrophic forgetting prevention
        self._calculate_importance_weights()
        
        # Generate new architecture
        new_sizes = self._generate_new_architecture()
        
        if new_sizes != self.current_sizes:
            old_state = self.state_dict()
            self.current_sizes = new_sizes
            self._build_network()
            
            # Transfer weights with importance preservation
            self._transfer_weights_with_importance(old_state)
            
            # Reset evolution tracking
            self.layer_usage = torch.zeros(len(self.current_sizes))
            
    def _calculate_importance_weights(self):
        """Calculate importance weights for parameters to prevent catastrophic forgetting"""
        for name, param in self.named_parameters():
            if param.grad is not None:
                # Fisher Information approximation
                importance = param.grad.data.clone().pow(2)
                
                if name in self.importance_weights:
                    # Accumulate importance over time
                    self.importance_weights[name] = 0.9 * self.importance_weights[name] + 0.1 * importance
                else:
                    self.importance_weights[name] = importance
    
    def _generate_new_architecture(self) -> List[int]:
        """Generate new architecture based on performance and usage patterns"""
        new_sizes = []
        
        for i, current_size in enumerate(self.current_sizes):
            usage_ratio = float(self.layer_usage[i]) / max(1, self.forward_count)
            
            # Adaptive sizing based on usage and performance
            if usage_ratio > 0.8:  # High usage - consider expansion
                growth_factor = np.random.uniform(1.1, 1.3)
                new_size = int(current_size * growth_factor)
            elif usage_ratio < 0.3:  # Low usage - consider reduction
                shrink_factor = np.random.uniform(0.7, 0.9)
                new_size = int(current_size * shrink_factor)
            else:  # Moderate usage - small random mutation
                mutation_factor = np.random.uniform(0.9, 1.1)
                new_size = int(current_size * mutation_factor)
            
            # Enforce reasonable bounds
            new_size = max(16, min(256, new_size))
            new_sizes.append(new_size)
        
        return new_sizes
    
    def _transfer_weights_with_importance(self, old_state: Dict):
        """Transfer weights from old architecture with importance preservation"""
        current_state = self.state_dict()
        
        for name, param in old_state.items():
            if name in current_state:
                old_shape = param.shape
                new_shape = current_state[name].shape
                
                if old_shape == new_shape:
                    # Direct transfer for same size
                    current_state[name] = param
                elif len(old_shape) == len(new_shape):
                    # Adaptive transfer for different sizes
                    if name in self.importance_weights:
                        # Use importance weights to guide transfer
                        importance = self.importance_weights[name]
                        current_state[name] = self._adaptive_weight_transfer(
                            param, current_state[name], importance
                        )
                    else:
                        # Simple truncation/padding
                        current_state[name] = self._simple_weight_transfer(
                            param, current_state[name]
                        )
        
        self.load_state_dict(current_state)
    
    def _adaptive_weight_transfer(self, old_weights: torch.Tensor, 
                                new_weights: torch.Tensor, 
                                importance: torch.Tensor) -> torch.Tensor:
        """Transfer weights using importance information"""
        old_shape = old_weights.shape
        new_shape = new_weights.shape
        
        # Find overlapping region
        overlap_shape = tuple(min(old_shape[i], new_shape[i]) for i in range(len(old_shape)))
        
        # Create slices for overlapping region
        old_slices = tuple(slice(0, overlap_shape[i]) for i in range(len(overlap_shape)))
        new_slices = tuple(slice(0, overlap_shape[i]) for i in range(len(overlap_shape)))
        
        # Transfer important weights
        result = new_weights.clone()
        if len(overlap_shape) > 0:
            old_overlap = old_weights[old_slices]
            importance_overlap = importance[old_slices] if importance.shape == old_shape else torch.ones_like(old_overlap)
            
            # Weight transfer based on importance
            result[new_slices] = old_overlap * importance_overlap + result[new_slices] * (1 - importance_overlap)
        
        return result
    
    def _simple_weight_transfer(self, old_weights: torch.Tensor, new_weights: torch.Tensor) -> torch.Tensor:
        """Simple weight transfer with truncation/padding"""
        old_shape = old_weights.shape
        new_shape = new_weights.shape
        
        # Find overlapping region
        overlap_shape = tuple(min(old_shape[i], new_shape[i]) for i in range(len(old_shape)))
        
        # Create slices for overlapping region
        old_slices = tuple(slice(0, overlap_shape[i]) for i in range(len(overlap_shape)))
        new_slices = tuple(slice(0, overlap_shape[i]) for i in range(len(overlap_shape)))
        
        # Transfer overlapping weights
        result = new_weights.clone()
        if len(overlap_shape) > 0:
            result[new_slices] = old_weights[old_slices]
        
        return result
    
    def record_performance(self, performance: float):
        """Record performance for evolution decisions"""
        self.layer_performance.append(performance)
    
    def add_previous_task_data(self, data: torch.Tensor, target: torch.Tensor):
        """Store data from previous tasks to prevent catastrophic forgetting"""
        self.previous_tasks_data.append((data, target))
    
    def get_evolution_stats(self) -> Dict:
        """Get evolution statistics"""
        return {
            'current_architecture': self.current_sizes,
            'forward_count': self.forward_count,
            'evolution_count': len(self.architecture_history),
            'layer_usage': self.layer_usage.tolist(),
            'recent_performance': np.mean(list(self.layer_performance)[-10:]) if self.layer_performance else 0.0,
            'architecture_history': self.architecture_history[-5:]  # Last 5 evolutions
        }


class FewShotLearner(nn.Module):
    def __init__(self, feature_dim: int = 64, support_size: int = 5):
        super().__init__()
        self.feature_dim = feature_dim
        self.support_size = support_size
        
        # Prototype network for few-shot learning
        self.prototype_network = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Attention mechanism for prototype weighting
        self.attention = nn.MultiheadAttention(feature_dim, num_heads=4, batch_first=True)
        
        # Support set storage
        self.support_features = deque(maxlen=support_size * 10)
        self.support_labels = deque(maxlen=support_size * 10)
        
    def forward(self, query_features: torch.Tensor) -> torch.Tensor:
        if len(self.support_features) < self.support_size:
            # Not enough support examples, return zero prediction
            return torch.zeros(query_features.shape[0], 1)
        
        # Get recent support examples
        support_batch = torch.stack(list(self.support_features)[-self.support_size:])
        support_labels_batch = torch.tensor(list(self.support_labels)[-self.support_size:])
        
        # Generate prototypes
        prototypes = self.prototype_network(support_batch)
        
        # Apply attention to weight prototypes
        query_expanded = query_features.unsqueeze(1).expand(-1, self.support_size, -1)
        attended_prototypes, attention_weights = self.attention(
            query_expanded, prototypes.unsqueeze(0).expand(query_features.shape[0], -1, -1),
            prototypes.unsqueeze(0).expand(query_features.shape[0], -1, -1)
        )
        
        # Weighted prediction based on prototype similarity
        similarities = F.cosine_similarity(
            query_expanded, attended_prototypes, dim=-1
        )
        
        # Weight similarities by support labels
        weighted_similarities = similarities * support_labels_batch.unsqueeze(0)
        predictions = torch.sum(weighted_similarities, dim=1, keepdim=True)
        
        return predictions
    
    def add_support_example(self, features: torch.Tensor, label: float):
        """Add new support example for few-shot learning"""
        self.support_features.append(features.detach())
        self.support_labels.append(label)


def create_enhanced_network(input_size: int = 64, 
                          initial_sizes: List[int] = [128, 64],
                          enable_few_shot: bool = True) -> SelfEvolvingNetwork:
    """Factory function for creating enhanced network with optional few-shot learning"""
    network = SelfEvolvingNetwork(input_size=input_size, initial_sizes=initial_sizes)
    
    if enable_few_shot:
        network.few_shot_learner = FewShotLearner(feature_dim=input_size)
    
    return network