# enhanced_neural.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
import math


class BoundedSigmoid(nn.Module):
    """Sigmoid activation with configurable bounds to prevent extreme values"""
    def __init__(self, min_value: float = 0.1, max_value: float = 1.0):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.range = max_value - min_value
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard sigmoid scaled to [min_value, max_value] range
        result = torch.sigmoid(x) * self.range + self.min_value
        
        # DEBUG: Log if output is below minimum (should never happen)
        if torch.any(result < self.min_value - 1e-6):
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"BoundedSigmoid error: output {result.min().item():.6f} below min {self.min_value}")
            logger.error(f"Input tensor: {x}")
            logger.error(f"Sigmoid output: {torch.sigmoid(x)}")
            logger.error(f"Range: {self.range}, Min: {self.min_value}")
        
        # CRITICAL FIX: Ensure result never goes below minimum
        result = torch.clamp(result, min=self.min_value, max=self.max_value)
        
        return result


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
        self.temperature = nn.Parameter(torch.ones(1, dtype=torch.float32))
        
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
        
        # Separate encoders for each timeframe - all expect same 32-feature input
        self.tf_1m_encoder = nn.Sequential(
            nn.Linear(32, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.tf_5m_encoder = nn.Sequential(
            nn.Linear(32, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.tf_15m_encoder = nn.Sequential(
            nn.Linear(32, feature_dim),
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
        self.pattern_memory = nn.Parameter(torch.randn(100, hidden_size, dtype=torch.float32) * 0.1)
        self.pattern_ages = nn.Parameter(torch.zeros(100, dtype=torch.float32))
        self.memory_importance = nn.Parameter(torch.ones(100, dtype=torch.float32))
        
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
        
        # Use consistent float32 precision
        self.float()
        
        # Evolution tracking with performance history
        self.layer_usage = torch.zeros(len(self.current_sizes), dtype=torch.float32)
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
            BoundedSigmoid(min_value=0.1, max_value=1.0)
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
            self.layer_usage = torch.zeros(len(self.current_sizes), dtype=torch.float32)
            
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
        
        # Use consistent float32 precision
        self.float()
        
        # Support set storage
        self.support_features = deque(maxlen=support_size * 10)
        self.support_labels = deque(maxlen=support_size * 10)
        
    def forward(self, query_features: torch.Tensor) -> torch.Tensor:
        if len(self.support_features) < self.support_size:
            # Not enough support examples, return zero prediction
            return torch.zeros(query_features.shape[0], 1, dtype=torch.float32, device=query_features.device)
        
        # Get recent support examples with consistent dtypes
        support_batch = torch.stack(list(self.support_features)[-self.support_size:]).to(dtype=torch.float32, device=query_features.device)
        support_labels_batch = torch.tensor(list(self.support_labels)[-self.support_size:], dtype=torch.float32, device=query_features.device)
        
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


class ActorCriticLoss(nn.Module):
    """Comprehensive loss function for actor-critic trading agent"""
    def __init__(self, confidence_weight: float = 0.3, position_weight: float = 0.2, 
                 risk_weight: float = 0.1, entropy_weight: float = 0.01):
        super().__init__()
        self.confidence_weight = confidence_weight
        self.position_weight = position_weight
        self.risk_weight = risk_weight
        self.entropy_weight = entropy_weight
        
        # Adaptive loss scaling
        self.loss_scales = nn.Parameter(torch.ones(4, dtype=torch.float32))
        
    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], 
                rewards: torch.Tensor, advantages: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute multi-objective loss for trading agent
        
        Args:
            outputs: Network outputs (action_logits, confidence, position_size, risk_params)
            targets: Target values for supervised components
            rewards: Reward signals for reinforcement learning
            advantages: Advantage estimates for policy gradient
        """
        losses = {}
        
        # 1. Policy Loss (Actor) - with entropy regularization
        action_logits = outputs['action_logits']
        action_probs = F.softmax(action_logits, dim=-1)
        
        if 'actions' in targets:
            # Cross-entropy loss for action prediction
            policy_loss = F.cross_entropy(action_logits, targets['actions'])
            
            # Add entropy regularization to encourage exploration
            entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=-1)
            entropy_loss = -self.entropy_weight * entropy.mean()
            
            losses['policy'] = policy_loss + entropy_loss
        else:
            # Policy gradient loss when no action targets available
            log_probs = F.log_softmax(action_logits, dim=-1)
            if 'action_indices' in targets:
                selected_log_probs = log_probs.gather(1, targets['action_indices'].unsqueeze(1))
                policy_loss = -(selected_log_probs * advantages.unsqueeze(1)).mean()
                losses['policy'] = policy_loss
            else:
                losses['policy'] = torch.tensor(0.0, device=action_logits.device)
        
        # 2. Value Loss (Critic) - confidence as value function
        confidence = outputs['confidence']
        if 'confidence_targets' in targets:
            # MSE loss for confidence prediction
            confidence_loss = F.mse_loss(confidence, targets['confidence_targets'])
            losses['confidence'] = self.confidence_weight * confidence_loss
        else:
            # Use rewards as confidence targets in RL setting
            confidence_loss = F.mse_loss(confidence.squeeze(), rewards)
            losses['confidence'] = self.confidence_weight * confidence_loss
        
        # 3. Position Sizing Loss
        position_size = outputs['position_size']
        if 'position_targets' in targets:
            position_loss = F.mse_loss(position_size, targets['position_targets'])
            losses['position'] = self.position_weight * position_loss
        else:
            # Regularization to prevent extreme position sizes
            position_reg = torch.mean((position_size - 1.0) ** 2)
            losses['position'] = self.position_weight * position_reg
        
        # 4. Risk Management Loss
        risk_params = outputs['risk_params']
        if 'risk_targets' in targets:
            risk_loss = F.mse_loss(risk_params, targets['risk_targets'])
            losses['risk'] = self.risk_weight * risk_loss
        else:
            # Encourage reasonable risk parameters
            risk_reg = torch.mean((risk_params - 0.5) ** 2)
            losses['risk'] = self.risk_weight * risk_reg
        
        # 5. Total Loss with adaptive scaling
        total_loss = torch.tensor(0.0, device=action_logits.device)
        loss_names = ['policy', 'confidence', 'position', 'risk']
        
        for i, name in enumerate(loss_names):
            if name in losses:
                scaled_loss = losses[name] * torch.abs(self.loss_scales[i])
                total_loss = total_loss + scaled_loss
                losses[f'{name}_scaled'] = scaled_loss
        
        losses['total'] = total_loss
        
        # 6. Additional metrics for monitoring
        losses['entropy'] = entropy.mean() if 'actions' in targets else torch.tensor(0.0)
        losses['confidence_mean'] = confidence.mean()
        losses['position_mean'] = position_size.mean()
        
        return losses


class TradingOptimizer:
    """Enhanced optimizer for trading neural networks with adaptive learning rates"""
    def __init__(self, networks: List[nn.Module], base_lr: float = 0.001):
        self.networks = networks
        self.base_lr = base_lr
        
        # Separate optimizers for different components
        self.policy_optimizer = torch.optim.AdamW(
            [p for net in networks for p in net.parameters() if 'action' in str(net)],
            lr=base_lr, weight_decay=1e-5
        )
        
        self.value_optimizer = torch.optim.AdamW(
            [p for net in networks for p in net.parameters() if 'confidence' in str(net)],
            lr=base_lr * 2, weight_decay=1e-4  # Higher LR for value function
        )
        
        self.feature_optimizer = torch.optim.AdamW(
            [p for net in networks for p in net.parameters() 
             if not any(x in str(net) for x in ['action', 'confidence'])],
            lr=base_lr * 0.5, weight_decay=1e-6  # Lower LR for feature learning
        )
        
        # Learning rate schedulers
        self.policy_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.policy_optimizer, mode='max', factor=0.8, patience=100, verbose=True
        )
        
        self.value_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.value_optimizer, mode='min', factor=0.9, patience=50, verbose=True
        )
        
        # Performance tracking
        self.performance_history = []
        
    def step(self, losses: Dict[str, torch.Tensor], performance_metric: float = None):
        """Perform optimization step with component-specific updates"""
        
        # Policy update
        if 'policy' in losses:
            self.policy_optimizer.zero_grad()
            losses['policy'].backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(
                [p for net in self.networks for p in net.parameters()], max_norm=1.0
            )
            self.policy_optimizer.step()
        
        # Value function update
        if 'confidence' in losses:
            self.value_optimizer.zero_grad()
            losses['confidence'].backward(retain_graph=True)
            self.value_optimizer.step()
        
        # Feature learning update
        if 'total' in losses:
            self.feature_optimizer.zero_grad()
            remaining_loss = losses['total'] - losses.get('policy', 0) - losses.get('confidence', 0)
            if remaining_loss.requires_grad:
                remaining_loss.backward()
                self.feature_optimizer.step()
        
        # Update learning rate schedulers
        if performance_metric is not None:
            self.performance_history.append(performance_metric)
            self.policy_scheduler.step(performance_metric)
            self.value_scheduler.step(losses.get('confidence', 0))
    
    def get_learning_rates(self) -> Dict[str, float]:
        """Get current learning rates for monitoring"""
        return {
            'policy': self.policy_optimizer.param_groups[0]['lr'],
            'value': self.value_optimizer.param_groups[0]['lr'],
            'feature': self.feature_optimizer.param_groups[0]['lr']
        }


class GPUMemoryManager:
    """Manages GPU memory allocation and cleanup for trading neural networks"""
    def __init__(self, max_memory_gb: float = 2.0):
        self.max_memory_bytes = max_memory_gb * 1024**3
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cleanup_threshold = 0.8  # Cleanup when 80% of max memory is used
        
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory statistics"""
        if not torch.cuda.is_available():
            return {'allocated': 0, 'reserved': 0, 'max_allocated': 0}
        
        stats = {
            'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**3  # GB
        }
        return stats
    
    def should_cleanup(self) -> bool:
        """Check if memory cleanup is needed"""
        if not torch.cuda.is_available():
            return False
        
        current_memory = torch.cuda.memory_allocated()
        return current_memory > (self.max_memory_bytes * self.cleanup_threshold)
    
    def cleanup_memory(self):
        """Perform GPU memory cleanup"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def optimize_memory_usage(self, model: nn.Module):
        """Optimize memory usage for a model"""
        if torch.cuda.is_available():
            # Enable memory efficient attention if available
            try:
                torch.backends.cuda.enable_flash_sdp(True)
            except:
                pass
            
            # Use gradient checkpointing for large models
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
    
    def memory_efficient_forward(self, model: nn.Module, *args, **kwargs):
        """Perform memory-efficient forward pass with automatic cleanup"""
        try:
            if self.should_cleanup():
                self.cleanup_memory()
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                return model(*args, **kwargs)
        
        except torch.cuda.OutOfMemoryError:
            # Emergency cleanup and retry
            self.cleanup_memory()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                return model(*args, **kwargs)
    
    def get_optimal_batch_size(self, model: nn.Module, input_shape: tuple) -> int:
        """Determine optimal batch size based on available memory"""
        if not torch.cuda.is_available():
            return 32  # CPU default
        
        # Start with a reasonable batch size and test
        batch_sizes = [64, 32, 16, 8, 4, 1]
        
        for batch_size in batch_sizes:
            try:
                # Create dummy input
                dummy_input = torch.randn(batch_size, *input_shape, device=self.device)
                
                # Test forward pass
                with torch.no_grad():
                    _ = model(dummy_input)
                
                # If successful, return this batch size
                del dummy_input
                self.cleanup_memory()
                return batch_size
            
            except torch.cuda.OutOfMemoryError:
                self.cleanup_memory()
                continue
        
        return 1  # Fallback to batch size 1


class MemoryEfficientSelfEvolvingNetwork(SelfEvolvingNetwork):
    """Memory-efficient version of SelfEvolvingNetwork with GPU management"""
    def __init__(self, input_size: int = 64, initial_sizes: List[int] = [128, 64],
                 evolution_frequency: int = 1000, max_memory_gb: float = 2.0):
        super().__init__(input_size, initial_sizes, evolution_frequency)
        
        self.memory_manager = GPUMemoryManager(max_memory_gb)
        self.memory_manager.optimize_memory_usage(self)
        
        # Use mixed precision training
        self.use_amp = torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def forward(self, x: torch.Tensor, timeframe_data: Dict = None) -> Dict[str, torch.Tensor]:
        """Memory-efficient forward pass"""
        return self.memory_manager.memory_efficient_forward(
            super().forward, x, timeframe_data
        )
    
    def train_step(self, loss_fn, optimizer, *args, **kwargs):
        """Memory-efficient training step with mixed precision"""
        if self.use_amp:
            with torch.cuda.amp.autocast():
                loss = loss_fn(*args, **kwargs)
            
            optimizer.zero_grad()
            self.scaler.scale(loss['total']).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss = loss_fn(*args, **kwargs)
            optimizer.zero_grad()
            loss['total'].backward()
            optimizer.step()
        
        # Periodic memory cleanup
        if self.forward_count % 100 == 0:
            self.memory_manager.cleanup_memory()
        
        return loss
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        return self.memory_manager.get_memory_stats()


def create_enhanced_network(input_size: int = 64, 
                          initial_sizes: List[int] = [128, 64],
                          enable_few_shot: bool = True,
                          memory_efficient: bool = True,
                          max_memory_gb: float = 2.0) -> SelfEvolvingNetwork:
    """Factory function for creating enhanced network with optional few-shot learning and memory management"""
    
    if memory_efficient:
        network = MemoryEfficientSelfEvolvingNetwork(
            input_size=input_size, 
            initial_sizes=initial_sizes,
            max_memory_gb=max_memory_gb
        )
    else:
        network = SelfEvolvingNetwork(input_size=input_size, initial_sizes=initial_sizes)
    
    if enable_few_shot:
        network.few_shot_learner = FewShotLearner(feature_dim=input_size)
    
    return network