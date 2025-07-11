"""
Specialized Networks for Different Trading Prediction Tasks

This module implements separate neural networks optimized for specific trading tasks:
- Price direction prediction
- Volatility estimation
- Risk assessment
- Market regime classification
- Position sizing optimization

Features:
- Task-specific architectures
- Shared feature extraction backbone
- Ensemble coordination
- Task-specific loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of trading prediction tasks"""
    PRICE_DIRECTION = "price_direction"
    VOLATILITY_ESTIMATION = "volatility_estimation"
    RISK_ASSESSMENT = "risk_assessment"
    MARKET_REGIME = "market_regime"
    POSITION_SIZING = "position_sizing"
    ENTRY_TIMING = "entry_timing"
    EXIT_TIMING = "exit_timing"


@dataclass
class TaskConfig:
    """Configuration for a specific prediction task"""
    task_type: TaskType
    output_dim: int
    hidden_sizes: List[int]
    activation: str = "relu"
    dropout_rate: float = 0.1
    use_batch_norm: bool = True
    loss_weight: float = 1.0


class SharedFeatureBackbone(nn.Module):
    """Shared feature extraction backbone for all tasks"""
    
    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
        
        # Attention mechanism for feature importance
        self.feature_attention = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Linear(output_dim // 2, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract shared features with attention"""
        features = self.feature_extractor(x)
        attention_weights = self.feature_attention(features)
        attended_features = features * attention_weights
        return attended_features


class PriceDirectionNetwork(nn.Module):
    """Network specialized for price direction prediction"""
    
    def __init__(self, input_dim: int = 64, hidden_sizes: List[int] = [32, 16]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.15)
            ])
            prev_dim = hidden_size
        
        # Output layer for 3-class classification (UP, DOWN, SIDEWAYS)
        layers.append(nn.Linear(prev_dim, 3))
        
        self.network = nn.Sequential(*layers)
        
        # Price momentum features
        self.momentum_processor = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict price direction with confidence"""
        # Process momentum features
        momentum_features = self.momentum_processor(features)
        
        # Main prediction
        logits = self.network(features)
        probabilities = F.softmax(logits, dim=-1)
        
        # Confidence based on prediction certainty
        confidence = torch.max(probabilities, dim=-1)[0]
        
        return {
            'direction_logits': logits,
            'direction_probs': probabilities,
            'confidence': confidence.unsqueeze(-1),
            'momentum_features': momentum_features
        }


class VolatilityEstimationNetwork(nn.Module):
    """Network specialized for volatility estimation"""
    
    def __init__(self, input_dim: int = 64, hidden_sizes: List[int] = [32, 24, 16]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ELU(),  # ELU for better gradient flow
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_size
        
        self.feature_network = nn.Sequential(*layers)
        
        # Multiple volatility outputs
        self.realized_vol = nn.Sequential(
            nn.Linear(prev_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Softplus()  # Ensure positive output
        )
        
        self.implied_vol = nn.Sequential(
            nn.Linear(prev_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Softplus()
        )
        
        # Volatility regime classifier
        self.vol_regime = nn.Sequential(
            nn.Linear(prev_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 3)  # LOW, MEDIUM, HIGH volatility
        )
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Estimate various volatility measures"""
        feature_repr = self.feature_network(features)
        
        realized_volatility = self.realized_vol(feature_repr)
        implied_volatility = self.implied_vol(feature_repr)
        vol_regime_logits = self.vol_regime(feature_repr)
        
        return {
            'realized_volatility': realized_volatility,
            'implied_volatility': implied_volatility,
            'volatility_regime_logits': vol_regime_logits,
            'volatility_regime_probs': F.softmax(vol_regime_logits, dim=-1)
        }


class RiskAssessmentNetwork(nn.Module):
    """Network specialized for risk assessment"""
    
    def __init__(self, input_dim: int = 64, hidden_sizes: List[int] = [48, 32, 16]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),  # GELU for smoother gradients
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_size
        
        self.feature_network = nn.Sequential(*layers)
        
        # Risk metrics outputs
        self.value_at_risk = nn.Sequential(
            nn.Linear(prev_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()  # VaR as percentage
        )
        
        self.expected_shortfall = nn.Sequential(
            nn.Linear(prev_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
        self.sharpe_prediction = nn.Sequential(
            nn.Linear(prev_dim, 8),
            nn.Tanh(),  # Sharpe can be negative
            nn.Linear(8, 1)
        )
        
        # Risk category classifier
        self.risk_category = nn.Sequential(
            nn.Linear(prev_dim, 12),
            nn.ReLU(),
            nn.Linear(12, 4)  # VERY_LOW, LOW, MEDIUM, HIGH risk
        )
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Assess various risk metrics"""
        feature_repr = self.feature_network(features)
        
        var = self.value_at_risk(feature_repr)
        es = self.expected_shortfall(feature_repr)
        sharpe = self.sharpe_prediction(feature_repr)
        risk_cat_logits = self.risk_category(feature_repr)
        
        return {
            'value_at_risk': var,
            'expected_shortfall': es,
            'sharpe_prediction': sharpe,
            'risk_category_logits': risk_cat_logits,
            'risk_category_probs': F.softmax(risk_cat_logits, dim=-1)
        }


class MarketRegimeNetwork(nn.Module):
    """Network for market regime classification"""
    
    def __init__(self, input_dim: int = 64, hidden_sizes: List[int] = [40, 24]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_size
        
        self.feature_network = nn.Sequential(*layers)
        
        # Regime classification (BULL, BEAR, SIDEWAYS, VOLATILE)
        self.regime_classifier = nn.Sequential(
            nn.Linear(prev_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )
        
        # Regime strength estimation
        self.regime_strength = nn.Sequential(
            nn.Linear(prev_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
        # Transition probability
        self.transition_prob = nn.Sequential(
            nn.Linear(prev_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4),  # Prob of transitioning to each regime
            nn.Softmax(dim=-1)
        )
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Classify market regime and estimate transitions"""
        feature_repr = self.feature_network(features)
        
        regime_logits = self.regime_classifier(feature_repr)
        regime_probs = F.softmax(regime_logits, dim=-1)
        strength = self.regime_strength(feature_repr)
        transition_probs = self.transition_prob(feature_repr)
        
        return {
            'regime_logits': regime_logits,
            'regime_probs': regime_probs,
            'regime_strength': strength,
            'transition_probs': transition_probs
        }


class PositionSizingNetwork(nn.Module):
    """Network for optimal position sizing"""
    
    def __init__(self, input_dim: int = 64, hidden_sizes: List[int] = [32, 24, 16]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_size
        
        self.feature_network = nn.Sequential(*layers)
        
        # Position sizing outputs
        self.base_size = nn.Sequential(
            nn.Linear(prev_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()  # Base position as fraction of capital
        )
        
        self.risk_multiplier = nn.Sequential(
            nn.Linear(prev_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Softplus()  # Risk-based sizing multiplier
        )
        
        # Kelly criterion estimation
        self.kelly_fraction = nn.Sequential(
            nn.Linear(prev_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Determine optimal position sizing"""
        feature_repr = self.feature_network(features)
        
        base = self.base_size(feature_repr)
        risk_mult = self.risk_multiplier(feature_repr)
        kelly = self.kelly_fraction(feature_repr)
        
        # Combined position size (0.5 to 3.0 range)
        position_size = base * risk_mult * 2.0 + 0.5
        position_size = torch.clamp(position_size, 0.5, 3.0)
        
        return {
            'base_size': base,
            'risk_multiplier': risk_mult,
            'kelly_fraction': kelly,
            'position_size': position_size
        }


class SpecializedNetworkEnsemble(nn.Module):
    """Ensemble of specialized networks for different trading tasks"""
    
    def __init__(self, input_dim: int = 64, shared_dim: int = 64):
        super().__init__()
        
        # Shared feature backbone
        self.shared_backbone = SharedFeatureBackbone(input_dim, 128, shared_dim)
        
        # Specialized networks
        self.price_direction_net = PriceDirectionNetwork(shared_dim)
        self.volatility_net = VolatilityEstimationNetwork(shared_dim)
        self.risk_assessment_net = RiskAssessmentNetwork(shared_dim)
        self.market_regime_net = MarketRegimeNetwork(shared_dim)
        self.position_sizing_net = PositionSizingNetwork(shared_dim)
        
        # Task coordination weights
        self.task_weights = nn.Parameter(torch.ones(5))
        
        # Cross-task attention for coordination
        self.task_coordination = nn.MultiheadAttention(
            embed_dim=shared_dim, num_heads=4, batch_first=True
        )
    
    def forward(self, x: torch.Tensor, tasks: Optional[List[TaskType]] = None) -> Dict[str, Any]:
        """Forward pass through specialized networks"""
        # Extract shared features
        shared_features = self.shared_backbone(x)
        
        # Prepare for cross-task attention
        shared_features_seq = shared_features.unsqueeze(1)  # Add sequence dimension
        
        # Apply cross-task attention for feature coordination
        coordinated_features, _ = self.task_coordination(
            shared_features_seq, shared_features_seq, shared_features_seq
        )
        coordinated_features = coordinated_features.squeeze(1)
        
        results = {'shared_features': shared_features}
        
        # Run all networks or only specified tasks
        if tasks is None or TaskType.PRICE_DIRECTION in tasks:
            results['price_direction'] = self.price_direction_net(coordinated_features)
        
        if tasks is None or TaskType.VOLATILITY_ESTIMATION in tasks:
            results['volatility'] = self.volatility_net(coordinated_features)
        
        if tasks is None or TaskType.RISK_ASSESSMENT in tasks:
            results['risk_assessment'] = self.risk_assessment_net(coordinated_features)
        
        if tasks is None or TaskType.MARKET_REGIME in tasks:
            results['market_regime'] = self.market_regime_net(coordinated_features)
        
        if tasks is None or TaskType.POSITION_SIZING in tasks:
            results['position_sizing'] = self.position_sizing_net(coordinated_features)
        
        # Add task coordination weights
        results['task_weights'] = F.softmax(self.task_weights, dim=0)
        
        return results


class MultiTaskLoss(nn.Module):
    """Multi-task loss function for specialized networks"""
    
    def __init__(self, task_weights: Optional[Dict[TaskType, float]] = None):
        super().__init__()
        
        self.task_weights = task_weights or {
            TaskType.PRICE_DIRECTION: 1.0,
            TaskType.VOLATILITY_ESTIMATION: 0.8,
            TaskType.RISK_ASSESSMENT: 0.6,
            TaskType.MARKET_REGIME: 0.4,
            TaskType.POSITION_SIZING: 0.7
        }
        
        # Adaptive task weighting
        self.adaptive_weights = nn.Parameter(torch.ones(len(self.task_weights)))
    
    def forward(self, predictions: Dict[str, Any], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss"""
        losses = {}
        total_loss = torch.tensor(0.0, device=next(iter(targets.values())).device)
        
        # Price direction loss
        if 'price_direction' in predictions and 'price_direction' in targets:
            pred = predictions['price_direction']['direction_logits']
            target = targets['price_direction']
            direction_loss = F.cross_entropy(pred, target)
            losses['price_direction'] = direction_loss
            total_loss += self.task_weights[TaskType.PRICE_DIRECTION] * direction_loss
        
        # Volatility estimation loss
        if 'volatility' in predictions and 'volatility' in targets:
            pred = predictions['volatility']['realized_volatility']
            target = targets['volatility']
            volatility_loss = F.mse_loss(pred, target)
            losses['volatility'] = volatility_loss
            total_loss += self.task_weights[TaskType.VOLATILITY_ESTIMATION] * volatility_loss
        
        # Risk assessment loss
        if 'risk_assessment' in predictions and 'risk_assessment' in targets:
            pred = predictions['risk_assessment']['value_at_risk']
            target = targets['risk_assessment']
            risk_loss = F.mse_loss(pred, target)
            losses['risk_assessment'] = risk_loss
            total_loss += self.task_weights[TaskType.RISK_ASSESSMENT] * risk_loss
        
        # Market regime loss
        if 'market_regime' in predictions and 'market_regime' in targets:
            pred = predictions['market_regime']['regime_logits']
            target = targets['market_regime']
            regime_loss = F.cross_entropy(pred, target)
            losses['market_regime'] = regime_loss
            total_loss += self.task_weights[TaskType.MARKET_REGIME] * regime_loss
        
        # Position sizing loss
        if 'position_sizing' in predictions and 'position_sizing' in targets:
            pred = predictions['position_sizing']['position_size']
            target = targets['position_sizing']
            sizing_loss = F.mse_loss(pred, target)
            losses['position_sizing'] = sizing_loss
            total_loss += self.task_weights[TaskType.POSITION_SIZING] * sizing_loss
        
        # Task coordination regularization
        if 'task_weights' in predictions:
            task_weights_norm = torch.norm(predictions['task_weights'], p=2)
            regularization = 0.01 * task_weights_norm
            total_loss += regularization
            losses['regularization'] = regularization
        
        losses['total'] = total_loss
        return losses


def create_specialized_ensemble(input_dim: int = 64, shared_dim: int = 64) -> SpecializedNetworkEnsemble:
    """Factory function to create specialized network ensemble"""
    return SpecializedNetworkEnsemble(input_dim, shared_dim)


def create_multi_task_loss(task_weights: Optional[Dict[TaskType, float]] = None) -> MultiTaskLoss:
    """Factory function to create multi-task loss"""
    return MultiTaskLoss(task_weights)