# adaptive_network.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any
import time
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DynamicLayer(nn.Module):
    def __init__(self, input_size: int, output_size: int, dropout_rate: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.linear(x)))


class AdaptiveTradingNetwork(nn.Module):
    def __init__(self, input_size: int = 64, hidden_sizes: List[int] = [128, 64]):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        
        self._build_network()
        
        # Use consistent float32 precision
        self.float()
    
    def _build_network(self):
        layers = []
        prev_size = self.input_size
        
        for hidden_size in self.hidden_sizes:
            layers.append(DynamicLayer(prev_size, hidden_size))
            prev_size = hidden_size
        
        self.backbone = nn.Sequential(*layers)
        
        # Output heads
        self.action_head = nn.Linear(prev_size, 3)
        self.confidence_head = nn.Linear(prev_size, 1)
        self.position_size_head = nn.Linear(prev_size, 1)
        self.risk_head = nn.Linear(prev_size, 4)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        
        action_logits = self.action_head(features)
        confidence = torch.sigmoid(self.confidence_head(features))
        position_size = torch.sigmoid(self.position_size_head(features)) * 3.0 + 0.5
        risk_outputs = torch.sigmoid(self.risk_head(features))
        
        return {
            'action_logits': action_logits,
            'confidence': confidence,
            'position_size': position_size,
            'use_stop': risk_outputs[:, 0:1],
            'stop_distance': risk_outputs[:, 1:2] * 0.05,
            'use_target': risk_outputs[:, 2:3],
            'target_distance': risk_outputs[:, 3:4] * 0.1,
        }
    
    def evolve_architecture(self, new_hidden_sizes: List[int]):
        old_state = self.state_dict()
        self.hidden_sizes = new_hidden_sizes
        self._build_network()
        
        # Transfer compatible weights
        current_state = self.state_dict()
        for name, param in old_state.items():
            if name in current_state and param.shape == current_state[name].shape:
                current_state[name] = param
        
        self.load_state_dict(current_state)


class FeatureLearner(nn.Module):
    def __init__(self, raw_feature_dim: int = 64, learned_feature_dim: int = 64):
        super().__init__()
        self.raw_feature_dim = raw_feature_dim
        self.learned_feature_dim = learned_feature_dim
        
        # Feature selection network
        self.feature_selector = nn.Sequential(
            nn.Linear(raw_feature_dim, raw_feature_dim // 2),
            nn.ReLU(),
            nn.Linear(raw_feature_dim // 2, raw_feature_dim),
            nn.Sigmoid()
        )
        
        # Feature combination network
        self.feature_combiner = nn.Sequential(
            nn.Linear(raw_feature_dim, learned_feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(learned_feature_dim * 2, learned_feature_dim)
        )
        
        # Use consistent float32 precision
        self.float()
        
        # Ensure all parameters are float32
        for param in self.parameters():
            param.data = param.data.float()
        
        # Feature importance tracking
        self.feature_importance = nn.Parameter(torch.ones(raw_feature_dim, dtype=torch.float32))
        self.usage_counts = torch.zeros(raw_feature_dim, dtype=torch.float32)
    
    def forward(self, raw_features: torch.Tensor) -> torch.Tensor:
        # Apply learned feature selection
        selection_weights = self.feature_selector(raw_features)
        selected_features = raw_features * selection_weights
        
        # Combine features in learned ways
        learned_features = self.feature_combiner(selected_features)
        
        # Update usage statistics
        with torch.no_grad():
            self.usage_counts += selection_weights.sum(dim=0)
        
        return learned_features
    
    def get_feature_importance(self) -> torch.Tensor:
        return self.feature_importance / (self.usage_counts + 1e-8)
    
    def prune_unused_features(self, threshold: float = 0.01):
        importance = self.get_feature_importance()
        mask = importance > threshold
        
        # TODO
        # This would require more complex implementation to actually modify network structure
        # For now, just zero out unimportant features
        with torch.no_grad():
            self.feature_importance.data *= mask.float()


class StateEncoder:
    def __init__(self):
        self.price_window = 10
        self.volume_window = 10
        
    def encode_market_data(self, market_data) -> torch.Tensor:
        features = []
        
        # Price movement features
        if market_data.prices_1m and len(market_data.prices_1m) >= self.price_window:
            prices = np.array(market_data.prices_1m[-self.price_window:])
            
            # Returns
            returns = np.diff(prices) / prices[:-1]
            features.extend(returns[-5:].tolist())
            
            # Moving averages
            short_ma = np.mean(prices[-3:])
            long_ma = np.mean(prices)
            ma_ratio = (short_ma - long_ma) / long_ma if long_ma > 0 else 0
            features.append(ma_ratio)
            
            # Volatility
            volatility = np.std(returns) if len(returns) > 1 else 0
            features.append(volatility)
            
            # Price position in range
            price_range = np.max(prices) - np.min(prices)
            if price_range > 0:
                position = (prices[-1] - np.min(prices)) / price_range
            else:
                position = 0.5
            features.append(position)
        else:
            features.extend([0.0] * 8)
        
        # Volume features
        if market_data.volumes_1m and len(market_data.volumes_1m) >= self.volume_window:
            volumes = np.array(market_data.volumes_1m[-self.volume_window:])
            
            # Volume momentum
            recent_vol = np.mean(volumes[-3:])
            avg_vol = np.mean(volumes)
            vol_momentum = (recent_vol - avg_vol) / avg_vol if avg_vol > 0 else 0
            features.append(vol_momentum)
            
            # Volume trend
            vol_trend = (volumes[-1] - volumes[0]) / volumes[0] if volumes[0] > 0 else 0
            features.append(vol_trend)
        else:
            features.extend([0.0] * 2)
        
        # Multi-timeframe features
        for timeframe_prices in [market_data.prices_5m, market_data.prices_15m]:
            if timeframe_prices and len(timeframe_prices) >= 3:
                tf_prices = np.array(timeframe_prices[-3:])
                tf_momentum = (tf_prices[-1] - tf_prices[0]) / tf_prices[0]
                features.append(tf_momentum)
            else:
                features.append(0.0)
        
        # Enhanced account state features
        features.extend([
            # Normalized account balance (using 50k as reference)
            min(1.0, market_data.account_balance / 50000),
            # Normalized buying power
            min(1.0, market_data.buying_power / 50000),
            # Daily PnL as percentage of account (already computed by TCP bridge)
            np.tanh(market_data.daily_pnl_pct * 100),  # Scale to reasonable range
            # Margin utilization (0-1 ratio)
            min(1.0, market_data.margin_utilization),
            # Buying power ratio (how much buying power vs net liquidation)
            min(2.0, market_data.buying_power_ratio),  # Cap at 2x leverage
            # Open positions normalized
            min(1.0, market_data.open_positions / 5.0),  # Assume max 5 positions
            # Available margin normalized
            min(1.0, market_data.available_margin / 25000)
        ])
        
        # Convert .NET Ticks to Unix timestamp
        timestamp = market_data.timestamp
        
        # Safe timestamp conversion with better error handling
        try:
            # .NET Ticks: 100-nanosecond intervals since January 1, 0001
            # Unix timestamp: seconds since January 1, 1970
            if timestamp > 1e15:  # Definitely .NET ticks (very large number)
                unix_timestamp = (timestamp - 621355968000000000) / 10000000
            else:
                unix_timestamp = timestamp  # Already Unix timestamp
            
            # Validate timestamp (reasonable range: 1970-2030)
            if unix_timestamp < 0 or unix_timestamp > 1893456000:
                unix_timestamp = time.time()  # Use current time as fallback
            
            # Additional validation for datetime conversion
            if unix_timestamp > 2147483647:  # Max 32-bit timestamp
                unix_timestamp = time.time()
                
            current_time = datetime.fromtimestamp(unix_timestamp)
            hour_norm = current_time.hour / 24.0
            minute_norm = current_time.minute / 60.0
            weekday_norm = current_time.weekday() / 6.0
            
        except (ValueError, OSError, OverflowError) as e:
            # Fallback to current time if conversion fails
            logger.warning(f"Timestamp conversion failed: {e}, using current time")
            current_time = datetime.now()
            hour_norm = current_time.hour / 24.0
            minute_norm = current_time.minute / 60.0
            weekday_norm = current_time.weekday() / 6.0
        
        features.extend([hour_norm, minute_norm, weekday_norm])
        
        # Pad to exactly 32 features for market data component
        while len(features) < 32:
            features.append(0.0)
        
        return torch.tensor(features[:32], dtype=torch.float32)
    
    def encode_intelligence_features(self, intelligence_features) -> torch.Tensor:
        return torch.tensor([
            intelligence_features.dna_signal,
            intelligence_features.micro_signal,
            intelligence_features.temporal_signal,
            intelligence_features.immune_signal,
            intelligence_features.overall_signal,
            intelligence_features.confidence,
            intelligence_features.price_momentum,
            intelligence_features.volume_momentum,
            intelligence_features.price_position,
            intelligence_features.volatility,
            intelligence_features.time_of_day,
            intelligence_features.pattern_score
        ], dtype=torch.float32)
    
    def create_full_state(self, market_data, intelligence_features, meta_context) -> torch.Tensor:
        market_state = self.encode_market_data(market_data)
        intelligence_state = self.encode_intelligence_features(intelligence_features)
        
        # Enhanced meta-learning context with account-aware features
        meta_state = torch.tensor([
            meta_context.get('recent_performance', 0.0),
            meta_context.get('consecutive_losses', 0.0) / 10.0,
            meta_context.get('trades_today', 0.0) / 20.0,
            meta_context.get('learning_efficiency', 0.0),
            meta_context.get('architecture_generation', 0.0) / 10.0,
            # Additional account context
            min(1.0, meta_context.get('position_count', 0.0) / 5.0),
            np.tanh(meta_context.get('time_since_last_trade', 0.0))
        ], dtype=torch.float32)
        
        # Combine all features with consistent dtype
        full_state = torch.cat([
            market_state,        # 32 features
            intelligence_state,  # 12 features  
            meta_state          # 7 features
        ]).to(dtype=torch.float32)
        
        # Pad or truncate to exactly 64 features for consistent input
        if len(full_state) < 64:
            padding = torch.zeros(64 - len(full_state), dtype=torch.float32)
            full_state = torch.cat([full_state, padding])
        else:
            full_state = full_state[:64]
        
        return full_state