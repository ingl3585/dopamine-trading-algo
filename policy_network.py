# policy_network.py - ENHANCED: Subsystem-aware neural network for strategic tool usage

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict

class SubsystemAwareAttention(nn.Module):
    """
    Attention mechanism that learns which subsystems to focus on in different market conditions
    """
    
    def __init__(self, subsystem_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        
        # Transform subsystem features
        self.subsystem_transform = nn.Linear(subsystem_dim, hidden_dim)
        
        # Query, Key, Value for attention
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        # Tool importance weights
        self.tool_importance = nn.Parameter(torch.randn(4))  # DNA, Micro, Temporal, Immune
        
        self.hidden_dim = hidden_dim
    
    def forward(self, subsystem_features, market_context):
        """
        subsystem_features: (batch, 16) - Features from DNA/Micro/Temporal/Immune
        market_context: (batch, hidden_dim) - Market state context
        """
        # Transform subsystem features
        subsystem_repr = self.subsystem_transform(subsystem_features)  # (batch, hidden_dim)
        
        # Create queries from market context
        Q = self.query(market_context).unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # Split subsystem representation into 4 tools (DNA, Micro, Temporal, Immune)
        tool_features = subsystem_repr.view(-1, 4, self.hidden_dim // 4)  # (batch, 4, 16)
        
        # Expand to full hidden dimension for each tool
        tool_features_expanded = torch.cat([
            tool_features, 
            tool_features, 
            tool_features, 
            tool_features
        ], dim=-1)  # (batch, 4, hidden_dim)
        
        # Keys and Values from tools
        K = self.key(tool_features_expanded)  # (batch, 4, hidden_dim)
        V = self.value(tool_features_expanded)  # (batch, 4, hidden_dim)
        
        # Attention weights
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.hidden_dim)
        
        # Apply tool importance bias
        tool_bias = self.tool_importance.unsqueeze(0).unsqueeze(0)  # (1, 1, 4)
        attention_scores = attention_scores + tool_bias
        
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch, 1, 4)
        
        # Weighted combination of tool features
        attended_features = torch.matmul(attention_weights, V).squeeze(1)  # (batch, hidden_dim)
        
        return attended_features, attention_weights.squeeze(1)

class MarketRegimeClassifier(nn.Module):
    """
    Classify current market regime to help with tool selection
    """
    
    def __init__(self, input_dim: int = 15, hidden_dim: int = 32):
        super().__init__()
        
        self.regime_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # trending, volatile, sideways, reversal
        )
    
    def forward(self, market_obs):
        regime_logits = self.regime_network(market_obs)
        regime_probs = F.softmax(regime_logits, dim=-1)
        return regime_probs

class EnhancedPolicyNetwork(nn.Module):
    """
    ENHANCED: Policy network that learns to strategically use your existing subsystems
    
    Key improvements:
    1. Subsystem-aware attention mechanism
    2. Market regime classification
    3. Tool trust learning
    4. Strategic risk management
    5. Contextual decision making
    """
    
    def __init__(self, market_obs_size: int = 15, subsystem_features_size: int = 16, 
                 hidden_size: int = 128):
        super().__init__()
        
        # Market observation encoder (LSTM for sequence processing)
        self.market_encoder = nn.LSTM(market_obs_size, hidden_size, batch_first=True)
        
        # Market regime classifier
        self.regime_classifier = MarketRegimeClassifier(market_obs_size, 32)
        
        # Subsystem attention mechanism
        self.subsystem_attention = SubsystemAwareAttention(subsystem_features_size, hidden_size)
        
        # Context fusion layer
        self.context_fusion = nn.Sequential(
            nn.Linear(hidden_size + 4, hidden_size),  # market + regime
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Decision layers that learn strategic combinations
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # market + attended subsystems
            nn.ReLU(),
            nn.Linear(hidden_size, 3)  # buy/sell/hold
        )
        
        # Risk management heads (AI learns when and how to use)
        self.use_stop_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.stop_distance_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.use_target_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.target_distance_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Exit timing (learns when to close positions)
        self.exit_confidence_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Overall confidence in decision
        self.overall_confidence_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Tool trust learning (learns which tools to trust when)
        self.tool_trust_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # DNA, Micro, Temporal, Immune
        )
        
        # Tool combination learning (learns synergies between tools)
        self.tool_combination_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 6)  # 6 possible pairs of 4 tools
        )

    def forward(self, market_obs, subsystem_features):
        """
        Forward pass with strategic subsystem usage
        
        Args:
            market_obs: (batch, seq_len, market_obs_size) - Market observations
            subsystem_features: (batch, subsystem_features_size) - Your subsystem outputs
        """
        batch_size = market_obs.shape[0]
        
        # Encode market observations
        market_encoded, _ = self.market_encoder(market_obs)
        market_context = market_encoded[:, -1]  # Last timestep (batch, hidden_size)
        
        # Classify market regime
        market_regime = self.regime_classifier(market_obs[:, -1])  # (batch, 4)
        
        # Fuse market context with regime
        fused_context = self.context_fusion(
            torch.cat([market_context, market_regime], dim=-1)
        )  # (batch, hidden_size)
        
        # Apply subsystem attention (learns which tools to focus on)
        attended_subsystems, attention_weights = self.subsystem_attention(
            subsystem_features, fused_context
        )  # (batch, hidden_size), (batch, 4)
        
        # Combine market context with attended subsystem insights
        combined_features = torch.cat([fused_context, attended_subsystems], dim=-1)  # (batch, hidden_size * 2)
        
        # Generate all decisions
        action_logits = self.action_head(combined_features)
        
        # Risk management decisions (AI learns when to use)
        use_stop = torch.sigmoid(self.use_stop_head(combined_features))
        stop_distance = torch.sigmoid(self.stop_distance_head(combined_features)) * 0.05  # 0-5%
        use_target = torch.sigmoid(self.use_target_head(combined_features))
        target_distance = torch.sigmoid(self.target_distance_head(combined_features)) * 0.10  # 0-10%
        
        # Exit and confidence
        exit_confidence = torch.sigmoid(self.exit_confidence_head(combined_features))
        overall_confidence = torch.sigmoid(self.overall_confidence_head(combined_features))
        
        # Tool trust and combinations
        tool_trust = F.softmax(self.tool_trust_head(combined_features), dim=-1)
        tool_combinations = torch.sigmoid(self.tool_combination_head(combined_features))
        
        return {
            'action_logits': action_logits,
            'use_stop': use_stop,
            'stop_distance': stop_distance,
            'use_target': use_target,
            'target_distance': target_distance,
            'exit_confidence': exit_confidence,
            'overall_confidence': overall_confidence,
            'tool_trust': tool_trust,  # Which individual tools to trust
            'tool_combinations': tool_combinations,  # Which tool combinations work
            'market_regime': market_regime,  # Market classification
            'attention_weights': attention_weights,  # Where AI is focusing
            'market_context': fused_context,  # For debugging/interpretation
            'attended_subsystems': attended_subsystems  # What the AI learned from your tools
        }

class ValueNetwork(nn.Module):
    """
    Value network for evaluating states (used with policy gradient methods)
    """
    
    def __init__(self, market_obs_size: int = 15, subsystem_features_size: int = 16, 
                 hidden_size: int = 128):
        super().__init__()
        
        # Similar structure to policy but outputs single value
        self.market_encoder = nn.LSTM(market_obs_size, hidden_size, batch_first=True)
        
        self.subsystem_processor = nn.Sequential(
            nn.Linear(subsystem_features_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size + hidden_size // 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, market_obs, subsystem_features):
        # Encode market
        market_encoded, _ = self.market_encoder(market_obs)
        market_context = market_encoded[:, -1]
        
        # Process subsystems
        subsystem_processed = self.subsystem_processor(subsystem_features)
        
        # Combine and predict value
        combined = torch.cat([market_context, subsystem_processed], dim=-1)
        value = self.value_head(combined)
        
        return value

class SubsystemInterpreter:
    """
    Helper class to interpret what the AI learned about your subsystems
    """
    
    @staticmethod
    def interpret_attention_weights(attention_weights: torch.Tensor) -> Dict[str, float]:
        """Convert attention weights to tool importance"""
        tools = ['DNA', 'Micro', 'Temporal', 'Immune']
        weights = attention_weights.cpu().numpy()
        
        if weights.ndim > 1:
            weights = weights.mean(axis=0)  # Average across batch
        
        return {tools[i]: float(weights[i]) for i in range(len(tools))}
    
    @staticmethod
    def interpret_market_regime(regime_probs: torch.Tensor) -> str:
        """Convert regime probabilities to regime name"""
        regimes = ['Trending', 'Volatile', 'Sideways', 'Reversal']
        probs = regime_probs.cpu().numpy()
        
        if probs.ndim > 1:
            probs = probs.mean(axis=0)  # Average across batch
        
        return regimes[np.argmax(probs)]
    
    @staticmethod
    def explain_decision(outputs: Dict[str, torch.Tensor]) -> str:
        """Generate human-readable explanation of AI's decision"""
        
        # Extract key information
        attention = SubsystemInterpreter.interpret_attention_weights(outputs['attention_weights'])
        regime = SubsystemInterpreter.interpret_market_regime(outputs['market_regime'])
        
        action_probs = F.softmax(outputs['action_logits'], dim=-1).cpu().numpy()
        if action_probs.ndim > 1:
            action_probs = action_probs.mean(axis=0)
        
        actions = ['Hold', 'Buy', 'Sell']
        predicted_action = actions[np.argmax(action_probs)]
        
        confidence = float(outputs['overall_confidence'].cpu().numpy().mean())
        
        # Primary tool
        primary_tool = max(attention.items(), key=lambda x: x[1])[0]
        
        # Risk management
        use_stop = float(outputs['use_stop'].cpu().numpy().mean()) > 0.5
        use_target = float(outputs['use_target'].cpu().numpy().mean()) > 0.5
        
        explanation = f"""
AI Decision Analysis:
- Action: {predicted_action} (confidence: {confidence:.2f})
- Market Regime: {regime}
- Primary Tool: {primary_tool} ({attention[primary_tool]:.2f} attention)
- Tool Focus: {', '.join([f'{k}({v:.2f})' for k, v in attention.items()])}
- Risk Management: Stop={'Yes' if use_stop else 'No'}, Target={'Yes' if use_target else 'No'}
"""
        
        return explanation