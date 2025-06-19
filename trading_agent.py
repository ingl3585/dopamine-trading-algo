# trading_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random # should not use random numbers

from collections import deque
from dataclasses import dataclass
from typing import List, Dict

from intelligence_engine import Features
from data_processor import MarketData

@dataclass
class Decision:
    action: str  # 'buy', 'sell', 'hold'
    confidence: float
    size: float
    stop_price: float = 0.0
    target_price: float = 0.0
    primary_tool: str = 'unknown'
    exploration: bool = False
    intelligence_data: Dict = None
    state_features: List = None


class TradingNetwork(nn.Module):
    def __init__(self, input_size=20, hidden_size=64):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
        )
        
        # Output heads
        self.action_head = nn.Linear(32, 3)  # Buy, Sell, Hold
        self.confidence_head = nn.Linear(32, 1)  # Confidence
        self.position_size_head = nn.Linear(32, 1)  # Position size
        self.risk_head = nn.Linear(32, 4)  # use_stop, stop_dist, use_target, target_dist
    
    def forward(self, x):
        features = self.network(x)
        
        action_logits = self.action_head(features)
        confidence = torch.sigmoid(self.confidence_head(features))
        position_size = torch.sigmoid(self.position_size_head(features)) * 3.0 + 0.5  # 0.5 to 3.5
        risk_outputs = torch.sigmoid(self.risk_head(features))
        
        return {
            'action_logits': action_logits,
            'confidence': confidence,
            'position_size': position_size,
            'use_stop': risk_outputs[:, 0:1],
            'stop_distance': risk_outputs[:, 1:2] * 0.05,  # Max 5%
            'use_target': risk_outputs[:, 2:3], 
            'target_distance': risk_outputs[:, 3:4] * 0.1,  # Max 10%
        }


class SimplifiedMetaLearner:
    """Simplified meta-learner for parameter adaptation"""
    
    def __init__(self):
        self.parameters = {
            'confidence_threshold': 0.6,
            'position_size_multiplier': 1.0,
            'stop_loss_pct': 0.015,
            'take_profit_pct': 0.03,
            'exploration_rate': 0.15,
        }
        
        self.outcomes = {name: deque(maxlen=50) for name in self.parameters}
        self.total_updates = 0
        self.successful_adaptations = 0
    
    def get_parameter(self, name):
        return self.parameters.get(name, 0.5)
    
    def update_parameter(self, name, outcome):
        if name not in self.parameters:
            return
            
        self.outcomes[name].append(outcome)
        self.total_updates += 1
        
        if len(self.outcomes[name]) < 10:
            return
        
        recent_outcomes = list(self.outcomes[name])[-20:]
        avg_outcome = np.mean(recent_outcomes)
        old_value = self.parameters[name]
        
        if avg_outcome > 0.05:  # Good performance
            if name == 'confidence_threshold':
                self.parameters[name] = max(0.3, old_value - 0.01)
            elif name == 'position_size_multiplier':
                self.parameters[name] = min(2.0, old_value + 0.05)
            elif name in ['stop_loss_pct', 'take_profit_pct']:
                self.parameters[name] = min(0.1, old_value + 0.001)
        
        elif avg_outcome < -0.05:  # Poor performance
            if name == 'confidence_threshold':
                self.parameters[name] = min(0.9, old_value + 0.01)
            elif name == 'position_size_multiplier':
                self.parameters[name] = max(0.5, old_value - 0.05)
            elif name in ['stop_loss_pct', 'take_profit_pct']:
                self.parameters[name] = max(0.005, old_value - 0.001)
        
        if abs(self.parameters[name] - old_value) > old_value * 0.05:
            self.successful_adaptations += 1
    
    def get_learning_efficiency(self):
        if self.total_updates == 0:
            return 0.0
        return self.successful_adaptations / self.total_updates


class TradingAgent:
    def __init__(self, intelligence, portfolio):
        self.intelligence = intelligence
        self.portfolio = portfolio
        
        # Meta-learner for parameter adaptation
        self.meta_learner = SimplifiedMetaLearner()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = TradingNetwork().to(self.device)
        self.target_network = TradingNetwork().to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        
        # Experience replay
        self.experience_buffer = deque(maxlen=10000)
        
        # Exploration
        self.epsilon = 0.3
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        
        # Statistics
        self.total_decisions = 0
        self.successful_trades = 0
        self.total_pnl = 0.0
        
    def decide(self, features: Features, market_data: MarketData) -> Decision:
        self.total_decisions += 1
        
        # Skip if insufficient confidence or too frequent trading
        if features.confidence < 0.3 or self._too_frequent():
            return Decision('hold', 0, 0)
        
        # Prepare comprehensive input features
        input_tensor = self._prepare_features(features, market_data)
        
        with torch.no_grad():
            outputs = self.network(input_tensor)
            
            # Get action probabilities
            action_probs = F.softmax(outputs['action_logits'], dim=-1).cpu().numpy()[0]
            confidence = float(outputs['confidence'].cpu().numpy()[0])
            
            # Exploration vs exploitation (should not use random numbers...)
            if random.random() < self.epsilon:
                action_idx = random.choice([0, 1, 2])  # Random action
                exploration = True
            else:
                action_idx = np.argmax(action_probs)  # Best action
                exploration = False
            
            # Check confidence threshold
            confidence_threshold = self.meta_learner.get_parameter('confidence_threshold')
            if confidence < confidence_threshold and not exploration:
                action_idx = 0  # Hold if not confident
            
            # Get other outputs
            position_size = float(outputs['position_size'].cpu().numpy()[0])
            use_stop = float(outputs['use_stop'].cpu().numpy()[0]) > 0.5
            stop_distance = float(outputs['stop_distance'].cpu().numpy()[0])
            use_target = float(outputs['use_target'].cpu().numpy()[0]) > 0.5
            target_distance = float(outputs['target_distance'].cpu().numpy()[0])
            
            # Apply meta-learner multipliers
            position_size *= self.meta_learner.get_parameter('position_size_multiplier')
            position_size = max(0.5, min(3.0, position_size))  # Clamp
        
        # Decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Convert to decision
        actions = ['hold', 'buy', 'sell']
        action = actions[action_idx]
        
        if action == 'hold' or confidence < 0.4:
            return Decision('hold', confidence, 0)
        
        # Calculate stop and target prices
        stop_price = 0
        target_price = 0
        
        if use_stop:
            if action == 'buy':
                stop_price = market_data.price * (1 - stop_distance)
            else:  # sell
                stop_price = market_data.price * (1 + stop_distance)
        
        if use_target:
            if action == 'buy':
                target_price = market_data.price * (1 + target_distance)
            else:  # sell
                target_price = market_data.price * (1 - target_distance)
        
        # Determine primary tool from subsystem signals
        primary_tool = self._get_primary_tool(features)
        
        # Prepare intelligence data for learning
        intelligence_data = {
            'subsystem_signals': {
                'dna': features.dna_signal,
                'micro': features.micro_signal,
                'temporal': features.temporal_signal,
                'immune': features.immune_signal
            },
            'overall_signal': features.overall_signal,
            'current_patterns': {}  # Will be filled by intelligence engine
        }
        
        # Store state features for learning
        state_features = input_tensor.squeeze().cpu().numpy().tolist()
        
        return Decision(
            action=action,
            confidence=confidence,
            size=position_size,
            stop_price=stop_price,
            target_price=target_price,
            primary_tool=primary_tool,
            exploration=exploration,
            intelligence_data=intelligence_data,
            state_features=state_features
        )
    
    def learn_from_trade(self, trade):
        if not hasattr(trade, 'features') or not hasattr(trade, 'decision_data'):
            return
            
        # Update statistics
        if trade.pnl > 0:
            self.successful_trades += 1
        self.total_pnl += trade.pnl
        
        # Normalize outcome for learning
        normalized_outcome = np.tanh(trade.pnl / 50.0)  # Normalize to [-1, 1]
        
        # Store experience for neural network training
        if hasattr(trade, 'state_features'):
            experience = {
                'state_features': trade.state_features,
                'action': ['hold', 'buy', 'sell'].index(trade.action),
                'reward': normalized_outcome,
                'done': True,
                'confidence': getattr(trade, 'confidence', 0.5),
                'position_size': getattr(trade, 'size', 1.0)
            }
            
            self.experience_buffer.append(experience)
        
        # Update meta-learner
        self.meta_learner.update_parameter('confidence_threshold', normalized_outcome)
        self.meta_learner.update_parameter('position_size_multiplier', normalized_outcome)
        
        if hasattr(trade, 'stop_used') and trade.stop_used:
            self.meta_learner.update_parameter('stop_loss_pct', normalized_outcome)
        if hasattr(trade, 'target_used') and trade.target_used:
            self.meta_learner.update_parameter('take_profit_pct', normalized_outcome)
        
        # Train network if enough experience
        if len(self.experience_buffer) >= 64:
            self._train_network()
    
    def _prepare_features(self, features: Features, market_data: MarketData) -> torch.Tensor:
        """Prepare comprehensive input features for neural network"""
        input_data = []
        
        # Market features (price changes)
        prices = market_data.prices_1m[-10:] if len(market_data.prices_1m) >= 10 else [market_data.price]
        if len(prices) >= 2:
            price_changes = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            input_data.extend(price_changes[-5:])  # Last 5 price changes
        else:
            input_data.extend([0.0] * 5)
        
        # Pad to exactly 5 features
        while len(input_data) < 5:
            input_data.append(0.0)
        input_data = input_data[:5]
        
        # Intelligence subsystem signals
        input_data.extend([
            features.dna_signal,
            features.micro_signal,
            features.temporal_signal,
            features.immune_signal
        ])
        
        # Overall intelligence signal and confidence
        input_data.extend([
            features.overall_signal,
            features.confidence
        ])
        
        # Basic features
        input_data.extend([
            features.price_momentum,
            features.volume_momentum,
            features.price_position,
            features.volatility,
            features.time_of_day,
            features.pattern_score
        ])
        
        # Account features
        input_data.extend([
            min(1.0, market_data.account_balance / 50000),  # Normalized balance
            min(1.0, market_data.buying_power / 50000),  # Normalized buying power
            np.tanh(market_data.daily_pnl / 1000),  # Normalized daily P&L
        ])
        
        # Meta-learning features
        input_data.extend([
            self.epsilon,  # Current exploration rate
            self.meta_learner.get_learning_efficiency(),  # Learning efficiency
        ])
        
        # Pad or truncate to exactly 20 features
        while len(input_data) < 20:
            input_data.append(0.0)
        input_data = input_data[:20]
        
        return torch.tensor(input_data, dtype=torch.float32, device=self.device).unsqueeze(0)
    
    def _get_primary_tool(self, features: Features) -> str:
        """Determine primary tool from subsystem signals"""
        signals = {
            'dna': abs(features.dna_signal),
            'micro': abs(features.micro_signal), 
            'temporal': abs(features.temporal_signal),
            'immune': abs(features.immune_signal)
        }
        
        if not any(signals.values()):
            return 'basic'
            
        return max(signals.items(), key=lambda x: x[1])[0]
    
    def _too_frequent(self) -> bool:
        recent_trades = self.portfolio.get_recent_trade_count(minutes=15)
        return recent_trades >= 3  # Max 3 trades per 15 minutes
    
    def _train_network(self):
        """Train the neural network on experiences"""
        if len(self.experience_buffer) < 32:
            return
        
        # Sample batch
        batch = random.sample(list(self.experience_buffer), 32)
        
        # Prepare tensors
        states = torch.tensor([exp['state_features'] for exp in batch], 
                            dtype=torch.float32, device=self.device)
        actions = torch.tensor([exp['action'] for exp in batch], 
                             dtype=torch.long, device=self.device)
        rewards = torch.tensor([exp['reward'] for exp in batch], 
                             dtype=torch.float32, device=self.device)
        
        # Forward pass
        outputs = self.network(states)
        action_logits = outputs['action_logits']
        
        # Calculate loss (simple policy gradient)
        action_probs = F.log_softmax(action_logits, dim=-1)
        selected_action_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Policy loss (maximize reward-weighted log probabilities)
        policy_loss = -(selected_action_probs * rewards).mean()
        
        # Confidence loss (predict absolute reward as confidence target)
        confidence_target = torch.abs(rewards).unsqueeze(1)
        confidence_loss = F.mse_loss(outputs['confidence'], confidence_target)
        
        # Total loss
        total_loss = policy_loss + 0.1 * confidence_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network occasionally
        if self.total_decisions % 100 == 0:
            self.target_network.load_state_dict(self.network.state_dict())
    
    def save_model(self, filepath: str):
        torch.save({
            'network_state': self.network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_decisions': self.total_decisions,
            'successful_trades': self.successful_trades,
            'total_pnl': self.total_pnl,
            'meta_learner_params': self.meta_learner.parameters
        }, filepath)
    
    def load_model(self, filepath: str):
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.network.load_state_dict(checkpoint['network_state'])
            self.target_network.load_state_dict(checkpoint['target_network_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.epsilon = checkpoint.get('epsilon', 0.3)
            self.total_decisions = checkpoint.get('total_decisions', 0)
            self.successful_trades = checkpoint.get('successful_trades', 0)
            self.total_pnl = checkpoint.get('total_pnl', 0.0)
            
            # Load meta-learner parameters
            if 'meta_learner_params' in checkpoint:
                self.meta_learner.parameters.update(checkpoint['meta_learner_params'])
        except FileNotFoundError:
            pass  # Start fresh
    
    def get_stats(self) -> dict:
        return {
            'total_decisions': self.total_decisions,
            'successful_trades': self.successful_trades,
            'success_rate': self.successful_trades / max(1, self.total_decisions),
            'total_pnl': self.total_pnl,
            'epsilon': self.epsilon,
            'experience_size': len(self.experience_buffer),
            'meta_learner_efficiency': self.meta_learner.get_learning_efficiency(),
            'confidence_threshold': self.meta_learner.get_parameter('confidence_threshold'),
            'position_multiplier': self.meta_learner.get_parameter('position_size_multiplier')
        }