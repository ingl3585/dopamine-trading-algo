# rl_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
import random
from collections import deque
from typing import Dict

log = logging.getLogger(__name__)

class SimpleNeuralNetwork(nn.Module):
    """Simple neural network for trading decisions"""
    
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

class SimpleRLAgent:
    """
    Simplified RL agent that learns trading decisions
    Much simpler than the original but still learns from outcomes
    """
    
    def __init__(self, meta_learner, config):
        self.meta_learner = meta_learner
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural network
        self.network = SimpleNeuralNetwork().to(self.device)
        self.target_network = SimpleNeuralNetwork().to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        
        # Optimizer
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
        
        log.info("Simple RL agent initialized")
    
    def select_action(self, market_data: Dict, intelligence_result: Dict, 
                     account_data: Dict, current_price: float) -> Dict:
        """Select trading action based on current state"""
        
        # Prepare input features
        features = self._prepare_features(market_data, intelligence_result, account_data)
        
        self.total_decisions += 1
        
        with torch.no_grad():
            features_tensor = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
            outputs = self.network(features_tensor)
            
            # Get action probabilities
            action_probs = F.softmax(outputs['action_logits'], dim=-1).cpu().numpy()[0]
            confidence = float(outputs['confidence'].cpu().numpy()[0])
            
            # Exploration vs exploitation
            if random.random() < self.epsilon:
                action = random.choice([0, 1, 2])  # Random action
                exploration = True
            else:
                action = np.argmax(action_probs)  # Best action
                exploration = False
            
            # Check confidence threshold
            confidence_threshold = self.meta_learner.get_parameter('confidence_threshold')
            if confidence < confidence_threshold and not exploration:
                action = 0  # Hold if not confident
            
            # Get other outputs
            position_size = float(outputs['position_size'].cpu().numpy()[0])
            use_stop = float(outputs['use_stop'].cpu().numpy()[0]) > 0.5
            stop_distance = float(outputs['stop_distance'].cpu().numpy()[0])
            use_target = float(outputs['use_target'].cpu().numpy()[0]) > 0.5
            target_distance = float(outputs['target_distance'].cpu().numpy()[0])
            
            # Apply meta-learner multipliers
            position_size *= self.meta_learner.get_parameter('position_size_multiplier')
            position_size = max(0.5, min(3.0, position_size))  # Clamp
            
            # Calculate actual prices
            stop_price = 0.0
            target_price = 0.0
            
            if use_stop and action != 0:
                if action == 1:  # Long
                    stop_price = current_price * (1 - stop_distance)
                else:  # Short
                    stop_price = current_price * (1 + stop_distance)
            
            if use_target and action != 0:
                if action == 1:  # Long
                    target_price = current_price * (1 + target_distance)
                else:  # Short
                    target_price = current_price * (1 - target_distance)
            
            # Determine primary tool (simplified)
            subsystem_signals = intelligence_result.get('subsystem_signals', {})
            primary_tool = max(subsystem_signals.items(), key=lambda x: abs(x[1]))[0] if subsystem_signals else 'dna'
        
        # Decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        decision = {
            'action': action,
            'confidence': confidence,
            'position_size': position_size,
            'use_stop': use_stop,
            'stop_price': stop_price,
            'use_target': use_target,
            'target_price': target_price,
            'primary_tool': primary_tool,
            'exploration': exploration,
            'epsilon': self.epsilon,
            'state_features': features,
            'intelligence_result': intelligence_result
        }
        
        # Store for learning
        self._store_decision(decision)
        
        log.info(f"Decision: {['HOLD', 'BUY', 'SELL'][action]} "
                f"(conf: {confidence:.3f}, size: {position_size:.1f}, "
                f"tool: {primary_tool}, explore: {exploration})")
        
        return decision
    
    def learn_from_outcome(self, decision: Dict, outcome_pnl: float, trade_data: Dict):
        """Learn from trade outcome"""
        
        # Update statistics
        if outcome_pnl > 0:
            self.successful_trades += 1
        self.total_pnl += outcome_pnl
        
        # Normalize outcome for learning
        normalized_outcome = np.tanh(outcome_pnl / 50.0)  # Normalize to [-1, 1]
        
        # Store experience
        experience = {
            'state_features': decision['state_features'],
            'action': decision['action'],
            'reward': normalized_outcome,
            'done': True,  # Each trade is episodic
            'confidence': decision['confidence'],
            'position_size': decision['position_size']
        }
        
        self.experience_buffer.append(experience)
        
        # Update meta-learner
        self.meta_learner.update_parameter('confidence_threshold', normalized_outcome)
        self.meta_learner.update_parameter('position_size_multiplier', normalized_outcome)
        
        if decision['use_stop']:
            self.meta_learner.update_parameter('stop_loss_pct', normalized_outcome)
        if decision['use_target']:
            self.meta_learner.update_parameter('take_profit_pct', normalized_outcome)
        
        # Train network if enough experience
        if len(self.experience_buffer) >= 64:
            self._train_network()
        
        log.info(f"Learning from outcome: ${outcome_pnl:.2f} -> {normalized_outcome:.3f}")
    
    def _prepare_features(self, market_data: Dict, intelligence_result: Dict, account_data: Dict) -> np.ndarray:
        """Prepare input features for neural network"""
        features = []
        
        # Market features (price changes, volume ratios)
        prices = market_data.get('price_1m', [4000])[-10:]  # Last 10 prices
        if len(prices) >= 2:
            price_changes = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            features.extend(price_changes[-5:])  # Last 5 price changes
        else:
            features.extend([0.0] * 5)
        
        # Pad to exactly 5 features
        while len(features) < 5:
            features.append(0.0)
        features = features[:5]
        
        # Intelligence signals
        subsystem_signals = intelligence_result.get('subsystem_signals', {})
        features.extend([
            subsystem_signals.get('dna', 0.0),
            subsystem_signals.get('micro', 0.0),
            subsystem_signals.get('temporal', 0.0),
            subsystem_signals.get('immune', 0.0)
        ])
        
        # Overall intelligence signal and confidence
        features.extend([
            intelligence_result.get('overall_signal', 0.0),
            intelligence_result.get('confidence', 0.0)
        ])
        
        # Account features
        balance = account_data.get('account_balance', 25000)
        buying_power = account_data.get('buying_power', 25000)
        daily_pnl = account_data.get('daily_pnl', 0.0)
        
        features.extend([
            min(1.0, balance / 50000),  # Normalized balance
            min(1.0, buying_power / 50000),  # Normalized buying power
            np.tanh(daily_pnl / 1000),  # Normalized daily P&L
        ])
        
        # Time features
        now = market_data.get('timestamp', 0)
        hour = (now % 86400) // 3600  # Hour of day
        features.extend([
            hour / 24.0,  # Normalized hour
            (hour % 12) / 12.0,  # Normalized AM/PM
        ])
        
        # Meta-learning features
        features.extend([
            self.epsilon,  # Current exploration rate
            self.meta_learner.get_learning_efficiency(),  # Learning efficiency
        ])
        
        # Pad or truncate to exactly 20 features
        while len(features) < 20:
            features.append(0.0)
        features = features[:20]
        
        return np.array(features, dtype=np.float32)
    
    def _store_decision(self, decision: Dict):
        """Store decision for potential learning"""
        # This is called when making a decision, actual learning happens in learn_from_outcome
        pass
    
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
        
        if self.total_decisions % 50 == 0:
            log.info(f"Training: loss={total_loss.item():.4f}, "
                    f"epsilon={self.epsilon:.3f}, "
                    f"success_rate={self.successful_trades/max(1, self.total_decisions):.2%}")
    
    def save_model(self, filepath: str):
        """Save model state"""
        torch.save({
            'network_state': self.network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_decisions': self.total_decisions,
            'successful_trades': self.successful_trades,
            'total_pnl': self.total_pnl
        }, filepath)
        log.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model state"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.network.load_state_dict(checkpoint['network_state'])
            self.target_network.load_state_dict(checkpoint['target_network_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.epsilon = checkpoint['epsilon']
            self.total_decisions = checkpoint['total_decisions']
            self.successful_trades = checkpoint['successful_trades']
            self.total_pnl = checkpoint['total_pnl']
            log.info(f"Model loaded from {filepath}")
        except Exception as e:
            log.info(f"Starting with fresh model: {e}")
    
    def get_status(self):
        """Get current status"""
        return {
            'total_decisions': self.total_decisions,
            'successful_trades': self.successful_trades,
            'success_rate': self.successful_trades / max(1, self.total_decisions),
            'total_pnl': self.total_pnl,
            'epsilon': self.epsilon,
            'experience_buffer_size': len(self.experience_buffer),
            'learning_efficiency': self.meta_learner.get_learning_efficiency()
        }

# Factory function
def create_rl_agent(meta_learner, config):
    return SimpleRLAgent(meta_learner, config)