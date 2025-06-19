# trading_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time

from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Any

from intelligence_engine import Features
from data_processor import MarketData
from meta_learner import MetaLearner
from adaptive_network import AdaptiveTradingNetwork, FeatureLearner, StateEncoder

@dataclass
class Decision:
    action: str
    confidence: float
    size: float
    stop_price: float = 0.0
    target_price: float = 0.0
    primary_tool: str = 'unknown'
    exploration: bool = False
    intelligence_data: Dict = None
    state_features: List = None


class TradingAgent:
    def __init__(self, intelligence, portfolio):
        self.intelligence = intelligence
        self.portfolio = portfolio
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Meta-learning system
        self.meta_learner = MetaLearner(state_dim=20)
        
        # Adaptive neural components
        initial_sizes = self.meta_learner.architecture_evolver.current_sizes
        self.network = AdaptiveTradingNetwork(input_size=20, hidden_sizes=initial_sizes).to(self.device)
        self.target_network = AdaptiveTradingNetwork(input_size=20, hidden_sizes=initial_sizes).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        
        # Feature learning
        self.feature_learner = FeatureLearner(raw_feature_dim=50, learned_feature_dim=20).to(self.device)
        self.state_encoder = StateEncoder()
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.network.parameters()) + 
            list(self.feature_learner.parameters()) + 
            list(self.meta_learner.subsystem_weights.parameters()) +
            list(self.meta_learner.exploration_strategy.parameters()),
            lr=0.001
        )
        
        # Experience replay
        self.experience_buffer = deque(maxlen=10000)
        
        # Statistics
        self.total_decisions = 0
        self.successful_trades = 0
        self.total_pnl = 0.0
        self.last_trade_time = 0.0
    
    def decide(self, features: Features, market_data: MarketData) -> Decision:
        self.total_decisions += 1
        
        # Check architecture evolution
        if self.meta_learner.should_evolve_architecture():
            self._evolve_architecture()
        
        # Create comprehensive state representation
        meta_context = self._get_meta_context(market_data)
        raw_state = self.state_encoder.create_full_state(market_data, features, meta_context)
        
        # Learn features and get final state
        learned_state = self.feature_learner(raw_state.unsqueeze(0).to(self.device))
        
        # Get subsystem contributions with adaptive weighting
        subsystem_signals = torch.tensor([
            features.dna_signal,
            features.micro_signal,
            features.temporal_signal,
            features.immune_signal
        ], device=self.device)
        
        subsystem_weights = self.meta_learner.get_subsystem_weights()
        weighted_signal = torch.sum(subsystem_signals * subsystem_weights)
        
        # Basic trading constraints (only very basic ones, everything else learned)
        if not self._should_consider_trading(market_data, meta_context):
            return Decision('hold', 0, 0)
        
        # Neural network decision
        with torch.no_grad():
            outputs = self.network(learned_state)
            
            # Get action probabilities
            action_probs = F.softmax(outputs['action_logits'], dim=-1).cpu().numpy()[0]
            confidence = float(outputs['confidence'].cpu().numpy()[0])
            
            # Learned exploration decision
            should_explore = self.meta_learner.should_explore(
                learned_state.squeeze(), 
                meta_context
            )
            
            if should_explore:
                # Exploration: use subsystem signal for action selection
                if weighted_signal > 0.1:
                    action_idx = 1  # Buy
                elif weighted_signal < -0.1:
                    action_idx = 2  # Sell
                else:
                    action_idx = 0  # Hold
                exploration = True
            else:
                # Exploitation: use network's best prediction
                action_idx = np.argmax(action_probs)
                exploration = False
            
            # Apply learned confidence threshold
            confidence_threshold = self.meta_learner.get_parameter('confidence_threshold')
            if confidence < confidence_threshold and not exploration:
                action_idx = 0
            
            # Get sizing and risk parameters
            position_size = float(outputs['position_size'].cpu().numpy()[0])
            use_stop = float(outputs['use_stop'].cpu().numpy()[0]) > 0.5
            stop_distance = float(outputs['stop_distance'].cpu().numpy()[0])
            use_target = float(outputs['use_target'].cpu().numpy()[0]) > 0.5
            target_distance = float(outputs['target_distance'].cpu().numpy()[0])
        
        # Convert to decision
        actions = ['hold', 'buy', 'sell']
        action = actions[action_idx]
        
        if action == 'hold':
            return Decision('hold', confidence, 0)
        
        # Calculate stop and target prices based on learned preferences
        stop_price = 0
        target_price = 0
        
        stop_distance_factor = self.meta_learner.get_parameter('stop_distance_factor')
        target_distance_factor = self.meta_learner.get_parameter('target_distance_factor')
        
        if use_stop:
            adjusted_distance = stop_distance_factor * (1 + stop_distance)
            if action == 'buy':
                stop_price = market_data.price * (1 - adjusted_distance)
            else:
                stop_price = market_data.price * (1 + adjusted_distance)
        
        if use_target:
            adjusted_distance = target_distance_factor * (1 + target_distance)
            if action == 'buy':
                target_price = market_data.price * (1 + adjusted_distance)
            else:
                target_price = market_data.price * (1 - adjusted_distance)
        
        # Determine primary tool
        primary_tool = self._get_primary_tool(subsystem_signals, subsystem_weights)
        
        # Store intelligence data for learning
        intelligence_data = {
            'subsystem_signals': subsystem_signals.cpu().numpy().tolist(),
            'subsystem_weights': subsystem_weights.cpu().numpy().tolist(),
            'weighted_signal': float(weighted_signal),
            'current_patterns': getattr(features, 'current_patterns', {})
        }
        
        self.last_trade_time = market_data.timestamp
        
        return Decision(
            action=action,
            confidence=confidence,
            size=position_size,
            stop_price=stop_price,
            target_price=target_price,
            primary_tool=primary_tool,
            exploration=exploration,
            intelligence_data=intelligence_data,
            state_features=learned_state.squeeze().cpu().numpy().tolist()
        )
    
    def learn_from_trade(self, trade):
        if not hasattr(trade, 'intelligence_data'):
            return
        
        # Update statistics
        if trade.pnl > 0:
            self.successful_trades += 1
        self.total_pnl += trade.pnl
        
        # Prepare comprehensive trade data for meta-learning
        trade_data = {
            'pnl': trade.pnl,
            'account_balance': getattr(trade.market_data, 'account_balance', 25000),
            'hold_time': trade.exit_time - trade.entry_time,
            'was_exploration': getattr(trade, 'exploration', False),
            'subsystem_contributions': torch.tensor(trade.intelligence_data.get('subsystem_signals', [0,0,0,0])),
            'subsystem_agreement': self._calculate_subsystem_agreement(trade.intelligence_data),
            'confidence': getattr(trade, 'confidence', 0.5),
            'primary_tool': getattr(trade, 'primary_tool', 'unknown'),
            'stop_used': getattr(trade, 'stop_used', False),
            'target_used': getattr(trade, 'target_used', False)
        }
        
        # Compute reward using adaptive reward engine
        reward = self.meta_learner.compute_reward(trade_data)
        
        # Store experience for neural network training
        if hasattr(trade, 'state_features') and trade.state_features:
            experience = {
                'state_features': trade.state_features,
                'action': ['hold', 'buy', 'sell'].index(trade.action),
                'reward': reward,
                'done': True,
                'trade_data': trade_data
            }
            
            self.experience_buffer.append(experience)
        
        # Meta-learning update
        self.meta_learner.learn_from_outcome(trade_data)
        
        # Train networks if enough experience
        if len(self.experience_buffer) >= 64:
            self._train_networks()
        
        # Periodic parameter adaptation
        if self.total_decisions % 50 == 0:
            self.meta_learner.adapt_parameters()
    
    def _should_consider_trading(self, market_data: MarketData, meta_context: Dict) -> bool:
        # Only very basic constraints - everything else learned by meta-learner
        
        # Learned loss tolerance
        loss_tolerance = self.meta_learner.get_parameter('loss_tolerance_factor')
        max_loss = market_data.account_balance * loss_tolerance
        if market_data.daily_pnl <= -max_loss:
            return False
        
        # Learned consecutive loss limit
        consecutive_limit = self.meta_learner.get_parameter('consecutive_loss_tolerance')
        if meta_context['consecutive_losses'] >= consecutive_limit:
            return False
        
        # Learned frequency limit
        frequency_limit = self.meta_learner.get_parameter('trade_frequency_base')
        time_since_last = market_data.timestamp - self.last_trade_time
        if time_since_last < (3600 / frequency_limit):
            return False
        
        return True
    
    def _get_meta_context(self, market_data: MarketData) -> Dict[str, float]:
        portfolio_summary = self.portfolio.get_summary()
        
        return {
            'recent_performance': np.tanh(portfolio_summary.get('daily_pnl', 0) / (market_data.account_balance * 0.01)),
            'consecutive_losses': portfolio_summary.get('consecutive_losses', 0),
            'position_count': portfolio_summary.get('pending_orders', 0),
            'trades_today': portfolio_summary.get('total_trades', 0),
            'time_since_last_trade': 0.0 if self.last_trade_time == 0 else (np.log(1 + (time.time() - self.last_trade_time) / 3600)),
            'learning_efficiency': self.meta_learner.get_learning_efficiency(),
            'architecture_generation': self.meta_learner.architecture_evolver.generations
        }
    
    def _get_primary_tool(self, signals: torch.Tensor, weights: torch.Tensor) -> str:
        weighted_signals = torch.abs(signals * weights)
        tool_names = ['dna', 'micro', 'temporal', 'immune']
        
        if torch.sum(weighted_signals) == 0:
            return 'basic'
        
        primary_idx = torch.argmax(weighted_signals)
        return tool_names[primary_idx]
    
    def _calculate_subsystem_agreement(self, intelligence_data: Dict) -> float:
        signals = intelligence_data.get('subsystem_signals', [0, 0, 0, 0])
        
        if not signals or all(s == 0 for s in signals):
            return 0.5
        
        # Calculate how much subsystems agree on direction
        positive_signals = sum(1 for s in signals if s > 0.1)
        negative_signals = sum(1 for s in signals if s < -0.1)
        total_signals = len([s for s in signals if abs(s) > 0.1])
        
        if total_signals == 0:
            return 0.5
        
        agreement = max(positive_signals, negative_signals) / total_signals
        return agreement
    
    def _evolve_architecture(self):
        new_sizes = self.meta_learner.evolve_architecture()
        
        # Evolve main network
        self.network.evolve_architecture(new_sizes)
        self.target_network.evolve_architecture(new_sizes)
        
        # Update optimizer with new parameters
        self.optimizer = optim.Adam(
            list(self.network.parameters()) + 
            list(self.feature_learner.parameters()) + 
            list(self.meta_learner.subsystem_weights.parameters()) +
            list(self.meta_learner.exploration_strategy.parameters()),
            lr=0.001
        )
    
    def _train_networks(self):
        if len(self.experience_buffer) < 32:
            return
        
        # Sample batch
        import random
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
        
        # Policy loss
        action_logits = outputs['action_logits']
        action_probs = F.log_softmax(action_logits, dim=-1)
        selected_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        policy_loss = -(selected_probs * rewards).mean()
        
        # Value losses
        confidence_target = torch.abs(rewards).unsqueeze(1)
        confidence_loss = F.mse_loss(outputs['confidence'], confidence_target)
        
        # Position size loss (reward-weighted)
        size_target = torch.clamp(torch.abs(rewards) * 2.0, 0.5, 3.0).unsqueeze(1)
        size_loss = F.mse_loss(outputs['position_size'], size_target)
        
        # Total loss
        total_loss = policy_loss + 0.1 * confidence_loss + 0.05 * size_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.network.parameters()) + list(self.feature_learner.parameters()), 
            1.0
        )
        self.optimizer.step()
        
        # Update target network
        if self.total_decisions % 100 == 0:
            self.target_network.load_state_dict(self.network.state_dict())
    
    def save_model(self, filepath: str):
        torch.save({
            'network_state': self.network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'feature_learner_state': self.feature_learner.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'total_decisions': self.total_decisions,
            'successful_trades': self.successful_trades,
            'total_pnl': self.total_pnl,
            'last_trade_time': self.last_trade_time
        }, filepath)
        
        # Save meta-learner separately
        self.meta_learner.save_state(filepath.replace('.pt', '_meta.pt'))
    
    def load_model(self, filepath: str):
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.network.load_state_dict(checkpoint['network_state'])
            self.target_network.load_state_dict(checkpoint['target_network_state'])
            self.feature_learner.load_state_dict(checkpoint['feature_learner_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            
            self.total_decisions = checkpoint.get('total_decisions', 0)
            self.successful_trades = checkpoint.get('successful_trades', 0)
            self.total_pnl = checkpoint.get('total_pnl', 0.0)
            self.last_trade_time = checkpoint.get('last_trade_time', 0.0)
            
            # Load meta-learner
            self.meta_learner.load_state(filepath.replace('.pt', '_meta.pt'))
            
        except FileNotFoundError:
            pass
    
    def get_stats(self) -> dict:
        return {
            'total_decisions': self.total_decisions,
            'successful_trades': self.successful_trades,
            'success_rate': self.successful_trades / max(1, self.total_decisions),
            'total_pnl': self.total_pnl,
            'experience_size': len(self.experience_buffer),
            'learning_efficiency': self.meta_learner.get_learning_efficiency(),
            'architecture_generation': self.meta_learner.architecture_evolver.generations,
            'current_sizes': self.meta_learner.architecture_evolver.current_sizes,
            'subsystem_weights': self.meta_learner.get_subsystem_weights().detach().cpu().numpy().tolist(),
            'key_parameters': {
                'confidence_threshold': self.meta_learner.get_parameter('confidence_threshold'),
                'position_size_factor': self.meta_learner.get_parameter('position_size_factor'),
                'loss_tolerance_factor': self.meta_learner.get_parameter('loss_tolerance_factor'),
                'stop_preference': self.meta_learner.get_parameter('stop_preference'),
                'target_preference': self.meta_learner.get_parameter('target_preference')
            }
        }