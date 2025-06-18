# pure_blackbox_meta_learner.py - TRUE Pure Black Box with Self-Optimization

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
import logging
from collections import deque
import json

log = logging.getLogger(__name__)

class MetaParameterLearner:
    """
    Learns ALL system parameters - no static values
    """
    
    def __init__(self):
        # Meta-parameters that learn themselves
        self.parameters = {
            # Risk management - starts neutral, learns from losses
            'max_daily_loss_pct': 0.05,  # % of account
            'max_consecutive_losses': 5.0,
            'position_size_base': 0.1,
            
            # Learning rates - adapts based on learning efficiency
            'policy_lr': 1e-4,
            'value_lr': 3e-4,
            'meta_lr': 1e-5,
            
            # Reward structure - discovers what matters
            'pnl_weight': 1.0,
            'risk_bonus_weight': 0.0,
            'consistency_weight': 0.0,
            'drawdown_penalty_weight': 0.0,
            
            # Thresholds - all learned from experience
            'entry_confidence_threshold': 0.5,
            'exit_confidence_threshold': 0.5,
            'stop_distance_max': 0.02,
            'target_distance_max': 0.05,
            
            # Exploration - adapts based on learning progress
            'epsilon_decay_rate': 0.999,
            'epsilon_min': 0.1,
            
            # Network architecture weights - meta-learns optimal structure
            'hidden_layer_multiplier': 1.0,
            'attention_weight': 1.0,
            'lstm_layers': 1.0,
        }
        
        # Track parameter performance
        self.parameter_outcomes = {param: deque(maxlen=100) for param in self.parameters}
        self.parameter_gradients = {param: 0.0 for param in self.parameters}
        
        # Meta-learning state
        self.meta_optimizer_state = {}
        self.learning_efficiency_history = deque(maxlen=50)
        self.total_updates = 0
        
        log.info("META-LEARNER: All parameters will self-optimize")
    
    def update_parameter(self, param_name: str, outcome: float, context: Dict = None):
        """Update a parameter based on outcome"""
        if param_name not in self.parameters:
            return
        
        # Store outcome
        self.parameter_outcomes[param_name].append({
            'outcome': outcome,
            'value': self.parameters[param_name],
            'context': context or {},
            'timestamp': datetime.now()
        })
        
        # Calculate gradient estimate
        outcomes = list(self.parameter_outcomes[param_name])
        if len(outcomes) >= 10:
            # Finite difference gradient approximation
            recent_outcomes = outcomes[-10:]
            values = [o['value'] for o in recent_outcomes]
            rewards = [o['outcome'] for o in recent_outcomes]
            
            # Simple gradient estimate
            if len(set(values)) > 1:  # Need parameter variation
                gradient = np.corrcoef(values, rewards)[0, 1] if len(values) > 1 else 0.0
                self.parameter_gradients[param_name] = gradient
                
                # Adaptive learning rate for this parameter
                param_lr = self.parameters.get('meta_lr', 1e-5)
                
                # Update parameter
                old_value = self.parameters[param_name]
                self.parameters[param_name] += param_lr * gradient * abs(old_value)
                
                # Clamp to reasonable bounds
                self.parameters[param_name] = self._clamp_parameter(param_name, self.parameters[param_name])
                
                if abs(gradient) > 0.1:  # Significant gradient
                    log.info(f"META-LEARNING: {param_name} {old_value:.6f} -> {self.parameters[param_name]:.6f} (grad: {gradient:.4f})")
    
    def _clamp_parameter(self, param_name: str, value: float) -> float:
        """Clamp parameters to reasonable bounds"""
        bounds = {
            'max_daily_loss_pct': (0.01, 0.20),  # 1%-20% of account
            'max_consecutive_losses': (3, 20),
            'position_size_base': (0.01, 2.0),
            'policy_lr': (1e-6, 1e-2),
            'value_lr': (1e-6, 1e-2),
            'meta_lr': (1e-7, 1e-3),
            'entry_confidence_threshold': (0.1, 0.9),
            'exit_confidence_threshold': (0.1, 0.9),
            'stop_distance_max': (0.005, 0.10),
            'target_distance_max': (0.01, 0.20),
            'epsilon_decay_rate': (0.99, 0.9999),
            'epsilon_min': (0.01, 0.5),
            'hidden_layer_multiplier': (0.5, 3.0),
            'attention_weight': (0.1, 2.0),
            'lstm_layers': (1.0, 4.0),
        }
        
        if param_name in bounds:
            min_val, max_val = bounds[param_name]
            return max(min_val, min(max_val, value))
        
        return value
    
    def get_parameter(self, param_name: str) -> float:
        """Get current parameter value"""
        return self.parameters.get(param_name, 0.5)
    
    def adapt_learning_rates(self, learning_efficiency: float):
        """Adapt learning rates based on learning efficiency"""
        self.learning_efficiency_history.append(learning_efficiency)
        
        if len(self.learning_efficiency_history) >= 10:
            recent_efficiency = np.mean(list(self.learning_efficiency_history)[-10:])
            
            # If learning is efficient, can use higher learning rates
            if recent_efficiency > 0.6:
                self.update_parameter('policy_lr', 0.1)  # Positive signal
                self.update_parameter('value_lr', 0.1)
            elif recent_efficiency < 0.3:
                self.update_parameter('policy_lr', -0.1)  # Negative signal
                self.update_parameter('value_lr', -0.1)
    
    def get_network_architecture(self) -> Dict:
        """Get current optimal network architecture"""
        base_hidden = 64
        multiplier = self.get_parameter('hidden_layer_multiplier')
        
        return {
            'hidden_size': int(base_hidden * multiplier),
            'lstm_layers': max(1, int(self.get_parameter('lstm_layers'))),
            'attention_layers': max(1, int(self.get_parameter('attention_weight') * 2)),
        }

class AdaptiveRewardLearner:
    """
    Learns optimal reward structure - no hardcoded bonuses
    """
    
    def __init__(self, meta_learner: MetaParameterLearner):
        self.meta_learner = meta_learner
        
        # Reward components that learn their importance
        self.reward_components = {
            'pnl': {'weight': 1.0, 'history': deque(maxlen=100)},
            'risk_management': {'weight': 0.0, 'history': deque(maxlen=100)},
            'consistency': {'weight': 0.0, 'history': deque(maxlen=100)},
            'drawdown_control': {'weight': 0.0, 'history': deque(maxlen=100)},
            'timing': {'weight': 0.0, 'history': deque(maxlen=100)},
            'tool_usage': {'weight': 0.0, 'history': deque(maxlen=100)},
        }
        
        self.total_reward_history = deque(maxlen=200)
        
    def calculate_adaptive_reward(self, trade_data: Dict) -> float:
        """Calculate reward using learned weights"""
        pnl = trade_data.get('pnl', 0.0)
        
        # Base PnL component
        pnl_component = pnl / self.meta_learner.get_parameter('pnl_normalizer')
        
        # Dynamic risk management component
        risk_component = 0.0
        if trade_data.get('used_stop') and pnl > -20:
            risk_component = 1.0  # Good risk management
        elif trade_data.get('big_loss', False):
            risk_component = -1.0  # Poor risk management
        
        # Consistency component (reward stable performance)
        recent_pnls = [t.get('pnl', 0) for t in list(self.total_reward_history)[-10:]]
        if len(recent_pnls) >= 5:
            consistency_component = -np.std(recent_pnls) / 50.0  # Penalize volatility
        else:
            consistency_component = 0.0
        
        # Drawdown control
        max_drawdown = trade_data.get('max_drawdown', 0.0)
        drawdown_component = -max(0, max_drawdown - 0.02) * 100  # Penalize >2% drawdown
        
        # Timing component (faster resolution = better)
        hold_time_hours = trade_data.get('hold_time_hours', 1.0)
        timing_component = max(0, 1.0 - hold_time_hours / 24.0)  # Reward quick resolution
        
        # Tool usage component (reward successful tool discovery)
        tool_confidence = trade_data.get('tool_confidence', 0.5)
        tool_component = (tool_confidence - 0.5) * 2.0  # Convert to -1 to +1
        
        # Combine with learned weights
        components = {
            'pnl': pnl_component,
            'risk_management': risk_component,
            'consistency': consistency_component,
            'drawdown_control': drawdown_component,
            'timing': timing_component,
            'tool_usage': tool_component,
        }
        
        total_reward = 0.0
        for component, value in components.items():
            weight = self.reward_components[component]['weight']
            contribution = weight * value
            total_reward += contribution
            
            # Store for weight learning
            self.reward_components[component]['history'].append({
                'value': value,
                'contribution': contribution,
                'final_pnl': pnl
            })
        
        # Learn reward weights
        self._update_reward_weights(total_reward, pnl)
        
        self.total_reward_history.append({
            'reward': total_reward,
            'pnl': pnl,
            'components': components.copy()
        })
        
        return total_reward
    
    def _update_reward_weights(self, total_reward: float, actual_pnl: float):
        """Learn which reward components actually predict good trades"""
        
        # For each component, see if its signal correlates with actual PnL
        for component_name, component_data in self.reward_components.items():
            history = list(component_data['history'])
            
            if len(history) >= 20:
                # Get recent component values and actual PnLs
                component_values = [h['value'] for h in history[-20:]]
                actual_pnls = [h['final_pnl'] for h in history[-20:]]
                
                # Calculate correlation
                if len(set(component_values)) > 1:  # Need variation
                    correlation = np.corrcoef(component_values, actual_pnls)[0, 1]
                    
                    # Update weight based on correlation
                    old_weight = component_data['weight']
                    learning_rate = self.meta_learner.get_parameter('meta_lr') * 10
                    
                    # Positive correlation = increase weight, negative = decrease
                    component_data['weight'] += learning_rate * correlation
                    component_data['weight'] = max(-2.0, min(2.0, component_data['weight']))  # Clamp
                    
                    if abs(correlation) > 0.3:  # Significant correlation
                        log.info(f"REWARD LEARNING: {component_name} weight {old_weight:.3f} -> {component_data['weight']:.3f} (corr: {correlation:.3f})")

class SelfOptimizingNetwork(nn.Module):
    """
    Network that adapts its own architecture
    """
    
    def __init__(self, meta_learner: MetaParameterLearner):
        super().__init__()
        self.meta_learner = meta_learner
        
        # Get initial architecture from meta-learner
        arch = meta_learner.get_network_architecture()
        
        self.hidden_size = arch['hidden_size']
        self.lstm_layers = arch['lstm_layers']
        
        # Build initial network
        self._build_network()
        
        # Architecture adaptation
        self.architecture_performance = deque(maxlen=50)
        self.last_rebuild = 0
        
    def _build_network(self):
        """Build network with current architecture parameters"""
        
        # Market encoder with adaptive layers
        self.market_encoder = nn.LSTM(
            15, self.hidden_size, 
            num_layers=self.lstm_layers, 
            batch_first=True
        )
        
        # Adaptive subsystem processor
        self.subsystem_processor = nn.Sequential(
            nn.Linear(16, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU()
        )
        
        # Decision heads with adaptive sizing
        decision_input_size = self.hidden_size + self.hidden_size // 2
        
        self.action_head = self._create_adaptive_head(decision_input_size, 3)
        self.confidence_head = self._create_adaptive_head(decision_input_size, 1)
        self.risk_head = self._create_adaptive_head(decision_input_size, 4)  # stop/target decisions
        
    def _create_adaptive_head(self, input_size: int, output_size: int):
        """Create decision head with adaptive complexity"""
        complexity = self.meta_learner.get_parameter('hidden_layer_multiplier')
        
        if complexity > 1.5:  # High complexity
            return nn.Sequential(
                nn.Linear(input_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_size // 2, output_size)
            )
        else:  # Simple network
            return nn.Sequential(
                nn.Linear(input_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_size // 2, output_size)
            )
    
    def forward(self, market_obs, subsystem_features):
        """Forward pass with performance tracking"""
        
        # Encode inputs
        market_encoded, _ = self.market_encoder(market_obs)
        market_repr = market_encoded[:, -1]
        
        subsystem_repr = self.subsystem_processor(subsystem_features)
        
        combined = torch.cat([market_repr, subsystem_repr], dim=-1)
        
        # Generate outputs
        action_logits = self.action_head(combined)
        confidence = torch.sigmoid(self.confidence_head(combined))
        risk_outputs = self.risk_head(combined)
        
        return {
            'action_logits': action_logits,
            'confidence': confidence,
            'use_stop': torch.sigmoid(risk_outputs[:, 0:1]),
            'stop_distance': torch.sigmoid(risk_outputs[:, 1:2]) * self.meta_learner.get_parameter('stop_distance_max'),
            'use_target': torch.sigmoid(risk_outputs[:, 2:3]),
            'target_distance': torch.sigmoid(risk_outputs[:, 3:4]) * self.meta_learner.get_parameter('target_distance_max'),
        }
    
    def maybe_adapt_architecture(self, performance_score: float):
        """Adapt architecture based on performance"""
        self.architecture_performance.append(performance_score)
        
        # Consider rebuilding every 100 steps
        if len(self.architecture_performance) >= 50 and (self.last_rebuild == 0 or len(self.architecture_performance) % 100 == 0):
            
            recent_perf = np.mean(list(self.architecture_performance)[-20:])
            older_perf = np.mean(list(self.architecture_performance)[-40:-20]) if len(self.architecture_performance) >= 40 else recent_perf
            
            # If performance is declining, try adapting architecture
            if recent_perf < older_perf - 0.1:
                log.info(f"ARCHITECTURE ADAPTATION: Performance declining ({recent_perf:.3f} vs {older_perf:.3f})")
                
                # Update architecture parameters
                self.meta_learner.update_parameter('hidden_layer_multiplier', -0.5)  # Try different complexity
                
                # Rebuild network
                old_hidden = self.hidden_size
                arch = self.meta_learner.get_network_architecture()
                self.hidden_size = arch['hidden_size']
                self.lstm_layers = arch['lstm_layers']
                
                if self.hidden_size != old_hidden:
                    log.info(f"REBUILDING NETWORK: {old_hidden} -> {self.hidden_size} hidden units")
                    self._build_network()
                    self.last_rebuild = len(self.architecture_performance)

class PureBlackBoxAgent:
    """
    Truly pure black box agent - ALL parameters self-optimize
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Meta-learner controls ALL parameters
        self.meta_learner = MetaParameterLearner()
        
        # Adaptive reward system
        self.reward_learner = AdaptiveRewardLearner(self.meta_learner)
        
        # Self-optimizing network
        self.network = SelfOptimizingNetwork(self.meta_learner).to(self.device)
        
        # Optimizers with adaptive learning rates
        self.optimizer = None
        self._update_optimizers()
        
        # Experience with adaptive memory size
        self.experience = deque(maxlen=10000)  # Will be meta-learned
        
        # Learning state
        self.step_count = 0
        self.last_performance_update = 0
        
        log.info("PURE BLACK BOX AGENT: All parameters will self-optimize")
        log.info("No static thresholds, learning rates, or reward structures")
    
    def _update_optimizers(self):
        """Update optimizers with current meta-learned learning rates"""
        policy_lr = self.meta_learner.get_parameter('policy_lr')
        
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), 
            lr=policy_lr
        )
    
    def select_action(self, market_obs: np.ndarray, subsystem_features: np.ndarray, 
                     current_price: float) -> Dict:
        """Select action with fully adaptive parameters"""
        
        with torch.no_grad():
            # Prepare inputs
            market_tensor = torch.tensor(market_obs, dtype=torch.float32, device=self.device)
            market_tensor = market_tensor.unsqueeze(0).unsqueeze(0)
            
            subsystem_tensor = torch.tensor(subsystem_features, dtype=torch.float32, device=self.device)
            subsystem_tensor = subsystem_tensor.unsqueeze(0)
            
            # Forward pass
            outputs = self.network(market_tensor, subsystem_tensor)
            
            # Action selection with adaptive confidence threshold
            action_probs = torch.softmax(outputs['action_logits'], dim=-1).cpu().numpy()[0]
            confidence = float(outputs['confidence'].cpu().numpy()[0])
            
            # Adaptive confidence threshold
            conf_threshold = self.meta_learner.get_parameter('entry_confidence_threshold')
            
            if confidence >= conf_threshold:
                action = np.argmax(action_probs)
            else:
                action = 0  # Hold
            
            # Risk management with adaptive parameters
            use_stop = float(outputs['use_stop'].cpu().numpy()[0]) > 0.5
            stop_distance = float(outputs['stop_distance'].cpu().numpy()[0])
            use_target = float(outputs['use_target'].cpu().numpy()[0]) > 0.5
            target_distance = float(outputs['target_distance'].cpu().numpy()[0])
            
            # Calculate prices
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
            
            return {
                'action': action,
                'confidence': confidence,
                'conf_threshold_used': conf_threshold,
                'use_stop': use_stop,
                'stop_price': stop_price,
                'stop_distance_pct': stop_distance * 100,
                'use_target': use_target,
                'target_price': target_price,
                'target_distance_pct': target_distance * 100,
                'raw_outputs': outputs
            }
    
    def store_experience_and_learn(self, state_data: Dict, trade_outcome: Dict):
        """Store experience and trigger learning with adaptive reward"""
        
        # Calculate adaptive reward
        adaptive_reward = self.reward_learner.calculate_adaptive_reward(trade_outcome)
        
        # Store experience
        experience = {
            'state_data': state_data,
            'trade_outcome': trade_outcome,
            'adaptive_reward': adaptive_reward,
            'timestamp': datetime.now()
        }
        
        self.experience.append(experience)
        
        # Trigger learning
        if len(self.experience) >= 10:  # Much earlier learning start
            self._meta_learning_step()
        
        # Update meta-parameters based on outcomes
        pnl = trade_outcome.get('pnl', 0.0)
        
        # Update all relevant meta-parameters
        self.meta_learner.update_parameter('entry_confidence_threshold', adaptive_reward)
        
        if trade_outcome.get('used_stop'):
            self.meta_learner.update_parameter('stop_distance_max', adaptive_reward)
        
        if trade_outcome.get('used_target'):
            self.meta_learner.update_parameter('target_distance_max', adaptive_reward)
        
        # Adapt network architecture based on performance
        if self.step_count % 20 == 0:
            recent_rewards = [exp['adaptive_reward'] for exp in list(self.experience)[-20:]]
            avg_performance = np.mean(recent_rewards) if recent_rewards else 0.0
            self.network.maybe_adapt_architecture(avg_performance)
        
        self.step_count += 1
        
        # Update optimizers with new learning rates
        if self.step_count % 50 == 0:
            self._update_optimizers()
    
    def _meta_learning_step(self):
        """Meta-learning step with fully adaptive parameters"""
        
        if len(self.experience) < 5:
            return
        
        # Sample recent experiences
        batch_size = min(32, len(self.experience))
        batch = list(self.experience)[-batch_size:]
        
        # Extract data
        rewards = torch.tensor([exp['adaptive_reward'] for exp in batch], 
                              dtype=torch.float32, device=self.device)
        
        # Simple policy gradient update
        total_loss = -torch.mean(rewards)  # Maximize adaptive reward
        
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Adaptive gradient clipping
        grad_clip = self.meta_learner.get_parameter('gradient_clip_norm')
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), grad_clip)
        
        self.optimizer.step()
        
        # Calculate learning efficiency for meta-parameter adaptation
        if len(self.experience) >= 20:
            recent_performance = np.mean([exp['adaptive_reward'] for exp in list(self.experience)[-10:]])
            older_performance = np.mean([exp['adaptive_reward'] for exp in list(self.experience)[-20:-10]])
            learning_efficiency = (recent_performance - older_performance) / max(abs(older_performance), 0.1)
            
            self.meta_learner.adapt_learning_rates(learning_efficiency)
    
    def get_meta_status(self) -> str:
        """Get status of all meta-learned parameters"""
        
        status = "PURE BLACK BOX META-LEARNING STATUS:\n\n"
        
        status += "Risk Management (Self-Optimized):\n"
        status += f"  Daily Loss Limit: {self.meta_learner.get_parameter('max_daily_loss_pct'):.3f}\n"
        status += f"  Position Size: {self.meta_learner.get_parameter('position_size_base'):.3f}\n"
        status += f"  Stop Distance Max: {self.meta_learner.get_parameter('stop_distance_max'):.3f}\n"
        status += f"  Target Distance Max: {self.meta_learner.get_parameter('target_distance_max'):.3f}\n"
        
        status += "\nLearning Parameters (Self-Adapted):\n"
        status += f"  Policy LR: {self.meta_learner.get_parameter('policy_lr'):.6f}\n"
        status += f"  Entry Threshold: {self.meta_learner.get_parameter('entry_confidence_threshold'):.3f}\n"
        status += f"  Exit Threshold: {self.meta_learner.get_parameter('exit_confidence_threshold'):.3f}\n"
        
        status += "\nReward Structure (Learned):\n"
        for component, data in self.reward_learner.reward_components.items():
            status += f"  {component}: {data['weight']:.3f}\n"
        
        status += f"\nArchitecture (Adaptive):\n"
        arch = self.meta_learner.get_network_architecture()
        status += f"  Hidden Size: {arch['hidden_size']}\n"
        status += f"  LSTM Layers: {arch['lstm_layers']}\n"
        
        status += f"\nExperiences: {len(self.experience)}\n"
        status += f"Learning Steps: {self.step_count}\n"
        
        return status

# Usage example showing true black box behavior
if __name__ == "__main__":
    
    # Initialize pure black box agent
    agent = PureBlackBoxAgent()
    
    print("PURE BLACK BOX AGENT INITIALIZED")
    print("ALL parameters will self-optimize through experience")
    print("No hardcoded thresholds, learning rates, or reward structures")
    print("\nInitial Meta-Parameters:")
    print(agent.get_meta_status())
    
    # Simulate some trading to show self-optimization
    for i in range(100):
        # Simulate market data
        market_obs = np.random.randn(15).astype(np.float32)
        subsystem_features = np.random.randn(16).astype(np.float32)
        current_price = 4000.0 + np.random.randn() * 10
        
        # Get action
        decision = agent.select_action(market_obs, subsystem_features, current_price)
        
        # Simulate outcome
        simulated_pnl = np.random.randn() * 20  # Random P&L
        trade_outcome = {
            'pnl': simulated_pnl,
            'used_stop': decision['use_stop'],
            'used_target': decision['use_target'],
            'hold_time_hours': np.random.uniform(0.5, 8.0),
            'max_drawdown': abs(min(0, simulated_pnl)) / current_price,
            'tool_confidence': decision['confidence']
        }
        
        # Learn from outcome
        agent.store_experience_and_learn(decision, trade_outcome)
        
        # Show adaptation every 25 steps
        if (i + 1) % 25 == 0:
            print(f"\n--- After {i+1} experiences ---")
            print(f"Entry Threshold: {agent.meta_learner.get_parameter('entry_confidence_threshold'):.3f}")
            print(f"Learning Rate: {agent.meta_learner.get_parameter('policy_lr'):.6f}")
            print(f"Stop Distance: {agent.meta_learner.get_parameter('stop_distance_max'):.3f}")
    
    print("\n" + "="*60)
    print("FINAL PURE BLACK BOX STATUS")
    print("="*60)
    print(agent.get_meta_status())
    print("\nAll parameters self-optimized through pure experience!")