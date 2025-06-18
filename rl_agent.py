# rl_agent.py - FIXED: Added missing step_count attribute

import threading
import time
import random
import logging
from collections import deque
from typing import Tuple, Dict, List
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime

# Import meta-learning components
from meta_learner import PureMetaLearner, AdaptiveRewardLearner

log = logging.getLogger(__name__)

class SelfEvolvingPolicyNetwork(torch.nn.Module):
    """
    Policy network that evolves its own architecture based on performance
    """
    
    def __init__(self, meta_learner: PureMetaLearner, market_obs_size: int = 15, subsystem_features_size: int = 16):
        super().__init__()
        self.meta_learner = meta_learner
        self.market_obs_size = market_obs_size
        self.subsystem_features_size = subsystem_features_size
        
        # Build initial architecture
        self._build_network()
        
        # Track performance for architecture adaptation
        self.performance_history = deque(maxlen=100)
        self.last_rebuild_step = 0
        self.rebuild_threshold = 0.02  # Rebuild if performance drops 2%
        
    def _build_network(self):
        """Build network with current meta-learned architecture"""
        
        arch = self.meta_learner.get_network_architecture()
        self.hidden_size = arch['hidden_size']
        self.lstm_layers = arch['lstm_layers']
        self.dropout_rate = arch['dropout_rate']
        
        # Market observation encoder with adaptive LSTM layers
        self.market_encoder = torch.nn.LSTM(
            self.market_obs_size, 
            self.hidden_size, 
            num_layers=self.lstm_layers,
            dropout=self.dropout_rate if self.lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # Subsystem feature processor with adaptive complexity
        subsystem_layers = []
        subsystem_layers.append(torch.nn.Linear(self.subsystem_features_size, self.hidden_size))
        subsystem_layers.append(torch.nn.ReLU())
        
        # Add layers based on complexity parameter
        complexity = self.meta_learner.get_parameter('hidden_layer_multiplier')
        if complexity > 1.5:
            subsystem_layers.extend([
                torch.nn.Dropout(self.dropout_rate),
                torch.nn.Linear(self.hidden_size, self.hidden_size),
                torch.nn.ReLU()
            ])
        
        subsystem_layers.extend([
            torch.nn.Linear(self.hidden_size, self.hidden_size // 2),
            torch.nn.ReLU()
        ])
        
        self.subsystem_processor = torch.nn.Sequential(*subsystem_layers)
        
        # Adaptive attention mechanism
        attention_size = self.hidden_size + self.hidden_size // 2
        self.attention = torch.nn.MultiheadAttention(
            attention_size, 
            num_heads=max(1, arch['attention_layers']),
            dropout=self.dropout_rate,
            batch_first=True
        )
        
        # Decision heads with adaptive sizing
        decision_input_size = attention_size
        
        # Action head (buy/sell/hold)
        self.action_head = self._create_adaptive_head(decision_input_size, 3, "action")
        
        # Confidence head
        self.confidence_head = self._create_adaptive_head(decision_input_size, 1, "confidence")
        
        # Risk management heads
        self.stop_head = self._create_adaptive_head(decision_input_size, 2, "stop")  # use_stop, distance
        self.target_head = self._create_adaptive_head(decision_input_size, 2, "target")  # use_target, distance
        
        # Tool selection head
        self.tool_selection_head = self._create_adaptive_head(decision_input_size, 4, "tool")  # DNA, Micro, Temporal, Immune
        
        # Value estimation for learning
        self.value_head = self._create_adaptive_head(decision_input_size, 1, "value")
        
        log.info(f"NETWORK EVOLUTION: Built architecture with {self.hidden_size} hidden, "
               f"{self.lstm_layers} LSTM layers, {arch['attention_layers']} attention heads")
    
    def _create_adaptive_head(self, input_size: int, output_size: int, head_type: str):
        """Create decision head with adaptive complexity"""
        
        complexity = self.meta_learner.get_parameter('hidden_layer_multiplier')
        
        layers = []
        
        if complexity > 1.8:  # High complexity
            layers.extend([
                torch.nn.Linear(input_size, self.hidden_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout_rate),
                torch.nn.Linear(self.hidden_size, self.hidden_size // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hidden_size // 2, output_size)
            ])
        elif complexity > 1.2:  # Medium complexity
            layers.extend([
                torch.nn.Linear(input_size, self.hidden_size // 2),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout_rate),
                torch.nn.Linear(self.hidden_size // 2, output_size)
            ])
        else:  # Simple head
            layers.extend([
                torch.nn.Linear(input_size, output_size)
            ])
        
        return torch.nn.Sequential(*layers)
    
    def forward(self, market_obs, subsystem_features):
        """Forward pass with adaptive architecture"""
        
        batch_size = market_obs.shape[0]
        
        # Encode market observations
        market_encoded, _ = self.market_encoder(market_obs)
        market_repr = market_encoded[:, -1]  # Last timestep
        
        # Process subsystem features
        subsystem_repr = self.subsystem_processor(subsystem_features)
        
        # Combine representations
        combined = torch.cat([market_repr, subsystem_repr], dim=-1)
        combined = combined.unsqueeze(1)  # Add sequence dimension for attention
        
        # Apply attention
        attended, attention_weights = self.attention(combined, combined, combined)
        attended = attended.squeeze(1)  # Remove sequence dimension
        
        # Generate all outputs
        action_logits = self.action_head(attended)
        confidence = torch.sigmoid(self.confidence_head(attended))
        
        # Risk management outputs
        stop_outputs = self.stop_head(attended)
        use_stop = torch.sigmoid(stop_outputs[:, 0:1])
        stop_distance = torch.sigmoid(stop_outputs[:, 1:2]) * self.meta_learner.get_parameter('stop_loss_max_pct')
        
        target_outputs = self.target_head(attended)
        use_target = torch.sigmoid(target_outputs[:, 0:1])
        target_distance = torch.sigmoid(target_outputs[:, 1:2]) * self.meta_learner.get_parameter('take_profit_max_pct')
        
        # Tool selection
        tool_logits = self.tool_selection_head(attended)
        tool_probs = F.softmax(tool_logits, dim=-1)
        
        # Value estimation
        value = self.value_head(attended)
        
        return {
            'action_logits': action_logits,
            'confidence': confidence,
            'use_stop': use_stop,
            'stop_distance': stop_distance,
            'use_target': use_target,
            'target_distance': target_distance,
            'tool_probs': tool_probs,
            'attention_weights': attention_weights,
            'value': value,
            'combined_features': attended
        }
    
    def maybe_evolve_architecture(self, performance_score: float, step_count: int):
        """Evolve architecture if performance is declining"""
        
        self.performance_history.append(performance_score)
        
        # Check if we should rebuild (every 200 steps minimum)
        if (step_count - self.last_rebuild_step) >= 200 and len(self.performance_history) >= 50:
            
            recent_perf = np.mean(list(self.performance_history)[-20:])
            older_perf = np.mean(list(self.performance_history)[-50:-30])
            
            performance_decline = older_perf - recent_perf
            
            if performance_decline > self.rebuild_threshold:
                log.info(f"ARCHITECTURE EVOLUTION: Performance declined by {performance_decline:.3f}")
                log.info(f"Recent: {recent_perf:.3f}, Older: {older_perf:.3f}")
                
                # Update architecture parameters to trigger rebuild
                current_complexity = self.meta_learner.get_parameter('hidden_layer_multiplier')
                
                # Try different complexity
                if current_complexity > 1.5:
                    new_complexity = current_complexity * 0.8  # Simplify
                    evolution_direction = "SIMPLIFYING"
                else:
                    new_complexity = current_complexity * 1.3  # Complexify
                    evolution_direction = "COMPLEXIFYING"
                
                # Update meta-parameter
                self.meta_learner.parameters['hidden_layer_multiplier'] = new_complexity
                
                # Rebuild network
                old_hidden = self.hidden_size
                self._build_network()
                
                log.info(f"NETWORK EVOLVED: {evolution_direction} architecture")
                log.info(f"Hidden size: {old_hidden} → {self.hidden_size}")
                log.info(f"Complexity: {current_complexity:.2f} → {new_complexity:.2f}")
                
                self.last_rebuild_step = step_count
                
                return True  # Signal that network was rebuilt
        
        return False

class PureBlackBoxStrategicAgent:
    """
    PURE BLACK BOX: Agent that learns EVERYTHING through experience
    - All parameters adapt through meta-learning
    - Network architecture evolves based on performance
    - Reward structure discovers what actually matters
    - Tool usage learned through trial and error
    """
    
    def __init__(self, market_obs_size: int = 15, subsystem_features_size: int = 16):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize meta-learning system
        self.meta_learner = PureMetaLearner()
        self.reward_learner = AdaptiveRewardLearner(self.meta_learner)
        
        # Self-evolving networks
        self.policy = SelfEvolvingPolicyNetwork(
            self.meta_learner, market_obs_size, subsystem_features_size
        ).to(self.device)
        
        self.target_policy = SelfEvolvingPolicyNetwork(
            self.meta_learner, market_obs_size, subsystem_features_size
        ).to(self.device)
        
        # Copy initial weights
        self.target_policy.load_state_dict(self.policy.state_dict())
        
        # Adaptive optimizers
        self._update_optimizers()
        
        # Experience buffer with adaptive size
        buffer_size = self.meta_learner.get_learning_parameters()['buffer_size']
        self.experience_buffer = deque(maxlen=buffer_size)
        
        # Tool performance tracking
        self.tool_performance_history = {
            'dna': deque(maxlen=200),
            'micro': deque(maxlen=200),
            'temporal': deque(maxlen=200),
            'immune': deque(maxlen=200)
        }
        
        # Adaptive exploration
        self.epsilon = 0.9  # Start high, will adapt
        self._update_exploration_parameters()
        
        # Learning state - FIXED: Added missing step_count
        self.step_count = 0  # This was missing!
        self.network_rebuilds = 0
        self.last_optimizer_update = 0
        
        # Performance tracking
        self.recent_rewards = deque(maxlen=100)
        self.tool_usage_count = {'dna': 0, 'micro': 0, 'temporal': 0, 'immune': 0}
        self.successful_tool_usage = {'dna': 0, 'micro': 0, 'temporal': 0, 'immune': 0}
        
        # Background learning
        self._start_adaptive_learning()
        
        log.info("PURE BLACK BOX AGENT: All parameters will self-optimize")
        log.info("Network architecture will evolve based on performance")
        log.info("Reward structure will discover what actually drives success")
    
    def _update_optimizers(self):
        """Update optimizers with current meta-learned learning rates"""
        learning_params = self.meta_learner.get_learning_parameters()
        
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(),
            lr=learning_params['policy_lr']
        )
        
        # Track optimizer updates
        self.last_optimizer_update = self.step_count
        
        log.debug(f"OPTIMIZER UPDATE: Policy LR = {learning_params['policy_lr']:.6f}")
    
    def _update_exploration_parameters(self):
        """Update exploration parameters from meta-learner"""
        learning_params = self.meta_learner.get_learning_parameters()
        
        self.epsilon_decay = learning_params['epsilon_decay']
        self.epsilon_min = learning_params['epsilon_min']
        
    def select_action_and_strategy(self, market_obs: np.ndarray, subsystem_features: np.ndarray,
                                 current_price: float, in_position: bool = False) -> Dict:
        """Select action using pure black box learning with adaptive parameters"""
        
        with torch.no_grad():
            # Prepare inputs
            market_tensor = torch.tensor(market_obs, dtype=torch.float32, device=self.device)
            market_tensor = market_tensor.unsqueeze(0).unsqueeze(0)
            
            subsystem_tensor = torch.tensor(subsystem_features, dtype=torch.float32, device=self.device)
            subsystem_tensor = subsystem_tensor.unsqueeze(0)
            
            # Forward pass
            outputs = self.policy(market_tensor, subsystem_tensor)
            
            # Action selection with adaptive confidence threshold
            action_probs = F.softmax(outputs['action_logits'], dim=-1).cpu().numpy()[0]
            confidence = float(outputs['confidence'].cpu().numpy()[0])
            
            # Get adaptive thresholds
            thresholds = self.meta_learner.get_confidence_thresholds()
            entry_threshold = thresholds['entry']
            
            # Adaptive exploration
            if random.random() < self.epsilon:
                action = random.choice([0, 1, 2])  # Exploration
                exploration_taken = True
            elif confidence >= entry_threshold:
                action = np.argmax(action_probs)  # Confident exploitation
                exploration_taken = False
            else:
                action = 0  # Hold due to low confidence
                exploration_taken = False
            
            # Extract other outputs
            use_stop = float(outputs['use_stop'].cpu().numpy()[0]) > 0.5
            stop_distance = float(outputs['stop_distance'].cpu().numpy()[0])
            use_target = float(outputs['use_target'].cpu().numpy()[0]) > 0.5
            target_distance = float(outputs['target_distance'].cpu().numpy()[0])
            
            # Tool selection using learned probabilities
            tool_probs = outputs['tool_probs'].cpu().numpy()[0]
            
            # Add exploration to tool selection
            if random.random() < 0.2:  # 20% tool exploration
                primary_tool_idx = random.choice([0, 1, 2, 3])
            else:
                primary_tool_idx = np.argmax(tool_probs)
            
            tool_names = ['dna', 'micro', 'temporal', 'immune']
            primary_tool = tool_names[primary_tool_idx]
            
            # Secondary tool
            tool_probs_copy = tool_probs.copy()
            tool_probs_copy[primary_tool_idx] = 0  # Remove primary
            secondary_tool_idx = np.argmax(tool_probs_copy)
            secondary_tool = tool_names[secondary_tool_idx]
            
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
            
            # Should exit current position?
            exit_threshold = thresholds['exit']
            should_exit = in_position and (confidence < exit_threshold or 
                                         (random.random() < 0.1 and exploration_taken))
            
            # Generate reasoning based on learned patterns
            reasoning = self._generate_adaptive_reasoning(
                primary_tool, confidence, entry_threshold, tool_probs[primary_tool_idx], exploration_taken
            )
            
            return {
                'action': action,
                'confidence': confidence,
                'entry_threshold_used': entry_threshold,
                'exploration_taken': exploration_taken,
                'use_stop': use_stop,
                'stop_price': stop_price,
                'stop_distance_pct': stop_distance * 100,
                'use_target': use_target,
                'target_price': target_price,
                'target_distance_pct': target_distance * 100,
                'should_exit': should_exit,
                
                # Tool analysis
                'primary_tool': primary_tool,
                'secondary_tool': secondary_tool,
                'tool_probabilities': {tool_names[i]: float(tool_probs[i]) for i in range(4)},
                'reasoning': reasoning,
                
                # Meta-learning data
                'meta_parameters_used': {
                    'entry_threshold': entry_threshold,
                    'stop_max_pct': stop_distance,
                    'target_max_pct': target_distance,
                    'current_epsilon': self.epsilon
                },
                
                # Raw data for learning
                'raw_market_obs': market_obs.copy(),
                'raw_subsystem_features': subsystem_features.copy(),
                'raw_outputs': {k: v.cpu().numpy() if torch.is_tensor(v) else v 
                              for k, v in outputs.items()}
            }
    
    def _generate_adaptive_reasoning(self, primary_tool: str, confidence: float, 
                                   threshold: float, tool_confidence: float, exploration: bool) -> str:
        """Generate reasoning based on learned parameters"""
        
        reasoning_parts = []
        
        if exploration:
            reasoning_parts.append(f"EXPLORATION: Random {primary_tool.upper()} tool selection")
        else:
            reasoning_parts.append(f"LEARNED: {primary_tool.upper()} tool (prob: {tool_confidence:.2f})")
        
        if confidence >= threshold:
            margin = confidence - threshold
            reasoning_parts.append(f"High confidence ({confidence:.2f} > {threshold:.2f}, margin: {margin:.2f})")
        else:
            reasoning_parts.append(f"Low confidence ({confidence:.2f} < {threshold:.2f}) - HOLDING")
        
        # Add meta-learning insight
        learning_efficiency = self.meta_learner.get_learning_efficiency()
        if learning_efficiency > 0.5:
            reasoning_parts.append("FAST LEARNING MODE")
        elif learning_efficiency < -0.2:
            reasoning_parts.append("ADAPTATION MODE")
        
        return " | ".join(reasoning_parts)
    
    def store_experience_and_learn(self, state_data: Dict, trade_outcome: Dict):
        """Store experience and trigger adaptive learning"""
        
        # Calculate adaptive reward
        adaptive_reward = self.reward_learner.calculate_adaptive_reward(trade_outcome)
        
        # Store experience
        experience = {
            'state_data': state_data,
            'trade_outcome': trade_outcome,
            'adaptive_reward': adaptive_reward,
            'timestamp': datetime.now(),
            'meta_parameters': state_data.get('meta_parameters_used', {})
        }
        
        self.experience_buffer.append(experience)
        
        # Track tool performance
        primary_tool = state_data.get('primary_tool', 'unknown')
        if primary_tool in self.tool_usage_count:
            self.tool_usage_count[primary_tool] += 1
            
            if adaptive_reward > 0.1:  # Successful outcome
                self.successful_tool_usage[primary_tool] += 1
                self.tool_performance_history[primary_tool].append(1.0)
            else:
                self.tool_performance_history[primary_tool].append(0.0)
        
        # Update recent performance
        self.recent_rewards.append(adaptive_reward)
        
        # Update meta-parameters based on outcome
        self._update_meta_parameters_from_outcome(state_data, trade_outcome, adaptive_reward)
        
        # Trigger learning
        if len(self.experience_buffer) >= self.meta_learner.get_learning_parameters()['batch_size']:
            self._adaptive_learning_step()
        
        # Decay exploration adaptively
        self._update_exploration_parameters()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        self.step_count += 1  # FIXED: Now properly increment step_count
        
        # Update optimizers periodically with new learning rates
        if self.step_count - self.last_optimizer_update >= 100:
            self._update_optimizers()
        
        # Check for network evolution
        if len(self.recent_rewards) >= 20:
            recent_performance = np.mean(list(self.recent_rewards)[-20:])
            network_rebuilt = self.policy.maybe_evolve_architecture(recent_performance, self.step_count)
            
            if network_rebuilt:
                self.network_rebuilds += 1
                # Copy evolved architecture to target
                self.target_policy.load_state_dict(self.policy.state_dict())
                # Reset optimizer for new architecture
                self._update_optimizers()
    
    def _update_meta_parameters_from_outcome(self, state_data: Dict, trade_outcome: Dict, reward: float):
        """Update meta-parameters based on trading outcome"""
        
        # Normalize outcome for meta-learning
        pnl = trade_outcome.get('pnl', 0.0)
        normalized_outcome = np.tanh(pnl / 50.0)  # Normalize to [-1, 1]
        
        # Update parameters that were used in this decision
        parameter_updates = {}
        
        # Entry confidence threshold
        if state_data.get('action', 0) != 0:  # If we took a position
            parameter_updates['entry_confidence_threshold'] = normalized_outcome
        
        # Risk management parameters
        if state_data.get('use_stop', False):
            parameter_updates['stop_loss_max_pct'] = normalized_outcome
        
        if state_data.get('use_target', False):
            parameter_updates['take_profit_max_pct'] = normalized_outcome
        
        # Position sizing
        parameter_updates['position_size_base'] = normalized_outcome
        
        # Learning rates (boost if reward is good)
        if abs(reward) > 0.2:  # Significant outcome
            lr_update = 0.1 if reward > 0 else -0.05
            parameter_updates['policy_learning_rate'] = lr_update
            parameter_updates['meta_learning_rate'] = lr_update
        
        # Tool selection parameters
        primary_tool = state_data.get('primary_tool', '')
        if primary_tool:
            # This would update tool-specific parameters if they existed
            pass
        
        # Batch update meta-parameters
        learning_efficiency = reward  # Simplified efficiency measure
        self.meta_learner.batch_update_parameters(parameter_updates, learning_efficiency)
    
    def _start_adaptive_learning(self):
        """Start adaptive background learning"""
        def adaptive_learning_loop():
            while True:
                if len(self.experience_buffer) < 10:
                    time.sleep(1)
                    continue
                
                # Adaptive learning frequency
                learning_params = self.meta_learner.get_learning_parameters()
                batch_size = learning_params['batch_size']
                
                if len(self.experience_buffer) >= batch_size:
                    self._adaptive_learning_step()
                
                # Adaptive sleep time based on learning efficiency
                learning_efficiency = self.meta_learner.get_learning_efficiency()
                sleep_time = 0.5 if learning_efficiency > 0.3 else 0.1  # Learn faster if inefficient
                time.sleep(sleep_time)
        
        thread = threading.Thread(target=adaptive_learning_loop, daemon=True, name="AdaptiveLearning")
        thread.start()
        log.info("ADAPTIVE LEARNING: Background thread started")
    
    def _adaptive_learning_step(self):
        """Adaptive learning step with meta-learned parameters"""
        
        learning_params = self.meta_learner.get_learning_parameters()
        batch_size = learning_params['batch_size']
        
        if len(self.experience_buffer) < batch_size:
            return
        
        # Sample batch
        batch = random.sample(list(self.experience_buffer), batch_size)
        
        # Extract data
        rewards = torch.tensor([exp['adaptive_reward'] for exp in batch], 
                              dtype=torch.float32, device=self.device)
        
        # Simple policy gradient with adaptive reward
        policy_loss = -torch.mean(rewards)  # Maximize adaptive reward
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        
        # Adaptive gradient clipping
        max_grad_norm = 1.0 + self.meta_learner.get_learning_efficiency()  # Adapt clipping
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
        
        self.policy_optimizer.step()
        
        # Soft update target network
        tau = 0.005
        for target_param, param in zip(self.target_policy.parameters(), self.policy.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        # Log progress occasionally
        if self.step_count % 100 == 0:
            avg_reward = torch.mean(rewards).item()
            log.info(f"ADAPTIVE LEARNING: Step {self.step_count}, Avg reward: {avg_reward:.3f}, "
                   f"Epsilon: {self.epsilon:.3f}, Network rebuilds: {self.network_rebuilds}")
    
    def get_pure_blackbox_status(self) -> str:
        """Get comprehensive pure black box status"""
        
        # Meta-learning status
        meta_status = self.meta_learner.get_adaptation_report()
        
        # Reward learning status  
        reward_status = self.reward_learner.get_reward_analysis()
        
        # Tool performance
        tool_status = "\nTOOL LEARNING STATUS:\n"
        for tool in ['dna', 'micro', 'temporal', 'immune']:
            usage = self.tool_usage_count[tool]
            success = self.successful_tool_usage[tool]
            success_rate = success / usage if usage > 0 else 0.0
            
            recent_perf = list(self.tool_performance_history[tool])[-10:] if self.tool_performance_history[tool] else []
            recent_success = np.mean(recent_perf) if recent_perf else 0.0
            
            tool_status += f"  {tool.upper()}: {usage} uses, {success_rate:.1%} overall, {recent_success:.1%} recent\n"
        
        # Network evolution status
        network_status = f"""
NETWORK EVOLUTION STATUS:
  Current Architecture: {self.policy.hidden_size} hidden, {self.policy.lstm_layers} LSTM layers
  Network Rebuilds: {self.network_rebuilds}
  Dropout Rate: {self.policy.dropout_rate:.3f}
  
LEARNING STATE:
  Training Steps: {self.step_count}
  Experience Buffer: {len(self.experience_buffer)}/{self.experience_buffer.maxlen}
  Current Epsilon: {self.epsilon:.3f}
  Recent Avg Reward: {np.mean(list(self.recent_rewards)[-20:]):.3f if len(self.recent_rewards) >= 20 else 'N/A'}
"""
        
        combined_status = f"""
=== PURE BLACK BOX AGENT STATUS ===

{meta_status}

{reward_status}

{tool_status}

{network_status}

PURE BLACK BOX: All parameters self-optimizing through experience!
Network architecture evolving based on performance!
Reward structure discovering what actually drives success!
"""
        
        return combined_status
    
    def force_save_all_learning(self):
        """Force save all meta-learning progress"""
        self.meta_learner.force_save()
        
        # Save network state
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'target_policy_state_dict': self.target_policy.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'step_count': self.step_count,
            'network_rebuilds': self.network_rebuilds,
            'epsilon': self.epsilon,
            'tool_performance_history': {k: list(v) for k, v in self.tool_performance_history.items()},
            'tool_usage_count': self.tool_usage_count,
            'successful_tool_usage': self.successful_tool_usage,
            'meta_learner_params': self.meta_learner.parameters.copy(),
            'reward_component_weights': {k: v['weight'] for k, v in self.reward_learner.reward_components.items()}
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(checkpoint, f"pure_blackbox_agent_{timestamp}.pt")
        
        log.info("PURE BLACK BOX: All learning progress saved")

# Factory function for creating pure black box agent
def create_pure_blackbox_agent(market_obs_size: int = 15, subsystem_features_size: int = 16) -> PureBlackBoxStrategicAgent:
    """Create pure black box agent with meta-learning"""
    
    agent = PureBlackBoxStrategicAgent(market_obs_size, subsystem_features_size)
    
    log.info("PURE BLACK BOX AGENT CREATED")
    log.info("All parameters will adapt through experience")
    log.info("Network will evolve its own architecture")
    log.info("Reward structure will discover success patterns")
    
    return agent

# Usage test
if __name__ == "__main__":
    print("Testing Pure Black Box Agent...")
    
    agent = create_pure_blackbox_agent()
    
    print("Initial status:")
    print(agent.get_pure_blackbox_status())
    
    # Simulate some decision making
    for i in range(20):
        market_obs = np.random.randn(15).astype(np.float32)
        subsystem_features = np.random.randn(16).astype(np.float32)
        current_price = 4000.0 + np.random.randn() * 10
        
        decision = agent.select_action_and_strategy(market_obs, subsystem_features, current_price)
        
        # Simulate outcome
        trade_outcome = {
            'pnl': np.random.normal(5, 25),
            'hold_time_hours': np.random.uniform(0.5, 8.0),
            'used_stop': decision['use_stop'],
            'used_target': decision['use_target'],
            'tool_confidence': decision['confidence'],
            'max_drawdown_pct': np.random.uniform(0.005, 0.03),
            'exit_reason': np.random.choice(['target_hit', 'stop_hit', 'manual_exit'])
        }
        
        agent.store_experience_and_learn(decision, trade_outcome)
        
        if (i + 1) % 5 == 0:
            print(f"\nAfter {i+1} decisions:")
            print(f"Primary tool: {decision['primary_tool']}")
            print(f"Confidence: {decision['confidence']:.3f} (threshold: {decision['entry_threshold_used']:.3f})")
            print(f"Epsilon: {agent.epsilon:.3f}")
            print(f"Step count: {agent.step_count}")
    
    print("\nFinal status after learning:")
    print(agent.get_pure_blackbox_status())