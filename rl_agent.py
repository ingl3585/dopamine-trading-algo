# rl_agent.py - ENHANCED: Strategic subsystem tool learning agent

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
from policy_network import EnhancedPolicyNetwork, ValueNetwork, SubsystemInterpreter

log = logging.getLogger(__name__)

class StrategicToolLearningAgent:
    """
    Enhanced RL agent that learns to strategically use your existing subsystems as tools
    
    Key Learning Objectives:
    1. When to trust each subsystem (DNA vs Micro vs Temporal vs Immune)
    2. How to combine subsystems for better decisions
    3. Which tools work best in different market regimes
    4. Optimal risk management for each tool type
    5. Exit timing based on tool confidence
    """
    
    def __init__(self, market_obs_size: int = 15, subsystem_features_size: int = 16):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Enhanced networks
        self.policy = EnhancedPolicyNetwork(market_obs_size, subsystem_features_size).to(self.device)
        self.value = ValueNetwork(market_obs_size, subsystem_features_size).to(self.device)
        self.target_policy = EnhancedPolicyNetwork(market_obs_size, subsystem_features_size).to(self.device)
        self.target_value = ValueNetwork(market_obs_size, subsystem_features_size).to(self.device)
        
        # Copy weights to target networks
        self.target_policy.load_state_dict(self.policy.state_dict())
        self.target_value.load_state_dict(self.value.state_dict())
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=3e-4)
        
        # Experience buffer
        self.experience_buffer = deque(maxlen=50000)
        
        # Tool performance tracking
        self.tool_performance_history = {
            'dna': deque(maxlen=200),
            'micro': deque(maxlen=200),
            'temporal': deque(maxlen=200),
            'immune': deque(maxlen=200)
        }
        
        # Tool combination tracking
        self.combination_performance = {
            'dna_micro': deque(maxlen=100),
            'dna_temporal': deque(maxlen=100),
            'dna_immune': deque(maxlen=100),
            'micro_temporal': deque(maxlen=100),
            'micro_immune': deque(maxlen=100),
            'temporal_immune': deque(maxlen=100)
        }
        
        # Learning parameters
        self.gamma = 0.99
        self.tau = 0.005  # Soft update rate
        self.sync_every = 1000
        self.step_count = 0
        self.learning_started = False
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.1
        
        # Performance metrics
        self.recent_rewards = deque(maxlen=100)
        self.tool_usage_count = {'dna': 0, 'micro': 0, 'temporal': 0, 'immune': 0}
        self.successful_tool_usage = {'dna': 0, 'micro': 0, 'temporal': 0, 'immune': 0}
        
        # Start background learning
        self._start_background_learning()
        
        log.info("Strategic Tool Learning Agent initialized")
        log.info("Learning objectives: Tool selection, combinations, regime adaptation, risk management")
    
    def select_action_and_strategy(self, market_obs: np.ndarray, subsystem_features: np.ndarray, 
                                 current_price: float, in_position: bool = False) -> Dict:
        """
        Select action and complete trading strategy using learned tool usage
        """
        with torch.no_grad():
            # Prepare inputs
            market_tensor = torch.tensor(market_obs, dtype=torch.float32, device=self.device)
            market_tensor = market_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, obs_size)
            
            subsystem_tensor = torch.tensor(subsystem_features, dtype=torch.float32, device=self.device)
            subsystem_tensor = subsystem_tensor.unsqueeze(0)  # (1, features_size)
            
            # Get policy outputs
            outputs = self.policy(market_tensor, subsystem_tensor)
            
            # Action selection with exploration
            action_probs = F.softmax(outputs['action_logits'], dim=-1).cpu().numpy()[0]
            
            if random.random() < self.epsilon and self.learning_started:
                action = random.choice([0, 1, 2])  # Exploration
            else:
                action = np.argmax(action_probs)  # Exploitation
            
            # Extract all decision components
            confidence = float(outputs['overall_confidence'].cpu().numpy()[0])
            use_stop_prob = float(outputs['use_stop'].cpu().numpy()[0])
            stop_distance = float(outputs['stop_distance'].cpu().numpy()[0])
            use_target_prob = float(outputs['use_target'].cpu().numpy()[0])
            target_distance = float(outputs['target_distance'].cpu().numpy()[0])
            exit_confidence = float(outputs['exit_confidence'].cpu().numpy()[0])
            
            # Tool analysis
            tool_trust = outputs['tool_trust'].cpu().numpy()[0]
            tool_combinations = outputs['tool_combinations'].cpu().numpy()[0]
            market_regime = outputs['market_regime'].cpu().numpy()[0]
            attention_weights = outputs['attention_weights'].cpu().numpy()[0]
            
            # Risk management decisions
            use_stop = random.random() < use_stop_prob
            use_target = random.random() < use_target_prob
            
            # Calculate actual prices
            stop_price = None
            target_price = None
            
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
            
            # Determine primary and secondary tools
            tool_names = ['dna', 'micro', 'temporal', 'immune']
            tool_ranking = sorted(enumerate(tool_trust), key=lambda x: x[1], reverse=True)
            primary_tool = tool_names[tool_ranking[0][0]]
            secondary_tool = tool_names[tool_ranking[1][0]] if len(tool_ranking) > 1 else None
            
            # Market regime interpretation
            regime_names = ['trending', 'volatile', 'sideways', 'reversal']
            market_regime_name = regime_names[np.argmax(market_regime)]
            
            # Should exit current position?
            should_exit = in_position and (exit_confidence > 0.7)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(primary_tool, secondary_tool, market_regime_name, 
                                               confidence, tool_trust[tool_ranking[0][0]])
            
            return {
                'action': action,
                'confidence': confidence,
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
                'tool_trust': {tool_names[i]: float(tool_trust[i]) for i in range(4)},
                'tool_combinations': tool_combinations,
                'market_regime': market_regime_name,
                'attention_weights': attention_weights,
                'reasoning': reasoning,
                
                # Raw data for experience storage
                'raw_market_obs': market_obs.copy(),
                'raw_subsystem_features': subsystem_features.copy(),
                'raw_outputs': {k: v.cpu().numpy() if torch.is_tensor(v) else v 
                              for k, v in outputs.items()}
            }
    
    def _generate_reasoning(self, primary_tool: str, secondary_tool: str, 
                          regime: str, confidence: float, primary_trust: float) -> str:
        """Generate human-readable reasoning for the decision"""
        
        reasoning_parts = []
        
        # Primary tool reasoning
        reasoning_parts.append(f"Primary tool: {primary_tool.upper()} (trust: {primary_trust:.2f})")
        
        # Secondary tool support
        if secondary_tool and primary_trust < 0.8:
            reasoning_parts.append(f"Supported by {secondary_tool.upper()}")
        
        # Market regime context
        reasoning_parts.append(f"Market regime: {regime}")
        
        # Confidence interpretation
        if confidence > 0.8:
            reasoning_parts.append("High confidence decision")
        elif confidence > 0.6:
            reasoning_parts.append("Moderate confidence")
        else:
            reasoning_parts.append("Low confidence - cautious approach")
        
        return " | ".join(reasoning_parts)
    
    def store_experience(self, state_data: Dict, reward: float, next_market_obs: np.ndarray,
                        next_subsystem_features: np.ndarray, done: bool):
        """Store experience with tool performance tracking"""
        
        experience = {
            'market_obs': state_data['raw_market_obs'],
            'subsystem_features': state_data['raw_subsystem_features'],
            'action': state_data['action'],
            'reward': reward,
            'next_market_obs': next_market_obs,
            'next_subsystem_features': next_subsystem_features,
            'done': done,
            'tool_trust': state_data['tool_trust'],
            'primary_tool': state_data['primary_tool'],
            'market_regime': state_data['market_regime']
        }
        
        self.experience_buffer.append(experience)
        
        # Track tool performance
        primary_tool = state_data['primary_tool']
        self.tool_usage_count[primary_tool] += 1
        
        # Record tool success/failure
        if reward > 0.01:  # Successful trade
            self.successful_tool_usage[primary_tool] += 1
            self.tool_performance_history[primary_tool].append(1.0)
        else:
            self.tool_performance_history[primary_tool].append(0.0)
        
        # Track tool combinations
        if state_data.get('secondary_tool'):
            combo_key = f"{primary_tool}_{state_data['secondary_tool']}"
            if combo_key in self.combination_performance:
                success = 1.0 if reward > 0.01 else 0.0
                self.combination_performance[combo_key].append(success)
        
        # Update recent performance
        self.recent_rewards.append(reward)
        
        # Decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def _start_background_learning(self):
        """Start background learning thread"""
        def learning_loop():
            while True:
                if len(self.experience_buffer) < 500:
                    time.sleep(1)
                    continue
                
                if not self.learning_started:
                    self.learning_started = True
                    log.info("Tool learning started - sufficient experience collected")
                
                self._train_step()
                time.sleep(0.1)
        
        thread = threading.Thread(target=learning_loop, daemon=True)
        thread.start()
        log.info("Background tool learning thread started")
    
    def _train_step(self):
        """Training step with focus on tool usage learning"""
        if len(self.experience_buffer) < 64:
            return
        
        # Sample batch
        batch = random.sample(self.experience_buffer, 64)
        
        # Prepare tensors
        market_obs = torch.stack([
            torch.tensor(exp['market_obs'], dtype=torch.float32)
            for exp in batch
        ]).unsqueeze(1).to(self.device)
        
        subsystem_features = torch.stack([
            torch.tensor(exp['subsystem_features'], dtype=torch.float32)
            for exp in batch
        ]).to(self.device)
        
        actions = torch.tensor([exp['action'] for exp in batch], 
                              dtype=torch.long, device=self.device)
        rewards = torch.tensor([exp['reward'] for exp in batch], 
                              dtype=torch.float32, device=self.device)
        
        next_market_obs = torch.stack([
            torch.tensor(exp['next_market_obs'], dtype=torch.float32)
            for exp in batch
        ]).unsqueeze(1).to(self.device)
        
        next_subsystem_features = torch.stack([
            torch.tensor(exp['next_subsystem_features'], dtype=torch.float32)
            for exp in batch
        ]).to(self.device)
        
        dones = torch.tensor([exp['done'] for exp in batch], 
                           dtype=torch.float32, device=self.device)
        
        # Current values and policy outputs
        current_policy_outputs = self.policy(market_obs, subsystem_features)
        current_values = self.value(market_obs, subsystem_features).squeeze()
        
        # Current action values
        current_action_values = current_policy_outputs['action_logits'].gather(1, actions.unsqueeze(1)).squeeze()
        
        # Target values
        with torch.no_grad():
            next_values = self.target_value(next_market_obs, next_subsystem_features).squeeze()
            target_values = rewards + (1 - dones) * self.gamma * next_values
        
        # Value loss
        value_loss = F.mse_loss(current_values, target_values)
        
        # Policy loss (actor-critic style)
        advantages = (target_values - current_values).detach()
        policy_loss = -torch.mean(current_action_values * advantages)
        
        # Tool trust regularization - encourage diverse tool usage
        tool_trust = current_policy_outputs['tool_trust']
        tool_diversity_loss = -torch.mean(torch.sum(tool_trust * torch.log(tool_trust + 1e-8), dim=1))
        
        # Combined loss
        total_policy_loss = policy_loss + 0.01 * tool_diversity_loss
        
        # Update value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), 1.0)
        self.value_optimizer.step()
        
        # Update policy network
        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_optimizer.step()
        
        self.step_count += 1
        
        # Soft update target networks
        if self.step_count % 10 == 0:
            self._soft_update_targets()
        
        # Hard update every so often
        if self.step_count % self.sync_every == 0:
            self.target_policy.load_state_dict(self.policy.state_dict())
            self.target_value.load_state_dict(self.value.state_dict())
            log.info(f"Tool learning: Networks synced at step {self.step_count}")
            
            # Log learning progress
            self._log_learning_progress()
    
    def _soft_update_targets(self):
        """Soft update of target networks"""
        for target_param, param in zip(self.target_policy.parameters(), self.policy.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_value.parameters(), self.value.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def _log_learning_progress(self):
        """Log learning progress and tool performance"""
        if len(self.recent_rewards) > 10:
            avg_reward = np.mean(list(self.recent_rewards)[-20:])
            log.info(f"Tool Learning Progress: Avg reward: {avg_reward:.4f}, Epsilon: {self.epsilon:.3f}")
            
            # Tool performance summary
            for tool in ['dna', 'micro', 'temporal', 'immune']:
                usage = self.tool_usage_count[tool]
                success = self.successful_tool_usage[tool]
                success_rate = success / usage if usage > 0 else 0.0
                log.info(f"  {tool.upper()}: {usage} uses, {success_rate:.2%} success")
    
    def get_tool_performance_report(self) -> str:
        """Generate comprehensive tool performance report"""
        
        report = f"""
=== STRATEGIC TOOL LEARNING REPORT ===

Learning Status:
- Training Steps: {self.step_count}
- Experience Buffer: {len(self.experience_buffer)} samples
- Learning Active: {self.learning_started}
- Exploration Rate: {self.epsilon:.3f}

Tool Usage & Performance:
"""
        
        for tool in ['dna', 'micro', 'temporal', 'immune']:
            usage = self.tool_usage_count[tool]
            success = self.successful_tool_usage[tool]
            success_rate = success / usage if usage > 0 else 0.0
            
            # Recent performance
            recent_perf = list(self.tool_performance_history[tool])[-20:] if self.tool_performance_history[tool] else []
            recent_success = np.mean(recent_perf) if recent_perf else 0.0
            
            report += f"  {tool.upper()}: {usage} uses, {success_rate:.1%} overall, {recent_success:.1%} recent\n"
        
        report += f"\nTool Combinations Performance:\n"
        for combo, history in self.combination_performance.items():
            if history:
                combo_success = np.mean(list(history))
                report += f"  {combo.replace('_', ' + ').upper()}: {combo_success:.1%} success ({len(history)} uses)\n"
        
        if self.recent_rewards:
            recent_avg = np.mean(list(self.recent_rewards)[-20:])
            report += f"\nRecent Performance: {recent_avg:.4f} avg reward (last 20 trades)\n"
        
        report += f"\nAI is learning optimal tool usage patterns for different market conditions!"
        
        return report
    
    def get_current_tool_preferences(self) -> Dict[str, float]:
        """Get current learned tool preferences"""
        preferences = {}
        
        for tool in ['dna', 'micro', 'temporal', 'immune']:
            usage = self.tool_usage_count[tool]
            success = self.successful_tool_usage[tool]
            
            if usage > 0:
                success_rate = success / usage
                # Weight by usage and success
                preference = (success_rate * 0.7) + (min(usage / 100, 1.0) * 0.3)
            else:
                preference = 0.5  # Neutral
            
            preferences[tool] = preference
        
        return preferences
    
    def save_model(self, filepath: str):
        """Save the trained models and learning state"""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'step_count': self.step_count,
            'epsilon': self.epsilon,
            'tool_usage_count': self.tool_usage_count,
            'successful_tool_usage': self.successful_tool_usage,
            'tool_performance_history': {k: list(v) for k, v in self.tool_performance_history.items()},
            'combination_performance': {k: list(v) for k, v in self.combination_performance.items()}
        }
        
        torch.save(checkpoint, filepath)
        log.info(f"Strategic tool learning model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model and learning state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])
        self.target_policy.load_state_dict(self.policy.state_dict())
        self.target_value.load_state_dict(self.value.state_dict())
        
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        
        self.step_count = checkpoint['step_count']
        self.epsilon = checkpoint['epsilon']
        self.tool_usage_count = checkpoint['tool_usage_count']
        self.successful_tool_usage = checkpoint['successful_tool_usage']
        
        # Restore performance history
        for tool, history in checkpoint['tool_performance_history'].items():
            self.tool_performance_history[tool] = deque(history, maxlen=200)
        
        for combo, history in checkpoint['combination_performance'].items():
            self.combination_performance[combo] = deque(history, maxlen=100)
        
        self.learning_started = True
        log.info(f"Strategic tool learning model loaded from {filepath}")
        log.info(f"Resumed at step {self.step_count} with epsilon {self.epsilon:.3f}")