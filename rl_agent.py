# File 2: Update rl_agent.py - Complete black box agent

import threading
import time
import random
from collections import deque
from typing import Tuple, Dict
import numpy as np
import torch
import torch.nn.functional as F
from policy_network import BlackBoxPolicyNetwork

class BlackBoxRLAgent:
    """
    COMPLETE black box RL agent that learns:
    - When to trade
    - How much to trade
    - Whether to use stops/targets
    - What levels to use
    - When to exit
    """
    
    def __init__(self, obs_size: int = 15):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Black box networks
        self.policy = BlackBoxPolicyNetwork(obs_size).to(self.device)
        self.target = BlackBoxPolicyNetwork(obs_size).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4)
        self.replay_buffer = deque(maxlen=100000)
        
        # Learning parameters
        self.gamma = 0.99
        self.sync_every = 2000
        self.step_count = 0
        
        # Start background learning
        threading.Thread(target=self._continuous_learning, daemon=True).start()
        
        log.info("BLACK BOX RL Agent initialized - learning EVERYTHING")
    
    def make_decision(self, obs: np.ndarray, current_price: float, in_position: bool = False) -> Dict:
        """
        AI makes COMPLETE trading decision - no human rules
        """
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=self.device)
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, obs_size)
            
            outputs = self.policy(x)
            
            # Extract decisions
            action_probs = torch.softmax(outputs['action_logits'], dim=-1).cpu().numpy()[0]
            action = np.random.choice(3, p=action_probs)
            
            position_size = float(outputs['position_size'].cpu().numpy()[0])
            use_stop_prob = float(outputs['use_stop'].cpu().numpy()[0])
            stop_distance = float(outputs['stop_distance'].cpu().numpy()[0])
            use_target_prob = float(outputs['use_target'].cpu().numpy()[0])
            target_distance = float(outputs['target_distance'].cpu().numpy()[0])
            exit_confidence = float(outputs['exit_confidence'].cpu().numpy()[0])
            overall_confidence = float(outputs['overall_confidence'].cpu().numpy()[0])
            
            # AI decides whether to use stops/targets (probabilistic)
            use_stop = random.random() < use_stop_prob
            use_target = random.random() < use_target_prob
            
            # Convert to actual prices
            stop_price = None
            target_price = None
            
            if use_stop and action != 0:
                if action == 1:  # Long position
                    stop_price = current_price * (1 - stop_distance)
                else:  # Short position
                    stop_price = current_price * (1 + stop_distance)
            
            if use_target and action != 0:
                if action == 1:  # Long position
                    target_price = current_price * (1 + target_distance)
                else:  # Short position
                    target_price = current_price * (1 - target_distance)
            
            # Exit decision for current position
            should_exit = in_position and (exit_confidence > 0.6)
            
            return {
                'action': action,  # 0=hold, 1=buy, 2=sell
                'position_size': position_size,
                'use_stop': use_stop,
                'stop_price': stop_price,
                'stop_distance_pct': stop_distance * 100,
                'use_target': use_target,
                'target_price': target_price,
                'target_distance_pct': target_distance * 100,
                'should_exit': should_exit,
                'overall_confidence': overall_confidence,
                'reasoning': f"AI_learned_decision_conf_{overall_confidence:.3f}"
            }
    
    def store_experience(self, state, action_dict, reward, next_state, done):
        """Store complete experience for learning"""
        self.replay_buffer.append((state, action_dict, reward, next_state, done))
    
    def _continuous_learning(self):
        """Background learning loop"""
        while True:
            if len(self.replay_buffer) < 1000:
                time.sleep(1)
                continue
            
            # Sample and train
            self._train_step()
            time.sleep(0.1)  # Don't hog CPU
    
    def _train_step(self):
        """Train the black box network"""
        if len(self.replay_buffer) < 64:
            return
        
        # Sample batch
        batch = random.sample(self.replay_buffer, 64)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        
        # Current Q values
        current_outputs = self.policy(states.unsqueeze(1))
        
        # Simple Q-learning on action head for now
        # (You could make this more sophisticated)
        action_indices = torch.tensor([a['action'] for a in actions], device=self.device)
        current_q = current_outputs['action_logits'].gather(1, action_indices.unsqueeze(1)).squeeze()
        
        # Target Q values
        with torch.no_grad():
            next_outputs = self.target(next_states.unsqueeze(1))
            next_q = next_outputs['action_logits'].max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Loss and update
        loss = F.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.step_count += 1
        
        # Sync target network
        if self.step_count % self.sync_every == 0:
            self.target.load_state_dict(self.policy.state_dict())
            log.info(f"BLACK BOX: Target network synced at step {self.step_count}")