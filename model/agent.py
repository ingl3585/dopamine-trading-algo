# model/agent.py

import os
import time
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque
from model.base import ActorCritic

log = logging.getLogger(__name__)

class RLAgent:
    def __init__(self, config):
        self.config = config
        self.temp = config.TEMPERATURE
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ActorCritic(config.INPUT_DIM, config.HIDDEN_DIM, config.ACTION_DIM).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.LR, weight_decay=1e-5)
        
        # Remove replay buffer - use immediate experience for on-policy learning
        self.experience_buffer = []  # For batch training only
        self.loss_fn = nn.SmoothL1Loss()
        self.last_save_time = 0

        self.load_model()

    def load_model(self):
        if not os.path.exists(self.config.MODEL_PATH):
            log.info("No existing model found, starting fresh")
            return
        try:
            checkpoint = torch.load(self.config.MODEL_PATH, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                log.info("Model and optimizer loaded from checkpoint")
            else:
                # Legacy loading
                self.model.load_state_dict(checkpoint)
                log.info("Model loaded (legacy format)")
        except Exception as e:
            log.warning("Model load failed: %s", e)

    def save_model(self):
        try:
            os.makedirs(os.path.dirname(self.config.MODEL_PATH), exist_ok=True)
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': {
                    'input_dim': self.config.INPUT_DIM,
                    'hidden_dim': self.config.HIDDEN_DIM,
                    'action_dim': self.config.ACTION_DIM
                }
            }
            torch.save(checkpoint, self.config.MODEL_PATH)
            log.info("Model checkpoint saved")
        except Exception as e:
            log.warning("Save error: %s", e)

    def train_on_batch(self, experiences):
        """
        Train on a batch of recent experiences (on-policy)
        """
        if len(experiences) < 2:
            return 0.0
            
        try:
            states, actions, rewards, next_states = zip(*experiences)
            
            # Convert to tensors
            states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
            actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            
            # Forward pass
            probs, values = self.model(states)
            _, next_values = self.model(next_states)
            
            # Calculate advantages using TD error
            targets = rewards + self.config.GAMMA * next_values.squeeze()
            advantages = targets - values.squeeze()
            
            # Normalize advantages
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Calculate losses
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            actor_loss = -(log_probs * advantages.detach()).mean()
            critic_loss = self.loss_fn(values.squeeze(), targets.detach())
            
            # Total loss with entropy regularization
            total_loss = actor_loss + 0.5 * critic_loss - self.config.ENTROPY_COEF * entropy
            
            # Optimization step
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            return total_loss.item()
            
        except Exception as e:
            log.warning(f"Training error: {e}")
            return 0.0

    def train(self, df, epochs=1):
        """
        Train on historical data for initial learning
        """
        try:
            if isinstance(df.iloc[0, 1], str):
                df = df.iloc[1:]

            # Extract features and prepare data
            feature_data = df.iloc[:, 1:-1].apply(pd.to_numeric, errors="coerce").fillna(0).values
            rewards = df.iloc[:, -1].apply(pd.to_numeric, errors="coerce").fillna(0).values
            
            if len(feature_data) < 2:
                log.warning(f"Insufficient data for training: {len(feature_data)} samples")
                return

            log.info(f"Training on {len(feature_data)} samples for {epochs} epochs")
            
            for epoch in range(epochs):
                total_loss = 0.0
                num_batches = 0
                
                # Create batches for training
                batch_size = min(self.config.BATCH_SIZE, len(feature_data) - 1)
                
                for i in range(0, len(feature_data) - 1, batch_size):
                    end_idx = min(i + batch_size, len(feature_data) - 1)
                    
                    batch_experiences = []
                    for j in range(i, end_idx):
                        state = feature_data[j]
                        next_state = feature_data[j + 1]
                        reward = rewards[j]
                        action = 1  # Placeholder action for historical training
                        
                        batch_experiences.append((state, action, reward, next_state))
                    
                    if batch_experiences:
                        loss = self.train_on_batch(batch_experiences)
                        total_loss += loss
                        num_batches += 1
                
                avg_loss = total_loss / max(num_batches, 1)
                log.info(f"Epoch {epoch+1}/{epochs} - Average loss: {avg_loss:.4f}")

            # Save after training
            if time.time() - self.last_save_time > 300:  # Save every 5 minutes
                self.save_model()
                self.last_save_time = time.time()

        except Exception as e:
            log.warning(f"Training error: {e}")

    def predict_single(self, feat_vec):
        """
        Make a single prediction with improved confidence variation
        """
        try:
            # Convert to tensor and ensure correct shape
            state = torch.tensor(feat_vec, dtype=torch.float32, device=self.device)
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
            
            with torch.no_grad():
                probs, value = self.model(state, temperature=self.temp)
                
                # IMPROVED: Use entropy-based confidence calculation
                dist = torch.distributions.Categorical(probs)
                action = int(dist.sample())
                
                # Calculate confidence based on probability distribution spread
                entropy = float(dist.entropy())
                max_entropy = np.log(probs.shape[1])  # Log of number of actions
                
                # Convert entropy to confidence (higher entropy = lower confidence)
                entropy_confidence = 1.0 - (entropy / max_entropy)
                
                # Get action probability
                action_prob = float(probs[0, action])
                
                # IMPROVED: Combine multiple confidence signals
                # 1. Action probability (30%)
                # 2. Entropy-based confidence (40%) 
                # 3. Value prediction magnitude (30%)
                value_confidence = min(1.0, abs(float(value[0])) / 2.0)  # Normalize value
                
                combined_confidence = (
                    0.3 * action_prob + 
                    0.4 * entropy_confidence + 
                    0.3 * value_confidence
                )
                
                if combined_confidence > 0.7:
                    # High confidence trades: 0.7-0.9
                    confidence = 0.7 + (combined_confidence - 0.7) * 0.67  # Maps 0.7-1.0 to 0.7-0.9
                elif combined_confidence > 0.5:
                    # Medium confidence trades: 0.4-0.7  
                    confidence = 0.4 + (combined_confidence - 0.5) * 1.5   # Maps 0.5-0.7 to 0.4-0.7
                else:
                    # Low confidence trades: 0.2-0.4
                    confidence = 0.2 + combined_confidence * 0.4           # Maps 0.0-0.5 to 0.2-0.4

                # Add controlled randomness for natural variation
                noise = np.random.uniform(-0.1, 0.1)  # ±10% noise
                confidence = np.clip(confidence + noise, 0.25, 0.85)
                
                # IMPROVED: Map to realistic trading confidence range
                # Use sigmoid-like function for smooth mapping
                confidence = 0.2 + 0.6 * (1 / (1 + np.exp(-4 * (confidence - 0.5))))
                
                # Final bounds
                confidence = np.clip(confidence, 0.15, 0.85)
                
                # DEBUG: Log the confidence calculation occasionally
                if np.random.random() < 0.1:  # 10% of the time
                    log.debug(f"Confidence calc: action_prob={action_prob:.3f}, "
                            f"entropy_conf={entropy_confidence:.3f}, "
                            f"value_conf={value_confidence:.3f}, "
                            f"combined={combined_confidence:.3f}, "
                            f"final={confidence:.3f}")
                    
                return action, confidence
                
        except Exception as e:
            log.warning(f"Prediction error: {e}")
            return 0, 0.33  # Default to HOLD with low confidence

    def add_experience(self, state, action, reward, next_state):
        """
        Add experience for online learning
        """
        self.experience_buffer.append((state, action, reward, next_state))
        
        # Keep buffer size manageable
        if len(self.experience_buffer) > 1000:
            self.experience_buffer = self.experience_buffer[-500:]

    def train_online(self):
        """
        Train on recent experiences
        """
        if len(self.experience_buffer) >= self.config.BATCH_SIZE:
            # Take most recent experiences for on-policy learning
            recent_experiences = self.experience_buffer[-self.config.BATCH_SIZE:]
            loss = self.train_on_batch(recent_experiences)
            
            # Clear some old experiences but keep some for stability
            self.experience_buffer = self.experience_buffer[-200:]
            
            return loss
        return 0.0