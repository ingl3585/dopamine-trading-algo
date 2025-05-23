# model/agent.py

import os
import time
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from model.base import ActorCritic

log = logging.getLogger(__name__)

class RLAgent:
    def __init__(self, config):
        self.config = config
        self.temp = config.TEMPERATURE
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ActorCritic(config.INPUT_DIM, config.HIDDEN_DIM, config.ACTION_DIM).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.LR, weight_decay=1e-5)
        
        self.experience_buffer = []
        self.loss_fn = nn.SmoothL1Loss()
        self.last_save_time = 0

        # Feature importance tracking for Ichimoku/EMA
        self.feature_importance = {
            'ichimoku_signals': 0.0,
            'ema_signals': 0.0,
            'momentum_signals': 0.0,
            'volume_signals': 0.0,
            'lwpe_signals': 0.0
        }

        self.load_model()

    def load_model(self):
        if not os.path.exists(self.config.MODEL_PATH):
            log.info("No existing model found, starting fresh with Ichimoku/EMA features")
            return
        try:
            checkpoint = torch.load(self.config.MODEL_PATH, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Check if model dimensions match
                if self._check_model_compatibility(checkpoint):
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    if 'optimizer_state_dict' in checkpoint:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    log.info("Model and optimizer loaded from checkpoint")
                else:
                    log.warning("Model dimensions incompatible, starting fresh")
            else:
                log.warning("Legacy model format detected, starting fresh for Ichimoku/EMA")
        except Exception as e:
            log.warning("Model load failed: %s, starting fresh", e)

    def _check_model_compatibility(self, checkpoint):
        """Check if saved model has compatible dimensions"""
        try:
            config_in_checkpoint = checkpoint.get('config', {})
            expected_input_dim = config_in_checkpoint.get('input_dim', 4)  # Old default
            return expected_input_dim == self.config.INPUT_DIM
        except:
            return False

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
                },
                'feature_importance': self.feature_importance,
                'version': 'ichimoku_ema_v1.0'
            }
            torch.save(checkpoint, self.config.MODEL_PATH)
            log.info("Model checkpoint saved with Ichimoku/EMA configuration")
        except Exception as e:
            log.warning("Save error: %s", e)

    def train_on_batch(self, experiences):
        if len(experiences) < 2:
            return 0.0
            
        try:
            states, actions, rewards, next_states = zip(*experiences)
            
            states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
            actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            
            probs, values = self.model(states)
            _, next_values = self.model(next_states)
            
            targets = rewards + self.config.GAMMA * next_values.squeeze()
            advantages = targets - values.squeeze()
            
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            actor_loss = -(log_probs * advantages.detach()).mean()
            critic_loss = self.loss_fn(values.squeeze(), targets.detach())
            
            total_loss = actor_loss + 0.5 * critic_loss - self.config.ENTROPY_COEF * entropy
            
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Update feature importance tracking
            self._update_feature_importance(states, advantages.detach())
            
            return total_loss.item()
            
        except Exception as e:
            log.warning(f"Training error: {e}")
            return 0.0

    def _update_feature_importance(self, states, advantages):
        """Track which features contribute most to good decisions"""
        try:
            if len(states) > 0:
                # Extract feature contributions (simplified approach)
                state_importance = torch.abs(states * advantages.unsqueeze(-1)).mean(dim=0)
                
                # Map to feature categories (indices based on FeatureVector order)
                self.feature_importance['ichimoku_signals'] = float(
                    (state_importance[2] + state_importance[3] + state_importance[4]).mean()
                )
                self.feature_importance['ema_signals'] = float(state_importance[5])
                self.feature_importance['momentum_signals'] = float(
                    (state_importance[6] + state_importance[7]).mean()
                )
                self.feature_importance['volume_signals'] = float(state_importance[1])
                self.feature_importance['lwpe_signals'] = float(state_importance[8])
                
        except Exception as e:
            log.debug(f"Feature importance update error: {e}")

    def train(self, df, epochs=1):
        try:
            if isinstance(df.iloc[0, 1], str):
                df = df.iloc[1:]

            # Updated for new feature columns
            feature_data = df.iloc[:, 1:-1].apply(pd.to_numeric, errors="coerce").fillna(0).values
            rewards = df.iloc[:, -1].apply(pd.to_numeric, errors="coerce").fillna(0).values
            
            if len(feature_data) < 2:
                log.warning(f"Insufficient data for training: {len(feature_data)} samples")
                return

            # Validate feature dimensions
            if feature_data.shape[1] != self.config.INPUT_DIM:
                log.error(f"Feature dimension mismatch: expected {self.config.INPUT_DIM}, got {feature_data.shape[1]}")
                return

            log.info(f"Training on {len(feature_data)} samples for {epochs} epochs (Ichimoku/EMA features)")
            
            for epoch in range(epochs):
                total_loss = 0.0
                num_batches = 0
                
                batch_size = min(self.config.BATCH_SIZE, len(feature_data) - 1)
                
                for i in range(0, len(feature_data) - 1, batch_size):
                    end_idx = min(i + batch_size, len(feature_data) - 1)
                    
                    batch_experiences = []
                    for j in range(i, end_idx):
                        state = feature_data[j]
                        next_state = feature_data[j + 1]
                        reward = rewards[j]
                        action = 1  # Default action for training
                        
                        batch_experiences.append((state, action, reward, next_state))
                    
                    if batch_experiences:
                        loss = self.train_on_batch(batch_experiences)
                        total_loss += loss
                        num_batches += 1
                
                avg_loss = total_loss / max(num_batches, 1)
                log.info(f"Epoch {epoch+1}/{epochs} - Average loss: {avg_loss:.4f}")

            if time.time() - self.last_save_time > 300:
                self.save_model()
                self.last_save_time = time.time()

        except Exception as e:
            log.warning(f"Training error: {e}")

    def predict_single(self, feat_vec):
        try:
            if len(feat_vec) != self.config.INPUT_DIM:
                log.warning(f"Feature vector dimension mismatch: expected {self.config.INPUT_DIM}, got {len(feat_vec)}")
                return 0, 0.33

            state = torch.tensor(feat_vec, dtype=torch.float32, device=self.device)
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
            
            with torch.no_grad():
                probs, value = self.model(state, temperature=self.temp)
                
                dist = torch.distributions.Categorical(probs)
                action = int(dist.sample())
                
                # Enhanced confidence calculation using Ichimoku/EMA signals
                confidence = self._calculate_enhanced_confidence(feat_vec, probs, value, action)
                
                return action, confidence
                
        except Exception as e:
            log.warning(f"Prediction error: {e}")
            return 0, 0.33

    def _calculate_enhanced_confidence(self, feat_vec, probs, value, action):
        """Calculate confidence using Ichimoku/EMA signal alignment"""
        try:
            # Extract signals from feature vector
            tenkan_kijun_signal = feat_vec[2]
            price_cloud_signal = feat_vec[3]
            future_cloud_signal = feat_vec[4]
            ema_cross_signal = feat_vec[5]
            tenkan_momentum = feat_vec[6]
            kijun_momentum = feat_vec[7]
            normalized_volume = feat_vec[1]
            lwpe = feat_vec[8]
            
            # Base confidence from model
            action_prob = float(probs[0, action])
            value_confidence = min(1.0, abs(float(value[0])) / 2.0)
            
            # Signal alignment confidence
            ichimoku_alignment = self._calculate_ichimoku_alignment(
                tenkan_kijun_signal, price_cloud_signal, future_cloud_signal, action
            )
            
            ema_alignment = self._calculate_ema_alignment(ema_cross_signal, action)
            
            momentum_alignment = self._calculate_momentum_alignment(
                tenkan_momentum, kijun_momentum, action
            )
            
            volume_confidence = self._calculate_volume_confidence(normalized_volume, lwpe)
            
            # Weighted combination
            signal_confidence = (
                self.config.ICHIMOKU_WEIGHT * ichimoku_alignment +
                self.config.EMA_WEIGHT * ema_alignment +
                self.config.MOMENTUM_WEIGHT * momentum_alignment +
                self.config.VOLUME_WEIGHT * volume_confidence +
                self.config.LWPE_WEIGHT * (lwpe if 0 <= lwpe <= 1 else 0.5)
            )
            
            # Combine model and signal confidence
            combined_confidence = (
                0.4 * action_prob +
                0.3 * signal_confidence +
                0.3 * value_confidence
            )
            
            # Apply confidence bounds and smoothing
            confidence = np.clip(combined_confidence, 0.15, 0.85)
            
            # Add small noise for exploration
            noise = np.random.uniform(-0.05, 0.05)
            confidence = np.clip(confidence + noise, 0.15, 0.85)
            
            return confidence
            
        except Exception as e:
            log.debug(f"Confidence calculation error: {e}")
            return 0.33

    def _calculate_ichimoku_alignment(self, tenkan_kijun, price_cloud, future_cloud, action):
        """Enhanced Ichimoku alignment calculation with neutral signal support"""
        if action == 0:  # Hold
            return 0.5
        
        expected_direction = 1 if action == 1 else -1  # Long=1, Short=-1
        
        alignments = []
        weights = []
        
        # Tenkan/Kijun signal
        if tenkan_kijun == expected_direction:
            alignments.append(1.0)
        elif tenkan_kijun == -expected_direction:
            alignments.append(0.0)
        elif tenkan_kijun == 0:
            alignments.append(0.5)  # Neutral
        weights.append(0.4)
        
        # Price vs Cloud signal
        if price_cloud == expected_direction:
            alignments.append(1.0)
        elif price_cloud == -expected_direction:
            alignments.append(0.0)
        elif price_cloud == 0:
            alignments.append(0.3)  # Inside cloud - less confident
        weights.append(0.4)
        
        # Future cloud signal
        if future_cloud == expected_direction:
            alignments.append(1.0)
        elif future_cloud == -expected_direction:
            alignments.append(0.0)
        elif future_cloud == 0:
            alignments.append(0.5)  # Neutral cloud
        weights.append(0.2)
        
        # Weighted average
        if alignments and weights:
            total_weight = sum(weights)
            weighted_sum = sum(a * w for a, w in zip(alignments, weights))
            return weighted_sum / total_weight
        
        return 0.5

    def _calculate_ema_alignment(self, ema_signal, action):
        """Enhanced EMA signal alignment with neutral support"""
        if action == 0:  # Hold
            return 0.5
        
        expected_direction = 1 if action == 1 else -1
        
        if ema_signal == expected_direction:
            return 1.0
        elif ema_signal == -expected_direction:
            return 0.0
        elif ema_signal == 0:
            return 0.4  # Neutral EMA - slightly negative for decision making
        
        return 0.5

    def _calculate_momentum_alignment(self, tenkan_momentum, kijun_momentum, action):
        """Enhanced momentum alignment calculation with neutral support"""
        if action == 0:  # Hold
            return 0.5
        
        expected_direction = 1 if action == 1 else -1
        
        alignments = []
        
        # Tenkan momentum
        if tenkan_momentum == expected_direction:
            alignments.append(1.0)
        elif tenkan_momentum == -expected_direction:
            alignments.append(0.0)
        elif tenkan_momentum == 0:
            alignments.append(0.5)  # Flat momentum
        
        # Kijun momentum
        if kijun_momentum == expected_direction:
            alignments.append(1.0)
        elif kijun_momentum == -expected_direction:
            alignments.append(0.0)
        elif kijun_momentum == 0:
            alignments.append(0.5)  # Flat momentum
        
        return sum(alignments) / len(alignments) if alignments else 0.5

    def _calculate_volume_confidence(self, normalized_volume, lwpe):
        """Calculate volume-based confidence"""
        # Higher volume generally increases confidence
        volume_conf = min(1.0, abs(normalized_volume) / 2.0)
        
        # LWPE near 0.5 indicates balanced market, extreme values indicate direction
        lwpe_conf = abs(lwpe - 0.5) * 2 if 0 <= lwpe <= 1 else 0
        
        return (volume_conf + lwpe_conf) / 2

    def add_experience(self, state, action, reward, next_state):
        self.experience_buffer.append((state, action, reward, next_state))
        
        if len(self.experience_buffer) > 1000:
            self.experience_buffer = self.experience_buffer[-500:]

    def train_online(self):
        if len(self.experience_buffer) >= self.config.BATCH_SIZE:
            recent_experiences = self.experience_buffer[-self.config.BATCH_SIZE:]
            loss = self.train_on_batch(recent_experiences)
            self.experience_buffer = self.experience_buffer[-200:]
            
            return loss
        return 0.0

    def get_feature_importance_summary(self):
        """Get summary of which features are most important"""
        return self.feature_importance