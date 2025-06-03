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

        # Multi-timeframe model (27 inputs)
        self.model = ActorCritic(config.INPUT_DIM, config.HIDDEN_DIM, config.ACTION_DIM).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.LR, weight_decay=1e-5)
        
        self.experience_buffer = []
        self.loss_fn = nn.SmoothL1Loss()
        self.last_save_time = 0

        # Multi-timeframe feature importance tracking
        self.feature_importance = {
            # 15-minute timeframe importance
            '15m_ichimoku_signals': 0.0,
            '15m_ema_signals': 0.0,
            '15m_momentum_signals': 0.0,
            '15m_volume_signals': 0.0,
            '15m_lwpe_signals': 0.0,
            
            # 5-minute timeframe importance
            '5m_ichimoku_signals': 0.0,
            '5m_ema_signals': 0.0,
            '5m_momentum_signals': 0.0,
            '5m_volume_signals': 0.0,
            '5m_lwpe_signals': 0.0,
            
            # 1-minute timeframe importance
            '1m_ichimoku_signals': 0.0,
            '1m_ema_signals': 0.0,
            '1m_momentum_signals': 0.0,
            '1m_volume_signals': 0.0,
            '1m_lwpe_signals': 0.0,
            
            # Overall timeframe importance
            'trend_15m_importance': 0.0,
            'momentum_5m_importance': 0.0,
            'entry_1m_importance': 0.0
        }

        self.load_model()

    def load_model(self):
        if not os.path.exists(self.config.MODEL_PATH):
            log.info("No existing multi-timeframe model found, starting fresh with 27 features")
            return
        try:
            checkpoint = torch.load(self.config.MODEL_PATH, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Check if model dimensions match
                if self._check_model_compatibility(checkpoint):
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    if 'optimizer_state_dict' in checkpoint:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    
                    # Load multi-timeframe feature importance if available
                    if 'feature_importance' in checkpoint:
                        self.feature_importance.update(checkpoint['feature_importance'])
                    
                    log.info("Multi-timeframe model and optimizer loaded from checkpoint")
                else:
                    log.warning("Model dimensions incompatible with 27-feature format, starting fresh")
            else:
                log.warning("Legacy model format detected, starting fresh for multi-timeframe")
        except Exception as e:
            log.warning("Multi-timeframe model load failed: %s, starting fresh", e)

    def _check_model_compatibility(self, checkpoint):
        """Check if saved model has compatible dimensions for 27 features"""
        try:
            config_in_checkpoint = checkpoint.get('config', {})
            expected_input_dim = config_in_checkpoint.get('input_dim', 9)  # Old default
            is_multiframe = config_in_checkpoint.get('is_multiframe', False)
            
            return expected_input_dim == self.config.INPUT_DIM and is_multiframe
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
                    'action_dim': self.config.ACTION_DIM,
                    'is_multiframe': True,
                    'timeframe_count': 3
                },
                'feature_importance': self.feature_importance,
                'version': 'multi_timeframe_v1.0'
            }
            torch.save(checkpoint, self.config.MODEL_PATH)
            log.info("Multi-timeframe model checkpoint saved with 27-feature configuration")
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
            
            # Update multi-timeframe feature importance tracking
            self._update_multiframe_feature_importance(states, advantages.detach())
            
            return total_loss.item()
            
        except Exception as e:
            log.warning(f"Multi-timeframe training error: {e}")
            return 0.0

    def _update_multiframe_feature_importance(self, states, advantages):
        """Track which timeframes and features contribute most to good decisions"""
        try:
            if len(states) > 0:
                # Extract feature contributions
                state_importance = torch.abs(states * advantages.unsqueeze(-1)).mean(dim=0)
                
                # 15-minute timeframe (indices 0-8)
                self.feature_importance['15m_ichimoku_signals'] = float(
                    (state_importance[2] + state_importance[3] + state_importance[4]).mean()
                )
                self.feature_importance['15m_ema_signals'] = float(state_importance[5])
                self.feature_importance['15m_momentum_signals'] = float(
                    (state_importance[6] + state_importance[7]).mean()
                )
                self.feature_importance['15m_volume_signals'] = float(state_importance[1])
                self.feature_importance['15m_lwpe_signals'] = float(state_importance[8])
                
                # 5-minute timeframe (indices 9-17)
                self.feature_importance['5m_ichimoku_signals'] = float(
                    (state_importance[11] + state_importance[12] + state_importance[13]).mean()
                )
                self.feature_importance['5m_ema_signals'] = float(state_importance[14])
                self.feature_importance['5m_momentum_signals'] = float(
                    (state_importance[15] + state_importance[16]).mean()
                )
                self.feature_importance['5m_volume_signals'] = float(state_importance[10])
                self.feature_importance['5m_lwpe_signals'] = float(state_importance[17])
                
                # 1-minute timeframe (indices 18-26)
                self.feature_importance['1m_ichimoku_signals'] = float(
                    (state_importance[20] + state_importance[21] + state_importance[22]).mean()
                )
                self.feature_importance['1m_ema_signals'] = float(state_importance[23])
                self.feature_importance['1m_momentum_signals'] = float(
                    (state_importance[24] + state_importance[25]).mean()
                )
                self.feature_importance['1m_volume_signals'] = float(state_importance[19])
                self.feature_importance['1m_lwpe_signals'] = float(state_importance[26])
                
                # Overall timeframe importance
                self.feature_importance['trend_15m_importance'] = float(state_importance[0:9].mean())
                self.feature_importance['momentum_5m_importance'] = float(state_importance[9:18].mean())
                self.feature_importance['entry_1m_importance'] = float(state_importance[18:27].mean())
                
        except Exception as e:
            log.debug(f"Multi-timeframe feature importance update error: {e}")

    def train(self, df, epochs=1):
        try:
            if isinstance(df.iloc[0, 1], str):
                df = df.iloc[1:]

            # Updated for 27-feature columns
            feature_data = df.iloc[:, 1:-1].apply(pd.to_numeric, errors="coerce").fillna(0).values
            rewards = df.iloc[:, -1].apply(pd.to_numeric, errors="coerce").fillna(0).values
            
            if len(feature_data) < 2:
                log.warning(f"Insufficient data for multi-timeframe training: {len(feature_data)} samples")
                return

            # Validate feature dimensions for 27-feature model
            if feature_data.shape[1] != self.config.INPUT_DIM:
                log.error(f"Feature dimension mismatch: expected {self.config.INPUT_DIM}, got {feature_data.shape[1]}")
                return

            log.info(f"Multi-timeframe training on {len(feature_data)} samples for {epochs} epochs (27 features)")
            
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
                log.info(f"Epoch {epoch+1}/{epochs} - Multi-timeframe average loss: {avg_loss:.4f}")

            if time.time() - self.last_save_time > 300:
                self.save_model()
                self.last_save_time = time.time()

        except Exception as e:
            log.warning(f"Multi-timeframe training error: {e}")

    def predict_single(self, feat_vec):
        try:
            if len(feat_vec) != self.config.INPUT_DIM:
                log.warning(f"Multi-timeframe feature vector dimension mismatch: expected {self.config.INPUT_DIM}, got {len(feat_vec)}")
                
                # Try to handle gracefully
                if len(feat_vec) == 9:
                    log.info("Received 9-feature vector, expanding to 27-feature format")
                    feat_vec = self._expand_9_to_27_features(feat_vec)
                else:
                    return 0, 0.33

            state = torch.tensor(feat_vec, dtype=torch.float32, device=self.device)
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
            
            with torch.no_grad():
                probs, value = self.model(state, temperature=self.temp)
                
                dist = torch.distributions.Categorical(probs)
                action = int(dist.sample())
                
                # Enhanced multi-timeframe confidence calculation
                confidence = self._calculate_multiframe_confidence(feat_vec, probs, value, action)
                
                return action, confidence
                
        except Exception as e:
            log.warning(f"Multi-timeframe prediction error: {e}")
            return 0, 0.33

    def _expand_9_to_27_features(self, feat_vec_9):
        """Expand 9-feature vector to 27-feature format for backward compatibility"""
        try:
            # Replicate the 9 features across all 3 timeframes
            # This is a fallback - not ideal but allows system to continue
            expanded = []
            
            # 15-minute (trend context)
            expanded.extend(feat_vec_9)
            
            # 5-minute (momentum context) 
            expanded.extend(feat_vec_9)
            
            # 1-minute (entry timing)
            expanded.extend(feat_vec_9)
            
            log.debug("Expanded 9-feature vector to 27-feature format")
            return expanded
            
        except Exception as e:
            log.warning(f"Feature expansion error: {e}")
            return feat_vec_9 + [0.0] * (27 - len(feat_vec_9))

    def _calculate_multiframe_confidence(self, feat_vec, probs, value, action):
        """Calculate confidence using multi-timeframe signal alignment"""
        try:
            # Extract timeframe features
            features_15m = feat_vec[0:9]    # Trend context
            features_5m = feat_vec[9:18]    # Momentum context  
            features_1m = feat_vec[18:27]   # Entry timing
            
            # Base confidence from model
            action_prob = float(probs[0, action])
            value_confidence = min(1.0, abs(float(value[0])) / 2.0)
            
            # Multi-timeframe alignment confidence
            trend_alignment = self._calculate_timeframe_alignment(features_15m, action, "15m")
            momentum_alignment = self._calculate_timeframe_alignment(features_5m, action, "5m")
            entry_alignment = self._calculate_timeframe_alignment(features_1m, action, "1m")
            
            # Check for trend-fighting scenarios
            trend_filter_penalty = self._calculate_trend_filter_penalty(
                features_15m, features_5m, features_1m, action
            )
            
            # Weighted timeframe combination
            timeframe_confidence = (
                self.config.TREND_15M_WEIGHT * trend_alignment +
                self.config.MOMENTUM_5M_WEIGHT * momentum_alignment +
                self.config.ENTRY_1M_WEIGHT * entry_alignment
            )
            
            # Combine model and timeframe confidence
            combined_confidence = (
                0.3 * action_prob +
                0.5 * timeframe_confidence +  # Higher weight on timeframe alignment
                0.2 * value_confidence
            )
            
            # Apply trend filter penalty
            combined_confidence *= (1.0 - trend_filter_penalty)
            
            # Apply bounds and smoothing
            confidence = np.clip(combined_confidence, 0.15, 0.90)
            
            # Add small noise for exploration
            noise = np.random.uniform(-0.03, 0.03)
            confidence = np.clip(confidence + noise, 0.15, 0.90)
            
            return confidence
            
        except Exception as e:
            log.debug(f"Multi-timeframe confidence calculation error: {e}")
            return 0.33

    def _calculate_timeframe_alignment(self, timeframe_features, action, timeframe_name):
        """Calculate signal alignment within a specific timeframe"""
        try:
            if action == 0:  # Hold
                return 0.5
            
            expected_direction = 1 if action == 1 else -1
            
            # Extract signals from timeframe (indices based on feature structure)
            tenkan_kijun = timeframe_features[2]
            price_cloud = timeframe_features[3] 
            future_cloud = timeframe_features[4]
            ema_cross = timeframe_features[5]
            tenkan_momentum = timeframe_features[6]
            kijun_momentum = timeframe_features[7]
            normalized_volume = timeframe_features[1]
            lwpe = timeframe_features[8]
            
            alignments = []
            weights = []
            
            # Ichimoku alignment with timeframe-specific weights
            if timeframe_name == "15m":
                # Trend timeframe - emphasize cloud and major signals
                if tenkan_kijun == expected_direction:
                    alignments.append(1.0)
                elif tenkan_kijun == -expected_direction:
                    alignments.append(0.0)
                elif tenkan_kijun == 0:
                    alignments.append(0.5)
                weights.append(0.3)
                
                if price_cloud == expected_direction:
                    alignments.append(1.0)
                elif price_cloud == -expected_direction:
                    alignments.append(0.0)
                elif price_cloud == 0:
                    alignments.append(0.2)  # Being in cloud is less decisive
                weights.append(0.4)  # High weight for trend timeframe
                
                if future_cloud == expected_direction:
                    alignments.append(1.0)
                elif future_cloud == -expected_direction:
                    alignments.append(0.0)
                elif future_cloud == 0:
                    alignments.append(0.5)
                weights.append(0.3)
                
            elif timeframe_name == "5m":
                # Momentum timeframe - emphasize EMA and momentum
                if ema_cross == expected_direction:
                    alignments.append(1.0)
                elif ema_cross == -expected_direction:
                    alignments.append(0.0)
                elif ema_cross == 0:
                    alignments.append(0.4)
                weights.append(0.4)
                
                if tenkan_kijun == expected_direction:
                    alignments.append(1.0)
                elif tenkan_kijun == -expected_direction:
                    alignments.append(0.0)
                elif tenkan_kijun == 0:
                    alignments.append(0.5)
                weights.append(0.3)
                
                # Momentum factors
                momentum_score = 0.0
                momentum_count = 0
                if tenkan_momentum != 0:
                    momentum_score += 1.0 if tenkan_momentum == expected_direction else 0.0
                    momentum_count += 1
                if kijun_momentum != 0:
                    momentum_score += 1.0 if kijun_momentum == expected_direction else 0.0
                    momentum_count += 1
                
                if momentum_count > 0:
                    alignments.append(momentum_score / momentum_count)
                    weights.append(0.3)
                
            else:  # 1m - entry timing
                # Entry timeframe - balance all signals
                if tenkan_kijun == expected_direction:
                    alignments.append(1.0)
                elif tenkan_kijun == -expected_direction:
                    alignments.append(0.0)
                elif tenkan_kijun == 0:
                    alignments.append(0.5)
                weights.append(0.25)
                
                if price_cloud == expected_direction:
                    alignments.append(1.0)
                elif price_cloud == -expected_direction:
                    alignments.append(0.0)
                elif price_cloud == 0:
                    alignments.append(0.3)
                weights.append(0.25)
                
                if ema_cross == expected_direction:
                    alignments.append(1.0)
                elif ema_cross == -expected_direction:
                    alignments.append(0.0)
                elif ema_cross == 0:
                    alignments.append(0.4)
                weights.append(0.25)
                
                # Volume and LWPE for entry timing
                volume_conf = min(1.0, abs(normalized_volume) / 2.0)
                lwpe_conf = abs(lwpe - 0.5) * 2 if 0 <= lwpe <= 1 else 0
                alignments.append((volume_conf + lwpe_conf) / 2)
                weights.append(0.25)
            
            # Weighted average
            if alignments and weights:
                total_weight = sum(weights)
                weighted_sum = sum(a * w for a, w in zip(alignments, weights))
                return weighted_sum / total_weight
            
            return 0.5
            
        except Exception as e:
            log.debug(f"Timeframe alignment calculation error for {timeframe_name}: {e}")
            return 0.5

    def _calculate_trend_filter_penalty(self, features_15m, features_5m, features_1m, action):
        """Calculate penalty for trading against strong trends"""
        try:
            if action == 0:  # Hold action
                return 0.0
            
            # Extract 15-minute trend signals
            tenkan_kijun_15m = features_15m[2]
            price_cloud_15m = features_15m[3]
            ema_cross_15m = features_15m[5]
            
            # Calculate 15-minute trend strength
            trend_signals = [tenkan_kijun_15m, price_cloud_15m, ema_cross_15m]
            non_zero_signals = [s for s in trend_signals if s != 0]
            
            if len(non_zero_signals) < 2:
                return 0.0  # Not enough signal for trend determination
            
            trend_direction = sum(non_zero_signals) / len(non_zero_signals)
            trend_strength = abs(trend_direction)
            
            # Check if action is against strong trend
            action_direction = 1 if action == 1 else -1
            
            if trend_strength > self.config.TREND_FILTER_STRENGTH:
                if (trend_direction > 0 and action_direction < 0) or \
                   (trend_direction < 0 and action_direction > 0):
                    # Trading against strong trend
                    penalty = trend_strength * 0.5  # Up to 50% confidence reduction
                    return min(penalty, 0.6)  # Cap penalty at 60%
            
            return 0.0
            
        except Exception as e:
            log.debug(f"Trend filter penalty calculation error: {e}")
            return 0.0

    def add_experience(self, state, action, reward, next_state):
        self.experience_buffer.append((state, action, reward, next_state))
        
        if len(self.experience_buffer) > 2000:  # Increased buffer for 27 features
            self.experience_buffer = self.experience_buffer[-1000:]

    def train_online(self):
        if len(self.experience_buffer) >= self.config.BATCH_SIZE:
            recent_experiences = self.experience_buffer[-self.config.BATCH_SIZE:]
            loss = self.train_on_batch(recent_experiences)
            self.experience_buffer = self.experience_buffer[-400:]  # Keep more history
            
            return loss
        return 0.0

    def get_feature_importance_summary(self):
        """Get summary of multi-timeframe feature importance"""
        return self.feature_importance

    def get_timeframe_analysis(self):
        """Get analysis of which timeframes are most important"""
        try:
            analysis = {
                'trend_15m_dominance': self.feature_importance.get('trend_15m_importance', 0),
                'momentum_5m_dominance': self.feature_importance.get('momentum_5m_importance', 0),
                'entry_1m_dominance': self.feature_importance.get('entry_1m_importance', 0)
            }
            
            # Determine dominant timeframe
            dominant_timeframe = max(analysis.items(), key=lambda x: x[1])
            analysis['dominant_timeframe'] = dominant_timeframe[0]
            analysis['dominant_strength'] = dominant_timeframe[1]
            
            return analysis
            
        except Exception as e:
            log.warning(f"Timeframe analysis error: {e}")
            return {}