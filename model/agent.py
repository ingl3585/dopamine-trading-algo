# model/agent.py

import os
import time
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from model.base import MultiTimeframeActorCritic, ActorCritic

log = logging.getLogger(__name__)

class RLAgent:
    def __init__(self, config):
        self.config = config
        self.temp = config.TEMPERATURE
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use enhanced multi-timeframe model by default
        self.use_enhanced_model = config.INPUT_DIM == 27 and hasattr(config, 'MULTI_TIMEFRAME_MODE') and config.MULTI_TIMEFRAME_MODE
        
        if self.use_enhanced_model:
            log.info("Initializing enhanced multi-timeframe model with attention mechanism")
            self.model = MultiTimeframeActorCritic(
                input_dim=config.INPUT_DIM, 
                hidden_dim=config.HIDDEN_DIM, 
                action_dim=config.ACTION_DIM
            ).to(self.device)
        else:
            log.info("Using legacy ActorCritic model")
            self.model = ActorCritic(
                config.INPUT_DIM, 
                config.HIDDEN_DIM, 
                config.ACTION_DIM
            ).to(self.device)
        
        # Enhanced optimizer for the larger model
        if self.use_enhanced_model:
            self.optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=config.LR, 
                weight_decay=1e-4,  # Increased regularization
                betas=(0.9, 0.95)   # Adjusted betas for stability
            )
            # Learning rate scheduler
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=1000, 
                eta_min=config.LR * 0.1
            )
        else:
            self.optimizer = optim.AdamW(self.model.parameters(), lr=config.LR, weight_decay=1e-5)
            self.scheduler = None
        
        self.experience_buffer = []
        self.loss_fn = nn.SmoothL1Loss()
        self.last_save_time = 0
        
        # Enhanced feature importance tracking for multi-timeframe
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
            'entry_1m_importance': 0.0,
            
            # Model attention weights (if using enhanced model)
            'learned_15m_weight': 0.0,
            'learned_5m_weight': 0.0,
            'learned_1m_weight': 0.0
        }
        
        # Performance tracking for enhanced model
        self.model_performance = {
            'attention_entropy': [],
            'timeframe_importance_evolution': [],
            'confidence_accuracy': [],
            'prediction_consistency': []
        }

        self.load_model()

    def load_model(self):
        if not os.path.exists(self.config.MODEL_PATH):
            log.info(f"No existing model found, starting fresh with {'enhanced multi-timeframe' if self.use_enhanced_model else 'standard'} architecture")
            return
        
        try:
            checkpoint = torch.load(self.config.MODEL_PATH, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Check model compatibility
                if self._check_enhanced_model_compatibility(checkpoint):
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    
                    if self.scheduler and 'scheduler_state_dict' in checkpoint:
                        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    
                    # Load enhanced feature importance if available
                    if 'feature_importance' in checkpoint:
                        self.feature_importance.update(checkpoint['feature_importance'])
                    
                    # Load performance tracking
                    if 'model_performance' in checkpoint:
                        self.model_performance.update(checkpoint['model_performance'])
                    
                    log.info(f"Enhanced multi-timeframe model loaded from checkpoint")
                else:
                    log.warning("Model dimensions incompatible with enhanced architecture, starting fresh")
            else:
                log.warning("Legacy model format detected, starting fresh for enhanced multi-timeframe")
        except Exception as e:
            log.warning(f"Enhanced model load failed: {e}, starting fresh")

    def _check_enhanced_model_compatibility(self, checkpoint):
        """Check if saved model is compatible with enhanced architecture"""
        try:
            config_in_checkpoint = checkpoint.get('config', {})
            expected_input_dim = config_in_checkpoint.get('input_dim', 9)
            is_enhanced = config_in_checkpoint.get('is_enhanced_multiframe', False)
            
            return (expected_input_dim == self.config.INPUT_DIM and 
                    is_enhanced == self.use_enhanced_model)
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
                    'is_enhanced_multiframe': self.use_enhanced_model,
                    'timeframe_count': 3,
                    'architecture_version': 'v2.0_attention'
                },
                'feature_importance': self.feature_importance,
                'model_performance': self.model_performance,
                'version': 'enhanced_multi_timeframe_v2.0'
            }
            
            if self.scheduler:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
            # Add model-specific info for enhanced model
            if self.use_enhanced_model and hasattr(self.model, 'get_timeframe_importance'):
                checkpoint['learned_timeframe_weights'] = self.model.get_timeframe_importance()
            
            torch.save(checkpoint, self.config.MODEL_PATH)
            log.info(f"Enhanced multi-timeframe model checkpoint saved")
        except Exception as e:
            log.warning(f"Enhanced model save error: {e}")

    def predict_single(self, feat_vec):
        try:
            if len(feat_vec) != self.config.INPUT_DIM:
                log.warning(f"Feature vector dimension mismatch: expected {self.config.INPUT_DIM}, got {len(feat_vec)}")
                
                if len(feat_vec) == 9:
                    log.info("Received 9-feature vector, expanding to 27-feature format")
                    feat_vec = self._expand_9_to_27_features(feat_vec)
                else:
                    return 0, 0.33

            state = torch.tensor(feat_vec, dtype=torch.float32, device=self.device)
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
            
            with torch.no_grad():
                if self.use_enhanced_model:
                    # Use enhanced model with attention analysis
                    probs, value, attention_weights, timeframe_weights = self.model(
                        state, temperature=self.temp, return_attention=True
                    )
                    
                    # Update learned timeframe importance
                    tf_weights = timeframe_weights.cpu().numpy()
                    self.feature_importance['learned_15m_weight'] = float(tf_weights[0])
                    self.feature_importance['learned_5m_weight'] = float(tf_weights[1])
                    self.feature_importance['learned_1m_weight'] = float(tf_weights[2])
                    
                    # Track attention entropy for analysis
                    att_entropy = self._calculate_attention_entropy(attention_weights)
                    self.model_performance['attention_entropy'].append(float(att_entropy))
                    
                else:
                    # Use standard model
                    probs, value = self.model(state, temperature=self.temp)
                
                dist = torch.distributions.Categorical(probs)
                action = int(dist.sample())
                
                # Enhanced confidence calculation
                confidence = self._calculate_multiframe_confidence(feat_vec, probs, value, action)
                
                return action, confidence
                
        except Exception as e:
            log.warning(f"Enhanced prediction error: {e}")
            return 0, 0.33

    def _calculate_attention_entropy(self, attention_weights):
        """Calculate attention entropy for analysis"""
        try:
            # attention_weights shape: [batch, 3, 3] (batch, timeframes, timeframes)
            avg_attention = attention_weights.mean(dim=0)  # Average across batch
            
            # Calculate entropy across timeframes
            entropy = 0.0
            for i in range(avg_attention.shape[0]):
                probs = avg_attention[i]
                probs = probs + 1e-8  # Avoid log(0)
                entropy += -(probs * torch.log(probs)).sum()
            
            return entropy / avg_attention.shape[0]  # Average entropy
        except:
            return 0.0

    def train_on_batch(self, experiences):
        if len(experiences) < 2:
            return 0.0
            
        try:
            states, actions, rewards, next_states = zip(*experiences)
            
            states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
            actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            
            # Forward pass
            if self.use_enhanced_model:
                probs, values = self.model(states)
                _, next_values = self.model(next_states)
            else:
                probs, values = self.model(states)
                _, next_values = self.model(next_states)
            
            # Calculate targets and advantages
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
            
            # Enhanced loss for better training
            if self.use_enhanced_model:
                # Add regularization for attention weights
                attention_reg = 0.0
                if hasattr(self.model, 'timeframe_attention'):
                    # Encourage diverse attention patterns
                    timeframe_weights = torch.softmax(self.model.timeframe_weights, dim=0)
                    attention_reg = -torch.sum(timeframe_weights * torch.log(timeframe_weights + 1e-8))
                    attention_reg *= 0.01  # Small regularization weight
                
                total_loss = (actor_loss + 0.5 * critic_loss - 
                             self.config.ENTROPY_COEF * entropy + attention_reg)
            else:
                total_loss = actor_loss + 0.5 * critic_loss - self.config.ENTROPY_COEF * entropy
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Enhanced gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            # Update learning rate if using scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Update feature importance tracking
            self._update_multiframe_feature_importance(states, advantages.detach())
            
            return total_loss.item()
            
        except Exception as e:
            log.warning(f"Enhanced training error: {e}")
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
        """
        Enhanced multi-timeframe confidence calculation with consistency bonuses
        """
        try:
            # Extract timeframe features
            features_15m = feat_vec[0:9]    # Trend context
            features_5m = feat_vec[9:18]    # Momentum context  
            features_1m = feat_vec[18:27]   # Entry timing
            
            # Calculate signal strength per timeframe with enhanced logic
            trend_strength = self._calculate_signal_strength(features_15m, action, "trend")
            momentum_strength = self._calculate_signal_strength(features_5m, action, "momentum")
            entry_strength = self._calculate_signal_strength(features_1m, action, "entry")
            
            # Calculate timeframe consistency bonus
            consistency_bonus = self._calculate_consistency_bonus(
                trend_strength, momentum_strength, entry_strength
            )
            
            # Calculate signal quality bonus
            quality_bonus = self._calculate_signal_quality_bonus(
                features_15m, features_5m, features_1m, action
            )
            
            # Base model confidence components
            action_prob = float(probs[0, action])
            value_confidence = min(1.0, abs(float(value[0])) / 2.0)
            
            # Enhanced timeframe confidence with consistency
            timeframe_confidence = (
                self.config.TREND_15M_WEIGHT * trend_strength +
                self.config.MOMENTUM_5M_WEIGHT * momentum_strength +
                self.config.ENTRY_1M_WEIGHT * entry_strength +
                0.15 * consistency_bonus +  # Bonus for aligned timeframes
                0.10 * quality_bonus        # Bonus for signal quality
            )
            
            # Check for trend-fighting scenarios (enhanced)
            trend_filter_penalty = self._calculate_enhanced_trend_filter_penalty(
                features_15m, features_5m, features_1m, action
            )
            
            # Final confidence combination with enhanced weighting
            combined_confidence = (
                0.20 * action_prob +           # Model prediction probability
                0.55 * timeframe_confidence +  # Multi-timeframe alignment (increased weight)
                0.15 * value_confidence +      # Value estimate confidence
                0.10 * self._calculate_volume_lwpe_bonus(feat_vec)  # Market condition bonus
            )
            
            # Apply trend filter penalty
            combined_confidence *= (1.0 - trend_filter_penalty)
            
            # Apply bounds and add controlled exploration noise
            confidence = np.clip(combined_confidence, 0.15, 0.95)
            
            # Adaptive noise based on confidence level
            if confidence > 0.8:
                noise_range = 0.02  # Low noise for high confidence
            elif confidence > 0.6:
                noise_range = 0.03  # Medium noise
            else:
                noise_range = 0.05  # Higher noise for low confidence (exploration)
            
            noise = np.random.uniform(-noise_range, noise_range)
            confidence = np.clip(confidence + noise, 0.15, 0.95)
            
            return confidence
            
        except Exception as e:
            log.debug(f"Enhanced confidence calculation error: {e}")
            return 0.33
        
    def _calculate_consistency_bonus(self, trend_str, momentum_str, entry_str):
        """
        Enhanced consistency bonus calculation
        """
        try:
            strengths = [trend_str, momentum_str, entry_str]
            
            # Perfect alignment (all strong in same direction)
            if all(s > 0.75 for s in strengths):
                return 1.0  # Perfect bullish alignment
            elif all(s < 0.25 for s in strengths):
                return 1.0  # Perfect bearish alignment
            
            # Strong majority alignment
            strong_bull = sum(1 for s in strengths if s > 0.65)
            strong_bear = sum(1 for s in strengths if s < 0.35)
            
            if strong_bull >= 2:
                return 0.8  # Strong bullish majority
            elif strong_bear >= 2:
                return 0.8  # Strong bearish majority
            
            # Moderate alignment
            moderate_bull = sum(1 for s in strengths if s > 0.55)
            moderate_bear = sum(1 for s in strengths if s < 0.45)
            
            if moderate_bull >= 2:
                return 0.5  # Moderate bullish consensus
            elif moderate_bear >= 2:
                return 0.5  # Moderate bearish consensus
            
            # Check for dangerous conflicts (trend vs entry)
            trend_bullish = trend_str > 0.6
            trend_bearish = trend_str < 0.4
            entry_bullish = entry_str > 0.6
            entry_bearish = entry_str < 0.4
            
            if (trend_bullish and entry_bearish) or (trend_bearish and entry_bullish):
                return -0.3  # Penalty for trend-fighting entry
            
            # Default for mixed/neutral signals
            return 0.0
            
        except Exception as e:
            log.debug(f"Consistency bonus calculation error: {e}")
            return 0.0

    def _calculate_signal_quality_bonus(self, features_15m, features_5m, features_1m, action):
        """
        Calculate bonus based on overall signal quality across timeframes
        """
        try:
            if action == 0:
                return 0.0
            
            expected_direction = 1 if action == 1 else -1
            quality_score = 0.0
            
            # Check for classic bullish/bearish setups on each timeframe
            for i, (features, tf_name, weight) in enumerate([
                (features_15m, "15m", 0.5),
                (features_5m, "5m", 0.3), 
                (features_1m, "1m", 0.2)
            ]):
                setup_quality = self._evaluate_timeframe_setup_quality(features, expected_direction, tf_name)
                quality_score += setup_quality * weight
            
            return max(0, min(1, quality_score))
            
        except Exception as e:
            log.debug(f"Signal quality bonus calculation error: {e}")
            return 0.0
        
    def _evaluate_timeframe_setup_quality(self, features, expected_direction, timeframe_name):
        """
        Evaluate the quality of trading setup for a specific timeframe
        """
        try:
            tenkan_kijun = features[2]
            price_cloud = features[3]
            future_cloud = features[4]
            ema_cross = features[5]
            tenkan_momentum = features[6]
            kijun_momentum = features[7]
            lwpe = features[8]
            
            # Classic bullish setup patterns
            if expected_direction == 1:
                # Perfect bullish setup
                if (tenkan_kijun > 0 and price_cloud > 0 and future_cloud > 0 and 
                    ema_cross > 0 and lwpe > 0.6):
                    return 1.0
                
                # Strong bullish setup
                bullish_signals = sum([
                    tenkan_kijun > 0, price_cloud > 0, future_cloud > 0,
                    ema_cross > 0, tenkan_momentum > 0, kijun_momentum > 0
                ])
                
                if bullish_signals >= 4:
                    return 0.8
                elif bullish_signals >= 3:
                    return 0.6
                elif bullish_signals >= 2:
                    return 0.4
                else:
                    return 0.1
            
            # Classic bearish setup patterns
            else:  # expected_direction == -1
                # Perfect bearish setup
                if (tenkan_kijun < 0 and price_cloud < 0 and future_cloud < 0 and 
                    ema_cross < 0 and lwpe < 0.4):
                    return 1.0
                
                # Strong bearish setup
                bearish_signals = sum([
                    tenkan_kijun < 0, price_cloud < 0, future_cloud < 0,
                    ema_cross < 0, tenkan_momentum < 0, kijun_momentum < 0
                ])
                
                if bearish_signals >= 4:
                    return 0.8
                elif bearish_signals >= 3:
                    return 0.6
                elif bearish_signals >= 2:
                    return 0.4
                else:
                    return 0.1
            
        except Exception as e:
            log.debug(f"Setup quality evaluation error for {timeframe_name}: {e}")
            return 0.0

    def _calculate_enhanced_trend_filter_penalty(self, features_15m, features_5m, features_1m, action):
        """
        Enhanced trend filter with multi-timeframe analysis
        """
        try:
            if action == 0:  # Hold action
                return 0.0
            
            # Extract 15-minute trend signals (primary filter)
            tenkan_kijun_15m = features_15m[2]
            price_cloud_15m = features_15m[3]
            ema_cross_15m = features_15m[5]
            
            # Extract 5-minute momentum signals (secondary filter)
            tenkan_kijun_5m = features_5m[2]
            ema_cross_5m = features_5m[5]
            
            # Calculate 15-minute trend strength
            trend_15m_signals = [tenkan_kijun_15m, price_cloud_15m, ema_cross_15m]
            non_zero_trend = [s for s in trend_15m_signals if s != 0]
            
            if len(non_zero_trend) >= 2:  # Sufficient signal for trend determination
                trend_direction_15m = sum(non_zero_trend) / len(non_zero_trend)
                trend_strength_15m = abs(trend_direction_15m)
            else:
                return 0.0  # No clear trend
            
            # Calculate 5-minute momentum direction
            momentum_5m_signals = [tenkan_kijun_5m, ema_cross_5m]
            non_zero_momentum = [s for s in momentum_5m_signals if s != 0]
            
            if len(non_zero_momentum) >= 1:
                momentum_direction_5m = sum(non_zero_momentum) / len(non_zero_momentum)
            else:
                momentum_direction_5m = 0
            
            # Determine action direction
            action_direction = 1 if action == 1 else -1
            
            # Primary penalty: Fighting strong 15-minute trend
            primary_penalty = 0.0
            if trend_strength_15m > self.config.TREND_FILTER_STRENGTH:
                if (trend_direction_15m > 0 and action_direction < 0) or \
                (trend_direction_15m < 0 and action_direction > 0):
                    # Base penalty for fighting trend
                    primary_penalty = trend_strength_15m * 0.6
                    
                    # Additional penalty if 5-minute momentum also opposes
                    if abs(momentum_direction_5m) > 0.3:
                        if (momentum_direction_5m > 0 and action_direction < 0) or \
                        (momentum_direction_5m < 0 and action_direction > 0):
                            primary_penalty += 0.2  # Double penalty
            
            # Secondary penalty: Early counter-trend entries
            secondary_penalty = 0.0
            if trend_strength_15m > 0.4:  # Medium strength trend
                if (trend_direction_15m > 0 and action_direction < 0) or \
                (trend_direction_15m < 0 and action_direction > 0):
                    # Check if this might be a premature reversal attempt
                    if abs(momentum_direction_5m) < 0.5:  # Weak counter-momentum
                        secondary_penalty = 0.15
            
            # Cap total penalty
            total_penalty = min(primary_penalty + secondary_penalty, 0.8)
            
            if total_penalty > 0.1:
                log.debug(f"Trend filter penalty: {total_penalty:.3f} for action {action} "
                        f"against 15m trend {trend_direction_15m:.2f}")
            
            return total_penalty
            
        except Exception as e:
            log.debug(f"Enhanced trend filter penalty calculation error: {e}")
            return 0.0

    def _calculate_volume_lwpe_bonus(self, feat_vec):
        """
        Calculate bonus based on volume and LWPE conditions across timeframes
        """
        try:
            # Extract volume and LWPE from all timeframes
            vol_15m = feat_vec[1]   # Normalized volume 15m
            lwpe_15m = feat_vec[8]  # LWPE 15m
            vol_5m = feat_vec[10]   # Normalized volume 5m
            lwpe_5m = feat_vec[17]  # LWPE 5m
            vol_1m = feat_vec[19]   # Normalized volume 1m
            lwpe_1m = feat_vec[26]  # LWPE 1m
            
            bonus = 0.0
            
            # Volume confirmation bonus
            high_volume_count = sum([
                abs(vol_15m) > 1.5,
                abs(vol_5m) > 1.5, 
                abs(vol_1m) > 1.5
            ])
            
            if high_volume_count >= 2:
                bonus += 0.15  # Multi-timeframe volume confirmation
            elif high_volume_count == 1:
                bonus += 0.08  # Single timeframe volume
            
            # LWPE extremes bonus (institutional flow detection)
            extreme_lwpe_count = sum([
                abs(lwpe_15m - 0.5) > 0.3,
                abs(lwpe_5m - 0.5) > 0.3,
                abs(lwpe_1m - 0.5) > 0.3
            ])
            
            if extreme_lwpe_count >= 2:
                bonus += 0.12  # Strong institutional flow
            elif extreme_lwpe_count == 1:
                bonus += 0.06  # Moderate institutional activity
            
            # LWPE direction consistency bonus
            lwpe_values = [lwpe_15m, lwpe_5m, lwpe_1m]
            all_bullish_flow = all(lwpe > 0.65 for lwpe in lwpe_values)
            all_bearish_flow = all(lwpe < 0.35 for lwpe in lwpe_values)
            
            if all_bullish_flow or all_bearish_flow:
                bonus += 0.10  # Consistent institutional flow direction
            
            return min(bonus, 0.25)  # Cap bonus at 25%
            
        except Exception as e:
            log.debug(f"Volume/LWPE bonus calculation error: {e}")
            return 0.0

    def _calculate_volume_boost(self, normalized_volume, timeframe_type):
        """
        Calculate volume boost factor based on timeframe
        """
        try:
            vol_abs = abs(normalized_volume)
            
            if timeframe_type == "trend":  # 15-minute
                # Trend timeframe needs sustained volume
                if vol_abs > 2.0:
                    return 0.15
                elif vol_abs > 1.5:
                    return 0.10
                elif vol_abs > 1.0:
                    return 0.05
                else:
                    return 0.0
            
            elif timeframe_type == "momentum":  # 5-minute
                # Momentum timeframe benefits from volume spikes
                if vol_abs > 2.5:
                    return 0.20
                elif vol_abs > 1.8:
                    return 0.12
                elif vol_abs > 1.2:
                    return 0.06
                else:
                    return 0.0
            
            else:  # "entry" - 1-minute
                # Entry timeframe needs immediate volume confirmation
                if vol_abs > 3.0:
                    return 0.25
                elif vol_abs > 2.0:
                    return 0.15
                elif vol_abs > 1.0:
                    return 0.08
                else:
                    return 0.0
            
        except Exception as e:
            log.debug(f"Volume boost calculation error: {e}")
            return 0.0

    def _calculate_lwpe_boost(self, lwpe, expected_direction):
        """
        Calculate LWPE boost based on institutional flow direction
        """
        try:
            # LWPE interpretation:
            # > 0.7: Strong buying pressure
            # 0.3-0.7: Neutral/mixed
            # < 0.3: Strong selling pressure
            
            if expected_direction == 1:  # Long position
                if lwpe > 0.75:
                    return 0.15  # Strong buying pressure supports long
                elif lwpe > 0.65:
                    return 0.08  # Moderate buying pressure
                elif lwpe < 0.25:
                    return -0.10  # Selling pressure opposes long
                else:
                    return 0.0
            
            else:  # Short position (expected_direction == -1)
                if lwpe < 0.25:
                    return 0.15  # Strong selling pressure supports short
                elif lwpe < 0.35:
                    return 0.08  # Moderate selling pressure
                elif lwpe > 0.75:
                    return -0.10  # Buying pressure opposes short
                else:
                    return 0.0
            
        except Exception as e:
            log.debug(f"LWPE boost calculation error: {e}")
            return 0.0
        
    def _calculate_signal_strength(self, timeframe_features, action, timeframe_type):
        """
        Enhanced signal strength calculation with timeframe-specific logic
        """
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
            
            # Timeframe-specific weighting schemes
            if timeframe_type == "trend":  # 15-minute
                weights = {
                    'price_cloud': 0.35,      # Cloud position most important for trend
                    'tenkan_kijun': 0.25,     # TK cross significant
                    'future_cloud': 0.20,     # Future cloud direction
                    'ema_cross': 0.15,        # EMA trend support
                    'momentum': 0.05          # Momentum less important for trend
                }
                # Trend requires stronger signals
                strength_threshold = 0.7
                
            elif timeframe_type == "momentum":  # 5-minute
                weights = {
                    'ema_cross': 0.30,        # EMA most important for momentum
                    'tenkan_kijun': 0.25,     # TK cross
                    'momentum': 0.25,         # Momentum signals important
                    'price_cloud': 0.15,      # Cloud position
                    'future_cloud': 0.05      # Future less important
                }
                strength_threshold = 0.6
                
            else:  # "entry" - 1-minute
                weights = {
                    'tenkan_kijun': 0.30,     # TK cross for entry timing
                    'price_cloud': 0.25,      # Current position
                    'ema_cross': 0.20,        # EMA confirmation
                    'momentum': 0.15,         # Short-term momentum
                    'future_cloud': 0.10      # Future direction
                }
                strength_threshold = 0.5  # Lower threshold for entry timing
            
            # Calculate weighted alignment score
            alignment_score = 0.0
            total_possible = 0.0
            
            # Price cloud alignment
            if price_cloud != 0:
                if price_cloud == expected_direction:
                    alignment_score += weights['price_cloud']
                else:
                    alignment_score -= weights['price_cloud'] * 0.6  # Penalty for opposing
                total_possible += weights['price_cloud']
            
            # Tenkan-Kijun alignment
            if tenkan_kijun != 0:
                if tenkan_kijun == expected_direction:
                    alignment_score += weights['tenkan_kijun']
                else:
                    alignment_score -= weights['tenkan_kijun'] * 0.6
                total_possible += weights['tenkan_kijun']
            
            # Future cloud alignment
            if future_cloud != 0:
                if future_cloud == expected_direction:
                    alignment_score += weights['future_cloud']
                else:
                    alignment_score -= weights['future_cloud'] * 0.6
                total_possible += weights['future_cloud']
            
            # EMA cross alignment
            if ema_cross != 0:
                if ema_cross == expected_direction:
                    alignment_score += weights['ema_cross']
                else:
                    alignment_score -= weights['ema_cross'] * 0.6
                total_possible += weights['ema_cross']
            
            # Momentum alignment (combined tenkan and kijun momentum)
            momentum_score = 0
            momentum_count = 0
            if tenkan_momentum != 0:
                momentum_score += 1 if tenkan_momentum == expected_direction else -0.6
                momentum_count += 1
            if kijun_momentum != 0:
                momentum_score += 1 if kijun_momentum == expected_direction else -0.6
                momentum_count += 1
            
            if momentum_count > 0:
                momentum_alignment = momentum_score / momentum_count
                alignment_score += weights['momentum'] * momentum_alignment
                total_possible += weights['momentum']
            
            # Normalize the alignment score
            if total_possible > 0:
                normalized_score = alignment_score / total_possible
            else:
                normalized_score = 0
            
            # Apply strength threshold and convert to [0, 1] range
            if abs(normalized_score) >= strength_threshold:
                # Strong signal gets bonus
                strength = (abs(normalized_score) + 0.2) if normalized_score > 0 else abs(normalized_score)
            else:
                # Weak signal gets penalty
                strength = abs(normalized_score) * 0.8
            
            # Convert to [0, 1] range
            strength = max(0, min(1, strength))
            
            # Volume and LWPE modulation
            volume_boost = self._calculate_volume_boost(normalized_volume, timeframe_type)
            lwpe_boost = self._calculate_lwpe_boost(lwpe, expected_direction)
            
            # Apply boosts
            final_strength = strength * (1 + volume_boost + lwpe_boost)
            final_strength = max(0, min(1, final_strength))
            
            return final_strength
            
        except Exception as e:
            log.debug(f"Signal strength calculation error for {timeframe_type}: {e}")
            return 0.5

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
        
    def get_model_analysis(self):
        """Get detailed analysis of the enhanced model"""
        try:
            analysis = {
                'model_type': 'enhanced_multi_timeframe' if self.use_enhanced_model else 'standard',
                'feature_importance': self.feature_importance.copy(),
                'model_performance': self.model_performance.copy()
            }
            
            if self.use_enhanced_model and hasattr(self.model, 'get_timeframe_importance'):
                analysis['learned_timeframe_weights'] = self.model.get_timeframe_importance()
            
            # Calculate recent performance metrics
            if self.model_performance['attention_entropy']:
                recent_entropy = self.model_performance['attention_entropy'][-10:]
                analysis['recent_attention_entropy'] = {
                    'mean': np.mean(recent_entropy),
                    'std': np.std(recent_entropy),
                    'trend': 'increasing' if recent_entropy[-1] > recent_entropy[0] else 'decreasing'
                }
            
            return analysis
            
        except Exception as e:
            log.warning(f"Model analysis error: {e}")
            return {'error': str(e)}