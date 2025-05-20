# model/agent.py

import os
import time
import random
import logging
from collections import deque

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
        self.temp   = config.TEMPERATURE
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ActorCritic(config.INPUT_DIM, config.HIDDEN_DIM, config.ACTION_DIM).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.LR)
        self.replay_buffer = deque(maxlen=10_000)
        self.loss_fn = nn.SmoothL1Loss()
        self.last_save_time = 0

        self.load_model()

    def load_model(self):
        if not os.path.exists(self.config.MODEL_PATH):
            return
        try:
            self.model.load_state_dict(torch.load(self.config.MODEL_PATH, map_location=self.device))
            log.info("Model loaded from %s", self.config.MODEL_PATH)

            buf_path = self.config.MODEL_PATH.replace(".pth", "_buffer.npy")
            if os.path.exists(buf_path):
                self.replay_buffer = deque(np.load(buf_path, allow_pickle=True), maxlen=10_000)
                log.info("Replay buffer loaded  (%d samples)", len(self.replay_buffer))
        except Exception as e:
            log.warning("Model load failed: %s", e)

    def save_model(self):
        try:
            os.makedirs(os.path.dirname(self.config.MODEL_PATH), exist_ok=True)
            torch.save(self.model.state_dict(), self.config.MODEL_PATH)
            np.save(self.config.MODEL_PATH.replace(".pth", "_buffer.npy"), np.array(self.replay_buffer, dtype=object))
            log.info("Model and buffer saved")
        except Exception as e:
            log.warning("Save error: %s", e)

    def train(self, df, epochs=1):
        try:
            if isinstance(df.iloc[0, 1], str):
                df = df.iloc[1:]

            data = df.iloc[:, 1:-1].apply(pd.to_numeric, errors="coerce").fillna(0).values
            if len(data) <= self.config.LOOKBACK:
                log.warning(f"Insufficient data for training: {len(data)} samples")
                return

            sequences = torch.tensor(data, dtype=torch.float32).unfold(0, self.config.LOOKBACK, 1).transpose(1, 2)
            total_seq = len(sequences) - 1
            bar_len = 30

            log.info(f"Starting training: {total_seq} sequences × {epochs} epoch(s)")

            for epoch in range(epochs):
                epoch_loss, num_batches, bar_tick = 0, 0, 0

                print(f"\rEpoch {epoch+1}/{epochs} [{' ' * bar_len}] 0% ", end="", flush=True)
                for i in range(total_seq - 1):
                    state = sequences[i].unsqueeze(0)
                    next_state = sequences[i + 1].unsqueeze(0)

                    price_change = (next_state[0, -1, 0] - state[0, -1, 0]).item()
                    atr = state[0, -1, 2].item() if state.shape[2] > 2 else 0.01
                    reward = price_change / (atr + 1e-6)

                    self.replay_buffer.append((state, 1, reward, next_state, False))

                    if len(self.replay_buffer) >= self.config.BATCH_SIZE:
                        epoch_loss += self._update_model()
                        num_batches += 1

                    pct = (i + 1) / total_seq
                    if pct >= bar_tick / bar_len:
                        filled = "=" * bar_tick + ">" + " " * (bar_len - bar_tick - 1)
                        print(f"\rEpoch {epoch+1}/{epochs} [{filled}] {pct*100:.0f}% ", end="", flush=True)
                        bar_tick += 1

                print(f"\rEpoch {epoch+1}/{epochs} [{'=' * bar_len}] 100% ", end="", flush=True)
                print()
                if num_batches:
                    avg_loss = epoch_loss / num_batches
                    print(f"Epoch {epoch+1}/{epochs} complete - Average loss: {avg_loss:.4f}")

            if time.time() - self.last_save_time > 3600:
                self.save_model()
                self.last_save_time = time.time()

        except Exception as e:
            log.warning(f"Training error: {e}")

    def _update_model(self):
        batch = random.sample(self.replay_buffer, self.config.BATCH_SIZE)
        states, actions, rewards, next_states, _ = zip(*batch)

        states = torch.cat(states).to(self.device)
        next_states = torch.cat(next_states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            _, next_values = self.model(next_states)
            targets = rewards + self.config.GAMMA * next_values.squeeze()

        probs, values = self.model(states, temperature=0.5)
        dist = torch.distributions.Categorical(probs)
        entropy = dist.entropy().mean()
        advantage = (targets - values.squeeze()).detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        actor_loss = -(dist.log_prob(actions) * advantage).mean()
        critic_loss = self.loss_fn(values.squeeze(), targets)
        l2_reg = sum(torch.norm(p) for p in self.model.parameters())

        loss = actor_loss + 0.5 * critic_loss - self.config.ENTROPY_COEF * 1.5 * entropy + 0.001 * l2_reg

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def predict_single(self, feat_vec):
        state = np.repeat(np.asarray(feat_vec, np.float32).reshape(1, -1), self.config.LOOKBACK, 0)
        with torch.no_grad():
            probs, _ = self.model(torch.tensor(state).unsqueeze(0).to(self.device), temperature=self.temp)
            dist = torch.distributions.Categorical(probs)
            action = int(dist.sample())
            conf = float(probs[0, action])
        return action, conf

    def calculate_improved_reward(self, price_change, atr, state_data=None):
        base_reward = price_change / (atr + 1e-6)
        consistency_reward = 0

        if hasattr(self, 'recent_rewards'):
            if len(self.recent_rewards) >= 10:
                win_rate = sum(1 for r in self.recent_rewards if r > 0) / len(self.recent_rewards)
                consistency_reward = 0.2 * (1.0 - 2.0 * abs(win_rate - 0.65))

            self.recent_rewards.append(base_reward)
            if len(self.recent_rewards) > 50:
                self.recent_rewards.pop(0)
        else:
            self.recent_rewards = [base_reward]

        volatility_reward = 0
        if state_data is not None and hasattr(self, 'prev_volatility'):
            current_volatility = state_data[0, -1, 2].item()
            vol_change = abs(current_volatility - self.prev_volatility)
            if vol_change > 0.0005:
                volatility_reward = 0.1 * min(base_reward, 1.0)
            self.prev_volatility = current_volatility
        else:
            self.prev_volatility = atr if state_data is None else state_data[0, -1, 2].item()

        return base_reward + consistency_reward + volatility_reward

    def push_sample(self, feat, action, reward):
        state = torch.tensor(np.repeat(np.asarray(feat, np.float32).reshape(1, -1), self.config.LOOKBACK, 0)).unsqueeze(0)
        self.replay_buffer.append((state, action, reward, state.clone(), False))