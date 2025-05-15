#!/usr/bin/env python
# ───────────────────────────────────────────────────────────────
#  actor_critic.py  –  simplified sockets, full ML intact
#  * list‑ens on 5556 (features) and 5557 (signals)
#  * one background thread to read feature packets
#  * send_signal() pushes JSON back to Ninja
#  * rest of the file = original RL agent / training code
# ───────────────────────────────────────────────────────────────
import socket, struct, json, threading, time, logging
import os, sys, random, traceback
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ───── logging ─────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler("rl_trader.log"), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

try:
    from arch import arch_model
    has_arch = True
except ImportError:
    log.warning("arch package not found. Using simple volatility estimation.")
    has_arch = False

try:
    from hmmlearn.hmm import GaussianHMM
    has_hmm = True
except ImportError:
    log.warning("hmmlearn package not found. Using simple regime detection.")
    has_hmm = False

HOST, FEAT_PORT, SIG_PORT = "localhost", 5556, 5557

class IO:
    """Two‑socket helper: receives feature vectors, sends signals."""
    def __init__(self):
        self._feat_srv = socket.socket(); self._feat_srv.bind((HOST, FEAT_PORT)); self._feat_srv.listen(1)
        self._sig_srv  = socket.socket(); self._sig_srv.bind((HOST, SIG_PORT));  self._sig_srv.listen(1)
        log.info("▲ waiting for NinjaTrader …")

        self.fsock, _ = self._feat_srv.accept()
        self.ssock, _ = self._sig_srv.accept()
        log.info("✔ NinjaTrader connected")

        self.on_features = lambda feat: None     # callback placeholder
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        stream = self.fsock
        while True:
            try:
                hdr = stream.recv(4, socket.MSG_WAITALL)
                if not hdr: break
                n = struct.unpack('<I', hdr)[0]
                data = stream.recv(n, socket.MSG_WAITALL)
                if len(data) != n: continue
                feat = json.loads(data.decode())['features']
                self.on_features(feat)
            except Exception as e:
                log.warning("recv error: %s", e); break

    def send_signal(self, sig: dict):
        try:
            blob = json.dumps(sig, separators=(',', ':')).encode()
            self.ssock.sendall(struct.pack('<I', len(blob)) + blob)
        except Exception as e:
            log.warning("send error: %s", e)

class Config:
    FEATURE_FILE = "C:\\Users\\ingle\\OneDrive\\Desktop\\Actor_Critic_ML_NT\\features.csv"
    MODEL_PATH = "C:\\Users\\ingle\\OneDrive\\Desktop\\Actor_Critic_ML_NT\\actor_critic_model.pth"
    
    INPUT_DIM   = 6
    HIDDEN_DIM  = 128
    ACTION_DIM  = 3
    LOOKBACK    = 20

    BATCH_SIZE  = 64
    GAMMA       = 0.99
    ENTROPY_COEF= 0.01
    LR          = 5e-4

    BASE_SIZE = 5
    CONS_SIZE = 2
    MIN_SIZE  = 1

class BayesianLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.dropout(lstm_out[:, -1, :])

class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super().__init__()
        self.feature_extractor = BayesianLSTM(input_dim, hidden_dim)
        
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim))
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1))
        
    def forward(self, x, temperature=1.0):
        features = self.feature_extractor(x)
        logits = self.actor(features) / temperature
        probs = F.softmax(logits, dim=-1)
        value = self.critic(features)
        return probs, value

class RLAgent:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("device %s", self.device)

        self.model = ActorCritic(config.INPUT_DIM,
                                 config.HIDDEN_DIM,
                                 config.ACTION_DIM).to(self.device)

        self.optimizer      = optim.AdamW(self.model.parameters(), lr=config.LR)
        self.replay_buffer  = deque(maxlen=10_000)
        self.loss_fn        = nn.SmoothL1Loss()
        self.last_save_time = 0                #  epoch timer

        self.load_model()      

    def load_model(self):
        if not os.path.exists(self.config.MODEL_PATH):
            return
        try:
            self.model.load_state_dict(
                torch.load(self.config.MODEL_PATH, map_location=self.device)
            )
            log.info("✓ model loaded from %s", self.config.MODEL_PATH)

            buf_path = self.config.MODEL_PATH.replace(".pth", "_buffer.npy")
            if os.path.exists(buf_path):
                self.replay_buffer = deque(np.load(buf_path, allow_pickle=True),
                                           maxlen=10_000)
                log.info("✓ replay buffer loaded  (%d samples)",
                         len(self.replay_buffer))
        except Exception as e:
            log.error("model load failed: %s", e)

    def save_model(self):
        try:
            torch.save(self.model.state_dict(), self.config.MODEL_PATH)
            np.save(self.config.MODEL_PATH.replace(".pth", "_buffer.npy"),
                    np.array(self.replay_buffer, dtype=object))
            log.info("✓ model + buffer saved")
        except Exception as e:
            log.error("save error: %s", e)
        
    def print_model_status(self):
        tot = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        log.info("model params: %d   replay: %d/%d",
                 tot, len(self.replay_buffer), self.config.BATCH_SIZE)

    def train(self, df, epochs=1):
        try:
            start_time = time.time()
            
            if isinstance(df.iloc[0, 1], str):
                df = df.iloc[1:]
            
            data = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0).values
            if len(data) <= self.config.LOOKBACK:
                log.warning(f"Insufficient data for training: {len(data)} samples")
                return
                
            data_tensor = torch.tensor(data, dtype=torch.float32)
            sequences = data_tensor.unfold(0, self.config.LOOKBACK, 1).transpose(1, 2)
            
            for epoch in range(epochs):
                epoch_loss = 0
                num_batches = 0
                
                for i in range(len(sequences) - 1):
                    state = sequences[i].unsqueeze(0)
                    next_state = sequences[i+1].unsqueeze(0)
                    
                    price_change = (next_state[0, -1, 0] - state[0, -1, 0]).item()
                    atr = state[0, -1, 4].item() if state.shape[2] > 4 else 0.01
                    reward = price_change / (atr + 1e-6)
                    
                    self.replay_buffer.append((state, None, reward, next_state, False))
                    
                    if len(self.replay_buffer) >= self.config.BATCH_SIZE:
                        loss = self._update_model()
                        epoch_loss += loss
                        num_batches += 1
                
                if num_batches > 0:
                    avg_loss = epoch_loss / num_batches
                    log.info(f"Epoch {epoch+1}/{epochs} complete - Avg loss: {avg_loss:.4f} - Buffer: {len(self.replay_buffer)} samples")
            
            if time.time() - self.last_save_time > 3600:
                if self.save_model():
                    self.last_save_time = time.time()
            
            log.info(f"Training completed in {time.time()-start_time:.2f} seconds")
            
        except Exception as e:
            log.error(f"Training error: {traceback.format_exc()}")

    def _update_model(self):
        batch = random.sample(self.replay_buffer, self.config.BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.cat(states).to(self.device)
        next_states = torch.cat(next_states).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            _, next_values = self.model(next_states)
            targets = rewards + self.config.GAMMA * next_values.squeeze()
        
        _, current_values = self.model(states)
        critic_loss = self.loss_fn(current_values.squeeze(), targets)
        
        probs, _ = self.model(states, temperature=0.5)
        dist = torch.distributions.Categorical(probs)
        entropy = dist.entropy().mean()
        advantage = targets - current_values.detach()
        actor_loss = -(dist.log_prob(torch.argmax(probs, dim=-1)) * advantage).mean()
        
        loss = actor_loss + 0.5 * critic_loss - self.config.ENTROPY_COEF * entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()

    def predict_single(self, feat_vec):
        """feat_vec: list[float] length==6 (latest bar only)."""
        state = np.repeat(np.asarray(feat_vec, np.float32)
                          .reshape(1, -1), self.config.LOOKBACK, 0)

        with torch.no_grad():
            probs, _ = self.model(torch.tensor(state)
                                  .unsqueeze(0).to(self.device))
            action   = int(torch.argmax(probs))
            conf     = float(probs[0, action])
        return action, conf

class MarketUtils:
    @staticmethod
    def detect_regime(prices, window=50):
        try:
            if len(prices) < 20:
                return 0
                
            if has_hmm and len(prices) > window:
                try:
                    returns = np.diff(np.log(prices))
                    features = np.column_stack([
                        returns,
                        pd.Series(returns).rolling(5).std().fillna(0).values,
                        pd.Series(prices).pct_change().rolling(10).mean().fillna(0).values
                    ])[1:]
                    
                    if len(features) > 10:
                        model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=100)
                        model.fit(features)
                        return model.predict(features)[-1]
                except Exception as e:
                    log.warning(f"HMM regime detection failed: {e}")
            
            ma_short = np.mean(prices[-10:])
            ma_long = np.mean(prices[-20:])
            return 0 if ma_short > ma_long else 1
            
        except Exception as e:
            log.error(f"Regime detection error: {e}")
            return 0

    @staticmethod
    def forecast_volatility(prices, window=14):
        try:
            if len(prices) < window + 1:
                return 0.01
                
            returns = 100 * pd.Series(prices).pct_change().dropna()
            
            if has_arch and len(returns) > 30:
                try:
                    model = arch_model(returns, vol='Garch', p=1, q=1)
                    res = model.fit(disp='off')
                    return np.sqrt(res.forecast(horizon=1).variance.values[-1, 0])
                except Exception as e:
                    log.warning(f"GARCH failed: {e}")
            
            return returns.ewm(span=window).std().iloc[-1] if not returns.empty else 0.01
            
        except Exception as e:
            log.error(f"Volatility forecasting error: {e}")
            return 0.01

def safe_read_csv(file_path, max_retries=3):
    for attempt in range(max_retries):
        try:
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                return None
                
            df = pd.read_csv(file_path, header=None)
            if len(df) == 0:
                return None
                
            numeric_df = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
            if numeric_df.isna().any().any():
                log.warning(f"CSV contains non-numeric data in attempt {attempt+1}")
                if attempt == max_retries - 1:
                    df = df.dropna()
                    if len(df) == 0:
                        return None
                    return df
                continue
                
            return df
            
        except Exception as e:
            log.warning(f"CSV read error (attempt {attempt+1}): {e}")
            time.sleep(1)
    
    return None

def clean_feature_file(file_path, max_lines=10000, max_size_mb=10):
    try:
        if os.path.getsize(file_path) > max_size_mb * 1024 * 1024:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            keep_lines = max(max_lines, 1000)
            new_lines = [lines[0]] + lines[-keep_lines:]
            
            with open(file_path, 'w') as f:
                f.writelines(new_lines)
            log.info(f"Trimmed feature file to {len(new_lines)} lines")
    except Exception as e:
        log.error(f"File cleaning error: {e}")

def main():
    cfg    = Config()
    agent  = RLAgent(cfg)
    io     = IO()

    # — callback executed every time NT sends a feature block —
    def handle_feat(feat):
        action, conf = agent.predict_single(feat)
        sig = {
            "action": action,
            "confidence": conf,
            "size": max(cfg.MIN_SIZE, int(conf * (cfg.BASE_SIZE if action!=1 else cfg.CONS_SIZE))),
            "timestamp": int(time.time())
        }
        io.send_signal(sig)
        log.info("sent signal %s", sig)

    io.on_features = handle_feat

    # keep the script running
    while True: time.sleep(3600)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("terminated by user")