import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import socket
import struct
import time
import os
import random
from collections import deque
import zmq
import json
import sys
import traceback
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rl_trader.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    from arch import arch_model
    has_arch = True
except ImportError:
    logger.warning("arch package not found. Using simple volatility estimation.")
    has_arch = False

try:
    from hmmlearn.hmm import GaussianHMM
    has_hmm = True
except ImportError:
    logger.warning("hmmlearn package not found. Using simple regime detection.")
    has_hmm = False

class Config:
    FEATURE_FILE = "C:\\Users\\ingle\\OneDrive\\Desktop\\Actor_Critic_ML_NT\\features.csv"
    SIGNAL_FILE = "C:\\Users\\ingle\\OneDrive\\Desktop\\Actor_Critic_ML_NT\\signal.txt"
    MODEL_PATH = "C:\\Users\\ingle\\OneDrive\\Desktop\\Actor_Critic_ML_NT\\actor_critic_model.pth"
    
    INPUT_DIM = 6
    HIDDEN_DIM = 128
    ACTION_DIM = 3
    
    LOOKBACK = 20
    BATCH_SIZE = 64
    GAMMA = 0.99
    ENTROPY_COEF = 0.01
    LR = 0.0005
    TRAIN_INTERVAL = 300
    
    TCP_HOST = "localhost"
    FEATURE_PORT = 5556
    SIGNAL_PORT = 5557
    SOCKET_TIMEOUT = 5.0
    RECV_BUFFER_SIZE = 4096

    USE_ZMQ = False
    POLL_INTERVAL = 0.1
    MAX_MSG_SIZE = 4096
    
    BASE_SIZE = 5
    CONSERVATIVE_SIZE = 2
    MIN_SIZE = 1

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
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.model = ActorCritic(config.INPUT_DIM, config.HIDDEN_DIM, config.ACTION_DIM).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.LR)
        self.replay_buffer = deque(maxlen=10000)
        self.loss_fn = nn.SmoothL1Loss()
        self.last_train_time = 0
        self.last_save_time = 0
        self.feature_conn = None
        self.signal_conn  = None

        
        self.load_model()
        self.print_model_status()

    def load_model(self):
        if os.path.exists(self.config.MODEL_PATH):
            try:
                state_dict = torch.load(self.config.MODEL_PATH, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info(f"Loaded model from {self.config.MODEL_PATH}")
                
                buffer_path = self.config.MODEL_PATH.replace(".pth", "_buffer.npy")
                if os.path.exists(buffer_path):
                    try:
                        buffer_data = np.load(buffer_path, allow_pickle=True)
                        self.replay_buffer = deque(buffer_data, maxlen=10000)
                        logger.info(f"Loaded replay buffer with {len(self.replay_buffer)} samples")
                    except Exception as e:
                        logger.error(f"Error loading replay buffer: {e}")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                logger.info("Initializing new model")

    def save_model(self):
        success = True
        try:
            torch.save(self.model.state_dict(), self.config.MODEL_PATH)
        except Exception as e:
            logger.error(f"Error saving model weights: {e}")
            success = False

        try:
            buffer_path = self.config.MODEL_PATH.replace(".pth", "_buffer.npy")
            np.save(buffer_path, np.array(self.replay_buffer, dtype=object))
        except Exception as e:
            logger.error(f"Error saving replay buffer: {e}")
            success = False

        if success:
            logger.info("Model and replay buffer saved")
        return success

    def setup_tcp_communication(self):
        try:
            # Feature socket
            self.feature_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.feature_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.feature_socket.bind((self.config.TCP_HOST, self.config.FEATURE_PORT))
            self.feature_socket.listen(1)
            
            # Signal socket
            self.signal_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.signal_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.signal_socket.bind((self.config.TCP_HOST, self.config.SIGNAL_PORT))
            self.signal_socket.listen(1)
            
            logger.info(f"TCP servers ready on ports {self.config.FEATURE_PORT} and {self.config.SIGNAL_PORT}")
            return True
        except Exception as e:
            logger.error(f"Failed to setup TCP: {str(e)}")
            return False

    def safe_recv(self, max_length=4096):
        """Safely receive a message with length prefix"""
        try:
            if not self.feature_conn:
                return None
                
            # First get the 4-byte length prefix
            length_bytes = self.feature_conn.recv(4, socket.MSG_WAITALL)
            if not length_bytes or len(length_bytes) != 4:
                return None
                
            # Convert length with proper endianness (little-endian)
            length = struct.unpack('<I', length_bytes)[0]
            
            # Validate length
            if length <= 0 or length > max_length:
                logger.warning(f"Invalid message length: {length} bytes (max {max_length})")
                # Discard any remaining data
                while True:
                    chunk = self.feature_conn.recv(4096)
                    if not chunk or len(chunk) < 4096:
                        break
                return None
            
            # Receive the actual message
            message = self.feature_conn.recv(length, socket.MSG_WAITALL)
            if not message or len(message) != length:
                logger.warning(f"Incomplete message. Expected {length}, got {len(message) if message else 0} bytes")
                return None
                
            return message
        except socket.timeout:
            return None
        except ConnectionResetError:
            logger.warning("Connection reset by peer")
            self.feature_conn = None
            return None
        except Exception as e:
            logger.error(f"Receive error: {str(e)[:200]}")
            return None

    def send_signal(self, signal, max_length=4096):
        """Send a signal with proper framing and validation"""
        try:
            # Validate signal structure first
            if not self.validate_signal(signal):
                logger.error("Invalid signal structure")
                return False
                
            if not self.signal_conn:
                return False
                
            # Convert to JSON and check size
            signal_str = json.dumps(signal, separators=(',', ':'))  # Compact JSON
            if len(signal_str) > max_length:
                logger.error(f"Signal too large: {len(signal_str)} bytes")
                return False
                
            # Send with length prefix
            msg = signal_str.encode('utf-8')
            length = struct.pack('<I', len(msg))
            self.signal_conn.sendall(length + msg)
            return True
            
        except socket.timeout:
            logger.warning("Signal send timeout")
            return False
        except BrokenPipeError:
            logger.warning("Signal connection broken")
            self.signal_conn = None
            return False
        except Exception as e:
            logger.error(f"Signal send error: {str(e)[:200]}")
            return False

    def perform_handshake(self, timeout=30):
        """Simplified handshake - waits for NinjaScript's READY first"""
        logger.info("Waiting for NinjaScript to connect and send READY...")
        
        try:
            # Accept connections if not already connected
            if not self.feature_conn or not self.signal_conn:
                # Accept feature connection with timeout
                self.feature_socket.settimeout(timeout)
                self.feature_conn, _ = self.feature_socket.accept()
                
                # Accept signal connection with timeout
                self.signal_socket.settimeout(timeout)
                self.signal_conn, _ = self.signal_socket.accept()
            
            # Wait for READY from NinjaScript (feature socket)
            ready_msg = self.feature_conn.recv(5)  # Should be "READY"
            if ready_msg != b'READY':
                logger.error(f"Unexpected handshake message: {ready_msg}")
                return False
                
            # Send our READY response (signal socket)
            self.signal_conn.sendall(b'READY')
            logger.info("Handshake completed successfully")
            return True
            
        except socket.timeout:
            logger.error("Handshake timed out waiting for NinjaScript")
            return False
        except Exception as e:
            logger.error(f"Handshake error: {str(e)}")
            return False

    def verify_connection(self):
        """Verify the connection is alive"""
        try:
            if not self.signal_conn or not self.feature_conn:
                return False
                
            # Send ping
            self.signal_conn.sendall(b'PING')
            
            # Wait for pong
            start = time.time()
            while time.time() - start < 1.0:
                try:
                    msg = self.feature_conn.recv(4, socket.MSG_WAITALL)
                    return msg == b'PONG'
                except socket.timeout:
                    continue
                    
            return False
        except Exception as e:
            logger.error(f"Connection verification failed: {e}")
            return False

    def validate_signal(self, signal):
        """Validate signal structure and values"""
        if not isinstance(signal, dict):
            return False
        
        required = {
            'action': lambda x: x in [0, 1, 2],
            'confidence': lambda x: 0 <= x <= 1,
            'size': lambda x: x >= 1,
            'regime': lambda x: x in [0, 1],
            'volatility': lambda x: x >= 0,
            'value_uncertainty': lambda x: x >= 0,
            'timestamp': lambda x: x > 0
        }
        
        return all(
            key in signal and validator(signal[key])
            for key, validator in required.items()
        )
        
    def print_model_status(self):
        total_params = 0
        logger.info("Model Architecture:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_count = param.numel()
                total_params += param_count
                logger.info(f"  {name}: {param_count} params (mean={param.data.mean():.4f}, std={param.data.std():.4f})")
        logger.info(f"Total trainable parameters: {total_params}")
        logger.info(f"Replay buffer: {len(self.replay_buffer)}/{self.config.BATCH_SIZE} samples")

    def train(self, df, epochs=1):
        try:
            start_time = time.time()
            
            if isinstance(df.iloc[0, 1], str):
                df = df.iloc[1:]
            
            data = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0).values
            if len(data) <= self.config.LOOKBACK:
                logger.warning(f"Insufficient data for training: {len(data)} samples")
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
                    logger.info(f"Epoch {epoch+1}/{epochs} complete - Avg loss: {avg_loss:.4f} - Buffer: {len(self.replay_buffer)} samples")
            
            if time.time() - self.last_save_time > 3600:
                if self.save_model():
                    self.last_save_time = time.time()
            
            logger.info(f"Training completed in {time.time()-start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Training error: {traceback.format_exc()}")

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

    def predict(self, state_np):
        with torch.no_grad():
            if not isinstance(state_np, np.ndarray):
                raise ValueError("Input must be numpy array")
                
            if np.isnan(state_np).any() or np.isinf(state_np).any():
                logger.warning("Invalid state (NaN/Inf values)")
                return 1, 0.5, 0.0
                
            if state_np.size != self.config.INPUT_DIM * self.config.LOOKBACK:
                logger.warning(f"Invalid shape: {state_np.shape}")
                return 1, 0.5, 0.0
                
            state = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            probs_list = []
            values = []
            for _ in range(5):
                probs, value = self.model(state)
                probs_list.append(probs.cpu())
                values.append(value.item())
            
            probs = torch.mean(torch.stack(probs_list), dim=0)
            value_uncertainty = np.std(values)
            action = torch.argmax(probs).item()
            confidence = probs[0][action].item()
            
            return action, confidence, value_uncertainty

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
                    logger.warning(f"HMM regime detection failed: {e}")
            
            ma_short = np.mean(prices[-10:])
            ma_long = np.mean(prices[-20:])
            return 0 if ma_short > ma_long else 1
            
        except Exception as e:
            logger.error(f"Regime detection error: {e}")
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
                    logger.warning(f"GARCH failed: {e}")
            
            return returns.ewm(span=window).std().iloc[-1] if not returns.empty else 0.01
            
        except Exception as e:
            logger.error(f"Volatility forecasting error: {e}")
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
                logger.warning(f"CSV contains non-numeric data in attempt {attempt+1}")
                if attempt == max_retries - 1:
                    df = df.dropna()
                    if len(df) == 0:
                        return None
                    return df
                continue
                
            return df
            
        except Exception as e:
            logger.warning(f"CSV read error (attempt {attempt+1}): {e}")
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
            logger.info(f"Trimmed feature file to {len(new_lines)} lines")
    except Exception as e:
        logger.error(f"File cleaning error: {e}")

def main():
    config = Config()
    agent = RLAgent(config)
    utils = MarketUtils()

    print("Waiting 30s for NinjaTrader to start…")
    time.sleep(30)

    # First setup communication
    if not agent.setup_tcp_communication():
        logger.error("Failed to setup TCP - falling back to file I/O")
        config.USE_ZMQ = False
    else:
        # give NT a beat
        time.sleep(10)
        if not agent.perform_handshake():
            logger.error("Handshake failed - falling back to file I/O")
            config.USE_ZMQ = False
        else:
            config.USE_ZMQ = True
    
    # Only then proceed with training
    if config.USE_ZMQ:
        df = safe_read_csv(config.FEATURE_FILE)
        if df is not None:
            agent.train(df)
    
    try:
        if not os.path.exists(config.FEATURE_FILE):
            with open(config.FEATURE_FILE, 'w') as f:
                f.write("Time,Close,FastEMA,SlowEMA,RSI,ATR,Volume\n")
            logger.info("Initialized feature file")
    except Exception as e:
        logger.error(f"Feature file initialization failed: {e}")
        sys.exit(1)
    
    df = safe_read_csv(config.FEATURE_FILE)
    if df is not None and len(df) > config.LOOKBACK + 1:
        logger.info(f"Starting initial training with {len(df)} samples")
        agent.train(df, epochs=3)
    
    logger.info("Starting main trading loop...")
    last_features_hash = None
    last_clean_time = time.time()
    
    try:
        while True:
            try:
                features = None
                
                if config.USE_ZMQ:
                    # Handle incoming connections if not connected
                    if not agent.feature_conn:
                        try:
                            agent.feature_conn, _ = agent.feature_socket.accept()
                            agent.feature_conn.settimeout(config.SOCKET_TIMEOUT)
                            logger.info("Accepted feature connection")
                        except socket.timeout:
                            pass
                            
                    if not agent.signal_conn:
                        try:
                            agent.signal_conn, _ = agent.signal_socket.accept()
                            agent.signal_conn.settimeout(config.SOCKET_TIMEOUT)
                            logger.info("Accepted signal connection")
                        except socket.timeout:
                            pass
                    
                    try:
                        message = agent.safe_recv()
                        if message == b'PING':
                            agent.feature_conn.sendall(b'PONG')
                            continue

                        if not message:
                            time.sleep(config.POLL_INTERVAL)
                            continue
                                
                        try:
                            data = json.loads(message.decode('utf-8'))
                            features = np.array(data['features'], dtype=np.float32).reshape(1, -1)
                        except json.JSONDecodeError:
                            logger.warning("Invalid JSON received")
                            continue
                        
                    except Exception as e:
                        logger.error(f"Feature receive error: {e}")
                        continue
                else:
                    df = safe_read_csv(config.FEATURE_FILE)
                    if df is not None and len(df) >= config.LOOKBACK:
                        numeric_data = df.iloc[-config.LOOKBACK:, 1:].apply(pd.to_numeric, errors='coerce')
                        features = numeric_data.fillna(0).values
                
                if features is None:
                    time.sleep(config.POLL_INTERVAL)
                    continue
                
                current_hash = hash(features.tobytes())
                if last_features_hash == current_hash:
                    time.sleep(config.POLL_INTERVAL)
                    continue
                last_features_hash = current_hash
                
                if features.shape[0] == 1 and config.LOOKBACK > 1:
                    features_expanded = np.repeat(features, config.LOOKBACK, axis=0)
                    action, confidence, value_uncertainty = agent.predict(features_expanded)
                else:
                    action, confidence, value_uncertainty = agent.predict(features)
                
                close_prices = features[:, 0] if features.shape[0] > 1 else np.array([features[0, 0]])
                regime = utils.detect_regime(close_prices)
                volatility = utils.forecast_volatility(close_prices)
                
                base_size = config.BASE_SIZE if regime == 0 else config.CONSERVATIVE_SIZE
                size = int(confidence * base_size / max(volatility, 0.01))
                size = np.clip(size, config.MIN_SIZE, base_size)
                
                signal = {
                    'action': int(action),
                    'confidence': float(confidence),
                    'size': int(size),
                    'regime': int(regime),
                    'volatility': float(volatility),
                    'value_uncertainty': float(value_uncertainty),
                    'timestamp': int(time.time())
                }
                
                if config.USE_ZMQ and agent.signal_conn:
                    agent.send_signal(signal)
                else:
                    with open(config.SIGNAL_FILE, 'w') as f:
                        json.dump(signal, f)
                
                logger.info(
                    f"Decision: {['Long', 'Flat', 'Short'][action]} | "
                    f"Confidence: {confidence:.1%} | Size: {size} | "
                    f"Regime: {['Trending', 'Choppy'][regime]} | "
                    f"Volatility: {volatility:.4f}"
                )
                
                current_time = time.time()
                
                if (len(agent.replay_buffer) >= config.BATCH_SIZE or 
                    current_time - agent.last_train_time > config.TRAIN_INTERVAL):
                    df = safe_read_csv(config.FEATURE_FILE)
                    if df is not None and len(df) > config.LOOKBACK + 1:
                        logger.info(f"Starting training with {len(df)} samples")
                        agent.train(df, epochs=2)
                        agent.last_train_time = current_time
                
                if current_time - last_clean_time > 3600:
                    clean_feature_file(config.FEATURE_FILE)
                    last_clean_time = current_time
                
                if random.random() < 0.05:
                    agent.print_model_status()

                if random.random() < 0.01:  # 1% chance to log message sizes
                    msg_size = len(json.dumps(signal))
                    logger.info(f"Current message size: {msg_size} bytes")
                    if msg_size > 2000:  # Warning threshold
                        logger.warning(f"Large message detected: {msg_size} bytes")
                
            except Exception as e:
                logger.error(f"Main loop error: {traceback.format_exc()}")
                time.sleep(5)
    
    except KeyboardInterrupt:
        logger.info("Shutdown initiated...")
        agent.save_model()
        
        # Close sockets
        if hasattr(agent, 'feature_conn') and agent.feature_conn:
            agent.feature_conn.close()
        if hasattr(agent, 'signal_conn') and agent.signal_conn:
            agent.signal_conn.close()
        if hasattr(agent, 'feature_socket') and agent.feature_socket:
            agent.feature_socket.close()
        if hasattr(agent, 'signal_socket') and agent.signal_socket:
            agent.signal_socket.close()
        
        logger.info("Shutdown complete")
        sys.exit(0)

if __name__ == "__main__":
    main()