import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import os
import random
from collections import deque
import zmq
import json
import sys
import traceback
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rl_trader.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Try to import arch_model for volatility forecasting
try:
    from arch import arch_model
    has_arch = True
except ImportError:
    logger.warning("arch package not found. Using simple volatility estimation instead.")
    has_arch = False

# Try to import GaussianHMM for regime detection
try:
    from hmmlearn.hmm import GaussianHMM
    has_hmm = True
except ImportError:
    logger.warning("hmmlearn package not found. Using simple regime detection instead.")
    has_hmm = False

# ------------------------- Configuration -------------------------
class Config:
    # Paths
    FEATURE_FILE = "C:\\Users\\ingle\\OneDrive\\Desktop\\Actor_Critic_ML_NT\\features.csv"
    SIGNAL_FILE = "C:\\Users\\ingle\\OneDrive\\Desktop\\Actor_Critic_ML_NT\\signal.txt"
    MODEL_PATH = "C:\\Users\\ingle\\OneDrive\\Desktop\\Actor_Critic_ML_NT\\actor_critic_model.pth"
    
    # Network Architecture
    INPUT_DIM = 6  # Close, FastEMA, SlowEMA, RSI, ATR, Volume
    HIDDEN_DIM = 128
    ACTION_DIM = 3  # [Long, Flat, Short]
    
    # Training Parameters
    LOOKBACK = 20
    BATCH_SIZE = 64
    GAMMA = 0.99
    ENTROPY_COEF = 0.01
    LR = 0.0005
    
    # Communication
    FEATURE_PORT = 5556  # For receiving features from NT
    SIGNAL_PORT = 5557   # For sending signals to NT
    USE_ZMQ = True
    POLL_INTERVAL = 0.1  # Seconds
    
    # Trading Parameters
    BASE_SIZE = 5
    CONSERVATIVE_SIZE = 2
    MIN_SIZE = 1

# ------------------------- Enhanced Bayesian LSTM -------------------------
class BayesianLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.dropout(lstm_out[:, -1, :])

# ------------------------- Actor-Critic Network -------------------------
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

# ------------------------- RL Agent with Experience Replay -------------------------
class RLAgent:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.model = ActorCritic(config.INPUT_DIM, config.HIDDEN_DIM, config.ACTION_DIM).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.LR)
        self.replay_buffer = deque(maxlen=5000)
        self.loss_fn = nn.SmoothL1Loss()
        
        if os.path.exists(config.MODEL_PATH):
            try:
                self.model.load_state_dict(torch.load(config.MODEL_PATH, map_location=self.device))
                logger.info("Loaded existing model")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
        
        self.setup_zmq_communication()

    def setup_zmq_communication(self):
        if self.config.USE_ZMQ:
            try:
                self.context = zmq.Context()
                
                # Feature receiver (PULL)
                self.feature_socket = self.context.socket(zmq.PULL)
                self.feature_socket.bind(f"tcp://*:{self.config.FEATURE_PORT}")
                
                # Signal publisher (PUB)
                self.signal_socket = self.context.socket(zmq.PUB)
                self.signal_socket.bind(f"tcp://*:{self.config.SIGNAL_PORT}")
                
                logger.info(f"ZMQ server started - Features: {self.config.FEATURE_PORT}, Signals: {self.config.SIGNAL_PORT}")
            except Exception as e:
                logger.error(f"ZMQ setup error: {e}")
                self.config.USE_ZMQ = False

    def train(self, df, epochs=1):
        try:
            logger.info(f"Training on {len(df)} samples, {epochs} epochs")
            
            # Skip first row which might be header
            if isinstance(df.iloc[0, 1], str):
                logger.info("Skipping header row")
                df = df.iloc[1:]
            
            # Ensure all data is numeric
            data = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0).values
            data = torch.tensor(data, dtype=torch.float32)
            
            if len(data) <= self.config.LOOKBACK:
                logger.warning(f"Not enough data for training: {len(data)} samples")
                return
                
            sequences = data.unfold(0, self.config.LOOKBACK, 1).transpose(1, 2)
            
            for i in range(len(sequences) - 1):
                state = sequences[i].unsqueeze(0)
                next_state = sequences[i+1].unsqueeze(0)
                price_change = (next_state[0, -1, 0] - state[0, -1, 0]).item()
                atr = state[0, -1, 4].item() if state.shape[2] > 4 else 0.01
                reward = price_change / (atr + 1e-6)
                self.replay_buffer.append((state, None, reward, next_state, False))
                
                if len(self.replay_buffer) >= self.config.BATCH_SIZE:
                    self._update_model()
            
            logger.info(f"Training complete. Buffer size: {len(self.replay_buffer)}")
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
        
        if random.random() < 0.01:
            try:
                torch.save(self.model.state_dict(), self.config.MODEL_PATH)
            except Exception as e:
                logger.error(f"Error saving model: {e}")

    def predict(self, state_np):
        with torch.no_grad():
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

# ------------------------- Market Analysis Utilities -------------------------
class MarketUtils:
    @staticmethod
    def detect_regime(prices, window=50):
        try:
            if has_hmm and len(prices) > window:
                returns = np.diff(np.log(prices))
                features = np.column_stack([
                    returns,
                    pd.Series(returns).rolling(5).std().fillna(0).values,
                    pd.Series(prices).pct_change().rolling(10).mean().fillna(0).values
                ])[1:]
                
                if len(features) > 10:  # Need sufficient data
                    model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=100)
                    model.fit(features)
                    return model.predict(features)[-1]
            
            # Simple alternative: trend detection using moving average
            if len(prices) > 20:
                ma_short = np.mean(prices[-10:])
                ma_long = np.mean(prices[-20:])
                return 0 if ma_short > ma_long else 1  # 0=Trending, 1=Choppy
            
            return 0  # Default to trending
        except Exception as e:
            logger.error(f"Regime detection error: {e}")
            return 0

    @staticmethod
    def forecast_volatility(prices, window=14):
        try:
            if len(prices) < window + 1:
                return 0.01  # Default low volatility
                
            returns = 100 * pd.Series(prices).pct_change().dropna()
            
            if has_arch and len(returns) > 30:
                try:
                    model = arch_model(returns, vol='Garch', p=1, q=1)
                    res = model.fit(disp='off')
                    return np.sqrt(res.forecast(horizon=1).variance.values[-1, 0])
                except:
                    pass  # Fall through to simple method
            
            # Simple alternative
            return returns.ewm(span=window).std().iloc[-1] if not returns.empty else 0.01
        except Exception as e:
            logger.error(f"Volatility forecasting error: {e}")
            return 0.01  # Default value

# ------------------------- CSV Safe Reading -------------------------
def safe_read_csv(file_path, retry_count=3, delay=1):
    """Safely read CSV with retries and proper error handling"""
    for attempt in range(retry_count):
        try:
            # First check if file exists and is not empty
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                logger.warning(f"File {file_path} doesn't exist or is empty")
                return None
                
            # Try to read with pandas
            df = pd.read_csv(file_path, header=None)
            
            # Validate data
            if len(df) == 0:
                logger.warning("CSV file is empty")
                return None
            
            return df
        except pd.errors.EmptyDataError:
            logger.warning("CSV file is empty")
            return None
        except pd.errors.ParserError as e:
            logger.warning(f"CSV parsing error (attempt {attempt+1}/{retry_count}): {e}")
            
            # Try to fix the file if it's the last attempt
            if attempt == retry_count - 1:
                try:
                    # Read raw file and fix potential issues
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                    
                    # Check for consistent columns
                    if lines:
                        first_line_cols = len(lines[0].strip().split(','))
                        fixed_lines = [lines[0]]  # Keep header
                        
                        for line in lines[1:]:
                            cols = line.strip().split(',')
                            if len(cols) == first_line_cols:
                                fixed_lines.append(line)
                            else:
                                logger.warning(f"Skipping inconsistent line: {line.strip()}")
                        
                        # Write fixed file
                        backup_path = file_path + ".bak"
                        os.rename(file_path, backup_path)
                        with open(file_path, 'w') as f:
                            f.writelines(fixed_lines)
                        
                        logger.info(f"Fixed CSV file. Backup saved to {backup_path}")
                        return pd.read_csv(file_path, header=None)
                except Exception as fix_e:
                    logger.error(f"Failed to fix CSV: {fix_e}")
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            time.sleep(delay)
    
    return None

# ------------------------- Main Trading Loop -------------------------
def main():
    config = Config()
    agent = RLAgent(config)
    utils = MarketUtils()
    
    # Initialize the feature file if it doesn't exist or is invalid
    feature_path = config.FEATURE_FILE
    try:
        if not os.path.exists(feature_path):
            with open(feature_path, 'w') as f:
                f.write("Time,Close,FastEMA,SlowEMA,RSI,ATR,Volume\n")
            logger.info(f"Created new features file: {feature_path}")
    except Exception as e:
        logger.error(f"Error creating feature file: {e}")
    
    # Initial training if data exists
    df = safe_read_csv(config.FEATURE_FILE)
    if df is not None and len(df) > config.LOOKBACK + 1:
        logger.info("Starting initial training...")
        agent.train(df, epochs=1)
    
    logger.info("Starting trading loop...")
    last_features_hash = None
    
    try:
        while True:
            try:
                features = None
                
                if config.USE_ZMQ:
                    try:
                        message = agent.feature_socket.recv_string(flags=zmq.NOBLOCK)
                        data = json.loads(message)
                        features = np.array(data['features'], dtype=np.float32).reshape(1, -1)
                    except zmq.Again:
                        time.sleep(config.POLL_INTERVAL)
                        continue
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON received")
                        continue
                else:
                    df = safe_read_csv(config.FEATURE_FILE)
                    if df is not None and len(df) >= config.LOOKBACK:
                        # Extract required columns (skip timestamp)
                        numeric_data = df.iloc[-config.LOOKBACK:, 1:].apply(pd.to_numeric, errors='coerce')
                        features = numeric_data.fillna(0).values
                
                if features is None:
                    time.sleep(config.POLL_INTERVAL)
                    continue
                
                # Check if features are new using hash
                current_hash = hash(features.tobytes())
                if last_features_hash is not None and current_hash == last_features_hash:
                    time.sleep(config.POLL_INTERVAL)
                    continue
                    
                last_features_hash = current_hash
                
                # Ensure proper dimensions for predict method
                if features.shape[0] == 1:  # Single row
                    # Create lookback window of identical rows if necessary
                    if config.LOOKBACK > 1:
                        features_expanded = np.repeat(features, config.LOOKBACK, axis=0)
                        action, confidence, value_uncertainty = agent.predict(features_expanded)
                    else:
                        action, confidence, value_uncertainty = agent.predict(features)
                else:
                    action, confidence, value_uncertainty = agent.predict(features)
                
                # Extract close prices for analysis
                if features.shape[0] == 1:
                    # Use single price if only one row
                    close_prices = np.array([features[0, 0]])
                else:
                    close_prices = features[:, 0]  # First column is Close price
                
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
                
                if config.USE_ZMQ:
                    agent.signal_socket.send_json(signal)
                else:
                    with open(config.SIGNAL_FILE, 'w') as f:
                        json.dump(signal, f)
                
                logger.info(f"Action: {['Long', 'Flat', 'Short'][action]} | "
                           f"Confidence: {confidence:.1%} | "
                           f"Size: {size} | "
                           f"Regime: {['Trending', 'Choppy'][regime]} | "
                           f"Volatility: {volatility:.4f} | "
                           f"Uncertainty: {value_uncertainty:.4f}")
                
                # To this more aggressive training schedule:
                if len(agent.replay_buffer) >= config.BATCH_SIZE or (time.time() % 300 < config.POLL_INTERVAL):  # Train every 5 minutes or when buffer is full
                    df = safe_read_csv(config.FEATURE_FILE)
                    if df is not None and len(df) > config.LOOKBACK + 1:
                        logger.info(f"Starting training with {len(df)} samples...")
                        agent.train(df, epochs=3)  # Increased epochs
            
            except Exception as e:
                logger.error(f"Main loop error: {traceback.format_exc()}")
                time.sleep(5)
    
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
        if config.USE_ZMQ:
            agent.feature_socket.close()
            agent.signal_socket.close()
            agent.context.term()
        sys.exit(0)

if __name__ == "__main__":
    main()