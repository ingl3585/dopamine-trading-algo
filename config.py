# config.py

class Config:
    # File paths
    FEATURE_FILE = r"C:\\Users\\ingle\\OneDrive\\Desktop\\Actor_Critic_ML_NT\\features\\features.csv"
    MODEL_PATH   = r"C:\\Users\\ingle\\OneDrive\\Desktop\\Actor_Critic_ML_NT\\model\\actor_critic_model.pth"

    # Model architecture - Optimized for enhanced Ichimoku/EMA features
    INPUT_DIM   = 9     # close, volume, tenkan_kijun, price_cloud, future_cloud, ema_cross, tenkan_momentum, kijun_momentum, lwpe
    HIDDEN_DIM  = 128   # Sufficient for 9 ternary signal inputs
    ACTION_DIM  = 3     # Hold, Long, Short
    LOOKBACK    = 1

    # Training parameters - Adjusted for ternary signal complexity
    BATCH_SIZE  = 32
    GAMMA       = 0.95
    ENTROPY_COEF= 0.05
    LR          = 1e-4

    # Position sizing
    BASE_SIZE   = 3
    MAX_SIZE    = 8
    MIN_SIZE    = 1

    # Prediction parameters - Adjusted for better neutral signal handling
    TEMPERATURE = 1.2  # Reduced from 1.8 for less randomness

    # Enhanced feature weights for ternary signal confidence calculation
    ICHIMOKU_WEIGHT = 0.30     # Increased weight for enhanced Ichimoku signals
    EMA_WEIGHT = 0.20          # Weight for EMA signals  
    MOMENTUM_WEIGHT = 0.15     # Weight for momentum signals (can be neutral)
    VOLUME_WEIGHT = 0.15       # Weight for volume signals
    LWPE_WEIGHT = 0.20         # Weight for LWPE signals

    # Enhanced risk management with neutral signal considerations
    CONFIDENCE_THRESHOLD = 0.45  # Lower threshold to allow more trades
    MAX_DRAWDOWN_PCT = 0.02     # 2% max drawdown per trade
    
    # Signal quality thresholds
    MIN_SIGNAL_ALIGNMENT = 0.5  # Reduced from 0.6
    NEUTRAL_SIGNAL_PENALTY = 0.05  # Reduced from 0.1
    
    # Feature normalization bounds
    PRICE_NORMALIZATION = True
    VOLUME_LOOKBACK = 20
    
    # Enhanced signal processing parameters
    SIGNAL_SMOOTHING = True      # Enable signal smoothing to reduce noise
    NEUTRAL_ZONE_SIZE = 0.001    # Wider neutral zone (0.1%)
    MOMENTUM_LOOKBACK = 5        # Longer momentum lookback
    
    # Validation parameters
    MAX_NEUTRAL_SIGNALS = 6      # Increased from 4 to be less restrictive
    SIGNAL_VALIDATION_STRICT = False  # Changed to False for looser validation