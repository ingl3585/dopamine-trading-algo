# config.py

class Config:
    # File paths
    FEATURE_FILE = r"C:\\Users\\ingle\\OneDrive\\Desktop\\Actor_Critic_ML_NT\\features\\features.csv"
    MODEL_PATH   = r"C:\\Users\\ingle\\OneDrive\\Desktop\\Actor_Critic_ML_NT\\model\\actor_critic_model.pth"

    # Model architecture - Updated for Ichimoku/EMA features
    INPUT_DIM   = 9     # close, volume, tenkan_kijun, price_cloud, future_cloud, ema_cross, tenkan_momentum, kijun_momentum, lwpe
    HIDDEN_DIM  = 128   # Increased for better feature processing
    ACTION_DIM  = 3     # Hold, Long, Short
    LOOKBACK    = 1

    # Training parameters
    BATCH_SIZE  = 32
    GAMMA       = 0.95
    ENTROPY_COEF= 0.05
    LR          = 1e-4

    # Position sizing
    BASE_SIZE   = 4
    MAX_SIZE    = 10
    MIN_SIZE    = 1

    # Prediction parameters
    TEMPERATURE = 2.0

    # Feature weights for confidence calculation
    ICHIMOKU_WEIGHT = 0.25     # Weight for Ichimoku signals
    EMA_WEIGHT = 0.20          # Weight for EMA signals  
    MOMENTUM_WEIGHT = 0.20     # Weight for momentum signals
    VOLUME_WEIGHT = 0.15       # Weight for volume signals
    LWPE_WEIGHT = 0.20         # Weight for LWPE signals

    # Risk management (replacing ATR-based system)
    CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to trade
    MAX_DRAWDOWN_PCT = 0.02    # 2% max drawdown per trade
    
    # Feature normalization bounds
    PRICE_NORMALIZATION = True
    VOLUME_LOOKBACK = 20