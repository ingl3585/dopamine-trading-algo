# config.py

class Config:
    # File paths
    FEATURE_FILE = r"C:\\Users\\ingle\\OneDrive\\Desktop\\Actor_Critic_ML_NT\\features\\features.csv"
    MODEL_PATH   = r"C:\\Users\\ingle\\OneDrive\\Desktop\\Actor_Critic_ML_NT\\model\\actor_critic_model.pth"

    # Model architecture - Optimized for Ichimoku/EMA features
    INPUT_DIM   = 9     # close, volume, tenkan_kijun, price_cloud, future_cloud, ema_cross, tenkan_momentum, kijun_momentum, lwpe
    HIDDEN_DIM  = 128   # Sufficient for 9 ternary signal inputs
    ACTION_DIM  = 3     # Hold, Long, Short
    LOOKBACK    = 1

    # Training parameters - Focused on signal quality
    BATCH_SIZE  = 32
    GAMMA       = 0.95
    ENTROPY_COEF= 0.05
    LR          = 1e-4

    # REMOVED: Position sizing parameters (now handled by NinjaScript)
    # BASE_SIZE, MAX_SIZE, MIN_SIZE moved to NinjaScript properties

    # Prediction parameters - Optimized for signal confidence
    TEMPERATURE = 1.2  # For exploration during signal generation

    # Enhanced feature weights for signal confidence calculation
    ICHIMOKU_WEIGHT = 0.30     # Ichimoku signals weight
    EMA_WEIGHT = 0.20          # EMA signals weight
    MOMENTUM_WEIGHT = 0.15     # Momentum signals weight
    VOLUME_WEIGHT = 0.15       # Volume signals weight
    LWPE_WEIGHT = 0.20         # LWPE signals weight

    # Signal quality thresholds - used for confidence calculation only
    CONFIDENCE_THRESHOLD = 0.45  # Minimum confidence for signal generation
    
    # Signal quality parameters
    MIN_SIGNAL_ALIGNMENT = 0.5
    NEUTRAL_SIGNAL_PENALTY = 0.05
    
    # Feature normalization bounds
    PRICE_NORMALIZATION = True
    VOLUME_LOOKBACK = 20
    
    # Enhanced signal processing parameters
    SIGNAL_SMOOTHING = True
    NEUTRAL_ZONE_SIZE = 0.001
    MOMENTUM_LOOKBACK = 5
    
    # Validation parameters
    MAX_NEUTRAL_SIGNALS = 6
    SIGNAL_VALIDATION_STRICT = False
    
    # NEW: Signal quality parameters for NinjaScript communication
    EXCELLENT_QUALITY_THRESHOLD = 0.85  # For "excellent" signal quality
    GOOD_QUALITY_THRESHOLD = 0.70       # For "good" signal quality
    POOR_QUALITY_THRESHOLD = 0.50       # Below this is "poor" signal quality
    
    # Signal generation mode
    PURE_ML_MODE = True  # All position management in NinjaScript