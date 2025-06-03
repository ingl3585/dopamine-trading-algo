# config.py

class Config:
    # File paths
    FEATURE_FILE = r"C:\\Users\\ingle\\OneDrive\\Desktop\\Actor_Critic_ML_NT\\features\\features_multiframe.csv"
    MODEL_PATH   = r"C:\\Users\\ingle\\OneDrive\\Desktop\\Actor_Critic_ML_NT\\model\\actor_critic_multiframe_model.pth"

    # Multi-timeframe model architecture - EXPANDED from 9 to 27 features
    INPUT_DIM   = 27    # 9 features × 3 timeframes (15m + 5m + 1m)
    HIDDEN_DIM  = 256   # Increased to handle 27-dimensional input
    ACTION_DIM  = 3     # Hold, Long, Short
    LOOKBACK    = 1

    # Training parameters - Adjusted for larger model
    BATCH_SIZE  = 64    # Increased batch size for stable training
    GAMMA       = 0.95
    ENTROPY_COEF= 0.05
    LR          = 5e-5  # Reduced learning rate for stability

    # Prediction parameters
    TEMPERATURE = 1.0   # Reduced for more decisive actions

    # Multi-timeframe feature weights for confidence calculation
    # 15-minute timeframe (trend context) - HIGH WEIGHT
    TREND_15M_WEIGHT = 0.40        # Primary trend direction
    
    # 5-minute timeframe (momentum context) - MEDIUM WEIGHT  
    MOMENTUM_5M_WEIGHT = 0.35      # Pullback vs continuation
    
    # 1-minute timeframe (entry timing) - LOWER WEIGHT
    ENTRY_1M_WEIGHT = 0.25         # Precise entry timing
    
    # Individual feature weights within each timeframe
    ICHIMOKU_WEIGHT = 0.35         # Ichimoku signals (per timeframe)
    EMA_WEIGHT = 0.25              # EMA signals (per timeframe)
    MOMENTUM_WEIGHT = 0.15         # Momentum signals (per timeframe)
    VOLUME_WEIGHT = 0.15           # Volume signals (per timeframe)
    LWPE_WEIGHT = 0.10             # LWPE signals (per timeframe)

    # Multi-timeframe signal quality thresholds
    CONFIDENCE_THRESHOLD = 0.50    # Increased for multi-timeframe reliability
    
    # Timeframe alignment parameters
    TREND_ALIGNMENT_THRESHOLD = 0.6     # Strong trend on 15m
    MOMENTUM_ALIGNMENT_THRESHOLD = 0.4  # Momentum on 5m
    ENTRY_ALIGNMENT_THRESHOLD = 0.3     # Entry timing on 1m
    
    # Anti-trend-fighting parameters
    MAX_COUNTER_TREND_CONFIDENCE = 0.7  # Don't trade against strong trends
    TREND_OVERRIDE_THRESHOLD = 0.8      # 15m trend can override lower timeframes
    
    # Signal quality parameters for enhanced analysis
    EXCELLENT_QUALITY_THRESHOLD = 0.85  # All timeframes aligned
    GOOD_QUALITY_THRESHOLD = 0.70       # 2/3 timeframes aligned
    POOR_QUALITY_THRESHOLD = 0.50       # Conflicting timeframes
    
    # Feature normalization bounds
    PRICE_NORMALIZATION = True
    VOLUME_LOOKBACK = 20
    
    # Enhanced signal processing for multi-timeframe
    SIGNAL_SMOOTHING = True
    NEUTRAL_ZONE_SIZE = 0.001
    MOMENTUM_LOOKBACK = 5
    
    # Multi-timeframe validation parameters
    MAX_NEUTRAL_SIGNALS_PER_TIMEFRAME = 6
    SIGNAL_VALIDATION_STRICT = True     # Stricter validation for 27 features
    
    # Feature vector structure (27 elements)
    # Indices 0-8:   15-minute features (trend context)
    # Indices 9-17:  5-minute features (momentum context)  
    # Indices 18-26: 1-minute features (entry timing)
    
    FEATURE_NAMES = [
        # 15-minute features (0-8)
        "close_15m", "norm_vol_15m", "tenkan_kijun_15m", "price_cloud_15m", 
        "future_cloud_15m", "ema_cross_15m", "tenkan_momentum_15m", 
        "kijun_momentum_15m", "lwpe_15m",
        
        # 5-minute features (9-17)
        "close_5m", "norm_vol_5m", "tenkan_kijun_5m", "price_cloud_5m", 
        "future_cloud_5m", "ema_cross_5m", "tenkan_momentum_5m", 
        "kijun_momentum_5m", "lwpe_5m",
        
        # 1-minute features (18-26)
        "close_1m", "norm_vol_1m", "tenkan_kijun_1m", "price_cloud_1m", 
        "future_cloud_1m", "ema_cross_1m", "tenkan_momentum_1m", 
        "kijun_momentum_1m", "lwpe_1m"
    ]
    
    # Timeframe-specific feature indices for analysis
    TIMEFRAME_15M_INDICES = list(range(0, 9))    # 0-8
    TIMEFRAME_5M_INDICES = list(range(9, 18))    # 9-17
    TIMEFRAME_1M_INDICES = list(range(18, 27))   # 18-26
    
    # Signal indices within each timeframe (same pattern for all)
    SIGNAL_INDICES_OFFSET = {
        'tenkan_kijun': 2,
        'price_cloud': 3,
        'future_cloud': 4,
        'ema_cross': 5,
        'tenkan_momentum': 6,
        'kijun_momentum': 7
    }
    
    # Multi-timeframe confidence calculation mode
    MULTI_TIMEFRAME_MODE = True
    
    # Trend filter enhancement
    ENABLE_TREND_FILTER = True          # Block counter-trend trades
    TREND_FILTER_STRENGTH = 0.7         # Minimum 15m trend to filter
    
    # Position sizing based on timeframe alignment
    BASE_SIZE_MULTIPLIER = 1.0
    ALIGNED_TIMEFRAMES_MULTIPLIER = 1.5  # When 2+ timeframes align
    ALL_TIMEFRAMES_MULTIPLIER = 2.0      # When all 3 timeframes align
    
    # Signal generation mode
    PURE_ML_MODE = True  # All position management in NinjaScript
    
    # Multi-timeframe debugging
    ENABLE_TIMEFRAME_LOGGING = True
    LOG_FEATURE_DISTRIBUTION = True
    LOG_TIMEFRAME_ALIGNMENT = True
    
    # Backward compatibility
    LEGACY_9_FEATURE_SUPPORT = True     # Fallback to 9 features if needed