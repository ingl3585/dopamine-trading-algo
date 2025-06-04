# config.py

class Config:
    # File paths
    FEATURE_FILE = r"C:\\Users\\ingle\\OneDrive\\Desktop\\Actor_Critic_ML_NT\\features\\features_multiframe.csv"
    MODEL_PATH   = r"C:\\Users\\ingle\\OneDrive\\Desktop\\Actor_Critic_ML_NT\\model\\actor_critic_enhanced_multiframe_model.pth"

    # Enhanced multi-timeframe model architecture
    INPUT_DIM   = 27    # 9 features × 3 timeframes (15m + 5m + 1m)
    HIDDEN_DIM  = 384   # Must be divisible by 3 (for 3 timeframes) and each part divisible by num_heads
    ACTION_DIM  = 3     # Hold, Long, Short
    LOOKBACK    = 1

    # Enhanced training parameters
    BATCH_SIZE  = 128   # Increased for more stable training
    GAMMA       = 0.95
    ENTROPY_COEF= 0.08  # Slightly increased for better exploration
    LR          = 3e-5  # Reduced for enhanced model stability

    # Enhanced prediction parameters
    TEMPERATURE = 0.9   # Slightly reduced for more decisive actions

    # Enhanced multi-timeframe feature weights
    # 15-minute timeframe (trend context) - HIGH WEIGHT
    TREND_15M_WEIGHT = 0.45        # Increased importance for trend
    
    # 5-minute timeframe (momentum context) - MEDIUM WEIGHT  
    MOMENTUM_5M_WEIGHT = 0.35      # Momentum signals
    
    # 1-minute timeframe (entry timing) - BALANCED WEIGHT
    ENTRY_1M_WEIGHT = 0.20         # Entry timing (reduced to emphasize higher TFs)
    
    # Enhanced individual feature weights within each timeframe
    ICHIMOKU_WEIGHT = 0.40         # Increased - Ichimoku signals (per timeframe)
    EMA_WEIGHT = 0.25              # EMA signals (per timeframe)
    MOMENTUM_WEIGHT = 0.15         # Momentum signals (per timeframe)
    VOLUME_WEIGHT = 0.12           # Volume signals (per timeframe)
    LWPE_WEIGHT = 0.08             # LWPE signals (per timeframe)

    # Enhanced multi-timeframe signal quality thresholds
    CONFIDENCE_THRESHOLD = 0.45    # Reduced for enhanced model sensitivity
    
    # Enhanced timeframe alignment parameters
    TREND_ALIGNMENT_THRESHOLD = 0.65    # Strong trend on 15m (increased)
    MOMENTUM_ALIGNMENT_THRESHOLD = 0.50 # Momentum on 5m (increased)
    ENTRY_ALIGNMENT_THRESHOLD = 0.35    # Entry timing on 1m (increased)
    
    # Enhanced anti-trend-fighting parameters
    MAX_COUNTER_TREND_CONFIDENCE = 0.65 # Reduced - more conservative against trends
    TREND_OVERRIDE_THRESHOLD = 0.75     # 15m trend override (reduced)
    TREND_FILTER_STRENGTH = 0.60        # Reduced threshold for trend filtering
    
    # Enhanced signal quality parameters
    EXCELLENT_QUALITY_THRESHOLD = 0.80  # All timeframes strongly aligned
    GOOD_QUALITY_THRESHOLD = 0.65       # 2/3 timeframes aligned
    POOR_QUALITY_THRESHOLD = 0.45       # Conflicting timeframes
    
    # Enhanced model architecture parameters
    MULTI_TIMEFRAME_MODE = True         # Enable enhanced model
    USE_ATTENTION_MECHANISM = True      # Enable attention layers
    ATTENTION_HEADS = 4                 # Number of attention heads (adjusted for divisibility)
    LAYER_NORM = True                   # Use layer normalization
    ENHANCED_DROPOUT = True             # Use adaptive dropout
    
    # Enhanced confidence calculation parameters
    CONFIDENCE_MODEL_WEIGHT = 0.20      # Model prediction weight
    CONFIDENCE_TIMEFRAME_WEIGHT = 0.55  # Multi-timeframe alignment weight (increased)
    CONFIDENCE_VALUE_WEIGHT = 0.15      # Value estimate weight
    CONFIDENCE_MARKET_WEIGHT = 0.10     # Market condition weight
    
    # Enhanced signal processing parameters
    SIGNAL_STRENGTH_THRESHOLD_15M = 0.70 # Trend signals need to be strong
    SIGNAL_STRENGTH_THRESHOLD_5M = 0.60  # Momentum signals medium strength
    SIGNAL_STRENGTH_THRESHOLD_1M = 0.50  # Entry signals can be weaker
    
    # Enhanced volume and LWPE parameters
    HIGH_VOLUME_THRESHOLD = 1.5         # Normalized volume threshold
    EXTREME_LWPE_THRESHOLD = 0.3        # Distance from 0.5 for extreme LWPE
    VOLUME_BOOST_MAX = 0.25            # Maximum volume boost
    LWPE_BOOST_MAX = 0.15              # Maximum LWPE boost
    
    # Enhanced trend filter parameters
    TREND_FILTER_PRIMARY_PENALTY = 0.6  # Base penalty for fighting trend
    TREND_FILTER_SECONDARY_PENALTY = 0.2 # Additional penalty for weak counter-momentum
    TREND_FILTER_MAX_PENALTY = 0.8      # Maximum total penalty
    
    # Enhanced consistency bonus parameters
    PERFECT_ALIGNMENT_BONUS = 1.0       # All timeframes strongly aligned
    STRONG_MAJORITY_BONUS = 0.8         # 2/3 timeframes strongly aligned  
    MODERATE_CONSENSUS_BONUS = 0.5      # 2/3 timeframes moderately aligned
    TREND_FIGHT_PENALTY = -0.3          # Penalty for trend vs entry conflict
    
    # Enhanced feature expansion parameters
    TREND_SIGNAL_AMPLIFICATION = 1.2    # Amplify strong signals for trend
    MOMENTUM_SIGNAL_AMPLIFICATION = 1.1 # Amplify momentum signals
    ENTRY_SIGNAL_AMPLIFICATION = 1.3    # Amplify weak signals for entry timing
    NOISE_DAMPENING_FACTOR = 0.6        # Dampen weak/noisy signals
    
    # Learning rate scheduling
    USE_LR_SCHEDULER = True             # Enable learning rate scheduling
    LR_SCHEDULER_TYPE = 'cosine'        # Cosine annealing
    LR_T_MAX = 1000                     # Scheduler period
    LR_ETA_MIN_RATIO = 0.1              # Minimum LR ratio
    
    # Model regularization
    WEIGHT_DECAY = 1e-4                 # L2 regularization (increased)
    GRADIENT_CLIP_NORM = 1.0            # Gradient clipping
    ATTENTION_REGULARIZATION = 0.01     # Attention diversity regularization
    
    # Enhanced experience buffer
    EXPERIENCE_BUFFER_SIZE = 2000       # Increased buffer size
    EXPERIENCE_BUFFER_MIN = 400         # Minimum experiences to keep
    
    # Enhanced training frequency
    ONLINE_TRAINING_FREQUENCY = 64      # Train every N steps
    BATCH_TRAINING_FREQUENCY = 256      # Batch train every N steps
    MODEL_SAVE_FREQUENCY = 300          # Save every N seconds
    
    # Enhanced validation and monitoring
    FEATURE_VALIDATION_STRICT = True    # Strict feature validation
    ENABLE_ATTENTION_MONITORING = True  # Monitor attention patterns
    ENABLE_PERFORMANCE_TRACKING = True  # Track model performance metrics
    LOG_ATTENTION_WEIGHTS = True        # Log attention analysis
    
    # Enhanced signal analysis parameters
    SETUP_QUALITY_WEIGHTS = {
        'perfect_setup': 1.0,           # All signals aligned
        'strong_setup_4_signals': 0.8,  # 4+ signals aligned
        'good_setup_3_signals': 0.6,    # 3 signals aligned
        'moderate_setup_2_signals': 0.4, # 2 signals aligned
        'weak_setup': 0.1               # <2 signals aligned
    }
    
    # Feature normalization bounds (enhanced)
    PRICE_NORMALIZATION = True
    VOLUME_LOOKBACK = 20
    VOLUME_NORMALIZATION_CLIP = 10.0    # Clip extreme volume values
    LWPE_BOUNDS_CHECK = True            # Ensure LWPE in [0,1]
    SIGNAL_BOUNDS_CHECK = True          # Ensure signals in [-1,0,1]
    
    # Enhanced signal processing for multi-timeframe
    SIGNAL_SMOOTHING = True
    NEUTRAL_ZONE_SIZE = 0.001
    MOMENTUM_LOOKBACK = 5
    TIMEFRAME_SPECIFIC_PROCESSING = True # Enable timeframe-specific signal processing
    
    # Multi-timeframe validation parameters (enhanced)
    MAX_NEUTRAL_SIGNALS_PER_TIMEFRAME = 6
    SIGNAL_VALIDATION_STRICT = True
    TIMEFRAME_CONSISTENCY_CHECK = True   # Check consistency across timeframes
    FEATURE_CORRELATION_MONITORING = True # Monitor feature correlations
    
    # Enhanced logging and debugging
    ENABLE_TIMEFRAME_LOGGING = True
    LOG_FEATURE_DISTRIBUTION = True
    LOG_TIMEFRAME_ALIGNMENT = True
    LOG_CONFIDENCE_BREAKDOWN = True     # Log confidence calculation details
    LOG_ATTENTION_ANALYSIS = True       # Log attention pattern analysis
    LOG_SIGNAL_QUALITY_STATS = True     # Log signal quality statistics
    
    # Performance optimization
    USE_MIXED_PRECISION = False         # Mixed precision training (if supported)
    DATALOADER_NUM_WORKERS = 2          # Number of data loading workers
    PIN_MEMORY = True                   # Pin memory for faster GPU transfer
    
    # Enhanced feature vector structure (27 elements) - DOCUMENTATION
    FEATURE_NAMES = [
        # 15-minute features (indices 0-8) - TREND CONTEXT
        "close_15m", "norm_vol_15m", "tenkan_kijun_15m", "price_cloud_15m", 
        "future_cloud_15m", "ema_cross_15m", "tenkan_momentum_15m", 
        "kijun_momentum_15m", "lwpe_15m",
        
        # 5-minute features (indices 9-17) - MOMENTUM CONTEXT
        "close_5m", "norm_vol_5m", "tenkan_kijun_5m", "price_cloud_5m", 
        "future_cloud_5m", "ema_cross_5m", "tenkan_momentum_5m", 
        "kijun_momentum_5m", "lwpe_5m",
        
        # 1-minute features (indices 18-26) - ENTRY TIMING
        "close_1m", "norm_vol_1m", "tenkan_kijun_1m", "price_cloud_1m", 
        "future_cloud_1m", "ema_cross_1m", "tenkan_momentum_1m", 
        "kijun_momentum_1m", "lwpe_1m"
    ]
    
    # Timeframe-specific feature indices for enhanced processing
    TIMEFRAME_15M_INDICES = list(range(0, 9))    # 0-8: Trend context
    TIMEFRAME_5M_INDICES = list(range(9, 18))    # 9-17: Momentum context
    TIMEFRAME_1M_INDICES = list(range(18, 27))   # 18-26: Entry timing
    
    # Enhanced signal indices within each timeframe
    SIGNAL_INDICES_OFFSET = {
        'close': 0,
        'normalized_volume': 1,
        'tenkan_kijun': 2,
        'price_cloud': 3,
        'future_cloud': 4,
        'ema_cross': 5,
        'tenkan_momentum': 6,
        'kijun_momentum': 7,
        'lwpe': 8
    }
    
    # Enhanced timeframe importance evolution tracking
    TRACK_IMPORTANCE_EVOLUTION = True   # Track how timeframe importance changes
    IMPORTANCE_HISTORY_LENGTH = 1000    # Keep N recent importance measurements
    
    # Enhanced model analysis parameters
    ANALYZE_ATTENTION_PATTERNS = True   # Analyze attention pattern evolution
    ANALYZE_PREDICTION_CONSISTENCY = True # Track prediction consistency
    ANALYZE_CONFIDENCE_ACCURACY = True  # Track confidence vs actual performance
    
    # Enhanced risk management integration
    ENABLE_DYNAMIC_CONFIDENCE_SCALING = True # Scale confidence based on market conditions
    MARKET_REGIME_DETECTION = True      # Detect market regime changes
    VOLATILITY_ADJUSTMENT = True        # Adjust signals based on volatility
    
    # Enhanced position sizing integration (for analysis)
    ANALYZE_POSITION_SIZING_IMPACT = True # Analyze how confidence affects sizing
    TRACK_RISK_ADJUSTED_RETURNS = True  # Track risk-adjusted performance
    
    # Backward compatibility flags
    LEGACY_9_FEATURE_SUPPORT = True     # Support for 9-feature inputs
    AUTO_EXPAND_FEATURES = True         # Automatically expand 9 to 27 features
    GRADUAL_MIGRATION_MODE = False      # Gradual migration from old to new model
    
    # Model versioning
    MODEL_VERSION = "2.0"               # Enhanced multi-timeframe version
    ARCHITECTURE_ID = "enhanced_attention_v2"
    FEATURE_VERSION = "27_feature_multiframe"
    
    # Debugging and development
    DEBUG_MODE = False                  # Enable debug features
    SAVE_INTERMEDIATE_STATES = False    # Save intermediate model states
    PROFILE_PERFORMANCE = False         # Profile model performance
    
    # Production deployment flags
    PRODUCTION_MODE = False             # Production optimization settings
    FAIL_SAFE_FALLBACKS = True          # Enable fallback mechanisms
    GRACEFUL_DEGRADATION = True         # Graceful degradation on errors