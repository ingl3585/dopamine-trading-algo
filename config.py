# config.py - ENHANCED with black box AI learning parameters

from dataclasses import dataclass
import os

@dataclass 
class ResearchConfig:
    """Enhanced configuration for black box intelligence engine with tool learning"""
    
    # TCP Configuration - ONLY essential communication settings
    TCP_HOST: str = "localhost"
    FEATURE_PORT: int = 5556
    SIGNAL_PORT: int = 5557
    
    # Black Box AI Learning Parameters (NEW)
    AI_MIN_CONFIDENCE: float = 0.4         # Minimum confidence to act
    AI_EXPLORATION_RATE: float = 0.1       # How much AI explores vs exploits
    AI_LEARNING_RATE: float = 1e-4         # Neural network learning rate
    AI_BATCH_SIZE: int = 64                 # Training batch size
    AI_MEMORY_SIZE: int = 50000             # Experience replay buffer size
    AI_SYNC_FREQUENCY: int = 1000           # Target network sync frequency
    
    # Tool Learning Thresholds (NEW)
    TOOL_MIN_USAGE: int = 20                # Min uses before trusting tool performance
    TOOL_SUCCESS_THRESHOLD: float = 0.6     # Success rate to consider tool reliable
    REGIME_ADAPTATION_RATE: float = 0.01    # How fast AI adapts to regime changes
    
    # Intelligence Engine Settings (ENHANCED)
    INTELLIGENCE_MIN_CONFIDENCE: float = 0.3    # Lowered for more AI exploration
    PATTERN_DISCOVERY_THRESHOLD: float = 0.5    # Pattern recognition threshold
    SUBSYSTEM_BOOTSTRAP_SAMPLES: int = 100      # Samples to bootstrap each subsystem
    
    # Memory Management (ENHANCED)
    MAX_PATTERN_HISTORY: int = 20000           # Increased for more learning
    PATTERN_CLEANUP_DAYS: int = 60             # Keep patterns longer for AI learning
    DNA_SEQUENCE_MIN_LENGTH: int = 10          # Minimum DNA sequence for learning
    
    # Learning Parameters  
    MIN_BOOTSTRAP_SAMPLES: int = 100           # Increased for better initialization
    CONTINUOUS_LEARNING: bool = True           # Enable continuous learning
    SAVE_LEARNING_CHECKPOINTS: bool = True     # Save AI progress regularly
    CHECKPOINT_FREQUENCY: int = 1000           # Save every N decisions
    
    # Performance Tracking (NEW)
    TRACK_TOOL_COMBINATIONS: bool = True       # Track which tool combos work
    PERFORMANCE_WINDOW: int = 100              # Rolling window for performance calc
    LOG_AI_DECISIONS: bool = True              # Detailed logging of AI reasoning
    
    # Risk Management Learning (NEW) - AI learns everything from scratch
    AI_LEARNS_RISK_MGMT: bool = True           # AI discovers stops/targets on its own
    NO_FIXED_RULES: bool = True                # No hardcoded risk management rules
    
    def __post_init__(self):
        """Create directories and validate settings"""
        # Create essential directories
        os.makedirs('patterns', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)          # For AI model checkpoints
        os.makedirs('logs', exist_ok=True)            # For detailed logs
        os.makedirs('reports', exist_ok=True)         # For performance reports
        
        # Validate learning parameters
        if self.AI_EXPLORATION_RATE < 0 or self.AI_EXPLORATION_RATE > 1:
            raise ValueError("AI_EXPLORATION_RATE must be between 0 and 1")
        
        if self.TOOL_SUCCESS_THRESHOLD < 0.5 or self.TOOL_SUCCESS_THRESHOLD > 1:
            raise ValueError("TOOL_SUCCESS_THRESHOLD should be between 0.5 and 1")
        
        if self.AI_MIN_CONFIDENCE > 0.8:
            raise ValueError("AI_MIN_CONFIDENCE too high - AI needs room to explore")

class RLConfigMixin:
    """
    Enhanced RL configuration for black box tool learning
    """
    # Model Persistence (ENHANCED)
    CHECKPOINT_DIR = "models"               # Auto-saved *.pt files
    EXPERIENCE_DIR = "experience"           # Experience replay archives
    BEST_MODEL_PATH = "models/best_ai.pt"   # Best performing model
    LATEST_MODEL_PATH = "models/latest_ai.pt"  # Most recent model
    
    # Trading Parameters (ENHANCED) - AI learns everything
    MAX_POSITION_SIZE = 1                   # Contracts (kept conservative)
    MAX_INTRADAY_DRAWDOWN_PCT = 3.0         # Safety limit only
    REWARD_SCALING = 1.0                    # Base reward scaling
    
    # AI Learning Rewards (NEW) - No fixed bonuses, AI discovers optimal behavior
    BASE_REWARD_ONLY: bool = True            # Use only P&L-based rewards initially
    LET_AI_DISCOVER_RISK_MGMT: bool = True   # AI learns stops/targets from scratch
    
    # Learning Schedule (NEW)
    EXPLORATION_DECAY = 0.9995              # How fast exploration decreases
    MIN_EXPLORATION = 0.05                  # Minimum exploration rate
    LEARNING_WARMUP_TRADES = 200            # Trades before serious learning
    
    # Performance Targets (NEW)
    TARGET_WIN_RATE = 0.6                   # Target win rate for AI
    TARGET_TOOL_DIVERSITY = 0.8             # Target for using all tools
    PERFORMANCE_EVALUATION_PERIOD = 100     # Evaluate every N trades
    
    # Advanced Learning (NEW)
    USE_PRIORITIZED_REPLAY = True           # Prioritize important experiences
    USE_DOUBLE_DQN = True                   # Better Q-value estimation
    USE_NOISY_NETWORKS = False              # Keep disabled for stability
    
    def get_model_save_path(self, model_type: str = "checkpoint") -> str:
        """Get path for saving models"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.CHECKPOINT_DIR}/ai_{model_type}_{timestamp}.pt"
    
    def get_performance_report_path(self) -> str:
        """Get path for performance reports"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"reports/ai_performance_{timestamp}.html"

# Combined configuration class
@dataclass
class BlackBoxConfig(ResearchConfig, RLConfigMixin):
    """
    Complete configuration for black box AI trading system
    """
    
    def __post_init__(self):
        super().__post_init__()
        
        # Create additional directories for black box AI
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.EXPERIENCE_DIR, exist_ok=True)
        
        # Validate advanced settings
        if self.MAX_INTRADAY_DRAWDOWN_PCT > 5.0:
            print("WARNING: High drawdown limit for AI learning phase")
        
        if self.LEARNING_WARMUP_TRADES < 100:
            print("WARNING: Low warmup trades - AI may not learn effectively")
    
    def get_learning_summary(self) -> str:
        """Get summary of learning configuration"""
        return f"""
Black Box AI Learning Configuration:
- Confidence Threshold: {self.AI_MIN_CONFIDENCE}
- Exploration Rate: {self.AI_EXPLORATION_RATE}
- Tool Success Threshold: {self.TOOL_SUCCESS_THRESHOLD}
- Memory Size: {self.AI_MEMORY_SIZE:,} experiences
- Warmup Period: {self.LEARNING_WARMUP_TRADES} trades
- Performance Window: {self.PERFORMANCE_WINDOW} trades
- Model Checkpoints: Every {self.CHECKPOINT_FREQUENCY} decisions
"""