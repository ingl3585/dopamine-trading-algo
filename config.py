# config.py - Clean configuration with only used variables

import os
from dataclasses import dataclass

@dataclass 
class ResearchConfig:
    """Clean configuration - only variables actually used in the code"""
    
    # TCP Configuration (used in tcp_bridge.py)
    TCP_HOST: str = "localhost"
    FEATURE_PORT: int = 5556
    SIGNAL_PORT: int = 5557
    
    # Safety Limits (used in SafetyManager)
    MAX_DAILY_LOSS: float = 200
    MAX_CONSECUTIVE_LOSSES: int = 6
    EXPLORATION_PHASE_SIZE: float = 0.1
    DEVELOPMENT_PHASE_SIZE: float = 0.5
    PRODUCTION_PHASE_SIZE: float = 1.0
    MAX_DAILY_TRADES_EXPLORATION: int = 3
    MAX_DAILY_TRADES_DEVELOPMENT: int = 8
    MAX_DAILY_TRADES_PRODUCTION: int = 15
    
    # AI Learning (used in rl_agent.py)
    AI_MIN_CONFIDENCE: float = 0.5
    AI_EXPLORATION_RATE: float = 0.8
    AI_LEARNING_RATE: float = 1e-4
    AI_MEMORY_SIZE: int = 50000
    
    # Confidence Learning (used in PureBlackBoxConfidenceLearner)
    NEUTRAL_THRESHOLD: float = 0.5
    MIN_LEARNING_SAMPLES: int = 5
    THRESHOLD_LEARNING_RATE: float = 0.02
    
    # Memory Management (used in intelligence engine)
    MAX_PATTERN_HISTORY: int = 20000
    PATTERN_CLEANUP_DAYS: int = 90
    
    def __post_init__(self):
        """Create essential directories only"""
        for directory in ['patterns', 'data', 'models', 'logs']:
            os.makedirs(directory, exist_ok=True)