# config.py

from dataclasses import dataclass

@dataclass 
class ResearchConfig:
    """Minimal configuration for pure intelligence engine"""
    
    # TCP Configuration - ONLY essential communication settings
    TCP_HOST: str = "localhost"
    FEATURE_PORT: int = 5556
    SIGNAL_PORT: int = 5557
    
    # Intelligence Engine Settings
    INTELLIGENCE_MIN_CONFIDENCE: float = 0.4    # Minimum confidence to act
    PATTERN_DISCOVERY_THRESHOLD: float = 0.5    # Pattern recognition threshold
    
    # Memory Management
    MAX_PATTERN_HISTORY: int = 10000           # Maximum patterns to remember
    PATTERN_CLEANUP_DAYS: int = 30             # Days before cleaning old patterns
    
    # Learning Parameters  
    MIN_BOOTSTRAP_SAMPLES: int = 50            # Minimum samples for bootstrap
    CONTINUOUS_LEARNING: bool = True           # Enable continuous learning
    
    def __post_init__(self):
        """Create directories if needed"""
        import os
        os.makedirs('patterns', exist_ok=True)
        os.makedirs('data', exist_ok=True)