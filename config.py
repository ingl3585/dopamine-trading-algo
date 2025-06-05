# config.py

from dataclasses import dataclass, field
from typing import List

@dataclass
class ResearchConfig:
    """Research-aligned configuration"""
    
    # Core indicators based on academic research
    RSI_PERIOD: int = 14
    BB_PERIOD: int = 20
    BB_STD: float = 2.0
    EMA_PERIOD: int = 20
    SMA_PERIOD: int = 50
    VOLUME_PERIOD: int = 20
    
    # Multi-timeframe settings
    TIMEFRAMES: List[str] = field(default_factory=lambda: ["15m", "5m"])
    
    # Logistic Regression parameters
    ML_LOOKBACK: int = 50
    ML_RETRAIN_FREQUENCY: int = 100
    MIN_TRAINING_SAMPLES: int = 30
    
    # Signal thresholds
    CONFIDENCE_THRESHOLD: float = 0.6
    SIGNAL_STRENGTH_THRESHOLD: float = 0.7
    
    # TCP Configuration
    TCP_HOST: str = "localhost"
    FEATURE_PORT: int = 5556
    SIGNAL_PORT: int = 5557
    
    # Model persistence
    MODEL_PATH: str = "models/logistic_model.joblib"
    SCALER_PATH: str = "models/feature_scaler.joblib"
    
    def __post_init__(self):
        """Ensure model directory exists"""
        import os
        os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)