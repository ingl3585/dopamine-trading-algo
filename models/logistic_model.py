# models/logistic_model.py

import numpy as np
import joblib
import os
import logging
from typing import Tuple, List
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from features.feature_extractor import ResearchFeatures
from config import ResearchConfig

log = logging.getLogger(__name__)

class LogisticSignalModel:
    """Logistic Regression model for signal generation"""
    
    def __init__(self, config: ResearchConfig):
        self.config = config
        self.model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Training data storage
        self.feature_history: List[ResearchFeatures] = []
        self.signal_history: List[int] = []
        
        self.load_model()
    
    def predict(self, features: ResearchFeatures) -> Tuple[int, float, str]:
        """Generate trading signal using Logistic Regression"""
        
        if not self.is_trained or features is None:
            return 0, 0.0, "model_not_ready"
        
        try:
            # Prepare features
            feature_array = features.to_array().reshape(1, -1)
            scaled_features = self.scaler.transform(feature_array)
            
            # Get prediction and probability
            prediction = self.model.predict(scaled_features)[0]
            probabilities = self.model.predict_proba(scaled_features)[0]
            confidence = probabilities[np.where(self.model.classes_ == prediction)[0][0]]
            quality = self._assess_quality(confidence)
            
            return prediction, confidence, quality
            
        except Exception as e:
            log.error(f"Prediction error: {e}")
            return 0, 0.0, "error"
    
    def add_training_sample(self, features: ResearchFeatures):
        """Add new training sample"""
        
        # Convert price change to signal class
        signal = self._generate_signal_from_features(features)
        
        self.feature_history.append(features)
        self.signal_history.append(signal)
        
        # Maintain limited history
        max_history = self.config.ML_LOOKBACK * 2
        if len(self.feature_history) > max_history:
            self.feature_history = self.feature_history[-self.config.ML_LOOKBACK:]
            self.signal_history = self.signal_history[-self.config.ML_LOOKBACK:]
        
        # Retrain only when we have natural diversity
        if self._should_retrain():
            self.train()
    
    def train(self):
        """Train the logistic regression model"""
        
        try:
            if len(self.feature_history) < self.config.MIN_TRAINING_SAMPLES:
                log.warning("Insufficient training samples")
                return
            
            # Prepare training data
            X = np.array([f.to_array() for f in self.feature_history])
            y = np.array(self.signal_history)
            
            # Validate we have multiple classes with sufficient samples
            unique_classes, counts = np.unique(y, return_counts=True)
            if len(unique_classes) < 2:
                log.warning("Need multiple signal classes for training")
                return
            
            # Require samples per class for sklearn
            min_samples_per_class = min(counts)
            if min_samples_per_class < 3:
                log.warning(f"Need at least 3 samples per class, got {min_samples_per_class}")
                return
            
            # Scale features
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            # Simple accuracy check on training data
            accuracy = self.model.score(X_scaled, y)
            log.info(f"Model trained on {len(X)} samples - accuracy: {accuracy:.3f}")
            
            # Save model
            self.save_model()
            
        except Exception as e:
            log.error(f"Training error: {e}")
    
    def save_model(self):
        """Save trained model and scaler"""
        try:
            if self.is_trained:
                joblib.dump(self.model, self.config.MODEL_PATH)
                joblib.dump(self.scaler, self.config.SCALER_PATH)
                log.info("Model saved successfully")
        except Exception as e:
            log.error(f"Model save error: {e}")
    
    def load_model(self):
        """Load saved model and scaler"""
        try:
            if os.path.exists(self.config.MODEL_PATH) and os.path.exists(self.config.SCALER_PATH):
                self.model = joblib.load(self.config.MODEL_PATH)
                self.scaler = joblib.load(self.config.SCALER_PATH)
                self.is_trained = True
                log.info("Model loaded successfully")
            else:
                log.info("No saved model found - will train from scratch")
        except Exception as e:
            log.error(f"Model load error: {e}")

    def _generate_signal_from_features(self, features: ResearchFeatures) -> int:
        """Generate signals - loose for training, strict for live trading"""
        
        # 1. Trend Analysis (15m timeframe)
        trend_bullish = (
            features.ema_trend_15m > 0 and           
            features.price_vs_sma_15m > -0.002 and  
            30 < features.rsi_15m < 75               
        )
        
        trend_bearish = (
            features.ema_trend_15m < 0 and           
            features.price_vs_sma_15m < 0.002 and   
            25 < features.rsi_15m < 70               
        )
        
        # 2. Entry Signals (5m timeframe)
        bb_oversold = features.bb_position_5m < 0.2      
        bb_overbought = features.bb_position_5m > 0.8    
        rsi_oversold = features.rsi_5m < 30              
        rsi_overbought = features.rsi_5m > 70            
        volume_above_avg = features.volume_ratio_5m > 0.8 
        price_above_ema = features.price_vs_sma_5m > 0.001   # Price 0.1% above 5m SMA
        price_below_ema = features.price_vs_sma_5m < -0.001  # Price 0.1% below 5m SMA 
        
        # 4. Signal Generation
        # PERFECT signals (all conditions + price position)
        perfect_buy = (trend_bullish and bb_oversold and rsi_oversold and 
                    volume_above_avg and price_below_ema)  # Buy when price below MA (pullback)
        
        perfect_sell = (trend_bearish and bb_overbought and rsi_overbought and 
                        volume_above_avg and price_above_ema)  # Sell when price above MA (rally)
        
        # GOOD signals (trend + price position + 2 other conditions)
        good_buy = (trend_bullish and price_below_ema and 
                    sum([bb_oversold, rsi_oversold, volume_above_avg]) >= 2)
        
        good_sell = (trend_bearish and price_above_ema and 
                    sum([bb_overbought, rsi_overbought, volume_above_avg]) >= 2)
        
        # BASIC signals (just trend + price breakout)
        basic_buy = (trend_bullish and price_above_ema and 
                    features.volume_ratio_5m > 1.0)  # Trend + breakout above MA
        
        basic_sell = (trend_bearish and price_below_ema and 
                    features.volume_ratio_5m > 1.0)  # Trend + breakdown below MA
        
        # Return signals with priority
        if perfect_buy or good_buy or basic_buy:
            return 1  # BUY
        elif perfect_sell or good_sell or basic_sell:
            return 2  # SELL
        else:
            return 0  # HOLD
    
    def _assess_quality(self, confidence: float) -> str:
        """Simplified quality assessment - research aligned"""
        if confidence >= 0.7:
            return "good"
        else:
            return "poor"
    
    def _should_retrain(self) -> bool:
        """Simplified retraining logic - research aligned"""
        
        if len(self.signal_history) < self.config.MIN_TRAINING_SAMPLES:
            return False
        
        # Only retrain every N samples
        if len(self.signal_history) % self.config.ML_RETRAIN_FREQUENCY != 0:
            return False
        
        # Simple check - need 3 classes
        unique_classes = len(np.unique(self.signal_history))
        return unique_classes >= 3