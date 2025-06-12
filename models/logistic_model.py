# models/logistic_model.py

import numpy as np
import joblib
import os
import logging
from typing import Tuple, List
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from features.feature_extractor import ResearchFeatures
from config import ResearchConfig

log = logging.getLogger(__name__)

class LogisticSignalModel:
    """Enhanced Logistic Regression model with ensemble support"""
    
    def __init__(self, config: ResearchConfig):
        self.config = config
        
        # Logistic regression
        self.model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
        
        # Random Forest for ensemble
        self.forest = RandomForestClassifier(
            n_estimators=20,          # Reduced trees
            max_depth=5,              # Reduced depth  
            min_samples_split=10,     # Require more samples to split
            min_samples_leaf=5,       # Require more samples per leaf
            max_features='sqrt',      # Limit features per tree
            random_state=42,
            class_weight='balanced'
        )
        
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Ensemble weights
        self.use_ensemble = True
        self.logistic_weight = 0.4
        self.forest_weight = 0.6
        
        # Training data storage
        self.feature_history: List[ResearchFeatures] = []
        self.signal_history: List[int] = []
        
        self.load_model()
    
    def predict(self, features: ResearchFeatures) -> Tuple[int, float, str]:
        """Generate trading signal - now with ensemble"""
        
        if not self.is_trained or features is None:
            return 0, 0.0, "model_not_ready"
        
        try:
            feature_array = features.to_array().reshape(1, -1)
            scaled_features = self.scaler.transform(feature_array)
            
            if self.use_ensemble:
                # Ensemble prediction
                log_probs = self.model.predict_proba(scaled_features)[0]
                rf_probs = self.forest.predict_proba(scaled_features)[0]
                
                # Weighted average
                ensemble_probs = (self.logistic_weight * log_probs + 
                                self.forest_weight * rf_probs)
                
                prediction = np.argmax(ensemble_probs)
                confidence = ensemble_probs[prediction]
                
                # Check model agreement for quality
                log_pred = np.argmax(log_probs)
                rf_pred = np.argmax(rf_probs)
                agreement = log_pred == rf_pred
                
                quality = self._assess_quality(confidence, agreement)
            else:
                # Original single model
                prediction = self.model.predict(scaled_features)[0]
                probabilities = self.model.predict_proba(scaled_features)[0]
                confidence = probabilities[np.where(self.model.classes_ == prediction)[0][0]]
                quality = self._assess_quality(confidence)
            
            return prediction, confidence, quality
            
        except Exception as e:
            log.error(f"Prediction error: {e}")
            return 0, 0.0, "error"
    
    def train(self):
        """Train both models"""
        
        try:
            if len(self.feature_history) < self.config.MIN_TRAINING_SAMPLES:
                log.warning("Insufficient training samples")
                return
            
            # Prepare training data
            X = np.array([f.to_array() for f in self.feature_history])
            y = np.array(self.signal_history)
            
            unique_classes, counts = np.unique(y, return_counts=True)
            if len(unique_classes) < 2:
                log.warning("Need multiple signal classes for training")
                return
            
            min_samples_per_class = min(counts)
            if min_samples_per_class < 3:
                log.warning(f"Need at least 3 samples per class, got {min_samples_per_class}")
                return
            
            # Scale features
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            
            # Train both models
            self.model.fit(X_scaled, y)
            self.forest.fit(X, y)
            self.is_trained = True
            
            # Performance comparison
            log_accuracy = self.model.score(X_scaled, y)
            rf_accuracy = self.forest.score(X, y)
            
            signal_dist = dict(zip(unique_classes, counts))
            log.info(f"Ensemble model trained on {len(X)} samples")
            log.info(f"Logistic: {log_accuracy:.3f}, Forest: {rf_accuracy:.3f}")
            log.info(f"Signal distribution: {signal_dist}")
            
            self.save_model()
            
        except Exception as e:
            log.error(f"Training error: {e}")
    
    def save_model(self):
        """Save both models"""
        try:
            if self.is_trained:
                joblib.dump(self.model, self.config.MODEL_PATH)
                joblib.dump(self.forest, self.config.MODEL_PATH.replace('.joblib', '_forest.joblib'))
                joblib.dump(self.scaler, self.config.SCALER_PATH)
                log.info("Ensemble model saved successfully")
        except Exception as e:
            log.error(f"Model save error: {e}")
    
    def load_model(self):
        """Load both models"""
        try:
            forest_path = self.config.MODEL_PATH.replace('.joblib', '_forest.joblib')
            
            if (os.path.exists(self.config.MODEL_PATH) and 
                os.path.exists(self.config.SCALER_PATH)):
                
                self.model = joblib.load(self.config.MODEL_PATH)
                self.scaler = joblib.load(self.config.SCALER_PATH)
                
                # Load forest if available
                if os.path.exists(forest_path):
                    self.forest = joblib.load(forest_path)
                    log.info("Ensemble model loaded successfully")
                else:
                    log.info("Logistic model loaded, forest will train from scratch")
                
                self.is_trained = True
            else:
                log.info("No saved model found - will train from scratch")
        except Exception as e:
            log.error(f"Model load error: {e}")
    
    def _assess_quality(self, confidence: float, agreement: bool = True) -> str:
        """Enhanced quality assessment"""
        base_quality = "poor"
        
        if confidence >= self.config.CONFIDENCE_HIGH:
            base_quality = "excellent"
        elif confidence >= self.config.CONFIDENCE_MODERATE:
            base_quality = "good"
        elif confidence >= self.config.CONFIDENCE_LOW:
            base_quality = "fair"
        
        # Boost quality if models agree
        if self.use_ensemble and agreement:
            if base_quality == "fair":
                return "good"
            elif base_quality == "good":
                return "excellent"
        
        return base_quality
    
    def add_training_sample(self, features: ResearchFeatures):
        """Add new training sample (unchanged)"""
        signal = self._generate_signal_from_features(features)
        
        self.feature_history.append(features)
        self.signal_history.append(signal)
        
        max_history = self.config.ML_LOOKBACK * 2
        if len(self.feature_history) > max_history:
            self.feature_history = self.feature_history[-self.config.ML_LOOKBACK:]
            self.signal_history = self.signal_history[-self.config.ML_LOOKBACK:]
        
        if self._should_retrain():
            self.train()

    def _generate_signal_from_features(self, features: ResearchFeatures) -> int:
        """BALANCED: Professional signal generation - prevents overfitting while maintaining quality"""
        
        # 1. MARKET REGIME IDENTIFICATION (Moderately Relaxed)
        trending_up = (
            features.ema_trend_15m > 0.0005 and          # RELAXED from 0.0008 (38% more permissive)
            features.price_vs_sma_15m > 0.002 and        # RELAXED from 0.003 (33% more permissive)
            features.rsi_15m > 35 and features.rsi_15m < 87  # Slightly wider range
        )
        
        trending_down = (
            features.ema_trend_15m < -0.0005 and         # RELAXED from -0.0008
            features.price_vs_sma_15m < -0.002 and       # RELAXED from -0.003
            features.rsi_15m > 13 and features.rsi_15m < 65  # Slightly wider range
        )
        
        # Range-bound: Slightly more permissive
        ranging_market = (
            abs(features.ema_trend_15m) < 0.0007 and     # INCREASED from 0.0005
            abs(features.ema_trend_5m) < 0.0004          # INCREASED from 0.0003
        )
        
        # 2. VOLUME ANALYSIS (Moderately Relaxed)
        strong_volume = features.volume_ratio_15m > 1.3 or features.volume_breakout_5m  # RELAXED from 1.5
        decent_volume = features.volume_ratio_5m > 1.0   # RELAXED from 1.1
        
        # 3. TRENDING MARKET SIGNALS (Highest Priority)
        
        # Band Walking (Moderately relaxed)
        if trending_up:
            upper_band_walk = (
                features.bb_position_15m > 0.7 and       # RELAXED from 0.75
                features.bb_position_5m > 0.55 and       # RELAXED from 0.6
                strong_volume and                        # Volume support
                features.rsi_15m < 82                    # RELAXED from 80
            )
            if upper_band_walk:
                return 1  # BUY - Ride the trend
        
        elif trending_down:
            lower_band_walk = (
                features.bb_position_15m < 0.3 and       # RELAXED from 0.25
                features.bb_position_5m < 0.45 and       # RELAXED from 0.4
                strong_volume and                        # Volume support
                features.rsi_15m > 18                    # RELAXED from 20
            )
            if lower_band_walk:
                return 2  # SELL - Ride the trend
        
        # 4. BREAKOUT SIGNALS (Moderately relaxed)
        
        if trending_up or not ranging_market:
            bullish_breakout = (
                features.bb_position_5m > 0.75 and       # RELAXED from 0.8
                features.rsi_5m < 78 and                 # RELAXED from 75
                features.volume_breakout_5m and          # Volume confirmation
                features.ema_trend_5m > 0.0002           # RELAXED from 0.0003
            )
            if bullish_breakout:
                return 1  # BUY - Breakout momentum
        
        elif trending_down or not ranging_market:
            bearish_breakout = (
                features.bb_position_5m < 0.25 and       # RELAXED from 0.2
                features.rsi_5m > 22 and                 # RELAXED from 25
                features.volume_breakout_5m and          # Volume confirmation
                features.ema_trend_5m < -0.0002          # RELAXED from -0.0003
            )
            if bearish_breakout:
                return 2  # SELL - Breakout momentum
        
        # 5. PULLBACK ENTRIES (Moderately relaxed)
        
        if trending_up:
            bullish_pullback = (
                features.rsi_5m < 48 and                 # RELAXED from 45
                features.bb_position_5m < 0.55 and       # RELAXED from 0.5
                decent_volume and                        # Volume support
                features.ema_trend_5m > -0.0002          # RELAXED from -0.0001
            )
            if bullish_pullback:
                return 1  # BUY - Pullback entry
        
        elif trending_down:
            bearish_pullback = (
                features.rsi_5m > 52 and                 # RELAXED from 55
                features.bb_position_5m > 0.45 and       # RELAXED from 0.5
                decent_volume and                        # Volume support
                features.ema_trend_5m < 0.0002           # RELAXED from 0.0001
            )
            if bearish_pullback:
                return 2  # SELL - Pullback entry
        
        # 6. MEAN REVERSION SIGNALS (More permissive in ranging markets)
        
        if ranging_market:
            oversold_reversal = (
                features.rsi_5m < 28 and                 # RELAXED from 25
                features.bb_position_5m < 0.2 and        # RELAXED from 0.15
                features.volume_ratio_5m > 1.1 and       # RELAXED from 1.3
                features.bb_position_15m < 0.35          # RELAXED from 0.3
            )
            if oversold_reversal:
                return 1  # BUY - Mean reversion
            
            overbought_reversal = (
                features.rsi_5m > 72 and                 # RELAXED from 75
                features.bb_position_5m > 0.8 and        # RELAXED from 0.85
                features.volume_ratio_5m > 1.1 and       # RELAXED from 1.3
                features.bb_position_15m > 0.65          # RELAXED from 0.7
            )
            if overbought_reversal:
                return 2  # SELL - Mean reversion
        
        # 7. SIMPLE MOMENTUM BACKUP (NEW - Catch missed opportunities)
        # Only if no other signals triggered and not in strong counter-trend
        
        if not trending_down:  # Not in strong downtrend
            simple_momentum_up = (
                features.ema_trend_5m > 0.0002 and       # Simple 5m uptrend
                features.rsi_5m > 45 and features.rsi_5m < 75 and
                features.volume_ratio_5m > 0.9 and       # Minimum volume
                features.bb_position_5m > 0.4             # Above lower region
            )
            if simple_momentum_up:
                return 1  # BUY - Simple momentum
        
        if not trending_up:  # Not in strong uptrend
            simple_momentum_down = (
                features.ema_trend_5m < -0.0002 and      # Simple 5m downtrend
                features.rsi_5m > 25 and features.rsi_5m < 55 and
                features.volume_ratio_5m > 0.9 and       # Minimum volume
                features.bb_position_5m < 0.6             # Below upper region
            )
            if simple_momentum_down:
                return 2  # SELL - Simple momentum
        
        # 8. QUALITY FILTERS (Only filter extreme cases)
        
        # Only filter truly problematic conditions
        extreme_conditions = (
            features.volume_ratio_5m < 0.6 or            # VERY weak volume only
            features.rsi_15m > 90 or features.rsi_15m < 10 or  # Extreme RSI only
            features.timeframe_alignment < 0.05          # VERY poor alignment only
        )
        
        if extreme_conditions:
            return 0  # HOLD - Avoid extreme conditions
        
        # 9. DEFAULT: HOLD
        return 0  # HOLD - No clear setup
    
    def _should_retrain(self) -> bool:
        """Enhanced retraining logic (unchanged)"""
        
        if len(self.signal_history) < self.config.MIN_TRAINING_SAMPLES:
            return False
        
        # Only retrain every N samples
        if len(self.signal_history) % self.config.ML_RETRAIN_FREQUENCY != 0:
            return False
        
        # Ensure signal diversity
        unique_classes = len(np.unique(self.signal_history))
        return unique_classes >= 3