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
            n_estimators=50,
            max_depth=8,
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
        """Enhanced signal generation with volume confirmation (unchanged)"""
        
        # 1. Trend Analysis (15m timeframe)
        trend_bullish = (
            features.ema_trend_15m > -0.002 and          # Allow slight negative trend
            features.price_vs_sma_15m > -0.005 and       # More lenient price position
            25 < features.rsi_15m < 80                    # Wider RSI range for trend-following
        )
        
        trend_bearish = (
            features.ema_trend_15m < 0.002 and           # Allow slight positive trend
            features.price_vs_sma_15m < 0.005 and        # More lenient price position
            20 < features.rsi_15m < 75                    # Wider RSI range
        )
        
        # 2. Entry Signals (5m timeframe)
        rsi_buy_zone = features.rsi_5m < 60              # Expanded from 30 (trend-following)
        rsi_sell_zone = features.rsi_5m > 40             # Expanded from 70 (trend-following)
        bb_buy_zone = features.bb_position_5m < 0.6      # Expanded zones
        bb_sell_zone = features.bb_position_5m > 0.4     # Expanded zones
        
        # Volume confirmation
        volume_confirm = (features.volume_ratio_5m > 1.0 or 
                         features.volume_breakout_5m)    # Either above average or breakout
        
        # 3. Multi-timeframe confluence signals
        confluence_buy = (
            trend_bullish and 
            (rsi_buy_zone or bb_buy_zone) and            # At least one entry signal
            volume_confirm                               # Volume confirmation required
        )
        
        confluence_sell = (
            trend_bearish and 
            (rsi_sell_zone or bb_sell_zone) and          # At least one entry signal
            volume_confirm                               # Volume confirmation required
        )
        
        # 4. Trend continuation signals (for band walking)
        band_walk_up = (features.bb_position_15m > 0.8 and 
                       features.bb_position_5m > 0.6 and
                       features.rsi_15m > 50)
        
        band_walk_down = (features.bb_position_15m < 0.2 and 
                         features.bb_position_5m < 0.4 and
                         features.rsi_15m < 50)
        
        # Return signals with priority
        if confluence_buy or band_walk_up:
            return 1  # BUY
        elif confluence_sell or band_walk_down:
            return 2  # SELL
        else:
            return 0  # HOLD
    
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