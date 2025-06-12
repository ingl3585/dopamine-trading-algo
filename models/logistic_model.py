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
        """IMPROVED: Research-backed signal generation with balanced quality/quantity"""
        
        # 1. MARKET REGIME IDENTIFICATION (Relaxed for more signals)
        trending_up = (
            features.ema_trend_15m > 0.0001 and          # RELAXED from 0.0003 (3x more permissive)
            features.price_vs_sma_15m > 0.0005 and      # RELAXED from 0.001 (2x more permissive)
            features.rsi_15m > 25 and features.rsi_15m < 85  # Wider range for more signals
        )
        
        trending_down = (
            features.ema_trend_15m < -0.0001 and         # RELAXED from -0.0003 (3x more permissive)        
            features.price_vs_sma_15m < -0.0005 and     # RELAXED from -0.001 (2x more permissive)
            features.rsi_15m > 15 and features.rsi_15m < 75  # Wider range for more signals
        )
        
        # Range-bound: More permissive for mean reversion
        ranging_market = (
            abs(features.ema_trend_15m) < 0.0015 and     # RELAXED from 0.001 (1.5x more permissive)
            abs(features.ema_trend_5m) < 0.0008          # RELAXED from 0.0005 (1.6x more permissive)
        )
        
        # 2. VOLUME ANALYSIS (Relaxed thresholds)
        strong_volume = features.volume_ratio_15m > 1.0 or features.volume_breakout_5m  # RELAXED from 1.1
        decent_volume = features.volume_ratio_5m > 0.7   # RELAXED from 0.8
        
        # 3. TIER 1 SIGNALS - Premium Quality (Your original logic with relaxed thresholds)
        
        # Band Walking (Premium signals)
        if trending_up:
            upper_band_walk = (
                features.bb_position_15m > 0.6 and       # RELAXED from 0.7
                features.bb_position_5m > 0.5 and        # RELAXED from 0.55
                strong_volume and                        # Volume support
                features.rsi_15m < 80
            )
            if upper_band_walk:
                return 1  # TIER 1 BUY - Premium signal
        
        elif trending_down:
            lower_band_walk = (
                features.bb_position_15m < 0.4 and       # RELAXED from 0.3
                features.bb_position_5m < 0.5 and        # RELAXED from 0.45
                strong_volume and                        # Volume support
                features.rsi_15m > 20
            )
            if lower_band_walk:
                return 2  # TIER 1 SELL - Premium signal
        
        # 4. TIER 2 SIGNALS - Standard Quality (NEW - Fill the signal gap)
        
        # Simple trend following (generates more signals)
        if trending_up or (features.ema_trend_5m > 0.00005 and not trending_down):  # RELAXED from 0.0001
            standard_buy = (
                features.rsi_5m > 35 and features.rsi_5m < 70 and    # Good RSI zone
                features.bb_position_5m > 0.3 and                    # Above lower band
                decent_volume and                                     # Minimum volume
                features.price_vs_sma_5m > -0.001                    # Not too far below SMA
            )
            if standard_buy:
                return 1  # TIER 2 BUY - Standard signal
        
        if trending_down or (features.ema_trend_5m < -0.00005 and not trending_up):  # RELAXED from -0.0001
            standard_sell = (
                features.rsi_5m > 30 and features.rsi_5m < 65 and    # Good RSI zone
                features.bb_position_5m < 0.7 and                    # Below upper band
                decent_volume and                                     # Minimum volume
                features.price_vs_sma_5m < 0.001                     # Not too far above SMA
            )
            if standard_sell:
                return 2  # TIER 2 SELL - Standard signal
        
        # 5. BREAKOUT SIGNALS (Moderately relaxed)
        
        if not ranging_market:  # Only in non-ranging conditions
            bullish_breakout = (
                features.bb_position_5m > 0.7 and       # RELAXED from 0.75
                features.rsi_5m < 75 and                 # RELAXED from 78
                features.volume_breakout_5m and          # Volume confirmation
                features.ema_trend_5m > 0.00005          # RELAXED from 0.0001
            )
            if bullish_breakout:
                return 1  # BUY - Breakout momentum
            
            bearish_breakout = (
                features.bb_position_5m < 0.3 and       # RELAXED from 0.25
                features.rsi_5m > 25 and                 # RELAXED from 22
                features.volume_breakout_5m and          # Volume confirmation
                features.ema_trend_5m < -0.00005         # RELAXED from -0.0001
            )
            if bearish_breakout:
                return 2  # SELL - Breakout momentum
        
        # 6. PULLBACK ENTRIES (More opportunities)
        
        if trending_up:
            bullish_pullback = (
                features.rsi_5m < 50 and                 # RELAXED from 48
                features.bb_position_5m < 0.6 and       # RELAXED from 0.55
                decent_volume and                        # Volume support
                features.ema_trend_5m > -0.0003          # RELAXED from -0.0002
            )
            if bullish_pullback:
                return 1  # BUY - Pullback entry
        
        elif trending_down:
            bearish_pullback = (
                features.rsi_5m > 50 and                 # RELAXED from 52
                features.bb_position_5m > 0.4 and       # RELAXED from 0.45
                decent_volume and                        # Volume support
                features.ema_trend_5m < 0.0003           # RELAXED from 0.0002
            )
            if bearish_pullback:
                return 2  # SELL - Pullback entry
        
        # 7. MEAN REVERSION SIGNALS (Enhanced for ranging markets)
        
        if ranging_market:
            oversold_reversal = (
                features.rsi_5m < 30 and                 # RELAXED from 28
                features.bb_position_5m < 0.25 and       # RELAXED from 0.2
                features.volume_ratio_5m > 0.7 and       # RELAXED from 0.9
                features.bb_position_15m < 0.4           # RELAXED from 0.35
            )
            if oversold_reversal:
                return 1  # BUY - Mean reversion
            
            overbought_reversal = (
                features.rsi_5m > 70 and                 # RELAXED from 72
                features.bb_position_5m > 0.75 and       # RELAXED from 0.8
                features.volume_ratio_5m > 0.7 and       # RELAXED from 0.9
                features.bb_position_15m > 0.6           # RELAXED from 0.65
            )
            if overbought_reversal:
                return 2  # SELL - Mean reversion
        
        # 8. MOMENTUM CONTINUATION (NEW - Catches more opportunities)
        
        # Simple momentum when not in strong opposite trend
        if not trending_down:
            momentum_up = (
                features.ema_trend_5m > 0.00005 and      # RELAXED from 0.0001
                features.rsi_5m > 40 and features.rsi_5m < 70 and  # RELAXED ranges
                features.volume_ratio_5m > 0.6 and       # RELAXED from 0.7
                features.bb_position_5m > 0.35           # RELAXED from 0.4
            )
            if momentum_up:
                return 1  # BUY - Momentum continuation
        
        if not trending_up:
            momentum_down = (
                features.ema_trend_5m < -0.00005 and     # RELAXED from -0.0001
                features.rsi_5m > 30 and features.rsi_5m < 60 and  # RELAXED ranges
                features.volume_ratio_5m > 0.6 and       # RELAXED from 0.7
                features.bb_position_5m < 0.65           # RELAXED from 0.6
            )
            if momentum_down:
                return 2  # SELL - Momentum continuation
        
        # 9. BASIC OPPORTUNITY SIGNALS (NEW - Simple signals for missed opportunities)
        
        # Very basic signals when conditions are reasonable
        basic_volume_ok = features.volume_ratio_5m > 0.6
        not_extreme_rsi = 20 < features.rsi_5m < 80
        
        if basic_volume_ok and not_extreme_rsi:
            # Simple trend + reasonable RSI
            if (features.ema_trend_5m > 0.0001 and     # RELAXED from 0.0002
                features.rsi_5m < 65 and 
                features.bb_position_5m > 0.25):
                return 1  # BASIC BUY
            
            if (features.ema_trend_5m < -0.0001 and    # RELAXED from -0.0002
                features.rsi_5m > 35 and 
                features.bb_position_5m < 0.75):
                return 2  # BASIC SELL
        
        # 10. SAFETY FILTERS (Keep essential protection)
        
        # Only filter truly dangerous conditions
        extreme_conditions = (
            features.volume_ratio_5m < 0.4 or            # VERY weak volume
            features.rsi_15m > 95 or features.rsi_15m < 5 or  # EXTREME RSI only
            features.timeframe_alignment < 0.02          # VERY poor alignment
        )
        
        if extreme_conditions:
            return 0  # HOLD - Dangerous conditions
        
        # 11. DEFAULT: HOLD
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