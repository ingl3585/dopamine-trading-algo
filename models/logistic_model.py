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
        """RESEARCH-VALIDATED: Professional signal generation with proper hierarchy"""
        
        # 1. MARKET REGIME IDENTIFICATION (Critical First Step)
        # Research shows this must come first to avoid wrong-footed trades
        trending_up = (
            features.ema_trend_15m > 0.0008 and          # Strong 15m uptrend
            features.price_vs_sma_15m > 0.003 and        # Price well above SMA
            features.rsi_15m > 40 and features.rsi_15m < 85  # Not extreme
        )
        
        trending_down = (
            features.ema_trend_15m < -0.0008 and         # Strong 15m downtrend  
            features.price_vs_sma_15m < -0.003 and       # Price well below SMA
            features.rsi_15m > 15 and features.rsi_15m < 60  # Not extreme
        )
        
        # Range-bound: Weak trends on both timeframes
        ranging_market = (
            abs(features.ema_trend_15m) < 0.0005 and     # Weak 15m trend
            abs(features.ema_trend_5m) < 0.0003          # Weak 5m trend
        )
        
        # 2. VOLUME ANALYSIS (Essential for All Signals)
        # Research: Volume confirmation is critical for signal reliability
        strong_volume = features.volume_ratio_15m > 1.5 or features.volume_breakout_5m
        decent_volume = features.volume_ratio_5m > 1.1
        
        # 3. TRENDING MARKET SIGNALS (Highest Priority - Research Validated)
        
        # Band Walking (John Bollinger's research - continuation signals)
        if trending_up:
            # Upper band walk = bullish continuation
            upper_band_walk = (
                features.bb_position_15m > 0.75 and      # Near/above upper band
                features.bb_position_5m > 0.6 and        # 5m confirmation
                strong_volume and                        # Volume support
                features.rsi_15m < 80                    # Not extremely overbought
            )
            if upper_band_walk:
                return 1  # BUY - Ride the trend
        
        elif trending_down:
            # Lower band walk = bearish continuation  
            lower_band_walk = (
                features.bb_position_15m < 0.25 and      # Near/below lower band
                features.bb_position_5m < 0.4 and        # 5m confirmation
                strong_volume and                        # Volume support
                features.rsi_15m > 20                    # Not extremely oversold
            )
            if lower_band_walk:
                return 2  # SELL - Ride the trend
        
        # 4. BREAKOUT SIGNALS (Trend Initiation - High Priority)
        
        if trending_up or not ranging_market:  # Uptrend or weak downtrend
            bullish_breakout = (
                features.bb_position_5m > 0.8 and        # Breaking above upper band
                features.rsi_5m < 75 and                 # Room to run
                features.volume_breakout_5m and          # Volume confirmation
                features.ema_trend_5m > 0.0003           # 5m momentum
            )
            if bullish_breakout:
                return 1  # BUY - Breakout momentum
        
        elif trending_down or not ranging_market:  # Downtrend or weak uptrend
            bearish_breakout = (
                features.bb_position_5m < 0.2 and        # Breaking below lower band
                features.rsi_5m > 25 and                 # Room to fall
                features.volume_breakout_5m and          # Volume confirmation
                features.ema_trend_5m < -0.0003          # 5m momentum
            )
            if bearish_breakout:
                return 2  # SELL - Breakout momentum
        
        # 5. PULLBACK ENTRIES (Trend Following - Medium Priority)
        # Research: High probability entries in established trends
        
        if trending_up:
            # Buy pullbacks in uptrends
            bullish_pullback = (
                features.rsi_5m < 45 and                 # Oversold in uptrend
                features.bb_position_5m < 0.5 and        # Below middle band
                decent_volume and                        # Volume support
                features.ema_trend_5m > -0.0001          # 5m not counter-trending
            )
            if bullish_pullback:
                return 1  # BUY - Pullback entry
        
        elif trending_down:
            # Sell bounces in downtrends
            bearish_pullback = (
                features.rsi_5m > 55 and                 # Overbought in downtrend
                features.bb_position_5m > 0.5 and        # Above middle band
                decent_volume and                        # Volume support
                features.ema_trend_5m < 0.0001           # 5m not counter-trending
            )
            if bearish_pullback:
                return 2  # SELL - Pullback entry
        
        # 6. MEAN REVERSION SIGNALS (Range-Bound Markets Only - Lower Priority)
        # Research: Only use in confirmed ranging markets
        
        if ranging_market:
            # Oversold reversal
            oversold_reversal = (
                features.rsi_5m < 25 and                 # Oversold
                features.bb_position_5m < 0.15 and       # Below lower band
                features.volume_ratio_5m > 1.3 and       # Volume spike
                features.bb_position_15m < 0.3           # 15m confirmation
            )
            if oversold_reversal:
                return 1  # BUY - Mean reversion
            
            # Overbought reversal
            overbought_reversal = (
                features.rsi_5m > 75 and                 # Overbought
                features.bb_position_5m > 0.85 and       # Above upper band
                features.volume_ratio_5m > 1.3 and       # Volume spike
                features.bb_position_15m > 0.7           # 15m confirmation
            )
            if overbought_reversal:
                return 2  # SELL - Mean reversion
        
        # 7. QUALITY FILTERS (Research-Based Risk Management)
        
        # Avoid low-quality setups
        low_quality_conditions = (
            features.volume_ratio_5m < 0.8 or            # Very weak volume
            (trending_up and features.rsi_15m > 85) or   # Extreme overbought in uptrend
            (trending_down and features.rsi_15m < 15) or # Extreme oversold in downtrend
            features.timeframe_alignment < 0.1           # Very poor alignment
        )
        
        if low_quality_conditions:
            return 0  # HOLD - Avoid low quality
        
        # 8. DEFAULT: HOLD
        # Research: "Do nothing" is often the best action
        return 0  # HOLD - No clear high-probability setup
    
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