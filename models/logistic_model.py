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
        """Train both models - CLEAN VERSION"""
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
            log.info("Scaling features and training ensemble models...")
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            
            # Train both models (fast, no progress bar needed)
            self.model.fit(X_scaled, y)
            self.forest.fit(X, y)
            self.is_trained = True
            
            # Performance comparison
            log_accuracy = self.model.score(X_scaled, y)
            rf_accuracy = self.forest.score(X, y)
            
            signal_dist = dict(zip(unique_classes, counts))
            log.info(f"Ensemble model trained: {len(X)} samples")
            log.info(f"Logistic Regression: {log_accuracy:.3f} accuracy")
            log.info(f"Random Forest: {rf_accuracy:.3f} accuracy")
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
        """BALANCED: More signals while maintaining quality - targeting ~75% HOLD, ~12% BUY, ~13% SELL"""
        
        # 1. BASIC SAFETY FILTERS (Only extreme danger zones)
        extreme_conditions = (
            features.volume_ratio_5m < 0.3 or                    # Very weak volume (was 0.4)
            features.rsi_15m > 95 or features.rsi_15m < 5 or     # Extreme 15m RSI (was 92/8)
            features.rsi_5m > 97 or features.rsi_5m < 3          # Extreme 5m RSI (was 95/5)
        )
        if extreme_conditions:
            return 0  # HOLD - Too dangerous
        
        # 2. MARKET REGIME ANALYSIS (Relaxed thresholds)
        strong_uptrend = (
            features.ema_trend_15m > 0.0002 and                  # Reduced from 0.0003
            features.price_vs_sma_15m > 0.001 and               # Reduced from 0.0015
            features.rsi_15m > 20 and features.rsi_15m < 88      # Wider range (was 25-85)
        )
        
        moderate_uptrend = (
            features.ema_trend_15m > 0.00005 and                 # Reduced from 0.0001
            features.price_vs_sma_15m > 0.0003 and              # Reduced from 0.0005
            features.rsi_15m > 25 and features.rsi_15m < 82      # Wider range (was 30-80)
        )
        
        strong_downtrend = (
            features.ema_trend_15m < -0.0002 and                 # Reduced from -0.0003
            features.price_vs_sma_15m < -0.001 and              # Reduced from -0.0015
            features.rsi_15m > 12 and features.rsi_15m < 80      # Wider range (was 15-75)
        )
        
        moderate_downtrend = (
            features.ema_trend_15m < -0.00005 and               # Reduced from -0.0001
            features.price_vs_sma_15m < -0.0003 and             # Reduced from -0.0005
            features.rsi_15m > 18 and features.rsi_15m < 75      # Wider range (was 20-70)
        )
        
        ranging_market = (
            abs(features.ema_trend_15m) < 0.0005 and            # Increased from 0.0004
            abs(features.ema_trend_5m) < 0.0008                 # Increased from 0.0006
        )
        
        # 3. VOLUME ANALYSIS (More permissive)
        exceptional_volume = features.volume_ratio_5m > 1.6     # Reduced from 1.8
        strong_volume = features.volume_ratio_5m > 1.0          # Reduced from 1.2
        good_volume = features.volume_ratio_5m > 0.7            # Reduced from 0.9
        decent_volume = features.volume_ratio_5m > 0.5          # Reduced from 0.6
        
        # 4. TIER 1 SIGNALS - PREMIUM QUALITY (More accessible)
        
        # Upper Band Walking (Strong uptrend + riding upper band)
        if strong_uptrend or moderate_uptrend:                   # Added moderate_uptrend
            upper_band_walk = (
                features.bb_position_15m > 0.6 and              # Reduced from 0.65
                features.bb_position_5m > 0.55 and              # Reduced from 0.6
                good_volume and                                  # Was strong_volume
                features.rsi_15m < 88 and                        # Relaxed from 85
                features.rsi_5m > 30 and features.rsi_5m < 85    # Wider range (was 35-80)
            )
            if upper_band_walk:
                return 1  # PREMIUM BUY - Band walking
        
        # Lower Band Walking (Strong downtrend + riding lower band)
        if strong_downtrend or moderate_downtrend:               # Added moderate_downtrend
            lower_band_walk = (
                features.bb_position_15m < 0.4 and              # Increased from 0.35
                features.bb_position_5m < 0.45 and              # Increased from 0.4
                good_volume and                                  # Was strong_volume
                features.rsi_15m > 12 and                        # Relaxed from 15
                features.rsi_5m > 15 and features.rsi_5m < 70    # Wider range (was 20-65)
            )
            if lower_band_walk:
                return 2  # PREMIUM SELL - Band walking
        
        # 5. TIER 2 SIGNALS - TREND CONTINUATION & PULLBACKS (More lenient)
        
        # Bullish Pullback in Uptrend (Buy the dip)
        if moderate_uptrend or strong_uptrend:
            bullish_pullback = (
                features.rsi_5m < 60 and features.rsi_5m > 25 and     # Wider range (was 55/30)
                features.bb_position_5m < 0.7 and features.bb_position_5m > 0.2 and  # Wider range
                decent_volume and                                     # Was good_volume
                features.ema_trend_5m > -0.0005                       # More lenient (was -0.0004)
            )
            if bullish_pullback:
                return 1  # BUY - Pullback entry
        
        # Bearish Pullback in Downtrend (Sell the bounce)
        if moderate_downtrend or strong_downtrend:
            bearish_pullback = (
                features.rsi_5m > 40 and features.rsi_5m < 75 and     # Wider range (was 45/70)
                features.bb_position_5m > 0.3 and features.bb_position_5m < 0.8 and  # Wider range
                decent_volume and                                     # Was good_volume
                features.ema_trend_5m < 0.0005                        # More lenient (was 0.0004)
            )
            if bearish_pullback:
                return 2  # SELL - Pullback entry
        
        # 6. BREAKOUT SIGNALS (More accessible volume requirements)
        
        if good_volume or strong_volume or exceptional_volume:    # Added good_volume
            # Bullish Breakout
            bullish_breakout = (
                features.bb_position_5m > 0.75 and                   # Reduced from 0.8
                features.rsi_5m > 40 and features.rsi_5m < 88 and    # Wider range (was 45-85)
                features.ema_trend_5m > 0.00005 and                  # Reduced from 0.0001
                (features.volume_breakout_5m or strong_volume)        # Added volume alternative
            )
            if bullish_breakout:
                return 1  # BUY - Breakout
            
            # Bearish Breakdown
            bearish_breakdown = (
                features.bb_position_5m < 0.25 and                   # Increased from 0.2
                features.rsi_5m > 12 and features.rsi_5m < 60 and    # Wider range (was 15-55)
                features.ema_trend_5m < -0.00005 and                 # Reduced from -0.0001
                (features.volume_breakout_5m or strong_volume)        # Added volume alternative
            )
            if bearish_breakdown:
                return 2  # SELL - Breakdown
        
        # 7. MOMENTUM CONTINUATION (More permissive)
        
        # Strong momentum alignment (relaxed requirements)
        strong_momentum_up = (
            features.ema_trend_15m > 0.00005 and                 # Reduced from 0.0001
            features.ema_trend_5m > 0.0001 and                   # Reduced from 0.0002
            features.ema_trend_1m > 0.00002 and                  # Reduced from 0.00005
            decent_volume                                         # Was good_volume
        )
        if strong_momentum_up and not strong_downtrend:
            momentum_long = (
                features.rsi_5m > 35 and features.rsi_5m < 82 and    # Wider range (was 40-78)
                features.bb_position_5m > 0.25                       # Reduced from 0.3
            )
            if momentum_long:
                return 1  # BUY - Momentum continuation
        
        strong_momentum_down = (
            features.ema_trend_15m < -0.00005 and                # Reduced from -0.0001
            features.ema_trend_5m < -0.0001 and                  # Reduced from -0.0002
            features.ema_trend_1m < -0.00002 and                 # Reduced from -0.00005
            decent_volume                                         # Was good_volume
        )
        if strong_momentum_down and not strong_uptrend:
            momentum_short = (
                features.rsi_5m > 18 and features.rsi_5m < 65 and    # Wider range (was 22-60)
                features.bb_position_5m < 0.75                       # Increased from 0.7
            )
            if momentum_short:
                return 2  # SELL - Momentum continuation
        
        # 8. MEAN REVERSION SIGNALS (More accessible)
        
        if ranging_market:
            # Oversold Reversal
            oversold_reversal = (
                features.rsi_5m < 35 and                              # Increased from 32
                features.bb_position_5m < 0.3 and                    # Increased from 0.25
                features.rsi_15m < 50 and                             # Increased from 45
                decent_volume and                                     # Was good_volume
                features.bb_position_15m < 0.45                       # Increased from 0.4
            )
            if oversold_reversal:
                return 1  # BUY - Mean reversion
            
            # Overbought Reversal
            overbought_reversal = (
                features.rsi_5m > 65 and                              # Reduced from 68
                features.bb_position_5m > 0.7 and                    # Reduced from 0.75
                features.rsi_15m > 50 and                             # Reduced from 55
                decent_volume and                                     # Was good_volume
                features.bb_position_15m > 0.55                       # Reduced from 0.6
            )
            if overbought_reversal:
                return 2  # SELL - Mean reversion
        
        # 9. STANDARD TREND FOLLOWING (Much more accessible)
        
        # Basic trend following without strict regime requirements
        if not strong_downtrend:  # Don't fight strong downtrend
            basic_long = (
                features.ema_trend_5m > 0.00005 and                  # Reduced from 0.0001
                features.rsi_5m > 32 and features.rsi_5m < 78 and    # Wider range (was 38-72)
                features.bb_position_5m > 0.25 and                   # Reduced from 0.3
                decent_volume and                                     # Same
                features.price_vs_sma_5m > -0.003                    # More lenient (was -0.002)
            )
            if basic_long:
                return 1  # BUY - Basic trend
        
        if not strong_uptrend:  # Don't fight strong uptrend
            basic_short = (
                features.ema_trend_5m < -0.00005 and                 # Reduced from -0.0001
                features.rsi_5m > 22 and features.rsi_5m < 68 and    # Wider range (was 28-62)
                features.bb_position_5m < 0.75 and                   # Increased from 0.7
                decent_volume and                                     # Same
                features.price_vs_sma_5m < 0.003                     # More lenient (was 0.002)
            )
            if basic_short:
                return 2  # SELL - Basic trend
        
        # 10. VOLUME SPIKE OPPORTUNITIES (More accessible)
        
        if strong_volume or exceptional_volume:                   # Removed exceptional_volume requirement
            # Volume spike long (assuming accumulation)
            if (features.rsi_5m > 30 and features.rsi_5m < 80 and    # Wider range (was 35-75)
                features.bb_position_5m > 0.2 and features.bb_position_5m < 0.9 and  # Wider range
                features.ema_trend_5m > -0.0003):                    # More lenient (was -0.0002)
                return 1  # BUY - Volume spike
            
            # Volume spike short (assuming distribution)  
            if (features.rsi_5m > 20 and features.rsi_5m < 70 and    # Wider range (was 25-65)
                features.bb_position_5m > 0.1 and features.bb_position_5m < 0.8 and  # Wider range
                features.ema_trend_5m < 0.0003):                     # More lenient (was 0.0002)
                return 2  # SELL - Volume spike
        
        # 11. CONFLUENCE SIGNALS (More accessible thresholds)
        
        # Multi-factor bullish confluence
        bullish_confluence = (
            features.timeframe_alignment > 0.5 and                   # Reduced from 0.6
            features.entry_timing_quality > 0.5 and                  # Reduced from 0.6
            features.rsi_5m > 30 and features.rsi_5m < 75 and        # Wider range (was 35-70)
            features.bb_position_5m > 0.2 and features.bb_position_5m < 0.85 and  # Wider range
            decent_volume                                             # Same
        )
        if bullish_confluence and not strong_downtrend:
            return 1  # BUY - Confluence
        
        # Multi-factor bearish confluence
        bearish_confluence = (
            features.timeframe_alignment > 0.5 and                   # Reduced from 0.6
            features.entry_timing_quality > 0.5 and                  # Reduced from 0.6
            features.rsi_5m > 25 and features.rsi_5m < 70 and        # Wider range (was 30-65)
            features.bb_position_5m > 0.15 and features.bb_position_5m < 0.8 and  # Wider range
            decent_volume and                                         # Same
            features.ema_trend_5m < -0.00002                          # More lenient (was -0.00005)
        )
        if bearish_confluence and not strong_uptrend:
            return 2  # SELL - Confluence
        
        # 12. NEW: SIMPLE DIRECTIONAL BIAS (Catch more basic setups)
        
        # Simple bullish bias (when most indicators point up)
        simple_bullish = (
            features.ema_trend_5m > 0 and                            # Any upward movement
            features.rsi_5m > 35 and features.rsi_5m < 75 and        # Reasonable RSI
            features.bb_position_5m > 0.3 and                        # Above lower third
            decent_volume and                                         # Basic volume
            features.price_vs_sma_5m > -0.001                        # Not far below SMA
        )
        if simple_bullish and not (strong_downtrend or moderate_downtrend):
            return 1  # BUY - Simple bias
        
        # Simple bearish bias (when most indicators point down)
        simple_bearish = (
            features.ema_trend_5m < 0 and                            # Any downward movement
            features.rsi_5m > 25 and features.rsi_5m < 65 and        # Reasonable RSI
            features.bb_position_5m < 0.7 and                        # Below upper third
            decent_volume and                                         # Basic volume
            features.price_vs_sma_5m < 0.001                         # Not far above SMA
        )
        if simple_bearish and not (strong_uptrend or moderate_uptrend):
            return 2  # SELL - Simple bias
        
        # 13. DEFAULT: HOLD (Wait for better setup)
        return 0  # HOLD - No qualifying setup
    
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