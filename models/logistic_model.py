# models/logistic_model.py

import numpy as np
import joblib
import os
import logging
from typing import Tuple, List
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
            
            # Convert to trading signal
            action, confidence = self._convert_prediction(prediction, probabilities)
            quality = self._assess_quality(confidence)
            
            return action, confidence, quality
            
        except Exception as e:
            log.error(f"Prediction error: {e}")
            return 0, 0.0, "error"
    
    def add_training_sample(self, features: ResearchFeatures, price_change: float):
        """Add new training sample"""
        
        # Convert price change to signal class
        signal = self._price_change_to_signal(price_change)
        
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
            
            # Validate we have multiple classes
            if len(np.unique(y)) < 2:
                log.warning("Need multiple signal classes for training")
                return
            
            # Split data if we have enough samples
            if len(X) > 40:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            else:
                X_train, y_train = X, y
                X_test, y_test = None, None
            
            # Scale features
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            self.is_trained = True
            
            # Evaluate if we have test data
            if X_test is not None:
                X_test_scaled = self.scaler.transform(X_test)
                accuracy = self.model.score(X_test_scaled, y_test)
                log.info(f"Model retrained - Test accuracy: {accuracy:.3f}")
            else:
                log.info("Model retrained on full dataset")
            
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
    
    def _convert_prediction(self, prediction: int, probabilities: np.ndarray) -> Tuple[int, float]:
        """Convert model prediction to trading signal"""
        if prediction == 2:  # Buy signal
            return 1, probabilities[2]
        elif prediction == 0:  # Sell signal
            return 2, probabilities[0]
        else:  # Hold signal
            return 0, probabilities[1]
    
    def _assess_quality(self, confidence: float) -> str:
        """Assess signal quality based on confidence"""
        if confidence >= 0.8:
            return "excellent"
        elif confidence >= 0.7:
            return "good"
        elif confidence >= 0.6:
            return "fair"
        else:
            return "poor"
    
    def _price_change_to_signal(self, price_change: float) -> int:
        # Make thresholds more sensitive to generate more varied signals
        if price_change > 0.001:  # Reduced from 0.002
            return 2  # Buy signal
        elif price_change < -0.001:  # Reduced from -0.002  
            return 0  # Sell signal
        else:
            return 1  # Hold signal
    
    def _should_retrain(self) -> bool:
        if len(self.signal_history) < self.config.MIN_TRAINING_SAMPLES:
            return False
        
        # Only train when we have natural class diversity
        unique_classes, counts = np.unique(self.signal_history, return_counts=True)
        min_samples_per_class = min(counts) if len(counts) > 0 else 0

        # Require at least 2 classes with minimum 3 samples each
        has_diversity = len(unique_classes) >= 2 and min_samples_per_class >= 3
        
        return has_diversity