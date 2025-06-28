# uncertainty_estimator.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class UncertaintyEstimator(nn.Module):
    def __init__(self, input_dim: int = 64, hidden_dim: int = 32):
        super().__init__()
        
        # Epistemic uncertainty network (model uncertainty)
        self.epistemic_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Aleatoric uncertainty network (data uncertainty)
        self.aleatoric_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Ensemble for uncertainty estimation
        self.ensemble_size = 5
        self.ensemble_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(self.ensemble_size)
        ])
        
        # Convert to double precision
        self.double()
        
        # Uncertainty tracking
        self.prediction_history = deque(maxlen=500)
        self.uncertainty_calibration = deque(maxlen=100)
        
        # Optimization
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Epistemic uncertainty (what the model doesn't know)
        epistemic_uncertainty = self.epistemic_net(x)
        
        # Aleatoric uncertainty (inherent data noise)
        aleatoric_uncertainty = self.aleatoric_net(x)
        
        # Ensemble predictions for uncertainty quantification
        ensemble_preds = []
        for net in self.ensemble_nets:
            pred = net(x)
            ensemble_preds.append(pred)
        
        ensemble_preds = torch.stack(ensemble_preds)
        ensemble_mean = torch.mean(ensemble_preds, dim=0)
        ensemble_std = torch.std(ensemble_preds, dim=0)
        
        return ensemble_mean, epistemic_uncertainty, aleatoric_uncertainty
    
    def estimate_uncertainty(self, features: torch.Tensor) -> float:
        
        with torch.no_grad():
            ensemble_mean, epistemic, aleatoric = self.forward(features)
            
            # Combine uncertainties
            total_uncertainty = epistemic + aleatoric
            
            # Add ensemble disagreement as additional uncertainty signal
            ensemble_preds = []
            for net in self.ensemble_nets:
                pred = net(features)
                ensemble_preds.append(pred)
            
            ensemble_disagreement = torch.std(torch.stack(ensemble_preds))
            
            # Final uncertainty estimate
            uncertainty = float(total_uncertainty + ensemble_disagreement * 0.5)
            
            return np.clip(uncertainty, 0.0, 1.0)
    
    def update_uncertainty(self, features: torch.Tensor, prediction: float, 
                          actual_outcome: float, confidence: float):
        
        # Calculate prediction error for calibration
        prediction_error = abs(prediction - actual_outcome)
        
        # Store for calibration tracking
        uncertainty_estimate = self.estimate_uncertainty(features)
        
        calibration_data = {
            'predicted_uncertainty': uncertainty_estimate,
            'actual_error': prediction_error,
            'confidence': confidence,
            'features': features.clone()
        }
        self.uncertainty_calibration.append(calibration_data)
        
        # Update uncertainty networks
        self._train_uncertainty_networks(features, prediction_error, confidence)
        
        # Store prediction history
        self.prediction_history.append({
            'prediction': prediction,
            'outcome': actual_outcome,
            'uncertainty': uncertainty_estimate,
            'error': prediction_error
        })
    
    def _train_uncertainty_networks(self, features: torch.Tensor, error: float, confidence: float):
        
        self.optimizer.zero_grad()
        
        # Forward pass
        ensemble_mean, epistemic, aleatoric = self.forward(features)
        
        # Target uncertainties based on actual error and confidence
        target_epistemic = torch.tensor(error * (1.0 - confidence), dtype=torch.float64)
        target_aleatoric = torch.tensor(error * confidence, dtype=torch.float64)
        
        # Uncertainty losses
        epistemic_loss = F.mse_loss(epistemic, target_epistemic.unsqueeze(0))
        aleatoric_loss = F.mse_loss(aleatoric, target_aleatoric.unsqueeze(0))
        
        # Ensemble diversity loss (encourage disagreement when uncertain)
        ensemble_preds = torch.stack([net(features) for net in self.ensemble_nets])
        ensemble_diversity = -torch.std(ensemble_preds)  # Negative to encourage diversity
        diversity_weight = error  # More diversity when errors are high
        
        total_loss = epistemic_loss + aleatoric_loss + diversity_weight * ensemble_diversity
        
        total_loss.backward()
        self.optimizer.step()
        
        logger.debug(f"Uncertainty training: epistemic_loss={epistemic_loss:.4f}, "
                    f"aleatoric_loss={aleatoric_loss:.4f}, diversity={ensemble_diversity:.4f}")
    
    def get_confidence_interval(self, features: torch.Tensor, confidence_level: float = 0.95) -> Tuple[float, float]:
        
        with torch.no_grad():
            # Get ensemble predictions
            ensemble_preds = []
            for net in self.ensemble_nets:
                pred = net(features)
                ensemble_preds.append(float(pred))
            
            # Calculate confidence interval
            ensemble_mean = np.mean(ensemble_preds)
            ensemble_std = np.std(ensemble_preds)
            
            # Add uncertainty estimates
            _, epistemic, aleatoric = self.forward(features)
            total_uncertainty = float(epistemic + aleatoric)
            
            # Expand interval based on uncertainty
            interval_width = ensemble_std + total_uncertainty
            z_score = 1.96 if confidence_level == 0.95 else 2.58  # 95% or 99%
            
            lower_bound = ensemble_mean - z_score * interval_width
            upper_bound = ensemble_mean + z_score * interval_width
            
            return lower_bound, upper_bound
    
    def calibrate_uncertainty(self) -> Dict[str, float]:
        
        if len(self.uncertainty_calibration) < 10:
            return {'calibration_error': 1.0, 'reliability': 0.0}
        
        # Calculate calibration metrics
        calibration_data = list(self.uncertainty_calibration)
        
        predicted_uncertainties = [d['predicted_uncertainty'] for d in calibration_data]
        actual_errors = [d['actual_error'] for d in calibration_data]
        
        # Expected Calibration Error (ECE)
        ece = self._calculate_ece(predicted_uncertainties, actual_errors)
        
        # Reliability (correlation between predicted uncertainty and actual error)
        correlation = np.corrcoef(predicted_uncertainties, actual_errors)[0, 1]
        reliability = correlation if not np.isnan(correlation) else 0.0
        
        # Sharpness (average predicted uncertainty - lower is better)
        sharpness = np.mean(predicted_uncertainties)
        
        return {
            'calibration_error': ece,
            'reliability': reliability,
            'sharpness': sharpness,
            'sample_count': len(calibration_data)
        }
    
    def _calculate_ece(self, predicted_uncertainties: List[float], 
                      actual_errors: List[float], num_bins: int = 10) -> float:
        
        # Bin predictions by uncertainty level
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        total_samples = len(predicted_uncertainties)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this bin
            in_bin = [(pred >= bin_lower) and (pred < bin_upper) 
                     for pred in predicted_uncertainties]
            
            prop_in_bin = sum(in_bin) / total_samples
            
            if prop_in_bin > 0:
                # Average uncertainty and error in this bin
                bin_uncertainties = [predicted_uncertainties[i] for i, x in enumerate(in_bin) if x]
                bin_errors = [actual_errors[i] for i, x in enumerate(in_bin) if x]
                
                avg_uncertainty = np.mean(bin_uncertainties)
                avg_error = np.mean(bin_errors)
                
                # Add to ECE
                ece += prop_in_bin * abs(avg_uncertainty - avg_error)
        
        return ece
    
    def get_uncertainty_stats(self) -> Dict[str, float]:
        
        if not self.prediction_history:
            return {}
        
        recent_history = list(self.prediction_history)[-50:]
        
        uncertainties = [h['uncertainty'] for h in recent_history]
        errors = [h['error'] for h in recent_history]
        
        # Calculate various uncertainty statistics
        stats = {
            'mean_uncertainty': np.mean(uncertainties),
            'std_uncertainty': np.std(uncertainties),
            'mean_error': np.mean(errors),
            'uncertainty_error_correlation': np.corrcoef(uncertainties, errors)[0, 1] if len(uncertainties) > 1 else 0.0,
            'high_uncertainty_ratio': sum(1 for u in uncertainties if u > 0.7) / len(uncertainties),
            'low_uncertainty_ratio': sum(1 for u in uncertainties if u < 0.3) / len(uncertainties)
        }
        
        # Handle NaN correlations
        if np.isnan(stats['uncertainty_error_correlation']):
            stats['uncertainty_error_correlation'] = 0.0
        
        return stats