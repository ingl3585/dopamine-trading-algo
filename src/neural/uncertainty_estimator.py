# uncertainty_estimator.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyEstimate:
    """Enhanced uncertainty estimation result"""
    mean_prediction: torch.Tensor
    total_uncertainty: torch.Tensor
    aleatoric_uncertainty: torch.Tensor
    epistemic_uncertainty: torch.Tensor
    confidence_interval: Tuple[torch.Tensor, torch.Tensor]
    ensemble_diversity: float
    calibration_score: float
    
    def get_confidence(self) -> torch.Tensor:
        """Get confidence score (1 - normalized uncertainty)"""
        normalized_uncertainty = torch.sigmoid(self.total_uncertainty)
        return 1.0 - normalized_uncertainty


class MonteCarloDropoutEstimator(nn.Module):
    """Monte Carlo Dropout for uncertainty estimation"""
    
    def __init__(self, model: nn.Module, dropout_rate: float = 0.1, n_samples: int = 50):
        super().__init__()
        self.model = model
        self.dropout_rate = dropout_rate
        self.n_samples = n_samples
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with Monte Carlo dropout"""
        self.model.train()  # Enable dropout
        predictions = []
        
        for _ in range(self.n_samples):
            with torch.no_grad():
                pred = self.model(x)
                if isinstance(pred, dict):
                    pred = pred.get('action_logits', list(pred.values())[0])
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean_prediction = torch.mean(predictions, dim=0)
        uncertainty = torch.var(predictions, dim=0)
        
        return mean_prediction, uncertainty


class DeepEnsemble(nn.Module):
    """Deep ensemble for uncertainty estimation"""
    
    def __init__(self, models: List[nn.Module], ensemble_size: int = 5):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.models = nn.ModuleList(models[:ensemble_size])
        
        if self.models:
            device = next(self.models[0].parameters()).device
            for model in self.models:
                model.to(device)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through ensemble"""
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                if isinstance(pred, dict):
                    pred = pred.get('action_logits', list(pred.values())[0])
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean_prediction = torch.mean(predictions, dim=0)
        variance = torch.var(predictions, dim=0)
        
        std = torch.sqrt(variance)
        lower_bound = mean_prediction - 1.96 * std
        upper_bound = mean_prediction + 1.96 * std
        
        return {
            'mean_prediction': mean_prediction,
            'variance': variance,
            'uncertainty': torch.mean(variance, dim=-1, keepdim=True),
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'ensemble_diversity': self._calculate_diversity(predictions)
        }
    
    def _calculate_diversity(self, predictions: torch.Tensor) -> float:
        """Calculate diversity metric for ensemble"""
        ensemble_size = predictions.shape[0]
        total_distance = 0.0
        count = 0
        
        for i in range(ensemble_size):
            for j in range(i + 1, ensemble_size):
                distance = torch.mean((predictions[i] - predictions[j]) ** 2)
                total_distance += distance.item()
                count += 1
        
        return total_distance / count if count > 0 else 0.0

class UncertaintyEstimator(nn.Module):
    def __init__(self, input_dim: int = 64, hidden_dim: int = 32, 
                 ensemble_models: Optional[List[nn.Module]] = None):
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
        
        # Enhanced ensemble for uncertainty estimation
        self.ensemble_size = 5
        if ensemble_models and len(ensemble_models) > 0:
            # Use provided models for deep ensemble
            self.deep_ensemble = DeepEnsemble(ensemble_models, self.ensemble_size)
            self.mc_dropout = MonteCarloDropoutEstimator(ensemble_models[0], n_samples=50)
        else:
            # Create simple ensemble as fallback
            self.ensemble_nets = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1)
                ) for _ in range(self.ensemble_size)
            ])
            self.deep_ensemble = None
            self.mc_dropout = None
        
        # Convert to double precision
        self.double()
        
        # Uncertainty tracking
        self.prediction_history = deque(maxlen=500)
        self.uncertainty_calibration = deque(maxlen=100)
        
        # Enhanced uncertainty statistics
        self.uncertainty_stats = {
            'mean_uncertainty': 0.0,
            'uncertainty_trend': 0.0,
            'calibration_error': 0.0,
            'ensemble_diversity': 0.0
        }
        
        # Optimization
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Epistemic uncertainty (what the model doesn't know)
        epistemic_uncertainty = self.epistemic_net(x)
        
        # Aleatoric uncertainty (inherent data noise)
        aleatoric_uncertainty = self.aleatoric_net(x)
        
        # Ensemble predictions for uncertainty quantification
        if hasattr(self, 'ensemble_nets'):
            ensemble_preds = []
            for net in self.ensemble_nets:
                pred = net(x)
                ensemble_preds.append(pred)
            
            ensemble_preds = torch.stack(ensemble_preds)
            ensemble_mean = torch.mean(ensemble_preds, dim=0)
            ensemble_std = torch.std(ensemble_preds, dim=0)
        else:
            # Use deep ensemble if available
            ensemble_mean = torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)
            ensemble_std = torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)
        
        return ensemble_mean, epistemic_uncertainty, aleatoric_uncertainty
    
    def estimate_uncertainty_enhanced(self, x: torch.Tensor, method: str = 'ensemble') -> UncertaintyEstimate:
        """Enhanced uncertainty estimation with multiple methods"""
        
        if method == 'ensemble' and self.deep_ensemble is not None:
            return self._ensemble_uncertainty(x)
        elif method == 'mc_dropout' and self.mc_dropout is not None:
            return self._mc_dropout_uncertainty(x)
        elif method == 'combined' and self.deep_ensemble is not None and self.mc_dropout is not None:
            return self._combined_uncertainty(x)
        else:
            # Fallback to original method
            return self._basic_uncertainty(x)
    
    def _ensemble_uncertainty(self, x: torch.Tensor) -> UncertaintyEstimate:
        """Estimate uncertainty using deep ensemble"""
        ensemble_results = self.deep_ensemble(x)
        
        # Get basic uncertainties
        _, epistemic_basic, aleatoric_basic = self.forward(x)
        
        # Combine ensemble variance with basic uncertainties
        epistemic_uncertainty = ensemble_results['variance'] + epistemic_basic
        aleatoric_uncertainty = aleatoric_basic
        
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        return UncertaintyEstimate(
            mean_prediction=ensemble_results['mean_prediction'],
            total_uncertainty=torch.mean(total_uncertainty, dim=-1, keepdim=True),
            aleatoric_uncertainty=torch.mean(aleatoric_uncertainty, dim=-1, keepdim=True),
            epistemic_uncertainty=torch.mean(epistemic_uncertainty, dim=-1, keepdim=True),
            confidence_interval=(ensemble_results['lower_bound'], ensemble_results['upper_bound']),
            ensemble_diversity=ensemble_results['ensemble_diversity'],
            calibration_score=self._calculate_calibration_score()
        )
    
    def _mc_dropout_uncertainty(self, x: torch.Tensor) -> UncertaintyEstimate:
        """Estimate uncertainty using Monte Carlo dropout"""
        mean_pred, uncertainty = self.mc_dropout(x)
        
        # Get basic uncertainties
        _, epistemic_basic, aleatoric_basic = self.forward(x)
        
        # Combine MC dropout uncertainty with basic uncertainties
        epistemic_uncertainty = uncertainty + epistemic_basic
        aleatoric_uncertainty = aleatoric_basic
        
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        # Approximate confidence intervals
        std = torch.sqrt(uncertainty)
        lower_bound = mean_pred - 1.96 * std
        upper_bound = mean_pred + 1.96 * std
        
        return UncertaintyEstimate(
            mean_prediction=mean_pred,
            total_uncertainty=torch.mean(total_uncertainty, dim=-1, keepdim=True),
            aleatoric_uncertainty=torch.mean(aleatoric_uncertainty, dim=-1, keepdim=True),
            epistemic_uncertainty=torch.mean(epistemic_uncertainty, dim=-1, keepdim=True),
            confidence_interval=(lower_bound, upper_bound),
            ensemble_diversity=0.0,
            calibration_score=self._calculate_calibration_score()
        )
    
    def _combined_uncertainty(self, x: torch.Tensor) -> UncertaintyEstimate:
        """Estimate uncertainty using combined methods"""
        ensemble_est = self._ensemble_uncertainty(x)
        mc_est = self._mc_dropout_uncertainty(x)
        
        # Combine predictions (weighted average)
        combined_prediction = 0.7 * ensemble_est.mean_prediction + 0.3 * mc_est.mean_prediction
        
        # Take maximum uncertainty for conservative estimate
        combined_uncertainty = torch.max(ensemble_est.total_uncertainty, mc_est.total_uncertainty)
        
        return UncertaintyEstimate(
            mean_prediction=combined_prediction,
            total_uncertainty=combined_uncertainty,
            aleatoric_uncertainty=ensemble_est.aleatoric_uncertainty,
            epistemic_uncertainty=torch.max(
                ensemble_est.epistemic_uncertainty, 
                mc_est.epistemic_uncertainty
            ),
            confidence_interval=ensemble_est.confidence_interval,
            ensemble_diversity=ensemble_est.ensemble_diversity,
            calibration_score=self._calculate_calibration_score()
        )
    
    def _basic_uncertainty(self, x: torch.Tensor) -> UncertaintyEstimate:
        """Basic uncertainty estimation (fallback)"""
        ensemble_mean, epistemic, aleatoric = self.forward(x)
        
        total_uncertainty = epistemic + aleatoric
        
        # Simple confidence intervals
        std = torch.sqrt(total_uncertainty)
        lower_bound = ensemble_mean - 1.96 * std
        upper_bound = ensemble_mean + 1.96 * std
        
        return UncertaintyEstimate(
            mean_prediction=ensemble_mean,
            total_uncertainty=total_uncertainty,
            aleatoric_uncertainty=aleatoric,
            epistemic_uncertainty=epistemic,
            confidence_interval=(lower_bound, upper_bound),
            ensemble_diversity=0.0,
            calibration_score=self._calculate_calibration_score()
        )
    
    def _calculate_calibration_score(self) -> float:
        """Calculate calibration score"""
        if len(self.uncertainty_calibration) < 50:
            return 0.5
        
        return max(0.0, 1.0 - self.uncertainty_stats['calibration_error'])
    
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
    
    def get_enhanced_uncertainty_stats(self) -> Dict[str, Any]:
        """Get enhanced uncertainty statistics"""
        base_stats = self.get_uncertainty_stats()
        
        enhanced_stats = {
            **base_stats,
            'uncertainty_methods': [],
            'ensemble_diversity': self.uncertainty_stats['ensemble_diversity'],
            'calibration_score': self._calculate_calibration_score()
        }
        
        # Add available methods
        if self.deep_ensemble is not None:
            enhanced_stats['uncertainty_methods'].append('deep_ensemble')
        if self.mc_dropout is not None:
            enhanced_stats['uncertainty_methods'].append('mc_dropout')
        if hasattr(self, 'ensemble_nets'):
            enhanced_stats['uncertainty_methods'].append('basic_ensemble')
        
        return enhanced_stats


def create_enhanced_uncertainty_estimator(
    input_dim: int = 64,
    hidden_dim: int = 32,
    ensemble_models: Optional[List[nn.Module]] = None
) -> UncertaintyEstimator:
    """Factory function to create enhanced uncertainty estimator"""
    return UncertaintyEstimator(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        ensemble_models=ensemble_models
    )