"""
Neural Network Manager - Centralized neural network operations

This module handles all neural network related operations:
1. Network initialization and configuration
2. Training and optimization
3. Architecture evolution
4. Forward pass orchestration  
5. Model persistence and loading
6. Performance tracking

Extracted from TradingAgent to separate concerns and improve maintainability.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import deque

from src.neural.adaptive_network import AdaptiveTradingNetwork, FeatureLearner, StateEncoder
from src.neural.enhanced_neural import (
    SelfEvolvingNetwork, FewShotLearner, ActorCriticLoss, 
    TradingOptimizer, create_enhanced_network
)
from src.neural.neural_architecture_search import (
    NeuralArchitectureSearch, PerformanceTracker, ArchitectureConfig,
    PerformanceMetrics, GradualTransitionManager
)
from src.neural.uncertainty_estimator import (
    UncertaintyEstimator, create_enhanced_uncertainty_estimator,
    UncertaintyEstimate
)
from src.agent.meta_learner import MetaLearner

logger = logging.getLogger(__name__)


class NetworkConfiguration:
    """Configuration for neural networks"""
    
    def __init__(self):
        self.input_size = 64
        self.feature_dim = 64
        self.enable_few_shot = True
        self.memory_efficient = True
        self.max_memory_gb = 2.0
        self.learning_rate = 0.001
        self.weight_decay = 1e-5
        

class NeuralNetworkManager:
    """
    Manages all neural network operations for the trading system.
    
    This class centralizes network creation, training, evolution, and inference,
    providing a clean interface for the trading agent while encapsulating
    all PyTorch-specific complexity.
    """
    
    def __init__(self, 
                 meta_learner: MetaLearner,
                 device: torch.device,
                 config: Optional[NetworkConfiguration] = None):
        """
        Initialize the neural network manager
        
        Args:
            meta_learner: Meta-learning component for architecture decisions
            device: PyTorch device for computations
            config: Network configuration (uses defaults if None)
        """
        self.meta_learner = meta_learner
        self.device = device
        self.config = config or NetworkConfiguration()
        
        # Initialize networks
        self._initialize_networks()
        
        # Initialize training components
        self._initialize_training_components()
        
        # Performance tracking
        self.ensemble_predictions = deque(maxlen=10)
        self.evolution_stats = {'generations': 0, 'improvements': 0}
        
        # Neural Architecture Search
        self.nas_system = NeuralArchitectureSearch(input_size=self.config.input_size)
        self.performance_tracker = PerformanceTracker()
        self.transition_manager = GradualTransitionManager()
        
        # Performance metrics tracking
        self.performance_history = deque(maxlen=200)
        self.training_times = deque(maxlen=100)
        self.memory_usage_history = deque(maxlen=100)
        
        # Enhanced uncertainty estimator
        self.enhanced_uncertainty_estimator = None
        
        # State tracking
        self.training_steps = 0
        self.last_evolution_step = 0
        
        logger.info("NeuralNetworkManager initialized")
    
    def _initialize_networks(self):
        """Initialize all neural networks"""
        try:
            # Get initial architecture sizes from meta-learner
            initial_sizes = self.meta_learner.architecture_evolver.current_sizes
            
            # Main enhanced network
            self.main_network = create_enhanced_network(
                input_size=self.config.input_size,
                initial_sizes=initial_sizes,
                enable_few_shot=self.config.enable_few_shot,
                memory_efficient=self.config.memory_efficient,
                max_memory_gb=self.config.max_memory_gb
            ).to(self.device)
            
            # Target network for stable training
            self.target_network = create_enhanced_network(
                input_size=self.config.input_size,
                initial_sizes=initial_sizes,
                enable_few_shot=False,  # Target network doesn't need few-shot
                memory_efficient=self.config.memory_efficient,
                max_memory_gb=1.0
            ).to(self.device)
            
            # Feature learning network
            self.feature_learner = FeatureLearner(
                raw_feature_dim=100,  # Match state creation
                learned_feature_dim=self.config.feature_dim,
            ).to(self.device)
            
            # Few-shot learning capability
            self.few_shot_learner = FewShotLearner(
                feature_dim=self.config.feature_dim
            ).to(self.device)
            
            # State encoder for creating enhanced states
            self.state_encoder = StateEncoder()
            
            # Initialize enhanced uncertainty estimator with ensemble models
            ensemble_models = [self.main_network, self.target_network]
            self.enhanced_uncertainty_estimator = create_enhanced_uncertainty_estimator(
                input_dim=self.config.input_size,
                hidden_dim=64,
                ensemble_models=ensemble_models
            ).to(self.device)
            
            logger.info(f"Networks initialized with sizes: {initial_sizes}")
            
        except Exception as e:
            logger.error(f"Error initializing networks: {e}")
            raise
    
    def _initialize_training_components(self):
        """Initialize training and optimization components"""
        try:
            # Enhanced loss function
            self.loss_function = ActorCriticLoss(
                confidence_weight=0.3,
                position_weight=0.2,
                risk_weight=0.1,
                entropy_weight=0.01
            ).to(self.device)
            
            # Advanced multi-component optimizer
            self.optimizer = TradingOptimizer(
                networks=[self.main_network, self.feature_learner, self.few_shot_learner],
                base_lr=self.config.learning_rate
            )
            
            # Legacy optimizer for compatibility
            self.unified_optimizer = optim.AdamW(
                list(self.main_network.parameters()) + 
                list(self.feature_learner.parameters()) + 
                list(self.few_shot_learner.parameters()) +
                list(self.meta_learner.subsystem_weights.parameters()) +
                list(self.meta_learner.exploration_strategy.parameters()),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            
            # Learning rate scheduler
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.unified_optimizer, mode='max', factor=0.8, patience=50
            )
            
            # Importance weights for catastrophic forgetting prevention
            self.importance_weights = {}
            
            logger.info("Training components initialized")
            
        except Exception as e:
            logger.error(f"Error initializing training components: {e}")
            raise
    
    def forward_pass(self, learned_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Perform forward pass through networks
        
        Args:
            learned_state: Processed feature state
            
        Returns:
            Dictionary of network outputs
        """
        try:
            with torch.no_grad():
                # Main network prediction
                main_outputs = self.main_network(learned_state)
                
                # Target network prediction for ensemble
                target_outputs = self.target_network(learned_state)
                
                # Few-shot prediction
                few_shot_prediction = self.few_shot_learner(learned_state)
                
                # Combine ensemble predictions
                ensemble_outputs = [main_outputs, target_outputs]
                combined_outputs = {}
                
                for key in main_outputs.keys():
                    combined_outputs[key] = torch.mean(
                        torch.stack([out[key] for out in ensemble_outputs]), 
                        dim=0
                    )
                
                # Add few-shot prediction
                combined_outputs['few_shot_prediction'] = few_shot_prediction
                
                # Store for ensemble uncertainty calculation
                self.ensemble_predictions.append({
                    'main': main_outputs,
                    'target': target_outputs,
                    'timestamp': torch.tensor(self.training_steps)
                })
                
                return combined_outputs
                
        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            # Return safe defaults
            return {
                'action_logits': torch.zeros(1, 3, device=self.device),
                'confidence': torch.tensor([[0.3]], device=self.device),
                'position_size': torch.tensor([[1.0]], device=self.device),
                'risk_params': torch.zeros(1, 4, device=self.device),
                'few_shot_prediction': torch.tensor([[0.0]], device=self.device)
            }
    
    def process_features(self, raw_state: torch.Tensor) -> torch.Tensor:
        """
        Process raw features through feature learner
        
        Args:
            raw_state: Raw feature state
            
        Returns:
            Learned feature representation
        """
        try:
            return self.feature_learner(raw_state.unsqueeze(0).to(dtype=torch.float32, device=self.device))
        except Exception as e:
            logger.error(f"Error processing features: {e}")
            # Return zero features as fallback
            return torch.zeros(1, self.config.feature_dim, device=self.device)
    
    def train_networks(self, experience_batch: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Train networks on experience batch
        
        Args:
            experience_batch: List of experience dictionaries
            
        Returns:
            Training loss breakdown
        """
        try:
            if len(experience_batch) < 16:
                logger.warning(f"Insufficient batch size: {len(experience_batch)}")
                return {'total_loss': 0.0}
            
            # Track training time
            import time
            start_time = time.time()
            
            # Prepare tensors
            states = torch.tensor(
                [exp['state_features'] for exp in experience_batch],
                dtype=torch.float64, device=self.device
            )
            actions = torch.tensor(
                [exp['action'] for exp in experience_batch],
                dtype=torch.long, device=self.device
            )
            rewards = torch.tensor(
                [exp['reward'] for exp in experience_batch],
                dtype=torch.float64, device=self.device
            )
            uncertainties = torch.tensor(
                [exp.get('uncertainty', 0.5) for exp in experience_batch],
                dtype=torch.float64, device=self.device
            )
            
            # Enhanced feature learning
            learned_features = self.feature_learner(states)
            
            # Forward pass through main network
            outputs = self.main_network(learned_features)
            
            # Calculate losses
            loss_breakdown = self._calculate_losses(
                outputs, actions, rewards, uncertainties, learned_features
            )
            
            # Backward pass with gradient clipping
            self.unified_optimizer.zero_grad()
            loss_breakdown['total_loss'].backward()
            
            # Update importance weights for catastrophic forgetting prevention
            self._update_importance_weights()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(self.main_network.parameters()) +
                list(self.feature_learner.parameters()) +
                list(self.few_shot_learner.parameters()),
                1.0
            )
            
            self.unified_optimizer.step()
            self.training_steps += 1
            
            # Track performance metrics
            training_time = time.time() - start_time
            self.training_times.append(training_time)
            
            # Track performance (inverse of loss for simplicity)
            performance = 1.0 / (1.0 + float(loss_breakdown['total_loss']))
            self.performance_history.append(performance)
            
            # Track memory usage
            memory_usage = self._get_current_memory_usage()
            self.memory_usage_history.append(memory_usage)
            
            # Update target network periodically
            if self.training_steps % 200 == 0:
                self.target_network.load_state_dict(self.main_network.state_dict())
                logger.debug("Target network updated")
            
            return {k: float(v) for k, v in loss_breakdown.items()}
            
        except Exception as e:
            logger.error(f"Error training networks: {e}")
            return {'total_loss': float('inf'), 'error': str(e)}
    
    def _calculate_losses(self,
                         outputs: Dict[str, torch.Tensor],
                         actions: torch.Tensor,
                         rewards: torch.Tensor,
                         uncertainties: torch.Tensor,
                         learned_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calculate comprehensive loss breakdown"""
        
        # Policy loss with uncertainty weighting
        action_logits = outputs['action_logits']
        action_probs = F.log_softmax(action_logits, dim=-1)
        selected_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Weight by inverse uncertainty
        uncertainty_weights = 1.0 / (uncertainties + 0.1)
        policy_loss = -(selected_probs * rewards * uncertainty_weights).mean()
        
        # Value losses with uncertainty consideration
        confidence_target = torch.abs(rewards).unsqueeze(1) * uncertainty_weights.unsqueeze(1)
        confidence_loss = F.mse_loss(outputs['confidence'], confidence_target)
        
        # Position size loss
        size_target = torch.clamp(torch.abs(rewards) * 2.0 * uncertainty_weights, 0.5, 3.0).unsqueeze(1)
        size_loss = F.mse_loss(outputs['position_size'], size_target)
        
        # Risk parameter loss
        risk_target = torch.sigmoid(rewards.unsqueeze(1).expand(-1, 4))
        risk_loss = F.mse_loss(outputs['risk_params'], risk_target)
        
        # Few-shot learning loss
        few_shot_predictions = self.few_shot_learner(learned_features)
        few_shot_loss = F.mse_loss(few_shot_predictions.squeeze(), rewards.unsqueeze(1))
        
        # Regularization for catastrophic forgetting prevention
        regularization_loss = torch.tensor(0.0, device=self.device)
        if self.importance_weights:
            for name, param in self.main_network.named_parameters():
                if name in self.importance_weights:
                    regularization_loss += torch.sum(self.importance_weights[name] * (param ** 2))
        
        # Total loss with adaptive weighting
        total_loss = (
            policy_loss +
            0.1 * confidence_loss +
            0.05 * size_loss +
            0.03 * risk_loss +
            0.02 * few_shot_loss +
            0.001 * regularization_loss
        )
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'confidence_loss': confidence_loss,
            'size_loss': size_loss,
            'risk_loss': risk_loss,
            'few_shot_loss': few_shot_loss,
            'regularization_loss': regularization_loss
        }
    
    def _update_importance_weights(self):
        """Update importance weights for catastrophic forgetting prevention"""
        for name, param in self.main_network.named_parameters():
            if param.grad is not None:
                # Fisher Information approximation
                importance = param.grad.data.clone().pow(2)
                
                if name in self.importance_weights:
                    # Exponential moving average of importance
                    self.importance_weights[name] = (
                        0.9 * self.importance_weights[name] + 0.1 * importance
                    )
                else:
                    self.importance_weights[name] = importance
    
    def should_evolve_architecture(self) -> bool:
        """Check if architecture should evolve using NAS system"""
        try:
            # Get current performance
            current_performance = self._calculate_current_performance()
            
            # Update NAS system with current performance
            self.performance_tracker.record_performance(current_performance)
            
            # Check if evolution is needed
            return self.performance_tracker.should_evolve()
        except Exception as e:
            logger.error(f"Error checking evolution criteria: {e}")
            return False
    
    def evolve_architecture(self) -> bool:
        """
        Evolve network architecture using NAS system
        
        Returns:
            True if evolution was successful
        """
        try:
            if self.training_steps - self.last_evolution_step < 1000:
                logger.debug("Too soon for architecture evolution")
                return False
            
            # Get current performance
            current_performance = self._calculate_current_performance()
            
            # Get new architecture from NAS system
            new_architecture = self.nas_system.suggest_mutation(current_performance)
            current_sizes = self.meta_learner.architecture_evolver.current_sizes
            
            if new_architecture.layer_sizes == current_sizes:
                logger.debug("No architecture change needed")
                return False
            
            logger.info(f"Evolving architecture from {current_sizes} to {new_architecture.layer_sizes}")
            
            # Start gradual transition
            old_config = ArchitectureConfig(layer_sizes=current_sizes)
            self.transition_manager.start_transition(old_config, new_architecture)
            
            # Apply architecture evolution
            self._apply_architecture_evolution(new_architecture)
            
            # Update tracking
            self.evolution_stats['generations'] += 1
            self.last_evolution_step = self.training_steps
            
            logger.info(f"Architecture evolved successfully. Generation: {self.evolution_stats['generations']}")
            return True
            
        except Exception as e:
            logger.error(f"Error evolving architecture: {e}")
            return False
    
    def _reinitialize_optimizer(self):
        """Reinitialize optimizer after architecture changes"""
        try:
            current_lr = self.unified_optimizer.param_groups[0]['lr']
            
            self.unified_optimizer = optim.AdamW(
                list(self.main_network.parameters()) +
                list(self.feature_learner.parameters()) +
                list(self.few_shot_learner.parameters()) +
                list(self.meta_learner.subsystem_weights.parameters()) +
                list(self.meta_learner.exploration_strategy.parameters()),
                lr=current_lr,
                weight_decay=self.config.weight_decay
            )
            
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.unified_optimizer, mode='max', factor=0.8, patience=50
            )
            
            logger.debug("Optimizer reinitialized after architecture evolution")
            
        except Exception as e:
            logger.error(f"Error reinitializing optimizer: {e}")
    
    def update_learning_rate(self, performance_metric: float):
        """Update learning rate based on performance"""
        try:
            self.scheduler.step(performance_metric)
        except Exception as e:
            logger.error(f"Error updating learning rate: {e}")
    
    def record_performance(self, reward: float):
        """Record network performance for evolution decisions"""
        try:
            self.main_network.record_performance(reward)
        except Exception as e:
            logger.error(f"Error recording performance: {e}")
    
    def calculate_ensemble_uncertainty(self, features: Optional[torch.Tensor] = None) -> float:
        """Calculate uncertainty from ensemble predictions"""
        try:
            # Use enhanced uncertainty estimator if available and features provided
            if self.enhanced_uncertainty_estimator is not None and features is not None:
                return self._calculate_enhanced_uncertainty(features)
            
            # Fallback to original ensemble uncertainty calculation
            if len(self.ensemble_predictions) < 2:
                return 0.5
            
            # Get recent predictions
            recent_predictions = list(self.ensemble_predictions)[-5:]
            
            # Calculate variance in action predictions
            action_variances = []
            for pred in recent_predictions:
                main_probs = F.softmax(pred['main']['action_logits'], dim=-1)
                target_probs = F.softmax(pred['target']['action_logits'], dim=-1)
                variance = torch.var(torch.stack([main_probs, target_probs]), dim=0)
                action_variances.append(torch.mean(variance))
            
            if action_variances:
                uncertainty = torch.mean(torch.stack(action_variances)).item()
                return min(1.0, uncertainty * 5.0)  # Scale to [0, 1]
            
            return 0.5
            
        except Exception as e:
            logger.error(f"Error calculating ensemble uncertainty: {e}")
            return 0.5
    
    def _calculate_enhanced_uncertainty(self, features: torch.Tensor) -> float:
        """Calculate enhanced uncertainty using multiple methods"""
        try:
            # Use combined uncertainty estimation for best results
            uncertainty_estimate = self.enhanced_uncertainty_estimator.estimate_uncertainty_enhanced(
                features, method='combined'
            )
            
            # Get scalar uncertainty value
            uncertainty_value = float(torch.mean(uncertainty_estimate.total_uncertainty))
            
            # Clamp to [0, 1] range
            return max(0.0, min(1.0, uncertainty_value))
            
        except Exception as e:
            logger.error(f"Error calculating enhanced uncertainty: {e}")
            return 0.5
    
    def get_uncertainty_estimate(self, features: torch.Tensor, method: str = 'combined') -> UncertaintyEstimate:
        """Get detailed uncertainty estimate"""
        try:
            if self.enhanced_uncertainty_estimator is not None:
                return self.enhanced_uncertainty_estimator.estimate_uncertainty_enhanced(
                    features, method=method
                )
            else:
                # Return basic uncertainty estimate
                basic_uncertainty = self.calculate_ensemble_uncertainty(features)
                return UncertaintyEstimate(
                    mean_prediction=torch.tensor([0.0]),
                    total_uncertainty=torch.tensor([basic_uncertainty]),
                    aleatoric_uncertainty=torch.tensor([basic_uncertainty * 0.5]),
                    epistemic_uncertainty=torch.tensor([basic_uncertainty * 0.5]),
                    confidence_interval=(torch.tensor([0.0]), torch.tensor([1.0])),
                    ensemble_diversity=0.0,
                    calibration_score=0.5
                )
        except Exception as e:
            logger.error(f"Error getting uncertainty estimate: {e}")
            return UncertaintyEstimate(
                mean_prediction=torch.tensor([0.0]),
                total_uncertainty=torch.tensor([0.5]),
                aleatoric_uncertainty=torch.tensor([0.25]),
                epistemic_uncertainty=torch.tensor([0.25]),
                confidence_interval=(torch.tensor([0.0]), torch.tensor([1.0])),
                ensemble_diversity=0.0,
                calibration_score=0.5
            )
    
    def save_networks(self, filepath: str):
        """Save all networks to file"""
        try:
            import os
            
            # Ensure directory exists
            dir_path = os.path.dirname(filepath)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            
            checkpoint = {
                'main_network_state': self.main_network.state_dict(),
                'target_network_state': self.target_network.state_dict(),
                'feature_learner_state': self.feature_learner.state_dict(),
                'few_shot_learner_state': self.few_shot_learner.state_dict(),
                'optimizer_state': self.unified_optimizer.state_dict(),
                'scheduler_state': self.scheduler.state_dict(),
                'importance_weights': self.importance_weights,
                'training_steps': self.training_steps,
                'evolution_stats': self.evolution_stats,
                'config': {
                    'input_size': self.config.input_size,
                    'feature_dim': self.config.feature_dim,
                    'learning_rate': self.config.learning_rate,
                    'weight_decay': self.config.weight_decay
                }
            }
            
            torch.save(checkpoint, filepath)
            logger.info(f"Networks saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving networks: {e}")
    
    def load_networks(self, filepath: str) -> bool:
        """
        Load networks from file
        
        Returns:
            True if loading was successful
        """
        try:
            # Handle PyTorch version compatibility
            try:
                checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            except Exception:
                checkpoint = torch.load(filepath, map_location=self.device)
            
            # Load network states with architecture compatibility check
            try:
                self.main_network.load_state_dict(checkpoint['main_network_state'])
                self.target_network.load_state_dict(checkpoint['target_network_state'])
                self.feature_learner.load_state_dict(checkpoint['feature_learner_state'])
                
                if 'few_shot_learner_state' in checkpoint:
                    self.few_shot_learner.load_state_dict(checkpoint['few_shot_learner_state'])
                    
            except Exception as arch_error:
                logger.warning(f"Architecture mismatch detected: {arch_error}")
                logger.warning("Starting with fresh neural networks")
                return False
            
            # Load optimizer and scheduler states
            self.unified_optimizer.load_state_dict(checkpoint['optimizer_state'])
            
            if 'scheduler_state' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            
            # Load importance weights and statistics
            if 'importance_weights' in checkpoint:
                self.importance_weights = checkpoint['importance_weights']
                
            self.training_steps = checkpoint.get('training_steps', 0)
            self.evolution_stats = checkpoint.get('evolution_stats', {'generations': 0, 'improvements': 0})
            
            logger.info(f"Networks loaded successfully from {filepath}")
            return True
            
        except FileNotFoundError:
            logger.info("No existing network checkpoint found, starting fresh")
            return False
        except Exception as e:
            logger.error(f"Error loading networks: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive network statistics"""
        try:
            current_lr = self.unified_optimizer.param_groups[0]['lr']
            
            return {
                'training_steps': self.training_steps,
                'current_learning_rate': current_lr,
                'evolution_stats': self.evolution_stats.copy(),
                'ensemble_size': len(self.ensemble_predictions),
                'importance_weights_count': len(self.importance_weights),
                'network_parameters': {
                    'main_network': sum(p.numel() for p in self.main_network.parameters()),
                    'feature_learner': sum(p.numel() for p in self.feature_learner.parameters()),
                    'few_shot_learner': sum(p.numel() for p in self.few_shot_learner.parameters())
                },
                'device': str(self.device),
                'memory_usage_gb': torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {'error': str(e)}
    
    def get_evolution_stats(self) -> Dict[str, Any]:
        """Get network evolution statistics"""
        nas_stats = self.nas_system.get_stats()
        transition_stats = self.transition_manager.get_transition_stats()
        
        return {
            'generations': self.evolution_stats['generations'],
            'improvements': self.evolution_stats['improvements'],
            'current_architecture': getattr(
                self.meta_learner.architecture_evolver, 'current_sizes', 'unknown'
            ),
            'last_evolution_step': self.last_evolution_step,
            'steps_since_evolution': self.training_steps - self.last_evolution_step,
            'nas_stats': nas_stats,
            'transition_stats': transition_stats,
            'performance_trend': self.performance_tracker.get_performance_trend()
        }
    
    def _calculate_current_performance(self) -> float:
        """Calculate current network performance"""
        try:
            if not self.performance_history:
                return 0.0
            
            # Use recent performance average
            recent_performance = list(self.performance_history)[-10:]
            return sum(recent_performance) / len(recent_performance)
        except Exception as e:
            logger.error(f"Error calculating current performance: {e}")
            return 0.0
    
    def _apply_architecture_evolution(self, new_architecture: ArchitectureConfig):
        """Apply architecture evolution to networks"""
        try:
            # Store current network state
            old_state = self.main_network.state_dict()
            
            # Update meta-learner with new architecture
            self.meta_learner.architecture_evolver.current_sizes = new_architecture.layer_sizes
            
            # Evolve networks
            self.main_network.evolve_architecture(new_architecture.layer_sizes)
            self.target_network.evolve_architecture(new_architecture.layer_sizes)
            
            # Update optimizer with new parameters
            self._reinitialize_optimizer()
            
            # Update performance tracking
            self._update_architecture_performance(new_architecture)
            
        except Exception as e:
            logger.error(f"Error applying architecture evolution: {e}")
            raise
    
    def _update_architecture_performance(self, architecture: ArchitectureConfig):
        """Update performance metrics for evolved architecture"""
        try:
            # Create performance metrics
            metrics = PerformanceMetrics(
                accuracy=self._calculate_current_performance(),
                loss=self._get_recent_loss(),
                training_time=self._get_average_training_time(),
                memory_usage=self._get_current_memory_usage(),
                convergence_speed=self._calculate_convergence_speed(),
                stability=self._calculate_stability()
            )
            
            # Update NAS system
            self.nas_system.update_architecture_performance(architecture, metrics)
            
        except Exception as e:
            logger.error(f"Error updating architecture performance: {e}")
    
    def _get_recent_loss(self) -> float:
        """Get recent training loss"""
        # This would be updated during training
        return 0.5  # Placeholder
    
    def _get_average_training_time(self) -> float:
        """Get average training time"""
        if not self.training_times:
            return 0.0
        return sum(self.training_times) / len(self.training_times)
    
    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
        return 0.0
    
    def _calculate_convergence_speed(self) -> float:
        """Calculate convergence speed metric"""
        if len(self.performance_history) < 20:
            return 0.5
        
        # Calculate how quickly performance improves
        recent = list(self.performance_history)[-20:]
        if len(recent) < 2:
            return 0.5
        
        # Simple gradient of performance
        improvements = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
        avg_improvement = sum(improvements) / len(improvements)
        
        # Normalize to 0-1 range
        return max(0.0, min(1.0, avg_improvement * 10 + 0.5))
    
    def _calculate_stability(self) -> float:
        """Calculate stability metric"""
        if len(self.performance_history) < 10:
            return 0.5
        
        # Calculate variance in recent performance
        recent = list(self.performance_history)[-10:]
        variance = np.var(recent)
        
        # Lower variance = higher stability
        stability = 1.0 / (1.0 + variance * 10)
        return max(0.0, min(1.0, stability))