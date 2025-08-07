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
import time
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
from src.core.state_coordinator import register_state_component
from src.core.unified_serialization import unified_serializer, serialize_component_state, deserialize_component_state
from src.core.storage_error_handler import storage_error_handler, handle_storage_operation
from src.core.data_integrity_validator import data_integrity_validator, validate_component_data, ValidationLevel

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
                 config: Dict[str, Any],
                 nas_system: Optional[Any] = None,
                 uncertainty_estimator: Optional[Any] = None,
                 pruning_manager: Optional[Any] = None,
                 specialized_networks: Optional[Any] = None,
                 meta_learner: Optional[Any] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize the neural network manager
        
        Args:
            config: Configuration dictionary
            nas_system: Neural Architecture Search system
            uncertainty_estimator: Uncertainty estimation component
            pruning_manager: Dynamic pruning manager
            specialized_networks: Specialized network ensemble
            meta_learner: Meta-learning component (optional)
            device: PyTorch device for computations (auto-detected if None)
        """
        # Store injected components
        self.nas_system = nas_system
        self.uncertainty_estimator = uncertainty_estimator
        self.pruning_manager = pruning_manager
        self.specialized_networks = specialized_networks
        self.meta_learner = meta_learner
        
        # Handle device
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        
        # Handle config
        if isinstance(config, dict):
            # Extract network configuration from dict
            network_config = NetworkConfiguration()
            network_config.input_size = config.get('input_size', 64)
            network_config.feature_dim = config.get('feature_dim', 64)
            network_config.learning_rate = config.get('learning_rate', 0.001)
            self.config = network_config
        else:
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
        self.last_save_time = time.time()
        self.save_interval = 300  # 5 minutes
        self.last_performance_time = time.time()
        self.force_save_counter = 0  # Counter to force saves even without performance data
        self.save_triggers = {
            'periodic': True,
            'performance_improvement': True,
            'training_milestones': True,
            'error_recovery': True,
            'force_periodic': True  # New trigger for guaranteed periodic saves
        }
        
        # Start background timer for independent model saving
        self._start_save_timer()
        
        # Register with state coordinator for unified state management
        self._register_state_management()
        
        # Load existing state if available
        self._load_persisted_state()
        
        logger.info("NeuralNetworkManager initialized with unified state management and periodic model saving")
    
    def _start_save_timer(self):
        """Start background timer for independent periodic model saving"""
        try:
            import threading
            
            def save_timer_worker():
                """Background worker that saves models every 5 minutes regardless of other conditions"""
                while True:
                    try:
                        time.sleep(self.save_interval)  # Wait 5 minutes
                        
                        # Force save regardless of experience storage or other conditions
                        if self.save_triggers['force_periodic']:
                            logger.info("Background timer triggered - forcing periodic model save")
                            self._force_save_models()
                            self.force_save_counter += 1
                        
                    except Exception as e:
                        logger.error(f"Error in background save timer: {e}")
                        # Continue the timer even if saving fails
                        continue
            
            # Start the timer thread as daemon so it doesn't prevent shutdown
            save_timer_thread = threading.Thread(target=save_timer_worker, daemon=True)
            save_timer_thread.start()
            
            logger.info(f"Background model save timer started - will save every {self.save_interval} seconds")
            
        except Exception as e:
            logger.error(f"Failed to start background save timer: {e}")
            # Continue without timer - fallback to manual saves
    
    def _initialize_networks(self):
        """Initialize all neural networks"""
        try:
            # Get initial architecture sizes from meta-learner or use defaults
            if self.meta_learner and hasattr(self.meta_learner, 'architecture_evolver'):
                initial_sizes = self.meta_learner.architecture_evolver.current_sizes
            else:
                # Use default architecture sizes
                initial_sizes = [128, 64, 32]
            
            # Main enhanced network
            self.main_network = create_enhanced_network(
                input_size=self.config.input_size,
                initial_sizes=initial_sizes,
                enable_few_shot=self.config.enable_few_shot,
                memory_efficient=self.config.memory_efficient,
                max_memory_gb=self.config.max_memory_gb
            ).to(self.device).float()
            
            # Target network for stable training
            self.target_network = create_enhanced_network(
                input_size=self.config.input_size,
                initial_sizes=initial_sizes,
                enable_few_shot=False,  # Target network doesn't need few-shot
                memory_efficient=self.config.memory_efficient,
                max_memory_gb=1.0
            ).to(self.device).float()
            
            # Feature learning network
            self.feature_learner = FeatureLearner(
                raw_feature_dim=100,  # Match state creation
                learned_feature_dim=self.config.feature_dim,
            ).to(self.device).float()
            
            # Few-shot learning capability
            self.few_shot_learner = FewShotLearner(
                feature_dim=self.config.feature_dim
            ).to(self.device).float()
            
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
            optimizer_params = (
                list(self.main_network.parameters()) + 
                list(self.feature_learner.parameters()) + 
                list(self.few_shot_learner.parameters())
            )
            
            # Add meta-learner parameters if available
            if self.meta_learner:
                if hasattr(self.meta_learner, 'subsystem_weights'):
                    optimizer_params += list(self.meta_learner.subsystem_weights.parameters())
                if hasattr(self.meta_learner, 'exploration_strategy'):
                    optimizer_params += list(self.meta_learner.exploration_strategy.parameters())
            
            self.unified_optimizer = optim.AdamW(
                optimizer_params,
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
                
                # Check for periodic model saves (but not too frequently)
                if self.training_steps % 50 == 0:  # Only check every 50 forward passes
                    save_result = self.check_and_save_models()
                    if save_result:
                        logger.debug(f"Periodic save completed during forward pass at step {self.training_steps}")
                
                return combined_outputs
                
        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            # Return safe defaults
            return {
                'action_logits': torch.zeros(1, 3, dtype=torch.float32, device=self.device),
                'confidence': torch.tensor([[0.3]], dtype=torch.float32, device=self.device),
                'position_size': torch.tensor([[1.0]], dtype=torch.float32, device=self.device),
                'risk_params': torch.zeros(1, 4, dtype=torch.float32, device=self.device),
                'few_shot_prediction': torch.tensor([[0.0]], dtype=torch.float32, device=self.device)
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
            return torch.zeros(1, self.config.feature_dim, dtype=torch.float32, device=self.device)
    
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
                dtype=torch.float32, device=self.device
            )
            actions = torch.tensor(
                [exp['action'] for exp in experience_batch],
                dtype=torch.long, device=self.device
            )
            rewards = torch.tensor(
                [exp['reward'] for exp in experience_batch],
                dtype=torch.float32, device=self.device
            )
            uncertainties = torch.tensor(
                [exp.get('uncertainty', 0.5) for exp in experience_batch],
                dtype=torch.float32, device=self.device
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
            
            # Multiple independent save triggers (decoupled from experience storage)
            save_triggered = False
            
            # 1. Check periodic saves
            if self.check_and_save_models():
                save_triggered = True
            
            # 2. Check training milestone triggers
            if self.save_triggers['training_milestones'] and self.training_steps % 500 == 0:
                logger.info(f"Training milestone reached: {self.training_steps} steps - forcing save")
                if self._force_save_models():
                    save_triggered = True
            
            # 3. Emergency save if no saves in a while
            if not save_triggered and (time.time() - self.last_save_time) > (self.save_interval * 1.5):
                logger.warning(f"Emergency save triggered - no saves for {time.time() - self.last_save_time:.0f} seconds")
                self._force_save_models()
            
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
        regularization_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
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
            if self.nas_system:
                new_architecture = self.nas_system.suggest_mutation(current_performance)
            else:
                logger.warning("NAS system not available, skipping architecture evolution")
                return False
                
            # Get current sizes from meta-learner or use defaults
            if self.meta_learner and hasattr(self.meta_learner, 'architecture_evolver'):
                current_sizes = self.meta_learner.architecture_evolver.current_sizes
            else:
                current_sizes = [128, 64, 32]
            
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
            
            # Build optimizer parameter list
            optimizer_params = (
                list(self.main_network.parameters()) +
                list(self.feature_learner.parameters()) +
                list(self.few_shot_learner.parameters())
            )
            
            # Add meta-learner parameters if available
            if self.meta_learner:
                if hasattr(self.meta_learner, 'subsystem_weights'):
                    optimizer_params += list(self.meta_learner.subsystem_weights.parameters())
                if hasattr(self.meta_learner, 'exploration_strategy'):
                    optimizer_params += list(self.meta_learner.exploration_strategy.parameters())
            
            self.unified_optimizer = optim.AdamW(
                optimizer_params,
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
        """Save all networks to file using PyTorch's native serialization"""
        try:
            import os
            
            # Ensure directory exists
            dir_path = os.path.dirname(filepath)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            
            # Validate that networks are in the correct state
            if not hasattr(self, 'main_network') or self.main_network is None:
                logger.error("Main network not initialized, cannot save")
                return
            
            # Create checkpoint with proper tensor preservation
            checkpoint = {
                'main_network_state': self.main_network.state_dict(),
                'target_network_state': self.target_network.state_dict(),
                'feature_learner_state': self.feature_learner.state_dict(),
                'few_shot_learner_state': self.few_shot_learner.state_dict(),
                'optimizer_state': self.unified_optimizer.state_dict(),
                'scheduler_state': self.scheduler.state_dict(),
                'importance_weights': self.importance_weights.tolist() if hasattr(self.importance_weights, 'tolist') else self.importance_weights,
                'training_steps': int(self.training_steps),
                'evolution_stats': dict(self.evolution_stats),
                'config': {
                    'input_size': int(self.config.input_size),
                    'feature_dim': int(self.config.feature_dim),
                    'learning_rate': float(self.config.learning_rate),
                    'weight_decay': float(self.config.weight_decay)
                },
                'version': '1.0',
                'pytorch_version': torch.__version__
            }
            
            # Save using PyTorch's native serialization (bypasses unified serialization)
            torch.save(checkpoint, filepath)
            
            # Verify the save worked by checking file size
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                logger.info(f"Networks saved to {filepath} ({file_size:,} bytes)")
            else:
                logger.error(f"Save verification failed - file {filepath} not found")
            
        except Exception as e:
            logger.error(f"Error saving networks: {e}")
            import traceback
            logger.error(f"Save traceback: {traceback.format_exc()}")
    
    def load_networks(self, filepath: str) -> bool:
        """
        Load networks from file with enhanced corruption detection and recovery
        
        Returns:
            True if loading was successful
        """
        try:
            logger.info(f"Loading neural networks from {filepath}")
            
            # Handle PyTorch version compatibility
            try:
                checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            except Exception as load_error:
                logger.warning(f"Failed to load with weights_only=False, trying compatibility mode: {load_error}")
                try:
                    checkpoint = torch.load(filepath, map_location=self.device)
                except Exception as compat_error:
                    logger.error(f"Failed to load checkpoint in compatibility mode: {compat_error}")
                    return False
            
            # Validate checkpoint structure
            if not isinstance(checkpoint, dict):
                logger.error(f"Invalid checkpoint format: expected dict, got {type(checkpoint)}")
                return False
            
            # Check for required network states
            required_states = ['main_network_state', 'target_network_state', 'feature_learner_state']
            missing_states = [state for state in required_states if state not in checkpoint]
            if missing_states:
                logger.error(f"Missing network states in checkpoint: {missing_states}")
                return False
            
            # Validate that network states are proper state_dicts (not corrupted to dicts)
            network_validation_errors = []
            for state_name in required_states:
                state_dict = checkpoint[state_name]
                if not isinstance(state_dict, dict):
                    network_validation_errors.append(f"{state_name}: not a dict ({type(state_dict)})")
                    continue
                
                # Check for tensor corruption: validate that values are tensors, not dicts
                for param_name, param_value in state_dict.items():
                    if isinstance(param_value, dict):
                        network_validation_errors.append(f"{state_name}.{param_name}: corrupted tensor (saved as dict)")
                    elif not torch.is_tensor(param_value):
                        network_validation_errors.append(f"{state_name}.{param_name}: not a tensor ({type(param_value)})")
            
            if network_validation_errors:
                logger.error("Detected corrupted network states:")
                for error in network_validation_errors[:10]:  # Show first 10 errors
                    logger.error(f"  - {error}")
                if len(network_validation_errors) > 10:
                    logger.error(f"  ... and {len(network_validation_errors) - 10} more errors")
                
                # Backup corrupted file for analysis
                try:
                    import shutil
                    import os
                    import time
                    backup_path = filepath + '.corrupted.' + str(int(time.time()))
                    shutil.move(filepath, backup_path)
                    logger.warning(f"Corrupted checkpoint moved to: {backup_path}")
                except Exception as backup_error:
                    logger.warning(f"Failed to backup corrupted checkpoint: {backup_error}")
                
                logger.error("Starting with fresh neural networks due to corruption")
                return False
            
            # Load network states with architecture compatibility check
            try:
                logger.debug("Loading main network state...")
                self.main_network.load_state_dict(checkpoint['main_network_state'])
                
                logger.debug("Loading target network state...")
                self.target_network.load_state_dict(checkpoint['target_network_state'])
                
                logger.debug("Loading feature learner state...")
                self.feature_learner.load_state_dict(checkpoint['feature_learner_state'])
                
                if 'few_shot_learner_state' in checkpoint:
                    logger.debug("Loading few-shot learner state...")
                    self.few_shot_learner.load_state_dict(checkpoint['few_shot_learner_state'])
                
                # Ensure all networks use float32 after loading
                self.main_network.float()
                self.target_network.float()
                self.feature_learner.float()
                if hasattr(self, 'few_shot_learner'):
                    self.few_shot_learner.float()
                    
                logger.debug("Network states loaded successfully")
                    
            except Exception as arch_error:
                logger.error(f"Architecture/compatibility error during network loading: {arch_error}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                logger.warning("Starting with fresh neural networks due to architecture mismatch")
                return False
            
            # Load optimizer and scheduler states with validation
            try:
                if 'optimizer_state' in checkpoint:
                    logger.debug("Loading optimizer state...")
                    optimizer_state = checkpoint['optimizer_state']
                    if isinstance(optimizer_state, dict):
                        self.unified_optimizer.load_state_dict(optimizer_state)
                    else:
                        logger.warning(f"Invalid optimizer state type: {type(optimizer_state)}")
                
                if 'scheduler_state' in checkpoint:
                    logger.debug("Loading scheduler state...")
                    scheduler_state = checkpoint['scheduler_state']
                    if isinstance(scheduler_state, dict):
                        self.scheduler.load_state_dict(scheduler_state)
                    else:
                        logger.warning(f"Invalid scheduler state type: {type(scheduler_state)}")
                        
            except Exception as opt_error:
                logger.warning(f"Failed to load optimizer/scheduler state (continuing with fresh): {opt_error}")
                # Not a critical error - networks are more important than optimizer state
            
            # Load importance weights and statistics with validation
            try:
                if 'importance_weights' in checkpoint:
                    weights = checkpoint['importance_weights']
                    if isinstance(weights, (list, tuple)):
                        self.importance_weights = weights
                    elif torch.is_tensor(weights):
                        self.importance_weights = weights.tolist()
                    else:
                        logger.warning(f"Invalid importance_weights type: {type(weights)}")
                        
                self.training_steps = int(checkpoint.get('training_steps', 0))
                
                evolution_stats = checkpoint.get('evolution_stats', {'generations': 0, 'improvements': 0})
                if isinstance(evolution_stats, dict):
                    self.evolution_stats = evolution_stats
                else:
                    logger.warning(f"Invalid evolution_stats type: {type(evolution_stats)}")
                    self.evolution_stats = {'generations': 0, 'improvements': 0}
                    
            except Exception as stats_error:
                logger.warning(f"Failed to load statistics (using defaults): {stats_error}")
                self.training_steps = 0
                self.evolution_stats = {'generations': 0, 'improvements': 0}
            
            # Validate networks are working after loading
            try:
                logger.debug("Validating loaded networks...")
                test_input = torch.zeros(1, self.config.input_size, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    main_output = self.main_network(test_input)
                    target_output = self.target_network(test_input)
                    feature_output = self.feature_learner(test_input)
                    
                # Validate output tensors
                if not torch.is_tensor(main_output):
                    raise ValueError(f"Main network returned non-tensor: {type(main_output)}")
                if not torch.is_tensor(target_output):  
                    raise ValueError(f"Target network returned non-tensor: {type(target_output)}")
                if not torch.is_tensor(feature_output):
                    raise ValueError(f"Feature learner returned non-tensor: {type(feature_output)}")
                    
                logger.debug("Network validation successful")
                    
            except Exception as validation_error:
                logger.error(f"Loaded networks failed validation: {validation_error}")
                logger.error("Starting with fresh neural networks due to validation failure")
                return False
            
            # Log checkpoint information
            version = checkpoint.get('version', 'unknown')
            pytorch_version = checkpoint.get('pytorch_version', 'unknown')
            logger.info(f"Networks loaded successfully from {filepath}")
            logger.info(f"  - Version: {version}, PyTorch: {pytorch_version}")
            logger.info(f"  - Training steps: {self.training_steps}")
            logger.info(f"  - Evolution stats: {self.evolution_stats}")
            
            return True
            
        except FileNotFoundError:
            logger.info("No existing network checkpoint found, starting fresh")
            return False
        except Exception as e:
            logger.error(f"Unexpected error loading networks: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
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
    
    def check_and_save_models(self) -> bool:
        """
        Check if models should be saved based on timer and save if needed.
        This method provides multiple fallback mechanisms to ensure models are saved.
        """
        try:
            current_time = time.time()
            time_since_last_save = current_time - self.last_save_time
            
            # Primary check: periodic timer
            should_save_periodic = time_since_last_save >= self.save_interval
            
            # Fallback check: if too much time has passed without any saves (emergency save)
            should_save_emergency = time_since_last_save >= (self.save_interval * 2)  # 10 minutes
            
            # Force save trigger
            should_save_force = self.force_save_counter > 0 and (time_since_last_save >= 60)  # At least 1 minute between saves
            
            if should_save_periodic or should_save_emergency or should_save_force:
                # Save models with appropriate messaging
                import os
                models_dir = 'models'
                os.makedirs(models_dir, exist_ok=True)
                model_path = os.path.join(models_dir, 'neural_networks.pt')
                
                # Determine save reason
                save_reason = "periodic"
                if should_save_emergency:
                    save_reason = "emergency"
                elif should_save_force:
                    save_reason = "force_triggered"
                
                self.save_networks(model_path)
                self.last_save_time = current_time
                
                # Also save a backup with timestamp for critical saves
                if should_save_emergency or should_save_force:
                    timestamp = int(current_time)
                    backup_path = os.path.join(models_dir, f'neural_networks_backup_{timestamp}.pt')
                    self.save_networks(backup_path)
                    logger.info(f"Models {save_reason} saved after {time_since_last_save:.0f} seconds with backup: {backup_path}")
                else:
                    logger.info(f"Models {save_reason} saved after {time_since_last_save:.0f} seconds")
                
                return True
            else:
                logger.debug(f"Models save check: {time_since_last_save:.0f}s elapsed, {self.save_interval - time_since_last_save:.0f}s remaining")
                return False
                
        except Exception as e:
            logger.error(f"Error in check_and_save_models: {e}")
            # Emergency save attempt on error
            try:
                logger.info("Attempting emergency save due to check_and_save_models error")
                self._force_save_models()
                return True
            except Exception as save_error:
                logger.error(f"Emergency save also failed: {save_error}")
                return False
    
    def record_performance(self, reward: float):
        """Record performance metric for model saving trigger"""
        try:
            # Convert reward to performance metric (0-1 range)
            performance = max(0.0, min(1.0, (reward + 1.0) / 2.0))  # Map [-1,1] to [0,1]
            self.performance_history.append(performance)
            self.last_performance_time = time.time()
            
            # Check for performance-based save triggers
            should_save = self._check_save_triggers(performance, reward)
            
            if should_save:
                logger.info(f"Performance-based save triggered: performance={performance:.3f}, reward={reward:.3f}")
                self._force_save_models()
            else:
                # Always check periodic saves when recording performance
                self.check_and_save_models()
            
            logger.debug(f"Performance recorded: {performance:.3f} (from reward: {reward:.3f})")
            
        except Exception as e:
            logger.error(f"Error recording performance: {e}")
            # Try to save models as error recovery
            try:
                if self.save_triggers['error_recovery']:
                    logger.info("Attempting error recovery save after performance recording failure")
                    self._force_save_models()
            except Exception as save_error:
                logger.error(f"Error recovery save also failed: {save_error}")
    
    def _check_save_triggers(self, performance: float, reward: float) -> bool:
        """Check if any save triggers are activated"""
        try:
            # Performance improvement trigger
            if self.save_triggers['performance_improvement'] and len(self.performance_history) >= 2:
                recent_avg = sum(list(self.performance_history)[-5:]) / min(5, len(self.performance_history))
                if performance > recent_avg + 0.1:  # Significant improvement
                    logger.debug("Performance improvement trigger activated")
                    return True
            
            # Training milestone trigger
            if self.save_triggers['training_milestones'] and self.training_steps > 0:
                if self.training_steps % 500 == 0:  # Every 500 training steps
                    logger.debug("Training milestone trigger activated")
                    return True
            
            # Exceptional performance trigger
            if performance > 0.8 or abs(reward) > 0.5:
                logger.debug("Exceptional performance trigger activated")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking save triggers: {e}")
            return False
    
    def _force_save_models(self) -> bool:
        """Force immediate model save regardless of timer"""
        try:
            import os
            models_dir = 'models'
            os.makedirs(models_dir, exist_ok=True)
            
            # Save with timestamp for forced saves
            timestamp = int(time.time())
            model_path = os.path.join(models_dir, f'neural_networks_forced_{timestamp}.pt')
            
            self.save_networks(model_path)
            
            # Also update the main model file
            main_model_path = os.path.join(models_dir, 'neural_networks.pt')
            self.save_networks(main_model_path)
            
            self.last_save_time = time.time()
            logger.info(f"Models force-saved to {model_path} and {main_model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error in force save models: {e}")
            return False
    
    def get_evolution_stats(self) -> Dict[str, Any]:
        """Get network evolution statistics"""
        nas_stats = self.nas_system.get_stats()
        transition_stats = self.transition_manager.get_transition_stats()
        
        return {
            'generations': self.evolution_stats['generations'],
            'improvements': self.evolution_stats['improvements'],
            'current_architecture': (
                self.meta_learner.architecture_evolver.current_sizes 
                if self.meta_learner and hasattr(self.meta_learner, 'architecture_evolver') 
                else [128, 64, 32]
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
            
            # Update meta-learner with new architecture if available
            if self.meta_learner and hasattr(self.meta_learner, 'architecture_evolver'):
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
    
    def _register_state_management(self):
        """Register with state coordinator for unified state management"""
        try:
            register_state_component(
                name="neural_network_manager",
                save_method=self._save_state_data,
                load_method=self._load_state_data,
                priority=85  # High priority for neural network state
            )
            logger.debug("NeuralNetworkManager registered with state coordinator")
        except Exception as e:
            logger.error(f"Failed to register neural network manager state management: {e}")
    
    @handle_storage_operation("neural_network_manager", "save_state")
    def _save_state_data(self) -> Dict[str, Any]:
        """
        Save neural network manager state data for state coordinator.
        
        Returns:
            Dictionary containing all neural network manager state data
        """
        try:
            # Prepare state data WITHOUT neural network states (save separately to avoid serialization issues)
            state_data = {
                'training_steps': self.training_steps,
                'last_evolution_step': self.last_evolution_step,
                'evolution_stats': dict(self.evolution_stats),
                'performance_history': list(self.performance_history),
                'training_times': list(self.training_times),
                'memory_usage_history': list(self.memory_usage_history),
                'config': {
                    'input_size': self.config.input_size,
                    'feature_dim': self.config.feature_dim,
                    'learning_rate': self.config.learning_rate,
                    'enable_few_shot': self.config.enable_few_shot,
                    'memory_efficient': self.config.memory_efficient,
                    'max_memory_gb': self.config.max_memory_gb
                },
                'saved_at': time.time(),
                'version': "2.1"  # Updated version - no longer includes network states due to serialization issues
            }
            
            # Save neural network weights separately using PyTorch's native serialization
            # This avoids the unified serialization system converting tensors to dictionaries
            try:
                import os
                models_dir = 'models'
                os.makedirs(models_dir, exist_ok=True)
                model_path = os.path.join(models_dir, 'neural_networks.pt')
                
                # Force save networks using PyTorch serialization during state save
                self.save_networks(model_path)
                state_data['networks_saved_to'] = model_path
                logger.debug("Neural network weights saved separately using PyTorch serialization")
                
            except Exception as e:
                logger.warning(f"Error saving network states to separate file: {e}")
                # Continue without network weights if they can't be saved
            
            # Validate data integrity before saving
            validation_report = validate_component_data(
                component="neural_network_manager",
                data=state_data,
                data_type="neural_state",
                level=ValidationLevel.STANDARD
            )
            
            if not validation_report.is_valid:
                logger.warning(f"Neural network manager state validation issues: {len(validation_report.issues)} issues found")
                for issue in validation_report.issues:
                    if issue.severity == "error":
                        logger.error(f"Neural network manager state error: {issue.message}")
            
            logger.debug("Neural network manager state data prepared for saving")
            return state_data
            
        except Exception as e:
            logger.error(f"Error preparing neural network manager state data: {e}")
            raise
    
    @handle_storage_operation("neural_network_manager", "load_state")
    def _load_state_data(self, state_data: Dict[str, Any]):
        """
        Load neural network manager state data from state coordinator.
        
        Args:
            state_data: Dictionary containing neural network manager state data
        """
        try:
            # Validate loaded data
            validation_report = validate_component_data(
                component="neural_network_manager",
                data=state_data,
                data_type="neural_state",
                level=ValidationLevel.STANDARD
            )
            
            if validation_report.result.value == "corrupted":
                logger.error("Neural network manager state data is corrupted, cannot load")
                return
            
            if not validation_report.is_valid:
                logger.warning(f"Neural network manager state validation issues during load: {len(validation_report.issues)} issues")
            
            # Load basic state
            if 'training_steps' in state_data:
                self.training_steps = state_data['training_steps']
            if 'last_evolution_step' in state_data:
                self.last_evolution_step = state_data['last_evolution_step']
            if 'evolution_stats' in state_data and isinstance(state_data['evolution_stats'], dict):
                self.evolution_stats.update(state_data['evolution_stats'])
            
            # Load performance history
            if 'performance_history' in state_data and isinstance(state_data['performance_history'], list):
                self.performance_history = deque(state_data['performance_history'], maxlen=200)
                logger.debug(f"Loaded {len(self.performance_history)} performance history entries")
            
            if 'training_times' in state_data and isinstance(state_data['training_times'], list):
                self.training_times = deque(state_data['training_times'], maxlen=100)
                logger.debug(f"Loaded {len(self.training_times)} training time entries")
            
            if 'memory_usage_history' in state_data and isinstance(state_data['memory_usage_history'], list):
                self.memory_usage_history = deque(state_data['memory_usage_history'], maxlen=100)
                logger.debug(f"Loaded {len(self.memory_usage_history)} memory usage entries")
            
            # Load neural network weights from separate file (using PyTorch serialization)
            try:
                # Check if networks were saved to a separate file (version 2.1+)
                if 'networks_saved_to' in state_data:
                    model_path = state_data['networks_saved_to']
                    logger.debug(f"Loading neural networks from separate file: {model_path}")
                    if self.load_networks(model_path):
                        logger.debug("Neural network weights loaded successfully from separate file")
                    else:
                        logger.warning("Failed to load neural networks from separate file")
                
                # Fallback: try to load from embedded state (older versions)
                elif any(key.endswith('_state') for key in state_data.keys()):
                    logger.warning("Detected embedded network states (old format) - attempting to load with corruption handling")
                    
                    if 'main_network_state' in state_data and hasattr(self, 'main_network') and self.main_network is not None:
                        main_state = state_data['main_network_state']
                        if isinstance(main_state, dict) and not any(isinstance(v, dict) for v in main_state.values()):
                            self.main_network.load_state_dict(main_state)
                            logger.debug("Loaded main network state (legacy)")
                        else:
                            logger.warning("Main network state appears corrupted (contains dict values instead of tensors)")
                    
                    if 'target_network_state' in state_data and hasattr(self, 'target_network') and self.target_network is not None:
                        target_state = state_data['target_network_state']
                        if isinstance(target_state, dict) and not any(isinstance(v, dict) for v in target_state.values()):
                            self.target_network.load_state_dict(target_state)
                            logger.debug("Loaded target network state (legacy)")
                        else:
                            logger.warning("Target network state appears corrupted (contains dict values instead of tensors)")
                    
                    if 'feature_learner_state' in state_data and hasattr(self, 'feature_learner') and self.feature_learner is not None:
                        feature_state = state_data['feature_learner_state']
                        if isinstance(feature_state, dict) and not any(isinstance(v, dict) for v in feature_state.values()):
                            self.feature_learner.load_state_dict(feature_state)
                            logger.debug("Loaded feature learner state (legacy)")
                        else:
                            logger.warning("Feature learner state appears corrupted (contains dict values instead of tensors)")
                
                else:
                    logger.debug("No neural network states found in state data")
                    
            except Exception as e:
                logger.warning(f"Error loading network states: {e}")
                # Continue without loading network weights if they can't be loaded
            
            logger.info("Neural network manager state loaded successfully from state coordinator")
            
        except Exception as e:
            logger.error(f"Error loading neural network manager state data: {e}")
            raise
    
    def _load_persisted_state(self):
        """Load persisted state if available"""
        try:
            # Try to load from unified serialization system
            state_data = deserialize_component_state("neural_network_manager")
            if state_data:
                self._load_state_data(state_data)
                logger.info("Neural network manager state loaded from persistent storage")
        except Exception as e:
            logger.debug(f"No existing neural network manager state found or failed to load: {e}")
    
    def disable_background_saves(self):
        """Disable background periodic saves if needed"""
        self.save_triggers['force_periodic'] = False
        logger.info("Background periodic saves disabled")
    
    def enable_background_saves(self):
        """Re-enable background periodic saves"""
        self.save_triggers['force_periodic'] = True
        logger.info("Background periodic saves enabled")
    
    def get_save_status(self) -> Dict[str, Any]:
        """Get current save status and statistics"""
        current_time = time.time()
        time_since_last_save = current_time - self.last_save_time
        
        return {
            'last_save_time': self.last_save_time,
            'time_since_last_save_seconds': time_since_last_save,
            'save_interval_seconds': self.save_interval,
            'time_until_next_save_seconds': max(0, self.save_interval - time_since_last_save),
            'force_save_counter': self.force_save_counter,
            'save_triggers': dict(self.save_triggers),
            'next_emergency_save_in_seconds': max(0, (self.save_interval * 2) - time_since_last_save)
        }