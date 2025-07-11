"""
Dynamic Network Pruning and Expansion

This module implements dynamic pruning and expansion techniques for neural networks,
allowing them to adapt their capacity based on performance and complexity requirements.

Features:
- Magnitude-based pruning
- Gradual pruning with performance monitoring
- Dynamic expansion when needed
- Structured and unstructured pruning
- Pruning sensitivity analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict, deque
import copy

logger = logging.getLogger(__name__)


@dataclass
class PruningConfig:
    """Configuration for pruning operations"""
    # Pruning thresholds
    magnitude_threshold: float = 0.01
    sensitivity_threshold: float = 0.05
    
    # Pruning schedules
    pruning_frequency: int = 1000  # Steps between pruning evaluations
    gradual_steps: int = 100       # Steps for gradual pruning
    
    # Expansion triggers
    performance_decline_threshold: float = 0.1
    capacity_utilization_threshold: float = 0.9
    
    # Pruning types
    structured_pruning: bool = False
    unstructured_pruning: bool = True
    
    # Safety limits
    min_sparsity: float = 0.1      # Minimum sparsity to maintain
    max_sparsity: float = 0.8      # Maximum sparsity allowed
    min_layer_size: int = 8        # Minimum neurons per layer


class LayerAnalyzer:
    """Analyzes layer importance and pruning sensitivity"""
    
    def __init__(self):
        self.importance_scores = {}
        self.sensitivity_scores = {}
        self.activation_stats = defaultdict(list)
        self.gradient_stats = defaultdict(list)
    
    def analyze_layer_importance(self, model: nn.Module, inputs: torch.Tensor) -> Dict[str, float]:
        """Analyze importance of each layer"""
        importance_scores = {}
        
        # Hook to collect activations
        activations = {}
        def get_activation(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activations[name] = output.detach()
            return hook
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hooks.append(module.register_forward_hook(get_activation(name)))
        
        # Forward pass
        with torch.no_grad():
            _ = model(inputs)
        
        # Calculate importance based on activation statistics
        for name, activation in activations.items():
            # Use activation magnitude as importance measure
            importance = torch.mean(torch.abs(activation)).item()
            importance_scores[name] = importance
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
        
        self.importance_scores.update(importance_scores)
        return importance_scores
    
    def analyze_pruning_sensitivity(self, model: nn.Module, 
                                  validation_data: torch.Tensor,
                                  original_performance: float) -> Dict[str, float]:
        """Analyze sensitivity of each layer to pruning"""
        sensitivity_scores = {}
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and hasattr(module, 'weight'):
                # Create temporary pruned version
                original_weight = module.weight.data.clone()
                
                # Apply temporary pruning (remove 10% smallest weights)
                weight_magnitude = torch.abs(module.weight.data)
                threshold = torch.quantile(weight_magnitude, 0.1)
                mask = weight_magnitude > threshold
                module.weight.data *= mask
                
                # Evaluate performance
                with torch.no_grad():
                    output = model(validation_data)
                    # Simple performance metric (you can customize this)
                    current_performance = torch.mean(output).item()
                
                # Calculate sensitivity
                performance_drop = abs(original_performance - current_performance)
                sensitivity_scores[name] = performance_drop
                
                # Restore original weights
                module.weight.data = original_weight
        
        self.sensitivity_scores.update(sensitivity_scores)
        return sensitivity_scores


class MagnitudePruner:
    """Implements magnitude-based pruning"""
    
    def __init__(self, config: PruningConfig):
        self.config = config
        self.pruning_masks = {}
        self.sparsity_levels = {}
    
    def calculate_pruning_masks(self, model: nn.Module, target_sparsity: float) -> Dict[str, torch.Tensor]:
        """Calculate pruning masks based on weight magnitudes"""
        masks = {}
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and hasattr(module, 'weight'):
                weight = module.weight.data
                weight_magnitude = torch.abs(weight)
                
                # Calculate threshold for target sparsity
                flat_weights = weight_magnitude.flatten()
                threshold = torch.quantile(flat_weights, target_sparsity)
                
                # Create mask
                mask = weight_magnitude > threshold
                masks[name] = mask
                
                # Track sparsity
                actual_sparsity = 1.0 - torch.mean(mask.float()).item()
                self.sparsity_levels[name] = actual_sparsity
                
                logger.debug(f"Layer {name}: target sparsity {target_sparsity:.3f}, "
                           f"actual sparsity {actual_sparsity:.3f}")
        
        self.pruning_masks.update(masks)
        return masks
    
    def apply_pruning_masks(self, model: nn.Module, masks: Dict[str, torch.Tensor]):
        """Apply pruning masks to model weights"""
        for name, module in model.named_modules():
            if name in masks and hasattr(module, 'weight'):
                module.weight.data *= masks[name]
                
                # Also handle bias if present
                if hasattr(module, 'bias') and module.bias is not None:
                    # For bias, use a simpler magnitude-based approach
                    bias_magnitude = torch.abs(module.bias.data)
                    bias_threshold = torch.quantile(bias_magnitude, 0.1)  # Prune 10% of biases
                    bias_mask = bias_magnitude > bias_threshold
                    module.bias.data *= bias_mask
    
    def gradual_pruning(self, model: nn.Module, 
                       current_step: int, 
                       total_steps: int,
                       final_sparsity: float) -> bool:
        """Implement gradual pruning schedule"""
        if current_step >= total_steps:
            return False
        
        # Calculate current target sparsity
        progress = current_step / total_steps
        current_sparsity = final_sparsity * progress
        
        # Apply pruning
        masks = self.calculate_pruning_masks(model, current_sparsity)
        self.apply_pruning_masks(model, masks)
        
        return True


class StructuredPruner:
    """Implements structured pruning (channel/neuron level)"""
    
    def __init__(self, config: PruningConfig):
        self.config = config
        self.pruned_channels = defaultdict(set)
    
    def prune_neurons(self, model: nn.Module, layer_name: str, num_neurons: int) -> bool:
        """Prune entire neurons from a layer"""
        try:
            # Find the layer
            layer = None
            for name, module in model.named_modules():
                if name == layer_name and isinstance(module, nn.Linear):
                    layer = module
                    break
            
            if layer is None:
                logger.warning(f"Layer {layer_name} not found for structured pruning")
                return False
            
            # Calculate neuron importance scores
            weight_norms = torch.norm(layer.weight.data, dim=1)
            
            # Find neurons to prune (smallest norms)
            _, indices_to_prune = torch.topk(weight_norms, num_neurons, largest=False)
            
            # Create new smaller layer
            old_out_features = layer.out_features
            new_out_features = old_out_features - num_neurons
            
            if new_out_features < self.config.min_layer_size:
                logger.warning(f"Cannot prune {num_neurons} neurons from {layer_name}: "
                             f"would result in {new_out_features} < {self.config.min_layer_size}")
                return False
            
            # Create mask for keeping neurons
            keep_mask = torch.ones(old_out_features, dtype=torch.bool)
            keep_mask[indices_to_prune] = False
            
            # Update weights and biases
            layer.weight.data = layer.weight.data[keep_mask]
            if layer.bias is not None:
                layer.bias.data = layer.bias.data[keep_mask]
            
            # Update layer dimensions
            layer.out_features = new_out_features
            
            # Track pruned neurons
            self.pruned_channels[layer_name].update(indices_to_prune.tolist())
            
            logger.info(f"Pruned {num_neurons} neurons from {layer_name}: "
                       f"{old_out_features} -> {new_out_features}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error pruning neurons from {layer_name}: {e}")
            return False


class DynamicExpander:
    """Implements dynamic network expansion"""
    
    def __init__(self, config: PruningConfig):
        self.config = config
        self.expansion_history = []
    
    def should_expand(self, performance_history: List[float], 
                     current_utilization: float) -> bool:
        """Determine if network should be expanded"""
        if len(performance_history) < 10:
            return False
        
        # Check for performance decline
        recent_perf = np.mean(performance_history[-5:])
        past_perf = np.mean(performance_history[-10:-5])
        performance_decline = (past_perf - recent_perf) / past_perf
        
        # Check capacity utilization
        high_utilization = current_utilization > self.config.capacity_utilization_threshold
        
        # Expansion conditions
        should_expand = (
            performance_decline > self.config.performance_decline_threshold and
            high_utilization
        )
        
        logger.debug(f"Expansion check: perf_decline={performance_decline:.4f}, "
                    f"utilization={current_utilization:.4f}, should_expand={should_expand}")
        
        return should_expand
    
    def expand_layer(self, model: nn.Module, layer_name: str, 
                    expansion_factor: float = 1.2) -> bool:
        """Expand a specific layer"""
        try:
            # Find the layer
            layer = None
            for name, module in model.named_modules():
                if name == layer_name and isinstance(module, nn.Linear):
                    layer = module
                    break
            
            if layer is None:
                logger.warning(f"Layer {layer_name} not found for expansion")
                return False
            
            # Calculate new size
            old_size = layer.out_features
            new_size = int(old_size * expansion_factor)
            
            # Create new weights with proper initialization
            new_weight = torch.zeros(new_size, layer.in_features, 
                                   dtype=layer.weight.dtype, device=layer.weight.device)
            new_weight[:old_size] = layer.weight.data
            
            # Initialize new neurons with small random values
            nn.init.xavier_uniform_(new_weight[old_size:], gain=0.1)
            
            # Update weights
            layer.weight.data = new_weight
            
            # Handle bias
            if layer.bias is not None:
                new_bias = torch.zeros(new_size, dtype=layer.bias.dtype, device=layer.bias.device)
                new_bias[:old_size] = layer.bias.data
                layer.bias.data = new_bias
            
            # Update layer dimensions
            layer.out_features = new_size
            
            # Record expansion
            self.expansion_history.append({
                'layer': layer_name,
                'old_size': old_size,
                'new_size': new_size,
                'expansion_factor': expansion_factor
            })
            
            logger.info(f"Expanded {layer_name}: {old_size} -> {new_size} neurons")
            return True
            
        except Exception as e:
            logger.error(f"Error expanding layer {layer_name}: {e}")
            return False


class DynamicPruningManager:
    """Main manager for dynamic pruning and expansion"""
    
    def __init__(self, config: Optional[PruningConfig] = None):
        self.config = config or PruningConfig()
        
        # Components
        self.analyzer = LayerAnalyzer()
        self.magnitude_pruner = MagnitudePruner(self.config)
        self.structured_pruner = StructuredPruner(self.config)
        self.expander = DynamicExpander(self.config)
        
        # State tracking
        self.step_count = 0
        self.last_pruning_step = 0
        self.performance_history = deque(maxlen=100)
        self.sparsity_history = deque(maxlen=100)
        
        # Model state
        self.original_model_state = None
        self.current_sparsity = 0.0
    
    def should_prune(self) -> bool:
        """Check if pruning should be performed at this step"""
        steps_since_last = self.step_count - self.last_pruning_step
        return steps_since_last >= self.config.pruning_frequency
    
    def evaluate_model_performance(self, model: nn.Module, 
                                 validation_data: torch.Tensor) -> float:
        """Evaluate current model performance"""
        model.eval()
        with torch.no_grad():
            outputs = model(validation_data)
            # Simple performance metric - customize as needed
            performance = float(torch.mean(torch.abs(outputs)))
        return performance
    
    def calculate_capacity_utilization(self, model: nn.Module) -> float:
        """Calculate current capacity utilization"""
        total_params = 0
        active_params = 0
        
        for module in model.modules():
            if isinstance(module, nn.Linear):
                total_params += module.weight.numel()
                if hasattr(module, 'weight'):
                    active_params += torch.count_nonzero(module.weight).item()
        
        utilization = active_params / total_params if total_params > 0 else 0.0
        return utilization
    
    def adaptive_pruning_step(self, model: nn.Module, 
                            validation_data: torch.Tensor,
                            performance_metric: float) -> Dict[str, Any]:
        """Perform one step of adaptive pruning/expansion"""
        results = {
            'action_taken': 'none',
            'sparsity_change': 0.0,
            'performance_impact': 0.0
        }
        
        try:
            self.step_count += 1
            self.performance_history.append(performance_metric)
            
            # Calculate current utilization
            current_utilization = self.calculate_capacity_utilization(model)
            
            # Check if we should expand
            if self.expander.should_expand(list(self.performance_history), current_utilization):
                # Find layer to expand (choose layer with highest importance)
                importance_scores = self.analyzer.analyze_layer_importance(model, validation_data)
                if importance_scores:
                    best_layer = max(importance_scores.items(), key=lambda x: x[1])[0]
                    if self.expander.expand_layer(model, best_layer):
                        results['action_taken'] = 'expansion'
                        results['expanded_layer'] = best_layer
            
            # Check if we should prune
            elif self.should_prune() and self.current_sparsity < self.config.max_sparsity:
                old_performance = performance_metric
                
                if self.config.structured_pruning:
                    # Structured pruning
                    sensitivity_scores = self.analyzer.analyze_pruning_sensitivity(
                        model, validation_data, old_performance
                    )
                    
                    # Prune from least sensitive layer
                    if sensitivity_scores:
                        least_sensitive = min(sensitivity_scores.items(), key=lambda x: x[1])[0]
                        if self.structured_pruner.prune_neurons(model, least_sensitive, 1):
                            results['action_taken'] = 'structured_pruning'
                            results['pruned_layer'] = least_sensitive
                
                elif self.config.unstructured_pruning:
                    # Unstructured pruning
                    target_sparsity = min(self.current_sparsity + 0.05, self.config.max_sparsity)
                    masks = self.magnitude_pruner.calculate_pruning_masks(model, target_sparsity)
                    self.magnitude_pruner.apply_pruning_masks(model, masks)
                    
                    self.current_sparsity = target_sparsity
                    results['action_taken'] = 'unstructured_pruning'
                    results['sparsity_change'] = target_sparsity - self.current_sparsity
                
                self.last_pruning_step = self.step_count
            
            # Update tracking
            self.sparsity_history.append(self.current_sparsity)
            
        except Exception as e:
            logger.error(f"Error in adaptive pruning step: {e}")
            results['error'] = str(e)
        
        return results
    
    def get_pruning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pruning statistics"""
        return {
            'current_sparsity': self.current_sparsity,
            'step_count': self.step_count,
            'last_pruning_step': self.last_pruning_step,
            'sparsity_history': list(self.sparsity_history),
            'performance_history': list(self.performance_history),
            'expansion_history': self.expander.expansion_history,
            'layer_importance': self.analyzer.importance_scores,
            'layer_sensitivity': self.analyzer.sensitivity_scores
        }


def create_dynamic_pruning_manager(config: Optional[PruningConfig] = None) -> DynamicPruningManager:
    """Factory function to create dynamic pruning manager"""
    return DynamicPruningManager(config)