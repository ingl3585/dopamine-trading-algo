# config.py - SIMPLIFIED but still adaptive

import os
import json
import logging
from datetime import datetime
from collections import defaultdict, deque

log = logging.getLogger(__name__)

class SimpleAdaptiveConfig:
    """
    Simplified adaptive configuration - fewer parameters, easier to understand
    """
    
    def __init__(self, config_file="data/adaptive_config.json"):
        self.config_file = config_file
        
        # Core adaptive parameters - much fewer than before
        self.parameters = {
            # Risk Management
            'position_size': 1.0,
            'max_daily_loss': 500.0,
            'max_consecutive_losses': 3,
            
            # Learning
            'confidence_threshold': 0.6,
            'learning_rate': 0.001,
            'exploration_rate': 0.15,
            
            # Timeouts
            'signal_timeout': 30.0,
        }
        
        # Track performance for each parameter
        self.parameter_performance = defaultdict(lambda: deque(maxlen=50))
        
        # Load existing config if available
        self.load_config()
        
        # Ensure directories exist
        for directory in ['data', 'models', 'patterns', 'logs']:
            os.makedirs(directory, exist_ok=True)
    
    def get_parameter(self, name, default=None):
        """Get a parameter value"""
        return self.parameters.get(name, default)
    
    def update_parameter(self, name, outcome):
        """Update parameter based on outcome"""
        if name not in self.parameters:
            log.warning(f"Unknown parameter: {name}")
            return
        
        # Simple adaptive update
        self.parameter_performance[name].append(outcome)
        
        # Need at least 5 samples before adapting
        if len(self.parameter_performance[name]) < 5:
            return
        
        # Calculate recent performance
        recent_performance = list(self.parameter_performance[name])[-10:]
        avg_performance = sum(recent_performance) / len(recent_performance)
        
        # Adapt based on performance
        if avg_performance > 0.1:  # Good performance
            if name == 'confidence_threshold':
                self.parameters[name] = max(0.3, self.parameters[name] - 0.02)
            elif name == 'exploration_rate':
                self.parameters[name] = max(0.05, self.parameters[name] - 0.01)
            elif name == 'position_size':
                self.parameters[name] = min(3.0, self.parameters[name] + 0.1)
        
        elif avg_performance < -0.1:  # Poor performance
            if name == 'confidence_threshold':
                self.parameters[name] = min(0.9, self.parameters[name] + 0.02)
            elif name == 'exploration_rate':
                self.parameters[name] = min(0.3, self.parameters[name] + 0.01)
            elif name == 'position_size':
                self.parameters[name] = max(0.5, self.parameters[name] - 0.1)
        
        log.info(f"Parameter {name} adapted to {self.parameters[name]:.3f} (performance: {avg_performance:.3f})")
    
    def load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    saved_data = json.load(f)
                    
                self.parameters.update(saved_data.get('parameters', {}))
                
                # Load performance history
                for name, history in saved_data.get('performance', {}).items():
                    self.parameter_performance[name] = deque(history, maxlen=50)
                
                log.info(f"Loaded adaptive config from {self.config_file}")
        except Exception as e:
            log.info(f"Starting with default config: {e}")
    
    def save_config(self):
        """Save configuration to file"""
        try:
            save_data = {
                'parameters': self.parameters,
                'performance': {name: list(history) for name, history in self.parameter_performance.items()},
                'saved_at': datetime.now().isoformat()
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(save_data, f, indent=2)
                
            log.info(f"Saved adaptive config to {self.config_file}")
        except Exception as e:
            log.error(f"Failed to save config: {e}")
    
    def get_status(self):
        """Get current configuration status"""
        return {
            'parameters': self.parameters.copy(),
            'total_adaptations': sum(len(history) for history in self.parameter_performance.values()),
            'recent_performance': {
                name: sum(list(history)[-5:]) / min(len(history), 5) 
                for name, history in self.parameter_performance.items() 
                if len(history) > 0
            }
        }

# Factory function
def create_config():
    return SimpleAdaptiveConfig()