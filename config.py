# adaptive_config.py - REPLACES config.py with pure adaptive parameters

import os
from datetime import datetime
from meta_learner import PureMetaLearner
import logging
import numpy as np

log = logging.getLogger(__name__)

class AdaptiveConfig:
    """
    PURE BLACK BOX: All configuration parameters adapt through experience
    NO STATIC VALUES - everything learned from trading outcomes
    """
    
    def __init__(self, db_path: str = "data/meta_parameters.db"):
        # Initialize meta-learner
        self.meta_learner = PureMetaLearner(db_path)
        
        # Only truly static values are network/file paths
        self.TCP_HOST = "localhost"
        self.FEATURE_PORT = 5556
        self.SIGNAL_PORT = 5557
        
        # Create essential directories
        for directory in ['patterns', 'data', 'models', 'logs', 'meta_learning']:
            os.makedirs(directory, exist_ok=True)
        
        log.info("ADAPTIVE CONFIG: All parameters will adapt through experience")
        log.info("Zero hardcoded thresholds - pure black box learning")
    
    # ADAPTIVE SAFETY LIMITS (learn from losses)
    @property
    def MAX_DAILY_LOSS(self) -> float:
        """Adaptive daily loss limit based on account size and learned risk tolerance"""
        # This would typically be account_size * learned_percentage
        # For demo, assuming $10,000 account
        account_size = 10000
        loss_pct = self.meta_learner.get_parameter('max_daily_loss_pct')
        return account_size * loss_pct
    
    @property
    def MAX_CONSECUTIVE_LOSSES(self) -> int:
        """Adaptive consecutive loss limit - learned from experience"""
        return int(self.meta_learner.get_parameter('max_consecutive_losses'))
    
    @property
    def EXPLORATION_PHASE_SIZE(self) -> float:
        """Adaptive exploration position size"""
        base_size = self.meta_learner.get_parameter('position_size_base')
        return base_size * 0.2  # Exploration uses 20% of learned base
    
    @property
    def DEVELOPMENT_PHASE_SIZE(self) -> float:
        """Adaptive development position size"""
        base_size = self.meta_learner.get_parameter('position_size_base')
        return base_size * 0.6  # Development uses 60% of learned base
    
    @property
    def PRODUCTION_PHASE_SIZE(self) -> float:
        """Adaptive production position size"""
        return self.meta_learner.get_parameter('position_size_base')
    
    @property
    def MAX_DAILY_TRADES_EXPLORATION(self) -> int:
        """Adaptive daily trade limits"""
        base_frequency = self.meta_learner.get_parameter('scaling_frequency')
        return max(10, int(20 * base_frequency))
    
    @property
    def MAX_DAILY_TRADES_DEVELOPMENT(self) -> int:
        base_frequency = self.meta_learner.get_parameter('scaling_frequency')
        return max(5, int(15 * base_frequency))
    
    @property
    def MAX_DAILY_TRADES_PRODUCTION(self) -> int:
        base_frequency = self.meta_learner.get_parameter('scaling_frequency')
        return max(8, int(25 * base_frequency))
    
    # ADAPTIVE AI LEARNING PARAMETERS
    @property
    def AI_MIN_CONFIDENCE(self) -> float:
        """Adaptive minimum confidence threshold"""
        return self.meta_learner.get_parameter('entry_confidence_threshold')
    
    @property
    def AI_EXPLORATION_RATE(self) -> float:
        """Adaptive exploration rate"""
        return self.meta_learner.get_parameter('epsilon_min')
    
    @property
    def AI_LEARNING_RATE(self) -> float:
        """Adaptive policy learning rate"""
        return self.meta_learner.get_parameter('policy_learning_rate')
    
    @property
    def AI_VALUE_LEARNING_RATE(self) -> float:
        """Adaptive value learning rate"""
        return self.meta_learner.get_parameter('value_learning_rate')
    
    @property
    def AI_MEMORY_SIZE(self) -> int:
        """Adaptive experience buffer size"""
        return int(self.meta_learner.get_parameter('experience_buffer_size'))
    
    # ADAPTIVE CONFIDENCE LEARNING
    @property
    def NEUTRAL_THRESHOLD(self) -> float:
        """Adaptive neutral threshold - learns what 'neutral' actually means"""
        return self.meta_learner.get_parameter('entry_confidence_threshold')
    
    @property
    def MIN_LEARNING_SAMPLES(self) -> int:
        """Adaptive minimum samples needed for learning"""
        learning_efficiency = len(self.meta_learner.learning_efficiency_history)
        return max(3, 10 - int(learning_efficiency / 10))  # Fewer samples as AI improves
    
    @property
    def THRESHOLD_LEARNING_RATE(self) -> float:
        """Adaptive threshold learning rate"""
        return self.meta_learner.get_parameter('meta_learning_rate') * 1000
    
    # ADAPTIVE MEMORY MANAGEMENT
    @property
    def MAX_PATTERN_HISTORY(self) -> int:
        """Adaptive pattern history size"""
        base_size = int(self.meta_learner.get_parameter('experience_buffer_size'))
        return base_size * 2  # Pattern memory is 2x experience buffer
    
    @property
    def PATTERN_CLEANUP_DAYS(self) -> int:
        """Adaptive pattern cleanup interval"""
        learning_rate = self.meta_learner.get_parameter('meta_learning_rate')
        # Faster learning = more frequent cleanup
        return max(30, int(90 * (1.0 - learning_rate * 10000)))
    
    # ADAPTIVE RISK MANAGEMENT
    @property
    def STOP_LOSS_MAX_PERCENT(self) -> float:
        """Adaptive maximum stop loss percentage"""
        return self.meta_learner.get_parameter('stop_loss_max_pct')
    
    @property
    def TAKE_PROFIT_MAX_PERCENT(self) -> float:
        """Adaptive maximum take profit percentage"""
        return self.meta_learner.get_parameter('take_profit_max_pct')
    
    @property
    def SCALING_CONFIDENCE_THRESHOLD(self) -> float:
        """Adaptive scaling confidence threshold"""
        return self.meta_learner.get_parameter('scaling_confidence_threshold')
    
    @property
    def EXIT_CONFIDENCE_THRESHOLD(self) -> float:
        """Adaptive exit confidence threshold"""
        return self.meta_learner.get_parameter('exit_confidence_threshold')
    
    # NETWORK ARCHITECTURE (ADAPTIVE)
    def get_network_architecture(self):
        """Get current optimal network architecture"""
        return self.meta_learner.get_network_architecture()
    
    def get_batch_size(self) -> int:
        """Adaptive batch size for training"""
        base_size = 32
        multiplier = self.meta_learner.get_parameter('batch_size_multiplier')
        return max(16, int(base_size * multiplier))
    
    # LEARNING EFFICIENCY TRACKING
    def update_parameter_from_outcome(self, param_name: str, outcome: float, context: dict = None):
        """Update any parameter based on trading outcome"""
        self.meta_learner.update_parameter(param_name, outcome, context)
    
    def get_learning_efficiency(self) -> float:
        """Get current learning efficiency"""
        if len(self.meta_learner.learning_efficiency_history) > 0:
            return self.meta_learner.learning_efficiency_history[-1]
        return 0.0
    
    def should_rebuild_network(self) -> bool:
        """Check if network should be rebuilt due to architecture changes"""
        return self.meta_learner.should_rebuild_network()
    
    # ADAPTATION REPORTING
    def get_adaptation_status(self) -> str:
        """Get comprehensive adaptation status"""
        
        status = f"""
=== ADAPTIVE CONFIGURATION STATUS ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CURRENT ADAPTIVE PARAMETERS:

Risk Management (Learning from Losses):
  Daily Loss Limit: ${self.MAX_DAILY_LOSS:.0f} ({self.meta_learner.get_parameter('max_daily_loss_pct'):.1%} of account)
  Consecutive Losses: {self.MAX_CONSECUTIVE_LOSSES}
  Position Size: {self.PRODUCTION_PHASE_SIZE:.3f}
  Stop Loss Max: {self.STOP_LOSS_MAX_PERCENT:.2%}
  Take Profit Max: {self.TAKE_PROFIT_MAX_PERCENT:.2%}

Learning Parameters (Self-Optimizing):
  Policy Learning Rate: {self.AI_LEARNING_RATE:.6f}
  Value Learning Rate: {self.AI_VALUE_LEARNING_RATE:.6f}
  Entry Confidence: {self.AI_MIN_CONFIDENCE:.3f}
  Scaling Confidence: {self.SCALING_CONFIDENCE_THRESHOLD:.3f}
  Exit Confidence: {self.EXIT_CONFIDENCE_THRESHOLD:.3f}

Memory & Architecture (Adaptive):
  Experience Buffer: {self.AI_MEMORY_SIZE:,}
  Batch Size: {self.get_batch_size()}
  Pattern History: {self.MAX_PATTERN_HISTORY:,}
  Cleanup Days: {self.PATTERN_CLEANUP_DAYS}

Network Architecture (Self-Evolving):
"""
        
        arch = self.get_network_architecture()
        for param, value in arch.items():
            status += f"  {param}: {value}\n"
        
        status += f"""
Learning Efficiency: {self.get_learning_efficiency():.3f}
Total Parameter Updates: {self.meta_learner.total_updates}

ALL PARAMETERS OPTIMIZING THROUGH PURE EXPERIENCE!
No hardcoded thresholds - everything learned from trading outcomes.
"""
        
        return status
    
    def force_save_parameters(self):
        """Force save all learned parameters"""
        self.meta_learner.force_save()
        log.info("ADAPTIVE CONFIG: All learned parameters saved")
    
    def get_meta_learner(self) -> PureMetaLearner:
        """Get the underlying meta-learner for advanced operations"""
        return self.meta_learner

# Factory function for creating adaptive config
def create_adaptive_config(db_path: str = "data/meta_parameters.db") -> AdaptiveConfig:
    """Create adaptive configuration system"""
    config = AdaptiveConfig(db_path)
    
    log.info("PURE BLACK BOX CONFIG: All parameters will adapt")
    log.info("Initial configuration based on learned parameters:")
    log.info(f"  Position Size: {config.PRODUCTION_PHASE_SIZE:.3f}")
    log.info(f"  Daily Loss Limit: ${config.MAX_DAILY_LOSS:.0f}")
    log.info(f"  Entry Confidence: {config.AI_MIN_CONFIDENCE:.3f}")
    log.info(f"  Learning Rate: {config.AI_LEARNING_RATE:.6f}")
    
    return config

# Usage example - drop-in replacement for existing config
if __name__ == "__main__":
    # Test adaptive config
    config = create_adaptive_config()
    
    print("ADAPTIVE CONFIGURATION TEST")
    print("="*50)
    print(config.get_adaptation_status())
    
    # Simulate learning from outcomes
    print("\nSimulating parameter adaptation...")
    
    for i in range(10):
        # Simulate a trading outcome
        outcome = np.random.normal(0, 1)  # Random outcome
        
        # Update parameters based on outcome
        config.update_parameter_from_outcome('position_size_base', outcome)
        config.update_parameter_from_outcome('entry_confidence_threshold', outcome)
        
        if i % 3 == 0:
            print(f"  After outcome {outcome:.3f}:")
            print(f"    Position Size: {config.PRODUCTION_PHASE_SIZE:.3f}")
            print(f"    Entry Confidence: {config.AI_MIN_CONFIDENCE:.3f}")
    
    print("\nFinal adapted configuration:")
    print(config.get_adaptation_status())