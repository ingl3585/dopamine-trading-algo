import os
from datetime import datetime
from meta_learner import PureMetaLearner
import logging

log = logging.getLogger(__name__)

class AdaptiveConfig:
    """
    PURE BLACK BOX: All configuration parameters adapt through experience.
    No static values. All behavior evolves from trading outcomes.
    """

    def __init__(self, db_path: str = "data/meta_parameters.db"):
        self.meta_learner = PureMetaLearner(db_path)

        # Network settings (fixed)
        self.TCP_HOST = "localhost"
        self.FEATURE_PORT = 5556
        self.SIGNAL_PORT = 5557

        # Ensure required directories exist
        for directory in ['patterns', 'data', 'models', 'logs', 'meta_learning']:
            os.makedirs(directory, exist_ok=True)

    # === Risk Management ===
    @property
    def MAX_DAILY_LOSS(self):
        return 10000 * self.meta_learner.get_parameter('max_daily_loss_pct')

    @property
    def MAX_CONSECUTIVE_LOSSES(self):
        return int(self.meta_learner.get_parameter('max_consecutive_losses'))

    @property
    def STOP_LOSS_MAX_PERCENT(self):
        return self.meta_learner.get_parameter('stop_loss_max_pct')

    @property
    def TAKE_PROFIT_MAX_PERCENT(self):
        return self.meta_learner.get_parameter('take_profit_max_pct')

    # === Position Sizing ===
    @property
    def PRODUCTION_PHASE_SIZE(self):
        return self.meta_learner.get_parameter('position_size_base')

    @property
    def EXPLORATION_PHASE_SIZE(self):
        return self.PRODUCTION_PHASE_SIZE * 0.2

    @property
    def DEVELOPMENT_PHASE_SIZE(self):
        return self.PRODUCTION_PHASE_SIZE * 0.6

    # === Trade Limits ===
    def _scaled_trades(self, factor):
        freq = self.meta_learner.get_parameter('scaling_frequency')
        return max(1, int(factor * freq))

    @property
    def MAX_DAILY_TRADES_EXPLORATION(self):
        return self._scaled_trades(20)

    @property
    def MAX_DAILY_TRADES_DEVELOPMENT(self):
        return self._scaled_trades(15)

    @property
    def MAX_DAILY_TRADES_PRODUCTION(self):
        return self._scaled_trades(25)

    # === Learning Parameters ===
    @property
    def AI_MIN_CONFIDENCE(self):
        return self.meta_learner.get_parameter('entry_confidence_threshold')

    @property
    def EXIT_CONFIDENCE_THRESHOLD(self):
        return self.meta_learner.get_parameter('exit_confidence_threshold')

    @property
    def SCALING_CONFIDENCE_THRESHOLD(self):
        return self.meta_learner.get_parameter('scaling_confidence_threshold')

    @property
    def AI_EXPLORATION_RATE(self):
        return self.meta_learner.get_parameter('epsilon_min')

    @property
    def AI_LEARNING_RATE(self):
        return self.meta_learner.get_parameter('policy_learning_rate')

    @property
    def AI_VALUE_LEARNING_RATE(self):
        return self.meta_learner.get_parameter('value_learning_rate')

    @property
    def AI_MEMORY_SIZE(self):
        return int(self.meta_learner.get_parameter('experience_buffer_size'))

    # === Architecture + Memory ===
    def get_network_architecture(self):
        return self.meta_learner.get_network_architecture()

    def get_batch_size(self):
        base = 32
        mult = self.meta_learner.get_parameter('batch_size_multiplier')
        return max(16, int(base * mult))

    @property
    def MAX_PATTERN_HISTORY(self):
        return self.AI_MEMORY_SIZE * 2

    @property
    def PATTERN_CLEANUP_DAYS(self):
        rate = self.meta_learner.get_parameter('meta_learning_rate')
        return max(30, int(90 * (1.0 - rate * 10000)))

    # === Utils ===
    def update_parameter_from_outcome(self, param_name, outcome, context=None):
        self.meta_learner.update_parameter(param_name, outcome, context)

    def get_learning_efficiency(self):
        hist = self.meta_learner.learning_efficiency_history
        return hist[-1] if hist else 0.0

    def should_rebuild_network(self):
        return self.meta_learner.should_rebuild_network()

    def force_save_parameters(self):
        self.meta_learner.force_save()

    def get_meta_learner(self):
        return self.meta_learner

# Factory

def create_adaptive_config(db_path="data/meta_parameters.db") -> AdaptiveConfig:
    return AdaptiveConfig(db_path)