# meta_learner.py

import logging
import sqlite3
import numpy as np
from datetime import datetime
from collections import deque
from typing import Dict

log = logging.getLogger(__name__)

class SimplifiedMetaLearner:
    """
    Simplified meta-learner that adapts core parameters based on outcomes
    Much simpler than the original but still learns
    """
    
    def __init__(self, db_path="data/meta_learner.db"):
        self.db_path = db_path
        
        # Core parameters that adapt
        self.parameters = {
            'position_size_multiplier': 1.0,
            'confidence_threshold': 0.6,
            'risk_per_trade': 0.02,
            'stop_loss_pct': 0.015,
            'take_profit_pct': 0.03,
            'exploration_rate': 0.15,
            'learning_rate': 0.001,
        }
        
        # Track outcomes for each parameter
        self.outcomes = {name: deque(maxlen=100) for name in self.parameters}
        
        # Simple statistics
        self.total_updates = 0
        self.successful_adaptations = 0
        
        # Initialize database
        self.init_db()
        self.load_parameters()
    
    def init_db(self):
        """Initialize SQLite database for persistence"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS parameters (
                name TEXT PRIMARY KEY,
                value REAL,
                last_updated TEXT,
                total_updates INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS outcomes (
                parameter_name TEXT,
                outcome REAL,
                timestamp TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_parameter(self, name):
        """Get current parameter value"""
        return self.parameters.get(name, 0.5)
    
    def update_parameter(self, name, outcome):
        """Update parameter based on trading outcome"""
        if name not in self.parameters:
            log.warning(f"Unknown parameter: {name}")
            return
        
        # Store outcome
        self.outcomes[name].append(outcome)
        self.total_updates += 1
        
        # Need sufficient data before adapting
        if len(self.outcomes[name]) < 10:
            return
        
        # Calculate recent performance
        recent_outcomes = list(self.outcomes[name])[-20:]
        avg_outcome = np.mean(recent_outcomes)
        
        # Simple adaptive rule
        old_value = self.parameters[name]
        
        if avg_outcome > 0.05:  # Good performance
            if name == 'confidence_threshold':
                self.parameters[name] = max(0.3, old_value - 0.01)  # Lower threshold = more trades
            elif name == 'position_size_multiplier':
                self.parameters[name] = min(2.0, old_value + 0.05)  # Increase size
            elif name == 'exploration_rate':
                self.parameters[name] = max(0.05, old_value - 0.005)  # Less exploration
            elif name in ['stop_loss_pct', 'take_profit_pct']:
                self.parameters[name] = min(0.1, old_value + 0.001)  # Wider stops/targets
        
        elif avg_outcome < -0.05:  # Poor performance
            if name == 'confidence_threshold':
                self.parameters[name] = min(0.9, old_value + 0.01)  # Higher threshold
            elif name == 'position_size_multiplier':
                self.parameters[name] = max(0.5, old_value - 0.05)  # Smaller size
            elif name == 'exploration_rate':
                self.parameters[name] = min(0.3, old_value + 0.005)  # More exploration
            elif name in ['stop_loss_pct', 'take_profit_pct']:
                self.parameters[name] = max(0.005, old_value - 0.001)  # Tighter stops/targets
        
        # Log significant changes
        if abs(self.parameters[name] - old_value) > old_value * 0.05:
            self.successful_adaptations += 1
            log.info(f"Adapted {name}: {old_value:.4f} → {self.parameters[name]:.4f} (outcome: {avg_outcome:.3f})")
        
        # Save to database
        self._save_parameter(name)
    
    def batch_update(self, parameter_outcomes: Dict[str, float]):
        """Update multiple parameters from a single trade"""
        for param_name, outcome in parameter_outcomes.items():
            self.update_parameter(param_name, outcome)
    
    def _save_parameter(self, name):
        """Save parameter to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO parameters (name, value, last_updated, total_updates)
            VALUES (?, ?, ?, ?)
        ''', (name, self.parameters[name], datetime.now().isoformat(), self.total_updates))
        
        # Save recent outcomes
        recent_outcomes = list(self.outcomes[name])[-5:]  # Last 5 outcomes
        for outcome in recent_outcomes:
            cursor.execute('''
                INSERT INTO outcomes (parameter_name, outcome, timestamp)
                VALUES (?, ?, ?)
            ''', (name, outcome, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def load_parameters(self):
        """Load parameters from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT name, value FROM parameters')
            rows = cursor.fetchall()
            
            for name, value in rows:
                if name in self.parameters:
                    self.parameters[name] = value
            
            conn.close()
            
            if rows:
                log.info(f"Loaded {len(rows)} parameters from database")
        except Exception as e:
            log.info(f"Starting with default parameters: {e}")
    
    def save_all(self):
        """Save all parameters"""
        for name in self.parameters:
            self._save_parameter(name)
        log.info("All parameters saved")
    
    def get_learning_efficiency(self):
        """Simple learning efficiency metric"""
        if self.total_updates == 0:
            return 0.0
        return self.successful_adaptations / self.total_updates
    
    def get_network_config(self):
        """Get neural network configuration"""
        return {
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.1,
            'learning_rate': self.get_parameter('learning_rate')
        }
    
    def get_status_report(self):
        """Get readable status report"""
        report = f"""
Meta-Learning Status:
  Total Updates: {self.total_updates}
  Successful Adaptations: {self.successful_adaptations}
  Learning Efficiency: {self.get_learning_efficiency():.2%}

Current Parameters:
"""
        for name, value in self.parameters.items():
            num_outcomes = len(self.outcomes[name])
            recent_perf = np.mean(list(self.outcomes[name])[-5:]) if num_outcomes >= 5 else 0.0
            report += f"  {name}: {value:.4f} (samples: {num_outcomes}, recent: {recent_perf:.3f})\n"
        
        return report

# Factory function
def create_meta_learner():
    return SimplifiedMetaLearner()