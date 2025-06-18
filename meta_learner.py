# meta_learner.py - FIXED: Added missing get_learning_efficiency method

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
import logging
import json
import sqlite3
from collections import deque, defaultdict
import threading
import time

log = logging.getLogger(__name__)

class MetaParameterDatabase:
    """Permanent storage for meta-learned parameters"""
    
    def __init__(self, db_path: str = "meta_parameters.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize meta-parameter storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS meta_parameters (
                param_name TEXT PRIMARY KEY,
                current_value REAL,
                learning_rate REAL,
                gradient REAL,
                last_updated TEXT,
                performance_history TEXT,
                bounds_min REAL,
                bounds_max REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reward_components (
                component_name TEXT PRIMARY KEY,
                weight REAL,
                correlation_history TEXT,
                last_updated TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS architecture_history (
                timestamp TEXT,
                hidden_size INTEGER,
                lstm_layers INTEGER,
                performance_score REAL,
                parameters_json TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_meta_parameter(self, name: str, value: float, gradient: float, 
                          performance_history: List[float], bounds: Tuple[float, float]):
        """Save meta-parameter state"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO meta_parameters 
            (param_name, current_value, gradient, last_updated, performance_history, bounds_min, bounds_max)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            name, value, gradient, datetime.now().isoformat(),
            json.dumps(performance_history), bounds[0], bounds[1]
        ))
        
        conn.commit()
        conn.close()
    
    def load_meta_parameters(self) -> Dict[str, Dict]:
        """Load all meta-parameters from storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM meta_parameters')
        rows = cursor.fetchall()
        
        parameters = {}
        for row in rows:
            name, value, lr, gradient, last_updated, perf_hist, bounds_min, bounds_max = row
            parameters[name] = {
                'value': value,
                'gradient': gradient,
                'performance_history': json.loads(perf_hist),
                'bounds': (bounds_min, bounds_max),
                'last_updated': datetime.fromisoformat(last_updated)
            }
        
        conn.close()
        return parameters

class PureMetaLearner:
    """
    Pure meta-learner that optimizes ALL system parameters
    No hardcoded values - everything learned from experience
    """
    
    def __init__(self, db_path: str = "meta_parameters.db"):
        self.db = MetaParameterDatabase(db_path)
        
        # Load existing parameters or initialize defaults
        saved_params = self.db.load_meta_parameters()
        
        # Core parameters that MUST be learned (no defaults)
        self.parameter_definitions = {
            # Risk Management
            'max_daily_loss_pct': {'initial': 0.02, 'bounds': (0.005, 0.20), 'lr': 1e-5},
            'max_consecutive_losses': {'initial': 5.0, 'bounds': (2.0, 15.0), 'lr': 1e-4},
            'position_size_base': {'initial': 0.5, 'bounds': (0.1, 2.0), 'lr': 1e-4},
            'stop_loss_max_pct': {'initial': 0.02, 'bounds': (0.005, 0.08), 'lr': 1e-5},
            'take_profit_max_pct': {'initial': 0.04, 'bounds': (0.01, 0.15), 'lr': 1e-5},
            
            # Learning Rates
            'policy_learning_rate': {'initial': 1e-4, 'bounds': (1e-6, 1e-2), 'lr': 1e-6},
            'value_learning_rate': {'initial': 3e-4, 'bounds': (1e-6, 1e-2), 'lr': 1e-6},
            'meta_learning_rate': {'initial': 1e-5, 'bounds': (1e-7, 1e-3), 'lr': 1e-7},
            
            # Confidence Thresholds
            'entry_confidence_threshold': {'initial': 0.5, 'bounds': (0.1, 0.9), 'lr': 1e-4},
            'exit_confidence_threshold': {'initial': 0.5, 'bounds': (0.1, 0.9), 'lr': 1e-4},
            'scaling_confidence_threshold': {'initial': 0.6, 'bounds': (0.2, 0.9), 'lr': 1e-4},
            
            # Exploration
            'epsilon_decay_rate': {'initial': 0.9995, 'bounds': (0.99, 0.9999), 'lr': 1e-6},
            'epsilon_min': {'initial': 0.15, 'bounds': (0.05, 0.5), 'lr': 1e-4},
            'exploration_bonus': {'initial': 0.1, 'bounds': (0.0, 1.0), 'lr': 1e-4},
            
            # Network Architecture
            'hidden_layer_multiplier': {'initial': 1.0, 'bounds': (0.5, 3.0), 'lr': 1e-4},
            'lstm_layers': {'initial': 1.0, 'bounds': (1.0, 4.0), 'lr': 1e-4},
            'attention_weight': {'initial': 1.0, 'bounds': (0.1, 3.0), 'lr': 1e-4},
            'dropout_rate': {'initial': 0.1, 'bounds': (0.0, 0.5), 'lr': 1e-4},
            
            # Memory Management
            'experience_buffer_size': {'initial': 10000, 'bounds': (1000, 100000), 'lr': 1e-3},
            'batch_size_multiplier': {'initial': 1.0, 'bounds': (0.5, 4.0), 'lr': 1e-4},
            
            # Advanced Features
            'scaling_frequency': {'initial': 0.3, 'bounds': (0.1, 0.8), 'lr': 1e-4},
            'partial_exit_frequency': {'initial': 0.2, 'bounds': (0.05, 0.6), 'lr': 1e-4},
            'hold_time_preference': {'initial': 0.5, 'bounds': (0.1, 2.0), 'lr': 1e-4},
        }
        
        # Initialize parameters
        self.parameters = {}
        self.parameter_gradients = {}
        self.parameter_outcomes = {}
        
        for name, config in self.parameter_definitions.items():
            if name in saved_params:
                # Load from database
                saved = saved_params[name]
                self.parameters[name] = saved['value']
                self.parameter_gradients[name] = saved['gradient']
                self.parameter_outcomes[name] = deque(saved['performance_history'], maxlen=200)
            else:
                # Initialize fresh
                self.parameters[name] = config['initial']
                self.parameter_gradients[name] = 0.0
                self.parameter_outcomes[name] = deque(maxlen=200)
        
        # Meta-learning state
        self.learning_efficiency_history = deque(maxlen=100)
        self.parameter_update_history = defaultdict(lambda: deque(maxlen=50))
        self.last_save_time = datetime.now()
        self.total_updates = 0
        
        # Background saving
        self._start_background_saver()
        
        log.info("PURE META-LEARNER: Initialized with zero hardcoded knowledge")
        log.info(f"Managing {len(self.parameters)} adaptive parameters")
        self._log_parameter_summary()
    
    def get_parameter(self, name: str) -> float:
        """Get current value of meta-learned parameter"""
        return self.parameters.get(name, 0.5)
    
    def update_parameter(self, name: str, outcome: float, context: Dict = None):
        """Update parameter based on outcome using gradient estimation"""
        
        if name not in self.parameters:
            log.warning(f"Unknown parameter: {name}")
            return
        
        # Store outcome with context
        outcome_data = {
            'outcome': outcome,
            'value': self.parameters[name],
            'context': context or {},
            'timestamp': datetime.now()
        }
        self.parameter_outcomes[name].append(outcome_data)
        
        # Need sufficient data for gradient estimation
        if len(self.parameter_outcomes[name]) < 10:
            return
        
        # Calculate gradient using recent outcomes
        recent_outcomes = list(self.parameter_outcomes[name])[-20:]
        
        # Finite difference gradient approximation
        values = [o['value'] for o in recent_outcomes]
        outcomes = [o['outcome'] for o in recent_outcomes]
        
        if len(set(values)) > 1:  # Need parameter variation for gradient
            try:
                # Calculate correlation as gradient proxy
                correlation = np.corrcoef(values, outcomes)[0, 1]
                if not np.isnan(correlation):
                    self.parameter_gradients[name] = correlation
                    
                    # Get learning rate and bounds
                    config = self.parameter_definitions[name]
                    learning_rate = config['lr']
                    bounds = config['bounds']
                    
                    # Adaptive learning rate based on recent performance
                    if len(self.learning_efficiency_history) > 5:
                        recent_efficiency = np.mean(list(self.learning_efficiency_history)[-5:])
                        learning_rate *= (1.0 + recent_efficiency)  # Boost if learning well
                    
                    # Update parameter
                    old_value = self.parameters[name]
                    gradient_step = learning_rate * correlation * abs(old_value)
                    
                    # Add exploration noise
                    exploration_noise = np.random.normal(0, learning_rate * 0.1)
                    
                    self.parameters[name] += gradient_step + exploration_noise
                    
                    # Clamp to bounds
                    self.parameters[name] = max(bounds[0], min(bounds[1], self.parameters[name]))
                    
                    # Track update
                    self.parameter_update_history[name].append({
                        'old_value': old_value,
                        'new_value': self.parameters[name],
                        'gradient': correlation,
                        'outcome': outcome,
                        'timestamp': datetime.now()
                    })
                    
                    self.total_updates += 1
                    
                    # Log significant changes
                    if abs(correlation) > 0.2 and abs(old_value - self.parameters[name]) > old_value * 0.01:
                        log.info(f"META-LEARNING: {name} {old_value:.6f} → {self.parameters[name]:.6f} "
                               f"(grad: {correlation:.4f}, outcome: {outcome:.4f})")
                        
            except Exception as e:
                log.warning(f"Gradient calculation error for {name}: {e}")
    
    def batch_update_parameters(self, outcomes_dict: Dict[str, float], learning_efficiency: float):
        """Update multiple parameters from a single trade outcome"""
        
        # Store learning efficiency
        self.learning_efficiency_history.append(learning_efficiency)
        
        # Update each parameter that has an outcome
        for param_name, outcome in outcomes_dict.items():
            self.update_parameter(param_name, outcome)
        
        # Meta-learning: adjust learning rates based on efficiency
        if len(self.learning_efficiency_history) >= 20:
            recent_efficiency = np.mean(list(self.learning_efficiency_history)[-10:])
            older_efficiency = np.mean(list(self.learning_efficiency_history)[-20:-10])
            
            efficiency_trend = recent_efficiency - older_efficiency
            
            # If learning efficiency is improving, can be more aggressive
            if efficiency_trend > 0.1:
                for name in self.parameter_definitions:
                    if 'learning_rate' in name:
                        self.update_parameter(name, 0.1)  # Boost learning rates
                        
            elif efficiency_trend < -0.1:
                for name in self.parameter_definitions:
                    if 'learning_rate' in name:
                        self.update_parameter(name, -0.1)  # Reduce learning rates
    
    def get_learning_efficiency(self) -> float:
        """Get current learning efficiency"""
        if len(self.learning_efficiency_history) > 0:
            return float(np.mean(list(self.learning_efficiency_history)[-10:]))
        return 0.0
    
    def get_network_architecture(self) -> Dict[str, int]:
        """Get current optimal network architecture"""
        base_hidden = 64
        multiplier = self.get_parameter('hidden_layer_multiplier')
        
        return {
            'hidden_size': max(32, int(base_hidden * multiplier)),
            'lstm_layers': max(1, int(self.get_parameter('lstm_layers'))),
            'attention_layers': max(1, int(self.get_parameter('attention_weight'))),
            'dropout_rate': self.get_parameter('dropout_rate'),
        }
    
    def get_risk_parameters(self) -> Dict[str, float]:
        """Get current risk management parameters"""
        return {
            'max_daily_loss_pct': self.get_parameter('max_daily_loss_pct'),
            'max_consecutive_losses': int(self.get_parameter('max_consecutive_losses')),
            'position_size_base': self.get_parameter('position_size_base'),
            'stop_loss_max_pct': self.get_parameter('stop_loss_max_pct'),
            'take_profit_max_pct': self.get_parameter('take_profit_max_pct'),
        }
    
    def get_learning_parameters(self) -> Dict[str, float]:
        """Get current learning parameters"""
        return {
            'policy_lr': self.get_parameter('policy_learning_rate'),
            'value_lr': self.get_parameter('value_learning_rate'),
            'meta_lr': self.get_parameter('meta_learning_rate'),
            'epsilon_decay': self.get_parameter('epsilon_decay_rate'),
            'epsilon_min': self.get_parameter('epsilon_min'),
            'batch_size': max(16, int(32 * self.get_parameter('batch_size_multiplier'))),
            'buffer_size': int(self.get_parameter('experience_buffer_size')),
        }
    
    def get_confidence_thresholds(self) -> Dict[str, float]:
        """Get current confidence thresholds"""
        return {
            'entry': self.get_parameter('entry_confidence_threshold'),
            'exit': self.get_parameter('exit_confidence_threshold'),
            'scaling': self.get_parameter('scaling_confidence_threshold'),
        }
    
    def should_rebuild_network(self) -> bool:
        """Determine if network architecture should be rebuilt"""
        
        # Check if architecture parameters changed significantly
        arch_params = ['hidden_layer_multiplier', 'lstm_layers', 'attention_weight', 'dropout_rate']
        
        for param in arch_params:
            updates = list(self.parameter_update_history[param])
            if len(updates) > 0:
                recent_update = updates[-1]
                change_pct = abs(recent_update['new_value'] - recent_update['old_value']) / recent_update['old_value']
                
                if change_pct > 0.1:  # 10% change triggers rebuild
                    log.info(f"ARCHITECTURE REBUILD: {param} changed by {change_pct:.1%}")
                    return True
        
        return False
    
    def _log_parameter_summary(self):
        """Log current parameter state"""
        log.info("Current Meta-Parameters:")
        
        # Group parameters by category
        categories = {
            'Risk Management': ['max_daily_loss_pct', 'max_consecutive_losses', 'position_size_base'],
            'Learning': ['policy_learning_rate', 'value_learning_rate', 'entry_confidence_threshold'],
            'Architecture': ['hidden_layer_multiplier', 'lstm_layers', 'attention_weight'],
        }
        
        for category, param_names in categories.items():
            log.info(f"  {category}:")
            for name in param_names:
                if name in self.parameters:
                    value = self.parameters[name]
                    gradient = self.parameter_gradients[name]
                    log.info(f"    {name}: {value:.6f} (grad: {gradient:.4f})")
    
    def _start_background_saver(self):
        """Start background thread to save parameters"""
        def save_loop():
            while True:
                try:
                    time.sleep(300)  # Save every 5 minutes
                    
                    if datetime.now() - self.last_save_time > timedelta(minutes=5):
                        self._save_all_parameters()
                        self.last_save_time = datetime.now()
                        
                except Exception as e:
                    log.error(f"Background save error: {e}")
        
        thread = threading.Thread(target=save_loop, daemon=True, name="MetaParameterSaver")
        thread.start()
        log.info("Background meta-parameter saver started")
    
    def _save_all_parameters(self):
        """Save all parameters to database"""
        for name, value in self.parameters.items():
            if name in self.parameter_definitions:
                config = self.parameter_definitions[name]
                gradient = self.parameter_gradients[name]
                performance_history = list(self.parameter_outcomes[name])[-50:]  # Save recent history
                
                outcome_values = [o['outcome'] for o in performance_history]
                bounds = config['bounds']
                
                self.db.save_meta_parameter(name, value, gradient, outcome_values, bounds)
        
        log.debug(f"META-LEARNER: Saved {len(self.parameters)} parameters to database")
    
    def force_save(self):
        """Force immediate save of all parameters"""
        self._save_all_parameters()
        log.info("META-LEARNER: Forced save completed")
    
    def get_adaptation_report(self) -> str:
        """Generate detailed adaptation report"""
        
        recent_efficiency = self.get_learning_efficiency()
        
        report = f"""
=== PURE BLACK BOX META-LEARNING REPORT ===

Total Parameter Updates: {self.total_updates}
Learning Efficiency: {recent_efficiency:.3f} (recent)

RISK MANAGEMENT (Self-Optimized):
"""
        
        risk_params = self.get_risk_parameters()
        for name, value in risk_params.items():
            gradient = self.parameter_gradients.get(name.replace('_pct', '_pct').replace('max_', 'max_'), 0.0)
            updates = len(self.parameter_update_history.get(name, []))
            report += f"  {name}: {value:.6f} (gradient: {gradient:.4f}, updates: {updates})\n"
        
        report += f"\nLEARNING PARAMETERS (Self-Adapted):\n"
        learning_params = self.get_learning_parameters()
        for name, value in learning_params.items():
            original_name = name + '_learning_rate' if name.endswith('_lr') else name
            gradient = self.parameter_gradients.get(original_name, 0.0)
            report += f"  {name}: {value:.6f} (gradient: {gradient:.4f})\n"
        
        report += f"\nCONFIDENCE THRESHOLDS (Experience-Based):\n"
        conf_params = self.get_confidence_thresholds()
        for name, value in conf_params.items():
            threshold_name = f"{name}_confidence_threshold"
            gradient = self.parameter_gradients.get(threshold_name, 0.0)
            samples = len(self.parameter_outcomes.get(threshold_name, []))
            report += f"  {name}: {value:.3f} (gradient: {gradient:.4f}, samples: {samples})\n"
        
        report += f"\nNETWORK ARCHITECTURE (Adaptive):\n"
        arch = self.get_network_architecture()
        for name, value in arch.items():
            report += f"  {name}: {value}\n"
        
        # Recent significant adaptations
        report += f"\nRECENT ADAPTATIONS:\n"
        recent_adaptations = 0
        for param_name, updates in self.parameter_update_history.items():
            recent_updates = [u for u in updates if 
                            (datetime.now() - u['timestamp']).total_seconds() < 3600]  # Last hour
            
            for update in recent_updates[-3:]:  # Last 3 updates
                change_pct = abs(update['new_value'] - update['old_value']) / update['old_value'] * 100
                if change_pct > 1:  # Significant change
                    report += f"  {param_name}: {change_pct:.1f}% change (outcome: {update['outcome']:.3f})\n"
                    recent_adaptations += 1
        
        if recent_adaptations == 0:
            report += "  No significant adaptations in last hour\n"
        
        report += f"\nALL PARAMETERS OPTIMIZING THROUGH PURE EXPERIENCE!"
        
        return report

class AdaptiveRewardLearner:
    """
    Learns optimal reward structure - discovers what actually matters for trading success
    """
    
    def __init__(self, meta_learner: PureMetaLearner):
        self.meta_learner = meta_learner
        
        # Reward components that discover their own importance
        self.reward_components = {
            'raw_pnl': {'weight': 1.0, 'correlation_history': deque(maxlen=100)},
            'risk_adjusted_pnl': {'weight': 0.0, 'correlation_history': deque(maxlen=100)},
            'sharpe_ratio': {'weight': 0.0, 'correlation_history': deque(maxlen=100)},
            'max_drawdown_control': {'weight': 0.0, 'correlation_history': deque(maxlen=100)},
            'consistency': {'weight': 0.0, 'correlation_history': deque(maxlen=100)},
            'hold_time_efficiency': {'weight': 0.0, 'correlation_history': deque(maxlen=100)},
            'tool_effectiveness': {'weight': 0.0, 'correlation_history': deque(maxlen=100)},
            'risk_management_usage': {'weight': 0.0, 'correlation_history': deque(maxlen=100)},
            'scaling_success': {'weight': 0.0, 'correlation_history': deque(maxlen=100)},
            'exit_timing': {'weight': 0.0, 'correlation_history': deque(maxlen=100)},
        }
        
        # Track reward learning
        self.reward_history = deque(maxlen=500)
        self.component_effectiveness = {}
        
    def calculate_adaptive_reward(self, trade_data: Dict) -> float:
        """Calculate reward using learned component weights"""
        
        pnl = trade_data.get('pnl', 0.0)
        
        # Calculate all reward components
        components = self._calculate_all_components(trade_data, pnl)
        
        # Combine using learned weights
        total_reward = 0.0
        component_contributions = {}
        
        for component_name, value in components.items():
            weight = self.reward_components[component_name]['weight']
            contribution = weight * value
            total_reward += contribution
            component_contributions[component_name] = contribution
            
            # Store for correlation learning
            self.reward_components[component_name]['correlation_history'].append({
                'component_value': value,
                'actual_pnl': pnl,
                'final_reward': total_reward,
                'timestamp': datetime.now()
            })
        
        # Learn component weights
        self._update_component_weights(components, pnl)
        
        # Store reward
        self.reward_history.append({
            'total_reward': total_reward,
            'pnl': pnl,
            'components': components.copy(),
            'contributions': component_contributions.copy(),
            'timestamp': datetime.now()
        })
        
        return total_reward
    
    def _calculate_all_components(self, trade_data: Dict, pnl: float) -> Dict[str, float]:
        """Calculate all reward components"""
        
        components = {}
        
        # Raw P&L (baseline)
        components['raw_pnl'] = pnl / 50.0  # Normalize for MNQ
        
        # Risk-adjusted P&L
        max_risk = trade_data.get('max_risk_taken', 50.0)
        components['risk_adjusted_pnl'] = pnl / max(max_risk, 10.0)
        
        # Sharpe-like ratio (return vs volatility)
        volatility = trade_data.get('price_volatility', 20.0)
        components['sharpe_ratio'] = pnl / max(volatility, 5.0)
        
        # Drawdown control
        max_drawdown = trade_data.get('max_drawdown_pct', 0.0)
        components['max_drawdown_control'] = -max_drawdown * 100  # Penalize drawdown
        
        # Consistency (reward predictable profits)
        recent_pnls = [r['pnl'] for r in list(self.reward_history)[-10:]]
        if len(recent_pnls) >= 3:
            pnl_std = np.std(recent_pnls)
            components['consistency'] = -pnl_std / 50.0  # Penalize volatility
        else:
            components['consistency'] = 0.0
        
        # Hold time efficiency
        hold_time_hours = trade_data.get('hold_time_hours', 1.0)
        if pnl > 0:
            components['hold_time_efficiency'] = pnl / max(hold_time_hours, 0.1)  # Profit per hour
        else:
            components['hold_time_efficiency'] = pnl * hold_time_hours  # Penalize long losses
        
        # Tool effectiveness
        tool_confidence = trade_data.get('tool_confidence', 0.5)
        if pnl > 0:
            components['tool_effectiveness'] = tool_confidence * 2.0 - 1.0  # -1 to +1
        else:
            components['tool_effectiveness'] = -(tool_confidence * 2.0 - 1.0)
        
        # Risk management usage
        used_stop = trade_data.get('used_stop', False)
        used_target = trade_data.get('used_target', False)
        stop_effectiveness = 1.0 if (used_stop and pnl > -30) else 0.0
        target_effectiveness = 1.0 if (used_target and pnl > 20) else 0.0
        components['risk_management_usage'] = stop_effectiveness + target_effectiveness
        
        # Scaling success
        scales_added = trade_data.get('scales_added', 0)
        if scales_added > 0:
            components['scaling_success'] = (pnl / scales_added) / 50.0 if pnl > 0 else -scales_added * 0.1
        else:
            components['scaling_success'] = 0.0
        
        # Exit timing
        exit_reason = trade_data.get('exit_reason', 'unknown')
        if exit_reason == 'target_hit':
            components['exit_timing'] = 0.5
        elif exit_reason == 'stop_hit' and pnl > -30:
            components['exit_timing'] = 0.3
        elif exit_reason == 'manual_exit' and pnl > 0:
            components['exit_timing'] = 0.2
        else:
            components['exit_timing'] = -0.1
        
        return components
    
    def _update_component_weights(self, components: Dict[str, float], actual_pnl: float):
        """Learn which components actually predict trading success"""
        
        learning_rate = self.meta_learner.get_parameter('meta_learning_rate') * 100
        
        for component_name, component_value in components.items():
            history = list(self.reward_components[component_name]['correlation_history'])
            
            if len(history) >= 20:
                # Calculate correlation with actual PnL
                component_values = [h['component_value'] for h in history[-20:]]
                actual_pnls = [h['actual_pnl'] for h in history[-20:]]
                
                if len(set(component_values)) > 1 and len(set(actual_pnls)) > 1:
                    try:
                        correlation = np.corrcoef(component_values, actual_pnls)[0, 1]
                        
                        if not np.isnan(correlation):
                            # Update weight based on correlation
                            old_weight = self.reward_components[component_name]['weight']
                            
                            # Positive correlation = increase weight
                            weight_update = learning_rate * correlation
                            self.reward_components[component_name]['weight'] += weight_update
                            
                            # Clamp weights
                            self.reward_components[component_name]['weight'] = max(
                                -2.0, min(2.0, self.reward_components[component_name]['weight'])
                            )
                            
                            # Log significant changes
                            if abs(correlation) > 0.3 and abs(weight_update) > 0.01:
                                log.info(f"REWARD LEARNING: {component_name} weight "
                                       f"{old_weight:.3f} → {self.reward_components[component_name]['weight']:.3f} "
                                       f"(correlation: {correlation:.3f})")
                    
                    except Exception as e:
                        log.warning(f"Correlation calculation error for {component_name}: {e}")
    
    def get_reward_analysis(self) -> str:
        """Get analysis of learned reward structure"""
        
        analysis = f"ADAPTIVE REWARD STRUCTURE ANALYSIS:\n\n"
        
        # Sort components by absolute weight
        sorted_components = sorted(
            self.reward_components.items(),
            key=lambda x: abs(x[1]['weight']),
            reverse=True
        )
        
        analysis += "Component Importance (Learned):\n"
        for component, data in sorted_components:
            weight = data['weight']
            samples = len(data['correlation_history'])
            
            # Calculate recent correlation
            recent_correlation = 0.0
            if len(data['correlation_history']) >= 10:
                recent_values = [h['component_value'] for h in list(data['correlation_history'])[-10:]]
                recent_pnls = [h['actual_pnl'] for h in list(data['correlation_history'])[-10:]]
                
                if len(set(recent_values)) > 1:
                    try:
                        recent_correlation = np.corrcoef(recent_values, recent_pnls)[0, 1]
                        if np.isnan(recent_correlation):
                            recent_correlation = 0.0
                    except:
                        recent_correlation = 0.0
            
            importance = "HIGH" if abs(weight) > 0.5 else "MEDIUM" if abs(weight) > 0.2 else "LOW"
            direction = "POSITIVE" if weight > 0 else "NEGATIVE" if weight < 0 else "NEUTRAL"
            
            analysis += f"  {component}: {weight:.3f} ({importance}, {direction}) "
            analysis += f"[correlation: {recent_correlation:.3f}, samples: {samples}]\n"
        
        # Recent reward statistics
        if len(self.reward_history) >= 10:
            recent_rewards = [r['total_reward'] for r in list(self.reward_history)[-10:]]
            recent_pnls = [r['pnl'] for r in list(self.reward_history)[-10:]]
            
            analysis += f"\nRecent Performance:\n"
            analysis += f"  Average Reward: {np.mean(recent_rewards):.3f}\n"
            analysis += f"  Average PnL: ${np.mean(recent_pnls):.2f}\n"
            if len(set(recent_rewards)) > 1 and len(set(recent_pnls)) > 1:
                try:
                    correlation = np.corrcoef(recent_rewards, recent_pnls)[0, 1]
                    if not np.isnan(correlation):
                        analysis += f"  Reward-PnL Correlation: {correlation:.3f}\n"
                except:
                    pass
        
        analysis += f"\nReward system learning what actually drives trading success!"
        
        return analysis

# Integration example
def integrate_meta_learning_with_existing_system():
    """Example of how to integrate meta-learning with existing components"""
    
    # Initialize meta-learner
    meta_learner = PureMetaLearner()
    reward_learner = AdaptiveRewardLearner(meta_learner)
    
    # Example usage in trade execution
    def execute_trade_with_meta_learning(market_data, subsystem_features):
        
        # Get adaptive parameters
        risk_params = meta_learner.get_risk_parameters()
        confidence_thresholds = meta_learner.get_confidence_thresholds()
        
        # Use adaptive position sizing
        position_size = risk_params['position_size_base']
        
        # Use adaptive confidence threshold
        entry_threshold = confidence_thresholds['entry']
        
        # ... rest of trading logic using adaptive parameters
        
        return {
            'position_size': position_size,
            'entry_threshold': entry_threshold,
            'risk_params': risk_params
        }
    
    # Example of recording outcomes for learning
    def record_trade_outcome_for_meta_learning(trade_data, outcome_pnl):
        
        # Calculate adaptive reward
        adaptive_reward = reward_learner.calculate_adaptive_reward(trade_data)
        
        # Update meta-parameters based on outcome
        parameter_outcomes = {
            'position_size_base': outcome_pnl / 100.0,  # Normalize
            'entry_confidence_threshold': adaptive_reward,
            'stop_loss_max_pct': adaptive_reward if trade_data.get('used_stop') else 0.0,
        }
        
        # Calculate learning efficiency
        learning_efficiency = adaptive_reward  # Simplified
        
        # Batch update parameters
        meta_learner.batch_update_parameters(parameter_outcomes, learning_efficiency)
        
        return adaptive_reward
    
    return meta_learner, reward_learner

if __name__ == "__main__":
    # Test the meta-learning system
    print("Testing Pure Black Box Meta-Learning System...")
    
    meta_learner = PureMetaLearner()
    reward_learner = AdaptiveRewardLearner(meta_learner)
    
    print(f"\nInitial state:")
    print(meta_learner.get_adaptation_report())
    
    # Simulate some trading outcomes
    for i in range(50):
        # Simulate trade data
        trade_data = {
            'pnl': np.random.normal(5, 30),  # Random P&L
            'hold_time_hours': np.random.uniform(0.5, 6.0),
            'used_stop': np.random.choice([True, False]),
            'used_target': np.random.choice([True, False]),
            'tool_confidence': np.random.uniform(0.3, 0.9),
            'max_drawdown_pct': np.random.uniform(0.005, 0.03),
            'scales_added': np.random.choice([0, 0, 0, 1, 2]),
            'exit_reason': np.random.choice(['target_hit', 'stop_hit', 'manual_exit', 'time_exit'])
        }
        
        # Calculate adaptive reward and update parameters
        adaptive_reward = reward_learner.calculate_adaptive_reward(trade_data)
        
        # Simulate parameter updates
        meta_learner.update_parameter('position_size_base', adaptive_reward)
        meta_learner.update_parameter('entry_confidence_threshold', adaptive_reward)
        
        if i % 10 == 9:
            print(f"\nAfter {i+1} trades:")
            print(f"Position size: {meta_learner.get_parameter('position_size_base'):.3f}")
            print(f"Entry threshold: {meta_learner.get_parameter('entry_confidence_threshold'):.3f}")
    
    print(f"\nFinal adaptation report:")
    print(meta_learner.get_adaptation_report())
    print(f"\nReward analysis:")
    print(reward_learner.get_reward_analysis())