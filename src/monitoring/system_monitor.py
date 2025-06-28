# system_monitor.py

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

class SystemMonitor:
    def __init__(self, update_interval: int = 30):
        self.update_interval = update_interval
        self.last_update = 0
        
    def get_system_status(self) -> Dict[str, Any]:
        status = {
            'timestamp': datetime.now().isoformat(),
            'meta_learning': self._get_meta_learning_status(),
            'trading_performance': self._get_trading_performance(),
            'intelligence_status': self._get_intelligence_status(),
            'architecture_status': self._get_architecture_status()
        }
        
        return status
    
    def _get_meta_learning_status(self) -> Dict[str, Any]:
        try:
            # This would be called from the running system
            # For monitoring, we'd need to expose these stats
            return {
                'parameters_learned': 8,  # Number of adaptive parameters
                'learning_efficiency': 0.0,  # Would come from meta_learner
                'successful_adaptations': 0,  # Would come from meta_learner
                'current_exploration_strategy': 'learned',
                'adaptive_components': {
                    'subsystem_weights': [0.25, 0.25, 0.25, 0.25],  # Would be dynamic
                    'reward_components': {},  # Would come from reward engine
                    'parameter_values': {}  # Would come from meta_learner
                }
            }
        except Exception:
            return {'status': 'unavailable'}
    
    def _get_trading_performance(self) -> Dict[str, Any]:
        try:
            # Would read from portfolio state
            portfolio_file = Path('data/portfolio.json')
            if portfolio_file.exists():
                with open(portfolio_file) as f:
                    portfolio_data = json.load(f)
                
                return {
                    'total_trades': portfolio_data.get('winning_trades', 0) + portfolio_data.get('losing_trades', 0),
                    'win_rate': portfolio_data.get('winning_trades', 0) / max(1, portfolio_data.get('winning_trades', 0) + portfolio_data.get('losing_trades', 0)),
                    'total_pnl': portfolio_data.get('total_pnl', 0.0),
                    'daily_pnl': portfolio_data.get('daily_pnl', 0.0),
                    'consecutive_losses': portfolio_data.get('consecutive_losses', 0)
                }
        except Exception:
            pass
        
        return {'status': 'no_data'}
    
    def _get_intelligence_status(self) -> Dict[str, Any]:
        try:
            # Would read from intelligence patterns
            patterns_file = Path('data/patterns.json')
            if patterns_file.exists():
                with open(patterns_file) as f:
                    patterns_data = json.load(f)
                
                return {
                    'dna_patterns': len(patterns_data.get('dna_patterns', {})),
                    'micro_patterns': len(patterns_data.get('micro_patterns', {})),
                    'temporal_patterns': len(patterns_data.get('temporal_patterns', {})),
                    'immune_patterns': len(patterns_data.get('immune_patterns', [])),
                    'recent_performance': patterns_data.get('recent_performance', 0.0),
                    'pattern_evolution': 'active'
                }
        except Exception:
            pass
        
        return {'status': 'learning'}
    
    def _get_architecture_status(self) -> Dict[str, Any]:
        try:
            # Would come from architecture evolver
            return {
                'current_architecture': [64, 32],  # Would be dynamic
                'generation': 0,  # Would track evolution cycles
                'recent_performance': 0.0,
                'evolution_triggered': False,
                'feature_learning': {
                    'learned_features': 20,
                    'feature_importance_discovered': True,
                    'active_feature_selection': True
                }
            }
        except Exception:
            pass
        
        return {'status': 'stable'}
    
    def should_update(self) -> bool:
        current_time = time.time()
        if current_time - self.last_update >= self.update_interval:
            self.last_update = current_time
            return True
        return False
    
    def print_status(self):
        if not self.should_update():
            return
        
        status = self.get_system_status()
        
        print(f"\n=== Black-Box Trading System Status ===")
        print(f"Time: {status['timestamp']}")
        
        # Meta-learning status
        meta = status['meta_learning']
        if 'parameters_learned' in meta:
            print(f"\nMeta-Learning:")
            print(f"  Adaptive Parameters: {meta['parameters_learned']}")
            print(f"  Learning Efficiency: {meta['learning_efficiency']:.3f}")
            print(f"  Exploration: {meta['current_exploration_strategy']}")
        
        # Trading performance
        trading = status['trading_performance']
        if 'total_trades' in trading:
            print(f"\nTrading Performance:")
            print(f"  Total Trades: {trading['total_trades']}")
            print(f"  Win Rate: {trading['win_rate']:.1%}")
            print(f"  Daily P&L: ${trading['daily_pnl']:.2f}")
            print(f"  Total P&L: ${trading['total_pnl']:.2f}")
        
        # Intelligence status
        intel = status['intelligence_status']
        if 'dna_patterns' in intel:
            print(f"\nIntelligence Subsystems:")
            print(f"  DNA Patterns: {intel['dna_patterns']}")
            print(f"  Micro Patterns: {intel['micro_patterns']}")
            print(f"  Temporal Patterns: {intel['temporal_patterns']}")
            print(f"  Immune Patterns: {intel['immune_patterns']}")
        
        # Architecture status
        arch = status['architecture_status']
        if 'current_architecture' in arch:
            print(f"\nNeural Architecture:")
            print(f"  Current Structure: {arch['current_architecture']}")
            print(f"  Generation: {arch['generation']}")
            print(f"  Evolution Status: {'Active' if arch['evolution_triggered'] else 'Stable'}")
        
        print("=" * 40)

def main():
    monitor = SystemMonitor(update_interval=10)
    
    print("Black-Box Trading System Monitor")
    print("Press Ctrl+C to exit")
    
    try:
        while True:
            monitor.print_status()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nMonitoring stopped")

if __name__ == "__main__":
    main()