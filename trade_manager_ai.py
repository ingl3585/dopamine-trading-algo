# trade_manager_ai.py

import logging
import time
import numpy as np
from datetime import datetime
from typing import Dict, Optional

log = logging.getLogger(__name__)

class SimpleTradeManager:
    """
    Simplified trade manager that coordinates between RL agent and trading system
    - Handles position sizing based on account data
    - Manages safety limits  
    - Coordinates signal generation
    - Tracks trade outcomes for learning
    """
    
    def __init__(self, rl_agent, intelligence_engine, tcp_bridge, config):
        self.rl_agent = rl_agent
        self.intelligence_engine = intelligence_engine
        self.tcp_bridge = tcp_bridge
        self.config = config
        
        # Current state
        self.current_account_data = {}
        self.current_price = 4000.0
        self.last_decision = None
        
        # Safety tracking
        self.daily_signals = 0
        self.consecutive_losses = 0
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.now().date()
        
        # Trade tracking for learning correlation
        self.pending_trades = {}  # Tracks signals waiting for outcomes
        self.trade_counter = 0
        
        # Performance statistics
        self.stats = {
            'total_signals': 0,
            'successful_trades': 0,
            'total_pnl': 0.0,
            'signals_today': 0,
            'avg_trade_duration': 0.0,
            'best_tool': 'unknown',
            'tool_performance': {'dna': [], 'micro': [], 'temporal': [], 'immune': []}
        }
        
        log.info("Simple trade manager initialized")
    
    def process_market_data(self, market_data: Dict) -> bool:
        """
        Process new market data and potentially generate trading signal
        Returns True if signal was sent
        """
        try:
            # Update account data
            self._update_account_data(market_data)
            
            # Reset daily counters if new day
            self._check_daily_reset()
            
            # Safety checks
            if not self._safety_check():
                return False
            
            # Extract price data
            prices_1m = market_data.get('price_1m', [])
            volumes_1m = market_data.get('volume_1m', [])
            
            if not prices_1m:
                return False
            
            self.current_price = prices_1m[-1]
            
            # Process through intelligence engine
            intelligence_result = self.intelligence_engine.process_market_data(
                prices_1m, volumes_1m, datetime.now()
            )
            
            # Check if we should make a trading decision
            if self._should_generate_signal(intelligence_result):
                return self._generate_trading_signal(market_data, intelligence_result)
            
            return False
            
        except Exception as e:
            log.error(f"Error processing market data: {e}")
            return False
    
    def _update_account_data(self, market_data: Dict):
        """Update current account data"""
        self.current_account_data = {
            'account_balance': market_data.get('account_balance', 25000),
            'buying_power': market_data.get('buying_power', 25000),
            'daily_pnl': market_data.get('daily_pnl', 0.0),
            'cash_value': market_data.get('cash_value', 25000),
            'timestamp': time.time()
        }
    
    def _check_daily_reset(self):
        """Reset daily counters if new day"""
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            log.info(f"New day - Yesterday: {self.daily_signals} signals, ${self.daily_pnl:.2f} P&L")
            
            self.daily_signals = 0
            self.daily_pnl = 0.0
            self.stats['signals_today'] = 0
            self.last_reset_date = current_date
            
            # Don't reset consecutive_losses - that persists across days
    
    def _safety_check(self) -> bool:
        """Check safety limits before generating signals"""
        
        # Daily signal limit
        max_daily_signals = 20
        if self.daily_signals >= max_daily_signals:
            return False
        
        # Consecutive loss limit
        max_consecutive_losses = self.config.get_parameter('max_consecutive_losses', 5)
        if self.consecutive_losses >= max_consecutive_losses:
            return False
        
        # Daily loss limit
        max_daily_loss = self.config.get_parameter('max_daily_loss', 500)
        if self.daily_pnl <= -max_daily_loss:
            return False
        
        return True
    
    def _should_generate_signal(self, intelligence_result: Dict) -> bool:
        """Check if conditions are right for signal generation"""
        
        # Check signal strength
        overall_signal = intelligence_result.get('overall_signal', 0.0)
        confidence = intelligence_result.get('confidence', 0.0)
        
        min_signal_strength = 0.2
        min_confidence = self.config.get_parameter('confidence_threshold', 0.6)
        
        signal_strong_enough = abs(overall_signal) >= min_signal_strength
        confidence_high_enough = confidence >= min_confidence
        
        return signal_strong_enough and confidence_high_enough
    
    def _generate_trading_signal(self, market_data: Dict, intelligence_result: Dict) -> bool:
        """Generate and send trading signal"""
        try:
            # Get decision from RL agent
            decision = self.rl_agent.select_action(
                market_data, intelligence_result, self.current_account_data, self.current_price
            )
            
            # Only process non-hold actions
            if decision['action'] == 0:  # Hold
                return False
            
            # Calculate position size using account data
            position_size = self._calculate_position_size(decision)
            
            # Store decision for learning correlation
            self.last_decision = decision
            self.last_decision['calculated_position_size'] = position_size
            
            # Send signal to NinjaTrader
            success = self.tcp_bridge.send_signal(
                action=decision['action'],
                confidence=decision['confidence'],
                position_size=position_size,
                stop_price=decision.get('stop_price', 0.0),
                target_price=decision.get('target_price', 0.0),
                tool_name=decision.get('primary_tool', 'unknown')
            )
            
            if success:
                # Update counters
                self.daily_signals += 1
                self.stats['total_signals'] += 1
                self.stats['signals_today'] += 1
                
                # Track for correlation
                self.trade_counter += 1
                trade_id = f"trade_{self.trade_counter}"
                
                self.pending_trades[trade_id] = {
                    'decision': decision,
                    'intelligence_result': intelligence_result,
                    'timestamp': datetime.now(),
                    'position_size': position_size
                }
                
                action_name = ['HOLD', 'BUY', 'SELL'][decision['action']]
                tool_name = decision.get('primary_tool', 'unknown')
                
                log.info(f"Signal sent: {action_name} using {tool_name.upper()} "
                        f"(conf: {decision['confidence']:.3f}, size: {position_size:.1f})")
                
                return True
            
            return False
            
        except Exception as e:
            log.error(f"Error generating signal: {e}")
            return False
    
    def _calculate_position_size(self, decision: Dict) -> float:
        """Calculate position size based on account data and risk management"""
        
        # Get base position size from RL agent
        base_size = decision.get('position_size', 1.0)
        
        # Get account information
        buying_power = self.current_account_data.get('buying_power', 25000)
        account_balance = self.current_account_data.get('account_balance', 25000)
        
        # Risk management parameters
        risk_per_trade = self.config.get_parameter('risk_per_trade', 0.02)  # 2% per trade
        position_multiplier = self.config.get_parameter('position_size', 1.0)
        
        # Calculate risk amount
        risk_amount = account_balance * risk_per_trade
        
        # MNQ specifics
        margin_per_contract = 500.0  # Approximate margin requirement
        point_value = 2.0  # $2 per point
        
        # Max contracts by margin
        max_by_margin = buying_power / margin_per_contract
        
        # Max contracts by risk (assuming 20 point stop)
        estimated_stop_distance = 20.0  # points
        max_by_risk = risk_amount / (estimated_stop_distance * point_value)
        
        # Use the most conservative limit
        max_contracts = min(max_by_margin * 0.8, max_by_risk)  # 80% of margin limit for safety
        
        # Apply base size and multiplier
        final_size = base_size * position_multiplier
        final_size = min(final_size, max_contracts)
        final_size = max(1.0, final_size)  # At least 1 contract
        final_size = min(final_size, 5.0)  # Max 5 contracts for safety
        
        return int(final_size)
    
    def process_trade_completion(self, completion_data: Dict):
        """Process trade completion and update learning"""
        try:
            pnl = completion_data.get('final_pnl', 0.0)
            tool_used = completion_data.get('tool_used', 'unknown')
            exit_reason = completion_data.get('exit_reason', 'unknown')
            
            # Update daily P&L
            self.daily_pnl += pnl
            self.stats['total_pnl'] += pnl
            
            # Update consecutive losses
            if pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
                self.stats['successful_trades'] += 1
            
            # Track tool performance
            if tool_used in self.stats['tool_performance']:
                self.stats['tool_performance'][tool_used].append(pnl)
                # Keep only recent performance
                if len(self.stats['tool_performance'][tool_used]) > 50:
                    self.stats['tool_performance'][tool_used] = \
                        self.stats['tool_performance'][tool_used][-50:]
            
            # Find matching pending trade for learning
            matching_trade = self._find_matching_trade(completion_data)
            
            if matching_trade:
                # Let RL agent learn from outcome
                self.rl_agent.learn_from_outcome(
                    matching_trade['decision'], pnl, completion_data
                )
                
                # Let intelligence engine learn
                self.intelligence_engine.learn_from_outcome(
                    matching_trade['intelligence_result'], pnl
                )
                
                # Update configuration parameters
                normalized_outcome = np.tanh(pnl / 50.0)
                self.config.update_parameter('confidence_threshold', normalized_outcome)
                self.config.update_parameter('position_size', normalized_outcome)
                
                log.info(f"Learning completed: P&L=${pnl:.2f}, Tool={tool_used}, Exit={exit_reason}")
            else:
                log.warning(f"No matching trade found for completion: P&L=${pnl:.2f}")
            
            # Update best performing tool
            self._update_best_tool()
            
        except Exception as e:
            log.error(f"Error processing trade completion: {e}")
    
    def _find_matching_trade(self, completion_data: Dict) -> Optional[Dict]:
        """Find matching pending trade for learning correlation"""
        
        tool_used = completion_data.get('tool_used', 'unknown')
        signal_timestamp = completion_data.get('signal_timestamp', 0)
        
        # Find best matching trade
        best_match = None
        best_score = 0
        
        for trade_id, trade_data in list(self.pending_trades.items()):
            # Score based on tool match and time proximity
            tool_match = 1.0 if trade_data['decision'].get('primary_tool') == tool_used else 0.0
            
            # Time difference (prefer recent trades)
            time_diff = abs((datetime.now() - trade_data['timestamp']).total_seconds())
            time_score = max(0, 1.0 - time_diff / 300.0)  # 5 minute window
            
            match_score = tool_match * 0.7 + time_score * 0.3
            
            if match_score > best_score and time_diff < 600:  # 10 minute max
                best_score = match_score
                best_match = trade_data
                best_match_id = trade_id
        
        # Remove matched trade
        if best_match and 'best_match_id' in locals():
            del self.pending_trades[best_match_id]
        
        return best_match
    
    def _update_best_tool(self):
        """Update best performing tool"""
        tool_avg_pnl = {}
        
        for tool, pnls in self.stats['tool_performance'].items():
            if len(pnls) >= 5:  # Need at least 5 trades
                tool_avg_pnl[tool] = np.mean(pnls[-10:])  # Recent performance
        
        if tool_avg_pnl:
            self.stats['best_tool'] = max(tool_avg_pnl.items(), key=lambda x: x[1])[0]
    
    def get_status(self) -> Dict:
        """Get current trade manager status"""
        
        # Calculate success rate
        total_completed = sum(len(pnls) for pnls in self.stats['tool_performance'].values())
        success_rate = self.stats['successful_trades'] / max(1, total_completed)
        
        # Average P&L per trade
        avg_pnl = self.stats['total_pnl'] / max(1, total_completed)
        
        return {
            'daily_signals': self.daily_signals,
            'consecutive_losses': self.consecutive_losses,
            'daily_pnl': self.daily_pnl,
            'total_signals': self.stats['total_signals'],
            'success_rate': success_rate,
            'avg_pnl_per_trade': avg_pnl,
            'best_tool': self.stats['best_tool'],
            'pending_trades': len(self.pending_trades),
            'account_balance': self.current_account_data.get('account_balance', 0),
            'buying_power': self.current_account_data.get('buying_power', 0),
            'tool_performance': {
                tool: np.mean(pnls[-5:]) if len(pnls) >= 5 else 0.0
                for tool, pnls in self.stats['tool_performance'].items()
            }
        }
    
    def get_performance_report(self) -> str:
        """Generate human-readable performance report"""
        status = self.get_status()
        
        report = f"""
=== TRADE MANAGER PERFORMANCE REPORT ===

Daily Status:
  Signals Today: {status['daily_signals']}
  Daily P&L: ${status['daily_pnl']:.2f}
  Consecutive Losses: {status['consecutive_losses']}

Overall Performance:
  Total Signals: {status['total_signals']}
  Success Rate: {status['success_rate']:.1%}
  Avg P&L per Trade: ${status['avg_pnl_per_trade']:.2f}
  Best Tool: {status['best_tool'].upper()}

Account Status:
  Balance: ${status['account_balance']:.0f}
  Buying Power: ${status['buying_power']:.0f}

Tool Performance (Recent Avg):"""
        
        for tool, avg_pnl in status['tool_performance'].items():
            if avg_pnl != 0:
                report += f"\n  {tool.upper()}: ${avg_pnl:.2f}"
        
        report += f"\n\nPending Trades: {status['pending_trades']}"
        
        return report
    
    def cleanup(self):
        """Cleanup resources"""
        log.info(f"Trade manager cleanup - {len(self.pending_trades)} pending trades")

# Factory function
def create_trade_manager(rl_agent, intelligence_engine, tcp_bridge, config):
    """Create simple trade manager"""
    return SimpleTradeManager(rl_agent, intelligence_engine, tcp_bridge, config)