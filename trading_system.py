# trading_system.py

import logging
import time

from data_processor import DataProcessor
from intelligence_engine import IntelligenceEngine
from trading_agent import TradingAgent
from risk_manager import RiskManager
from tcp_bridge import TCPServer
from portfolio import Portfolio
from config import Config

logger = logging.getLogger(__name__)

class TradingSystem:
    def __init__(self):
        logger.info("Initializing trading system")
        
        self.portfolio = Portfolio()
        self.data_processor = DataProcessor()
        self.intelligence = IntelligenceEngine()
        self.agent = TradingAgent(self.intelligence, self.portfolio)
        self.risk_manager = RiskManager(self.portfolio, self.agent.meta_learner)
        
        self.tcp_server = TCPServer()
        self.tcp_server.on_market_data = self._process_market_data
        self.tcp_server.on_trade_completion = self._process_trade_completion
        
        self.running = False
        self.last_save = time.time()
        self.last_account_update = time.time()
        
        # Account monitoring
        self.last_account_balance = 0.0
        self.account_change_threshold = 0.05  # 5% change triggers adaptation
        
        # Load previous state if available
        self._load_state()

    def start(self):
        logger.info("Starting trading system")
        self.tcp_server.start()
        self.running = True
        
        try:
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        finally:
            self.shutdown()

    def _main_loop(self):
        while self.running:
            time.sleep(0.1)
            
            # Save state every 5 minutes
            if time.time() - self.last_save > 300:
                self._save_state()
                self.last_save = time.time()
            
            # Log performance summary every 30 minutes
            if time.time() - self.last_account_update > 1800:
                self._log_performance_summary()
                self.last_account_update = time.time()

    def _process_market_data(self, raw_data):
        try:
            market_data = self.data_processor.process(raw_data)
            
            if not market_data:
                return
            
            # Check for significant account changes and adapt
            self._check_account_adaptation(market_data)
                
            features = self.intelligence.extract_features(market_data)
            
            decision = self.agent.decide(features, market_data)
            if decision.action == 'hold':
                return
                
            order = self.risk_manager.validate_order(decision, market_data)
            if order:
                # Enhanced order validation with account context
                if self._validate_order_with_account_context(order, market_data):
                    success = self.tcp_server.send_signal(order)
                    
                    if success:
                        # Store enhanced order with intelligence data
                        order.features = features
                        order.market_data = market_data
                        order.intelligence_data = decision.intelligence_data
                        order.decision_data = {
                            'primary_tool': decision.primary_tool,
                            'exploration': decision.exploration,
                            'state_features': decision.state_features
                        }
                        
                        self.portfolio.add_pending_order(order)
                        
                        # Log with account context
                        account_risk = (order.size * 600) / market_data.account_balance * 100  # Estimate risk %
                        logger.info(f"Order placed: {order.action.upper()} {order.size} @ {order.price:.2f} "
                                  f"(Risk: {account_risk:.1f}%, Balance: ${market_data.account_balance:.0f})")
                    else:
                        logger.warning("Failed to send signal to NinjaTrader")
                else:
                    logger.info("Order rejected by account context validation")
                
        except Exception as e:
            logger.error(f"Error processing market data: {e}")

    def _process_trade_completion(self, completion_data):
        try:
            trade = self.portfolio.complete_trade(completion_data)
            if trade:
                # Enhanced learning with intelligence data
                self.agent.learn_from_trade(trade)
                self.intelligence.learn_from_outcome(trade)
                
                # Log trade with account impact
                account_impact = trade.account_risk_pct * 100
                logger.info(f"Trade completed: {trade.action.upper()} "
                          f"P&L: ${trade.pnl:.2f} ({account_impact:.2f}% of account) "
                          f"Reason: {trade.exit_reason}")
                
                # Check for significant account changes after trade
                if hasattr(trade, 'exit_account_balance') and trade.exit_account_balance > 0:
                    self._update_account_tracking(trade.exit_account_balance)
                
        except Exception as e:
            logger.error(f"Error processing trade completion: {e}")

    def _check_account_adaptation(self, market_data):
        """Check if we need to adapt to account balance changes"""
        current_balance = market_data.account_balance
        
        if self.last_account_balance == 0.0:
            self.last_account_balance = current_balance
            logger.info(f"Initial account balance: ${current_balance:.2f}")
            return
        
        # Check for significant balance changes
        balance_change = abs(current_balance - self.last_account_balance) / self.last_account_balance
        
        if balance_change > self.account_change_threshold:
            logger.info(f"Significant account change detected: "
                       f"${self.last_account_balance:.2f} -> ${current_balance:.2f} "
                       f"({balance_change:.1%})")
            
            # Trigger meta-learner adaptation
            self.agent.meta_learner.adapt_to_account_size(current_balance)
            self.last_account_balance = current_balance
    
    def _update_account_tracking(self, new_balance):
        """Update account tracking after trade"""
        if self.last_account_balance > 0:
            change = (new_balance - self.last_account_balance) / self.last_account_balance
            if abs(change) > 0.01:  # 1% change
                logger.info(f"Account balance updated: ${self.last_account_balance:.2f} -> ${new_balance:.2f}")
        
        self.last_account_balance = new_balance
    
    def _validate_order_with_account_context(self, order, market_data) -> bool:
        """Additional validation with account context"""
        
        # Estimated margin requirement for order
        estimated_margin = order.size * 600  # $600 per MNQ contract estimate
        
        # Check available margin
        if estimated_margin > market_data.available_margin * 0.8:  # Use max 80% of available
            logger.warning(f"Order rejected: Insufficient margin "
                          f"(Need: ${estimated_margin:.0f}, Available: ${market_data.available_margin:.0f})")
            return False
        
        # Check account risk percentage
        account_risk_pct = estimated_margin / market_data.account_balance
        max_risk_pct = 0.1 if market_data.account_balance < 10000 else 0.2  # More conservative for smaller accounts
        
        if account_risk_pct > max_risk_pct:
            logger.warning(f"Order rejected: Risk too high "
                          f"({account_risk_pct:.1%} > {max_risk_pct:.1%} of account)")
            return False
        
        # Check margin utilization after order
        projected_margin_usage = (market_data.margin_used + estimated_margin) / market_data.net_liquidation
        if projected_margin_usage > 0.8:  # Don't exceed 80% total margin usage
            logger.warning(f"Order rejected: Total margin usage would be too high "
                          f"({projected_margin_usage:.1%})")
            return False
        
        return True
    
    def _log_performance_summary(self):
        """Log comprehensive performance summary with account metrics"""
        try:
            portfolio_summary = self.portfolio.get_summary()
            agent_stats = self.agent.get_stats()
            intelligence_stats = self.intelligence.get_stats()
            
            logger.info("=== Performance Summary ===")
            logger.info(f"Account Balance: ${portfolio_summary.get('current_balance', 0):.2f}")
            logger.info(f"Session Return: {portfolio_summary.get('session_return_pct', 0):.2f}%")
            logger.info(f"Max Drawdown: {portfolio_summary.get('max_drawdown_pct', 0):.2f}%")
            logger.info(f"Total Trades: {portfolio_summary['total_trades']}")
            logger.info(f"Win Rate: {portfolio_summary['win_rate']:.1%}")
            logger.info(f"Daily P&L: ${portfolio_summary['daily_pnl']:.2f}")
            logger.info(f"Profit Factor: {portfolio_summary.get('profit_factor', 0):.2f}")
            logger.info(f"Avg Risk/Trade: {portfolio_summary.get('avg_risk_per_trade_pct', 0):.2f}%")
            logger.info(f"Margin Usage: ${portfolio_summary.get('current_margin_usage', 0):.0f}")
            
            logger.info(f"Agent Learning Efficiency: {agent_stats['learning_efficiency']:.2%}")
            logger.info(f"Architecture Generation: {agent_stats['architecture_generation']}")
            logger.info(f"Intelligence Patterns: DNA={intelligence_stats['dna_patterns']}, "
                       f"Micro={intelligence_stats['micro_patterns']}, "
                       f"Temporal={intelligence_stats['temporal_patterns']}")
            logger.info("=" * 30)
            
        except Exception as e:
            logger.error(f"Error logging performance summary: {e}")

    def _save_state(self):
        try:
            self.agent.save_model('models/agent.pt')
            self.intelligence.save_patterns('data/patterns.json')
            self.portfolio.save_state('data/portfolio.json')
            
            # Save system state with account tracking
            system_state = {
                'last_account_balance': self.last_account_balance,
                'account_change_threshold': self.account_change_threshold,
                'saved_at': time.time()
            }
            
            import json
            with open('data/system_state.json', 'w') as f:
                json.dump(system_state, f)
            
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def _load_state(self):
        try:
            self.agent.load_model('models/agent.pt')
            self.intelligence.load_patterns('data/patterns.json')
            self.portfolio.load_state('data/portfolio.json')
            
            # Load system state
            import json
            try:
                with open('data/system_state.json', 'r') as f:
                    system_state = json.load(f)
                    self.last_account_balance = system_state.get('last_account_balance', 0.0)
                    self.account_change_threshold = system_state.get('account_change_threshold', 0.05)
            except FileNotFoundError:
                pass
            
        except Exception as e:
            logger.info("Starting with fresh state")

    def shutdown(self):
        logger.info("Shutting down trading system")
        self.running = False
        
        self._save_state()
        self.tcp_server.stop()
        
        # Final performance summary
        final_summary = self.portfolio.get_summary()
        logger.info(f"Final session summary: {final_summary}")
        
        # Log final account metrics
        account_perf = self.portfolio.get_account_performance()
        if account_perf:
            logger.info(f"Session return: {account_perf.get('session_return_pct', 0):.2f}%")
            logger.info(f"Max drawdown: {account_perf.get('max_drawdown_pct', 0):.2f}%")
            logger.info(f"Profit factor: {account_perf.get('profit_factor', 0):.2f}")