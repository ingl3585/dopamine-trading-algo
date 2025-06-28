# trading_system.py

import logging
import time

from src.market_analysis.data_processor import DataProcessor
from src.intelligence.intelligence_engine import IntelligenceEngine
from src.agent.trading_agent import TradingAgent
from src.risk.risk_manager import RiskManager
from src.communication.tcp_bridge import TCPServer
from src.portfolio.portfolio_manager import PortfolioManager
from src.core.config import Config

logger = logging.getLogger(__name__)

# AI Trading Personality Integration
try:
    from src.personality.config_manager import PersonalityConfigManager
    from src.personality.trading_personality import TradingPersonality, TriggerEvent
    PERSONALITY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Personality system not available: {e}")
    PERSONALITY_AVAILABLE = False

logger = logging.getLogger(__name__)

class TradingSystem:
    def __init__(self):
        logger.info("Initializing trading system with historical bootstrapping")
        self.config = Config()
        self.portfolio = PortfolioManager(self.config)
        self.data_processor = DataProcessor()
        self.intelligence = IntelligenceEngine(self.config)
        self.agent = TradingAgent(self.intelligence, self.portfolio)
        self.risk_manager = RiskManager(self.portfolio, self.agent.meta_learner, self.agent)
        
        self.tcp_server = TCPServer()
        self.tcp_server.on_market_data = self._process_market_data
        self.tcp_server.on_trade_completion = self._process_trade_completion
        self.tcp_server.on_historical_data = self._process_historical_data
        
        self.running = False
        self.last_save = time.time()
        self.last_account_update = time.time()
        self.ready_for_trading = False
        
        # Enhanced tracking
        self.total_decisions = 0
        self.data_updates_received = 0
        self.last_detailed_log = time.time()
        
        # Account monitoring
        self.last_account_balance = 0.0
        self.account_change_threshold = 0.05
        
        # AI Trading Personality Integration
        self.personality = None
        if PERSONALITY_AVAILABLE:
            try:
                personality_config_manager = PersonalityConfigManager()
                personality_config = personality_config_manager.get_personality_config()
                if personality_config_manager.get_integration_config().enabled:
                    self.personality = TradingPersonality(personality_config)
                    logger.info(f"AI Trading Personality '{personality_config.personality_name}' initialized")
                else:
                    logger.info("AI Trading Personality disabled in configuration")
            except Exception as e:
                logger.error(f"Failed to initialize AI Trading Personality: {e}")
                self.personality = None
        
        # Load previous state if available
        self._load_state()

    def start(self):
        logger.info("Starting trading system - waiting for historical data...")
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
            
            # Wait for historical data processing before starting trading
            if not self.ready_for_trading:
                if self.tcp_server.is_ready_for_live_trading() and self.intelligence.historical_processed:
                    self.ready_for_trading = True
                    logger.info("=== READY FOR LIVE TRADING ===")
                    self._log_bootstrap_summary()
                continue
            
            # Enhanced status logging every 2 minutes during live trading
            current_time = time.time()
            if current_time - self.last_detailed_log > 120:  # Every 2 minutes
                self._log_detailed_status()
                self.last_detailed_log = current_time
            
            # Save state every 5 minutes
            if current_time - self.last_save > 300:
                self._save_state()
                self.last_save = current_time
            
            # Log performance summary every 30 minutes
            if current_time - self.last_account_update > 1800:
                self._log_performance_summary()
                self.last_account_update = current_time

    def _process_historical_data(self, historical_data):
        """Process historical data for pattern bootstrapping and priming the data processor."""
        try:
            logger.info("Processing historical data for pattern learning...")
            
            # Bootstrap the intelligence engine with historical patterns
            self.intelligence.bootstrap_from_historical_data(historical_data)

            # Prime the data processor with the same historical data
            self.data_processor.prime_with_historical_data(historical_data)
            logger.info("Data processor primed with historical data.")
            
            # Signal that historical processing is complete
            logger.info("Historical data processing complete")
            
        except Exception as e:
            logger.error(f"Error processing historical data: {e}")

    def _process_market_data(self, raw_data):
        # Only process live market data if we're ready for trading
        if not self.ready_for_trading:
            return
            
        self.data_updates_received += 1
            
        try:
            # Enhanced logging for debugging
            if self.data_updates_received % 20 == 0:  # Log every 20 data updates
                logger.info(f"Processing market data update #{self.data_updates_received}")
            
            market_data = self.data_processor.process(raw_data)
            
            if not market_data:
                logger.warning("Market data processing returned None")
                return
            
            # Log market data quality every 10 updates
            if self.data_updates_received % 10 == 0:
                logger.info(f"Market data: Price={market_data.price:.2f}, "
                           f"1m_bars={len(market_data.prices_1m)}, "
                           f"5m_bars={len(market_data.prices_5m)}, "
                           f"15m_bars={len(market_data.prices_15m)}")
            
            # Check for significant account changes and adapt
            self._check_account_adaptation(market_data)
                
            features = self.intelligence.extract_features(market_data)
            
            # Log intelligence analysis every 5 updates
            if self.data_updates_received % 5 == 0:
                logger.info(f"Intelligence: DNA={features.dna_signal:.3f}, "
                           f"Temporal={features.temporal_signal:.3f}, "
                           f"Immune={features.immune_signal:.3f}, "
                           f"Overall={features.overall_signal:.3f}, "
                           f"Confidence={features.confidence:.3f}")
            
            decision = self.agent.decide(features, market_data)
            self.total_decisions += 1
            
            # Always log decision details
            logger.info(f"Decision #{self.total_decisions}: {decision.action.upper()} "
                       f"(Size: {decision.size:.1f}, Conf: {decision.confidence:.3f}, "
                       f"Tool: {decision.primary_tool}, Exploration: {decision.exploration})")
            
            if decision.action == 'hold':
                return
                
            order = self.risk_manager.validate_order(decision, market_data)
            if order:
                logger.info(f"Risk manager approved order: {order.action.upper()} {order.size}")
                
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
                    account_risk = (order.size * 100) / market_data.account_balance * 100
                    logger.info(f"Order placed: {order.action.upper()} {order.size} @ {order.price:.2f} "
                                    f"(Risk: {account_risk:.1f}%, Balance: ${market_data.account_balance:.0f})")
                    
                    # AI Personality Commentary moved to periodic system status updates
                else:
                    logger.warning("Failed to send signal to NinjaTrader")
            else:
                logger.info(f"Risk manager rejected order")
                
        except Exception as e:
            import traceback
            logger.error(f"Error processing market data: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.error(f"Raw data keys: {list(raw_data.keys()) if isinstance(raw_data, dict) else 'Not a dict'}")

    def _learn_from_outcome(self, outcome):
        """Sequential learning coordinator - eliminates race conditions"""
        # Phase 1: Subsystems learn first (no conflicts)
        self.intelligence.learn_from_outcome(outcome)
        # Phase 2: RL agent learns second (uses updated subsystem weights)
        self.agent.learn_from_trade(outcome)

    # Rest of the methods remain the same...
    def _process_trade_completion(self, completion_data):
        try:
            # Debug: Log what we're receiving from NinjaTrader
            logger.info(f"Raw completion data: {completion_data}")
            trade = self.portfolio.complete_trade(completion_data)
            if trade:
                # Sequential learning phases to prevent race conditions
                self._learn_from_outcome(trade)
                
                # Process trade outcome for advanced risk learning
                trade_outcome = {
                    'pnl': trade.pnl,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'size': trade.size,
                    'exit_reason': trade.exit_reason,
                    'account_balance': trade.exit_account_balance
                }
                self.risk_manager.process_trade_outcome(trade_outcome)
                
                # Log trade with account impact and risk metrics
                account_impact = trade.account_risk_pct * 100
                risk_summary = self.risk_manager.get_risk_summary()
                
                logger.info(f"Trade completed: {trade.action.upper()} "
                          f"P&L: ${trade.pnl:.2f} ({account_impact:.2f}% of account) "
                          f"Reason: {trade.exit_reason} "
                          f"Regime: {risk_summary.get('current_regime', 'unknown')}")
                
                # AI Personality Commentary removed - only triggers on system status
                
                # Check for significant account changes after trade
                if hasattr(trade, 'exit_account_balance') and trade.exit_account_balance > 0:
                    self._update_account_tracking(trade.exit_account_balance)
                
        except Exception as e:
            logger.error(f"Error processing trade completion: {e}")

    def _log_bootstrap_summary(self):
        """Log summary of historical pattern learning"""
        try:
            bootstrap_stats = self.intelligence.bootstrap_stats
            intelligence_stats = self.intelligence.get_stats()
            
            logger.info("=== BOOTSTRAP SUMMARY ===")
            logger.info(f"Historical bars processed: {bootstrap_stats['total_bars_processed']}")
            logger.info(f"Patterns discovered: {bootstrap_stats['patterns_discovered']}")
            logger.info(f"Bootstrap time: {bootstrap_stats['bootstrap_time']:.1f}s")
            logger.info(f"DNA patterns: {intelligence_stats['dna_patterns']}")
            logger.info(f"Micro patterns: {intelligence_stats['micro_patterns']}")
            logger.info(f"Temporal patterns: {intelligence_stats['temporal_patterns']}")
            logger.info(f"Immune patterns: {intelligence_stats['immune_patterns']}")
            logger.info("System ready for live trading with pre-learned patterns")
            logger.info("=" * 30)
            
        except Exception as e:
            logger.error(f"Error logging bootstrap summary: {e}")

    def _log_detailed_status(self):
        """Log detailed system status during live trading"""
        try:
            logger.info("=== SYSTEM STATUS ===")
            logger.info(f"Data updates received: {self.data_updates_received}")
            logger.info(f"Total decisions made: {self.total_decisions}")
            
            # Portfolio status
            portfolio_summary = self.portfolio.get_summary()
            logger.info(f"Portfolio: {portfolio_summary['total_trades']} trades, "
                       f"Win rate: {portfolio_summary['win_rate']:.1%}, "
                       f"Daily P&L: ${portfolio_summary['daily_pnl']:.2f}")
            
            # Agent status
            agent_stats = self.agent.get_stats()
            logger.info(f"Agent: {agent_stats['total_decisions']} decisions, "
                       f"Success rate: {agent_stats['success_rate']:.1%}, "
                       f"Learning efficiency: {agent_stats['learning_efficiency']:.1%}")
            
            # Intelligence status
            intelligence_stats = self.intelligence.get_stats()
            logger.info(f"Intelligence: DNA={intelligence_stats['dna_patterns']}, "
                       f"Micro={intelligence_stats['micro_patterns']}, "
                       f"Temporal={intelligence_stats['temporal_patterns']}, "
                       f"Immune={intelligence_stats['immune_patterns']}")
            
            # Connection status
            logger.info(f"TCP: {self.tcp_server.data_received} data messages, "
                       f"{self.tcp_server.signals_sent} signals sent")
            
            # AI Personality Commentary - Only on System Status
            if self.personality:
                self._trigger_personality_system_update()
            
            logger.info("=" * 20)
            
        except Exception as e:
            logger.error(f"Error logging detailed status: {e}")

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
            if abs(change) > 0.01:
                logger.info(f"Account balance updated: ${self.last_account_balance:.2f} -> ${new_balance:.2f}")
        
        self.last_account_balance = new_balance
    
    def _log_performance_summary(self):
        """Log comprehensive performance summary with account metrics"""
        try:
            portfolio_summary = self.portfolio.get_summary()
            agent_stats = self.agent.get_stats()
            intelligence_stats = self.intelligence.get_stats()
            risk_summary = self.risk_manager.get_risk_summary()
            
            logger.info("=== PERFORMANCE SUMMARY ===")
            logger.info(f"Account Balance: ${portfolio_summary.get('current_balance', 0):.2f}")
            logger.info(f"Session Return: {portfolio_summary.get('session_return_pct', 0):.2f}%")
            logger.info(f"Max Drawdown: {portfolio_summary.get('max_drawdown_pct', 0):.2f}%")
            logger.info(f"Total Trades: {portfolio_summary['total_trades']}")
            logger.info(f"Win Rate: {portfolio_summary['win_rate']:.1%}")
            logger.info(f"Daily P&L: ${portfolio_summary['daily_pnl']:.2f}")
            logger.info(f"Profit Factor: {portfolio_summary.get('profit_factor', 0):.2f}")
            
            logger.info(f"Agent Learning Efficiency: {agent_stats['learning_efficiency']:.2%}")
            logger.info(f"Architecture Generation: {agent_stats['architecture_generation']}")
            logger.info(f"Intelligence Patterns: DNA={intelligence_stats['dna_patterns']}, "
                       f"Micro={intelligence_stats['micro_patterns']}, "
                       f"Temporal={intelligence_stats['temporal_patterns']}, "
                       f"Immune={intelligence_stats['immune_patterns']}")
            
            # Advanced risk metrics
            logger.info(f"Risk Regime: {risk_summary.get('current_regime', 'unknown')}")
            logger.info(f"Portfolio Heat: {risk_summary.get('portfolio_heat', 0):.1%}")
            tail_metrics = risk_summary.get('tail_risk_metrics', {})
            if tail_metrics:
                logger.info(f"VaR 95%: {tail_metrics.get('var_95', 0):.4f}")
                logger.info(f"Black Swan Prob: {tail_metrics.get('black_swan_probability', 0):.4f}")
            
            if intelligence_stats['historical_processed']:
                logger.info(f"Bootstrap Stats: {intelligence_stats['bootstrap_stats']['total_bars_processed']} bars processed")
            
            logger.info("=" * 30)
            
        except Exception as e:
            logger.error(f"Error logging performance summary: {e}")

    def _save_state(self):
        """
        Persist agent + intelligence + portfolio + minimal runtime stats.
        Handles NumPy scalars so json.dump never throws
        “Object of type int64 is not JSON serializable”.
        """
        try:
            import os, json, time, numpy as np

            # --- helper ----------------------------------------------------
            def _np_encoder(obj):
                """Convert NumPy scalars/arrays to vanilla Python types."""
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, (np.bool_,)):
                    return bool(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                raise TypeError(f"{type(obj)} is not JSON serializable")
            # ----------------------------------------------------------------

            os.makedirs("models", exist_ok=True)
            os.makedirs("data",   exist_ok=True)

            # binary / torch checkpoints
            self.agent.save_model("models/agent.pt")
            self.intelligence.save_patterns("data/patterns.json")  # already JSON-safe
            self.portfolio.save_state("data/portfolio.json")

            # tiny runtime snapshot
            system_state = {
                "last_account_balance" : float(self.last_account_balance),
                "account_change_threshold": float(self.account_change_threshold),
                "ready_for_trading"    : bool(self.ready_for_trading),
                "total_decisions"      : int(self.total_decisions),
                "data_updates_received": int(self.data_updates_received),
                "saved_at"             : time.time()
            }

            with open("data/system_state.json", "w") as f:
                json.dump(system_state, f, indent=2, default=_np_encoder)

        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def _load_state(self):
        try:
            self.agent.load_model('models/agent.pt')
            self.portfolio.load_state('data/portfolio.json')
            
            # Load system state
            import json
            try:
                with open('data/system_state.json', 'r') as f:
                    system_state = json.load(f)
                    self.last_account_balance = system_state.get('last_account_balance', 0.0)
                    self.account_change_threshold = system_state.get('account_change_threshold', 0.05)
                    self.total_decisions = system_state.get('total_decisions', 0)
                    self.data_updates_received = system_state.get('data_updates_received', 0)
                    # Don't restore ready_for_trading - always wait for fresh historical data
            except FileNotFoundError:
                pass
            
            # Log if we have existing patterns
            if self.intelligence.historical_processed:
                logger.info("Loaded existing historical patterns - will still wait for fresh bootstrap")
            
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
        
        # Log final account metrics and bootstrap stats
        account_perf = self.portfolio.get_account_performance()
        if account_perf:
            logger.info(f"Session return: {account_perf.get('session_return_pct', 0):.2f}%")
            logger.info(f"Max drawdown: {account_perf.get('max_drawdown_pct', 0):.2f}%")
            logger.info(f"Profit factor: {account_perf.get('profit_factor', 0):.2f}")
        
        # Log final intelligence stats
        intelligence_stats = self.intelligence.get_stats()
        if intelligence_stats['historical_processed']:
            logger.info(f"Final pattern count: DNA={intelligence_stats['dna_patterns']}, "
                       f"Micro={intelligence_stats['micro_patterns']}, "
                       f"Temporal={intelligence_stats['temporal_patterns']}")
        
        # Shutdown personality system
        if self.personality:
            self.personality.shutdown()
            
        logger.info("Trading system shutdown complete")

    def _trigger_personality_commentary(self, event, features, market_data, order=None, decision=None):
        """Trigger AI personality commentary for trading events"""
        if not self.personality:
            return
            
        try:
            # Build comprehensive context for personality
            context = {
                'subsystem_signals': {
                    'dna': float(features.dna_signal),
                    'temporal': float(features.temporal_signal), 
                    'immune': float(features.immune_signal),
                    'microstructure': float(getattr(features, 'microstructure_signal', 0.0)),
                    'dopamine': float(getattr(features, 'dopamine_signal', 0.0)),
                    'regime': float(getattr(features, 'regime_signal', 0.0))
                },
                'market_data': {
                    'price': float(market_data.price),
                    'volatility': float(getattr(market_data, 'volatility', 0.02)),
                    'trend_strength': float(getattr(features, 'trend_strength', 0.0)),
                    'volume_regime': float(getattr(features, 'volume_regime', 0.5)),
                    'regime': getattr(market_data, 'regime', 'normal')
                },
                'portfolio_state': self._get_portfolio_state(),
                'decision_context': {
                    'decision_type': event.value if hasattr(event, 'value') else str(event),
                    'confidence': float(features.confidence),
                    'overall_signal': float(features.overall_signal),
                    'primary_tool': getattr(decision, 'primary_tool', 'unknown') if decision else 'unknown',
                    'exploration': getattr(decision, 'exploration', False) if decision else False
                }
            }
            
            # Add order-specific context if available
            if order:
                context['order_details'] = {
                    'action': order.action,
                    'size': float(order.size),
                    'price': float(order.price)
                }
            
            # Trigger async commentary (don't wait for it)
            import asyncio
            try:
                # Try to run in existing event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule as task in running loop
                    loop.create_task(self._async_personality_commentary(event, context))
                else:
                    # Create new event loop
                    asyncio.run(self._async_personality_commentary(event, context))
            except RuntimeError:
                # No event loop, create new one
                asyncio.run(self._async_personality_commentary(event, context))
                
        except Exception as e:
            logger.error(f"Error triggering personality commentary: {e}")

    async def _async_personality_commentary(self, event, context):
        """Async handler for personality commentary - DISABLED to only allow system status commentary"""
        # Commentary disabled except for system status updates
        pass

    def _get_portfolio_state(self):
        """Get portfolio state for personality context"""
        try:
            summary = self.portfolio.get_summary()
            
            # Get current position from market data (NinjaTrader real-time data)
            current_position_size = 0
            if hasattr(self, 'last_market_data') and self.last_market_data:
                current_position_size = getattr(self.last_market_data, 'total_position_size', 0)
            
            # Create positions dict with real-time position data
            positions = dict(self.portfolio.positions) if hasattr(self.portfolio, 'positions') else {}
            
            # If we have a real-time position but no portfolio positions, create one
            if current_position_size != 0 and not positions:
                positions['current'] = {
                    'size': current_position_size,
                    'direction': 'long' if current_position_size > 0 else 'short',
                    'entry_price': getattr(self.last_market_data, 'price', 0),
                    'unrealized_pnl': getattr(self.last_market_data, 'unrealized_pnl', 0)
                }
            
            return {
                'positions': positions,
                'current_position_size': current_position_size,  # Add explicit position size
                'unrealized_pnl': float(getattr(self.last_market_data, 'unrealized_pnl', summary.get('total_pnl', 0.0))),
                'daily_pnl': float(getattr(self.last_market_data, 'daily_pnl', summary.get('daily_pnl', 0.0))),
                'recent_performance': [trade.pnl for trade in list(self.portfolio.trade_history)[-5:]] if hasattr(self.portfolio, 'trade_history') else []
            }
        except Exception as e:
            logger.warning(f"Error getting portfolio state for personality: {e}")
            return {
                'positions': {},
                'current_position_size': 0,
                'unrealized_pnl': 0.0,
                'daily_pnl': 0.0,
                'recent_performance': []
            }

    def _trigger_personality_system_update(self):
        """Trigger personality commentary during system status updates"""
        if not self.personality:
            return
            
        try:
            # Get current system state for personality context
            portfolio_summary = self.portfolio.get_summary()
            agent_stats = self.agent.get_stats()
            intelligence_stats = self.intelligence.get_stats()
            
            # Create comprehensive context for periodic update
            context = {
                'subsystem_signals': {
                    'dna': float(intelligence_stats.get('dna_patterns', 0) / 300.0),  # Normalize
                    'temporal': float(intelligence_stats.get('temporal_patterns', 0) / 50.0),
                    'immune': float(intelligence_stats.get('immune_patterns', 0) / 10.0),
                    'microstructure': float(intelligence_stats.get('micro_patterns', 0) / 100.0),
                    'dopamine': 0.5,  # Would need actual dopamine signal
                    'regime': 0.0     # Would need actual regime signal
                },
                'market_data': {
                    'price': 0.0,  # Current price not available in status context
                    'volatility': 0.02,  # Default volatility
                    'trend_strength': 0.0,
                    'volume_regime': 0.5,
                    'regime': 'normal'
                },
                'portfolio_state': self._get_portfolio_state(),
                'decision_context': {
                    'decision_type': 'periodic_update',
                    'confidence': float(agent_stats.get('success_rate', 0.5)),
                    'overall_signal': 0.0,
                    'total_decisions': self.total_decisions,
                    'recent_activity': 'monitoring' if self.total_decisions % 10 > 5 else 'active',
                    'agent_learning_efficiency': float(agent_stats.get('learning_efficiency', 0.0)),
                    'recent_rewards': agent_stats.get('recent_rewards', []),
                    'current_strategy': agent_stats.get('current_strategy', 'unknown'),
                    'exploration_rate': agent_stats.get('exploration_rate', 0.0),
                    'meta_learner_updates': agent_stats.get('meta_learner_updates', 0),
                    'successful_adaptations': agent_stats.get('successful_adaptations', 0)
                },
                'system_performance': {
                    'data_updates': self.data_updates_received,
                    'decisions_made': self.total_decisions,
                    'win_rate': portfolio_summary.get('win_rate', 0.0),
                    'daily_pnl': portfolio_summary.get('daily_pnl', 0.0)
                }
            }
            
            # Trigger async commentary (don't wait for it)
            import asyncio
            try:
                # Try to run in existing event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule as task in running loop
                    loop.create_task(self._async_personality_system_commentary(context))
                else:
                    # Create new event loop
                    asyncio.run(self._async_personality_system_commentary(context))
            except RuntimeError:
                # No event loop, create new one
                asyncio.run(self._async_personality_system_commentary(context))
                
        except Exception as e:
            logger.error(f"Error triggering personality system update: {e}")

    async def _async_personality_system_commentary(self, context):
        """Async handler for periodic personality commentary"""
        try:
            commentary = await self.personality.process_trading_event(TriggerEvent.PERIODIC_UPDATE, context)
            if commentary:
                # Log the commentary prominently 
                logger.info(f"AI {self.personality.config.personality_name}: {commentary.text}")
                
        except Exception as e:
            logger.error(f"Error in async personality system commentary: {e}")