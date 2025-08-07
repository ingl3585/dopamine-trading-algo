# trading_system.py - Unified Trading System Orchestrator

import asyncio
import logging
import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

# Core system imports
from src.core.market_data_processor import MarketDataProcessor as DataProcessor
from src.intelligence.intelligence_engine import IntelligenceEngine
from src.agent.trading_agent_v2 import TradingAgent
from src.risk.risk_manager import RiskManager
from src.communication.tcp_bridge import TCPServer
from src.portfolio.portfolio import Portfolio
from src.core.config_manager import Config
from src.core.dependency_registry import registry, register_service
from src.core.state_coordinator import state_coordinator, register_state_component
from src.core.event_bus import EventBus, EventType, EventDrivenComponent
from src.shared.types import TradeDecision

logger = logging.getLogger(__name__)

# AI Trading Personality Integration
try:
    from src.personality.config_manager import PersonalityConfigManager
    from src.personality.trading_personality import TradingPersonality, TriggerEvent
    from src.personality.personality_integration import PersonalityIntegration, PersonalityIntegrationConfig
    PERSONALITY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Personality system not available: {e}")
    PERSONALITY_AVAILABLE = False

@dataclass
class SystemHealth:
    """System health status"""
    overall_health: str  # healthy, warning, critical
    component_health: Dict[str, str]
    event_bus_health: str
    integration_health: str
    performance_health: str
    issues: List[str]
    recommendations: List[str]

class TradingSystem:
    """
    Unified Trading System Orchestrator
    
    Combines the best features from all orchestrator implementations:
    - Production-ready TCP integration with NinjaTrader
    - Dependency injection and state coordination
    - Event-driven architecture with health monitoring
    - Async/await patterns for optimal performance
    - Clean separation of concerns following SOLID principles
    
    Architecture:
    - Core Infrastructure: TCP bridge, data processing, configuration
    - AI Intelligence: Multi-subsystem analysis and pattern recognition
    - Trading Agent: Reinforcement learning with dopamine pathways
    - Risk Management: Dynamic risk assessment and position sizing
    - Portfolio Management: Position tracking and performance analysis
    - Event System: Async event-driven component communication
    - Health Monitoring: System resilience and error recovery
    """
    def __init__(self, config_file: Optional[str] = None):
        logger.info("Initializing unified trading system orchestrator")
        
        # Core configuration and state
        self.config = Config(config_file) if config_file else Config()
        self.running = False
        self.system_initialized = False
        self.bootstrap_complete = False
        
        # Initialize core components
        self.portfolio = Portfolio()
        self.data_processor = DataProcessor(self.config.settings)
        self.intelligence = IntelligenceEngine(self.config)
        self.agent = TradingAgent(self.intelligence, self.portfolio)
        self.risk_manager = RiskManager(self.portfolio, self.agent.meta_learner, self.agent)
        
        # Event-driven architecture
        event_config = {
            'max_event_queue_size': 1000,
            'event_thread_pool_size': 4,
            'event_history_size': 10000
        }
        self.event_bus = EventBus(event_config)
        self.event_handlers = {}
        
        # System state tracking
        self.last_account_update = time.time()
        self.ready_for_trading = False
        self.total_decisions = 0
        self.data_updates_received = 0
        self.trade_count = 0
        self.last_decision_time = None
        
        # Account monitoring
        self.last_account_balance = 0.0
        self.account_change_threshold = 0.05
        
        # Health monitoring (currently handled by event system)
        # Future enhancement: dedicated SystemHealthMonitor for advanced system diagnostics
        # self.health_monitor = SystemHealthMonitor(self.event_bus)
        # self.health_data = {}
        
        # Historical data for bootstrap
        self.historical_data_cache = []
        self.current_market_data = {}
        
        # Setup core systems
        self._setup_dependency_injection()
        self._setup_state_coordination()
        self._setup_event_system()
        self._setup_tcp_server()
        
        # Initialize personality system
        self._setup_personality_system()
        
        # Load previous state if available using state coordinator
        self._load_coordinated_state()
        
        logger.info("Unified trading system orchestrator initialized successfully")
    
    def _setup_tcp_server(self):
        """Setup TCP server for NinjaTrader communication"""
        try:
            self.tcp_server = TCPServer(
                data_port=self.config.get('tcp_data_port', 5556),
                signal_port=self.config.get('tcp_signal_port', 5557)
            )
            self.tcp_server.on_market_data = self._process_market_data
            self.tcp_server.on_trade_completion = self._process_trade_completion
            self.tcp_server.on_historical_data = self._process_historical_data
            
            logger.info("TCP server configured for NinjaTrader communication")
        except Exception as e:
            logger.error(f"Failed to setup TCP server: {e}")
            raise
    
    def _setup_event_system(self):
        """Setup event-driven architecture"""
        try:
            # Register core event handlers
            self.event_bus.register_handler(
                'system_events',
                self._handle_system_events,
                [EventType.SYSTEM_STARTED, EventType.SYSTEM_STOPPED, EventType.SYSTEM_ERROR],
                priority=9
            )
            
            self.event_bus.register_handler(
                'market_events',
                self._handle_market_events,
                [EventType.MARKET_DATA_RECEIVED, EventType.PRICE_CHANGE],
                priority=8
            )
            
            self.event_bus.register_handler(
                'trading_events',
                self._handle_trading_events,
                [EventType.TRADE_SIGNAL_GENERATED, EventType.POSITION_OPENED, EventType.POSITION_CLOSED],
                priority=7
            )
            
            logger.info("Event system initialized")
        except Exception as e:
            logger.error(f"Failed to setup event system: {e}")
    
    def _setup_personality_system(self):
        """Setup AI trading personality integration"""
        self.personality = None
        self.personality_integration = None
        
        if PERSONALITY_AVAILABLE:
            try:
                personality_config_manager = PersonalityConfigManager()
                integration_config = personality_config_manager.get_integration_config()
                if integration_config.enabled:
                    integration_config_obj = PersonalityIntegrationConfig(
                        enabled=True,
                        personality_name=integration_config.personality_name,
                        auto_commentary=integration_config.auto_commentary,
                        llm_model=integration_config.llm_model,
                        llm_api_key=integration_config.llm_api_key
                    )
                    self.personality_integration = PersonalityIntegration(integration_config_obj)
                    self.personality = self.personality_integration.personality
                    logger.info(f"AI Trading Personality '{integration_config.personality_name}' initialized")
                else:
                    logger.info("AI Trading Personality disabled in configuration")
            except Exception as e:
                logger.error(f"Failed to initialize AI Trading Personality: {e}")
                self.personality = None
                self.personality_integration = None
    
    def _setup_dependency_injection(self):
        """Setup dependency injection to break circular imports"""
        try:
            # Register key services that might have circular dependencies
            register_service('config', type(self.config), lambda: self.config)
            register_service('portfolio', type(self.portfolio), lambda: self.portfolio)
            register_service('intelligence', type(self.intelligence), lambda: self.intelligence)
            register_service('agent', type(self.agent), lambda: self.agent)
            register_service('risk_manager', type(self.risk_manager), lambda: self.risk_manager)
            
            # Register adaptation engine when needed
            def create_adaptation_engine():
                from src.agent.real_time_adaptation import RealTimeAdaptationEngine
                return RealTimeAdaptationEngine(model_dim=64)
            
            register_service('adaptation_engine', None, create_adaptation_engine)
            
            logger.info("Dependency injection setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup dependency injection: {e}")
            raise
    
    def _setup_state_coordination(self):
        """Setup coordinated state management for all components"""
        try:
            # Register portfolio state management
            register_state_component(
                'portfolio',
                lambda: self.portfolio.get_persistent_state(),
                lambda state: self.portfolio.load_persistent_state(state),
                priority=10
            )
            
            # Register agent state management (high priority)
            register_state_component(
                'agent',
                lambda: {'model_state': 'placeholder'},  # Agent will implement proper state methods
                lambda state: None,  # Agent will implement proper load
                priority=20
            )
            
            # Register intelligence state management
            register_state_component(
                'intelligence',
                lambda: self.intelligence.save_patterns('data/intelligence_state.json'),  # Use proper filepath
                lambda state: None,  # Intelligence will implement proper load
                priority=15
            )
            
            # Enable auto-save every 5 minutes
            state_coordinator.enable_auto_save(300)
            
            logger.info("State coordination setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup state coordination: {e}")
            raise
    
    def _load_coordinated_state(self):
        """Load state using state coordinator"""
        try:
            latest_state_file = state_coordinator.get_latest_state_file()
            if latest_state_file:
                logger.info(f"Loading coordinated state from {latest_state_file}")
                state_coordinator.load_state(str(latest_state_file))
            else:
                logger.info("No previous state file found, starting fresh")
        except Exception as e:
            logger.error(f"Failed to load coordinated state: {e}")
            # Fallback to individual component loading
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
        import threading
        
        # Use event-driven approach instead of polling
        stop_event = threading.Event()
        
        def periodic_tasks():
            while self.running and not stop_event.is_set():
                current_time = time.time()
                
                # Check auto-save every 5 minutes (state coordinator handles the timing)
                state_coordinator.check_auto_save()
                
                # Wait with interruptible sleep
                stop_event.wait(timeout=60)  # Check every minute instead of every 0.1s
        
        # Start periodic tasks in background
        periodic_thread = threading.Thread(target=periodic_tasks, daemon=True)
        periodic_thread.start()
        
        # Main loop for readiness check (minimal CPU usage)
        while self.running:
            # Wait for historical data processing before starting trading
            if not self.ready_for_trading:
                if self.tcp_server.is_ready_for_live_trading() and self.intelligence.historical_processed:
                    self.ready_for_trading = True
                    logger.info("=== READY FOR LIVE TRADING ===")
                    self._log_bootstrap_summary()
                    # Switch to event-driven mode after ready
                    break
                else:
                    time.sleep(1.0)  # Check readiness every second, not 0.1s
            else:
                break
        
        # Once ready, minimal polling for shutdown
        while self.running:
            time.sleep(5.0)  # Much less frequent polling
        
        # Signal periodic tasks to stop
        stop_event.set()
        periodic_thread.join(timeout=2)
            

    def _process_historical_data(self, historical_data):
        """Process historical data for pattern bootstrapping and priming the data processor."""
        try:
            logger.info("Processing historical data for pattern learning...")
            
            # Validate historical data quality
            if not self.data_processor.validate_historical_data(historical_data):
                logger.error("Historical data validation failed - system not ready for trading")
                return
            
            # Bootstrap the intelligence engine with historical patterns
            self.intelligence.bootstrap_from_historical_data(historical_data)

            # Prime the data processor with the same historical data
            self.data_processor.prime_with_historical_data(historical_data)
            logger.info("Data processor primed with historical data.")
            
            # Signal that historical processing is complete
            logger.info("Historical data processing complete")
            
            # Generate initial personality commentary after learning
            self._generate_initial_commentary()
            
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
            
            # Check for new higher timeframe bars and trigger enhanced analysis
            if self.data_processor.check_and_reset_15m_bar_flag():
                logger.info(f"New 15m bar: {market_data.price:.2f} (Account: ${market_data.account_balance:.0f}, Daily P&L: ${market_data.daily_pnl:.2f})")
                self._trigger_15m_commentary(market_data)
            
            # New 1H bar triggers enhanced regime analysis
            if self.data_processor.check_and_reset_1h_bar_flag():
                logger.info(f"New 1H bar: {market_data.price:.2f} - Triggering enhanced regime analysis")
                self._trigger_1h_analysis(market_data)
            
            # New 4H bar triggers major trend analysis  
            if self.data_processor.check_and_reset_4h_bar_flag():
                logger.info(f"New 4H bar: {market_data.price:.2f} - Triggering major trend analysis")
                self._trigger_4h_analysis(market_data)
            
            # Check for significant account changes and adapt
            self._check_account_adaptation(market_data)
                
            features = self.intelligence.extract_features(market_data)
            
            # Simplified intelligence logging every 20 updates
            if self.data_updates_received % 20 == 0:
                logger.info(f"Intelligence: Overall={features.overall_signal:.3f}, Confidence={features.confidence:.3f}")
            
            decision = self.agent.decide(features, market_data)
            self.total_decisions += 1
            
            # Enhanced decision logging with confidence monitoring
            confidence_status = "CRITICAL" if decision.confidence < 0.2 else "LOW" if decision.confidence < 0.4 else "NORMAL"
            confidence_change = ""
            
            # Track confidence changes
            if hasattr(self.agent, '_last_logged_confidence'):
                change = decision.confidence - self.agent._last_logged_confidence
                confidence_change = f" ({change:+.3f})"
            self.agent._last_logged_confidence = decision.confidence
            
            # Log detailed confidence information
            logger.info(f"Decision #{self.total_decisions}: {decision.action.upper()} "
                       f"(Size: {decision.size:.1f}, Conf: {decision.confidence:.3f}{confidence_change} [{confidence_status}], "
                       f"Tool: {decision.primary_tool}, Exploration: {decision.exploration})")
            
            # Additional confidence monitoring for problematic values
            if decision.confidence < 0.3:
                recovery_factor = getattr(self.agent, 'confidence_recovery_factor', 1.0)
                logger.warning(f"LOW CONFIDENCE DETECTED: {decision.confidence:.3f} "
                              f"(Recovery factor: {recovery_factor:.2f})")
                # Position rejection tracking removed
            
            # LLM commentary now handled by 15-minute bar triggers only
            # (Per-decision commentary disabled to avoid duplicate messages)
            
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
        """Thread-safe sequential learning coordinator - eliminates race conditions"""
        import threading
        
        # Use lock to ensure atomic learning sequence
        if not hasattr(self, '_learning_lock'):
            self._learning_lock = threading.Lock()
        
        with self._learning_lock:
            try:
                # Phase 1: Intelligence subsystems learn first (establish base patterns)
                logger.debug("Phase 1: Intelligence learning from outcome")
                self.intelligence.learn_from_outcome(outcome)
                
                # Phase 2: Agent learns second (uses updated subsystem state)
                logger.debug("Phase 2: Agent learning from outcome") 
                self.agent.learn_from_trade(outcome)
                
                # Phase 3: Risk learning engine updates (uses both intelligence and agent state)
                logger.debug("Phase 3: Risk learning from outcome")
                if hasattr(self.risk_manager, 'risk_learning'):
                    self.risk_manager.risk_learning.learn_from_outcome(outcome)
                
                logger.debug("Sequential learning completed successfully")
                
            except Exception as e:
                logger.error(f"Error in sequential learning: {e}")
                # Continue operation even if learning fails
                pass

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
                
                # Debug: Log portfolio statistics after trade completion
                portfolio_summary = self.portfolio.get_summary()
                logger.info(f"Portfolio after trade: {portfolio_summary['total_trades']} trades, "
                           f"Win rate: {portfolio_summary['win_rate']:.1%}, "
                           f"Daily P&L: ${portfolio_summary['daily_pnl']:.2f}")
                
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
            logger.info(f"Subsystem Patterns: DNA={intelligence_stats['dna_patterns']} (sequences), "
                       f"Micro={intelligence_stats['micro_patterns']} (analyzed), "
                       f"Temporal={intelligence_stats['temporal_patterns']} (cycles), "
                       f"Immune={intelligence_stats['immune_patterns']} (antibodies), "
                       f"Dopamine={intelligence_stats['dopamine_patterns']} (updates)")
            logger.info("System ready for live trading with pre-learned patterns")
            logger.info("=" * 30)
            
        except Exception as e:
            logger.error(f"Error logging bootstrap summary: {e}")

    def _log_detailed_status(self):
        """DEPRECATED: Heavy system monitoring replaced with 15m commentary"""
        # This method was called every 2 minutes and generated excessive logs
        # Now replaced with intelligent commentary triggered on 15-minute bars
        pass

    def _check_account_adaptation(self, market_data):
        """Thread-safe account balance change detection and adaptation"""
        if not hasattr(self, '_account_lock'):
            import threading
            self._account_lock = threading.Lock()
        
        with self._account_lock:
            current_balance = market_data.account_balance
            current_time = time.time()
            
            # Initialize on first run
            if self.last_account_balance == 0.0:
                self.last_account_balance = current_balance
                self.last_account_update = current_time
                logger.info(f"Initial account balance: ${current_balance:.2f}")
                return
            
            # Rate limiting: only check every 30 seconds to avoid race conditions
            if current_time - self.last_account_update < 30:
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
                self.last_account_update = current_time
    
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
                       f"Immune={intelligence_stats['immune_patterns']}, "
                       f"Dopamine={intelligence_stats['dopamine_patterns']}")
            
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

    def _generate_initial_commentary(self):
        """Generate initial personality commentary after learning from historical data"""
        if not self.personality_integration or not self.personality_integration.is_enabled():
            return
            
        try:
            # Get initial system state after learning
            agent_stats = self.agent.get_stats()
            intelligence_stats = self.intelligence.get_stats()
            
            # Create dummy market data for initial context (using last known data)
            dummy_market_data = type('MockMarketData', (), {
                'price': 0.0,
                'account_balance': 25000.0,
                'daily_pnl': 0.0,
                'net_liquidation': 25000.0,
                'timestamp': time.time(),
                'prices_1m': [22400.0] * 25,  # Dummy price data
                'volumes_1m': [1000] * 25,   # Dummy volume data
                'prices_5m': [22400.0] * 10,
                'volumes_5m': [5000] * 10,
                'prices_15m': [22400.0] * 5,
                'volumes_15m': [15000] * 5,
                'volatility': 0.02,
                'position_size': 0,
                'unrealized_pnl': 0.0,
                'margin_utilization': 0.0,
                'buying_power': 25000.0,
                'buying_power_ratio': 1.0,
                'daily_pnl_pct': 0.0
            })()
            
            # Extract initial features
            features = self.intelligence.extract_features(dummy_market_data)
            
            logger.info(f"Initial Commentary Trigger: Learning complete, "
                       f"Neural confidence={features.confidence:.3f}, "
                       f"Patterns learned={intelligence_stats.get('total_patterns', 0)}")
            
            # Generate initial commentary
            import asyncio
            import threading
            
            def run_initial_commentary():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    # Use special trigger for initial commentary
                    from src.personality.trading_personality import TriggerEvent
                    commentary = loop.run_until_complete(
                        self.personality_integration.process_trading_decision(
                            None,  # No decision yet
                            features,
                            dummy_market_data,
                            self.agent,
                            trigger_event=TriggerEvent.SYSTEM_START
                        )
                    )
                    
                    if commentary:
                        print(f"\n{'='*80}")
                        print(f"[DOPAMINE INITIAL ANALYSIS] {commentary}")
                        print(f"{'='*80}\n")
                        logger.info(f"[INITIAL] {commentary}")
                    
                    loop.close()
                except Exception as e:
                    logger.warning(f"Initial commentary failed: {e}")
            
            # Run in background thread
            commentary_thread = threading.Thread(target=run_initial_commentary, daemon=True)
            commentary_thread.start()
            
        except Exception as e:
            logger.error(f"Error generating initial commentary: {e}")

    def _trigger_15m_commentary(self, market_data):
        """Trigger LLM commentary on new 15-minute bars"""
        if not self.personality_integration:
            return
            
        try:
            # Get current system state for comprehensive context
            portfolio_summary = self.portfolio.get_summary()
            agent_stats = self.agent.get_stats()
            intelligence_stats = self.intelligence.get_stats()
            
            # Extract features for current market conditions
            features = self.intelligence.extract_features(market_data)
            
            logger.info(f"15m Commentary Trigger: Price={market_data.price:.2f}, "
                       f"Overall_Signal={features.overall_signal:.3f}, "
                       f"Trades={portfolio_summary['total_trades']}, "
                       f"P&L=${portfolio_summary['daily_pnl']:.2f}")
            
            # Trigger the periodic commentary function (now on 15m bars)
            import asyncio
            if not hasattr(self, '_commentary_loop'):
                # Create event loop for async commentary if needed
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Run commentary generation
                commentary = loop.run_until_complete(
                    self.personality_integration.periodic_commentary()
                )
                
                if commentary:
                    # Single unified output for 15-minute commentary
                    logger.info(f"\n\n[DOPAMINE] {commentary}\n")
                else:
                    logger.debug("No commentary generated for this 15m bar")
                    
        except Exception as e:
            logger.error(f"Error triggering 15m commentary: {e}")

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
    
    def _save_coordinated_state(self):
        """Save state using state coordinator (replaces _save_state)"""
        try:
            logger.debug("Triggering coordinated state save")
            success = state_coordinator.save_state()
            if success:
                logger.info("Coordinated state save completed successfully")
            else:
                logger.warning("Coordinated state save had some failures")
        except Exception as e:
            logger.error(f"Error in coordinated state save: {e}")
            # Fallback to old method
            self._save_state()
    
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
        
        self._save_coordinated_state()
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
                       f"Temporal={intelligence_stats['temporal_patterns']}, "
                       f"Immune={intelligence_stats['immune_patterns']}, "
                       f"Dopamine={intelligence_stats['dopamine_patterns']}")
        
        # Shutdown personality system
        if self.personality:
            self.personality.shutdown()
            
        logger.info("Trading system shutdown complete")

    def _handle_system_events(self, event):
        """Handle system-level events"""
        logger.debug(f"System event received: {event}")
        
    def _handle_market_events(self, event):
        """Handle market-related events"""
        logger.debug(f"Market event received: {event}")
        
    def _handle_trading_events(self, event):
        """Handle trading-related events"""
        logger.debug(f"Trading event received: {event}")

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
            return {
                'positions': dict(self.portfolio.positions) if hasattr(self.portfolio, 'positions') else {},
                'unrealized_pnl': float(summary.get('total_pnl', 0.0)),
                'daily_pnl': float(summary.get('daily_pnl', 0.0)),
                'recent_performance': [trade.pnl for trade in list(self.portfolio.trade_history)[-5:]] if hasattr(self.portfolio, 'trade_history') else []
            }
        except Exception as e:
            logger.warning(f"Error getting portfolio state for personality: {e}")
            return {
                'positions': {},
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
            
            # Old periodic commentary system removed - using enhanced real-time system instead
            pass
                
        except Exception as e:
            logger.error(f"Error triggering personality system update: {e}")

    # Removed old periodic commentary system - now using enhanced real-time system
    
    def _trigger_1h_analysis(self, market_data):
        """Trigger enhanced regime and trend analysis on new 1H bars"""
        try:
            # Extract enhanced features with multi-timeframe analysis
            features = self.intelligence.extract_features(market_data)
            
            # Log enhanced 1H analysis
            logger.info(f"1H ANALYSIS - Trend: {features.higher_tf_bias:.4f}, "
                       f"1H TF: {features.trend_1h:.4f}, "
                       f"Alignment: {features.trend_alignment:.3f}, "
                       f"Regime: {features.regime_confidence:.3f}")
            
            # Trigger adaptation engine update with enhanced context
            if hasattr(self.agent, 'adaptation_engine'):
                self.agent.adaptation_engine.process_regime_change({
                    'timeframe': '1H',
                    'trend_1h': features.trend_1h,
                    'higher_tf_bias': features.higher_tf_bias,
                    'trend_alignment': features.trend_alignment,
                    'volatility_1h': features.volatility_1h
                })
            
            # Update meta-learner with higher timeframe insights
            self.agent.meta_learner.update_regime_parameters({
                'trend_strength_1h': abs(features.trend_1h),
                'trend_direction_1h': 1 if features.trend_1h > 0 else -1,
                'volatility_regime_1h': 'high' if features.volatility_1h > 0.03 else 'normal'
            })
            
        except Exception as e:
            logger.error(f"Error in 1H analysis: {e}")
    
    def _trigger_4h_analysis(self, market_data):
        """Trigger major trend and bias analysis on new 4H bars"""
        try:
            # Extract enhanced features with multi-timeframe analysis
            features = self.intelligence.extract_features(market_data)
            
            # Log major 4H trend analysis
            logger.info(f"4H MAJOR TREND ANALYSIS - Bias: {features.higher_tf_bias:.4f}, "
                       f"4H TF: {features.trend_4h:.4f}, "
                       f"Multi-TF Alignment: {features.trend_alignment:.3f}, "
                       f"Vol 4H: {features.volatility_4h:.4f}")
            
            # Update trading bias based on 4H analysis
            trend_4h = features.trend_4h
            if abs(trend_4h) > 0.005:  # Significant 4H trend
                bias_strength = min(abs(trend_4h) * 10, 1.0)  # Scale to 0-1
                bias_direction = 1 if trend_4h > 0 else -1
                
                logger.info(f"4H BIAS UPDATE: Direction={bias_direction}, Strength={bias_strength:.3f}")
                
                # Update meta-learner with major trend bias
                self.agent.meta_learner.update_trend_bias({
                    'direction': bias_direction,
                    'strength': bias_strength,
                    'timeframe': '4H',
                    'confidence': features.regime_confidence
                })
            
            # Trigger portfolio rebalancing assessment for major trend changes
            portfolio_summary = self.portfolio.get_summary()
            current_position = portfolio_summary.get('total_position_size', 0)
            
            # Alert if position is against major 4H trend
            if current_position != 0 and trend_4h != 0:
                position_bias = 1 if current_position > 0 else -1
                trend_bias = 1 if trend_4h > 0 else -1
                
                if position_bias != trend_bias and abs(trend_4h) > 0.01:
                    logger.warning(f"POSITION VS 4H TREND CONFLICT: Position={position_bias}, 4H Trend={trend_bias:.4f}")
            
            # Enhanced personality commentary for major timeframe changes
            if self.personality_integration:
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                commentary = loop.run_until_complete(
                    self.personality_integration.process_market_event(
                        '4h_trend_change',
                        {
                            'trend_4h': trend_4h,
                            'higher_tf_bias': features.higher_tf_bias,
                            'trend_alignment': getattr(features, 'trend_alignment', 0.0),
                            'current_position': current_position
                        }
                    )
                )
                
                if commentary:
                    logger.info(f"\n\n[DOPAMINE 4H ANALYSIS] {commentary}\n")
            
        except Exception as e:
            logger.error(f"Error in 4H analysis: {e}")


# Note: Main entry point moved to /main.py for better organization and features