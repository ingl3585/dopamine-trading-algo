"""
Main Entry Point - Coordinates all domains using DDD architecture
"""

import asyncio
import logging
import sys
from datetime import datetime
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor

from src.core.config import Config
from src.intelligence import create_intelligence_engine
from src.services import create_trading_service
from src.market_data import create_market_processor
from src.risk import create_risk_manager
from src.shared.types import TradeDecision
from src.communication.tcp_bridge import TCPServer

logger = logging.getLogger(__name__)

class TradingSystemOrchestrator:
    """
    Main system orchestrator coordinating all domains
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.running = False
        
        # Initialize all domains using factories
        logger.info("Initializing AI domain...")
        self.intelligence_engine = create_intelligence_engine(config)
        
        logger.info("Initializing Trading domain...")
        self.trading_service = create_trading_service(config)
        
        logger.info("Initializing Market domain...")
        self.market_processor = create_market_processor(config)
        
        logger.info("Initializing Risk domain...")
        self.risk_manager = create_risk_manager(config)
        
        # System state
        self.bootstrap_complete = False
        self.trade_count = 0
        self.last_decision_time = None
        
        # Component registry for integration
        self.registered_components = {}
        
        # TCP Bridge for NinjaTrader
        self.tcp_server = TCPServer(
            data_port=config.get('tcp_data_port', 5556),
            signal_port=config.get('tcp_signal_port', 5557)
        )
        self.historical_data_cache = []
        self.current_market_data = {}
        
        # Setup TCP callbacks
        self.tcp_server.on_market_data = self._handle_market_data
        self.tcp_server.on_historical_data = self._handle_historical_data
        self.tcp_server.on_trade_completion = self._handle_trade_completion
        
        # Link TCP server to trading service
        if hasattr(self.trading_service, 'repository') and hasattr(self.trading_service.repository, 'set_tcp_server'):
            self.trading_service.repository.set_tcp_server(self.tcp_server)
        
        logger.info("Trading system orchestrator initialized successfully")
    
    def register_component(self, name: str, component, startup_callback=None, shutdown_callback=None):
        """Register a component with the orchestrator for integration"""
        try:
            self.registered_components[name] = {
                'component': component,
                'startup_callback': startup_callback,
                'shutdown_callback': shutdown_callback
            }
            logger.info(f"Registered component: {name}")
            
            # Execute startup callback if provided
            if startup_callback:
                startup_callback()
                
        except Exception as e:
            logger.error(f"Error registering component {name}: {e}")
    
    def start_system(self) -> bool:
        """Start the trading system (synchronous wrapper for async start)"""
        try:
            import asyncio
            
            # Check if we're already in an event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, create a task instead
                logger.info("Creating async task for system start...")
                task = loop.create_task(self.start())
                logger.info("Trading system start task created successfully")
                return True
            except RuntimeError:
                # No running loop, we can run directly
                logger.info("Starting trading system...")
                asyncio.run(self.start())
                return True
                
        except Exception as e:
            logger.error(f"Error starting trading system: {e}")
            return False
    
    async def start(self):
        """Start the complete trading system"""
        try:
            logger.info("Starting Actor-Critic ML Trading System...")
            
            # Phase 1: Start TCP Server and wait for NinjaTrader connection
            await self._start_tcp_server()
            
            # Phase 2: Historical Bootstrap
            if not self.bootstrap_complete:
                await self._bootstrap_historical_data()
            
            # Phase 3: Live Trading Loop
            self.running = True
            await self._run_live_trading_loop()
            
        except Exception as e:
            logger.error(f"Critical error in trading system: {e}")
            await self.shutdown()
    
    async def _bootstrap_historical_data(self):
        """Bootstrap AI subsystems with historical data"""
        try:
            logger.info("Starting historical bootstrap phase...")
            
            # Get historical market data from NinjaTrader
            historical_data = await self._get_historical_data()
            
            if not historical_data or len(historical_data) < 100:
                logger.warning("Insufficient historical data for bootstrap")
                self.bootstrap_complete = True
                return
            
            logger.info(f"Bootstrapping with {len(historical_data)} historical data points...")
            
            # Process historical data and train subsystems
            bootstrap_count = 0
            for data_point in historical_data:
                try:
                    # Process historical market data (no account fields expected)
                    market_data = self.market_processor.process_data(data_point, is_historical=True)
                    market_features = self.market_processor.extract_features(market_data)
                    
                    # Get historical prices and volumes for AI analysis
                    historical_context = self.market_processor.get_historical_data(50)
                    
                    # Analyze with AI subsystems
                    ai_signals = self.intelligence_engine.analyze_market(historical_context, market_features)
                    
                    # Simulate trading decision for learning
                    overall_signal = ai_signals['overall'].value
                    
                    # Create learning context for subsystems with standardized market_state
                    standardized_market_state = self._create_standardized_market_state(market_features, historical_context)
                    
                    learning_context = {
                        'dna_sequence': self._extract_dna_sequence(historical_context, market_features),
                        'cycles_info': self._extract_cycles_info(ai_signals),
                        'market_state': standardized_market_state,
                        'microstructure_signal': ai_signals['microstructure'].value,
                        'is_bootstrap': True
                    }
                    
                    # Simulate outcome for learning (use next data point if available)
                    outcome = self._calculate_bootstrap_outcome(data_point, historical_data, bootstrap_count)
                    
                    # Learn from simulated outcome
                    self.intelligence_engine.learn_from_outcome(outcome, learning_context)
                    
                    bootstrap_count += 1
                    
                    if bootstrap_count % 50 == 0:
                        logger.info(f"Bootstrap progress: {bootstrap_count}/{len(historical_data)}")
                        
                except Exception as e:
                    logger.warning(f"Error in bootstrap data point {bootstrap_count}: {e}")
                    continue
            
            logger.info(f"Historical bootstrap completed: {bootstrap_count} data points processed")
            self.bootstrap_complete = True
            
        except Exception as e:
            logger.error(f"Error in historical bootstrap: {e}")
            self.bootstrap_complete = True  # Continue with live trading
    
    async def _run_live_trading_loop(self):
        """Main live trading loop"""
        logger.info("Starting live trading loop...")
        
        while self.running:
            try:
                # Get current market data
                market_data_raw = await self._get_current_market_data()
                
                if not market_data_raw:
                    await asyncio.sleep(1)  # Wait before retry
                    continue
                
                # Process market data
                market_data = self.market_processor.process_data(market_data_raw)
                market_features = self.market_processor.extract_features(market_data)
                
                # Update trading service with latest account data from NinjaTrader
                if hasattr(self.trading_service, 'update_account_data'):
                    self.trading_service.update_account_data(market_data)
                
                # Get account information (now from real NinjaTrader data)
                account_info = self.trading_service.get_account_info()
                
                # Get historical context for AI analysis
                historical_context = self.market_processor.get_historical_data(100)
                
                # AI Analysis - Get signals from all subsystems
                ai_signals = self.intelligence_engine.analyze_market(historical_context, market_features)
                
                # Risk Assessment
                trade_decision = self._create_trade_decision(ai_signals, market_features)
                risk_level = self.risk_manager.assess_risk(trade_decision, account_info)
                
                # Position Sizing
                position_size = self.risk_manager.size_position(
                    ai_signals['overall'].value, account_info, risk_level
                )
                
                # Update trade decision with risk-adjusted size
                trade_decision.size = position_size
                
                # Execute Trade (if signal is strong enough and risk is acceptable)
                if self._should_execute_trade(trade_decision, risk_level, ai_signals):
                    trade_outcome = self.trading_service.execute_trade(trade_decision)
                    
                    # Update risk manager with outcome
                    if trade_outcome.success:
                        self.risk_manager.update_trade_outcome(
                            trade_outcome.pnl, trade_outcome.duration
                        )
                        
                        # Learn from trade outcome with standardized market_state
                        standardized_market_state = self._create_standardized_market_state(market_features, historical_context)
                        
                        learning_context = {
                            'dna_sequence': self._extract_dna_sequence(historical_context, market_features),
                            'cycles_info': self._extract_cycles_info(ai_signals),
                            'market_state': standardized_market_state,
                            'microstructure_signal': ai_signals['microstructure'].value,
                            'is_bootstrap': False
                        }
                        
                        self.intelligence_engine.learn_from_outcome(trade_outcome.pnl, learning_context)
                        
                        self.trade_count += 1
                        self.last_decision_time = datetime.now()
                        
                        logger.info(f"Trade {self.trade_count} executed: {trade_decision.action} "
                                  f"size={position_size:.1f} pnl={trade_outcome.pnl:.2f}")
                
                # Portfolio Management
                positions = self.trading_service.manage_positions()
                portfolio_analytics = self.risk_manager.get_risk_metrics()
                
                # Log trading decision on each bar
                self._log_trading_decision(ai_signals, risk_level, market_data)
                
                # Wait before next iteration
                await asyncio.sleep(self.config.get('trading_interval_seconds', 60))
                
            except KeyboardInterrupt:
                logger.info("Received shutdown signal")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(5)  # Wait before retry
        
        logger.info("Live trading loop stopped")
    
    def _create_trade_decision(self, ai_signals: Dict, market_features: Dict) -> TradeDecision:
        """Create trade decision from AI signals"""
        overall_signal = ai_signals['overall'].value
        confidence = ai_signals['overall'].confidence
        
        # Determine action based on signal strength and confidence
        if overall_signal > 0.1 and confidence > 0.6:
            action = "buy"
        elif overall_signal < -0.1 and confidence > 0.6:
            action = "sell"
        else:
            action = "hold"
        
        return TradeDecision(
            action=action,
            size=1.0,  # Will be updated by risk management
            confidence=confidence,
            reasoning={
                'overall_signal': overall_signal,
                'dna_signal': ai_signals['dna'].value,
                'temporal_signal': ai_signals['temporal'].value,
                'immune_signal': ai_signals['immune'].value,
                'microstructure_signal': ai_signals['microstructure'].value,
                'consensus_strength': ai_signals.get('consensus_strength', 0.0),
                'volatility': market_features.get('volatility', 0.02)
            }
        )
    
    def _should_execute_trade(self, decision: TradeDecision, risk_level: float, ai_signals: Dict) -> bool:
        """Determine if trade should be executed"""
        # Don't trade if action is hold
        if decision.action == "hold":
            return False
        
        # Don't trade if risk is too high
        if risk_level > 0.8:
            logger.debug(f"Trade rejected: risk too high ({risk_level:.2f})")
            return False
        
        # Don't trade if confidence is too low
        if decision.confidence < 0.6:
            logger.debug(f"Trade rejected: confidence too low ({decision.confidence:.2f})")
            return False
        
        # Don't trade if consensus is too weak
        consensus = ai_signals.get('consensus_strength', 0.0)
        if consensus < 0.3:
            logger.debug(f"Trade rejected: consensus too weak ({consensus:.2f})")
            return False
        
        # Rate limiting - don't trade too frequently
        if self.last_decision_time:
            time_since_last = (datetime.now() - self.last_decision_time).total_seconds()
            min_interval = self.config.get('min_trade_interval_seconds', 300)  # 5 minutes
            
            if time_since_last < min_interval:
                logger.debug(f"Trade rejected: too soon since last trade ({time_since_last:.0f}s)")
                return False
        
        return True
    
    def _extract_dna_sequence(self, historical_context: Dict, market_features: Dict) -> str:
        """Extract DNA sequence for learning - ensures valid string format"""
        try:
            prices = historical_context.get('prices', [])
            volumes = historical_context.get('volumes', [])
            
            if len(prices) < 20 or len(volumes) < 20:
                # Return a minimal valid sequence instead of empty string
                vol = market_features.get('volatility', 0.02)
                momentum = market_features.get('price_momentum', 0.0)
                return f"MINIMAL_ABCD_{vol:.3f}_{momentum:.3f}"
            
            # Use orchestrator DNA subsystem to encode current market state
            if hasattr(self.intelligence_engine, 'orchestrator') and hasattr(self.intelligence_engine.orchestrator, 'dna_subsystem'):
                dna_sequence = self.intelligence_engine.orchestrator.dna_subsystem.encode_market_state(
                    prices[-20:], volumes[-20:],
                    market_features.get('volatility', 0.02),
                    market_features.get('price_momentum', 0.0)
                )
                # Ensure we have a valid sequence
                if not dna_sequence or len(dna_sequence) < 3:
                    vol = market_features.get('volatility', 0.02)
                    momentum = market_features.get('price_momentum', 0.0)
                    return f"FALLBACK_ABCD_{vol:.3f}_{momentum:.3f}"
                return dna_sequence
            else:
                # Simple DNA encoding fallback
                vol = market_features.get('volatility', 0.02)
                momentum = market_features.get('price_momentum', 0.0)
                return f"PATTERN_{len(prices)}_VOL_{vol:.3f}_MOM_{momentum:.3f}"
        except Exception as e:
            logger.error(f"Error extracting DNA sequence: {e}")
            # Return a valid fallback sequence instead of empty string
            return f"ERROR_FALLBACK_{hash(str(e)) % 10000}"
    
    def _create_standardized_market_state(self, market_features: Dict, historical_context: Dict) -> Dict:
        """Create standardized market state structure for all subsystems"""
        try:
            # Extract core market data
            volatility = market_features.get('volatility', 0.02)
            price_momentum = market_features.get('price_momentum', 0.0)
            volume_momentum = market_features.get('volume_momentum', 0.0)
            
            # Calculate time-based features
            current_time = datetime.now()
            time_of_day = (current_time.hour * 60 + current_time.minute) / (24 * 60)  # Normalized 0-1
            
            # Determine market regime based on volatility and momentum
            if volatility > 0.05:
                regime = "high_volatility"
            elif volatility < 0.01:
                regime = "low_volatility"
            elif abs(price_momentum) > 0.03:
                regime = "trending"
            else:
                regime = "ranging"
            
            # Create standardized structure
            standardized_state = {
                'volatility': volatility,
                'price_momentum': price_momentum,
                'volume_momentum': volume_momentum,
                'time_of_day': time_of_day,
                'regime': regime,
                'regime_confidence': market_features.get('regime_confidence', 0.5),
                # Additional fields for immune system
                'market_session': self._get_market_session(current_time),
                'volatility_percentile': min(1.0, volatility / 0.1),  # Normalized volatility
                'momentum_strength': abs(price_momentum),
                'volume_activity': abs(volume_momentum)
            }
            
            return standardized_state
            
        except Exception as e:
            logger.error(f"Error creating standardized market state: {e}")
            # Return minimal valid structure
            return {
                'volatility': 0.02,
                'price_momentum': 0.0,
                'volume_momentum': 0.0,
                'time_of_day': 0.5,
                'regime': 'ranging',
                'regime_confidence': 0.5,
                'market_session': 'regular',
                'volatility_percentile': 0.2,
                'momentum_strength': 0.0,
                'volume_activity': 0.0
            }
    
    def _get_market_session(self, current_time: datetime) -> str:
        """Determine market session based on time"""
        hour = current_time.hour
        if 9 <= hour <= 16:
            return "regular"
        elif 4 <= hour <= 9:
            return "premarket"
        elif 16 <= hour <= 20:
            return "afterhours"
        else:
            return "overnight"
    
    def _extract_cycles_info(self, ai_signals: Dict) -> list:
        """Extract cycles info for learning - ensures List[Dict] format"""
        try:
            # Get recent cycles from temporal subsystem via orchestrator
            if (hasattr(self.intelligence_engine, 'orchestrator') and 
                hasattr(self.intelligence_engine.orchestrator, 'temporal_subsystem') and
                hasattr(self.intelligence_engine.orchestrator.temporal_subsystem, 'dominant_cycles')):
                
                cycles = self.intelligence_engine.orchestrator.temporal_subsystem.dominant_cycles
                if len(cycles) > 0:
                    recent_cycle = list(cycles)[-1]
                    # Ensure we return List[Dict] format
                    if isinstance(recent_cycle, list):
                        return recent_cycle  # Already a list of dicts
                    elif isinstance(recent_cycle, dict):
                        return [recent_cycle]  # Wrap single dict in list
                    else:
                        logger.warning(f"Unexpected cycle format: {type(recent_cycle)}")
            
            # Fallback - create cycle dict from AI signals if available
            if 'temporal' in ai_signals and hasattr(ai_signals['temporal'], 'value'):
                # Temporal subsystem expects List[Dict], not List[float]
                cycle_dict = {
                    'frequency': 1.0/60.0,  # Default 1-hour cycle frequency
                    'amplitude': abs(ai_signals['temporal'].value),
                    'phase': 0.0,
                    'period': 60,  # Default 1-hour cycle
                    'window_size': 64  # Default window size
                }
                return [cycle_dict]
            
            return []
        except Exception as e:
            logger.error(f"Error extracting cycles info: {e}")
            return []
    
    def _calculate_bootstrap_outcome(self, current_data: Dict, historical_data: list, index: int) -> float:
        """Calculate outcome for bootstrap learning"""
        try:
            # Use next data point to calculate outcome
            if index + 1 < len(historical_data):
                next_data = historical_data[index + 1]
                current_price = current_data.get('close', 0.0)
                next_price = next_data.get('close', 0.0)
                
                if current_price > 0:
                    return (next_price - current_price) / current_price
            
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating bootstrap outcome: {e}")
            return 0.0
    
    async def _start_tcp_server(self):
        """Start TCP server and wait for NinjaTrader connection"""
        try:
            logger.info("Starting TCP server for NinjaTrader connection...")
            
            # Start TCP server in background thread
            with ThreadPoolExecutor() as executor:
                future = executor.submit(self.tcp_server.start)
                await asyncio.get_event_loop().run_in_executor(None, future.result)
            
            logger.info("TCP server started, waiting for historical data...")
            
        except Exception as e:
            logger.error(f"Error starting TCP server: {e}")
            raise
    
    async def _get_historical_data(self) -> list:
        """Get historical data for bootstrap from NinjaTrader"""
        try:
            logger.info("Waiting for historical data from NinjaTrader...")
            logger.info("ResearchStrategy.cs sends historical data on next full 1-minute bar - please wait...")
            
            # Wait for historical data to be received via TCP
            timeout_seconds = self.config.get('historical_data_timeout', 120)
            start_time = datetime.now()
            last_data_count = 0
            
            while True:
                # Check if we have received historical data
                current_data_count = len(self.historical_data_cache)
                if current_data_count > 0 and current_data_count != last_data_count:
                    logger.info(f"Historical data arriving: {current_data_count} data points received")
                    last_data_count = current_data_count
                    
                    # Wait a bit more to ensure all data is received
                    await asyncio.sleep(2)
                    if len(self.historical_data_cache) == current_data_count:
                        # Data count stabilized, we're done
                        break
                
                # Check timeout
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > timeout_seconds:
                    logger.warning(f"Historical data timeout after {timeout_seconds}s")
                    break
                
                # Log progress every 15 seconds
                if int(elapsed) % 15 == 0 and elapsed > 0:
                    logger.info(f"Still waiting for historical data... ({elapsed:.0f}s elapsed)")
                
                await asyncio.sleep(1)
            
            if self.historical_data_cache:
                logger.info(f"Historical data received: {len(self.historical_data_cache)} data points")
                return self.historical_data_cache
            else:
                logger.warning("No historical data received, proceeding with live data only")
                return []
                
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return []
    
    async def _get_current_market_data(self) -> Dict:
        """Get current market data from NinjaTrader"""
        try:
            if not self.current_market_data:
                return None
            
            # Extract OHLCV from TCP data
            tcp_data = self.current_market_data.copy()
            
            # Log what data we have for debugging
            logger.debug(f"Current market data keys: {list(tcp_data.keys())}")
            
            # Check if this is live data format from ResearchStrategy.cs
            data_type = tcp_data.get('type', 'unknown')
            
            if data_type == 'live_data' or ('price_1m' in tcp_data and 'account_balance' in tcp_data):
                # Use 1-minute price data if available
                prices_1m = tcp_data.get('price_1m', [])
                volumes_1m = tcp_data.get('volume_1m', [])
                
                logger.debug(f"Live data processing: {len(prices_1m)} prices, {len(volumes_1m)} volumes")
                
                if prices_1m and len(prices_1m) >= 4:
                    # Use last 4 prices as OHLC
                    recent_prices = prices_1m[-4:]
                    current_volume = volumes_1m[-1] if volumes_1m else 1000.0
                    
                    result = {
                        'timestamp': tcp_data.get('timestamp', datetime.now().timestamp()),
                        'open': recent_prices[0],
                        'high': max(recent_prices),
                        'low': min(recent_prices),
                        'close': recent_prices[-1],
                        'volume': current_volume,
                        # All required account data from NinjaTrader
                        'account_balance': tcp_data.get('account_balance', 0.0),
                        'buying_power': tcp_data.get('buying_power', 0.0),
                        'daily_pnl': tcp_data.get('daily_pnl', 0.0),
                        'unrealized_pnl': tcp_data.get('unrealized_pnl', 0.0),
                        'net_liquidation': tcp_data.get('net_liquidation', tcp_data.get('account_balance', 0.0)),
                        'margin_used': tcp_data.get('margin_used', 0.0),
                        'available_margin': tcp_data.get('available_margin', tcp_data.get('buying_power', 0.0)),
                        'open_positions': tcp_data.get('open_positions', 0),
                        'total_position_size': tcp_data.get('total_position_size', 0)
                    }
                    
                    logger.info(f"Live market data processed: Price={result['close']:.2f}, Volume={result['volume']:.0f}, Account=${result['account_balance']:.2f}")
                    return result
                    
                elif prices_1m and len(prices_1m) >= 1:
                    # Limited price data, use what we have
                    current_price = prices_1m[-1]
                    current_volume = volumes_1m[-1] if volumes_1m else 1000.0
                    
                    result = {
                        'timestamp': tcp_data.get('timestamp', datetime.now().timestamp()),
                        'open': current_price,
                        'high': current_price,
                        'low': current_price,
                        'close': current_price,
                        'volume': current_volume,
                        # All required account data from NinjaTrader
                        'account_balance': tcp_data.get('account_balance', 0.0),
                        'buying_power': tcp_data.get('buying_power', 0.0),
                        'daily_pnl': tcp_data.get('daily_pnl', 0.0),
                        'unrealized_pnl': tcp_data.get('unrealized_pnl', 0.0),
                        'net_liquidation': tcp_data.get('net_liquidation', tcp_data.get('account_balance', 0.0)),
                        'margin_used': tcp_data.get('margin_used', 0.0),
                        'available_margin': tcp_data.get('available_margin', tcp_data.get('buying_power', 0.0)),
                        'open_positions': tcp_data.get('open_positions', 0),
                        'total_position_size': tcp_data.get('total_position_size', 0)
                    }
                    
                    logger.info(f"Limited live market data processed: Price={result['close']:.2f}, Account=${result['account_balance']:.2f}")
                    return result
                else:
                    # No price data yet
                    logger.warning(f"Live data format detected but no price data: {list(tcp_data.keys())}")
                    return None
            else:
                # Unknown format or insufficient data
                logger.warning(f"Unrecognized market data format. Type: {data_type}, Keys: {list(tcp_data.keys())}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting current market data: {e}")
            return None
    
    def _log_trading_decision(self, ai_signals: Dict, risk_level: float, market_data):
        """Log simple trading decision on each bar"""
        try:
            # Extract key values (market_data is MarketData object, not Dict)
            price = market_data.close
            overall_signal = ai_signals['overall'].value
            overall_confidence = ai_signals['overall'].confidence
            consensus_strength = ai_signals.get('consensus_strength', 0.0)
            
            # Determine action
            if overall_signal > 0.1 and overall_confidence > 0.6 and consensus_strength > 0.3 and risk_level < 0.8:
                action = "BUY"
            elif overall_signal < -0.1 and overall_confidence > 0.6 and consensus_strength > 0.3 and risk_level < 0.8:
                action = "SELL"
            else:
                action = "HOLD"
            
            # Log simple one-line decision
            logger.info(f"Price: ${price:.2f} | Signal: {overall_signal:.3f} | Conf: {overall_confidence:.2f} | Consensus: {consensus_strength:.2f} | Risk: {risk_level:.2f} | Action: {action}")
            
        except Exception as e:
            logger.error(f"Error logging trading decision: {e}")
    
    def _handle_market_data(self, data: Dict):
        """Handle incoming market data from NinjaTrader"""
        try:
            self.current_market_data = data
            
            # Detect live data type and log appropriately
            data_type = data.get('type', 'unknown')
            if data_type == 'live_data':
                logger.info(f"Live market data received: {len(data.get('price_1m', []))} 1m prices, account_balance: {data.get('account_balance', 'N/A')}")
            elif data_type == 'historical_data':
                logger.info(f"Historical data received in live handler - redirecting")
                self._handle_historical_data(data)
            else:
                # Legacy format or unspecified - try to detect by fields
                if 'price_1m' in data and 'account_balance' in data:
                    logger.info(f"Live data detected (legacy format): {len(data.get('price_1m', []))} 1m prices")
                elif 'bars_15m' in data or 'bars_5m' in data or 'bars_1m' in data:
                    logger.info(f"Historical data detected in live handler - redirecting")
                    self._handle_historical_data(data)
                else:
                    logger.debug(f"Market data updated: {list(data.keys())}")
                    
        except Exception as e:
            logger.error(f"Error handling market data: {e}")
    
    def _handle_historical_data(self, data: Dict):
        """Handle incoming historical data from NinjaTrader"""
        try:
            # Extract historical bars from different timeframes
            bars_15m = data.get('bars_15m', [])
            bars_5m = data.get('bars_5m', [])
            bars_1m = data.get('bars_1m', [])
            
            # Convert to our format
            for bars in [bars_15m, bars_5m, bars_1m]:
                for bar in bars:
                    if isinstance(bar, dict) and 'close' in bar:
                        self.historical_data_cache.append({
                            'timestamp': bar.get('timestamp', datetime.now().timestamp()),
                            'open': bar.get('open', 0.0),
                            'high': bar.get('high', 0.0),
                            'low': bar.get('low', 0.0),
                            'close': bar.get('close', 0.0),
                            'volume': bar.get('volume', 0.0)
                        })
            
            logger.info(f"Historical data processed: {len(self.historical_data_cache)} total bars")
            
        except Exception as e:
            logger.error(f"Error handling historical data: {e}")
    
    def _handle_trade_completion(self, data: Dict):
        """Handle trade completion notifications from NinjaTrader"""
        try:
            logger.info(f"Trade completion received: {data}")
            # This could trigger learning updates or position management
        except Exception as e:
            logger.error(f"Error handling trade completion: {e}")
    
    async def shutdown(self):
        """Graceful shutdown"""
        try:
            logger.info("Shutting down trading system...")
            self.running = False
            
            # Stop TCP server
            if hasattr(self, 'tcp_server'):
                self.tcp_server.stop()
            
            # Close any open positions if configured
            if self.config.get('close_positions_on_shutdown', False):
                positions = self.trading_service.manage_positions()
                # Implementation for closing positions would go here
            
            # Get final statistics
            intelligence_stats = self.intelligence_engine.get_comprehensive_stats()
            risk_stats = self.risk_manager.get_risk_metrics()
            
            logger.info("=== FINAL STATISTICS ===")
            logger.info(f"Total trades: {self.trade_count}")
            logger.info(f"AI subsystems: {intelligence_stats}")
            logger.info(f"Risk metrics: {risk_stats}")
            logger.info("========================")
            
            logger.info("Trading system shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

def main():
    """Main entry point"""
    try:
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Load configuration
        config = Config()
        
        # Create and start trading system
        orchestrator = TradingSystemOrchestrator(config)
        
        # Run the trading system
        asyncio.run(orchestrator.start())
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Critical error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()