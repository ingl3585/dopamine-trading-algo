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
        
        self.config = Config()
        self.portfolio = Portfolio()
        self.data_processor = DataProcessor()
        self.intelligence = IntelligenceEngine()
        self.agent = TradingAgent(self.intelligence, self.portfolio)
        self.risk_manager = RiskManager(self.portfolio, self.config)
        
        self.tcp_server = TCPServer()
        self.tcp_server.on_market_data = self._process_market_data
        self.tcp_server.on_trade_completion = self._process_trade_completion
        
        self.running = False
        self.last_save = time.time()

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
            
            if time.time() - self.last_save > 300:  # Save every 5 minutes
                self._save_state()
                self.last_save = time.time()

    def _process_market_data(self, raw_data):
        try:
            market_data = self.data_processor.process(raw_data)
            
            if not market_data:
                return
                
            features = self.intelligence.extract_features(market_data)
            
            decision = self.agent.decide(features, market_data)
            if decision.action == 'hold':
                return
                
            order = self.risk_manager.validate_order(decision, market_data)
            if order:
                self.tcp_server.send_signal(order)
                
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
                
        except Exception as e:
            logger.error(f"Error processing market data: {e}")

    def _process_trade_completion(self, completion_data):
        try:
            trade = self.portfolio.complete_trade(completion_data)
            if trade:
                # Enhanced learning with intelligence data
                self.agent.learn_from_trade(trade)
                self.intelligence.learn_from_outcome(trade)
                
        except Exception as e:
            logger.error(f"Error processing trade completion: {e}")

    def _save_state(self):
        try:
            self.agent.save_model('models/agent.pt')
            self.intelligence.save_patterns('data/patterns.json')
            self.portfolio.save_state('data/portfolio.json')
            
        except Exception as e:
            logger.error(f"Error saving state: {e}")

    def shutdown(self):
        logger.info("Shutting down trading system")
        self.running = False
        
        self._save_state()
        self.tcp_server.stop()
        
        logger.info(f"Session summary: {self.portfolio.get_summary()}")