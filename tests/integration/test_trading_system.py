"""
Integration Tests - Trading System End-to-End Testing
"""

import asyncio
import unittest
import logging
from unittest.mock import Mock, patch
from datetime import datetime

# Disable logging during tests
logging.disable(logging.CRITICAL)

from src.core.main import TradingSystemOrchestrator
from src.core.config import Config
from src.shared.types import TradeDecision, AccountInfo

class TestTradingSystemIntegration(unittest.TestCase):
    """
    Integration tests for the complete trading system
    """
    
    def setUp(self):
        """Set up test environment"""
        # Create test configuration
        self.config = Config()
        self.config.update_setting('log_level', 'CRITICAL')
        self.config.update_setting('tcp_data_port', 5558)  # Different port for testing
        self.config.update_setting('tcp_signal_port', 5559)
        self.config.update_setting('trading_interval_seconds', 1)  # Fast for testing
        self.config.update_setting('historical_data_timeout', 5)  # Short timeout
        
        # Mock TCP server to avoid actual network connections
        self.tcp_server_mock = Mock()
        self.tcp_server_mock.is_ready_for_live_trading.return_value = True
        self.tcp_server_mock.start.return_value = None
        self.tcp_server_mock.stop.return_value = None
        
    def test_orchestrator_initialization(self):
        """Test that the trading system orchestrator initializes correctly"""
        with patch('src.core.main.TCPServer', return_value=self.tcp_server_mock):
            orchestrator = TradingSystemOrchestrator(self.config)
            
            # Verify all domains are initialized
            self.assertIsNotNone(orchestrator.intelligence_engine)
            self.assertIsNotNone(orchestrator.trading_service)
            self.assertIsNotNone(orchestrator.market_processor)
            self.assertIsNotNone(orchestrator.risk_manager)
            self.assertIsNotNone(orchestrator.tcp_server)
            
            # Verify initial state
            self.assertFalse(orchestrator.bootstrap_complete)
            self.assertEqual(orchestrator.trade_count, 0)
            self.assertIsNone(orchestrator.last_decision_time)
    
    def test_ai_domain_integration(self):
        """Test AI domain integration and signal generation"""
        with patch('src.core.main.TCPServer', return_value=self.tcp_server_mock):
            orchestrator = TradingSystemOrchestrator(self.config)
            
            # Test AI analysis with sample data
            sample_historical_context = {
                'prices': [15000 + i for i in range(100)],
                'volumes': [1000 + i*10 for i in range(100)],
                'timestamps': [datetime.now().timestamp() + i for i in range(100)]
            }
            
            sample_market_features = {
                'price_momentum': 0.05,
                'volume_momentum': 0.02,
                'volatility': 0.015,
                'price_position': 0.7,
                'time_of_day': 0.5,
                'pattern_score': 0.6,
                'confidence': 0.8
            }
            
            # This should not raise exceptions
            ai_signals = orchestrator.intelligence_engine.analyze_market(
                sample_historical_context, sample_market_features
            )
            
            # Verify AI signals structure
            required_signals = ['overall', 'dna', 'temporal', 'immune', 'microstructure']
            for signal_name in required_signals:
                self.assertIn(signal_name, ai_signals)
                self.assertTrue(hasattr(ai_signals[signal_name], 'value'))
                self.assertTrue(hasattr(ai_signals[signal_name], 'confidence'))
    
    def test_risk_management_integration(self):
        """Test risk management domain integration"""
        with patch('src.core.main.TCPServer', return_value=self.tcp_server_mock):
            orchestrator = TradingSystemOrchestrator(self.config)
            
            # Create sample trade decision
            decision = TradeDecision(
                action="buy",
                size=1.0,
                confidence=0.8,
                reasoning={'overall_signal': 0.5, 'volatility': 0.02}
            )
            
            # Create sample account info
            account = AccountInfo(
                buying_power=25000.0,
                position_size=0.0,
                daily_pnl=0.0
            )
            
            # Test risk assessment
            risk_level = orchestrator.risk_manager.assess_risk(decision, account)
            self.assertIsInstance(risk_level, float)
            self.assertGreaterEqual(risk_level, 0.0)
            self.assertLessEqual(risk_level, 1.0)
            
            # Test position sizing
            position_size = orchestrator.risk_manager.size_position(0.5, account, risk_level)
            self.assertIsInstance(position_size, float)
            self.assertGreaterEqual(position_size, 0.0)
    
    def test_market_data_processing(self):
        """Test market data processing domain"""
        with patch('src.core.main.TCPServer', return_value=self.tcp_server_mock):
            orchestrator = TradingSystemOrchestrator(self.config)
            
            # Test market data processing
            sample_raw_data = {
                'timestamp': datetime.now().timestamp(),
                'open': 15000.0,
                'high': 15010.0,
                'low': 14995.0,
                'close': 15005.0,
                'volume': 1500.0
            }
            
            market_data = orchestrator.market_processor.process_data(sample_raw_data)
            self.assertIsNotNone(market_data)
            self.assertEqual(market_data.close, 15005.0)
            
            # Test feature extraction
            features = orchestrator.market_processor.extract_features(market_data)
            self.assertIsInstance(features, dict)
            self.assertIn('price_momentum', features)
            self.assertIn('volatility', features)
    
    def test_trade_decision_creation(self):
        """Test trade decision creation logic"""
        with patch('src.core.main.TCPServer', return_value=self.tcp_server_mock):
            orchestrator = TradingSystemOrchestrator(self.config)
            
            # Mock AI signals
            from src.shared.types import Signal
            ai_signals = {
                'overall': Signal(value=0.6, confidence=0.8),
                'dna': Signal(value=0.5, confidence=0.7),
                'temporal': Signal(value=0.7, confidence=0.8),
                'immune': Signal(value=0.4, confidence=0.6),
                'microstructure': Signal(value=0.6, confidence=0.9),
                'consensus_strength': 0.7
            }
            
            market_features = {
                'volatility': 0.02,
                'price_momentum': 0.03
            }
            
            # Test trade decision creation
            decision = orchestrator._create_trade_decision(ai_signals, market_features)
            self.assertIsInstance(decision, TradeDecision)
            self.assertIn(decision.action, ['buy', 'sell', 'hold'])
            self.assertIsInstance(decision.confidence, float)
            self.assertIsInstance(decision.reasoning, dict)
    
    def test_should_execute_trade_logic(self):
        """Test trade execution decision logic"""
        with patch('src.core.main.TCPServer', return_value=self.tcp_server_mock):
            orchestrator = TradingSystemOrchestrator(self.config)
            
            # Test cases for trade execution
            from src.shared.types import Signal
            
            # Case 1: Strong signal should execute
            strong_decision = TradeDecision(
                action="buy",
                size=1.0,
                confidence=0.8,
                reasoning={}
            )
            
            strong_signals = {
                'consensus_strength': 0.8
            }
            
            should_execute = orchestrator._should_execute_trade(
                strong_decision, 0.3, strong_signals
            )
            self.assertTrue(should_execute)
            
            # Case 2: High risk should not execute
            should_not_execute = orchestrator._should_execute_trade(
                strong_decision, 0.9, strong_signals
            )
            self.assertFalse(should_not_execute)
            
            # Case 3: Hold action should not execute
            hold_decision = TradeDecision(
                action="hold",
                size=1.0,
                confidence=0.8,
                reasoning={}
            )
            
            should_not_execute_hold = orchestrator._should_execute_trade(
                hold_decision, 0.3, strong_signals
            )
            self.assertFalse(should_not_execute_hold)
    
    def test_system_graceful_shutdown(self):
        """Test system graceful shutdown"""
        with patch('src.core.main.TCPServer', return_value=self.tcp_server_mock):
            orchestrator = TradingSystemOrchestrator(self.config)
            
            # Test shutdown doesn't raise exceptions
            async def test_shutdown():
                await orchestrator.shutdown()
            
            # Run shutdown test
            asyncio.run(test_shutdown())
            
            # Verify TCP server was stopped
            self.tcp_server_mock.stop.assert_called_once()

class TestConfigurationSystem(unittest.TestCase):
    """Test configuration management"""
    
    def test_config_initialization(self):
        """Test configuration loads correctly"""
        config = Config()
        
        # Test required settings exist
        required_settings = [
            'tcp_data_port', 'tcp_signal_port', 'max_position_size',
            'max_daily_loss', 'leverage', 'contract_value'
        ]
        
        for setting in required_settings:
            self.assertIsNotNone(config.get(setting))
    
    def test_config_validation(self):
        """Test configuration validation"""
        # This should not raise any validation errors
        config = Config()
        
        # Test that ports are in valid range
        self.assertGreaterEqual(config.get('tcp_data_port'), 1024)
        self.assertLessEqual(config.get('tcp_data_port'), 65535)
        
        # Test that position size is reasonable
        self.assertGreater(config.get('max_position_size'), 0)
        self.assertLessEqual(config.get('max_position_size'), 1)
    
    def test_environment_specific_config(self):
        """Test environment-specific configuration loading"""
        import os
        
        # Test development environment
        os.environ['TRADING_ENV'] = 'development'
        config = Config()
        self.assertEqual(config.get('environment'), 'development')
        
        # Clean up
        if 'TRADING_ENV' in os.environ:
            del os.environ['TRADING_ENV']

if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)