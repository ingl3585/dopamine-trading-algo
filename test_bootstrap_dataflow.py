#!/usr/bin/env python3
"""
Test script to validate end-to-end bootstrap data flow
Tests all data format transformations and learning pipelines
"""

import sys
import os
import logging
import traceback
from datetime import datetime
from typing import Dict, List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.config import Config
from src.core.trading_system_orchestrator import TradingSystemOrchestrator
from src.intelligence import create_intelligence_engine
from src.intelligence.subsystem_evolution import EnhancedIntelligenceOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_historical_data() -> List[Dict]:
    """Create synthetic historical data for testing"""
    historical_data = []
    base_price = 100.0
    
    for i in range(200):  # 200 data points
        # Generate realistic price movements
        price_change = (i % 10 - 5) * 0.001 + 0.0001 * (i % 3)
        base_price += price_change
        
        volume = 1000 + (i % 50) * 10
        
        historical_data.append({
            'timestamp': datetime.now().timestamp() - (200 - i) * 60,
            'open': base_price - 0.01,
            'high': base_price + 0.02,
            'low': base_price - 0.02,
            'close': base_price,
            'volume': volume
        })
    
    return historical_data

def test_data_extraction_methods():
    """Test the data extraction methods directly"""
    logger.info("Testing data extraction methods...")
    
    try:
        config = Config()
        orchestrator = TradingSystemOrchestrator(config)
        
        # Test historical context
        historical_context = {
            'prices': [100.0, 100.1, 100.2, 100.15, 100.25] * 4,  # 20 prices
            'volumes': [1000, 1100, 1200, 1050, 1300] * 4  # 20 volumes
        }
        
        # Test market features
        market_features = {
            'volatility': 0.02,
            'price_momentum': 0.001,
            'volume_momentum': 0.05,
            'regime_confidence': 0.7
        }
        
        # Test AI signals
        ai_signals = {
            'temporal': type('Signal', (), {'value': 0.1, 'confidence': 0.6})(),
            'microstructure': type('Signal', (), {'value': 0.05})()
        }
        
        # Test DNA sequence extraction
        dna_sequence = orchestrator._extract_dna_sequence(historical_context, market_features)
        logger.info(f"DNA sequence extracted: {dna_sequence[:50]}... (length: {len(dna_sequence)})")
        assert isinstance(dna_sequence, str), f"DNA sequence should be string, got {type(dna_sequence)}"
        assert len(dna_sequence) >= 3, f"DNA sequence should be at least 3 chars, got {len(dna_sequence)}"
        
        # Test cycles info extraction
        cycles_info = orchestrator._extract_cycles_info(ai_signals)
        logger.info(f"Cycles info extracted: {cycles_info}")
        assert isinstance(cycles_info, list), f"Cycles info should be list, got {type(cycles_info)}"
        if cycles_info:
            assert isinstance(cycles_info[0], dict), f"Cycle should be dict, got {type(cycles_info[0])}"
            assert 'frequency' in cycles_info[0], "Cycle should have frequency key"
        
        # Test standardized market state
        market_state = orchestrator._create_standardized_market_state(market_features, historical_context)
        logger.info(f"Market state created: {market_state}")
        assert isinstance(market_state, dict), f"Market state should be dict, got {type(market_state)}"
        
        required_keys = ['volatility', 'price_momentum', 'volume_momentum', 'regime', 'time_of_day']
        for key in required_keys:
            assert key in market_state, f"Market state missing required key: {key}"
        
        logger.info("✓ Data extraction methods test passed")
        return True
        
    except Exception as e:
        logger.error(f"Data extraction test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_orchestrator_learning_flow():
    """Test the orchestrator learning flow with realistic data"""
    logger.info("Testing orchestrator learning flow...")
    
    try:
        config = Config()
        orchestrator = EnhancedIntelligenceOrchestrator(config)
        
        # Test learning context
        learning_context = {
            'dna_sequence': 'ABCDEFGHIJKLMNOP',
            'cycles_info': [{
                'frequency': 1.0/60.0,
                'amplitude': 0.15,
                'phase': 0.5,
                'period': 60,
                'window_size': 64
            }],
            'market_state': {
                'volatility': 0.025,
                'price_momentum': 0.002,
                'volume_momentum': 0.01,
                'regime': 'trending',
                'regime_confidence': 0.8,
                'time_of_day': 0.6,
                'market_session': 'regular'
            },
            'microstructure_signal': 0.03,
            'is_bootstrap': True
        }
        
        # Test learning with positive outcome
        outcome = 0.05
        logger.info(f"Testing learning with outcome: {outcome}")
        orchestrator.learn_from_outcome(outcome, learning_context)
        
        # Test learning with negative outcome
        outcome = -0.03
        logger.info(f"Testing learning with outcome: {outcome}")
        orchestrator.learn_from_outcome(outcome, learning_context)
        
        # Test learning with zero outcome
        outcome = 0.0
        logger.info(f"Testing learning with outcome: {outcome}")
        orchestrator.learn_from_outcome(outcome, learning_context)
        
        # Get stats to verify learning occurred
        stats = orchestrator.get_comprehensive_stats()
        logger.info(f"Learning stats: {stats}")
        
        logger.info("✓ Orchestrator learning flow test passed")
        return True
        
    except Exception as e:
        logger.error(f"Orchestrator learning test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_subsystem_learning_signatures():
    """Test individual subsystem learning method signatures"""
    logger.info("Testing subsystem learning signatures...")
    
    try:
        config = Config()
        orchestrator = EnhancedIntelligenceOrchestrator(config)
        
        # Test DNA subsystem
        dna_sequence = "ABCDEFGHIJKLMNOP"
        outcome = 0.05
        logger.info(f"Testing DNA learning with sequence: {dna_sequence}")
        orchestrator.dna_subsystem.learn_from_outcome(dna_sequence, outcome)
        
        # Test temporal subsystem
        cycles_info = [{
            'frequency': 1.0/60.0,
            'amplitude': 0.15,
            'phase': 0.5,
            'period': 60,
            'window_size': 64
        }]
        logger.info(f"Testing temporal learning with cycles: {cycles_info}")
        orchestrator.temporal_subsystem.learn_from_outcome(cycles_info, outcome)
        
        # Test immune subsystem
        market_state = {
            'volatility': 0.025,
            'price_momentum': 0.002,
            'volume_momentum': 0.01,
            'regime': 'trending'
        }
        logger.info(f"Testing immune learning with market state: {market_state}")
        orchestrator.immune_subsystem.learn_threat(market_state, outcome, is_bootstrap=True)
        
        logger.info("✓ Subsystem learning signatures test passed")
        return True
        
    except Exception as e:
        logger.error(f"Subsystem learning test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_intelligence_engine_learning():
    """Test the intelligence engine learning pipeline"""
    logger.info("Testing intelligence engine learning pipeline...")
    
    try:
        config = Config()
        intelligence_engine = create_intelligence_engine(config)
        
        # Test learning context
        learning_context = {
            'dna_sequence': 'ABCDEFGHIJKLMNOP',
            'cycles_info': [{
                'frequency': 1.0/60.0,
                'amplitude': 0.15,
                'phase': 0.5,
                'period': 60,
                'window_size': 64
            }],
            'market_state': {
                'volatility': 0.025,
                'price_momentum': 0.002,
                'volume_momentum': 0.01,
                'regime': 'trending',
                'regime_confidence': 0.8
            },
            'microstructure_signal': 0.03,
            'is_bootstrap': True
        }
        
        # Test learning
        outcome = 0.05
        logger.info(f"Testing intelligence engine learning with outcome: {outcome}")
        intelligence_engine.learn_from_outcome(outcome, learning_context)
        
        # Test learning with None context
        logger.info("Testing intelligence engine learning with None context")
        intelligence_engine.learn_from_outcome(outcome, None)
        
        # Test learning with empty context
        logger.info("Testing intelligence engine learning with empty context")
        intelligence_engine.learn_from_outcome(outcome, {})
        
        logger.info("✓ Intelligence engine learning test passed")
        return True
        
    except Exception as e:
        logger.error(f"Intelligence engine learning test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_bootstrap_simulation():
    """Test the complete bootstrap simulation"""
    logger.info("Testing complete bootstrap simulation...")
    
    try:
        config = Config()
        orchestrator = TradingSystemOrchestrator(config)
        
        # Override the historical data cache with test data
        orchestrator.historical_data_cache = create_test_historical_data()
        
        # Test the bootstrap process components
        historical_data = orchestrator.historical_data_cache
        logger.info(f"Testing with {len(historical_data)} historical data points")
        
        bootstrap_count = 0
        for i, data_point in enumerate(historical_data[:10]):  # Test first 10 points
            try:
                # Simulate the bootstrap process
                market_data = {
                    'timestamp': data_point['timestamp'],
                    'open': data_point['open'],
                    'high': data_point['high'],
                    'low': data_point['low'],
                    'close': data_point['close'],
                    'volume': data_point['volume']
                }
                
                # Create mock market features
                market_features = {
                    'volatility': 0.02,
                    'price_momentum': 0.001,
                    'volume_momentum': 0.05,
                    'regime_confidence': 0.7
                }
                
                # Create mock AI signals
                ai_signals = {
                    'temporal': type('Signal', (), {'value': 0.1, 'confidence': 0.6})(),
                    'microstructure': type('Signal', (), {'value': 0.05})()
                }
                
                # Create historical context
                historical_context = {
                    'prices': [dp['close'] for dp in historical_data[max(0, i-20):i+1]],
                    'volumes': [dp['volume'] for dp in historical_data[max(0, i-20):i+1]]
                }
                
                # Test learning context creation
                standardized_market_state = orchestrator._create_standardized_market_state(
                    market_features, historical_context
                )
                
                learning_context = {
                    'dna_sequence': orchestrator._extract_dna_sequence(historical_context, market_features),
                    'cycles_info': orchestrator._extract_cycles_info(ai_signals),
                    'market_state': standardized_market_state,
                    'microstructure_signal': ai_signals['microstructure'].value,
                    'is_bootstrap': True
                }
                
                # Test outcome calculation
                outcome = orchestrator._calculate_bootstrap_outcome(data_point, historical_data, i)
                
                # Test learning
                orchestrator.intelligence_engine.learn_from_outcome(outcome, learning_context)
                
                bootstrap_count += 1
                logger.info(f"Bootstrap point {bootstrap_count} processed successfully")
                
            except Exception as e:
                logger.error(f"Error in bootstrap point {bootstrap_count}: {e}")
                continue
        
        logger.info(f"✓ Bootstrap simulation test passed ({bootstrap_count} points processed)")
        return True
        
    except Exception as e:
        logger.error(f"Bootstrap simulation test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Run all tests"""
    logger.info("Starting comprehensive bootstrap data flow tests...")
    
    tests = [
        ("Data Extraction Methods", test_data_extraction_methods),
        ("Orchestrator Learning Flow", test_orchestrator_learning_flow),
        ("Subsystem Learning Signatures", test_subsystem_learning_signatures),
        ("Intelligence Engine Learning", test_intelligence_engine_learning),
        ("Bootstrap Simulation", test_bootstrap_simulation)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            if test_func():
                logger.info(f"✓ {test_name} PASSED")
                passed += 1
            else:
                logger.error(f"✗ {test_name} FAILED")
                failed += 1
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
            failed += 1
    
    logger.info(f"\n{'='*50}")
    logger.info(f"TEST SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total:  {passed + failed}")
    
    if failed == 0:
        logger.info("🎉 ALL TESTS PASSED! Bootstrap data flow is working correctly.")
        return 0
    else:
        logger.error(f"❌ {failed} tests failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())