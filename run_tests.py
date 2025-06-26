#!/usr/bin/env python3
"""
Test Runner - Execute integration tests for the trading system
"""

import sys
import unittest
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def run_tests():
    """Run all integration tests"""
    print("🧪 Running Actor-Critic ML Trading System Integration Tests...")
    print("=" * 60)
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent / "tests"
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        buffer=True
    )
    
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ All tests passed!")
        print(f"Ran {result.testsRun} tests successfully")
        return 0
    else:
        print("❌ Some tests failed!")
        print(f"Ran {result.testsRun} tests, {len(result.failures)} failures, {len(result.errors)} errors")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback}")
        
        return 1

if __name__ == "__main__":
    # Suppress logging during tests unless explicitly needed
    logging.basicConfig(level=logging.CRITICAL)
    
    sys.exit(run_tests())