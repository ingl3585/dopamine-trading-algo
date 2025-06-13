# trading_system.py

import os
import threading
import logging
import time
from typing import Dict
from datetime import datetime
from config import ResearchConfig
from tcp_bridge import TCPBridge
from advanced_market_intelligence import AdvancedMarketIntelligence
from trade_manager_ai import TradeManagerAI

log = logging.getLogger(__name__)

class TradingSystem:
    """Pure Black Box Market Intelligence Engine"""

    def __init__(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        self.config = ResearchConfig()

        # Core intelligence
        self.intelligence_engine = AdvancedMarketIntelligence()
        self.intelligence_engine.start_continuous_learning()

        # TCP bridge FIRST (so we can pass it to TradeManagerAI)
        self.tcp_bridge = TCPBridge(self.config)
        self.tcp_bridge.on_market_data = self.process_raw_market_data
        self.tcp_bridge.on_trade_completion = self.on_trade_completed

        # Trade manager with RL agent
        self.trade_manager = TradeManagerAI(
            self.intelligence_engine,
            self.tcp_bridge
        )

        # Book-keeping
        self.in_trade = False
        self.entry_time = None
        self.entry_prices = []
        self.entry_volumes = []

        self.signal_count = 0
        self.bootstrap_complete = False
        self.stats = {
            'total_signals': 0,
            'intelligence_signals': 0,
            'patterns_discovered': 0
        }

        log.info("Pure Black Box Intelligence Engine initialized")
    
    def start(self):
        try:
            self.tcp_bridge.start()
            while True:
                time.sleep(1)

                if self.in_trade and self.entry_time:
                    should_exit, reason = self.trade_manager.should_exit_now(
                        self.entry_prices, self.entry_volumes, self.entry_time
                    )
                    if should_exit:
                        print(f"[INTELLIGENCE EXIT] Triggered: {reason}")
                        self.tcp_bridge.send_signal(0, 0.5, f"auto_exit_{reason}")
                        self.in_trade = False
                        self.entry_time = None
                        self.entry_prices = []
                        self.entry_volumes = []

        except KeyboardInterrupt:
            log.info("Shutdown requested")
        finally:
            self.stop()
    
    def bootstrap_intelligence_from_history(self, price_15m, volume_15m, price_5m, volume_5m, price_1m, volume_1m):
        """Feed historical data directly to intelligence for pattern discovery"""
        log.info("BOOTSTRAPPING pure intelligence from historical patterns...")
        
        # Feed different timeframes to intelligence for pattern learning
        patterns_learned = 0
        
        # Process 1-minute data in chunks for micro-pattern discovery
        if len(price_1m) >= 200:
            for i in range(100, len(price_1m), 20):  # Every 20 bars
                chunk_prices = price_1m[i-100:i]
                chunk_volumes = volume_1m[i-100:i] if volume_1m else [1000] * 100
                
                # Let intelligence discover patterns
                self.intelligence_engine.process_market_data(chunk_prices, chunk_volumes)
                patterns_learned += 1
                
                if patterns_learned >= 50:  # Enough for bootstrap
                    break
        
        # Process 5-minute data for broader patterns
        if len(price_5m) >= 100:
            for i in range(50, len(price_5m), 10):  # Every 10 bars
                chunk_prices = price_5m[i-50:i]
                chunk_volumes = volume_5m[i-50:i] if volume_5m else [500] * 50
                
                self.intelligence_engine.process_market_data(chunk_prices, chunk_volumes)
                patterns_learned += 1
                
                if patterns_learned >= 100:  # Enough patterns
                    break
        
        self.bootstrap_complete = True
        log.info(f"PURE INTELLIGENCE BOOTSTRAPPED - {patterns_learned} pattern samples processed")
        log.info("Ready for live pattern discovery and trading")
    
    def process_raw_market_data(self, data: Dict):
        """Process raw market data - NO FEATURE EXTRACTION"""
        try:
            # Extract raw price/volume data
            price_15m = data.get("price_15m", [])
            volume_15m = data.get("volume_15m", [])
            price_5m = data.get("price_5m", [])
            volume_5m = data.get("volume_5m", [])
            price_1m = data.get("price_1m", [])
            volume_1m = data.get("volume_1m", [])
            
            print(f"=== RAW DATA RECEIVED ===")
            print(f"15m: {len(price_15m)} bars, 5m: {len(price_5m)} bars, 1m: {len(price_1m)} bars")
            
            # Bootstrap intelligence on first substantial data
            if not self.bootstrap_complete and len(price_5m) >= 100:
                self.bootstrap_intelligence_from_history(
                    price_15m, volume_15m, price_5m, volume_5m, price_1m, volume_1m
                )
            
            # Choose best data for intelligence processing
            if len(price_1m) >= 50:
                # Use 1-minute data for precision
                intel_prices = price_1m[-200:] if len(price_1m) >= 200 else price_1m
                intel_volumes = volume_1m[-200:] if len(volume_1m) >= 200 else volume_1m
                timeframe = "1m"
            elif len(price_5m) >= 20:
                # Fallback to 5-minute data
                intel_prices = price_5m[-100:] if len(price_5m) >= 100 else price_5m
                intel_volumes = volume_5m[-100:] if len(volume_5m) >= 100 else volume_5m
                timeframe = "5m"
            else:
                print("Insufficient data for intelligence processing")
                return
            
            print(f"Processing with PURE INTELLIGENCE using {timeframe} data...")

            if price_1m:
                self.trade_manager.on_new_bar(data)
            
            # PURE INTELLIGENCE DECISION - No traditional models
            intelligence_result = self.intelligence_engine.process_market_data(
                intel_prices, intel_volumes, datetime.now()
            )
            
            # Direct signal from intelligence
            final_signal = self.convert_intelligence_to_signal(intelligence_result)
            
            # Send signal if intelligence is confident
            if final_signal['send_signal']:
                self.tcp_bridge.send_signal(
                    final_signal['action'], 
                    final_signal['confidence'], 
                    final_signal['quality']
                )
                log.info(f"PURE INTELLIGENCE: {final_signal['reasoning']}")
                self.stats['intelligence_signals'] += 1

                # New: track entry info
                if final_signal['action'] in [1, 2]:
                    self.in_trade = True
                    self.entry_time = datetime.now()
                    self.entry_prices = intel_prices.copy()
                    self.entry_volumes = intel_volumes.copy()
            else:
                print(f"INTELLIGENCE HOLDING: {final_signal['reasoning']}")
            
            self.signal_count += 1
            self.stats['total_signals'] += 1
            
            print("=== PURE INTELLIGENCE PROCESSING COMPLETE ===\n")
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            log.error(f"Raw data processing error: {e}")
    
    def convert_intelligence_to_signal(self, intelligence_result):
        """Convert pure intelligence output to trading signal"""
        
        intel_signal = intelligence_result.get('signal_strength', 0)
        intel_confidence = intelligence_result.get('confidence', 0)
        is_dangerous = intelligence_result.get('is_dangerous_pattern', False)
        is_beneficial = intelligence_result.get('is_beneficial_pattern', False)
        
        # 1. IMMUNE SYSTEM OVERRIDE - Intelligence recognizes danger
        if is_dangerous and intel_confidence > 0.4:
            return {
                'action': 0,  # HOLD
                'confidence': 0.0,
                'quality': 'dangerous_pattern',
                'send_signal': False,
                'reasoning': f"DANGEROUS PATTERN DETECTED - Intelligence override (conf: {intel_confidence:.3f})"
            }
        
        # 2. HIGH CONFIDENCE INTELLIGENCE SIGNAL
        if intel_confidence > 0.7 and abs(intel_signal) > 0.2:
            action = self.signal_to_action(intel_signal)
            return {
                'action': action,
                'confidence': intel_confidence,
                'quality': 'high_confidence_intelligence',
                'send_signal': True,
                'reasoning': f"HIGH CONFIDENCE INTELLIGENCE - Signal: {intel_signal:.3f}, Conf: {intel_confidence:.3f}"
            }
        
        # 3. BENEFICIAL PATTERN SIGNAL
        if is_beneficial and intel_confidence > 0.5 and abs(intel_signal) > 0.15:
            action = self.signal_to_action(intel_signal)
            return {
                'action': action,
                'confidence': intel_confidence * 1.1,  # Slight boost for beneficial
                'quality': 'beneficial_pattern',
                'send_signal': True,
                'reasoning': f"BENEFICIAL PATTERN - Signal: {intel_signal:.3f}, Conf: {intel_confidence:.3f}"
            }
        
        # 4. MODERATE CONFIDENCE INTELLIGENCE
        if intel_confidence > 0.6 and abs(intel_signal) > 0.25:
            action = self.signal_to_action(intel_signal)
            return {
                'action': action,
                'confidence': intel_confidence * 0.9,  # Slight discount
                'quality': 'moderate_intelligence',
                'send_signal': True,
                'reasoning': f"MODERATE INTELLIGENCE - Signal: {intel_signal:.3f}, Conf: {intel_confidence:.3f}"
            }
        
        # 5. STRONG SIGNAL, LOWER CONFIDENCE
        if intel_confidence > 0.4 and abs(intel_signal) > 0.35:
            action = self.signal_to_action(intel_signal)
            return {
                'action': action,
                'confidence': intel_confidence * 0.8,
                'quality': 'strong_signal_intelligence',
                'send_signal': True,
                'reasoning': f"STRONG SIGNAL - Signal: {intel_signal:.3f}, Conf: {intel_confidence:.3f}"
            }
        
        # 6. INTELLIGENCE SAYS HOLD
        return {
            'action': 0,
            'confidence': intel_confidence,
            'quality': 'intelligence_neutral',
            'send_signal': False,
            'reasoning': f"INTELLIGENCE NEUTRAL - Signal: {intel_signal:.3f}, Conf: {intel_confidence:.3f}"
        }
    
    def signal_to_action(self, signal_strength):
        """Convert intelligence signal strength to action"""
        if signal_strength > 0.1:     # Lower threshold for pure intelligence
            return 1  # BUY
        elif signal_strength < -0.1:  # Lower threshold for pure intelligence
            return 2  # SELL
        else:
            return 0  # HOLD
    
    def on_trade_completed(self, completion_data):
        """Feed trade outcomes back to intelligence for learning"""
        try:
            exit_price = completion_data.get('exit_price', 0)
            exit_reason = completion_data.get('exit_reason', 'unknown')
            duration_minutes = completion_data.get('duration_minutes', 0)
            
            # Calculate simple outcome for learning
            if exit_reason in ['intelligence_exit', 'signal_exit']:
                outcome = 0.005  # Small positive for intelligence-driven exits
            elif exit_reason == 'session_close':
                outcome = 0.001  # Neutral for session close
            else:
                outcome = -0.005  # Small negative for other exits
            
            # Feed back to intelligence for pattern learning
            self.intelligence_engine.record_trade_outcome(
                datetime.now(), 
                outcome,
                exit_price, 
                exit_price
            )
            
            log.info(f"Trade outcome fed to intelligence: {exit_reason} -> {outcome}")

            self.in_trade = False
            self.entry_time = None
            self.entry_prices = []
            self.entry_volumes = []
            
        except Exception as e:
            log.error(f"Trade completion error: {e}")
    
    def stop(self):
        """Stop the Pure Intelligence Engine"""
        log.info("Stopping Pure Black Box Intelligence Engine...")
        
        # Stop TCP bridge
        try:
            self.tcp_bridge.stop()
        except Exception as e:
            log.warning(f"TCP bridge stop error: {e}")
        
        # Export intelligence knowledge base
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            knowledge_file = f"pure_intelligence_patterns_{timestamp}.json"
            self.intelligence_engine.export_knowledge_base(knowledge_file)
            log.info(f"Pure intelligence patterns exported to {knowledge_file}")
        except Exception as e:
            log.warning(f"Knowledge export error: {e}")
        
        # Print final statistics
        total = self.stats['total_signals']
        if total > 0:
            log.info("=== PURE BLACK BOX INTELLIGENCE FINAL REPORT ===")
            log.info(f"Total Data Samples Processed: {total}")
            log.info(f"Intelligence Signals Generated: {self.stats['intelligence_signals']} ({self.stats['intelligence_signals']/total*100:.1f}%)")
            
            try:
                intel_status = self.intelligence_engine.get_system_status()
                log.info(f"DNA Patterns Discovered: {intel_status['total_dna_patterns']}")
                log.info(f"Micro Patterns Found: {intel_status['total_micro_patterns']}")
                log.info(f"Temporal Patterns Learned: {intel_status['total_temporal_patterns']}")
                log.info(f"Intelligence Win Rate: {intel_status['win_rate']:.2%}")
                log.info(f"System Weights Learned: {intel_status['system_weights']}")
                log.info("Pure Black Box Intelligence - All Patterns Preserved Forever")
            except Exception as e:
                log.warning(f"Status report error: {e}")
        
        log.info("Pure Black Box Intelligence Engine stopped")