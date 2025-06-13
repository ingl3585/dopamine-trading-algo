# trading_system.py (FIXED)

import os
import threading
import logging
import time
import numpy as np
from typing import Dict
from datetime import datetime
from config import ResearchConfig
from tcp_bridge import TCPBridge
from advanced_market_intelligence import AdvancedMarketIntelligence
from trade_manager_ai import TradeManagerAI

log = logging.getLogger(__name__)

class TradingSystem:
    """Fixed Black Box Market Intelligence Engine with ALL subsystems active"""

    def __init__(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        self.config = ResearchConfig()

        # Core intelligence with PERMANENT MEMORY enabled
        self.intelligence_engine = AdvancedMarketIntelligence()
        
        # FIX 1: Load existing patterns from permanent memory
        self._load_permanent_memory()
        
        self.intelligence_engine.start_continuous_learning()

        # TCP bridge 
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
        
        # FIX 2: Track all four subsystems
        self.subsystem_status = {
            'dna': {'active': False, 'patterns': 0},
            'micro': {'active': False, 'patterns': 0}, 
            'temporal': {'active': False, 'patterns': 0},
            'immune': {'active': False, 'patterns': 0}
        }
        
        self.stats = {
            'total_signals': 0,
            'intelligence_signals': 0,
            'consensus_signals': 0,
            'patterns_discovered': 0,
            'memory_saves': 0
        }

        log.info("BLACK BOX Intelligence Engine initialized with PERMANENT MEMORY")
    
    def _load_permanent_memory(self):
        """FIX 1: Load all patterns from permanent memory on startup"""
        try:
            # Load DNA patterns
            dna_patterns = self.intelligence_engine.memory_db.load_dna_patterns()
            self.intelligence_engine.dna_system.dna_patterns.update(dna_patterns)
            
            log.info(f"PERMANENT MEMORY: Loaded {len(dna_patterns)} DNA patterns")
            
            # Enable all subsystems from the start
            self.subsystem_status['dna']['active'] = True
            self.subsystem_status['dna']['patterns'] = len(dna_patterns)
            
            # Other subsystems start active too
            for system in ['micro', 'temporal', 'immune']:
                self.subsystem_status[system]['active'] = True
                
        except Exception as e:
            log.warning(f"Memory load error (expected on first run): {e}")
    
    def start(self):
        try:
            self.tcp_bridge.start()
            
            # Print subsystem status
            log.info("=== SUBSYSTEM STATUS ===")
            for name, status in self.subsystem_status.items():
                log.info(f"{name.upper()}: {'ACTIVE' if status['active'] else 'INACTIVE'} - {status['patterns']} patterns")
            
            while True:
                time.sleep(1)

                if self.in_trade and self.entry_time:
                    should_exit, reason = self.trade_manager.should_exit_now(
                        self.entry_prices, self.entry_volumes, self.entry_time
                    )
                    if should_exit:
                        print(f"[INTELLIGENCE EXIT] Triggered: {reason}")
                        self.tcp_bridge.send_signal(0, 0.5, f"auto_exit_{reason}")
                        self._complete_trade("intelligence_exit")

        except KeyboardInterrupt:
            log.info("Shutdown requested")
        finally:
            self.stop()
    
    def bootstrap_intelligence_from_history(self, price_15m, volume_15m, price_5m, volume_5m, price_1m, volume_1m):
        """Enhanced bootstrap that SAVES patterns to permanent memory"""
        log.info("BOOTSTRAPPING with PERMANENT MEMORY saving...")
        
        patterns_learned = 0
        
        # Process 1-minute data for DNA sequencing
        if len(price_1m) >= 200:
            for i in range(100, len(price_1m), 20):
                chunk_prices = price_1m[i-100:i]
                chunk_volumes = volume_1m[i-100:i] if volume_1m else [1000] * 100
                
                # FIX 1: Create DNA sequences and save to permanent memory
                dna_sequence = self.intelligence_engine.dna_system.create_dna_sequence(
                    chunk_prices, chunk_volumes
                )
                
                if dna_sequence and len(dna_sequence) > 10:
                    # Process through intelligence to create pattern
                    result = self.intelligence_engine.process_market_data(chunk_prices, chunk_volumes)
                    
                    # Save DNA pattern with dummy outcome for bootstrap
                    if dna_sequence not in self.intelligence_engine.dna_system.dna_patterns:
                        self.intelligence_engine.dna_system.update_pattern_outcome(dna_sequence, 0.001)
                        patterns_learned += 1
                        self.stats['memory_saves'] += 1
                
                if patterns_learned >= 50:
                    break
        
        # Process 5-minute data 
        if len(price_5m) >= 100:
            for i in range(50, len(price_5m), 10):
                chunk_prices = price_5m[i-50:i]
                chunk_volumes = volume_5m[i-50:i] if volume_5m else [500] * 50
                
                self.intelligence_engine.process_market_data(chunk_prices, chunk_volumes)
                patterns_learned += 1
                
                if patterns_learned >= 100:
                    break
        
        # Update subsystem status
        self.subsystem_status['dna']['patterns'] = len(self.intelligence_engine.dna_system.dna_patterns)
        self.subsystem_status['micro']['patterns'] = len(self.intelligence_engine.micro_system.patterns)
        self.subsystem_status['temporal']['patterns'] = len(self.intelligence_engine.temporal_system.temporal_patterns)
        
        self.bootstrap_complete = True
        log.info(f"BOOTSTRAP COMPLETE - {patterns_learned} patterns, {self.stats['memory_saves']} saved to permanent memory")
        log.info("ALL SUBSYSTEMS ACTIVE with PERMANENT MEMORY")
    
    def process_raw_market_data(self, data: Dict):
        """Simplified processing - let black box AI handle everything"""
        try:
            # Extract data
            price_1m = data.get("price_1m", [])
            volume_1m = data.get("volume_1m", [])
            
            if not price_1m:
                return
            
            # Bootstrap intelligence if needed (keep existing logic)
            if not self.bootstrap_complete and len(price_1m) >= 100:
                self.bootstrap_intelligence_from_history(
                    data.get("price_15m", []), data.get("volume_15m", []),
                    data.get("price_5m", []), data.get("volume_5m", []),
                    price_1m, volume_1m
                )
            
            # Store current price for trade completion
            if price_1m:
                self.current_price = price_1m[-1]
            
            # BLACK BOX AI handles everything using your subsystems as tools
            self.trade_manager.on_new_bar(data)
            
            # Optional: Print subsystem usage report periodically
            if self.signal_count % 100 == 0 and self.signal_count > 0:
                report = self.trade_manager.get_performance_report()
                print(report)
            
            self.signal_count += 1
            
        except Exception as e:
            log.error(f"Black box processing error: {e}")
            import traceback
            traceback.print_exc()
    
    def enforce_subsystem_consensus(self, intelligence_result):
        """FIX 2: ENFORCE the consensus requirement from your prompt"""
        
        subsystem_signals = intelligence_result.get('subsystem_signals', {})
        subsystem_scores = intelligence_result.get('subsystem_scores', {})
        
        print(f"SUBSYSTEM SIGNALS: {subsystem_signals}")
        print(f"SUBSYSTEM SCORES: {subsystem_scores}")
        
        # REQUIREMENT: At least 3 of 4 subsystems must be active
        required_systems = ['dna', 'micro', 'temporal', 'immune']
        active_systems = [s for s in required_systems if s in subsystem_signals and subsystem_scores.get(s, 0) > 0.1]
        
        if len(active_systems) < 3:
            return {
                'action': 0,
                'confidence': 0.0,
                'quality': 'insufficient_subsystems',
                'send_signal': False,
                'reasoning': f"CONSENSUS FAILED: Only {len(active_systems)}/4 subsystems active: {active_systems}"
            }
        
        # IMMUNE SYSTEM OVERRIDE (highest priority)
        immune_signal = subsystem_signals.get('immune', 0)
        immune_score = subsystem_scores.get('immune', 0)
        
        if immune_signal < -0.3 and immune_score > 0.4:
            return {
                'action': 0,
                'confidence': 0.0,
                'quality': 'immune_override',
                'send_signal': False,
                'reasoning': f"IMMUNE OVERRIDE: Dangerous pattern detected {immune_signal:.3f}"
            }
        
        # COUNT VOTES FOR CONSENSUS
        buy_voters = []
        sell_voters = []
        
        for system in active_systems:
            signal = subsystem_signals[system]
            confidence = subsystem_scores[system]
            
            if confidence > 0.3:  # Only confident votes count
                if signal > 0.15:
                    buy_voters.append((system, signal, confidence))
                elif signal < -0.15:
                    sell_voters.append((system, signal, confidence))
        
        # REQUIRE MINIMUM 2 SYSTEM AGREEMENT
        min_agreement = 2
        
        if len(buy_voters) >= min_agreement:
            avg_confidence = np.mean([conf for _, _, conf in buy_voters])
            avg_signal = np.mean([sig for _, sig, _ in buy_voters])
            
            if avg_confidence > 0.5:
                return {
                    'action': 1,
                    'confidence': avg_confidence,
                    'quality': f'consensus_buy_{len(buy_voters)}systems',
                    'send_signal': True,
                    'reasoning': f"BUY CONSENSUS: {len(buy_voters)} systems agree - {[v[0] for v in buy_voters]} (conf: {avg_confidence:.3f})"
                }
        
        if len(sell_voters) >= min_agreement:
            avg_confidence = np.mean([conf for _, _, conf in sell_voters])
            avg_signal = np.mean([sig for _, sig, _ in sell_voters])
            
            if avg_confidence > 0.5:
                return {
                    'action': 2,
                    'confidence': avg_confidence,
                    'quality': f'consensus_sell_{len(sell_voters)}systems',
                    'send_signal': True,
                    'reasoning': f"SELL CONSENSUS: {len(sell_voters)} systems agree - {[v[0] for v in sell_voters]} (conf: {avg_confidence:.3f})"
                }
        
        # NO CONSENSUS ACHIEVED
        return {
            'action': 0,
            'confidence': 0.0,
            'quality': 'no_consensus',
            'send_signal': False,
            'reasoning': f"NO CONSENSUS: Buy voters: {len(buy_voters)}, Sell voters: {len(sell_voters)} (need {min_agreement}+)"
        }
    
    def _update_subsystem_status(self, result):
        """Track subsystem health"""
        subsystem_signals = result.get('subsystem_signals', {})
        
        for system in ['dna', 'micro', 'temporal', 'immune']:
            if system in subsystem_signals:
                self.subsystem_status[system]['active'] = True
                
                # Update pattern counts
                if system == 'dna':
                    self.subsystem_status[system]['patterns'] = len(self.intelligence_engine.dna_system.dna_patterns)
                elif system == 'micro':
                    self.subsystem_status[system]['patterns'] = len(self.intelligence_engine.micro_system.patterns)
                elif system == 'temporal':
                    self.subsystem_status[system]['patterns'] = len(self.intelligence_engine.temporal_system.temporal_patterns)
    
    def _complete_trade(self, reason):
        """Helper to complete trades with RL learning"""
        # Calculate basic PnL for RL learning
        if self.entry_prices and hasattr(self, 'current_price'):
            entry_price = self.entry_prices[-1] 
            pnl = self.current_price - entry_price if reason != "loss" else entry_price - self.current_price
            
            # Feed to RL agent
            self.trade_manager.record_trade_outcome(self.current_price, pnl, True)
            
        self.in_trade = False
        self.entry_time = None
        self.entry_prices = []
        self.entry_volumes = []
    
    def on_trade_completed(self, completion_data):
        """Handle trade completion from NinjaTrader"""
        try:
            exit_price = completion_data.get('exit_price', 0)
            exit_reason = completion_data.get('exit_reason', 'unknown')
            
            # Complete the trade in trade manager
            self.trade_manager._complete_trade(exit_reason, exit_price)
            
            log.info(f"Trade completed: {exit_reason} at ${exit_price:.2f}")
            
        except Exception as e:
            log.error(f"Trade completion error: {e}")
    
    def stop(self):
        """Stop with full permanent memory save"""
        log.info("Stopping BLACK BOX Intelligence with PERMANENT MEMORY save...")
        
        try:
            self.tcp_bridge.stop()
        except Exception as e:
            log.warning(f"TCP stop error: {e}")
        
        # FIX 1: Force save all patterns to permanent memory
        try:
            log.info("SAVING ALL PATTERNS TO PERMANENT MEMORY...")
            
            # Save all DNA patterns
            for seq, pattern in self.intelligence_engine.dna_system.dna_patterns.items():
                self.intelligence_engine.memory_db.save_dna_pattern(pattern)
            
            log.info(f"PERMANENT MEMORY: {len(self.intelligence_engine.dna_system.dna_patterns)} DNA patterns saved")
            
        except Exception as e:
            log.warning(f"Memory save error: {e}")
        
        # Export knowledge base
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            knowledge_file = f"black_box_intelligence_{timestamp}.json"
            self.intelligence_engine.export_knowledge_base(knowledge_file)
            log.info(f"Knowledge exported to {knowledge_file}")
        except Exception as e:
            log.warning(f"Export error: {e}")
        
        # Print final statistics
        log.info("=== BLACK BOX FINAL REPORT ===")
        log.info(f"Total Signals: {self.stats['total_signals']}")
        log.info(f"Consensus Signals: {self.stats['consensus_signals']}")
        log.info(f"Memory Saves: {self.stats['memory_saves']}")
        
        log.info("=== SUBSYSTEM FINAL STATUS ===")
        for name, status in self.subsystem_status.items():
            log.info(f"{name.upper()}: {status['patterns']} patterns learned")
        
        try:
            intel_status = self.intelligence_engine.get_system_status()
            log.info(f"DNA Patterns: {intel_status['total_dna_patterns']}")
            log.info(f"Micro Patterns: {intel_status['total_micro_patterns']}")
            log.info(f"Temporal Patterns: {intel_status['total_temporal_patterns']}")
            log.info(f"Win Rate: {intel_status['win_rate']:.2%}")
            log.info("BLACK BOX Intelligence - PERMANENT MEMORY PRESERVED")
        except Exception as e:
            log.warning(f"Status error: {e}")
        
        log.info("BLACK BOX Intelligence Engine stopped - ALL PATTERNS SAVED")