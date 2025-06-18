# trading_system.py - PURE BLACK BOX: Zero hardcoded knowledge

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
from trade_manager_ai import PureBlackBoxTradeManager  # CHANGED to pure version

log = logging.getLogger(__name__)

class PureBlackBoxTradingSystem:
    """PURE BLACK BOX: AI learns everything from scratch - zero hardcoded knowledge"""

    def __init__(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        self.config = ResearchConfig()

        # Core intelligence with permanent memory
        self.intelligence_engine = AdvancedMarketIntelligence()
        
        # Load existing patterns from permanent memory
        self._load_permanent_memory()
        self.intelligence_engine.start_continuous_learning()

        # TCP bridge 
        self.tcp_bridge = TCPBridge(self.config)
        self.tcp_bridge.on_market_data = self.process_pure_blackbox_data
        self.tcp_bridge.on_trade_completion = self.on_trade_completed

        # PURE BLACK BOX trade manager - no wisdom, no biases
        self.trade_manager = PureBlackBoxTradeManager(
            self.intelligence_engine,
            self.tcp_bridge
        )

        # Pure statistics
        self.signal_count = 0
        self.bootstrap_complete = False
        
        self.stats = {
            'total_signals': 0,
            'pure_ai_decisions': 0,
            'tool_discoveries': 0,
            'patterns_learned': 0,
            'memory_saves': 0,
            'learning_iterations': 0
        }

        log.info("PURE BLACK BOX Trading System - Zero hardcoded knowledge")
        log.info("AI will discover ALL trading strategies through experience")
        log.info("No wisdom, no biases, no preset rules - pure learning")
    
    def _load_permanent_memory(self):
        """Load patterns from permanent memory"""
        try:
            dna_patterns = self.intelligence_engine.memory_db.load_dna_patterns()
            self.intelligence_engine.dna_system.dna_patterns.update(dna_patterns)
            
            log.info(f"PERMANENT MEMORY: Loaded {len(dna_patterns)} learned patterns")
            
        except Exception as e:
            log.info(f"Starting fresh - no previous learning found: {e}")
    
    def start(self):
        try:
            self.tcp_bridge.start()
            
            log.info("=== PURE BLACK BOX AI SYSTEM STARTED ===")
            log.info("Learning Mode: Pure discovery - zero preset knowledge")
            log.info("AI will learn optimal subsystem usage through trial and error")
            log.info("Progressive safety limits will scale with learning progress")
            log.info("Press Ctrl+C to stop and save learned knowledge")
            
            while True:
                time.sleep(1)
                
                # Pure black box - no manual interventions
                # AI controls all decisions

        except KeyboardInterrupt:
            log.info("Shutdown requested - saving pure learning")
        finally:
            self.stop()
    
    def bootstrap_pure_learning(self, price_15m, volume_15m, price_5m, volume_5m, price_1m, volume_1m):
        """Bootstrap with minimal neutral patterns - no trading knowledge"""
        log.info("PURE BOOTSTRAP: Creating neutral pattern library...")
        
        patterns_learned = 0
        
        # Create neutral patterns without any trading assumptions
        if len(price_1m) >= 200:
            for i in range(100, len(price_1m), 30):  # Less frequent sampling
                chunk_prices = price_1m[i-100:i]
                chunk_volumes = volume_1m[i-100:i] if volume_1m else [1000] * 100
                
                # Create DNA sequences with neutral outcomes
                dna_sequence = self.intelligence_engine.dna_system.create_dna_sequence(
                    chunk_prices, chunk_volumes
                )
                
                if dna_sequence and len(dna_sequence) > 10:
                    # Process through intelligence
                    result = self.intelligence_engine.process_market_data(chunk_prices, chunk_volumes)
                    
                    # Save with completely neutral outcome - no bias
                    if dna_sequence not in self.intelligence_engine.dna_system.dna_patterns:
                        self.intelligence_engine.dna_system.update_pattern_outcome(dna_sequence, 0.0)  # Truly neutral
                        patterns_learned += 1
                        self.stats['memory_saves'] += 1
                
                if patterns_learned >= 30:  # Fewer bootstrap patterns
                    break
        
        # Minimal micro pattern creation
        if len(price_5m) >= 100:
            for i in range(50, len(price_5m), 15):
                chunk_prices = price_5m[i-50:i]
                chunk_volumes = volume_5m[i-50:i] if volume_5m else [500] * 50
                
                self.intelligence_engine.process_market_data(chunk_prices, chunk_volumes)
                patterns_learned += 1
                
                if patterns_learned >= 50:  # Minimal bootstrap
                    break
        
        self.bootstrap_complete = True
        self.stats['patterns_learned'] = patterns_learned
        
        log.info(f"PURE BOOTSTRAP COMPLETE: {patterns_learned} neutral patterns created")
        log.info("AI ready for pure discovery learning!")
    
    def process_pure_blackbox_data(self, data: Dict):
        """PURE BLACK BOX: Zero hardcoded trading logic"""
        try:
            # Extract data
            price_1m = data.get("price_1m", [])
            volume_1m = data.get("volume_1m", [])
            
            if not price_1m:
                return
            
            # Minimal bootstrap if needed
            if not self.bootstrap_complete and len(price_1m) >= 100:
                self.bootstrap_pure_learning(
                    data.get("price_15m", []), data.get("volume_15m", []),
                    data.get("price_5m", []), data.get("volume_5m", []),
                    price_1m, volume_1m
                )
            
            # Store current price
            if price_1m:
                self.current_price = price_1m[-1]
            
            # PURE BLACK BOX: AI discovers everything
            self.trade_manager.on_new_bar(data)
            
            self.signal_count += 1
            self.stats['total_signals'] += 1
            self.stats['learning_iterations'] += 1
            
            # Pure learning progress report
            if self.signal_count % 50 == 0 and self.signal_count > 0:
                log.info(f"PURE LEARNING PROGRESS: {self.signal_count} market updates processed")
                
                # Tool discovery progress
                discoveries = self.trade_manager.trade_stats['tool_discovery']
                total_tools_tried = sum(discoveries.values())
                if total_tools_tried > 0:
                    log.info(f"Tool Discovery: {total_tools_tried} total experiments")
                    for tool, count in discoveries.items():
                        if count > 0:
                            log.info(f"  {tool.upper()}: {count} experiments")
            
            # Comprehensive report less frequently
            if self.signal_count % 200 == 0 and self.signal_count > 0:
                report = self.trade_manager.get_performance_report()
                print("\n" + "="*60)
                print("PURE BLACK BOX LEARNING REPORT")
                print("="*60)
                print(report)
                print("="*60 + "\n")
            
        except Exception as e:
            log.error(f"Pure black box error: {e}")
            import traceback
            traceback.print_exc()
    
    def on_trade_completed(self, completion_data):
        """Handle trade completion for pure learning"""
        try:
            exit_price = completion_data.get('exit_price', 0)
            exit_reason = completion_data.get('exit_reason', 'unknown')
            duration_minutes = completion_data.get('duration_minutes', 0)
            
            # Feed to pure AI for learning
            self.trade_manager._complete_trade(exit_reason, exit_price)
            
            log.info(f"PURE LEARNING: Trade completed - {exit_reason} at ${exit_price:.2f}")
            
            self.stats['pure_ai_decisions'] += 1
            
        except Exception as e:
            log.error(f"Trade completion error: {e}")
    
    def stop(self):
        """Stop with pure learning summary"""
        log.info("Stopping PURE BLACK BOX system...")
        
        try:
            self.tcp_bridge.stop()
        except Exception as e:
            log.warning(f"TCP stop error: {e}")
        
        # Save all learned patterns
        try:
            log.info("SAVING ALL PURE LEARNING...")
            
            # Save DNA patterns
            for seq, pattern in self.intelligence_engine.dna_system.dna_patterns.items():
                self.intelligence_engine.memory_db.save_dna_pattern(pattern)
            
            log.info(f"PURE MEMORY: {len(self.intelligence_engine.dna_system.dna_patterns)} patterns saved")
            
        except Exception as e:
            log.warning(f"Memory save error: {e}")
        
        # Export pure learned knowledge
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            knowledge_file = f"pure_blackbox_learning_{timestamp}.json"
            self.intelligence_engine.export_knowledge_base(knowledge_file)
            log.info(f"Pure AI learning exported to {knowledge_file}")
        except Exception as e:
            log.warning(f"Export error: {e}")
        
        # Pure learning summary
        log.info("\n" + "="*60)
        log.info("PURE BLACK BOX LEARNING SUMMARY")
        log.info("="*60)
        
        log.info(f"Market Updates Processed: {self.stats['total_signals']}")
        log.info(f"Pure AI Decisions: {self.stats['pure_ai_decisions']}")
        log.info(f"Learning Iterations: {self.stats['learning_iterations']}")
        log.info(f"Patterns Created: {self.stats['patterns_learned']}")
        
        # Tool discovery summary
        try:
            discoveries = self.trade_manager.trade_stats['tool_discovery']
            total_experiments = sum(discoveries.values())
            
            log.info(f"\nPURE TOOL DISCOVERY:")
            log.info(f"Total Experiments: {total_experiments}")
            
            for tool, count in discoveries.items():
                percentage = (count / max(1, total_experiments)) * 100
                log.info(f"  {tool.upper()}: {count} uses ({percentage:.1f}%)")
            
            # Learning phase
            current_phase = self.trade_manager.safety_manager.current_phase
            log.info(f"\nLearning Phase: {current_phase}")
            
        except Exception as e:
            log.warning(f"Discovery summary error: {e}")
        
        # Subsystem learning
        try:
            intel_status = self.intelligence_engine.get_system_status()
            log.info(f"\nSUBSYSTEM PATTERNS DISCOVERED:")
            log.info(f"DNA Sequences: {intel_status['total_dna_patterns']}")
            log.info(f"Micro Patterns: {intel_status['total_micro_patterns']}")
            log.info(f"Temporal Patterns: {intel_status['total_temporal_patterns']}")
            
            if intel_status['win_rate'] > 0:
                log.info(f"Current Win Rate: {intel_status['win_rate']:.1%}")
            
        except Exception as e:
            log.warning(f"Intel summary error: {e}")
        
        # AI model saving
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"pure_ai_model_{timestamp}.pt"
            self.trade_manager.agent.save_model(model_path)
            log.info(f"Pure AI model saved to {model_path}")
        except Exception as e:
            log.warning(f"Model save error: {e}")
        
        log.info("\n" + "="*60)
        log.info("PURE BLACK BOX: All discoveries preserved")
        log.info("Next startup will continue pure learning from this state")
        log.info("AI learned optimal subsystem usage through pure experience")
        log.info("Zero hardcoded knowledge - everything discovered")
        log.info("="*60)

# Alias for backwards compatibility
TradingSystem = PureBlackBoxTradingSystem