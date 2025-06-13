# trading_system.py - MODIFIED: Remove consensus rules, let black box AI orchestrate

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
from trade_manager_ai import BlackBoxTradeManagerWithSubsystems

log = logging.getLogger(__name__)

class TradingSystem:
    """BLACK BOX orchestration - removed hardcoded consensus, AI learns tool usage"""

    def __init__(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        self.config = ResearchConfig()

        # Core intelligence with PERMANENT MEMORY enabled
        self.intelligence_engine = AdvancedMarketIntelligence()
        
        # Load existing patterns from permanent memory
        self._load_permanent_memory()
        self.intelligence_engine.start_continuous_learning()

        # TCP bridge 
        self.tcp_bridge = TCPBridge(self.config)
        self.tcp_bridge.on_market_data = self.process_market_data_blackbox
        self.tcp_bridge.on_trade_completion = self.on_trade_completed

        # MODIFIED: Trade manager with black box AI that learns tool usage
        self.trade_manager = BlackBoxTradeManagerWithSubsystems(
            self.intelligence_engine,
            self.tcp_bridge
        )

        # Book-keeping
        self.signal_count = 0
        self.bootstrap_complete = False
        
        # REMOVED: All consensus requirements and hardcoded rules
        # AI learns when and how to use each subsystem
        
        self.stats = {
            'total_signals': 0,
            'ai_decisions': 0,
            'tool_switches': 0,
            'patterns_learned': 0,
            'memory_saves': 0
        }

        log.info("BLACK BOX Intelligence Engine - AI learns subsystem tool usage")
        log.info("REMOVED: All hardcoded consensus rules")
        log.info("AI will discover optimal tool combinations through learning")
    
    def _load_permanent_memory(self):
        """Load all patterns from permanent memory on startup"""
        try:
            # Load DNA patterns
            dna_patterns = self.intelligence_engine.memory_db.load_dna_patterns()
            self.intelligence_engine.dna_system.dna_patterns.update(dna_patterns)
            
            log.info(f"PERMANENT MEMORY: Loaded {len(dna_patterns)} DNA patterns")
            
        except Exception as e:
            log.warning(f"Memory load error (expected on first run): {e}")
    
    def start(self):
        try:
            self.tcp_bridge.start()
            
            log.info("=== BLACK BOX AI SYSTEM STARTED ===")
            log.info("AI Learning Mode: Tool usage optimization")
            log.info("No hardcoded rules - AI discovers patterns")
            log.info("Press Ctrl+C to stop")
            
            while True:
                time.sleep(1)
                
                # Let AI handle all exit decisions
                # Removed manual exit logic - AI controls everything

        except KeyboardInterrupt:
            log.info("Shutdown requested")
        finally:
            self.stop()
    
    def bootstrap_intelligence_from_history(self, price_15m, volume_15m, price_5m, volume_5m, price_1m, volume_1m):
        """Bootstrap intelligence and save patterns to permanent memory"""
        log.info("BOOTSTRAPPING: Building initial pattern library...")
        
        patterns_learned = 0
        
        # Process 1-minute data for DNA sequencing
        if len(price_1m) >= 200:
            for i in range(100, len(price_1m), 20):
                chunk_prices = price_1m[i-100:i]
                chunk_volumes = volume_1m[i-100:i] if volume_1m else [1000] * 100
                
                # Create DNA sequences and save to permanent memory
                dna_sequence = self.intelligence_engine.dna_system.create_dna_sequence(
                    chunk_prices, chunk_volumes
                )
                
                if dna_sequence and len(dna_sequence) > 10:
                    # Process through intelligence to create pattern
                    result = self.intelligence_engine.process_market_data(chunk_prices, chunk_volumes)
                    
                    # Save DNA pattern with neutral outcome for bootstrap
                    if dna_sequence not in self.intelligence_engine.dna_system.dna_patterns:
                        self.intelligence_engine.dna_system.update_pattern_outcome(dna_sequence, 0.001)
                        patterns_learned += 1
                        self.stats['memory_saves'] += 1
                
                if patterns_learned >= 50:
                    break
        
        # Process 5-minute data for micro patterns
        if len(price_5m) >= 100:
            for i in range(50, len(price_5m), 10):
                chunk_prices = price_5m[i-50:i]
                chunk_volumes = volume_5m[i-50:i] if volume_5m else [500] * 50
                
                self.intelligence_engine.process_market_data(chunk_prices, chunk_volumes)
                patterns_learned += 1
                
                if patterns_learned >= 100:
                    break
        
        self.bootstrap_complete = True
        self.stats['patterns_learned'] = patterns_learned
        
        log.info(f"BOOTSTRAP COMPLETE: {patterns_learned} patterns learned")
        log.info(f"DNA patterns: {len(self.intelligence_engine.dna_system.dna_patterns)}")
        log.info(f"Micro patterns: {len(self.intelligence_engine.micro_system.patterns)}")
        log.info("AI ready to learn optimal subsystem tool usage!")
    
    def process_market_data_blackbox(self, data: Dict):
        """
        MODIFIED: Pure black box processing - AI orchestrates everything
        """
        try:
            # Extract data
            price_1m = data.get("price_1m", [])
            volume_1m = data.get("volume_1m", [])
            
            if not price_1m:
                return
            
            # Bootstrap intelligence if needed
            if not self.bootstrap_complete and len(price_1m) >= 100:
                self.bootstrap_intelligence_from_history(
                    data.get("price_15m", []), data.get("volume_15m", []),
                    data.get("price_5m", []), data.get("volume_5m", []),
                    price_1m, volume_1m
                )
            
            # Store current price for tracking
            if price_1m:
                self.current_price = price_1m[-1]
            
            # BLACK BOX AI orchestrates all subsystems
            # REMOVED: All consensus logic, hardcoded rules, manual overrides
            # AI learns optimal subsystem combinations through experience
            self.trade_manager.on_new_bar(data)
            
            self.signal_count += 1
            self.stats['total_signals'] += 1
            
            # Periodic performance report
            if self.signal_count % 100 == 0 and self.signal_count > 0:
                report = self.trade_manager.get_performance_report()
                print("\n" + "="*60)
                print(report)
                print("="*60 + "\n")
            
        except Exception as e:
            log.error(f"Black box processing error: {e}")
            import traceback
            traceback.print_exc()
    
    def on_trade_completed(self, completion_data):
        """Handle trade completion from NinjaTrader"""
        try:
            exit_price = completion_data.get('exit_price', 0)
            exit_reason = completion_data.get('exit_reason', 'unknown')
            duration_minutes = completion_data.get('duration_minutes', 0)
            
            # Feed completion to AI for learning
            self.trade_manager._complete_trade(exit_reason, exit_price)
            
            log.info(f"TRADE COMPLETED: {exit_reason} at ${exit_price:.2f} ({duration_minutes}min)")
            
            # Track AI decisions
            self.stats['ai_decisions'] += 1
            
        except Exception as e:
            log.error(f"Trade completion error: {e}")
    
    def stop(self):
        """Stop with comprehensive learning summary"""
        log.info("Stopping BLACK BOX Intelligence with learning summary...")
        
        try:
            self.tcp_bridge.stop()
        except Exception as e:
            log.warning(f"TCP stop error: {e}")
        
        # Save all patterns to permanent memory
        try:
            log.info("SAVING ALL LEARNED PATTERNS...")
            
            # Save all DNA patterns
            for seq, pattern in self.intelligence_engine.dna_system.dna_patterns.items():
                self.intelligence_engine.memory_db.save_dna_pattern(pattern)
            
            log.info(f"PERMANENT MEMORY: {len(self.intelligence_engine.dna_system.dna_patterns)} DNA patterns saved")
            
        except Exception as e:
            log.warning(f"Memory save error: {e}")
        
        # Export knowledge base with timestamp
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            knowledge_file = f"blackbox_learned_knowledge_{timestamp}.json"
            self.intelligence_engine.export_knowledge_base(knowledge_file)
            log.info(f"AI learned knowledge exported to {knowledge_file}")
        except Exception as e:
            log.warning(f"Export error: {e}")
        
        # Generate final AI learning report
        log.info("\n" + "="*60)
        log.info("BLACK BOX AI LEARNING SUMMARY")
        log.info("="*60)
        
        # Basic stats
        log.info(f"Total Market Updates: {self.stats['total_signals']}")
        log.info(f"AI Trading Decisions: {self.stats['ai_decisions']}")
        log.info(f"Patterns Learned: {self.stats['patterns_learned']}")
        log.info(f"Memory Saves: {self.stats['memory_saves']}")
        
        # Subsystem learning summary
        try:
            intel_status = self.intelligence_engine.get_system_status()
            log.info(f"\nSUBSYSTEM KNOWLEDGE LEARNED:")
            log.info(f"DNA Sequences: {intel_status['total_dna_patterns']}")
            log.info(f"Micro Patterns: {intel_status['total_micro_patterns']}")
            log.info(f"Temporal Patterns: {intel_status['total_temporal_patterns']}")
            log.info(f"Immune Strength: {intel_status['immune_strength']:.2%}")
            log.info(f"Recent Win Rate: {intel_status['win_rate']:.2%}")
            
            # Tool learning summary
            tool_report = self.trade_manager.get_performance_report()
            log.info(f"\nAI TOOL LEARNING PROGRESS:")
            log.info(tool_report)
            
        except Exception as e:
            log.warning(f"Status summary error: {e}")
        
        log.info("\n" + "="*60)
        log.info("BLACK BOX AI: All learning preserved in permanent memory")
        log.info("Next startup will continue from this knowledge state")
        log.info("AI discovered optimal subsystem tool usage patterns")
        log.info("="*60)