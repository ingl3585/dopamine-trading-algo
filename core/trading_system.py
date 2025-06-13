# core/trading_system.py - CLEAN VERSION following your original prompt

import os
import threading
import logging
import numpy as np
from typing import Dict
from datetime import datetime
from config import ResearchConfig
from features.feature_extractor import FeatureExtractor
from models.logistic_model import LogisticSignalModel
from communication.tcp_bridge import TCPBridge
from advanced_market_intelligence import AdvancedMarketIntelligence
import queue
import time

log = logging.getLogger(__name__)

class TradingSystem:
    """Market Intelligence Engine - Black Box Learning System"""
    
    def __init__(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        self.config = ResearchConfig()
        self.feature_extractor = FeatureExtractor(self.config)
        self.traditional_model = LogisticSignalModel(self.config)
        
        # The core intelligence engine - this is the "black box"
        self.intelligence_engine = AdvancedMarketIntelligence()
        self.intelligence_engine.start_continuous_learning()
        
        # TCP Bridge
        self.tcp_bridge = TCPBridge(self.config)
        self.tcp_bridge.on_market_data = self.process_market_data
        self.tcp_bridge.on_trade_completion = self.on_trade_completed
        
        # Background intelligence processing
        self.intelligence_queue = queue.Queue()
        self.start_intelligence_processor()
        
        # Shutdown handling
        self.shutdown_event = threading.Event()
        
        # State tracking - MINIMAL, let intelligence learn
        self.signal_count = 0
        self.intelligence_active = False
        
        # Statistics for monitoring
        self.stats = {
            'total_signals': 0,
            'intelligence_enhanced': 0,
            'intelligence_filtered': 0,
            'traditional_signals': 0,
            'consensus_signals': 0
        }
        
        log.info("Market Intelligence Engine initialized - Black Box Learning Active")
    
    def start_intelligence_processor(self):
        """Background intelligence processing - non-blocking"""
        def intelligence_worker():
            while True:
                try:
                    data = self.intelligence_queue.get(timeout=1)
                    if data is None:  # Shutdown signal
                        break
                    
                    # Process with intelligence engine
                    prices = data['prices']
                    volumes = data['volumes']
                    callback = data['callback']
                    
                    # The intelligence engine does its pattern recognition magic
                    intelligence_result = self.intelligence_engine.process_market_data(
                        prices, volumes, datetime.now()
                    )
                    
                    # Execute callback with result
                    callback(intelligence_result)
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    log.error(f"Intelligence processing error: {e}")
        
        thread = threading.Thread(target=intelligence_worker, daemon=True, name="IntelligenceProcessor")
        thread.start()
        log.info("Intelligence processor thread started")
    
    def start(self):
        """Start the Market Intelligence Engine"""
        try:
            log.info("Starting Market Intelligence Engine")
            log.info("Strategy: Multi-Layer AI with Continuous Learning")
            log.info("Philosophy: Black Box Pattern Discovery")
            
            self.tcp_bridge.start()
            
            # Simple loop - just wait for Ctrl+C
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            log.info("Shutdown requested")
        finally:
            self.stop()
    
    def process_market_data(self, data: Dict):
        """Process market data - SIMPLE DEBUG VERSION"""
        try:
            # Extract all timeframe data
            price_15m = data.get("price_15m", [])
            volume_15m = data.get("volume_15m", [])
            price_5m = data.get("price_5m", [])
            volume_5m = data.get("volume_5m", [])
            price_1m = data.get("price_1m", [])
            volume_1m = data.get("volume_1m", [])
            
            print(f"=== DATA RECEIVED ===")
            print(f"15m: {len(price_15m)} bars, 5m: {len(price_5m)} bars, 1m: {len(price_1m)} bars")
            
            # Extract features for traditional model
            features = self.feature_extractor.extract_features(
                price_15m, volume_15m, price_5m, volume_5m, price_1m, volume_1m
            )

            if features is None:
                print("FAILED: No features extracted")
                return

            print(f"SUCCESS: Features extracted")
            print(f"  RSI 5m: {features.rsi_5m:.1f}")
            print(f"  BB pos 5m: {features.bb_position_5m:.3f}")
            print(f"  EMA trend 5m: {features.ema_trend_5m:.6f}")

            # Generate traditional signal (baseline)
            traditional_action, traditional_confidence, traditional_quality = \
                self.traditional_model.predict(features)
            
            print(f"Traditional signal: action={traditional_action}, conf={traditional_confidence:.3f}")

            # ADD TRAINING SAMPLE IMMEDIATELY - THIS IS THE KEY FIX
            print("Adding training sample...")
            self.traditional_model.add_training_sample(features)
            print(f"Training samples so far: {len(self.traditional_model.signal_history)}")
            
            # Check if model is trained
            print(f"Model is trained: {self.traditional_model.is_trained}")
            
            # Only do intelligence processing if we have a good traditional signal
            if traditional_confidence > 0.3:  # Lower threshold for testing
                print("Queuing intelligence processing...")
                
                # Queue intelligence processing (non-blocking)
                def intelligence_callback(intelligence_result):
                    print("Intelligence callback received")
                    final_signal = self.fuse_intelligence_signals(
                        traditional_action, traditional_confidence, traditional_quality,
                        intelligence_result
                    )
                    
                    # Send the final signal if it meets criteria
                    if final_signal['send_signal']:
                        print(f"SENDING SIGNAL: {final_signal['reasoning']}")
                        self.tcp_bridge.send_signal(
                            final_signal['action'], 
                            final_signal['confidence'], 
                            final_signal['quality']
                        )
                        log.info(f"INTELLIGENCE SIGNAL: {final_signal['reasoning']}")
                    else:
                        print(f"NO SIGNAL: {final_signal['reasoning']}")
                
                # Prepare data for intelligence engine
                intel_prices = price_1m if len(price_1m) >= 50 else price_5m
                intel_volumes = volume_1m if len(volume_1m) >= 50 else volume_5m
                
                # Queue for background processing
                self.intelligence_queue.put({
                    'prices': intel_prices[-200:],
                    'volumes': intel_volumes[-200:],
                    'callback': intelligence_callback
                })
            else:
                print(f"Skipping intelligence (low confidence: {traditional_confidence:.3f})")
            
            self.signal_count += 1
            
            # Activate intelligence after we have some data
            if not self.intelligence_active and self.signal_count > 5:  # Reduced threshold
                self.intelligence_active = True
                log.info("Intelligence engine activated - Pattern learning engaged")
            
            print("=== PROCESSING COMPLETE ===\n")
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            log.error(f"Market data processing error: {e}")
    
    def fuse_intelligence_signals(self, traditional_action, traditional_confidence, 
                                 traditional_quality, intelligence_result):
        """Signal fusion using black box intelligence - CORE LOGIC"""
        
        self.stats['total_signals'] += 1
        
        # Extract intelligence insights
        intel_signal = intelligence_result.get('signal_strength', 0)
        intel_confidence = intelligence_result.get('confidence', 0)
        is_dangerous = intelligence_result.get('is_dangerous_pattern', False)
        is_beneficial = intelligence_result.get('is_beneficial_pattern', False)
        
        # BLACK BOX DECISION LOGIC
        
        # 1. IMMUNE SYSTEM OVERRIDE - Dangerous patterns
        if is_dangerous and intel_confidence > 0.5:
            self.stats['intelligence_filtered'] += 1
            return {
                'action': 0,  # HOLD
                'confidence': 0.0,
                'quality': 'dangerous_filtered',
                'send_signal': False,
                'reasoning': f"DANGEROUS PATTERN DETECTED - Override traditional signal"
            }
        
        # 2. BENEFICIAL PATTERN BOOST
        if is_beneficial and intel_confidence > 0.6 and traditional_action != 0:
            boost_factor = 1.0 + (intel_confidence * 0.4)  # Up to 40% boost
            final_confidence = min(traditional_confidence * boost_factor, 0.95)
            self.stats['intelligence_enhanced'] += 1
            return {
                'action': traditional_action,
                'confidence': final_confidence,
                'quality': 'intelligence_enhanced',
                'send_signal': final_confidence >= self.config.CONFIDENCE_THRESHOLD,
                'reasoning': f"BENEFICIAL PATTERN BOOST - Traditional {traditional_confidence:.3f} → {final_confidence:.3f}"
            }
        
        # 3. CONSENSUS VALIDATION - Both systems agree
        intel_action = self.signal_to_action(intel_signal)
        if (traditional_action == intel_action and traditional_action != 0 and 
            intel_confidence > 0.5):
            
            # High confidence consensus
            consensus_confidence = min(
                (traditional_confidence * 0.7 + intel_confidence * 0.3) * 1.15,  # 15% consensus bonus
                0.95
            )
            self.stats['consensus_signals'] += 1
            return {
                'action': traditional_action,
                'confidence': consensus_confidence,
                'quality': 'consensus_validated',
                'send_signal': consensus_confidence >= self.config.CONFIDENCE_THRESHOLD,
                'reasoning': f"CONSENSUS SIGNAL - Both systems agree: {consensus_confidence:.3f}"
            }
        
        # 4. INTELLIGENCE OVERRIDE - High confidence intelligence signal
        if (intel_confidence > 0.75 and intel_action != 0 and 
            abs(intel_signal) > 0.3):
            
            self.stats['intelligence_enhanced'] += 1
            return {
                'action': intel_action,
                'confidence': intel_confidence * 0.9,  # Slight discount for override
                'quality': 'intelligence_override',
                'send_signal': True,
                'reasoning': f"INTELLIGENCE OVERRIDE - High confidence: {intel_confidence:.3f}"
            }
        
        # 5. TRADITIONAL SIGNAL - Intelligence not confident enough
        if traditional_confidence >= self.config.CONFIDENCE_THRESHOLD:
            self.stats['traditional_signals'] += 1
            return {
                'action': traditional_action,
                'confidence': traditional_confidence,
                'quality': traditional_quality,
                'send_signal': True,
                'reasoning': f"TRADITIONAL SIGNAL - Intelligence neutral: {traditional_confidence:.3f}"
            }
        
        # 6. NO SIGNAL - Nothing confident enough
        return {
            'action': 0,
            'confidence': 0.0,
            'quality': 'no_consensus',
            'send_signal': False,
            'reasoning': f"NO CLEAR SIGNAL - Awaiting better setup"
        }
    
    def signal_to_action(self, signal_strength):
        """Convert intelligence signal strength to action"""
        if signal_strength > 0.2:
            return 1  # BUY
        elif signal_strength < -0.2:
            return 2  # SELL
        else:
            return 0  # HOLD
    
    def on_trade_completed(self, completion_data):
        """Handle trade completion - Feed back to intelligence"""
        try:
            exit_price = completion_data.get('exit_price', 0)
            exit_reason = completion_data.get('exit_reason', 'unknown')
            duration_minutes = completion_data.get('duration_minutes', 0)
            
            # Simple PnL calculation (this would be more sophisticated in practice)
            # For now, just tell intelligence about the trade outcome
            
            # Record with intelligence engine for learning
            outcome = 0.01 if exit_reason in ['take_profit', 'target'] else -0.01
            
            self.intelligence_engine.record_trade_outcome(
                datetime.now(), 
                outcome,
                exit_price, 
                exit_price
            )
            
            log.info(f"Trade completed - Intelligence updated: {exit_reason}")
            
        except Exception as e:
            log.error(f"Trade completion error: {e}")
    
    def stop(self):
        """Stop the Market Intelligence Engine - KEEP YOUR FULL VERSION"""
        log.info("Stopping Market Intelligence Engine...")
        
        # Stop intelligence processing
        try:
            self.intelligence_queue.put(None)
        except Exception as e:
            log.warning(f"Intelligence queue stop error: {e}")
        
        # Stop TCP bridge
        try:
            self.tcp_bridge.stop()
        except Exception as e:
            log.warning(f"TCP bridge stop error: {e}")
        
        # Save traditional model
        try:
            self.traditional_model.save_model()
        except Exception as e:
            log.warning(f"Model save error: {e}")
        
        # Export intelligence knowledge base
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            knowledge_file = f"intelligence_patterns_{timestamp}.json"
            self.intelligence_engine.export_knowledge_base(knowledge_file)
            log.info(f"Intelligence patterns exported to {knowledge_file}")
        except Exception as e:
            log.warning(f"Knowledge export error: {e}")
        
        # Print final statistics
        total = self.stats['total_signals']
        if total > 0:
            log.info("=== MARKET INTELLIGENCE FINAL REPORT ===")
            log.info(f"Total Signals Processed: {total}")
            log.info(f"Intelligence Enhanced: {self.stats['intelligence_enhanced']} ({self.stats['intelligence_enhanced']/total*100:.1f}%)")
            log.info(f"Intelligence Filtered: {self.stats['intelligence_filtered']} ({self.stats['intelligence_filtered']/total*100:.1f}%)")
            log.info(f"Consensus Signals: {self.stats['consensus_signals']} ({self.stats['consensus_signals']/total*100:.1f}%)")
            log.info(f"Traditional Signals: {self.stats['traditional_signals']} ({self.stats['traditional_signals']/total*100:.1f}%)")
            
            # Get intelligence insights
            try:
                intel_status = self.intelligence_engine.get_system_status()
                log.info(f"DNA Patterns Learned: {intel_status['total_dna_patterns']}")
                log.info(f"Intelligence Win Rate: {intel_status['win_rate']:.2%}")
                log.info("Market Intelligence Engine - Learning Complete")
            except:
                pass
        
        log.info("Market Intelligence Engine stopped")