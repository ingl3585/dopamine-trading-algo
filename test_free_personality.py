#!/usr/bin/env python3
"""
Free AI Trading Personality Test Script

Tests the completely free setup of the AI Trading Personality system
"""

import asyncio
import logging
import sys
import time
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add src to path for imports
sys.path.insert(0, 'src')

class FreePersonalityTester:
    """Test the free personality setup"""
    
    def __init__(self):
        self.test_results = {}
        self.personality = None
        self.config_manager = None
        
    async def run_all_tests(self):
        """Run complete test suite"""
        
        print("🆓 FREE AI Trading Personality Test Suite")
        print("=" * 50)
        
        # Test basic imports
        await self.test_imports()
        
        # Test configuration loading
        await self.test_configuration()
        
        # Test personality initialization
        await self.test_personality_init()
        
        # Test emotional states
        await self.test_emotional_states()
        
        # Test LLM integration (mock and Ollama)
        await self.test_llm_integration()
        
        # Test voice synthesis
        await self.test_voice_synthesis()
        
        # Test memory system
        await self.test_memory_system()
        
        # Test trading scenarios
        await self.test_trading_scenarios()
        
        # Show results
        self.show_test_results()
    
    async def test_imports(self):
        """Test that all required modules can be imported"""
        
        print("\n📦 Testing Module Imports")
        print("-" * 25)
        
        try:
            from personality.trading_personality import TradingPersonality, TriggerEvent, PersonalityConfig
            print("✅ trading_personality imported")
            
            from personality.config_manager import PersonalityConfigManager
            print("✅ config_manager imported")
            
            from personality.emotional_engine import EmotionalStateEngine
            print("✅ emotional_engine imported")
            
            from personality.llm_client import LLMClient
            print("✅ llm_client imported")
            
            from personality.personality_memory import PersonalityMemory
            print("✅ personality_memory imported")
            
            from personality.voice_synthesis import VoiceSynthesizer
            print("✅ voice_synthesis imported")
            
            from personality.personality_integration import PersonalityIntegration
            print("✅ personality_integration imported")
            
            self.test_results['imports'] = True
            
        except Exception as e:
            print(f"❌ Import failed: {e}")
            self.test_results['imports'] = False
    
    async def test_configuration(self):
        """Test configuration loading"""
        
        print("\n⚙️ Testing Configuration")
        print("-" * 25)
        
        try:
            from personality.config_manager import PersonalityConfigManager
            
            # Test free config
            if os.path.exists('config/personality_config_free.json'):
                self.config_manager = PersonalityConfigManager('config/personality_config_free.json')
                print("✅ Free config loaded")
            else:
                self.config_manager = PersonalityConfigManager()
                print("✅ Default config loaded")
            
            # Test configuration validation
            is_valid, errors = self.config_manager.validate_config()
            if is_valid:
                print("✅ Configuration is valid")
            else:
                print(f"⚠️ Configuration warnings: {errors}")
            
            # Show config summary
            summary = self.config_manager.get_config_summary()
            print(f"Personality: {summary['personality_name']}")
            print(f"LLM Model: {summary['llm_model']}")
            print(f"Voice Enabled: {summary['voice_enabled']}")
            
            self.test_results['configuration'] = True
            
        except Exception as e:
            print(f"❌ Configuration test failed: {e}")
            self.test_results['configuration'] = False
    
    async def test_personality_init(self):
        """Test personality initialization"""
        
        print("\n🤖 Testing Personality Initialization")
        print("-" * 35)
        
        try:
            from personality.trading_personality import TradingPersonality
            
            # Get configuration
            if self.config_manager:
                config = self.config_manager.get_personality_config()
            else:
                from personality.trading_personality import PersonalityConfig
                config = PersonalityConfig()
            
            # Initialize personality
            self.personality = TradingPersonality(config)
            print(f"✅ Personality '{config.personality_name}' initialized")
            
            # Test basic methods
            emotional_state = self.personality.get_current_emotional_state()
            print(f"✅ Current emotion: {emotional_state.primary_emotion.value}")
            print(f"   Confidence: {emotional_state.confidence:.2f}")
            print(f"   Stability: {emotional_state.emotional_stability:.2f}")
            
            summary = self.personality.get_personality_summary()
            print(f"✅ Personality summary generated")
            
            self.test_results['personality_init'] = True
            
        except Exception as e:
            print(f"❌ Personality initialization failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['personality_init'] = False
    
    async def test_emotional_states(self):
        """Test emotional state processing"""
        
        print("\n🎭 Testing Emotional States")
        print("-" * 25)
        
        if not self.personality:
            print("❌ No personality available for testing")
            self.test_results['emotional_states'] = False
            return
        
        try:
            # Test different emotional scenarios
            scenarios = [
                {
                    'name': 'Confident Bull Market',
                    'context': {
                        'subsystem_signals': {'dna': 0.8, 'temporal': 0.6, 'immune': 0.2, 'microstructure': 0.7, 'dopamine': 0.9},
                        'market_context': {'volatility': 0.02, 'trend_strength': 0.8},
                        'portfolio_state': {'unrealized_pnl': 200.0},
                        'recent_performance': [100, 150, 75],
                        'system_confidence': 0.85
                    }
                },
                {
                    'name': 'Fearful Bear Market',
                    'context': {
                        'subsystem_signals': {'dna': -0.7, 'temporal': -0.4, 'immune': -0.9, 'microstructure': -0.6, 'dopamine': -0.8},
                        'market_context': {'volatility': 0.08, 'trend_strength': -0.7},
                        'portfolio_state': {'unrealized_pnl': -300.0},
                        'recent_performance': [-50, -100, -75],
                        'system_confidence': 0.3
                    }
                }
            ]
            
            for scenario in scenarios:
                print(f"\n   Testing: {scenario['name']}")
                
                # Update emotional state
                emotional_state = self.personality.emotional_engine.update_emotional_state(scenario['context'])
                
                print(f"   Emotion: {emotional_state.primary_emotion.value}")
                print(f"   Confidence: {emotional_state.confidence:.2f}")
                print(f"   Fear: {emotional_state.fear:.2f}")
                print(f"   Excitement: {emotional_state.excitement:.2f}")
            
            print("✅ Emotional state testing completed")
            self.test_results['emotional_states'] = True
            
        except Exception as e:
            print(f"❌ Emotional state testing failed: {e}")
            self.test_results['emotional_states'] = False
    
    async def test_llm_integration(self):
        """Test LLM integration (mock and Ollama)"""
        
        print("\n🧠 Testing LLM Integration")
        print("-" * 25)
        
        if not self.personality:
            print("❌ No personality available for testing")
            self.test_results['llm_integration'] = False
            return
        
        try:
            from personality.trading_personality import TriggerEvent
            
            # Test with simple context
            test_context = {
                'subsystem_signals': {'dna': 0.5, 'temporal': 0.3, 'immune': 0.1, 'microstructure': 0.4, 'dopamine': 0.6},
                'market_data': {'volatility': 0.025, 'trend_strength': 0.4},
                'portfolio_state': {'unrealized_pnl': 100.0, 'daily_pnl': 25.0},
                'decision_context': {'decision_type': 'test', 'confidence': 0.7}
            }
            
            # Test commentary generation
            print("   Testing commentary generation...")
            commentary = await self.personality.process_trading_event(
                TriggerEvent.PERIODIC_UPDATE,
                test_context
            )
            
            if commentary:
                print(f"✅ Commentary generated: {commentary.text[:100]}...")
                print(f"   Confidence: {commentary.confidence:.2f}")
                print(f"   Themes: {', '.join(commentary.key_themes[:3])}")
            else:
                print("⚠️ No commentary generated")
            
            # Test manual query
            print("   Testing manual query...")
            response = await self.personality.manual_query("How do you feel about the current market?")
            
            if response:
                print(f"✅ Manual query response: {response.text[:100]}...")
            else:
                print("⚠️ No manual query response")
            
            self.test_results['llm_integration'] = True
            
        except Exception as e:
            print(f"❌ LLM integration test failed: {e}")
            self.test_results['llm_integration'] = False
    
    async def test_voice_synthesis(self):
        """Test voice synthesis"""
        
        print("\n🔊 Testing Voice Synthesis")
        print("-" * 25)
        
        try:
            from personality.voice_synthesis import VoiceSynthesizer
            
            # Test voice synthesizer initialization
            voice_config = {'service': 'local', 'enabled': True}
            voice_synthesizer = VoiceSynthesizer(voice_config)
            
            print("✅ Voice synthesizer initialized")
            print(f"   Available: {voice_synthesizer.is_available()}")
            
            # Test voice info
            voice_info = voice_synthesizer.get_voice_info()
            print(f"   Service: {voice_info['tts_service']}")
            print(f"   Voice ID: {voice_info['voice_id']}")
            
            # Test basic synthesis (mock)
            test_text = "Hello from your AI trading personality!"
            print(f"   Testing synthesis: '{test_text}'")
            
            audio_path = await voice_synthesizer.synthesize_speech(test_text)
            if audio_path:
                print("✅ Audio synthesis successful")
            else:
                print("⚠️ Audio synthesis returned None")
            
            self.test_results['voice_synthesis'] = True
            
        except Exception as e:
            print(f"❌ Voice synthesis test failed: {e}")
            self.test_results['voice_synthesis'] = False
    
    async def test_memory_system(self):
        """Test personality memory system"""
        
        print("\n🧠 Testing Memory System")
        print("-" * 22)
        
        try:
            from personality.personality_memory import PersonalityMemory
            
            # Initialize memory system
            memory = PersonalityMemory(memory_file="data/test_personality_memory.json")
            print("✅ Memory system initialized")
            
            # Add test memories
            memory_id = memory.add_memory(
                event_type="test_trade",
                emotional_state="confident",
                market_context={'volatility': 0.02},
                decision_context={'action': 'buy'},
                commentary="Test commentary",
                confidence=0.8,
                key_themes=['bullish', 'momentum']
            )
            
            print(f"✅ Memory added: {memory_id}")
            
            # Update with outcome
            memory.update_memory_outcome(memory_id, 150.0)
            print("✅ Memory outcome updated")
            
            # Get memory stats
            stats = memory.get_memory_stats()
            print(f"   Memory entries: {stats['short_term_entries']}")
            print(f"   Emotional patterns: {stats['emotional_patterns_count']}")
            
            # Test contextual guidance
            guidance = memory.get_contextual_guidance({
                'market_context': {'volatility': 0.02},
                'emotional_context': {'primary_emotion': 'confident'}
            })
            
            print("✅ Contextual guidance generated")
            
            self.test_results['memory_system'] = True
            
        except Exception as e:
            print(f"❌ Memory system test failed: {e}")
            self.test_results['memory_system'] = False
    
    async def test_trading_scenarios(self):
        """Test complete trading scenarios"""
        
        print("\n📈 Testing Trading Scenarios")
        print("-" * 28)
        
        if not self.personality:
            print("❌ No personality available for testing")
            self.test_results['trading_scenarios'] = False
            return
        
        try:
            from personality.trading_personality import TriggerEvent
            
            # Simulate a complete trading sequence
            scenarios = [
                (TriggerEvent.POSITION_ENTRY, "Position Entry", 0.8),
                (TriggerEvent.PROFIT_TARGET, "Profit Target Hit", 0.9),
                (TriggerEvent.POSITION_EXIT, "Position Exit", 0.7)
            ]
            
            base_context = {
                'subsystem_signals': {'dna': 0.6, 'temporal': 0.4, 'immune': 0.2, 'microstructure': 0.5, 'dopamine': 0.7},
                'market_data': {'volatility': 0.03, 'trend_strength': 0.5},
                'portfolio_state': {'unrealized_pnl': 75.0, 'daily_pnl': 50.0},
                'decision_context': {'decision_type': 'trading_sequence', 'confidence': 0.75}
            }
            
            for event, description, confidence in scenarios:
                print(f"\n   Testing: {description}")
                
                # Update context confidence
                base_context['decision_context']['confidence'] = confidence
                
                # Process event
                commentary = await self.personality.process_trading_event(event, base_context)
                
                if commentary:
                    print(f"   Response: {commentary.text[:80]}...")
                    print(f"   Confidence: {commentary.confidence:.2f}")
                else:
                    print(f"   No response generated")
                
                # Add some performance feedback
                if event == TriggerEvent.PROFIT_TARGET:
                    self.personality.update_performance_feedback(150.0, {'trade_type': 'profit_target'})
                    print("   Performance feedback added")
                
                await asyncio.sleep(0.5)  # Small delay between events
            
            print("✅ Trading scenario testing completed")
            self.test_results['trading_scenarios'] = True
            
        except Exception as e:
            print(f"❌ Trading scenario testing failed: {e}")
            self.test_results['trading_scenarios'] = False
    
    def show_test_results(self):
        """Show final test results"""
        
        print("\n🎯 Test Results Summary")
        print("=" * 25)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
        
        print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED! Your free AI Trading Personality setup is ready!")
            print("\nNext steps:")
            print("1. If using Ollama, ensure it's running: ollama serve")
            print("2. Copy config/personality_config_free.json to config/personality_config.json")
            print("3. Run: python personality_demo.py")
            print("4. Integrate with your trading system using PERSONALITY_INTEGRATION_GUIDE.md")
        else:
            print("⚠️ Some tests failed. Check the errors above.")
            print("\nTroubleshooting:")
            if not self.test_results.get('imports'):
                print("- Install dependencies: pip install numpy scipy pandas rich pyttsx3 aiohttp")
            if not self.test_results.get('llm_integration'):
                print("- For Ollama: Download from https://ollama.ai/download and run 'ollama pull llama2:7b-chat'")
                print("- Or use mock mode: copy config/personality_config_mock.json to config/personality_config.json")
            if not self.test_results.get('voice_synthesis'):
                print("- For Windows TTS: pip install pyttsx3")
        
        # Cleanup
        if self.personality:
            self.personality.shutdown()

async def main():
    """Main test function"""
    
    print("🆓 Starting Free AI Trading Personality Test")
    print("This will test the completely free setup without any API keys")
    print()
    
    tester = FreePersonalityTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())