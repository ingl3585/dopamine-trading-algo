#!/usr/bin/env python3
"""
AI Trading Personality Demo Script

Demonstrates the AI Trading Personality system functionality
"""

import asyncio
import json
import logging
import sys
import time
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add src to path for imports
sys.path.insert(0, 'src')

from personality.trading_personality import TradingPersonality, TriggerEvent, PersonalityConfig
from personality.config_manager import PersonalityConfigManager
from personality.personality_integration import PersonalityIntegration, PersonalityIntegrationConfig

class PersonalityDemo:
    """Demo class for AI Trading Personality system"""
    
    def __init__(self):
        self.config_manager = PersonalityConfigManager()
        self.personality = None
        self.integration = None
        
    async def initialize(self):
        """Initialize the personality system"""
        
        print("🤖 Initializing AI Trading Personality Demo")
        print("=" * 50)
        
        # Show configuration summary
        config_summary = self.config_manager.get_config_summary()
        print(f"Configuration Status: {config_summary['status']}")
        print(f"Personality: {config_summary['personality_name']}")
        print(f"LLM Model: {config_summary['llm_model']}")
        print(f"Voice Enabled: {config_summary['voice_enabled']}")
        print(f"Available Personalities: {config_summary['available_personalities']}")
        
        if config_summary['errors']:
            print(f"Configuration Errors: {config_summary['errors']}")
        
        print()
        
        # Initialize personality system
        personality_config = self.config_manager.get_personality_config()
        integration_config = self.config_manager.get_integration_config()
        
        self.personality = TradingPersonality(personality_config)
        self.integration = PersonalityIntegration(integration_config)
        
        print(f"✅ Personality system initialized for '{personality_config.personality_name}'")
        print()
    
    async def demo_emotional_states(self):
        """Demonstrate different emotional states"""
        
        print("🎭 Demonstrating Emotional States")
        print("-" * 30)
        
        # Test different market scenarios
        scenarios = [
            {
                'name': 'Bull Market Confidence',
                'context': {
                    'subsystem_signals': {'dna': 0.8, 'temporal': 0.6, 'immune': 0.2, 'microstructure': 0.7, 'dopamine': 0.9},
                    'market_data': {'volatility': 0.02, 'trend_strength': 0.8, 'regime': 'bull'},
                    'portfolio_state': {'unrealized_pnl': 150.0, 'daily_pnl': 75.0, 'open_positions': 1},
                    'decision_context': {'decision_type': 'position_entry', 'confidence': 0.85}
                }
            },
            {
                'name': 'Bear Market Fear',
                'context': {
                    'subsystem_signals': {'dna': -0.7, 'temporal': -0.4, 'immune': -0.9, 'microstructure': -0.6, 'dopamine': -0.8},
                    'market_data': {'volatility': 0.08, 'trend_strength': -0.7, 'regime': 'bear'},
                    'portfolio_state': {'unrealized_pnl': -250.0, 'daily_pnl': -120.0, 'open_positions': 2},
                    'decision_context': {'decision_type': 'stop_loss', 'confidence': 0.3}
                }
            },
            {
                'name': 'Sideways Confusion',
                'context': {
                    'subsystem_signals': {'dna': 0.1, 'temporal': -0.2, 'immune': 0.3, 'microstructure': -0.1, 'dopamine': 0.0},
                    'market_data': {'volatility': 0.015, 'trend_strength': 0.1, 'regime': 'sideways'},
                    'portfolio_state': {'unrealized_pnl': 25.0, 'daily_pnl': -10.0, 'open_positions': 0},
                    'decision_context': {'decision_type': 'analysis', 'confidence': 0.4}
                }
            }
        ]
        
        for scenario in scenarios:
            print(f"\n📊 Scenario: {scenario['name']}")
            
            # Process through personality system
            commentary = await self.personality.process_trading_event(
                TriggerEvent.PERIODIC_UPDATE, 
                scenario['context']
            )
            
            if commentary:
                emotional_state = self.personality.get_current_emotional_state()
                
                print(f"Emotional State: {emotional_state.primary_emotion.value}")
                print(f"Confidence: {emotional_state.confidence:.2f}")
                print(f"Fear: {emotional_state.fear:.2f}")
                print(f"Excitement: {emotional_state.excitement:.2f}")
                print(f"Intensity: {emotional_state.emotional_intensity:.2f}")
                print(f"Commentary: {commentary.text}")
                print(f"Key Themes: {', '.join(commentary.key_themes)}")
            else:
                print("No commentary generated")
        
        print()
    
    async def demo_trading_events(self):
        """Demonstrate personality responses to trading events"""
        
        print("📈 Demonstrating Trading Event Responses")
        print("-" * 40)
        
        # Simulate trading events
        events = [
            (TriggerEvent.POSITION_ENTRY, "Entering long position"),
            (TriggerEvent.PROFIT_TARGET, "Profit target hit!"),
            (TriggerEvent.STOP_LOSS, "Stop loss triggered"),
            (TriggerEvent.IMMUNE_WARNING, "Risk system alert"),
            (TriggerEvent.DOPAMINE_SPIKE, "Positive momentum surge")
        ]
        
        base_context = {
            'subsystem_signals': {'dna': 0.5, 'temporal': 0.3, 'immune': 0.1, 'microstructure': 0.4, 'dopamine': 0.6},
            'market_data': {'volatility': 0.025, 'trend_strength': 0.4, 'regime': 'normal'},
            'portfolio_state': {'unrealized_pnl': 50.0, 'daily_pnl': 25.0, 'open_positions': 1},
            'decision_context': {'decision_type': 'trading_event', 'confidence': 0.6}
        }
        
        for event, description in events:
            print(f"\n🔔 Event: {description}")
            
            commentary = await self.personality.process_trading_event(event, base_context)
            
            if commentary:
                print(f"Response: {commentary.text}")
                print(f"Confidence: {commentary.confidence:.2f}")
                print(f"Urgency: {commentary.emotional_intensity:.2f}")
            else:
                print("No response generated")
            
            # Small delay between events
            await asyncio.sleep(0.5)
        
        print()
    
    async def demo_manual_queries(self):
        """Demonstrate manual query handling"""
        
        print("💬 Demonstrating Manual Queries")
        print("-" * 30)
        
        queries = [
            "How are you feeling about the market right now?",
            "What's your confidence level on this trade?",
            "Should I be worried about this volatility spike?",
            "What do your subsystems tell you about this setup?",
            "Are you seeing any opportunities today?"
        ]
        
        for query in queries:
            print(f"\n❓ Query: {query}")
            
            response = await self.personality.manual_query(query)
            
            if response:
                print(f"Response: {response.text}")
            else:
                print("No response generated")
            
            await asyncio.sleep(0.3)
        
        print()
    
    async def demo_personality_learning(self):
        """Demonstrate personality learning from outcomes"""
        
        print("🧠 Demonstrating Personality Learning")
        print("-" * 35)
        
        # Simulate some trades with outcomes
        trades = [
            {'outcome': 150.0, 'context': {'decision_type': 'aggressive_entry', 'confidence': 0.8}},
            {'outcome': -75.0, 'context': {'decision_type': 'defensive_exit', 'confidence': 0.4}},
            {'outcome': 200.0, 'context': {'decision_type': 'momentum_trade', 'confidence': 0.9}},
            {'outcome': -120.0, 'context': {'decision_type': 'contrarian_bet', 'confidence': 0.3}},
            {'outcome': 300.0, 'context': {'decision_type': 'trend_follow', 'confidence': 0.85}}
        ]
        
        print("Simulating trade outcomes for learning...")
        
        for i, trade in enumerate(trades, 1):
            print(f"Trade {i}: {'+' if trade['outcome'] > 0 else ''}{trade['outcome']:.0f}")
            
            # Update personality with trade outcome
            self.personality.update_performance_feedback(trade['outcome'], trade['context'])
            
            await asyncio.sleep(0.2)
        
        # Show personality summary after learning
        print("\nPersonality Summary After Learning:")
        summary = self.personality.get_personality_summary()
        
        print(f"Current Emotion: {summary['emotional_metrics']['primary_emotion']}")
        print(f"Confidence: {summary['emotional_metrics']['confidence']:.2f}")
        print(f"Recent Performance: {summary['recent_performance']['avg_confidence']:.2f}")
        print(f"Common Themes: {[theme[0] for theme in summary['recent_performance']['common_themes'][:3]]}")
        
        print()
    
    async def demo_personality_consistency(self):
        """Demonstrate personality consistency across interactions"""
        
        print("🎯 Demonstrating Personality Consistency")
        print("-" * 38)
        
        # Multiple similar interactions to test consistency
        similar_context = {
            'subsystem_signals': {'dna': 0.6, 'temporal': 0.5, 'immune': 0.2, 'microstructure': 0.5, 'dopamine': 0.7},
            'market_data': {'volatility': 0.03, 'trend_strength': 0.5, 'regime': 'normal'},
            'portfolio_state': {'unrealized_pnl': 100.0, 'daily_pnl': 50.0, 'open_positions': 1},
            'decision_context': {'decision_type': 'hold_position', 'confidence': 0.7}
        }
        
        print("Generating multiple responses to similar market conditions...")
        
        responses = []
        for i in range(3):
            commentary = await self.personality.process_trading_event(
                TriggerEvent.PERIODIC_UPDATE, 
                similar_context
            )
            
            if commentary:
                responses.append(commentary.text)
                print(f"Response {i+1}: {commentary.text}")
            
            await asyncio.sleep(1)  # Allow time between responses
        
        # Analyze consistency
        if len(responses) > 1:
            print(f"\nConsistency Analysis:")
            print(f"Response count: {len(responses)}")
            print(f"Average length: {sum(len(r.split()) for r in responses) / len(responses):.1f} words")
            
            # Simple consistency check - look for repeated themes
            all_words = ' '.join(responses).lower().split()
            common_words = [word for word in set(all_words) if all_words.count(word) > 1 and len(word) > 4]
            print(f"Consistent themes: {common_words[:5]}")
        
        print()
    
    async def show_personality_status(self):
        """Show current personality system status"""
        
        print("📊 Personality System Status")
        print("-" * 28)
        
        if self.integration:
            status = self.integration.get_personality_status()
            
            print(f"System Enabled: {status['enabled']}")
            print(f"Current Emotion: {status.get('current_emotion', 'unknown')}")
            
            if 'emotional_metrics' in status:
                metrics = status['emotional_metrics']
                print(f"Confidence: {metrics['confidence']:.2f}")
                print(f"Fear: {metrics['fear']:.2f}")
                print(f"Excitement: {metrics['excitement']:.2f}")
                print(f"Emotional Intensity: {metrics['intensity']:.2f}")
                print(f"Emotional Stability: {metrics['stability']:.2f}")
            
            if 'recent_activity' in status:
                activity = status['recent_activity']
                print(f"Recent Commentary: {activity.get('commentary_count', 0)} entries")
                print(f"Average Confidence: {activity.get('avg_confidence', 0):.2f}")
        else:
            print("Personality integration not available")
        
        print()
    
    async def run_demo(self):
        """Run the complete demo"""
        
        try:
            await self.initialize()
            
            # Run demo sections
            await self.demo_emotional_states()
            await self.demo_trading_events()
            await self.demo_manual_queries()
            await self.demo_personality_learning()
            await self.demo_personality_consistency()
            await self.show_personality_status()
            
            print("🎉 Demo completed successfully!")
            print("\nThe AI Trading Personality system is now ready for integration")
            print("with your trading system. Key features demonstrated:")
            print("- Emotional state management")
            print("- Context-aware commentary generation")
            print("- Trading event processing")
            print("- Manual query handling")
            print("- Performance-based learning")
            print("- Personality consistency")
            
        except Exception as e:
            print(f"❌ Demo failed with error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.personality:
                self.personality.shutdown()

async def main():
    """Main demo function"""
    
    print("🚀 AI Trading Personality Demo Starting...")
    print()
    
    demo = PersonalityDemo()
    await demo.run_demo()

if __name__ == "__main__":
    asyncio.run(main())