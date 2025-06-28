"""
Test Suite for AI Trading Personality System

Comprehensive tests for all personality system components
"""

import asyncio
import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import dataclass
from typing import Dict, Any

# Import personality system components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.personality.trading_personality import TradingPersonality, PersonalityConfig, TriggerEvent
from src.personality.emotional_engine import EmotionalStateEngine, EmotionalMetrics, EmotionalState
from src.personality.llm_client import LLMClient, CommentaryRequest, CommentaryResponse
from src.personality.personality_memory import PersonalityMemory, MemoryEntry
from src.personality.voice_synthesis import VoiceSynthesizer, VoiceSettings
from src.personality.personality_integration import PersonalityIntegration, PersonalityIntegrationConfig
from src.personality.config_manager import PersonalityConfigManager
from src.shared.types import Features

@dataclass
class MockDecision:
    action: str = "buy"
    confidence: float = 0.7
    size: float = 1.0
    primary_tool: str = "dna"
    exploration: bool = False
    uncertainty_estimate: float = 0.3

@dataclass
class MockMarketData:
    prices_1m: list = None
    unrealized_pnl: float = 0.0
    daily_pnl: float = 0.0
    account_balance: float = 25000.0
    open_positions: int = 0
    margin_utilization: float = 0.0
    
    def __post_init__(self):
        if self.prices_1m is None:
            self.prices_1m = [100.0, 101.0, 102.0]

class TestEmotionalStateEngine:
    """Test the emotional state engine"""
    
    def setup_method(self):
        self.engine = EmotionalStateEngine()
    
    def test_initialization(self):
        assert self.engine.base_confidence == 0.6
        assert self.engine.fear_sensitivity == 0.8
        assert len(self.engine.emotion_history) == 0
    
    def test_confidence_calculation(self):
        # Test with strong consensus
        subsystem_signals = {
            'dna': 0.8,
            'temporal': 0.7,
            'immune': 0.6,
            'microstructure': 0.9,
            'dopamine': 0.5
        }
        
        recent_performance = [0.5, 0.3, 0.8, 0.2]  # Mixed performance
        
        confidence = self.engine._calculate_confidence(
            subsystem_signals, recent_performance, 0.8
        )
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be above average with strong signals
    
    def test_fear_calculation(self):
        # Test with threatening conditions
        subsystem_signals = {'immune': -0.8, 'dopamine': -0.5}
        portfolio_state = {'unrealized_pnl': -500.0, 'position_exposure': 0.8}
        market_context = {'volatility': 0.08}
        
        fear = self.engine._calculate_fear(
            subsystem_signals, portfolio_state, market_context
        )
        
        assert 0.0 <= fear <= 1.0
        assert fear > 0.3  # Should show significant fear
    
    def test_excitement_calculation(self):
        # Test with positive conditions
        subsystem_signals = {'dna': 0.9, 'dopamine': 0.8}
        portfolio_state = {'unrealized_pnl': 500.0}
        recent_performance = [0.8, 0.9, 0.7, 0.6, 0.8]  # Good streak
        
        excitement = self.engine._calculate_excitement(
            subsystem_signals, portfolio_state, recent_performance
        )
        
        assert 0.0 <= excitement <= 1.0
        assert excitement > 0.4  # Should show excitement
    
    def test_emotional_state_update(self):
        trading_context = {
            'subsystem_signals': {
                'dna': 0.5,
                'temporal': 0.3,
                'immune': -0.2,
                'microstructure': 0.7,
                'dopamine': 0.1
            },
            'portfolio_state': {
                'unrealized_pnl': 100.0,
                'daily_pnl': 50.0
            },
            'market_context': {
                'volatility': 0.025,
                'regime_confidence': 0.8
            },
            'recent_performance': [0.3, 0.7, 0.1],
            'system_confidence': 0.6
        }
        
        emotional_state = self.engine.update_emotional_state(trading_context)
        
        assert isinstance(emotional_state, EmotionalMetrics)
        assert isinstance(emotional_state.primary_emotion, EmotionalState)
        assert 0.0 <= emotional_state.confidence <= 1.0
        assert 0.0 <= emotional_state.emotional_intensity <= 1.0

class TestLLMClient:
    """Test the LLM client"""
    
    def setup_method(self):
        self.client = LLMClient()
    
    def test_initialization(self):
        assert self.client.personality_name == "Alex"
        assert self.client.temperature == 0.7
        assert len(self.client.conversation_memory) == 0
    
    @pytest.mark.asyncio
    async def test_commentary_generation(self):
        request = CommentaryRequest(
            trigger_event="position_entry",
            market_context={'volatility': 0.03, 'trend_strength': 0.05},
            emotional_context={'primary_emotion': 'confident', 'confidence_level': 0.8},
            subsystem_context={'subsystem_signals': {'dna': 0.7, 'temporal': 0.5}},
            portfolio_context={'unrealized_pnl': 100.0, 'daily_pnl': 50.0}
        )
        
        response = await self.client.generate_commentary(request)
        
        assert isinstance(response, CommentaryResponse)
        assert len(response.text) > 0
        assert 0.0 <= response.confidence <= 1.0
        assert 0.0 <= response.emotional_intensity <= 1.0
        assert isinstance(response.key_themes, list)
    
    def test_prompt_building(self):
        request = CommentaryRequest(
            trigger_event="immune_warning",
            market_context={'volatility': 0.06},
            emotional_context={'primary_emotion': 'fearful', 'fear_level': 0.8},
            subsystem_context={'subsystem_signals': {'immune': -0.8}},
            portfolio_context={'unrealized_pnl': -200.0}
        )
        
        prompt = self.client._build_commentary_prompt(request)
        
        assert "Alex" in prompt
        assert "immune_warning" in prompt
        assert "fearful" in prompt
        assert len(prompt) > 100

class TestPersonalityMemory:
    """Test the personality memory system"""
    
    def setup_method(self):
        # Use temporary file for testing
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_file.close()
        self.memory = PersonalityMemory(memory_file=self.temp_file.name)
    
    def teardown_method(self):
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_initialization(self):
        assert len(self.memory.short_term_memory) == 0
        assert len(self.memory.long_term_memory) == 0
        assert isinstance(self.memory.personality_traits.base_confidence, float)
    
    def test_add_memory(self):
        memory_id = self.memory.add_memory(
            event_type="position_entry",
            emotional_state="confident",
            market_context={'volatility': 0.02},
            decision_context={'action': 'buy', 'confidence': 0.8},
            commentary="Taking a long position here",
            confidence=0.8,
            key_themes=['bullish_bias', 'confidence']
        )
        
        assert memory_id is not None
        assert len(self.memory.short_term_memory) == 1
        assert len(self.memory.session_memory) == 1
        
        # Verify memory content
        memory_entry = self.memory.short_term_memory[0]
        assert memory_entry.event_type == "position_entry"
        assert memory_entry.emotional_state == "confident"
        assert "bullish_bias" in memory_entry.key_themes
    
    def test_memory_outcome_update(self):
        memory_id = self.memory.add_memory(
            event_type="position_entry",
            emotional_state="confident",
            market_context={},
            decision_context={},
            commentary="Test trade"
        )
        
        success = self.memory.update_memory_outcome(memory_id, 150.0)
        assert success
        
        # Verify outcome was recorded
        memory_entry = self.memory.short_term_memory[0]
        assert memory_entry.outcome == 150.0
    
    def test_personality_context(self):
        # Add some memories first
        for i in range(3):
            self.memory.add_memory(
                event_type=f"test_event_{i}",
                emotional_state="analytical",
                market_context={'volatility': 0.02 + i * 0.01},
                decision_context={},
                commentary=f"Test commentary {i}"
            )
        
        context = self.memory.get_personality_context()
        
        assert 'personality_traits' in context
        assert 'recent_emotional_pattern' in context
        assert 'session_context' in context
    
    def test_save_load_memory(self):
        # Add some test data
        self.memory.add_memory(
            event_type="test_save",
            emotional_state="optimistic",
            market_context={'volatility': 0.03},
            decision_context={'confidence': 0.9},
            commentary="Save test commentary"
        )
        
        # Save to file
        self.memory.save_memory()
        assert os.path.exists(self.temp_file.name)
        
        # Create new memory instance and load
        new_memory = PersonalityMemory(memory_file=self.temp_file.name)
        
        # Verify data was loaded
        assert new_memory.emotional_consistency_score == self.memory.emotional_consistency_score
        assert new_memory.last_emotional_state == self.memory.last_emotional_state

class TestTradingPersonality:
    """Test the main trading personality class"""
    
    def setup_method(self):
        config = PersonalityConfig(personality_name="TestBot")
        self.personality = TradingPersonality(config)
    
    @pytest.mark.asyncio
    async def test_trading_event_processing(self):
        context = {
            'subsystem_signals': {
                'dna': 0.7,
                'temporal': 0.5,
                'immune': -0.1,
                'microstructure': 0.6,
                'dopamine': 0.3
            },
            'market_data': {
                'volatility': 0.025,
                'price_momentum': 0.02,
                'regime_confidence': 0.8
            },
            'portfolio_state': {
                'unrealized_pnl': 250.0,
                'daily_pnl': 100.0,
                'open_positions': 1
            },
            'decision_context': {
                'action': 'buy',
                'confidence': 0.8,
                'primary_tool': 'dna'
            }
        }
        
        response = await self.personality.process_trading_event(
            TriggerEvent.POSITION_ENTRY, context
        )
        
        assert response is not None
        assert isinstance(response, CommentaryResponse)
        assert len(response.text) > 0
    
    @pytest.mark.asyncio
    async def test_manual_query(self):
        query = "How are you feeling about the current market?"
        
        response = await self.personality.manual_query(query)
        
        assert response is not None
        assert isinstance(response, CommentaryResponse)
        assert len(response.text) > 0
    
    def test_emotional_state_tracking(self):
        emotional_state = self.personality.get_current_emotional_state()
        
        assert isinstance(emotional_state, EmotionalMetrics)
        assert isinstance(emotional_state.primary_emotion, EmotionalState)
    
    def test_personality_summary(self):
        summary = self.personality.get_personality_summary()
        
        assert 'personality_name' in summary
        assert 'current_emotional_state' in summary
        assert 'emotional_metrics' in summary
        assert summary['personality_name'] == "TestBot"

class TestPersonalityIntegration:
    """Test the personality integration layer"""
    
    def setup_method(self):
        config = PersonalityIntegrationConfig(
            personality_name="TestIntegration",
            voice_enabled=False,
            auto_commentary=True
        )
        self.integration = PersonalityIntegration(config)
    
    @pytest.mark.asyncio
    async def test_trading_decision_processing(self):
        decision = MockDecision()
        features = Features(
            price_momentum=0.02,
            volume_momentum=0.1,
            price_position=0.6,
            volatility=0.025,
            time_of_day=0.5,
            pattern_score=0.7,
            confidence=0.8,
            dna_signal=0.7,
            micro_signal=0.3,
            temporal_signal=0.5,
            immune_signal=-0.1,
            microstructure_signal=0.6,
            dopamine_signal=0.2,
            overall_signal=0.4
        )
        market_data = MockMarketData()
        
        commentary = await self.integration.process_trading_decision(
            decision, features, market_data
        )
        
        if self.integration.is_enabled():
            assert commentary is not None
            assert len(commentary) > 0
    
    @pytest.mark.asyncio
    async def test_trade_completion_processing(self):
        trade_outcome = 150.0
        trade_context = {
            'entry_price': 100.0,
            'exit_price': 101.5,
            'exit_reason': 'profit_target',
            'duration': 300,
            'timestamp': 1000000000
        }
        
        commentary = await self.integration.process_trade_completion(
            trade_outcome, trade_context
        )
        
        if self.integration.is_enabled():
            assert commentary is not None
            assert len(commentary) > 0
    
    def test_personality_status(self):
        status = self.integration.get_personality_status()
        
        assert 'enabled' in status
        if status['enabled']:
            assert 'personality_name' in status
            assert 'current_emotion' in status
            assert 'emotional_metrics' in status

class TestConfigManager:
    """Test the configuration manager"""
    
    def setup_method(self):
        # Use temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(
            mode='w', delete=False, suffix='.json'
        )
        
        # Write test config
        test_config = {
            "personality": {
                "enabled": True,
                "personality_name": "TestConfig",
                "base_confidence": 0.7
            },
            "llm": {
                "model_name": "test-model",
                "temperature": 0.8
            },
            "voice": {
                "enabled": False
            },
            "personalities": {
                "test_personality": {
                    "name": "Test",
                    "traits": ["test_trait"]
                }
            }
        }
        
        json.dump(test_config, self.temp_config)
        self.temp_config.close()
        
        self.config_manager = PersonalityConfigManager(self.temp_config.name)
    
    def teardown_method(self):
        if os.path.exists(self.temp_config.name):
            os.unlink(self.temp_config.name)
    
    def test_config_loading(self):
        assert self.config_manager.config is not None
        assert self.config_manager.config.personality.personality_name == "TestConfig"
        assert self.config_manager.config.llm_config['model_name'] == "test-model"
    
    def test_personality_config_retrieval(self):
        personality_config = self.config_manager.get_personality_config("test_personality")
        
        assert isinstance(personality_config, PersonalityConfig)
        assert personality_config.personality_name == "Test"
    
    def test_available_personalities(self):
        personalities = self.config_manager.get_available_personalities()
        
        assert isinstance(personalities, dict)
        assert "test_personality" in personalities
    
    def test_config_validation(self):
        is_valid, errors = self.config_manager.validate_config()
        
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)
    
    def test_config_summary(self):
        summary = self.config_manager.get_config_summary()
        
        assert 'status' in summary
        assert 'personality_name' in summary
        assert 'available_personalities' in summary

class TestVoiceSynthesis:
    """Test the voice synthesis system"""
    
    def setup_method(self):
        config = {
            'tts_service': 'mock',
            'voice_id': 'test_voice'
        }
        self.synthesizer = VoiceSynthesizer(config)
    
    @pytest.mark.asyncio
    async def test_speech_synthesis(self):
        text = "This is a test of the voice synthesis system."
        emotional_context = {
            'primary_emotion': 'confident',
            'emotional_intensity': 0.7
        }
        
        # This will use the mock TTS client
        audio_path = await self.synthesizer.synthesize_speech(text, emotional_context)
        
        # Mock should return a path
        assert audio_path is not None
    
    @pytest.mark.asyncio
    async def test_commentary_speaking(self):
        commentary = "I'm feeling confident about this trade setup."
        
        success = await self.synthesizer.speak_commentary(
            commentary, "confident", 0.8
        )
        
        # Mock should succeed
        assert success == True
    
    def test_voice_settings_adjustment(self):
        emotional_context = {
            'primary_emotion': 'excited',
            'emotional_intensity': 0.9
        }
        
        adjusted_settings = self.synthesizer._adjust_voice_for_emotion(emotional_context)
        
        assert adjusted_settings.speed > self.synthesizer.voice_settings.speed
        assert adjusted_settings.emotion_intensity == 0.9
    
    def test_text_preparation(self):
        text = "The P&L is looking good! DNA signals are strong."
        
        clean_text = self.synthesizer._prepare_text_for_synthesis(text)
        
        assert "profit and loss" in clean_text
        assert "D.N.A." in clean_text

# Integration Tests
class TestSystemIntegration:
    """Test full system integration"""
    
    @pytest.mark.asyncio
    async def test_full_personality_pipeline(self):
        """Test complete pipeline from decision to commentary"""
        
        # Setup
        config = PersonalityIntegrationConfig(
            personality_name="IntegrationTest",
            voice_enabled=False
        )
        integration = PersonalityIntegration(config)
        
        if not integration.is_enabled():
            pytest.skip("Personality system disabled")
        
        # Create mock trading decision
        decision = MockDecision(action="buy", confidence=0.8)
        
        features = Features(
            price_momentum=0.03,
            volume_momentum=0.15,
            price_position=0.7,
            volatility=0.02,
            time_of_day=0.4,
            pattern_score=0.8,
            confidence=0.8,
            dna_signal=0.8,
            micro_signal=0.4,
            temporal_signal=0.6,
            immune_signal=0.1,
            microstructure_signal=0.7,
            dopamine_signal=0.5,
            overall_signal=0.6
        )
        
        market_data = MockMarketData(
            unrealized_pnl=200.0,
            daily_pnl=350.0,
            open_positions=1
        )
        
        # Process through personality system
        commentary = await integration.process_trading_decision(
            decision, features, market_data
        )
        
        assert commentary is not None
        assert len(commentary) > 10
        
        # Simulate trade completion
        trade_outcome = 180.0
        trade_context = {
            'exit_reason': 'profit_target',
            'timestamp': 1000000000
        }
        
        completion_commentary = await integration.process_trade_completion(
            trade_outcome, trade_context
        )
        
        assert completion_commentary is not None
        
        # Check personality status
        status = integration.get_personality_status()
        assert status['enabled'] == True
        assert 'current_emotion' in status

# Performance Tests
class TestPerformance:
    """Test system performance under load"""
    
    @pytest.mark.asyncio
    async def test_concurrent_commentary_generation(self):
        """Test multiple concurrent commentary requests"""
        
        config = PersonalityConfig(personality_name="PerformanceTest")
        personality = TradingPersonality(config)
        
        # Create multiple contexts
        contexts = []
        for i in range(5):
            context = {
                'subsystem_signals': {
                    'dna': 0.5 + i * 0.1,
                    'temporal': 0.3,
                    'immune': 0.0,
                    'microstructure': 0.4,
                    'dopamine': 0.2
                },
                'market_data': {'volatility': 0.02 + i * 0.005},
                'portfolio_state': {'unrealized_pnl': i * 50},
                'decision_context': {'confidence': 0.7}
            }
            contexts.append(context)
        
        # Process concurrently
        tasks = [
            personality.process_trading_event(TriggerEvent.PERIODIC_UPDATE, context)
            for context in contexts
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check results
        successful_responses = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_responses) >= 3  # At least 60% success rate

if __name__ == "__main__":
    # Run basic tests
    print("Running AI Trading Personality System Tests...")
    
    # Run a simple test
    async def run_basic_test():
        engine = EmotionalStateEngine()
        
        test_context = {
            'subsystem_signals': {'dna': 0.7, 'dopamine': 0.3},
            'portfolio_state': {'unrealized_pnl': 100.0},
            'market_context': {'volatility': 0.03},
            'recent_performance': [0.5, 0.8, 0.2],
            'system_confidence': 0.7
        }
        
        emotional_state = engine.update_emotional_state(test_context)
        print(f"Emotional State: {emotional_state.primary_emotion.value}")
        print(f"Confidence: {emotional_state.confidence:.2f}")
        print(f"Intensity: {emotional_state.emotional_intensity:.2f}")
        
        # Test LLM client
        client = LLMClient()
        request = CommentaryRequest(
            trigger_event="test",
            market_context=test_context['market_context'],
            emotional_context={'primary_emotion': emotional_state.primary_emotion.value},
            subsystem_context=test_context['subsystem_signals'],
            portfolio_context=test_context['portfolio_state']
        )
        
        response = await client.generate_commentary(request)
        print(f"Commentary: {response.text}")
    
    # Run the test
    asyncio.run(run_basic_test())
    print("Basic tests completed successfully!")