"""
Voice Synthesis for AI Trading Personality

Converts text commentary to natural speech for conversational trading experience
"""

import asyncio
import logging
import time
from typing import Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class VoiceStyle(Enum):
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    EXCITED = "excited"
    CAUTIOUS = "cautious"
    ANALYTICAL = "analytical"

@dataclass
class VoiceSettings:
    voice_id: str = "default"
    speed: float = 1.0
    pitch: float = 1.0
    volume: float = 0.8
    style: VoiceStyle = VoiceStyle.PROFESSIONAL
    emotion_intensity: float = 0.5

class VoiceSynthesizer:
    """
    Voice synthesis system for AI trading personality
    
    This is a framework for voice synthesis integration.
    Can be connected to various TTS services like:
    - ElevenLabs
    - Azure Cognitive Services
    - Google Cloud Text-to-Speech
    - Amazon Polly
    - Local TTS engines
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Voice configuration
        self.voice_settings = VoiceSettings(
            voice_id=self.config.get('voice_id', 'alex_trader'),
            speed=self.config.get('speed', 1.0),
            pitch=self.config.get('pitch', 1.0),
            volume=self.config.get('volume', 0.8)
        )
        
        # TTS Service configuration
        self.tts_service = self.config.get('tts_service', 'elevenlabs')
        self.api_key = self.config.get('api_key', '')
        self.base_url = self.config.get('base_url', '')
        
        # Audio settings
        self.audio_format = self.config.get('audio_format', 'mp3')
        self.sample_rate = self.config.get('sample_rate', 44100)
        
        # Synthesis tracking
        self.synthesis_queue = asyncio.Queue(maxsize=10)
        self.is_speaking = False
        self.last_synthesis_time = 0.0
        
        # Callbacks
        self.audio_ready_callbacks: list = []
        self.speaking_state_callbacks: list = []
        
        # Initialize TTS client
        self.tts_client = self._initialize_tts_client()
        
    async def synthesize_speech(self, text: str, emotional_context: Dict = None) -> Optional[str]:
        """
        Convert text to speech with emotional context
        
        Args:
            text: Text to synthesize
            emotional_context: Emotional state for voice modulation
            
        Returns:
            str: Path to generated audio file or None if failed
        """
        
        try:
            # Adjust voice settings based on emotional context
            voice_settings = self._adjust_voice_for_emotion(emotional_context)
            
            # Clean and prepare text
            clean_text = self._prepare_text_for_synthesis(text)
            
            # Add to synthesis queue
            synthesis_request = {
                'text': clean_text,
                'settings': voice_settings,
                'timestamp': time.time()
            }
            
            await self.synthesis_queue.put(synthesis_request)
            
            # Process synthesis
            audio_path = await self._process_synthesis(synthesis_request)
            
            # Notify callbacks
            await self._notify_audio_ready(audio_path, text)
            
            return audio_path
            
        except Exception as e:
            logger.error(f"Speech synthesis error: {e}")
            return None
    
    async def speak_commentary(self, commentary_text: str, emotional_state: str = "analytical",
                             emotional_intensity: float = 0.5) -> bool:
        """
        Speak trading commentary with appropriate emotional tone
        
        Args:
            commentary_text: Trading commentary to speak
            emotional_state: Current emotional state
            emotional_intensity: Intensity of emotion (0-1)
            
        Returns:
            bool: Success status
        """
        
        if self.is_speaking:
            logger.warning("Already speaking, queueing new commentary")
        
        emotional_context = {
            'primary_emotion': emotional_state,
            'emotional_intensity': emotional_intensity,
            'trading_context': True
        }
        
        try:
            self.is_speaking = True
            await self._notify_speaking_state(True)
            
            audio_path = await self.synthesize_speech(commentary_text, emotional_context)
            
            if audio_path:
                # Play audio (implementation depends on platform)
                await self._play_audio(audio_path)
                logger.info(f"Spoke commentary: {commentary_text[:50]}...")
                return True
            else:
                logger.error("Failed to synthesize speech")
                return False
                
        except Exception as e:
            logger.error(f"Error speaking commentary: {e}")
            return False
        finally:
            self.is_speaking = False
            await self._notify_speaking_state(False)
    
    def _initialize_tts_client(self):
        """Initialize TTS client based on configured service"""
        
        if self.tts_service == 'elevenlabs':
            return self._init_elevenlabs_client()
        elif self.tts_service == 'azure':
            return self._init_azure_client()
        elif self.tts_service == 'google':
            return self._init_google_client()
        elif self.tts_service == 'amazon':
            return self._init_amazon_client()
        else:
            return self._init_local_client()
    
    def _init_elevenlabs_client(self):
        """Initialize ElevenLabs TTS client"""
        try:
            # Placeholder for ElevenLabs integration
            # from elevenlabs import ElevenLabs
            # return ElevenLabs(api_key=self.api_key)
            
            logger.info("ElevenLabs TTS client initialized (placeholder)")
            return MockTTSClient("ElevenLabs")
        except ImportError:
            logger.warning("ElevenLabs package not available, using mock client")
            return MockTTSClient("ElevenLabs")
    
    def _init_azure_client(self):
        """Initialize Azure Cognitive Services TTS client"""
        try:
            # Placeholder for Azure integration
            # import azure.cognitiveservices.speech as speechsdk
            # speech_config = speechsdk.SpeechConfig(subscription=self.api_key, region="eastus")
            # return speechsdk.SpeechSynthesizer(speech_config=speech_config)
            
            logger.info("Azure TTS client initialized (placeholder)")
            return MockTTSClient("Azure")
        except ImportError:
            logger.warning("Azure Speech package not available, using mock client")
            return MockTTSClient("Azure")
    
    def _init_google_client(self):
        """Initialize Google Cloud TTS client"""
        try:
            # Placeholder for Google Cloud integration
            # from google.cloud import texttospeech
            # return texttospeech.TextToSpeechClient()
            
            logger.info("Google Cloud TTS client initialized (placeholder)")
            return MockTTSClient("Google")
        except ImportError:
            logger.warning("Google Cloud TTS package not available, using mock client")
            return MockTTSClient("Google")
    
    def _init_amazon_client(self):
        """Initialize Amazon Polly TTS client"""
        try:
            # Placeholder for Amazon Polly integration
            # import boto3
            # return boto3.client('polly', region_name='us-east-1')
            
            logger.info("Amazon Polly TTS client initialized (placeholder)")
            return MockTTSClient("Amazon")
        except ImportError:
            logger.warning("Amazon Polly package not available, using mock client")
            return MockTTSClient("Amazon")
    
    def _init_local_client(self):
        """Initialize local TTS client"""
        try:
            # Try to use pyttsx3 for real local TTS
            import pyttsx3
            engine = pyttsx3.init()
            
            # Configure voice settings
            voices = engine.getProperty('voices')
            if voices:
                # Try to find a good voice
                for voice in voices:
                    if 'english' in voice.name.lower() or 'zira' in voice.name.lower():
                        engine.setProperty('voice', voice.id)
                        break
            
            # Set speaking rate and volume
            engine.setProperty('rate', 200)  # Speed of speech
            engine.setProperty('volume', 0.8)  # Volume level (0.0 to 1.0)
            
            logger.info("Local TTS (pyttsx3) client initialized successfully")
            return LocalTTSClient(engine)
            
        except ImportError:
            logger.warning("pyttsx3 not available. Install with: pip install pyttsx3")
            logger.info("Using mock TTS client")
            return MockTTSClient("Local")
        except Exception as e:
            logger.error(f"Error initializing local TTS: {e}")
            return MockTTSClient("Local")
    
    def _adjust_voice_for_emotion(self, emotional_context: Dict = None) -> VoiceSettings:
        """Adjust voice settings based on emotional context"""
        
        if not emotional_context:
            return self.voice_settings
        
        # Copy base settings
        adjusted_settings = VoiceSettings(
            voice_id=self.voice_settings.voice_id,
            speed=self.voice_settings.speed,
            pitch=self.voice_settings.pitch,
            volume=self.voice_settings.volume,
            style=self.voice_settings.style
        )
        
        # Adjust based on emotion
        emotion = emotional_context.get('primary_emotion', 'analytical')
        intensity = emotional_context.get('emotional_intensity', 0.5)
        
        if emotion == 'excited':
            adjusted_settings.speed = min(1.3, self.voice_settings.speed + 0.2)
            adjusted_settings.pitch = min(1.2, self.voice_settings.pitch + 0.1)
            adjusted_settings.style = VoiceStyle.EXCITED
            
        elif emotion == 'fearful' or emotion == 'cautious':
            adjusted_settings.speed = max(0.8, self.voice_settings.speed - 0.1)
            adjusted_settings.pitch = max(0.9, self.voice_settings.pitch - 0.05)
            adjusted_settings.style = VoiceStyle.CAUTIOUS
            
        elif emotion == 'confident':
            adjusted_settings.speed = min(1.1, self.voice_settings.speed + 0.05)
            adjusted_settings.volume = min(1.0, self.voice_settings.volume + 0.1)
            adjusted_settings.style = VoiceStyle.PROFESSIONAL
            
        elif emotion == 'confused':
            adjusted_settings.speed = max(0.9, self.voice_settings.speed - 0.05)
            adjusted_settings.style = VoiceStyle.ANALYTICAL
            
        else:  # analytical, optimistic, etc.
            adjusted_settings.style = VoiceStyle.ANALYTICAL
        
        # Apply intensity scaling
        if intensity > 0.7:
            # High intensity - more pronounced changes
            speed_change = (adjusted_settings.speed - self.voice_settings.speed) * 1.5
            pitch_change = (adjusted_settings.pitch - self.voice_settings.pitch) * 1.5
            
            adjusted_settings.speed = max(0.7, min(1.5, self.voice_settings.speed + speed_change))
            adjusted_settings.pitch = max(0.8, min(1.3, self.voice_settings.pitch + pitch_change))
        
        adjusted_settings.emotion_intensity = intensity
        
        return adjusted_settings
    
    def _prepare_text_for_synthesis(self, text: str) -> str:
        """Clean and prepare text for speech synthesis"""
        
        # Remove markdown formatting
        clean_text = text.replace('**', '').replace('*', '')
        
        # Handle trading-specific terms for better pronunciation
        trading_replacements = {
            'P&L': 'profit and loss',
            'PnL': 'profit and loss',
            'pnl': 'profit and loss',
            'DNA': 'D.N.A.',
            'AI': 'A.I.',
            'VIX': 'V.I.X.',
            'SPY': 'S.P.Y.',
            'QQQ': 'Q.Q.Q.',
            'NYSE': 'New York Stock Exchange',
            'NASDAQ': 'NASDAQ',
            'S&P': 'S and P',
            'volatility': 'volatility',
            'cryptocurrency': 'crypto currency'
        }
        
        for term, replacement in trading_replacements.items():
            clean_text = clean_text.replace(term, replacement)
        
        # Add pauses for better flow
        clean_text = clean_text.replace('. ', '. <break time="0.3s"/> ')
        clean_text = clean_text.replace('! ', '! <break time="0.4s"/> ')
        clean_text = clean_text.replace('? ', '? <break time="0.4s"/> ')
        clean_text = clean_text.replace(', ', ', <break time="0.2s"/> ')
        
        # Ensure reasonable length
        if len(clean_text) > 500:
            clean_text = clean_text[:497] + '...'
        
        return clean_text
    
    async def _process_synthesis(self, synthesis_request: Dict) -> Optional[str]:
        """Process speech synthesis request"""
        
        try:
            text = synthesis_request['text']
            settings = synthesis_request['settings']
            
            # Generate unique filename
            timestamp = int(time.time() * 1000)
            audio_filename = f"trading_voice_{timestamp}.{self.audio_format}"
            audio_path = f"data/audio/{audio_filename}"
            
            # Call TTS service
            success = await self._call_tts_service(text, settings, audio_path)
            
            if success:
                self.last_synthesis_time = time.time()
                return audio_path
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error processing synthesis: {e}")
            return None
    
    async def _call_tts_service(self, text: str, settings: VoiceSettings, output_path: str) -> bool:
        """Call the configured TTS service"""
        
        try:
            # This would contain the actual TTS service integration
            # For now, using mock implementation
            
            if hasattr(self.tts_client, 'synthesize'):
                result = await self.tts_client.synthesize(text, settings, output_path)
                return result
            else:
                # Mock success for development
                logger.info(f"Mock TTS synthesis: {text[:50]}...")
                return True
                
        except Exception as e:
            logger.error(f"TTS service call failed: {e}")
            return False
    
    async def _play_audio(self, audio_path: str):
        """Play synthesized audio"""
        
        try:
            # Platform-specific audio playback
            # This would integrate with system audio or web audio APIs
            
            # For development, just log the action
            logger.info(f"Playing audio: {audio_path}")
            
            # Simulate audio playback time
            await asyncio.sleep(2.0)
            
        except Exception as e:
            logger.error(f"Audio playback error: {e}")
    
    async def _notify_audio_ready(self, audio_path: str, original_text: str):
        """Notify callbacks that audio is ready"""
        
        for callback in self.audio_ready_callbacks:
            try:
                await callback(audio_path, original_text)
            except Exception as e:
                logger.error(f"Error in audio ready callback: {e}")
    
    async def _notify_speaking_state(self, is_speaking: bool):
        """Notify callbacks of speaking state change"""
        
        for callback in self.speaking_state_callbacks:
            try:
                await callback(is_speaking)
            except Exception as e:
                logger.error(f"Error in speaking state callback: {e}")
    
    def add_audio_ready_callback(self, callback: Callable):
        """Add callback for when audio synthesis is complete"""
        self.audio_ready_callbacks.append(callback)
    
    def add_speaking_state_callback(self, callback: Callable):
        """Add callback for speaking state changes"""
        self.speaking_state_callbacks.append(callback)
    
    def is_available(self) -> bool:
        """Check if voice synthesis is available"""
        return self.tts_client is not None
    
    def get_voice_info(self) -> Dict:
        """Get current voice configuration info"""
        return {
            'tts_service': self.tts_service,
            'voice_id': self.voice_settings.voice_id,
            'is_available': self.is_available(),
            'is_speaking': self.is_speaking,
            'settings': {
                'speed': self.voice_settings.speed,
                'pitch': self.voice_settings.pitch,
                'volume': self.voice_settings.volume,
                'style': self.voice_settings.style.value
            }
        }

class LocalTTSClient:
    """Real local TTS client using pyttsx3"""
    
    def __init__(self, engine):
        self.engine = engine
        self.service_name = "Local TTS"
    
    async def synthesize(self, text: str, settings: VoiceSettings, output_path: str) -> bool:
        """Real local TTS synthesis"""
        
        try:
            # Adjust voice settings based on emotion
            rate = int(200 * settings.speed)
            volume = settings.volume
            
            self.engine.setProperty('rate', rate)
            self.engine.setProperty('volume', volume)
            
            logger.info(f"Local TTS synthesizing: '{text[:50]}...' "
                       f"(speed: {settings.speed:.1f}, volume: {settings.volume:.1f})")
            
            # For real-time speaking (no file output needed for local TTS)
            await asyncio.get_event_loop().run_in_executor(
                None, self.engine.say, text
            )
            await asyncio.get_event_loop().run_in_executor(
                None, self.engine.runAndWait
            )
            
            # Create a marker file to indicate success
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(f"Local TTS spoken: {text}")
            
            return True
            
        except Exception as e:
            logger.error(f"Local TTS synthesis error: {e}")
            return False

class MockTTSClient:
    """Mock TTS client for development and testing"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
    
    async def synthesize(self, text: str, settings: VoiceSettings, output_path: str) -> bool:
        """Mock synthesis that just logs the request"""
        
        logger.info(f"Mock {self.service_name} TTS: '{text[:50]}...' "
                   f"(style: {settings.style.value}, intensity: {settings.emotion_intensity:.2f})")
        
        # Simulate processing time
        await asyncio.sleep(0.5)
        
        # Create mock audio file (empty file for testing)
        try:
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(f"Mock audio for: {text[:100]}")
            return True
        except Exception as e:
            logger.error(f"Error creating mock audio file: {e}")
            return False