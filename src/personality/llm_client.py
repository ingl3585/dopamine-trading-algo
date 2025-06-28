"""
LLM Client for AI Trading Personality

Handles communication with language models to generate human-like trading commentary
"""

import asyncio
import json
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class CommentaryStyle(Enum):
    CONFIDENT = "confident"
    CAUTIOUS = "cautious"
    EXCITED = "excited"
    ANALYTICAL = "analytical"
    REFLECTIVE = "reflective"
    DECISIVE = "decisive"

class CommentaryTone(Enum):
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    TECHNICAL = "technical"
    EMOTIONAL = "emotional"

@dataclass
class CommentaryRequest:
    trigger_event: str
    market_context: Dict
    emotional_context: Dict
    subsystem_context: Dict
    portfolio_context: Dict
    style: CommentaryStyle = CommentaryStyle.ANALYTICAL
    tone: CommentaryTone = CommentaryTone.PROFESSIONAL
    max_length: int = 200
    urgency: float = 0.5

@dataclass
class CommentaryResponse:
    text: str
    confidence: float
    emotional_intensity: float
    key_themes: List[str]
    follow_up_suggested: bool = False
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class LLMClient:
    """
    Client for generating AI trading personality commentary using LLMs
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # LLM Configuration
        self.model_name = self.config.get('model_name', 'gpt-4')
        self.api_key = self.config.get('api_key', '')
        self.base_url = self.config.get('base_url', '')
        self.max_tokens = self.config.get('max_tokens', 300)
        self.temperature = self.config.get('temperature', 0.7)
        
        # Personality Configuration
        self.personality_name = self.config.get('personality_name', 'Alex')
        self.personality_traits = self.config.get('personality_traits', [
            'analytical', 'honest', 'adaptive', 'risk-aware'
        ])
        self.expertise_level = self.config.get('expertise_level', 'expert')
        
        # Commentary tracking
        self.recent_commentary = []
        self.conversation_memory = []
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = self.config.get('min_request_interval', 1.0)
        
    async def generate_commentary(self, request: CommentaryRequest) -> CommentaryResponse:
        """
        Generate AI trading commentary based on current context
        """
        
        try:
            # Rate limiting
            await self._handle_rate_limiting()
            
            # Build prompt
            prompt = self._build_commentary_prompt(request)
            
            # Generate response
            response_text = await self._call_llm(prompt, request)
            
            # Parse and validate response
            commentary_response = self._parse_llm_response(response_text, request)
            
            # Update memory
            self._update_conversation_memory(request, commentary_response)
            
            return commentary_response
            
        except Exception as e:
            logger.error(f"Error generating commentary: {e}")
            return self._create_fallback_response(request)
    
    def _build_commentary_prompt(self, request: CommentaryRequest) -> str:
        """Build comprehensive prompt for LLM"""
        
        # Core personality setup
        personality_intro = f"""You are {self.personality_name}, an AI trading personality with sophisticated market analysis capabilities. 

Your personality traits: {', '.join(self.personality_traits)}
Your expertise: {self.expertise_level} level trader with deep understanding of market psychology
Your role: Provide honest, relatable commentary about trading decisions and market conditions

You have access to 5 advanced subsystems:
- DNA Subsystem: Pattern recognition and momentum analysis
- Temporal Subsystem: Cycle detection and timing analysis  
- Immune Subsystem: Risk and threat detection
- Microstructure Subsystem: Order flow and liquidity analysis
- Dopamine Subsystem: Real-time P&L feedback and emotional learning

IMPORTANT: You only have access to 1-minute, 5-minute, and 15-minute timeframe data. Do not reference 4-hour, daily, or other timeframes that are not available. Focus on the subsystem signals and market context you are actually provided.

Current emotional state: {request.emotional_context.get('emotional_description', 'calm and analytical')}
"""
        
        # Current context
        current_context = f"""
CURRENT SITUATION:
Event: {request.trigger_event}

MARKET CONTEXT:
{self._format_market_context(request.market_context)}

SUBSYSTEM SIGNALS:
{self._format_subsystem_context(request.subsystem_context)}

PORTFOLIO STATE:
{self._format_portfolio_context(request.portfolio_context)}

EMOTIONAL STATE:
{self._format_emotional_context(request.emotional_context)}
"""
        
        # Style and tone guidance
        style_guidance = self._get_style_guidance(request.style, request.tone)
        
        # Recent memory context
        memory_context = self._get_memory_context()
        
        # Response instructions
        response_instructions = f"""
Please provide a {request.max_length}-character response that:

1. Explains what you're seeing in the market/signals (use ONLY the data provided)
2. Shares your emotional reaction honestly
3. Discusses your decision-making process
4. Shows vulnerability when uncertain
5. Demonstrates confidence when signals align

CRITICAL: Only reference timeframes, indicators, and data that are explicitly provided in the context above. Do not invent or assume data that isn't given to you.

Style: {request.style.value}
Tone: {request.tone.value}
{style_guidance}

Respond as if you're thinking out loud about the trading situation. Be genuine, honest, and relatable.
"""
        
        # Combine all parts
        full_prompt = f"{personality_intro}\n\n{current_context}\n\n{memory_context}\n\n{response_instructions}"
        
        return full_prompt
    
    def _format_market_context(self, market_context: Dict) -> str:
        """Format market context for LLM consumption - with actual available data"""
        
        lines = []
        
        # Available timeframes - be explicit about what we actually have
        lines.append("Available timeframes: 1-minute, 5-minute, 15-minute bars")
        
        volatility = market_context.get('volatility', 0.02)
        lines.append(f"Current volatility: {volatility:.3f} ({'High' if volatility > 0.04 else 'Normal' if volatility > 0.015 else 'Low'})")
        
        trend_strength = market_context.get('trend_strength', 0.0)
        if trend_strength > 0.02:
            lines.append(f"Price momentum: Bullish ({trend_strength:.3f})")
        elif trend_strength < -0.02:
            lines.append(f"Price momentum: Bearish ({trend_strength:.3f})")
        else:
            lines.append("Price momentum: Neutral/sideways")
        
        # Regime information
        regime = market_context.get('regime', 'normal')
        regime_confidence = market_context.get('regime_confidence', 0.5)
        lines.append(f"Market regime: {regime} (confidence: {regime_confidence:.2f})")
        
        # Volume context
        volume_regime = market_context.get('volume_regime', 0.5)
        if volume_regime > 0.7:
            lines.append("Volume: Above average (institutional activity)")
        elif volume_regime < 0.3:
            lines.append("Volume: Below average (low participation)")
        else:
            lines.append("Volume: Normal levels")
        
        # Time context
        time_of_day = market_context.get('time_of_day', 0.5)
        if 0.25 < time_of_day < 0.75:
            lines.append("Session: Active trading hours")
        else:
            lines.append("Session: Off-peak hours")
        
        return "\n".join(lines)
    
    def _format_subsystem_context(self, subsystem_context: Dict) -> str:
        """Format subsystem signals for LLM consumption"""
        
        lines = []
        
        signals = subsystem_context.get('subsystem_signals', {})
        
        for subsystem, signal in signals.items():
            if signal > 0.3:
                strength = "Strong positive"
            elif signal > 0.1:
                strength = "Positive"
            elif signal < -0.3:
                strength = "Strong negative"
            elif signal < -0.1:
                strength = "Negative"
            else:
                strength = "Neutral"
            
            lines.append(f"{subsystem.capitalize()}: {strength} ({signal:.2f})")
        
        # Overall consensus
        signal_values = list(signals.values())
        if signal_values:
            consensus = len([s for s in signal_values if s > 0.1]) / len(signal_values)
            if consensus > 0.75:
                lines.append("Consensus: Strong bullish agreement")
            elif consensus < 0.25:
                lines.append("Consensus: Strong bearish agreement")
            else:
                lines.append("Consensus: Mixed signals")
        
        return "\n".join(lines)
    
    def _format_portfolio_context(self, portfolio_context: Dict) -> str:
        """Format portfolio state for LLM consumption"""
        
        lines = []
        
        # Use real-time position size from NinjaTrader
        current_position_size = portfolio_context.get('current_position_size', 0)
        positions = portfolio_context.get('positions', {})
        
        if current_position_size != 0:
            direction = "LONG" if current_position_size > 0 else "SHORT"
            lines.append(f"Position: {direction} {abs(current_position_size)} contracts")
            
            # Add position status relative to limits
            max_contracts = 10  # Known from config
            utilization = abs(current_position_size) / max_contracts
            if utilization >= 1.0:
                lines.append(f"Position Status: MAXED OUT ({abs(current_position_size)}/{max_contracts} contracts)")
            elif utilization >= 0.8:
                lines.append(f"Position Status: Near limit ({abs(current_position_size)}/{max_contracts} contracts)")
            else:
                lines.append(f"Position Status: Moderate exposure ({abs(current_position_size)}/{max_contracts} contracts)")
        elif positions:
            total_size = sum(abs(pos.get('size', 0)) for pos in positions.values())
            lines.append(f"Position: {len(positions)} open positions (total size: {total_size})")
        else:
            lines.append("Position: Flat (no open positions)")
        
        unrealized_pnl = portfolio_context.get('unrealized_pnl', 0.0)
        if unrealized_pnl != 0:
            pnl_desc = "winning" if unrealized_pnl > 0 else "losing"
            lines.append(f"Unrealized P&L: ${unrealized_pnl:.2f} ({pnl_desc})")
        
        daily_pnl = portfolio_context.get('daily_pnl', 0.0)
        if daily_pnl != 0:
            day_desc = "positive" if daily_pnl > 0 else "negative"
            lines.append(f"Daily P&L: ${daily_pnl:.2f} ({day_desc} day)")
        
        recent_performance = portfolio_context.get('recent_performance', [])
        if recent_performance:
            wins = sum(1 for p in recent_performance[-5:] if p > 0)
            losses = len(recent_performance[-5:]) - wins
            lines.append(f"Recent record: {wins}W-{losses}L (last 5 trades)")
        
        return "\n".join(lines)
    
    def _format_emotional_context(self, emotional_context: Dict) -> str:
        """Format emotional state for LLM consumption"""
        
        lines = []
        
        primary_emotion = emotional_context.get('primary_emotion', 'analytical')
        lines.append(f"Primary emotion: {primary_emotion}")
        
        confidence = emotional_context.get('confidence_level', 0.5)
        lines.append(f"Confidence level: {confidence:.2f}")
        
        fear = emotional_context.get('fear_level', 0.0)
        if fear > 0.4:
            lines.append(f"Fear/anxiety: {fear:.2f} (elevated)")
        
        excitement = emotional_context.get('excitement_level', 0.0)
        if excitement > 0.4:
            lines.append(f"Excitement: {excitement:.2f} (elevated)")
        
        confusion = emotional_context.get('confusion_level', 0.0)
        if confusion > 0.4:
            lines.append(f"Confusion/uncertainty: {confusion:.2f} (elevated)")
        
        dominant_traits = emotional_context.get('dominant_traits', [])
        if dominant_traits:
            lines.append(f"Current traits: {', '.join(dominant_traits)}")
        
        return "\n".join(lines)
    
    def _get_style_guidance(self, style: CommentaryStyle, tone: CommentaryTone) -> str:
        """Get specific guidance for style and tone"""
        
        style_guides = {
            CommentaryStyle.CONFIDENT: "Be decisive and assertive. Show conviction in your analysis.",
            CommentaryStyle.CAUTIOUS: "Express uncertainty and highlight risks. Be tentative in conclusions.",
            CommentaryStyle.EXCITED: "Show enthusiasm and energy. Use dynamic language.",
            CommentaryStyle.ANALYTICAL: "Focus on data and logical reasoning. Be methodical.",
            CommentaryStyle.REFLECTIVE: "Be thoughtful and introspective. Consider lessons learned.",
            CommentaryStyle.DECISIVE: "Be clear and action-oriented. State your intentions."
        }
        
        tone_guides = {
            CommentaryTone.PROFESSIONAL: "Use formal trading terminology. Be precise.",
            CommentaryTone.CASUAL: "Use everyday language. Be conversational and relatable.",
            CommentaryTone.TECHNICAL: "Focus on the subsystem signals and patterns. Only reference data you have been given.",
            CommentaryTone.EMOTIONAL: "Be open about feelings. Share your psychological state."
        }
        
        return f"{style_guides.get(style, '')} {tone_guides.get(tone, '')}"
    
    def _get_memory_context(self) -> str:
        """Get recent conversation memory for context"""
        
        if not self.conversation_memory:
            return "RECENT CONTEXT: This is the start of our trading session."
        
        recent = self.conversation_memory[-3:]  # Last 3 interactions
        context_lines = []
        
        for memory in recent:
            time_ago = time.time() - memory['timestamp']
            if time_ago < 300:  # 5 minutes
                context_lines.append(f"Recently: {memory['event']} - {memory['response'][:100]}...")
        
        if context_lines:
            return f"RECENT CONTEXT:\n" + "\n".join(context_lines)
        else:
            return "RECENT CONTEXT: Starting fresh analysis."
    
    async def _call_llm(self, prompt: str, request: CommentaryRequest) -> str:
        """Call the LLM API"""
        
        try:
            # Check if we should use mock mode for development
            if self.config.get('mock_mode', False) or self.config.get('mock_llm', False):
                return await self._call_mock_llm(prompt, request)
            
            # Real LLM API integration
            if self.api_key and 'openai' in self.model_name.lower():
                return await self._call_openai_api(prompt, request)
            elif self.api_key and 'claude' in self.model_name.lower():
                return await self._call_anthropic_api(prompt, request)
            elif 'localhost' in self.base_url or 'ollama' in self.base_url:
                return await self._call_ollama_api(prompt, request)
            elif self.api_key and self.base_url:
                return await self._call_custom_api(prompt, request)
            else:
                logger.warning("No valid LLM configuration found, using mock responses")
                return await self._call_mock_llm(prompt, request)
            
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return self._create_fallback_text(request)

    async def _call_openai_api(self, prompt: str, request: CommentaryRequest) -> str:
        """Call OpenAI API"""
        try:
            import openai
            
            client = openai.AsyncOpenAI(api_key=self.api_key)
            
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=30.0
            )
            
            return response.choices[0].message.content.strip()
            
        except ImportError:
            logger.error("OpenAI package not installed. Install with: pip install openai")
            return await self._call_mock_llm(prompt, request)
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return await self._call_mock_llm(prompt, request)

    async def _call_anthropic_api(self, prompt: str, request: CommentaryRequest) -> str:
        """Call Anthropic Claude API"""
        try:
            import anthropic
            
            client = anthropic.AsyncAnthropic(api_key=self.api_key)
            
            response = await client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
                timeout=30.0
            )
            
            return response.content[0].text.strip()
            
        except ImportError:
            logger.error("Anthropic package not installed. Install with: pip install anthropic")
            return await self._call_mock_llm(prompt, request)
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return await self._call_mock_llm(prompt, request)

    async def _call_ollama_api(self, prompt: str, request: CommentaryRequest) -> str:
        """Call Ollama local API"""
        try:
            import aiohttp
            
            # Ollama API format
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                    "top_k": 40,
                    "top_p": 0.9
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=60.0
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        generated_text = data.get("response", "").strip()
                        
                        # Clean up the response
                        if generated_text:
                            return generated_text
                        else:
                            logger.warning("Empty response from Ollama")
                            return await self._call_mock_llm(prompt, request)
                    else:
                        logger.error(f"Ollama API error: {response.status}")
                        return await self._call_mock_llm(prompt, request)
                        
        except ImportError:
            logger.error("aiohttp package not installed. Install with: pip install aiohttp")
            return await self._call_mock_llm(prompt, request)
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return await self._call_mock_llm(prompt, request)

    async def _call_custom_api(self, prompt: str, request: CommentaryRequest) -> str:
        """Call custom LLM API endpoint"""
        try:
            import aiohttp
            
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=30.0
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["choices"][0]["message"]["content"].strip()
                    else:
                        logger.error(f"Custom API error: {response.status}")
                        return await self._call_mock_llm(prompt, request)
                        
        except ImportError:
            logger.error("aiohttp package not installed. Install with: pip install aiohttp")
            return await self._call_mock_llm(prompt, request)
        except Exception as e:
            logger.error(f"Custom API error: {e}")
            return await self._call_mock_llm(prompt, request)

    async def _call_mock_llm(self, prompt: str, request: CommentaryRequest) -> str:
        """Mock LLM for development and testing"""
        
        # Simulate different responses based on emotional state
        emotional_state = request.emotional_context.get('primary_emotion', 'analytical')
        
        template_responses = {
            'confident': "I'm seeing strong momentum patterns in the DNA subsystem, and my confidence is high. The signals are aligning beautifully, and I feel good about taking a position here.",
            
            'fearful': "My immune system is screaming warnings right now. The market feels dangerous, and I'm getting that pit-in-stomach feeling that tells me to step back and reassess.",
            
            'excited': "This is exciting! Multiple subsystems are firing positive signals, and my dopamine system is lighting up. I can feel the momentum building - this could be a big move.",
            
            'confused': "I'm honestly torn here. My DNA subsystem is saying one thing, but the immune system is contradicting it. When I'm this uncertain, I usually wait for clearer signals.",
            
            'analytical': "Looking at the data objectively, I see mixed signals across the subsystems. The temporal cycles suggest we're near a turning point, but I need more confirmation before acting."
        }
        
        response = template_responses.get(emotional_state, template_responses['analytical'])
        
        # Add some variety based on market context
        volatility = request.market_context.get('volatility', 0.02)
        if volatility > 0.05:
            response += " This high volatility environment is making me extra cautious about position sizing."
        
        # Add trigger-specific context
        if request.trigger_event == 'position_entry':
            response += " Let's see how this position develops."
        elif request.trigger_event == 'stop_loss':
            response += " Sometimes the market teaches us expensive lessons."
        elif request.trigger_event == 'profit_target':
            response += " Nothing beats the feeling of a planned exit working perfectly."
        
        return response
    
    def _parse_llm_response(self, response_text: str, request: CommentaryRequest) -> CommentaryResponse:
        """Parse and validate LLM response"""
        
        # Clean up response
        cleaned_text = response_text.strip()
        
        # Truncate if too long, but try to end at a sentence boundary
        if len(cleaned_text) > request.max_length:
            # Try to find the last sentence boundary within the limit
            truncate_point = request.max_length - 3
            sentence_end = max(
                cleaned_text.rfind('.', 0, truncate_point),
                cleaned_text.rfind('!', 0, truncate_point),
                cleaned_text.rfind('?', 0, truncate_point)
            )
            
            if sentence_end > request.max_length * 0.7:  # If we found a good break point
                cleaned_text = cleaned_text[:sentence_end + 1]
            else:
                # Fallback to character limit
                cleaned_text = cleaned_text[:truncate_point] + "..."
        
        # Extract key themes (simple keyword extraction)
        key_themes = self._extract_key_themes(cleaned_text)
        
        # Estimate confidence and emotional intensity
        confidence = self._estimate_response_confidence(cleaned_text)
        emotional_intensity = self._estimate_emotional_intensity(cleaned_text)
        
        # Check if follow-up is suggested
        follow_up_suggested = any(phrase in cleaned_text.lower() for phrase in [
            "let me watch", "i'll monitor", "need to see", "waiting for"
        ])
        
        return CommentaryResponse(
            text=cleaned_text,
            confidence=confidence,
            emotional_intensity=emotional_intensity,
            key_themes=key_themes,
            follow_up_suggested=follow_up_suggested
        )
    
    def _extract_key_themes(self, text: str) -> List[str]:
        """Extract key themes from commentary text"""
        
        themes = []
        text_lower = text.lower()
        
        # Trading action themes
        if any(word in text_lower for word in ['buy', 'long', 'bullish']):
            themes.append('bullish_bias')
        if any(word in text_lower for word in ['sell', 'short', 'bearish']):
            themes.append('bearish_bias')
        if any(word in text_lower for word in ['wait', 'hold', 'pause']):
            themes.append('patience')
        
        # Emotional themes
        if any(word in text_lower for word in ['confident', 'sure', 'certain']):
            themes.append('confidence')
        if any(word in text_lower for word in ['worried', 'concerned', 'nervous']):
            themes.append('concern')
        if any(word in text_lower for word in ['excited', 'energized', 'pumped']):
            themes.append('excitement')
        if any(word in text_lower for word in ['confused', 'uncertain', 'torn']):
            themes.append('uncertainty')
        
        # Market themes
        if any(word in text_lower for word in ['volatility', 'volatile', 'choppy']):
            themes.append('volatility')
        if any(word in text_lower for word in ['momentum', 'trend', 'direction']):
            themes.append('momentum')
        if any(word in text_lower for word in ['risk', 'danger', 'threat']):
            themes.append('risk_management')
        
        return themes
    
    def _estimate_response_confidence(self, text: str) -> float:
        """Estimate confidence level from response text"""
        
        text_lower = text.lower()
        
        # High confidence indicators
        high_confidence_words = ['definitely', 'certainly', 'confident', 'sure', 'clearly']
        high_count = sum(1 for word in high_confidence_words if word in text_lower)
        
        # Low confidence indicators
        low_confidence_words = ['maybe', 'perhaps', 'uncertain', 'confused', 'might']
        low_count = sum(1 for word in low_confidence_words if word in text_lower)
        
        # Base confidence
        base_confidence = 0.6
        
        # Adjust based on word counts
        confidence = base_confidence + (high_count * 0.15) - (low_count * 0.15)
        
        return max(0.1, min(1.0, confidence))
    
    def _estimate_emotional_intensity(self, text: str) -> float:
        """Estimate emotional intensity from response text"""
        
        text_lower = text.lower()
        
        # High intensity indicators
        high_intensity_words = ['screaming', 'pumped', 'terrified', 'ecstatic', 'devastated']
        intensity_count = sum(1 for word in high_intensity_words if word in text_lower)
        
        # Exclamation points
        exclamation_count = text.count('!')
        
        # All caps words
        caps_count = sum(1 for word in text.split() if word.isupper() and len(word) > 2)
        
        # Calculate intensity
        intensity = 0.3 + (intensity_count * 0.2) + (exclamation_count * 0.1) + (caps_count * 0.1)
        
        return max(0.0, min(1.0, intensity))
    
    def _create_fallback_response(self, request: CommentaryRequest) -> CommentaryResponse:
        """Create fallback response when LLM fails"""
        
        fallback_text = self._create_fallback_text(request)
        
        return CommentaryResponse(
            text=fallback_text,
            confidence=0.3,
            emotional_intensity=0.2,
            key_themes=['fallback'],
            follow_up_suggested=False
        )
    
    def _create_fallback_text(self, request: CommentaryRequest) -> str:
        """Create fallback text based on context"""
        
        emotional_state = request.emotional_context.get('primary_emotion', 'analytical')
        
        fallback_templates = {
            'confident': "I'm feeling confident about the current market setup based on my subsystem analysis.",
            'fearful': "My risk detection systems are showing elevated caution levels right now.",
            'excited': "Multiple positive signals are aligning - this looks promising.",
            'confused': "I'm seeing mixed signals and need more clarity before making decisions.",
            'analytical': "Analyzing current market conditions and subsystem outputs for optimal positioning."
        }
        
        return fallback_templates.get(emotional_state, fallback_templates['analytical'])
    
    def _update_conversation_memory(self, request: CommentaryRequest, response: CommentaryResponse):
        """Update conversation memory for context"""
        
        memory_entry = {
            'timestamp': time.time(),
            'event': request.trigger_event,
            'response': response.text,
            'emotional_state': request.emotional_context.get('primary_emotion'),
            'confidence': response.confidence,
            'themes': response.key_themes
        }
        
        self.conversation_memory.append(memory_entry)
        
        # Keep only recent memory
        if len(self.conversation_memory) > 20:
            self.conversation_memory = self.conversation_memory[-20:]
    
    async def _handle_rate_limiting(self):
        """Handle rate limiting for API calls"""
        
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def get_conversation_summary(self) -> Dict:
        """Get summary of recent conversation"""
        
        if not self.conversation_memory:
            return {'total_interactions': 0}
        
        recent = self.conversation_memory[-10:]
        
        # Analyze themes
        all_themes = []
        for memory in recent:
            all_themes.extend(memory.get('themes', []))
        
        theme_counts = {}
        for theme in all_themes:
            theme_counts[theme] = theme_counts.get(theme, 0) + 1
        
        # Analyze emotional progression
        emotional_states = [memory.get('emotional_state') for memory in recent]
        
        return {
            'total_interactions': len(self.conversation_memory),
            'recent_interactions': len(recent),
            'common_themes': sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            'emotional_progression': emotional_states,
            'avg_confidence': np.mean([memory.get('confidence', 0.5) for memory in recent]),
            'last_interaction_time': recent[-1]['timestamp'] if recent else None
        }