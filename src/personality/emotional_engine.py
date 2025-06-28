"""
Emotional State Engine for AI Trading Personality

Maps trading system states to human-like emotional responses
"""

import numpy as np
import time
from typing import Dict, List, Optional
from collections import deque
from dataclasses import dataclass
from enum import Enum

class EmotionalState(Enum):
    CONFIDENT = "confident"
    CAUTIOUS = "cautious" 
    EXCITED = "excited"
    FEARFUL = "fearful"
    CONFUSED = "confused"
    OPTIMISTIC = "optimistic"
    DEFENSIVE = "defensive"
    ANALYTICAL = "analytical"

@dataclass
class EmotionalMetrics:
    confidence: float = 0.5
    fear: float = 0.0
    excitement: float = 0.0
    confusion: float = 0.0
    pain: float = 0.0
    optimism: float = 0.5
    aggression: float = 0.3
    patience: float = 0.5
    
    # Derived states
    primary_emotion: EmotionalState = EmotionalState.ANALYTICAL
    emotional_intensity: float = 0.5
    emotional_stability: float = 0.8

class EmotionalStateEngine:
    """
    Converts trading system metrics into human-like emotional states
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Emotional history for stability
        self.emotion_history = deque(maxlen=50)
        self.confidence_history = deque(maxlen=20)
        self.fear_history = deque(maxlen=20)
        
        # Current emotional state
        self.current_emotions = EmotionalMetrics()
        
        # Personality traits (configurable)
        self.base_confidence = self.config.get('base_confidence', 0.6)
        self.fear_sensitivity = self.config.get('fear_sensitivity', 0.8)
        self.excitement_threshold = self.config.get('excitement_threshold', 0.7)
        self.emotional_dampening = self.config.get('emotional_dampening', 0.7)
        
        # Learning parameters
        self.adaptation_rate = 0.1
        self.stability_preference = 0.8
        
    def update_emotional_state(self, trading_context: Dict) -> EmotionalMetrics:
        """
        Update emotional state based on current trading context
        
        Args:
            trading_context: Dictionary containing:
                - subsystem_signals: Dict of all subsystem signals
                - portfolio_state: Current positions and P&L
                - market_context: Volatility, regime, etc.
                - recent_performance: Recent trade outcomes
                - system_confidence: Model uncertainty metrics
        
        Returns:
            EmotionalMetrics: Current emotional state
        """
        
        # Extract key components
        subsystem_signals = trading_context.get('subsystem_signals', {})
        portfolio_state = trading_context.get('portfolio_state', {})
        market_context = trading_context.get('market_context', {})
        recent_performance = trading_context.get('recent_performance', [])
        system_confidence = trading_context.get('system_confidence', 0.5)
        
        # Calculate core emotional components
        confidence = self._calculate_confidence(
            subsystem_signals, recent_performance, system_confidence
        )
        
        fear = self._calculate_fear(
            subsystem_signals, portfolio_state, market_context
        )
        
        excitement = self._calculate_excitement(
            subsystem_signals, portfolio_state, recent_performance
        )
        
        confusion = self._calculate_confusion(
            subsystem_signals, market_context, system_confidence
        )
        
        pain = self._calculate_pain(
            portfolio_state, recent_performance
        )
        
        optimism = self._calculate_optimism(
            recent_performance, market_context, confidence
        )
        
        aggression = self._calculate_aggression(
            confidence, excitement, fear, recent_performance
        )
        
        patience = self._calculate_patience(
            recent_performance, market_context, confusion
        )
        
        # Apply emotional dampening for stability
        confidence = self._dampen_emotion(confidence, self.confidence_history)
        fear = self._dampen_emotion(fear, self.fear_history)
        
        # Determine primary emotional state
        primary_emotion = self._determine_primary_emotion(
            confidence, fear, excitement, confusion, optimism
        )
        
        # Calculate emotional intensity and stability
        emotional_intensity = self._calculate_emotional_intensity(
            confidence, fear, excitement, confusion, pain
        )
        
        emotional_stability = self._calculate_emotional_stability()
        
        # Update current state
        self.current_emotions = EmotionalMetrics(
            confidence=confidence,
            fear=fear,
            excitement=excitement,
            confusion=confusion,
            pain=pain,
            optimism=optimism,
            aggression=aggression,
            patience=patience,
            primary_emotion=primary_emotion,
            emotional_intensity=emotional_intensity,
            emotional_stability=emotional_stability
        )
        
        # Update history
        self.emotion_history.append({
            'timestamp': time.time(),
            'emotions': self.current_emotions,
            'context': trading_context
        })
        
        self.confidence_history.append(confidence)
        self.fear_history.append(fear)
        
        return self.current_emotions
    
    def _calculate_confidence(self, subsystem_signals: Dict, 
                            recent_performance: List, system_confidence: float) -> float:
        """Calculate overall confidence level"""
        
        # Subsystem consensus strength
        signals = list(subsystem_signals.values())
        if signals:
            # Check for consensus (all signals in same direction)
            positive_signals = sum(1 for s in signals if s > 0.1)
            negative_signals = sum(1 for s in signals if s < -0.1)
            consensus_strength = max(positive_signals, negative_signals) / len(signals)
            
            # Signal strength
            signal_strength = np.mean([abs(s) for s in signals])
        else:
            consensus_strength = 0.5
            signal_strength = 0.5
        
        # Recent performance impact
        if recent_performance:
            winning_rate = sum(1 for p in recent_performance[-10:] if p > 0) / min(10, len(recent_performance))
            performance_confidence = min(1.0, winning_rate * 1.5)
        else:
            performance_confidence = 0.5
        
        # Combine factors
        confidence = (
            self.base_confidence * 0.3 +
            consensus_strength * 0.25 +
            signal_strength * 0.2 +
            performance_confidence * 0.15 +
            system_confidence * 0.1
        )
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _calculate_fear(self, subsystem_signals: Dict, portfolio_state: Dict, 
                       market_context: Dict) -> float:
        """Calculate fear/anxiety level"""
        
        # Immune system warnings
        immune_signal = subsystem_signals.get('immune', 0.0)
        immune_fear = max(0, -immune_signal) * 2.0  # Negative immune = fear
        
        # P&L-based fear (unrealized losses)
        unrealized_pnl = portfolio_state.get('unrealized_pnl', 0.0)
        pnl_fear = max(0, -unrealized_pnl / 1000.0)  # Normalize by account size
        
        # Volatility-based fear
        volatility = market_context.get('volatility', 0.02)
        volatility_fear = min(1.0, volatility * 20)  # High volatility = fear
        
        # Position size fear (overexposure)
        position_exposure = portfolio_state.get('position_exposure', 0.0)
        exposure_fear = max(0, (position_exposure - 0.3) * 2)  # Fear when > 30% exposed
        
        # Dopamine system negative feedback
        dopamine_signal = subsystem_signals.get('dopamine', 0.0)
        dopamine_fear = max(0, -dopamine_signal)
        
        # Combine fears
        total_fear = (
            immune_fear * 0.3 +
            pnl_fear * 0.25 +
            volatility_fear * 0.2 +
            exposure_fear * 0.15 +
            dopamine_fear * 0.1
        )
        
        return np.clip(total_fear * self.fear_sensitivity, 0.0, 1.0)
    
    def _calculate_excitement(self, subsystem_signals: Dict, portfolio_state: Dict,
                            recent_performance: List) -> float:
        """Calculate excitement/enthusiasm level"""
        
        # Strong positive signals
        signals = list(subsystem_signals.values())
        strong_signals = [s for s in signals if s > 0.5]
        signal_excitement = len(strong_signals) / max(1, len(signals))
        
        # Positive P&L momentum
        unrealized_pnl = portfolio_state.get('unrealized_pnl', 0.0)
        pnl_excitement = min(1.0, max(0, unrealized_pnl / 500.0))
        
        # Winning streak
        if recent_performance:
            recent_wins = []
            for p in reversed(recent_performance[-5:]):
                if p > 0:
                    recent_wins.append(p)
                else:
                    break
            streak_excitement = min(1.0, len(recent_wins) / 5.0)
        else:
            streak_excitement = 0.0
        
        # Dopamine positive feedback
        dopamine_signal = subsystem_signals.get('dopamine', 0.0)
        dopamine_excitement = max(0, dopamine_signal)
        
        # DNA momentum patterns
        dna_signal = subsystem_signals.get('dna', 0.0)
        momentum_excitement = max(0, dna_signal * 0.8)
        
        # Combine excitement factors
        total_excitement = (
            signal_excitement * 0.25 +
            pnl_excitement * 0.25 +
            streak_excitement * 0.2 +
            dopamine_excitement * 0.15 +
            momentum_excitement * 0.15
        )
        
        return np.clip(total_excitement, 0.0, 1.0)
    
    def _calculate_confusion(self, subsystem_signals: Dict, market_context: Dict,
                           system_confidence: float) -> float:
        """Calculate confusion/uncertainty level"""
        
        # Conflicting subsystem signals
        signals = list(subsystem_signals.values())
        if len(signals) > 1:
            signal_variance = np.var(signals)
            signal_confusion = min(1.0, signal_variance * 4.0)
        else:
            signal_confusion = 0.0
        
        # Low system confidence
        confidence_confusion = 1.0 - system_confidence
        
        # Unusual market conditions
        volatility = market_context.get('volatility', 0.02)
        if volatility > 0.06:  # Very high volatility
            volatility_confusion = 0.8
        elif volatility < 0.005:  # Very low volatility
            volatility_confusion = 0.4
        else:
            volatility_confusion = 0.0
        
        # Regime uncertainty
        regime_confidence = market_context.get('regime_confidence', 0.5)
        regime_confusion = 1.0 - regime_confidence
        
        # Combine confusion factors
        total_confusion = (
            signal_confusion * 0.4 +
            confidence_confusion * 0.25 +
            volatility_confusion * 0.2 +
            regime_confusion * 0.15
        )
        
        return np.clip(total_confusion, 0.0, 1.0)
    
    def _calculate_pain(self, portfolio_state: Dict, recent_performance: List) -> float:
        """Calculate emotional pain from losses"""
        
        # Unrealized loss pain
        unrealized_pnl = portfolio_state.get('unrealized_pnl', 0.0)
        unrealized_pain = max(0, -unrealized_pnl / 1000.0)
        
        # Recent realized losses
        if recent_performance:
            recent_losses = [p for p in recent_performance[-5:] if p < 0]
            if recent_losses:
                loss_pain = min(1.0, abs(sum(recent_losses)) / 1000.0)
            else:
                loss_pain = 0.0
        else:
            loss_pain = 0.0
        
        # Consecutive losses
        if recent_performance:
            consecutive_losses = 0
            for p in reversed(recent_performance):
                if p < 0:
                    consecutive_losses += 1
                else:
                    break
            consecutive_pain = min(1.0, consecutive_losses / 5.0)
        else:
            consecutive_pain = 0.0
        
        # Peak drawdown pain
        account_balance = portfolio_state.get('account_balance', 25000)
        peak_balance = portfolio_state.get('peak_balance', account_balance)
        if peak_balance > account_balance:
            drawdown_pain = (peak_balance - account_balance) / peak_balance
        else:
            drawdown_pain = 0.0
        
        # Combine pain factors
        total_pain = (
            unrealized_pain * 0.3 +
            loss_pain * 0.3 +
            consecutive_pain * 0.2 +
            drawdown_pain * 0.2
        )
        
        return np.clip(total_pain, 0.0, 1.0)
    
    def _calculate_optimism(self, recent_performance: List, market_context: Dict,
                          confidence: float) -> float:
        """Calculate optimism/hope level"""
        
        # Recent performance trend
        if len(recent_performance) >= 5:
            recent_trend = np.mean(recent_performance[-5:])
            trend_optimism = min(1.0, max(0, recent_trend / 200.0 + 0.5))
        else:
            trend_optimism = 0.5
        
        # Market opportunities (low volatility = good for trends)
        volatility = market_context.get('volatility', 0.02)
        if 0.01 < volatility < 0.04:  # Sweet spot
            market_optimism = 0.8
        else:
            market_optimism = 0.4
        
        # Confidence-based optimism
        confidence_optimism = confidence
        
        # Forward-looking indicators
        trend_strength = market_context.get('trend_strength', 0.0)
        trend_optimism_component = min(1.0, abs(trend_strength))
        
        # Combine optimism factors
        total_optimism = (
            trend_optimism * 0.3 +
            market_optimism * 0.25 +
            confidence_optimism * 0.25 +
            trend_optimism_component * 0.2
        )
        
        return np.clip(total_optimism, 0.0, 1.0)
    
    def _calculate_aggression(self, confidence: float, excitement: float,
                            fear: float, recent_performance: List) -> float:
        """Calculate risk-taking aggression level"""
        
        # High confidence + excitement = more aggressive
        confidence_aggression = (confidence + excitement) / 2.0
        
        # Fear reduces aggression
        fear_reduction = fear * 0.8
        
        # Recent wins increase aggression
        if recent_performance:
            recent_wins = sum(1 for p in recent_performance[-3:] if p > 0)
            win_aggression = min(1.0, recent_wins / 3.0)
        else:
            win_aggression = 0.3
        
        # Base personality aggression
        base_aggression = self.config.get('base_aggression', 0.4)
        
        # Combine factors
        total_aggression = (
            confidence_aggression * 0.4 +
            win_aggression * 0.3 +
            base_aggression * 0.3 -
            fear_reduction
        )
        
        return np.clip(total_aggression, 0.0, 1.0)
    
    def _calculate_patience(self, recent_performance: List, market_context: Dict,
                          confusion: float) -> float:
        """Calculate patience/waiting willingness"""
        
        # Recent losses increase patience (wait for better setups)
        if recent_performance:
            recent_losses = sum(1 for p in recent_performance[-3:] if p < 0)
            loss_patience = min(1.0, recent_losses / 3.0 * 0.8)
        else:
            loss_patience = 0.5
        
        # High confusion increases patience
        confusion_patience = confusion * 0.8
        
        # Low volatility = can afford to be patient
        volatility = market_context.get('volatility', 0.02)
        volatility_patience = max(0.2, 1.0 - volatility * 20)
        
        # Base personality patience
        base_patience = self.config.get('base_patience', 0.6)
        
        # Combine patience factors
        total_patience = (
            loss_patience * 0.3 +
            confusion_patience * 0.25 +
            volatility_patience * 0.25 +
            base_patience * 0.2
        )
        
        return np.clip(total_patience, 0.0, 1.0)
    
    def _determine_primary_emotion(self, confidence: float, fear: float,
                                 excitement: float, confusion: float,
                                 optimism: float) -> EmotionalState:
        """Determine the dominant emotional state"""
        
        emotions = {
            EmotionalState.CONFIDENT: confidence,
            EmotionalState.FEARFUL: fear,
            EmotionalState.EXCITED: excitement,
            EmotionalState.CONFUSED: confusion,
            EmotionalState.OPTIMISTIC: optimism,
            EmotionalState.CAUTIOUS: (fear + confusion) / 2,
            EmotionalState.DEFENSIVE: fear * 1.2,
            EmotionalState.ANALYTICAL: 1.0 - confusion
        }
        
        # Find primary emotion
        primary = max(emotions.items(), key=lambda x: x[1])
        
        # Require minimum threshold for non-analytical states
        if primary[1] < 0.4 and primary[0] != EmotionalState.ANALYTICAL:
            return EmotionalState.ANALYTICAL
        
        return primary[0]
    
    def _calculate_emotional_intensity(self, confidence: float, fear: float,
                                     excitement: float, confusion: float,
                                     pain: float) -> float:
        """Calculate overall emotional intensity"""
        
        # High values in any emotion = high intensity
        max_emotion = max(confidence, fear, excitement, confusion, pain)
        
        # Variance in emotions = intensity
        emotion_variance = np.var([confidence, fear, excitement, confusion, pain])
        
        # Combine factors
        intensity = (max_emotion * 0.7 + min(1.0, emotion_variance * 2) * 0.3)
        
        return np.clip(intensity, 0.0, 1.0)
    
    def _calculate_emotional_stability(self) -> float:
        """Calculate emotional stability over time"""
        
        if len(self.emotion_history) < 5:
            return 0.8  # Default stable
        
        # Variance in recent emotions
        recent_emotions = list(self.emotion_history)[-5:]
        
        confidence_values = [e['emotions'].confidence for e in recent_emotions]
        fear_values = [e['emotions'].fear for e in recent_emotions]
        
        confidence_stability = 1.0 - min(1.0, np.var(confidence_values) * 2)
        fear_stability = 1.0 - min(1.0, np.var(fear_values) * 2)
        
        # Overall stability
        stability = (confidence_stability + fear_stability) / 2.0
        
        return np.clip(stability, 0.0, 1.0)
    
    def _dampen_emotion(self, current_value: float, history: deque) -> float:
        """Apply emotional dampening for stability"""
        
        if not history:
            return current_value
        
        # Recent average
        recent_avg = np.mean(list(history)[-3:])
        
        # Dampen extreme changes
        dampened = (
            current_value * (1 - self.emotional_dampening) +
            recent_avg * self.emotional_dampening
        )
        
        return dampened
    
    def get_emotional_description(self) -> str:
        """Get human-readable emotional state description"""
        
        emotions = self.current_emotions
        
        # Primary emotion descriptions
        emotion_descriptions = {
            EmotionalState.CONFIDENT: "feeling confident and decisive",
            EmotionalState.FEARFUL: "experiencing fear and anxiety",
            EmotionalState.EXCITED: "excited and enthusiastic",
            EmotionalState.CONFUSED: "uncertain and conflicted",
            EmotionalState.OPTIMISTIC: "optimistic about opportunities",
            EmotionalState.CAUTIOUS: "cautious and careful",
            EmotionalState.DEFENSIVE: "defensive and risk-averse",
            EmotionalState.ANALYTICAL: "calm and analytical"
        }
        
        base_description = emotion_descriptions.get(
            emotions.primary_emotion, "in a neutral state"
        )
        
        # Add intensity modifier
        if emotions.emotional_intensity > 0.8:
            intensity_modifier = "very "
        elif emotions.emotional_intensity > 0.6:
            intensity_modifier = "quite "
        elif emotions.emotional_intensity < 0.3:
            intensity_modifier = "slightly "
        else:
            intensity_modifier = ""
        
        # Add stability modifier
        if emotions.emotional_stability < 0.4:
            stability_modifier = " and emotionally volatile"
        elif emotions.emotional_stability > 0.8:
            stability_modifier = " and emotionally stable"
        else:
            stability_modifier = ""
        
        return f"{intensity_modifier}{base_description}{stability_modifier}"
    
    def get_emotional_context_for_llm(self) -> Dict:
        """Get emotional context optimized for LLM consumption"""
        
        emotions = self.current_emotions
        
        return {
            'primary_emotion': emotions.primary_emotion.value,
            'emotional_description': self.get_emotional_description(),
            'confidence_level': emotions.confidence,
            'fear_level': emotions.fear,
            'excitement_level': emotions.excitement,
            'confusion_level': emotions.confusion,
            'pain_level': emotions.pain,
            'optimism_level': emotions.optimism,
            'aggression_level': emotions.aggression,
            'patience_level': emotions.patience,
            'emotional_intensity': emotions.emotional_intensity,
            'emotional_stability': emotions.emotional_stability,
            'should_be_cautious': emotions.fear > 0.6 or emotions.confusion > 0.7,
            'should_be_aggressive': emotions.confidence > 0.7 and emotions.excitement > 0.6,
            'is_emotionally_stable': emotions.emotional_stability > 0.7,
            'dominant_traits': self._get_dominant_traits()
        }
    
    def _get_dominant_traits(self) -> List[str]:
        """Get list of dominant personality traits"""
        
        emotions = self.current_emotions
        traits = []
        
        if emotions.confidence > 0.7:
            traits.append("confident")
        if emotions.fear > 0.6:
            traits.append("fearful")
        if emotions.excitement > 0.6:
            traits.append("excited")
        if emotions.confusion > 0.6:
            traits.append("uncertain")
        if emotions.pain > 0.5:
            traits.append("hurt")
        if emotions.optimism > 0.7:
            traits.append("optimistic")
        if emotions.aggression > 0.7:
            traits.append("aggressive")
        if emotions.patience > 0.7:
            traits.append("patient")
        
        return traits if traits else ["analytical"]