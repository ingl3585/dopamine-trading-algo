"""
Personality Memory System

Maintains consistency and context for the AI trading personality across sessions
"""

import json
import logging
import time
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class MemoryEntry:
    timestamp: float
    event_type: str
    emotional_state: str
    market_context: Dict
    decision_context: Dict
    commentary: str
    outcome: Optional[float] = None  # P&L outcome if available
    confidence: float = 0.5
    key_themes: List[str] = None
    
    def __post_init__(self):
        if self.key_themes is None:
            self.key_themes = []

@dataclass
class PersonalityTraits:
    base_confidence: float = 0.6
    risk_tolerance: float = 0.5
    emotional_stability: float = 0.7
    learning_rate: float = 0.1
    memory_weight: float = 0.3
    consistency_preference: float = 0.8
    
    # Learned traits
    preferred_market_conditions: List[str] = None
    successful_patterns: Dict[str, float] = None
    emotional_triggers: Dict[str, float] = None
    
    def __post_init__(self):
        if self.preferred_market_conditions is None:
            self.preferred_market_conditions = []
        if self.successful_patterns is None:
            self.successful_patterns = {}
        if self.emotional_triggers is None:
            self.emotional_triggers = {}

class PersonalityMemory:
    """
    Manages personality memory, learning, and consistency for the AI trading personality
    """
    
    def __init__(self, config: Dict = None, memory_file: str = "data/personality_memory.json"):
        self.config = config or {}
        self.memory_file = memory_file
        
        # Memory storage
        self.short_term_memory = deque(maxlen=100)  # Recent interactions
        self.long_term_memory = deque(maxlen=1000)  # Extended history
        self.session_memory = deque(maxlen=50)      # Current session
        
        # Personality learning
        self.personality_traits = PersonalityTraits()
        self.learned_preferences = defaultdict(float)
        self.emotional_patterns = defaultdict(list)
        self.decision_patterns = defaultdict(list)
        
        # Context tracking
        self.current_session_id = self._generate_session_id()
        self.session_start_time = time.time()
        self.last_emotional_state = 'analytical'
        self.emotional_consistency_score = 0.8
        
        # Performance tracking
        self.commentary_effectiveness = defaultdict(list)
        self.emotional_accuracy = defaultdict(list)
        
        # Load existing memory
        self.load_memory()
    
    def add_memory(self, event_type: str, emotional_state: str, 
                   market_context: Dict, decision_context: Dict,
                   commentary: str, confidence: float = 0.5,
                   key_themes: List[str] = None) -> str:
        """
        Add new memory entry and update personality learning
        
        Returns:
            str: Memory ID for later reference
        """
        
        memory_entry = MemoryEntry(
            timestamp=time.time(),
            event_type=event_type,
            emotional_state=emotional_state,
            market_context=market_context.copy(),
            decision_context=decision_context.copy(),
            commentary=commentary,
            confidence=confidence,
            key_themes=key_themes or []
        )
        
        # Generate memory ID
        memory_id = f"{int(memory_entry.timestamp)}_{event_type}"
        
        # Add to memory stores
        self.short_term_memory.append(memory_entry)
        self.long_term_memory.append(memory_entry)
        self.session_memory.append(memory_entry)
        
        # Update personality learning
        self._update_personality_learning(memory_entry)
        
        # Update consistency tracking
        self._update_emotional_consistency(emotional_state)
        
        # Log memory addition
        logger.debug(f"Added memory: {event_type} with emotion {emotional_state}")
        
        return memory_id
    
    def update_memory_outcome(self, memory_id: str, outcome: float):
        """Update memory entry with actual outcome (P&L)"""
        
        # Find memory entry by ID (approximate matching)
        timestamp_str = memory_id.split('_')[0]
        target_timestamp = float(timestamp_str)
        
        # Search recent memories for matching entry
        for memory_store in [self.short_term_memory, self.session_memory]:
            for entry in memory_store:
                if abs(entry.timestamp - target_timestamp) < 60:  # Within 1 minute
                    entry.outcome = outcome
                    self._update_outcome_learning(entry)
                    logger.debug(f"Updated memory outcome: {outcome}")
                    return True
        
        return False
    
    def get_personality_context(self) -> Dict:
        """Get current personality context for commentary generation"""
        
        return {
            'personality_traits': asdict(self.personality_traits),
            'recent_emotional_pattern': self._get_recent_emotional_pattern(),
            'preferred_conditions': self._get_preferred_conditions(),
            'learned_biases': self._get_learned_biases(),
            'consistency_score': self.emotional_consistency_score,
            'session_context': self._get_session_context(),
            'successful_strategies': self._get_successful_strategies()
        }
    
    def get_contextual_guidance(self, current_context: Dict) -> Dict:
        """Get personality guidance based on current context and memory"""
        
        guidance = {
            'emotional_guidance': self._get_emotional_guidance(current_context),
            'behavioral_guidance': self._get_behavioral_guidance(current_context),
            'consistency_requirements': self._get_consistency_requirements(),
            'learning_adjustments': self._get_learning_adjustments(current_context)
        }
        
        return guidance
    
    def _update_personality_learning(self, memory_entry: MemoryEntry):
        """Update personality traits based on new memory"""
        
        # Learn from emotional states
        self.emotional_patterns[memory_entry.emotional_state].append({
            'timestamp': memory_entry.timestamp,
            'market_context': memory_entry.market_context,
            'confidence': memory_entry.confidence,
            'themes': memory_entry.key_themes
        })
        
        # Learn from decision contexts
        decision_type = memory_entry.decision_context.get('decision_type', 'unknown')
        self.decision_patterns[decision_type].append({
            'emotional_state': memory_entry.emotional_state,
            'confidence': memory_entry.confidence,
            'market_volatility': memory_entry.market_context.get('volatility', 0.02)
        })
        
        # Update learned preferences
        for theme in memory_entry.key_themes:
            self.learned_preferences[theme] += 0.1
        
        # Decay old preferences
        for key in self.learned_preferences:
            self.learned_preferences[key] *= 0.99
    
    def _update_outcome_learning(self, memory_entry: MemoryEntry):
        """Update learning based on actual outcomes"""
        
        if memory_entry.outcome is None:
            return
        
        # Analyze prediction accuracy
        predicted_confidence = memory_entry.confidence
        actual_success = 1.0 if memory_entry.outcome > 0 else 0.0
        
        # Update emotional accuracy tracking
        self.emotional_accuracy[memory_entry.emotional_state].append({
            'predicted_confidence': predicted_confidence,
            'actual_success': actual_success,
            'outcome_magnitude': abs(memory_entry.outcome),
            'timestamp': memory_entry.timestamp
        })
        
        # Update commentary effectiveness
        commentary_length = len(memory_entry.commentary.split())
        self.commentary_effectiveness[memory_entry.event_type].append({
            'length': commentary_length,
            'confidence': predicted_confidence,
            'outcome': memory_entry.outcome,
            'themes': memory_entry.key_themes
        })
        
        # Adjust personality traits based on outcomes
        self._adjust_personality_traits(memory_entry)
    
    def _adjust_personality_traits(self, memory_entry: MemoryEntry):
        """Adjust personality traits based on outcomes"""
        
        if memory_entry.outcome is None:
            return
        
        outcome_success = memory_entry.outcome > 0
        confidence_was_high = memory_entry.confidence > 0.7
        
        # Adjust base confidence
        if outcome_success and confidence_was_high:
            self.personality_traits.base_confidence += 0.01
        elif not outcome_success and confidence_was_high:
            self.personality_traits.base_confidence -= 0.01
        
        # Adjust risk tolerance based on outcomes
        market_volatility = memory_entry.market_context.get('volatility', 0.02)
        if outcome_success and market_volatility > 0.03:
            self.personality_traits.risk_tolerance += 0.005
        elif not outcome_success and market_volatility > 0.03:
            self.personality_traits.risk_tolerance -= 0.005
        
        # Bound adjustments
        self.personality_traits.base_confidence = np.clip(
            self.personality_traits.base_confidence, 0.3, 0.9
        )
        self.personality_traits.risk_tolerance = np.clip(
            self.personality_traits.risk_tolerance, 0.2, 0.8
        )
    
    def _update_emotional_consistency(self, current_emotional_state: str):
        """Update emotional consistency tracking"""
        
        if self.last_emotional_state == current_emotional_state:
            self.emotional_consistency_score = min(1.0, self.emotional_consistency_score + 0.05)
        else:
            # Check if transition makes sense
            valid_transitions = {
                'confident': ['excited', 'analytical'],
                'fearful': ['cautious', 'defensive'],
                'excited': ['confident', 'optimistic'],
                'confused': ['analytical', 'cautious'],
                'analytical': ['confident', 'cautious', 'confused'],
                'cautious': ['fearful', 'analytical'],
                'defensive': ['fearful', 'cautious'],
                'optimistic': ['excited', 'confident']
            }
            
            if current_emotional_state in valid_transitions.get(self.last_emotional_state, []):
                self.emotional_consistency_score = max(0.5, self.emotional_consistency_score - 0.02)
            else:
                self.emotional_consistency_score = max(0.3, self.emotional_consistency_score - 0.05)
        
        self.last_emotional_state = current_emotional_state
    
    def _get_recent_emotional_pattern(self) -> Dict:
        """Get recent emotional state patterns"""
        
        if not self.short_term_memory:
            return {'dominant_emotion': 'analytical', 'stability': 0.8}
        
        recent_emotions = [entry.emotional_state for entry in list(self.short_term_memory)[-10:]]
        
        # Count emotion frequencies
        emotion_counts = defaultdict(int)
        for emotion in recent_emotions:
            emotion_counts[emotion] += 1
        
        # Find dominant emotion
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
        
        # Calculate emotional stability
        if len(set(recent_emotions)) <= 2:
            stability = 0.9
        elif len(set(recent_emotions)) <= 4:
            stability = 0.7
        else:
            stability = 0.4
        
        return {
            'dominant_emotion': dominant_emotion,
            'recent_emotions': recent_emotions[-5:],
            'stability': stability,
            'emotion_distribution': dict(emotion_counts)
        }
    
    def _get_preferred_conditions(self) -> Dict:
        """Get learned market condition preferences"""
        
        preferences = {}
        
        # Analyze successful emotional states by market conditions
        for emotion, pattern_list in self.emotional_patterns.items():
            if len(pattern_list) < 3:
                continue
            
            # Calculate average confidence in different volatility regimes
            low_vol_confidence = []
            high_vol_confidence = []
            
            for pattern in pattern_list:
                volatility = pattern['market_context'].get('volatility', 0.02)
                confidence = pattern['confidence']
                
                if volatility < 0.02:
                    low_vol_confidence.append(confidence)
                elif volatility > 0.04:
                    high_vol_confidence.append(confidence)
            
            if low_vol_confidence:
                preferences[f'{emotion}_low_vol'] = np.mean(low_vol_confidence)
            if high_vol_confidence:
                preferences[f'{emotion}_high_vol'] = np.mean(high_vol_confidence)
        
        return preferences
    
    def _get_learned_biases(self) -> Dict:
        """Get learned behavioral biases"""
        
        biases = {}
        
        # Overconfidence bias
        if self.emotional_accuracy.get('confident'):
            confident_accuracy = [
                entry['actual_success'] for entry in self.emotional_accuracy['confident']
            ]
            if len(confident_accuracy) >= 5:
                actual_accuracy = np.mean(confident_accuracy)
                if actual_accuracy < 0.6:  # Low success rate when confident
                    biases['overconfidence'] = 1.0 - actual_accuracy
        
        # Loss aversion
        if self.emotional_patterns.get('fearful'):
            fear_patterns = self.emotional_patterns['fearful']
            fear_in_losses = sum(1 for p in fear_patterns 
                               if p['market_context'].get('daily_pnl', 0) < 0)
            if len(fear_patterns) > 0:
                biases['loss_aversion'] = fear_in_losses / len(fear_patterns)
        
        # Recency bias
        recent_preferences = dict(list(self.learned_preferences.items())[-5:])
        all_preferences = dict(self.learned_preferences)
        
        if recent_preferences and all_preferences:
            recent_weight = sum(recent_preferences.values())
            total_weight = sum(all_preferences.values())
            if total_weight > 0:
                biases['recency_bias'] = recent_weight / total_weight
        
        return biases
    
    def _get_session_context(self) -> Dict:
        """Get current session context"""
        
        session_duration = time.time() - self.session_start_time
        
        session_emotions = [entry.emotional_state for entry in self.session_memory]
        session_confidence = [entry.confidence for entry in self.session_memory]
        
        return {
            'session_id': self.current_session_id,
            'duration_minutes': session_duration / 60,
            'interaction_count': len(self.session_memory),
            'session_emotions': session_emotions,
            'avg_confidence': np.mean(session_confidence) if session_confidence else 0.5,
            'emotional_range': len(set(session_emotions)) if session_emotions else 0
        }
    
    def _get_successful_strategies(self) -> Dict:
        """Get strategies that have been successful"""
        
        strategies = {}
        
        # Analyze commentary effectiveness
        for event_type, effectiveness_list in self.commentary_effectiveness.items():
            if len(effectiveness_list) < 3:
                continue
            
            positive_outcomes = [e for e in effectiveness_list if e['outcome'] > 0]
            if positive_outcomes:
                avg_confidence = np.mean([e['confidence'] for e in positive_outcomes])
                avg_length = np.mean([e['length'] for e in positive_outcomes])
                
                strategies[f'{event_type}_successful'] = {
                    'confidence_level': avg_confidence,
                    'commentary_length': avg_length,
                    'success_rate': len(positive_outcomes) / len(effectiveness_list)
                }
        
        return strategies
    
    def _get_emotional_guidance(self, current_context: Dict) -> Dict:
        """Get emotional guidance based on memory and context"""
        
        guidance = {}
        
        # Check for similar past situations
        current_volatility = current_context.get('market_context', {}).get('volatility', 0.02)
        current_emotion = current_context.get('emotional_context', {}).get('primary_emotion', 'analytical')
        
        # Find similar historical situations
        similar_situations = []
        for entry in self.short_term_memory:
            entry_volatility = entry.market_context.get('volatility', 0.02)
            if abs(entry_volatility - current_volatility) < 0.01:
                similar_situations.append(entry)
        
        if similar_situations:
            # Analyze what worked in similar situations
            successful_emotions = []
            for entry in similar_situations:
                if entry.outcome and entry.outcome > 0:
                    successful_emotions.append(entry.emotional_state)
            
            if successful_emotions:
                guidance['recommended_emotion'] = max(set(successful_emotions), 
                                                    key=successful_emotions.count)
        
        # Consistency guidance
        if self.emotional_consistency_score < 0.5:
            guidance['consistency_warning'] = "Consider emotional stability"
        
        return guidance
    
    def _get_behavioral_guidance(self, current_context: Dict) -> Dict:
        """Get behavioral guidance based on learned patterns"""
        
        guidance = {}
        
        # Risk tolerance guidance
        recent_outcomes = [entry.outcome for entry in self.short_term_memory 
                         if entry.outcome is not None]
        
        if recent_outcomes:
            recent_success_rate = sum(1 for o in recent_outcomes[-5:] if o > 0) / min(5, len(recent_outcomes))
            
            if recent_success_rate < 0.4:
                guidance['risk_adjustment'] = "Consider reducing risk tolerance"
            elif recent_success_rate > 0.8:
                guidance['risk_adjustment'] = "Consider slightly increasing risk tolerance"
        
        # Commentary style guidance
        current_market_volatility = current_context.get('market_context', {}).get('volatility', 0.02)
        
        if current_market_volatility > 0.05:
            guidance['commentary_style'] = "Use cautious and detailed explanations"
        elif current_market_volatility < 0.015:
            guidance['commentary_style'] = "Can be more analytical and decisive"
        
        return guidance
    
    def _get_consistency_requirements(self) -> Dict:
        """Get requirements for maintaining personality consistency"""
        
        return {
            'emotional_consistency_score': self.emotional_consistency_score,
            'minimum_consistency': 0.6,
            'requires_stability': self.emotional_consistency_score < 0.6,
            'last_emotional_state': self.last_emotional_state,
            'personality_stability': self.personality_traits.emotional_stability
        }
    
    def _get_learning_adjustments(self, current_context: Dict) -> Dict:
        """Get learning-based adjustments for current situation"""
        
        adjustments = {}
        
        # Confidence adjustments based on historical accuracy
        current_emotion = current_context.get('emotional_context', {}).get('primary_emotion', 'analytical')
        
        if current_emotion in self.emotional_accuracy:
            historical_accuracy = self.emotional_accuracy[current_emotion]
            if len(historical_accuracy) >= 3:
                recent_accuracy = np.mean([h['actual_success'] for h in historical_accuracy[-5:]])
                
                if recent_accuracy < 0.5:
                    adjustments['confidence_adjustment'] = -0.2
                elif recent_accuracy > 0.8:
                    adjustments['confidence_adjustment'] = 0.1
        
        # Market condition adjustments
        current_volatility = current_context.get('market_context', {}).get('volatility', 0.02)
        preferred_conditions = self._get_preferred_conditions()
        
        emotion_vol_key = f'{current_emotion}_{"high" if current_volatility > 0.03 else "low"}_vol'
        if emotion_vol_key in preferred_conditions:
            if preferred_conditions[emotion_vol_key] > 0.7:
                adjustments['context_bonus'] = 0.1
            elif preferred_conditions[emotion_vol_key] < 0.4:
                adjustments['context_penalty'] = -0.1
        
        return adjustments
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return f"session_{int(time.time())}_{np.random.randint(1000, 9999)}"
    
    def save_memory(self):
        """Save memory to file"""
        
        try:
            memory_data = {
                'personality_traits': asdict(self.personality_traits),
                'learned_preferences': dict(self.learned_preferences),
                'emotional_patterns': {k: v[-10:] for k, v in self.emotional_patterns.items()},  # Keep recent
                'decision_patterns': {k: v[-10:] for k, v in self.decision_patterns.items()},
                'emotional_accuracy': {k: v[-20:] for k, v in self.emotional_accuracy.items()},
                'commentary_effectiveness': {k: v[-20:] for k, v in self.commentary_effectiveness.items()},
                'emotional_consistency_score': self.emotional_consistency_score,
                'last_emotional_state': self.last_emotional_state,
                'session_context': self._get_session_context(),
                'saved_at': datetime.now().isoformat()
            }
            
            # Convert long-term memory to serializable format
            memory_data['long_term_memory'] = [
                {
                    'timestamp': entry.timestamp,
                    'event_type': entry.event_type,
                    'emotional_state': entry.emotional_state,
                    'commentary': entry.commentary[:200],  # Truncate for storage
                    'confidence': entry.confidence,
                    'outcome': entry.outcome,
                    'key_themes': entry.key_themes
                }
                for entry in list(self.long_term_memory)[-100:]  # Keep recent 100
            ]
            
            import os
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            
            with open(self.memory_file, 'w') as f:
                json.dump(memory_data, f, indent=2, default=str)
            
            logger.info("Personality memory saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving personality memory: {e}")
    
    def load_memory(self):
        """Load memory from file"""
        
        try:
            with open(self.memory_file, 'r') as f:
                memory_data = json.load(f)
            
            # Restore personality traits
            if 'personality_traits' in memory_data:
                traits_data = memory_data['personality_traits']
                self.personality_traits = PersonalityTraits(**traits_data)
            
            # Restore learned preferences
            if 'learned_preferences' in memory_data:
                self.learned_preferences = defaultdict(float, memory_data['learned_preferences'])
            
            # Restore patterns
            if 'emotional_patterns' in memory_data:
                self.emotional_patterns = defaultdict(list, memory_data['emotional_patterns'])
            
            if 'decision_patterns' in memory_data:
                self.decision_patterns = defaultdict(list, memory_data['decision_patterns'])
            
            # Restore tracking data
            if 'emotional_accuracy' in memory_data:
                self.emotional_accuracy = defaultdict(list, memory_data['emotional_accuracy'])
            
            if 'commentary_effectiveness' in memory_data:
                self.commentary_effectiveness = defaultdict(list, memory_data['commentary_effectiveness'])
            
            # Restore state
            self.emotional_consistency_score = memory_data.get('emotional_consistency_score', 0.8)
            self.last_emotional_state = memory_data.get('last_emotional_state', 'analytical')
            
            logger.info("Personality memory loaded successfully")
            
        except FileNotFoundError:
            logger.info("No existing personality memory found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading personality memory: {e}")
    
    def get_memory_stats(self) -> Dict:
        """Get memory system statistics"""
        
        return {
            'short_term_entries': len(self.short_term_memory),
            'long_term_entries': len(self.long_term_memory),
            'session_entries': len(self.session_memory),
            'learned_preferences_count': len(self.learned_preferences),
            'emotional_patterns_count': len(self.emotional_patterns),
            'decision_patterns_count': len(self.decision_patterns),
            'emotional_consistency_score': self.emotional_consistency_score,
            'personality_traits': asdict(self.personality_traits),
            'session_duration_minutes': (time.time() - self.session_start_time) / 60,
            'total_outcomes_tracked': sum(
                len(entries) for entries in self.emotional_accuracy.values()
            )
        }