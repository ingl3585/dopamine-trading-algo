"""
Shared Intelligence Types - Clean interfaces for intelligence subsystem communication

This module provides shared types and interfaces for intelligence subsystems to communicate
with the trading agent while maintaining clean architecture principles.

Key design principles:
- Clear separation between intelligence signals and dopamine processing
- Type-safe interfaces using protocols and dataclasses
- Immutable data structures for reliable communication
- Extensible design for future intelligence subsystems
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Protocol, Any
import time


class IntelligenceSignalType(Enum):
    """Types of intelligence signals that subsystems can provide"""
    PATTERN_RECOGNITION = "pattern_recognition"    # DNA subsystem
    TEMPORAL_ANALYSIS = "temporal_analysis"        # Temporal subsystem
    ANOMALY_DETECTION = "anomaly_detection"        # Immune subsystem
    MICROSTRUCTURE = "microstructure"              # Microstructure subsystem
    REGIME_ANALYSIS = "regime_analysis"            # Cross-subsystem regime detection


class SignalStrength(Enum):
    """Standardized signal strength levels"""
    VERY_WEAK = "very_weak"      # 0.0 - 0.2
    WEAK = "weak"                # 0.2 - 0.4
    MODERATE = "moderate"        # 0.4 - 0.6
    STRONG = "strong"            # 0.6 - 0.8
    VERY_STRONG = "very_strong"  # 0.8 - 1.0


@dataclass(frozen=True)
class IntelligenceSignal:
    """
    Immutable intelligence signal from a subsystem
    
    This is the primary communication mechanism between intelligence subsystems
    and the trading agent's dopamine processing system.
    """
    signal_type: IntelligenceSignalType
    strength: float                    # Normalized signal strength [-1.0, 1.0]
    confidence: float                  # Confidence in the signal [0.0, 1.0]
    direction: str                     # 'bullish', 'bearish', 'neutral'
    timeframe: str                     # Timeframe this signal applies to
    metadata: Dict[str, Any]           # Additional subsystem-specific data
    timestamp: float                   # When the signal was generated
    subsystem_id: str                  # Identifier of the generating subsystem
    
    def __post_init__(self):
        """Validate signal constraints"""
        if not -1.0 <= self.strength <= 1.0:
            import logging
            import traceback
            logger = logging.getLogger(__name__)
            logger.error(f"Invalid signal strength {self.strength} from subsystem {self.subsystem_id}")
            logger.error(f"Signal creation traceback:\n{''.join(traceback.format_stack())}")
            
            # Auto-normalize the signal instead of crashing
            import numpy as np
            self.strength = float(np.tanh(self.strength))
            self.strength = max(-1.0, min(1.0, self.strength))
            logger.warning(f"Auto-normalized signal strength to {self.strength}")
            
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0.0, 1.0], got {self.confidence}")
        if self.direction not in ['bullish', 'bearish', 'neutral']:
            raise ValueError(f"Direction must be bullish/bearish/neutral, got {self.direction}")
    
    @property
    def signal_strength_category(self) -> SignalStrength:
        """Get categorized signal strength"""
        abs_strength = abs(self.strength)
        if abs_strength < 0.2:
            return SignalStrength.VERY_WEAK
        elif abs_strength < 0.4:
            return SignalStrength.WEAK
        elif abs_strength < 0.6:
            return SignalStrength.MODERATE
        elif abs_strength < 0.8:
            return SignalStrength.STRONG
        else:
            return SignalStrength.VERY_STRONG
    
    @property
    def weighted_strength(self) -> float:
        """Get confidence-weighted signal strength"""
        return self.strength * self.confidence


@dataclass(frozen=True)
class IntelligenceContext:
    """
    Contextual information accompanying intelligence signals
    
    Provides market context and environmental factors that can influence
    how signals should be interpreted and processed.
    """
    market_regime: str                 # Current market regime
    volatility_level: float           # Normalized volatility [0.0, 1.0]
    volume_profile: str               # 'low', 'normal', 'high'
    time_of_day: str                  # Trading session context
    market_conditions: Dict[str, Any] # Additional market condition data
    signal_aggregation_window: int    # Number of recent signals to consider
    
    def is_high_volatility(self) -> bool:
        """Check if current volatility is considered high"""
        return self.volatility_level > 0.7
    
    def is_low_volume(self) -> bool:
        """Check if current volume is considered low"""
        return self.volume_profile == 'low'


@dataclass(frozen=True)
class IntelligenceUpdate:
    """
    Complete intelligence update containing multiple signals and context
    
    This is the comprehensive update sent from intelligence subsystems
    to the trading agent for processing.
    """
    signals: List[IntelligenceSignal]
    context: IntelligenceContext
    primary_signal: Optional[IntelligenceSignal]  # Most important signal
    signal_consensus: float                       # Agreement between signals [-1.0, 1.0]
    update_timestamp: float
    
    def __post_init__(self):
        """Validate update consistency"""
        if not self.signals:
            raise ValueError("IntelligenceUpdate must contain at least one signal")
        
        # Set primary signal if not provided
        if self.primary_signal is None:
            # Choose signal with highest confidence-weighted strength
            object.__setattr__(self, 'primary_signal', 
                max(self.signals, key=lambda s: abs(s.weighted_strength)))
    
    @property
    def signal_count(self) -> int:
        """Number of signals in this update"""
        return len(self.signals)
    
    @property
    def average_confidence(self) -> float:
        """Average confidence across all signals"""
        if not self.signals:
            return 0.0
        return sum(s.confidence for s in self.signals) / len(self.signals)
    
    @property
    def strongest_signal_strength(self) -> float:
        """Strength of the strongest signal"""
        if not self.signals:
            return 0.0
        return max(abs(s.strength) for s in self.signals)


class IntelligenceProvider(Protocol):
    """
    Protocol for intelligence subsystems to provide signals
    
    All intelligence subsystems should implement this protocol to ensure
    consistent communication with the trading agent.
    """
    
    def get_current_signals(self, market_data: Dict[str, Any]) -> List[IntelligenceSignal]:
        """Get current intelligence signals based on market data"""
        ...
    
    def get_signal_context(self, market_data: Dict[str, Any]) -> IntelligenceContext:
        """Get contextual information for signal interpretation"""
        ...
    
    def process_market_update(self, market_data: Dict[str, Any]) -> IntelligenceUpdate:
        """Process market update and return complete intelligence update"""
        ...
    
    def get_subsystem_health(self) -> Dict[str, Any]:
        """Get health and performance metrics for the subsystem"""
        ...


class IntelligenceConsumer(Protocol):
    """
    Protocol for components that consume intelligence signals
    
    The trading agent's dopamine manager implements this protocol to process
    intelligence signals and integrate them into trading decisions.
    """
    
    def process_intelligence_update(self, update: IntelligenceUpdate) -> Dict[str, Any]:
        """Process intelligence update and return processing results"""
        ...
    
    def handle_signal_consensus(self, signals: List[IntelligenceSignal]) -> float:
        """Calculate consensus between multiple signals"""
        ...
    
    def integrate_intelligence_context(self, context: IntelligenceContext) -> Dict[str, Any]:
        """Integrate intelligence context into decision making"""
        ...


class IntelligenceAggregator(ABC):
    """
    Abstract base class for aggregating intelligence from multiple subsystems
    
    Provides common functionality for collecting, filtering, and aggregating
    intelligence signals from multiple subsystems.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.signal_history: List[IntelligenceUpdate] = []
        self.last_update_time = 0.0
    
    @abstractmethod
    def aggregate_signals(self, updates: List[IntelligenceUpdate]) -> IntelligenceUpdate:
        """Aggregate multiple intelligence updates into a single update"""
        pass
    
    @abstractmethod
    def filter_signals(self, signals: List[IntelligenceSignal], 
                      context: IntelligenceContext) -> List[IntelligenceSignal]:
        """Filter signals based on context and quality criteria"""
        pass
    
    def calculate_signal_consensus(self, signals: List[IntelligenceSignal]) -> float:
        """
        Calculate consensus between signals using a weighted approach
        
        Returns a value in [-1.0, 1.0] indicating the strength and direction
        of consensus between signals.
        """
        if not signals:
            return 0.0
        
        # Group signals by direction
        bullish_strength = sum(s.weighted_strength for s in signals if s.direction == 'bullish')
        bearish_strength = sum(abs(s.weighted_strength) for s in signals if s.direction == 'bearish')
        neutral_count = sum(1 for s in signals if s.direction == 'neutral')
        
        # Calculate net consensus
        net_strength = bullish_strength - bearish_strength
        
        # Weight by number of signals and reduce for neutral signals
        total_signals = len(signals)
        neutral_factor = 1.0 - (neutral_count / total_signals * 0.5)
        
        # Normalize to [-1.0, 1.0] range
        max_possible = total_signals  # If all signals were max strength in same direction
        consensus = (net_strength / max_possible) * neutral_factor
        
        return max(-1.0, min(1.0, consensus))
    
    def store_update(self, update: IntelligenceUpdate, max_history: int = 100):
        """Store intelligence update in history"""
        self.signal_history.append(update)
        if len(self.signal_history) > max_history:
            self.signal_history = self.signal_history[-max_history:]
        self.last_update_time = update.update_timestamp
    
    def get_recent_signals(self, timeframe_seconds: float = 300.0) -> List[IntelligenceSignal]:
        """Get all signals from recent updates within the specified timeframe"""
        cutoff_time = time.time() - timeframe_seconds
        recent_signals = []
        
        for update in self.signal_history:
            if update.update_timestamp >= cutoff_time:
                recent_signals.extend(update.signals)
        
        return recent_signals


# Utility functions for signal processing

def create_intelligence_signal(signal_type: IntelligenceSignalType, strength: float, 
                             confidence: float, direction: str, timeframe: str,
                             subsystem_id: str, **metadata) -> IntelligenceSignal:
    """
    Utility function to create a properly formatted intelligence signal
    
    Args:
        signal_type: Type of intelligence signal
        strength: Signal strength [-1.0, 1.0]
        confidence: Confidence level [0.0, 1.0]
        direction: Signal direction ('bullish', 'bearish', 'neutral')
        timeframe: Applicable timeframe
        subsystem_id: ID of the generating subsystem
        **metadata: Additional metadata
    
    Returns:
        IntelligenceSignal: Properly formatted signal
    """
    return IntelligenceSignal(
        signal_type=signal_type,
        strength=strength,
        confidence=confidence,
        direction=direction,
        timeframe=timeframe,
        metadata=metadata,
        timestamp=time.time(),
        subsystem_id=subsystem_id
    )


def create_intelligence_context(market_regime: str, volatility_level: float,
                              volume_profile: str, time_of_day: str,
                              **market_conditions) -> IntelligenceContext:
    """
    Utility function to create intelligence context
    
    Args:
        market_regime: Current market regime
        volatility_level: Volatility level [0.0, 1.0]
        volume_profile: Volume profile ('low', 'normal', 'high')
        time_of_day: Time of day context
        **market_conditions: Additional market conditions
    
    Returns:
        IntelligenceContext: Properly formatted context
    """
    return IntelligenceContext(
        market_regime=market_regime,
        volatility_level=volatility_level,
        volume_profile=volume_profile,
        time_of_day=time_of_day,
        market_conditions=market_conditions,
        signal_aggregation_window=20  # Default window
    )


def merge_intelligence_signals(signals: List[IntelligenceSignal], 
                             merge_strategy: str = 'weighted_average') -> IntelligenceSignal:
    """
    Merge multiple intelligence signals into a single signal
    
    Args:
        signals: List of signals to merge
        merge_strategy: Strategy for merging ('weighted_average', 'strongest', 'consensus')
    
    Returns:
        IntelligenceSignal: Merged signal
    """
    if not signals:
        raise ValueError("Cannot merge empty signal list")
    
    if len(signals) == 1:
        return signals[0]
    
    if merge_strategy == 'weighted_average':
        # Calculate weighted average based on confidence
        total_weight = sum(s.confidence for s in signals)
        if total_weight == 0:
            return signals[0]  # Fallback if all confidences are zero
        
        weighted_strength = sum(s.strength * s.confidence for s in signals) / total_weight
        avg_confidence = sum(s.confidence for s in signals) / len(signals)
        
        # Determine direction from weighted strength
        if weighted_strength > 0.1:
            direction = 'bullish'
        elif weighted_strength < -0.1:
            direction = 'bearish'
        else:
            direction = 'neutral'
        
        # Merge metadata
        merged_metadata = {}
        for signal in signals:
            merged_metadata.update(signal.metadata)
        merged_metadata['merged_from'] = [s.subsystem_id for s in signals]
        
        return IntelligenceSignal(
            signal_type=signals[0].signal_type,  # Use first signal's type
            strength=weighted_strength,
            confidence=avg_confidence,
            direction=direction,
            timeframe=signals[0].timeframe,  # Use first signal's timeframe
            metadata=merged_metadata,
            timestamp=time.time(),
            subsystem_id='merged_signal'
        )
    
    elif merge_strategy == 'strongest':
        # Return the signal with highest absolute strength
        return max(signals, key=lambda s: abs(s.strength))
    
    elif merge_strategy == 'consensus':
        # Return signal only if there's consensus, otherwise neutral
        bullish_signals = [s for s in signals if s.direction == 'bullish']
        bearish_signals = [s for s in signals if s.direction == 'bearish']
        
        if len(bullish_signals) > len(bearish_signals) * 2:
            strongest_bullish = max(bullish_signals, key=lambda s: s.strength)
            return strongest_bullish
        elif len(bearish_signals) > len(bullish_signals) * 2:
            strongest_bearish = max(bearish_signals, key=lambda s: abs(s.strength))
            return strongest_bearish
        else:
            # No consensus, return neutral signal
            avg_confidence = sum(s.confidence for s in signals) / len(signals)
            return IntelligenceSignal(
                signal_type=signals[0].signal_type,
                strength=0.0,
                confidence=avg_confidence,
                direction='neutral',
                timeframe=signals[0].timeframe,
                metadata={'consensus_failed': True, 'signal_count': len(signals)},
                timestamp=time.time(),
                subsystem_id='consensus_merger'
            )
    
    else:
        raise ValueError(f"Unknown merge strategy: {merge_strategy}")


# Export main types and functions
__all__ = [
    'IntelligenceSignalType',
    'SignalStrength', 
    'IntelligenceSignal',
    'IntelligenceContext',
    'IntelligenceUpdate',
    'IntelligenceProvider',
    'IntelligenceConsumer',
    'IntelligenceAggregator',
    'create_intelligence_signal',
    'create_intelligence_context',
    'merge_intelligence_signals'
]