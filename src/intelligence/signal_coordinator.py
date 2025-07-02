"""
Signal Coordinator - Manages subsystem interactions and prevents feedback loops
"""

import logging
import numpy as np
import time
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Types of signals that can be coordinated"""
    TRADING = "trading"
    REWARD = "reward" 
    CONFIDENCE = "confidence"
    RISK = "risk"

@dataclass
class SubsystemSignal:
    """Signal from a subsystem"""
    subsystem_name: str
    signal_type: SignalType
    value: float
    confidence: float
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class CoordinatedSignal:
    """Final coordinated signal output"""
    signal_type: SignalType
    value: float
    confidence: float
    contributing_subsystems: List[str]
    dampening_applied: float
    consensus_strength: float
    timestamp: float

class SignalCoordinator:
    """
    Coordinates signals from multiple subsystems to prevent feedback loops
    and provide dampened, consensus-based outputs
    """
    
    def __init__(self, dampening_factor: float = 0.8, consensus_threshold: float = 0.6):
        self.dampening_factor = dampening_factor  # Reduce signal strength when systems agree too much
        self.consensus_threshold = consensus_threshold  # Agreement threshold for dampening
        
        # Signal history for feedback detection
        self.signal_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.coordinated_history: deque = deque(maxlen=200)
        
        # Feedback loop detection
        self.feedback_detection_window = 20
        self.feedback_threshold = 0.95  # Correlation threshold for feedback detection
        self.detected_feedback_loops: Dict[str, float] = {}
        
        # Subsystem performance tracking
        self.subsystem_performance: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self.subsystem_weights: Dict[str, float] = defaultdict(lambda: 1.0)
        
        # Consensus tracking
        self.consensus_history: deque = deque(maxlen=100)
        self.recent_consensus_strength = 0.5
        
    def coordinate_signals(self, signals: List[SubsystemSignal]) -> CoordinatedSignal:
        """
        Coordinate multiple subsystem signals into a single output
        Applies dampening and feedback loop prevention
        """
        if not signals:
            return self._create_neutral_signal(SignalType.TRADING)
        
        # Group signals by type
        signals_by_type = defaultdict(list)
        for signal in signals:
            signals_by_type[signal.signal_type].append(signal)
        
        # Process each signal type separately (for now, focus on trading signals)
        trading_signals = signals_by_type.get(SignalType.TRADING, [])
        if not trading_signals:
            return self._create_neutral_signal(SignalType.TRADING)
        
        # Store raw signals for feedback detection
        for signal in trading_signals:
            self.signal_history[signal.subsystem_name].append({
                'value': signal.value,
                'confidence': signal.confidence,
                'timestamp': signal.timestamp
            })
        
        # Detect and handle feedback loops
        feedback_adjustment = self._detect_and_handle_feedback(trading_signals)
        
        # Calculate consensus
        consensus_strength = self._calculate_consensus(trading_signals)
        self.consensus_history.append(consensus_strength)
        self.recent_consensus_strength = consensus_strength
        
        # Apply dynamic weights based on subsystem performance
        weighted_signals = self._apply_performance_weights(trading_signals)
        
        # Calculate base coordinated value
        coordinated_value = self._calculate_weighted_average(weighted_signals)
        
        # Apply consensus-based dampening
        dampening_applied = self._apply_consensus_dampening(
            coordinated_value, consensus_strength, feedback_adjustment
        )
        
        # Calculate final confidence
        final_confidence = self._calculate_coordinated_confidence(weighted_signals, consensus_strength)
        
        # Create coordinated signal
        coordinated_signal = CoordinatedSignal(
            signal_type=SignalType.TRADING,
            value=dampening_applied,
            confidence=final_confidence,
            contributing_subsystems=[s.subsystem_name for s in trading_signals],
            dampening_applied=abs(coordinated_value - dampening_applied),
            consensus_strength=consensus_strength,
            timestamp=time.time()
        )
        
        # Store coordinated signal for history
        self.coordinated_history.append({
            'value': coordinated_signal.value,
            'confidence': coordinated_signal.confidence,
            'consensus': consensus_strength,
            'timestamp': coordinated_signal.timestamp
        })
        
        # Log coordination details
        self._log_coordination_details(trading_signals, coordinated_signal)
        
        return coordinated_signal
    
    def update_subsystem_performance(self, subsystem_name: str, performance_score: float):
        """Update performance tracking for a subsystem (0.0 to 1.0)"""
        self.subsystem_performance[subsystem_name].append(performance_score)
        
        # Update weight based on recent performance
        recent_scores = list(self.subsystem_performance[subsystem_name])
        if len(recent_scores) >= 10:
            avg_performance = np.mean(recent_scores[-10:])
            # Weight ranges from 0.5 to 1.5 based on performance
            self.subsystem_weights[subsystem_name] = 0.5 + avg_performance
    
    def _detect_and_handle_feedback(self, signals: List[SubsystemSignal]) -> float:
        """Detect feedback loops and return adjustment factor"""
        if len(self.coordinated_history) < self.feedback_detection_window:
            return 1.0
        
        # Get recent coordinated signals
        recent_coordinated = [h['value'] for h in list(self.coordinated_history)[-self.feedback_detection_window:]]
        
        # Check correlation between each subsystem and coordinated output
        max_correlation = 0.0
        feedback_subsystem = None
        
        for signal in signals:
            subsystem_name = signal.subsystem_name
            if len(self.signal_history[subsystem_name]) < self.feedback_detection_window:
                continue
            
            recent_subsystem = [h['value'] for h in list(self.signal_history[subsystem_name])[-self.feedback_detection_window:]]
            
            if len(recent_subsystem) == len(recent_coordinated):
                correlation = np.corrcoef(recent_subsystem, recent_coordinated)[0, 1]
                if abs(correlation) > max_correlation:
                    max_correlation = abs(correlation)
                    feedback_subsystem = subsystem_name
        
        # If high correlation detected, apply feedback reduction
        if max_correlation > self.feedback_threshold:
            self.detected_feedback_loops[feedback_subsystem] = max_correlation
            feedback_reduction = 1.0 - (max_correlation - self.feedback_threshold) * 0.5
            logger.warning(f"Feedback loop detected in {feedback_subsystem}: {max_correlation:.3f}, "
                          f"applying {(1-feedback_reduction)*100:.1f}% reduction")
            return feedback_reduction
        
        return 1.0
    
    def _calculate_consensus(self, signals: List[SubsystemSignal]) -> float:
        """Calculate consensus strength among subsystems"""
        if len(signals) < 2:
            return 1.0
        
        values = [s.value for s in signals]
        
        # Calculate standard deviation normalized by signal range
        signal_std = np.std(values)
        signal_range = max(values) - min(values) if len(values) > 1 else 1.0
        
        # Consensus is higher when signals are closer together
        if signal_range > 0:
            consensus = 1.0 - min(signal_std / signal_range, 1.0)
        else:
            consensus = 1.0  # Perfect consensus when all signals identical
        
        return consensus
    
    def _apply_performance_weights(self, signals: List[SubsystemSignal]) -> List[Tuple[SubsystemSignal, float]]:
        """Apply performance-based weights to signals"""
        weighted_signals = []
        
        for signal in signals:
            weight = self.subsystem_weights.get(signal.subsystem_name, 1.0)
            # Also consider signal confidence in weighting
            final_weight = weight * signal.confidence
            weighted_signals.append((signal, final_weight))
        
        return weighted_signals
    
    def _calculate_weighted_average(self, weighted_signals: List[Tuple[SubsystemSignal, float]]) -> float:
        """Calculate weighted average of signals"""
        if not weighted_signals:
            return 0.0
        
        total_weighted_value = 0.0
        total_weight = 0.0
        
        for signal, weight in weighted_signals:
            total_weighted_value += signal.value * weight
            total_weight += weight
        
        return total_weighted_value / max(total_weight, 1e-8)
    
    def _apply_consensus_dampening(self, value: float, consensus_strength: float, feedback_adjustment: float) -> float:
        """Apply dampening based on consensus strength and feedback detection"""
        # High consensus can indicate overconfidence or herding - apply dampening
        if consensus_strength > self.consensus_threshold:
            excess_consensus = consensus_strength - self.consensus_threshold
            dampening_strength = excess_consensus * self.dampening_factor
            dampened_value = value * (1.0 - dampening_strength)
        else:
            dampened_value = value
        
        # Apply feedback adjustment
        final_value = dampened_value * feedback_adjustment
        
        # Ensure bounded output
        return np.clip(final_value, -1.0, 1.0)
    
    def _calculate_coordinated_confidence(self, weighted_signals: List[Tuple[SubsystemSignal, float]], consensus_strength: float) -> float:
        """Calculate confidence of coordinated signal"""
        if not weighted_signals:
            return 0.0
        
        # Base confidence from weighted average of individual confidences
        total_confidence = 0.0
        total_weight = 0.0
        
        for signal, weight in weighted_signals:
            total_confidence += signal.confidence * weight
            total_weight += weight
        
        base_confidence = total_confidence / max(total_weight, 1e-8)
        
        # Adjust confidence based on consensus
        # Moderate consensus increases confidence, extreme consensus decreases it
        consensus_adjustment = 1.0
        if consensus_strength > 0.9:  # Very high consensus - reduce confidence
            consensus_adjustment = 0.8
        elif consensus_strength > 0.7:  # High consensus - slight increase
            consensus_adjustment = 1.1
        elif consensus_strength < 0.3:  # Low consensus - reduce confidence
            consensus_adjustment = 0.7
        
        final_confidence = base_confidence * consensus_adjustment
        
        return np.clip(final_confidence, 0.0, 1.0)
    
    def _create_neutral_signal(self, signal_type: SignalType) -> CoordinatedSignal:
        """Create a neutral coordinated signal when no input signals available"""
        return CoordinatedSignal(
            signal_type=signal_type,
            value=0.0,
            confidence=0.0,
            contributing_subsystems=[],
            dampening_applied=0.0,
            consensus_strength=0.0,
            timestamp=time.time()
        )
    
    def _log_coordination_details(self, input_signals: List[SubsystemSignal], output_signal: CoordinatedSignal):
        """Log detailed coordination information"""
        if logger.isEnabledFor(logging.DEBUG):
            signal_summary = ", ".join([
                f"{s.subsystem_name}={s.value:.3f}(conf:{s.confidence:.3f})"
                for s in input_signals
            ])
            
            logger.debug(f"Signal coordination: [{signal_summary}] -> "
                        f"value={output_signal.value:.3f}, conf={output_signal.confidence:.3f}, "
                        f"consensus={output_signal.consensus_strength:.3f}, "
                        f"dampening={output_signal.dampening_applied:.3f}")
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """Get coordinator status for monitoring"""
        active_feedback_loops = {k: v for k, v in self.detected_feedback_loops.items() if v > self.feedback_threshold}
        
        return {
            'subsystem_weights': dict(self.subsystem_weights),
            'recent_consensus_strength': self.recent_consensus_strength,
            'active_feedback_loops': active_feedback_loops,
            'signals_processed': len(self.coordinated_history),
            'dampening_factor': self.dampening_factor,
            'consensus_threshold': self.consensus_threshold,
            'feedback_threshold': self.feedback_threshold
        }