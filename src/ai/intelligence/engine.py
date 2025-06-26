"""
Intelligence Engine - Orchestrates all AI subsystems using DDD patterns
Coordinates DNA, Temporal, Immune, and Microstructure subsystems
"""

import numpy as np
import logging
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Optional

from src.shared.types import MarketData, Signal
from .subsystems.dna.domain import DNASubsystem
from .subsystems.temporal.domain import FFTTemporalSubsystem
from .subsystems.immune.domain import EvolvingImmuneSystem
from .subsystems.microstructure.domain import MarketMicrostructureEngine

logger = logging.getLogger(__name__)

class IntelligenceEngine:
    """
    Enhanced Intelligence Orchestrator per prompt.txt:
    - Coordinates all four subsystems (DNA, Temporal, Immune, Microstructure)
    - Implements swarm intelligence for consensus building
    - Manages performance attribution and tool evolution
    - Provides clean domain interface for the trading system
    """
    
    def __init__(self):
        # Initialize all subsystems
        self.dna_subsystem = DNASubsystem()
        self.temporal_subsystem = FFTTemporalSubsystem()
        self.immune_subsystem = EvolvingImmuneSystem()
        self.microstructure_engine = MarketMicrostructureEngine()
        
        # Swarm intelligence coordination
        self.subsystem_votes = deque(maxlen=200)
        self.consensus_history = deque(maxlen=100)
        self.performance_attribution = defaultdict(lambda: deque(maxlen=50))
        
        # Tool evolution and lifecycle management
        self.tool_lifecycle = {
            'dna': {'birth_time': datetime.now(), 'performance_history': deque(maxlen=100)},
            'temporal': {'birth_time': datetime.now(), 'performance_history': deque(maxlen=100)},
            'immune': {'birth_time': datetime.now(), 'performance_history': deque(maxlen=100)},
            'microstructure': {'birth_time': datetime.now(), 'performance_history': deque(maxlen=100)}
        }
        
        # Hybrid tool breeding
        self.hybrid_tools = {}
        self.tool_breeding_frequency = 500
        self.decision_count = 0
        self.bootstrap_complete = False

    def analyze_market(self, market_data: MarketData, market_features: Dict) -> Dict[str, Signal]:
        """
        Main analysis method - coordinates all subsystems
        Returns signals from each subsystem plus overall consensus
        """
        try:
            self.decision_count += 1
            
            # Convert market data to format needed by subsystems
            prices = market_data.get('prices', [])
            volumes = market_data.get('volumes', [])
            timestamps = market_data.get('timestamps', [])
            
            # DNA Subsystem Analysis
            dna_signal = self._analyze_dna_patterns(prices, volumes, market_features)
            
            # Temporal Subsystem Analysis
            temporal_signal = self._analyze_temporal_cycles(prices, timestamps)
            
            # Immune Subsystem Analysis
            immune_signal = self._analyze_threat_patterns(market_features)
            
            # Microstructure Analysis
            microstructure_signal = self._analyze_microstructure(market_data, market_features)
            
            # Swarm Intelligence - Enhanced voting with performance attribution
            votes = {
                'dna': dna_signal,
                'temporal': temporal_signal,
                'immune': immune_signal,
                'microstructure': microstructure_signal
            }
            
            # Track individual tool performance
            for tool, signal_value in votes.items():
                if not (np.isnan(signal_value) or np.isinf(signal_value)):
                    self.tool_lifecycle[tool]['performance_history'].append(signal_value)
            
            # Calculate enhanced consensus
            consensus_strength = self._calculate_enhanced_consensus(votes)
            active_weights = self._calculate_dynamic_weights(market_features, votes)
            
            # Weighted overall signal with dynamic activation
            overall_signal = sum(votes[tool] * active_weights.get(tool, 0.0) for tool in votes.keys())
            
            # Boost signal based on consensus
            if consensus_strength > 0.8:
                overall_signal *= 1.4  # Strong consensus boost
            elif consensus_strength < 0.3:
                overall_signal *= 0.6  # Low consensus penalty
            
            # Store voting data
            self.subsystem_votes.append({
                'votes': votes.copy(),
                'consensus': consensus_strength,
                'weights': active_weights.copy(),
                'timestamp': datetime.now()
            })
            self.consensus_history.append(consensus_strength)
            
            # Check for hybrid tool creation
            if self.decision_count % self.tool_breeding_frequency == 0:
                self._attempt_tool_breeding(votes, market_features)
            
            # Return structured signals
            return {
                'dna': Signal(dna_signal, 0.8, 'dna_subsystem', datetime.now()),
                'temporal': Signal(temporal_signal, 0.7, 'temporal_subsystem', datetime.now()),
                'immune': Signal(immune_signal, 0.9, 'immune_subsystem', datetime.now()),
                'microstructure': Signal(microstructure_signal, 0.75, 'microstructure_engine', datetime.now()),
                'overall': Signal(overall_signal, consensus_strength, 'intelligence_engine', datetime.now()),
                'consensus_strength': consensus_strength,
                'active_weights': active_weights
            }
            
        except Exception as e:
            logger.error(f"Critical error in intelligence analysis: {e}")
            # Return safe fallback signals
            fallback_signal = Signal(0.0, 0.0, 'fallback', datetime.now())
            return {
                'dna': fallback_signal,
                'temporal': fallback_signal,
                'immune': fallback_signal,
                'microstructure': fallback_signal,
                'overall': fallback_signal,
                'consensus_strength': 0.0,
                'active_weights': {'dna': 0.25, 'temporal': 0.25, 'immune': 0.25, 'microstructure': 0.25}
            }

    def learn_from_outcome(self, outcome: float, context: Dict):
        """Learn from trade outcome to improve all subsystems"""
        try:
            # DNA subsystem learning
            dna_sequence = context.get('dna_sequence', '')
            if dna_sequence:
                self.dna_subsystem.learn_from_outcome(dna_sequence, outcome)
            
            # Temporal subsystem learning
            cycles_info = context.get('cycles_info', [])
            if cycles_info:
                self.temporal_subsystem.learn_from_outcome(cycles_info, outcome)
            
            # Immune subsystem learning
            market_state = context.get('market_state', {})
            if market_state:
                is_bootstrap = context.get('is_bootstrap', False)
                self.immune_subsystem.learn_threat(market_state, outcome, is_bootstrap)
            
            # Performance attribution
            if self.subsystem_votes:
                recent_vote = self.subsystem_votes[-1]
                votes = recent_vote.get('votes', {})
                weights = recent_vote.get('weights', {})
                
                for tool, signal in votes.items():
                    if tool in weights:
                        attribution_score = outcome * signal * weights[tool]
                        self.performance_attribution[tool].append(attribution_score)
            
            # Update hybrid tool performance
            for hybrid_name, hybrid_data in self.hybrid_tools.items():
                hybrid_data['performance_history'].append(outcome * 0.5)
            
            # Periodic evolution
            if len(self.subsystem_votes) % 100 == 0:
                self._evolve_tools()
                
        except Exception as e:
            logger.error(f"Error in learning from outcome: {e}")

    def _analyze_dna_patterns(self, prices: List[float], volumes: List[float], 
                             market_features: Dict) -> float:
        """Analyze DNA patterns with enhanced 16-base encoding"""
        try:
            if len(prices) < 20 or len(volumes) < 20:
                return 0.05  # Baseline signal
            
            volatility = market_features.get('volatility', 0)
            momentum = market_features.get('price_momentum', 0)
            
            dna_sequence = self.dna_subsystem.encode_market_state(
                prices[-20:], volumes[-20:], volatility, momentum
            )
            dna_signal = self.dna_subsystem.analyze_sequence(dna_sequence)
            
            # Validate and ensure minimum baseline
            if np.isnan(dna_signal) or np.isinf(dna_signal):
                return 0.05
            elif dna_signal == 0.0 and len(self.dna_subsystem.sequences) > 0:
                return 0.05
            
            return dna_signal
            
        except Exception as e:
            logger.error(f"DNA analysis failed: {e}")
            return 0.05

    def _analyze_temporal_cycles(self, prices: List[float], timestamps: Optional[List[float]]) -> float:
        """Analyze temporal cycles using FFT"""
        try:
            if len(prices) < 32:
                return 0.02  # Baseline signal
            
            temporal_signal = self.temporal_subsystem.analyze_cycles(prices, timestamps)
            
            if np.isnan(temporal_signal) or np.isinf(temporal_signal):
                return 0.02
            elif temporal_signal == 0.0:
                return 0.02
            
            return temporal_signal
            
        except Exception as e:
            logger.error(f"Temporal analysis failed: {e}")
            return 0.02

    def _analyze_threat_patterns(self, market_features: Dict) -> float:
        """Analyze threat patterns using immune system"""
        try:
            immune_signal = self.immune_subsystem.detect_threats(market_features)
            
            if np.isnan(immune_signal) or np.isinf(immune_signal):
                return -0.01
            elif immune_signal == 0.0 and len(self.immune_subsystem.antibodies) == 0:
                return -0.01
            
            return immune_signal
            
        except Exception as e:
            logger.error(f"Immune analysis failed: {e}")
            return -0.01

    def _analyze_microstructure(self, market_data: Dict, market_features: Dict) -> float:
        """Analyze market microstructure patterns"""
        try:
            microstructure_signal = self.microstructure_engine.get_microstructure_signal(
                market_data, market_features
            )
            
            if np.isnan(microstructure_signal) or np.isinf(microstructure_signal):
                return 0.0
            
            return microstructure_signal
            
        except Exception as e:
            logger.error(f"Microstructure analysis failed: {e}")
            return 0.0

    def _calculate_enhanced_consensus(self, votes: Dict[str, float]) -> float:
        """Enhanced consensus calculation with disagreement analysis"""
        signals = list(votes.values())
        
        if not signals:
            return 0.0
        
        # Directional agreement
        positive_signals = sum(1 for s in signals if s > 0.1)
        negative_signals = sum(1 for s in signals if s < -0.1)
        neutral_signals = sum(1 for s in signals if abs(s) <= 0.1)
        
        total_signals = len(signals)
        directional_consensus = max(positive_signals, negative_signals, neutral_signals) / total_signals
        
        # Magnitude agreement
        signal_magnitudes = [abs(s) for s in signals]
        magnitude_std = np.std(signal_magnitudes) if len(signal_magnitudes) > 1 else 0
        magnitude_consensus = 1.0 / (1.0 + magnitude_std)
        
        # Combined consensus
        return (directional_consensus * 0.7 + magnitude_consensus * 0.3)

    def _calculate_dynamic_weights(self, market_features: Dict, votes: Dict[str, float]) -> Dict[str, float]:
        """Calculate dynamic weights based on market conditions"""
        weights = {'dna': 0.3, 'temporal': 0.3, 'immune': 0.2, 'microstructure': 0.2}
        
        # Adjust based on market volatility
        volatility = market_features.get('volatility', 0.02)
        if volatility > 0.05:  # High volatility
            weights['immune'] += 0.15
            weights['dna'] -= 0.05
            weights['temporal'] -= 0.05
            weights['microstructure'] -= 0.05
        elif volatility < 0.01:  # Low volatility
            weights['temporal'] += 0.15
            weights['microstructure'] += 0.1
            weights['immune'] -= 0.15
            weights['dna'] -= 0.1
        
        # Adjust based on recent tool performance
        for tool in weights.keys():
            if tool in self.tool_lifecycle:
                recent_performance = list(self.tool_lifecycle[tool]['performance_history'])[-10:]
                if len(recent_performance) >= 5:
                    avg_performance = np.mean([abs(p) for p in recent_performance])
                    if avg_performance > 0.3:
                        weights[tool] *= 1.2
                    elif avg_performance < 0.1:
                        weights[tool] *= 0.8
        
        # Normalize weights
        total_weight = sum(weights.values())
        return {tool: weight / total_weight for tool, weight in weights.items()}

    def _attempt_tool_breeding(self, votes: Dict[str, float], market_features: Dict):
        """Attempt to create hybrid tools from successful combinations"""
        tool_pairs = [('dna', 'temporal'), ('dna', 'immune'), ('temporal', 'microstructure')]
        
        for tool1, tool2 in tool_pairs:
            signal1, signal2 = votes[tool1], votes[tool2]
            
            # Check for complementary behavior
            if abs(signal1) > 0.3 and abs(signal2) > 0.3 and np.sign(signal1) == np.sign(signal2):
                hybrid_name = f"{tool1}_{tool2}_hybrid"
                
                if hybrid_name not in self.hybrid_tools:
                    self.hybrid_tools[hybrid_name] = {
                        'parents': (tool1, tool2),
                        'birth_time': datetime.now(),
                        'performance_history': deque(maxlen=50),
                        'activation_conditions': self._create_activation_conditions(market_features),
                        'combination_strategy': 'weighted_average'
                    }
                    logger.info(f"Created hybrid tool: {hybrid_name}")

    def _create_activation_conditions(self, market_features: Dict) -> Dict:
        """Create activation conditions for hybrid tools"""
        return {
            'min_volatility': market_features.get('volatility', 0.02) * 0.8,
            'max_volatility': market_features.get('volatility', 0.02) * 1.2,
            'momentum_threshold': abs(market_features.get('price_momentum', 0)) * 0.5
        }

    def _evolve_tools(self):
        """Evolve tools based on performance"""
        # Evolve immune system
        self.immune_subsystem.evolve_antibodies()
        
        # Tool lifecycle management
        tools_to_remove = []
        for hybrid_name, hybrid_data in self.hybrid_tools.items():
            if len(hybrid_data['performance_history']) >= 20:
                avg_performance = np.mean(list(hybrid_data['performance_history']))
                if avg_performance < -0.2:
                    tools_to_remove.append(hybrid_name)
        
        for tool_name in tools_to_remove:
            del self.hybrid_tools[tool_name]
            logger.info(f"Removed underperforming hybrid tool: {tool_name}")

    def get_comprehensive_stats(self) -> Dict:
        """Get comprehensive intelligence statistics"""
        dna_stats = self.dna_subsystem.get_evolution_stats()
        immune_stats = self.immune_subsystem.get_immune_stats()
        temporal_stats = self.temporal_subsystem.get_cycle_stats()
        microstructure_stats = self.microstructure_engine.get_microstructure_stats()
        
        # Performance attribution stats
        attribution_stats = {}
        for tool, scores in self.performance_attribution.items():
            if scores:
                attribution_stats[tool] = {
                    'avg_attribution': np.mean(list(scores)),
                    'attribution_volatility': np.std(list(scores)),
                    'positive_attributions': sum(1 for s in scores if s > 0),
                    'total_attributions': len(scores)
                }
        
        return {
            'dna_evolution': dna_stats,
            'immune_system': immune_stats,
            'temporal_cycles': temporal_stats,
            'microstructure': microstructure_stats,
            'recent_consensus': np.mean(list(self.consensus_history)) if self.consensus_history else 0.0,
            'performance_attribution': attribution_stats,
            'hybrid_tools': {
                'count': len(self.hybrid_tools),
                'tools': list(self.hybrid_tools.keys())
            },
            'decision_count': self.decision_count
        }